"""What one model request is made of, by name — and the one way to build one (Builder).

A request used to be a list of messages each agent concatenated by hand: the profile, the
procedures, the policy, the retrieved window, the transcript, this turn's tool calls, all
inside one system string or scattered across messages. Nothing downstream could say how big
any of it was, or which part grew. `PromptBuilder` assembles the same messages from named
sections and returns, beside them, a `PromptManifest`: every section, where it came from,
how many tokens it costs, and which leading sections form the stable prefix a provider can
cache. The CONTEXT card and the LLM log draw from the manifest; the provider never sees it.

Section names are code identifiers (the vocabulary below); what a person reads for each is
configuration (`observability.context_card.labels`).
"""

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.llm.tokenizer import Tokenizer, shared_tokenizer
from agent_platform.llm.types import Message, Role

# The section vocabulary. An agent may add its own names; these are the ones every agent shares.
PROFILE = "profile"
PROCEDURES = "procedures"
POLICY = "policy"
INSTRUCTIONS = "instructions"
CONTEXT_WINDOW = "context_window"
WINDOW_STABLE = "window_stable"
# The chunk metadata key the Cache strategy marks a stable chunk with (`context.types.STABLE`).
STABLE_MARK = "stable"
USER_QUERY = "user_query"
HISTORY = "history"
TOOL_LOOP = "tool_loop"
PAYLOAD = "payload"


class ChunkLike(Protocol):
    """A window chunk as the builder reads it — the context package's `Chunk`, structurally."""

    @property
    def content(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def source(self) -> str: ...


@dataclass(frozen=True)
class PromptPart:
    """One attributable piece of text: what it is, where it came from, the text itself."""

    name: str
    text: str
    source: str = ""


class PromptSection(BaseModel):
    """One named part of a request, with its cost; `parts` break it down further."""

    name: str
    role: Role
    content: str
    tokens: int
    source: str = ""
    # Byte-identical from call to call within a session: part of the cacheable prefix.
    stable: bool = False
    parts: list["PromptSection"] = Field(default_factory=list)


class PromptManifest(BaseModel):
    """Every section of one request, in the order the model reads them, and their costs."""

    sections: list[PromptSection] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    tool_schema_tokens: int = 0
    tokenizer: str = ""
    exact: bool = False
    # Why the call was made (`observability.llm_purposes`), set by whoever reports it.
    purpose: str = ""
    # What a context window's strategies and failure checks did before this request.
    window_notes: list[str] = Field(default_factory=list)
    # True once a card has been drawn for this request, so the chain does not draw a second.
    announced: bool = False
    # Whether this model was sent the same stable prefix before (None: not yet known), so a
    # provider's prefix cache could serve it.
    prefix_seen: bool | None = None

    @property
    def total_tokens(self) -> int:
        return sum(section.tokens for section in self.sections) + self.tool_schema_tokens

    @property
    def prefix_tokens(self) -> int:
        return sum(section.tokens for section in self._prefix())

    @property
    def prefix_signature(self) -> str:
        """A short digest of the stable prefix: equal digests mean a provider cache can hit."""
        joined = "\x1f".join(section.content for section in self._prefix())
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
        return digest[: get_settings().context.prompt.signature_chars] if joined else ""

    def _prefix(self) -> list[PromptSection]:
        prefix: list[PromptSection] = []
        for section in self.sections:
            if not section.stable:
                break
            prefix.append(section)
        return prefix


@dataclass(frozen=True)
class TurnLayout:
    """A tool loop's fixed opening, so every step's request is broken down the same way.

    A loop's messages grow step by step — system, history, the question, then the model's
    calls and their results. The layout remembers what the opening was made of (the
    instruction parts, how many history messages), and `build` sections each step's
    messages from it.
    """

    instructions: tuple[PromptPart, ...] = ()
    # System parts that change per turn, after the window (session notes, today's date).
    tail: tuple[PromptPart, ...] = ()
    history: int = 0
    # The context window the system message ends with, and the chunks it was made of.
    window_text: str = ""
    window_chunks: tuple[Any, ...] = ()
    window_heading: str = ""
    window_notes: tuple[str, ...] = ()

    def build(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> "BuiltPrompt":
        rest = [message for message in messages if message.role != "system"]
        system = next((m.content for m in messages if m.role == "system"), "")
        builder = PromptBuilder()
        if self.instructions:
            builder.instructions(self.instructions)
        else:
            builder.system(INSTRUCTIONS, system)
        builder.window(self.window_heading, self.window_text, self.window_chunks)
        builder.notes(self.window_notes)
        for part in self.tail:
            builder.system(part.name, part.text, source=part.source, stable=False)
        asked = self.history + 1
        return (
            builder.messages(HISTORY, rest[: self.history])
            .messages(USER_QUERY, rest[self.history : asked])
            .messages(TOOL_LOOP, rest[asked:])
            .tools(tools)
            .build()
        )


@dataclass(frozen=True)
class BuiltPrompt:
    """The messages a provider is sent, and the manifest that describes them."""

    messages: list[Message]
    manifest: PromptManifest


class PromptBuilder:
    """Builds a request's messages from named sections, counting each as it goes.

    System sections are joined, in the order added, into the one system message every
    provider expects first; every other message keeps the order it was added in.
    """

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or shared_tokenizer()
        self._settings = get_settings().context.prompt
        self._system: list[PromptSection] = []
        self._conversation: list[tuple[PromptSection, list[Message]]] = []
        self._tools: list[dict[str, Any]] = []
        self._notes: list[str] = []

    def system(
        self, name: str, text: str, *, source: str = "", stable: bool = True
    ) -> "PromptBuilder":
        """A named block of the system message; empty text adds nothing."""
        if text.strip():
            self._system.append(self._section(name, "system", text, source, stable))
        return self

    def system_parts(
        self, name: str, parts: Sequence[PromptPart], *, stable: bool = True
    ) -> "PromptBuilder":
        """One system section built of attributable parts (each procedure, say)."""
        kept = [part for part in parts if part.text.strip()]
        if not kept:
            return self
        text = self._settings.separator.join(part.text for part in kept)
        if len(kept) == 1:
            return self.system(name, text, source=kept[0].source, stable=stable)
        section = self._section(name, "system", text, "", stable)
        section.parts = [
            self._section(p.source or p.name, "system", p.text, p.source, stable) for p in kept
        ]
        self._system.append(section)
        return self

    def instructions(self, parts: Sequence[PromptPart], *, stable: bool = True) -> "PromptBuilder":
        """The agent's instructions — its profile, then each procedure — one section per kind."""
        for name in dict.fromkeys(part.name for part in parts):
            self.system_parts(name, [part for part in parts if part.name == name], stable=stable)
        return self

    def window(self, heading: str, text: str, chunks: Iterable[ChunkLike] = ()) -> "PromptBuilder":
        """The assembled context window: one system block, broken down by what it holds."""
        if not text.strip():
            return self
        chunks = list(chunks)
        stable = [chunk for chunk in chunks if _stable(chunk)]
        rest = [chunk for chunk in chunks if not _stable(chunk)]
        joined = self._settings.separator.join(chunk.content for chunk in [*stable, *rest])
        if stable and joined == text:
            # The Cache strategy's chunks lead the window and never change: they extend the
            # cacheable prefix rather than starting the part of the request that varies.
            self._window_block(WINDOW_STABLE, heading, stable, stable=True)
            return self._window_block(CONTEXT_WINDOW, "", rest, stable=False)
        block = f"{heading}\n{text}" if heading else text
        section = self._section(CONTEXT_WINDOW, "system", block, "", stable=False)
        section.parts = self._window_parts(chunks)
        self._system.append(section)
        return self

    def _window_block(
        self, name: str, heading: str, chunks: list[ChunkLike], *, stable: bool
    ) -> "PromptBuilder":
        if not chunks:
            return self
        text = self._settings.separator.join(chunk.content for chunk in chunks)
        block = f"{heading}\n{text}" if heading else text
        section = self._section(name, "system", block, "", stable=stable)
        section.parts = self._window_parts(chunks)
        self._system.append(section)
        return self

    def notes(self, lines: Iterable[str]) -> "PromptBuilder":
        """What the window's strategies and checks did, for the CONTEXT card."""
        self._notes.extend(lines)
        return self

    def message(self, name: str, role: Role, text: str, *, source: str = "") -> "PromptBuilder":
        """One non-system message, its own section."""
        return self.messages(name, [Message(role=role, content=text)], source=source)

    def messages(
        self, name: str, messages: Sequence[Message], *, source: str = ""
    ) -> "PromptBuilder":
        """Several non-system messages counted as one section (the transcript, the tool loop)."""
        if not messages:
            return self
        rendered = (
            messages[0].content
            if len(messages) == 1 and not messages[0].tool_calls
            else self._settings.separator.join(render_message(item) for item in messages)
        )
        role = messages[0].role
        self._conversation.append(
            (self._section(name, role, rendered, source, False), list(messages))
        )
        return self

    def tools(self, specs: Sequence[dict[str, Any]] | None) -> "PromptBuilder":
        """The tool schemas offered with the request."""
        self._tools = list(specs or [])
        return self

    def build(self) -> BuiltPrompt:
        messages: list[Message] = []
        if self._system:
            joined = self._settings.separator.join(section.content for section in self._system)
            messages.append(Message(role="system", content=joined))
        for _, group in self._conversation:
            messages.extend(group)
        manifest = PromptManifest(
            sections=[*self._system, *(section for section, _ in self._conversation)],
            tools=tool_names(self._tools),
            tool_schema_tokens=self._count(json.dumps(self._tools)) if self._tools else 0,
            tokenizer=self._tokenizer.name,
            exact=self._tokenizer.exact,
            window_notes=list(self._notes),
        )
        return BuiltPrompt(messages=messages, manifest=manifest)

    def _window_parts(self, chunks: Iterable[ChunkLike]) -> list[PromptSection]:
        """Chunks grouped by what they are — retrieved, working memory, long-term memory…"""
        groups: dict[str, list[ChunkLike]] = {}
        for chunk in chunks:
            groups.setdefault(self._part_name(chunk), []).append(chunk)
        return [
            self._section(
                name,
                "system",
                self._settings.separator.join(chunk.content for chunk in members),
                ", ".join(dict.fromkeys(chunk.source for chunk in members if chunk.source)),
                False,
            )
            for name, members in groups.items()
        ]

    def _part_name(self, chunk: ChunkLike) -> str:
        parts = self._settings.window_parts
        for prefix, name in parts.by_source.items():
            if chunk.source.startswith(prefix):
                return name
        return parts.by_kind.get(chunk.kind, chunk.kind)

    def _section(
        self, name: str, role: Role, text: str, source: str, stable: bool
    ) -> PromptSection:
        return PromptSection(
            name=name,
            role=role,
            content=text,
            tokens=self._count(text),
            source=source,
            stable=stable,
        )

    def _count(self, text: str) -> int:
        return self._tokenizer.count(text)


def _stable(chunk: ChunkLike) -> bool:
    """Marked by the Cache strategy as the same every turn."""
    return bool(getattr(chunk, "metadata", {}).get(STABLE_MARK))


def render_message(message: Message) -> str:
    """A message as a person reads it in a section: role, text, and any tool calls."""
    lines = [f"{message.role}: {message.content}" if message.content else f"{message.role}:"]
    for call in message.tool_calls or []:
        lines.append(f"  → {call.name}({call.arguments})")
    return "\n".join(lines)


def tool_names(specs: Iterable[Any]) -> list[str]:
    """The function names in OpenAI-style tool specs (or objects with a `name`)."""
    names: list[str] = []
    for tool in specs:
        if isinstance(tool, dict):
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
                continue
        name = getattr(tool, "name", None)
        if name:
            names.append(str(name))
    return names


def manifest_from_messages(
    messages: Sequence[Message],
    tools: Sequence[dict[str, Any]] | None = None,
    tokenizer: Tokenizer | None = None,
) -> PromptManifest:
    """A manifest for a request no builder made: the system message, then each role's turn.

    Coarser than a built one — the system prompt is one section — but every request gets a
    CONTEXT card, and its token counts are the same tokenizer's.
    """
    builder = PromptBuilder(tokenizer)
    names = get_settings().context.prompt.role_sections
    for message in messages:
        if message.role == "system":
            builder.system(INSTRUCTIONS, message.content, stable=True)
        else:
            builder.messages(names.get(message.role, message.role), [message])
    return builder.tools(tools).build().manifest


__all__ = [
    "CONTEXT_WINDOW",
    "HISTORY",
    "INSTRUCTIONS",
    "PAYLOAD",
    "POLICY",
    "PROCEDURES",
    "PROFILE",
    "TOOL_LOOP",
    "USER_QUERY",
    "BuiltPrompt",
    "ChunkLike",
    "PromptBuilder",
    "PromptManifest",
    "PromptPart",
    "PromptSection",
    "TurnLayout",
    "manifest_from_messages",
    "render_message",
    "tool_names",
]
