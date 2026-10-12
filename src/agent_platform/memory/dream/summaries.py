"""One summary per session: exact outcomes, plus a checked sentence of intent.

The outcome part is a template over the structured actions — what was
called, with what, and the first sentence of what it returned — so it
cannot state an outcome that did not happen. The intent part is a
generated sentence ("the customer wanted …") and is kept only when the
NLI checker finds it entailed by the conversation itself. A summary that
misremembers is worse than none: recall would feed it back as fact.
"""

import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.memory.dream.events import DreamEvent, MemoryAction
from agent_platform.scoring.entailment import EntailmentModel

# The category a dream gives the summary that stands for a session.
SUMMARY_CATEGORY = "session_summary"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
# Between a tool's namespace and its name: `support.x` in the registry,
# `support__x` as the model's function name.
_NAMESPACE = re.compile(r"^.*(?:\.|__)")
_SPEAKERS = ("Customer", "Agent")
_WHITESPACE = re.compile(r"\s+")
# A token that reads as a word or a number, once edge punctuation is off.
_WORD = re.compile(r"[\w$][\w$.,'/-]*")
_EDGE_PUNCTUATION = "()[]{}\"',.;:!?–—-"


@runtime_checkable
class DialogueSummarizer(Protocol):
    """Turns a transcript into one short summary, or None if it cannot."""

    async def summarize(self, dialogue: str) -> str | None: ...


class SessionSummary(BaseModel):
    """What a dream will remember about one session."""

    text: str
    outcomes: list[str]
    # None: no intent sentence was attempted. False: one was, and NLI refused it.
    intent_kept: bool | None = None


class SessionSummarizer:
    """Builds a session's summary from its turn events.

    With no dialogue summarizer it is the plain template — exact outcomes
    only — which is the default any agent gets; an agent that adds a
    generator must add the entailment check that keeps it honest.
    """

    def __init__(
        self,
        entailment: EntailmentModel | None,
        threshold: float,
        outcome_chars: int,
        summarizer: DialogueSummarizer | None = None,
        lookup_tools: list[str] | None = None,
        forbidden_words: list[str] | None = None,
        min_word_ratio: float = 0.0,
    ) -> None:
        # Words a generated sentence may not use — gendered pronouns: the
        # summarizer cannot know them, and memory must not invent them.
        self._forbidden = frozenset(word.casefold() for word in forbidden_words or [])
        # Below this share of real words, the output is decoding noise.
        self._min_word_ratio = min_word_ratio
        # Read-only tools: condensed to what was looked up, never quoted.
        self._lookups = frozenset(lookup_tools or [])
        self._entailment = entailment
        self._threshold = threshold
        self._outcome_chars = outcome_chars
        self._summarizer = summarizer

    async def summarize(self, events: list[DreamEvent]) -> SessionSummary | None:
        """The session's summary, or None when nothing was done worth remembering."""
        actions = [action for event in events for action in event.actions]
        if not actions:
            return None
        looked_up = _looked_up([a for a in actions if tool_name(a.name) in self._lookups])
        outcomes = [self._outcome(a) for a in actions if tool_name(a.name) not in self._lookups]
        intent, kept = await self._intent(events)
        parts = [intent] if intent else []
        if looked_up:
            parts.append(f"Looked up {looked_up}.")
        if outcomes:
            parts.append(f"Outcomes: {'; '.join(outcomes)}.")
        return SessionSummary(text=" ".join(parts), outcomes=outcomes, intent_kept=kept)

    def _outcome(self, action: MemoryAction) -> str:
        """`tool (order_id 10441) → first sentence of its result`, exactly as recorded."""
        tool = tool_name(action.name)
        arguments = ", ".join(
            f"{key} {value}" for key, value in action.arguments.items() if key != "raw"
        )
        result = _SENTENCE_END.split(str(action.result or "").strip(), maxsplit=1)[0].rstrip(".")
        result = (
            result
            if len(result) <= self._outcome_chars
            else result[: self._outcome_chars - 1] + "…"
        )
        called = f"{tool} ({arguments})" if arguments else tool
        return f"{called} → {result}" if result else called

    async def _intent(self, events: list[DreamEvent]) -> tuple[str, bool | None]:
        """A generated sentence of intent, kept only if the transcript entails it."""
        if self._summarizer is None or self._entailment is None:
            return "", None
        dialogue = _transcript(events)
        generated = await self._summarizer.summarize(dialogue)
        if not generated:
            return "", None
        # One sentence, checked on its own. A multi-sentence hypothesis
        # passed NLI with an invented clause in its second half.
        sentence = _WHITESPACE.sub(" ", _SENTENCE_END.split(generated.strip(), maxsplit=1)[0])
        if not self._plausible(sentence):
            return "", False
        verdict = await self._entailment.entails(dialogue, sentence)
        if verdict.label == "entailment" and verdict.score >= self._threshold:
            return sentence.rstrip(".") + ".", True
        return "", False

    def _plausible(self, sentence: str) -> bool:
        """Mostly words, and none it may not use."""
        tokens = sentence.split()
        if not tokens:
            return False
        words = [token for token in tokens if _WORD.fullmatch(token.strip(_EDGE_PUNCTUATION))]
        if len(words) / len(tokens) < self._min_word_ratio:
            return False
        return not any(
            token.strip(_EDGE_PUNCTUATION).casefold() in self._forbidden for token in tokens
        )


def tool_name(qualified: str) -> str:
    """A tool's own name, without its namespace."""
    return _NAMESPACE.sub("", qualified)


def _looked_up(lookups: list[MemoryAction]) -> str:
    """`order_id 10482, 10216` — every value each lookup was asked about, once."""
    values: dict[str, list[str]] = {}
    for action in lookups:
        for key, value in action.arguments.items():
            seen = values.setdefault(key, [])
            if key != "raw" and str(value) not in seen:
                seen.append(str(value))
    named = "; ".join(f"{key} {', '.join(found)}" for key, found in values.items() if found)
    # A lookup with no arguments (list every order) still happened.
    bare = sorted({tool_name(a.name) for a in lookups if not a.arguments})
    return "; ".join(part for part in (named, ", ".join(bare)) if part)


def _transcript(events: list[DreamEvent]) -> str:
    """The session as a two-speaker dialogue, the shape the summarizer was trained on."""
    customer, agent = _SPEAKERS
    return "\n".join(
        f"{customer}: {event.user_input}\n{agent}: {event.assistant_output}" for event in events
    )
