"""What the model knows about the session besides the question: working memory, and earlier research.

*Working memory* is the session's recent turns and its state (`scope_phase`, `run`,
`run_status`, noted by the agent); the turns go to the model as the conversation they were.
*Earlier research* is the archive's run index: a few
runs whose topic shares enough words with the question, offered as pointers ("open it with
/show"), never as facts — a pointer is not evidence, and nothing here can ground a claim.

Both are assembled for each question and written back after each answer, so a follow-up ("and
tomorrow?") has something to refer to.
"""

import re

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.llm.types import Message
from agentic_commons.logging import instrument
from agentic_commons.memory import MemoryFacade
from research_agent.archive.store import ResearchArchive, RunIndexRow
from research_agent.config.settings import AnswerContextSettings, AnswerMessages

_WORD = re.compile(r"\w+")
_ROLE_NAMES = {"user": "User", "assistant": "Assistant"}


class SessionContext(BaseModel):
    """The session as one question sees it."""

    # Appended to the system prompt: the session state, then earlier research.
    blocks: list[str] = Field(default_factory=list)
    # Recent turns, oldest first, between the system prompt and the question.
    history: list[Message] = Field(default_factory=list)
    state: dict[str, str] = Field(default_factory=dict)
    runs: list[str] = Field(default_factory=list)

    @property
    def turns(self) -> int:
        return len(self.history) // 2

    @property
    def empty(self) -> bool:
        return not (self.history or self.state or self.runs)

    def tokens(self) -> int:
        """How big the context is, in the platform's own estimate."""
        characters = sum(len(block) for block in self.blocks) + sum(
            len(message.content) for message in self.history
        )
        return characters // get_settings().context.chars_per_token


@instrument("research_agent")
class AnswerContext:
    """Assembles a question's session context, and remembers its answer."""

    def __init__(
        self,
        memory: MemoryFacade | None,
        archive: ResearchArchive,
        settings: AnswerContextSettings,
        messages: AnswerMessages,
    ) -> None:
        self._memory = memory
        self._archive = archive
        self._settings = settings
        self._messages = messages

    async def assemble(self, session_id: str, question: str) -> SessionContext:
        history, state = self._working(session_id)
        runs = self._earlier_runs(question)
        return SessionContext(
            blocks=self._blocks(state, runs),
            history=history,
            state=state,
            runs=[run.run_id for run in runs],
        )

    def remember(self, session_id: str, question: str, answer: str) -> None:
        """Add one completed turn to the session's working memory."""
        if self._memory is not None and self._memory.working is not None:
            self._memory.working.get(session_id).add_history(question, answer)

    def _blocks(self, state: dict[str, str], runs: list[RunIndexRow]) -> list[str]:
        notes = self._messages
        blocks = []
        if state:
            lines = [notes.state_line.format(key=k, value=v) for k, v in state.items()]
            blocks.append("\n".join([notes.session_state_heading, *lines]))
        if runs:
            lines = [
                notes.earlier_line.format(run_id=r.run_id, status=r.status, topic=r.topic)
                for r in runs
            ]
            blocks.append("\n".join([notes.earlier_heading, *lines]))
        return blocks

    def _working(self, session_id: str) -> tuple[list[Message], dict[str, str]]:
        """The recent turns as messages, and the session state."""
        if self._memory is None or self._memory.working is None:
            return [], {}
        memory = self._memory.working.get(session_id)
        keep = self._settings.history_turns * 2
        recent = memory.history[-keep:] if keep and memory.history else []
        history = [Message(**item) for item in recent if isinstance(item, dict)]
        return history, {str(key): str(value) for key, value in memory.state.items()}

    def _earlier_runs(self, question: str) -> list[RunIndexRow]:
        """Archived runs whose topic shares enough words with the question, best match first."""
        if not self._settings.recall_limit:
            return []
        wanted = self._words(question)
        scored = [
            (len(wanted & self._words(f"{row.topic} {row.archetype}")), row)
            for row in self._archive.history(self._settings.recall_scan)
        ]
        ranked = sorted(
            (item for item in scored if item[0] >= self._settings.min_overlap),
            key=lambda item: -item[0],
        )
        return [row for _, row in ranked[: self._settings.recall_limit]]

    def _words(self, text: str) -> set[str]:
        floor = self._settings.min_word_chars
        return {word for word in _WORD.findall(text.casefold()) if len(word) >= floor}


__all__ = ["AnswerContext", "SessionContext"]
