"""What the model knows about the session besides the question: its working memory.

*Working memory* is the session's recent turns and its state (`scope_phase`, `run`,
`run_status`, noted by the agent). Nothing from an earlier run is offered: runs are not
kept, and every question is researched fresh from the web.

With the `context` feature, working memory reaches the model through the platform's context
window — the same five-stage pipeline, strategies, health check and card the support agent
uses — so the window is ranked, fitted to its budget and checked before the model sees it.
Withheld, the turns go to the model as the conversation they were and the state as a plain
block.

It is assembled for each question and written back after each answer, so a follow-up
("and tomorrow?") has something to refer to.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field

from agent_platform.context import ContextWindow, Task
from agent_platform.context.types import AssembledContext
from agent_platform.llm.tokenizer import count_tokens
from agent_platform.llm.types import Message
from agent_platform.logging import instrument
from agent_platform.memory import MemoryFacade
from agent_platform.observability import StageReporter
from research_agent.config.settings import AnswerContextSettings, AnswerMessages

# What the context window is assembled for, as the pipeline's `Task.kind`.
ANSWER_TASK = "direct_answer"
Reporter = Callable[[], StageReporter | None]


class SessionContext(BaseModel):
    """The session as one question sees it."""

    # Appended to the system prompt: the session state.
    blocks: list[str] = Field(default_factory=list)
    # With the `context` feature: the assembled window it reaches the model through instead.
    window: AssembledContext | None = None
    # Recent turns, oldest first, between the system prompt and the question.
    history: list[Message] = Field(default_factory=list)
    state: dict[str, str] = Field(default_factory=dict)

    @property
    def turns(self) -> int:
        return len(self.history) // 2

    @property
    def empty(self) -> bool:
        return not (self.history or self.state)

    def tokens(self) -> int:
        """How big the context is, by the shared tokenizer."""
        window = self.window.text if self.window is not None else ""
        texts = [*self.blocks, window, *(message.content for message in self.history)]
        return sum(count_tokens(text) for text in texts)


@instrument("research_agent")
class AnswerContext:
    """Assembles a question's session context, and remembers its answer."""

    def __init__(
        self,
        memory: MemoryFacade | None,
        settings: AnswerContextSettings,
        messages: AnswerMessages,
        window: ContextWindow | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self._memory = memory
        self._settings = settings
        self._messages = messages
        self._window = window
        self._reporter = reporter or (lambda: None)

    async def assemble(self, session_id: str, question: str) -> SessionContext:
        history, state = self._working(session_id)
        if self._window is not None:
            task = Task(query=question, session_id=session_id, kind=ANSWER_TASK)
            result = await self._window.build(task, self._reporter())
            window = result.window if result.text else None
            return SessionContext(window=window, state=state)
        return SessionContext(blocks=self._state_block(state), history=history, state=state)

    def observe_quality(self, session_id: str, score: float) -> None:
        """How well an answer held up, for the window's rot check (nothing without a window)."""
        if self._window is not None:
            self._window.observe_quality(session_id, score)

    def remember(self, session_id: str, question: str, answer: str) -> None:
        """Add one completed turn to the session's working memory."""
        if self._memory is not None and self._memory.working is not None:
            self._memory.working.get(session_id).add_history(question, answer)

    def _state_block(self, state: dict[str, str]) -> list[str]:
        if not state:
            return []
        notes = self._messages
        lines = [notes.state_line.format(key=k, value=v) for k, v in state.items()]
        return ["\n".join([notes.session_state_heading, *lines])]

    def _working(self, session_id: str) -> tuple[list[Message], dict[str, str]]:
        """The recent turns as messages, and the session state."""
        if self._memory is None or self._memory.working is None:
            return [], {}
        memory = self._memory.working.get(session_id)
        keep = self._settings.history_turns * 2
        recent = memory.history[-keep:] if keep and memory.history else []
        history = [Message(**item) for item in recent if isinstance(item, dict)]
        return history, {str(key): str(value) for key, value in memory.state.items()}


__all__ = ["AnswerContext", "SessionContext"]
