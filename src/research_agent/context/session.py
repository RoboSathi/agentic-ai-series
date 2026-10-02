"""What the model knows about the session besides the question: working memory, and earlier research.

*Working memory* is the session's recent turns and its state (`scope_phase`, `run`,
`run_status`, noted by the agent). *Earlier research* is the archive's run index: a few runs
whose topic shares enough words with the question, offered as pointers ("open it with
/show"), never as facts.

With the `context` feature, both reach the model through the platform's context window —
the same five-stage pipeline, strategies, health check and card the support agent uses — so
the window is ranked, fitted to its budget and checked before the model sees it. Withheld,
the turns go to the model as the conversation they were and the rest as plain blocks.

Both are assembled for each question and written back after each answer, so a follow-up
("and tomorrow?") has something to refer to.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.context import ContextWindow, Task
from agentic_commons.llm.types import Message
from agentic_commons.logging import instrument
from agentic_commons.memory import MemoryFacade
from agentic_commons.observability import StageReporter
from research_agent.config.settings import AnswerContextSettings, AnswerMessages
from research_agent.context.earlier_runs import EarlierRunsSource

# What the context window is assembled for, as the pipeline's `Task.kind`.
ANSWER_TASK = "direct_answer"
Reporter = Callable[[], StageReporter | None]


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
        earlier: EarlierRunsSource,
        settings: AnswerContextSettings,
        messages: AnswerMessages,
        window: ContextWindow | None = None,
        reporter: Reporter | None = None,
    ) -> None:
        self._memory = memory
        self._earlier = earlier
        self._settings = settings
        self._messages = messages
        self._window = window
        self._reporter = reporter or (lambda: None)

    async def assemble(self, session_id: str, question: str) -> SessionContext:
        history, state = self._working(session_id)
        earlier = self._earlier.matching(question)
        runs = [run.run_id for run in earlier]
        if self._window is not None:
            task = Task(query=question, session_id=session_id, kind=ANSWER_TASK)
            result = await self._window.build(task, self._reporter())
            return SessionContext(
                blocks=[result.text] if result.text else [], state=state, runs=runs
            )
        pointers = self._earlier.block(earlier)
        blocks = [*self._state_block(state), *([pointers] if pointers else [])]
        return SessionContext(blocks=blocks, history=history, state=state, runs=runs)

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
