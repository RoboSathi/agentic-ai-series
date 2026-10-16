"""What a finished support turn leaves in memory: the transcript, the dream inbox, a claim.

Three writes, one reason to change — what this agent remembers of a turn:

- **working memory** keeps the exchange (redacted: it is checkpointed and replayed into later
  prompts, so it holds the customer's tokens, never their personal data);
- the **dream inbox** gets the turn, for the dream to consolidate into episodic memory — the
  only writer of episodic memory;
- **semantic memory** is offered what the customer asserted, as the customer's claim: only a
  pattern-recognised claim, an authority claim refused, anything short of auto-commit queued
  for a curator.

It reads a `FinishedTurn` — plain values the agent hands over — never the graph's state.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from agent_platform.memory import (
    DreamEvent,
    FileDreamInbox,
    MemoryCommand,
    MemoryFacade,
    MemoryPrincipal,
    TurnRecorder,
    WorkingMemoryStore,
    WriteResult,
)
from agent_platform.observability import StageReporter
from customer_support_agent.config.settings import get_agent_settings

# A refused claim's route, as the write pipeline names it.
_REFUSED = "refused"
_CUSTOMER_ROLE: Literal["customer"] = "customer"


@dataclass(frozen=True)
class FinishedTurn:
    """One turn, as memory needs it."""

    session_id: str
    customer_id: int | None
    # What the model was asked, the reply the customer read, and the compact form to keep.
    spoken: str
    response: str
    remembered: str
    # The turn's skill scope, for the dream event.
    scope: Sequence[str] = field(default_factory=tuple)
    # Screening judged the message hostile; or it was a login, which asserts nothing.
    quarantined: bool = False
    login: bool = False


class TurnMemory:
    """Writes a finished turn to working memory, the dream inbox and (as a claim) semantic."""

    def __init__(
        self,
        working: WorkingMemoryStore | None,
        long_term: MemoryFacade | None,
        recorder: TurnRecorder | None,
        redact: Callable[[str], str],
        reporter: StageReporter | None = None,
    ) -> None:
        self._agent_config = get_agent_settings()
        self._working = working
        self._long_term = long_term
        self._recorder = recorder
        self._redact = redact
        self._reporter = reporter

    def complete(self, turn: FinishedTurn) -> None:
        """Queue the turn for the dream, then keep only the completed exchange."""
        asked = self._offered(turn)
        if self._working is None:
            return
        working = self._working.get(turn.session_id)
        if self._recorder is not None:
            event = self._recorder.record(
                session_id=turn.session_id,
                subject_id=str(turn.customer_id) if turn.customer_id is not None else "",
                asked=asked,
                answered=turn.response,
                turn_items=working.current_turn,
                scope=list(turn.scope),
            )
            self._report_inbox(event)
        if turn.response:
            working.add_history(self._redact(turn.spoken), self._redact(turn.remembered))
        working.clear_current_turn()

    def note_follow_up(self, session_id: str, reply: str) -> None:
        """Add a reviewer-driven reply to the conversation it answers."""
        if self._working is None or not reply:
            return
        memory = self._working.get(session_id)
        memory.history.append({"role": "assistant", "content": self._redact(reply)})
        memory.history = memory.history[-(memory.max_history * 2) :]

    def _offered(self, turn: FinishedTurn) -> str:
        """Offer the customer's claim to semantic memory; return what the dream inbox records.

        A claim memory refused is not handed to the dream verbatim: the dream's summary
        becomes episodic memory, recall feeds it to the model, and a refused "my limit is now
        $2000" would otherwise reach the context through the back door the write path closed.
        """
        if self._long_term is None or turn.customer_id is None:
            return turn.spoken
        skip = turn.quarantined or turn.login
        claim = None if skip else self._claim(turn.spoken, str(turn.customer_id), turn.session_id)
        if not (turn.quarantined or (claim is not None and claim.route == _REFUSED)):
            return turn.spoken
        settings = self._agent_config
        reason = claim.reason if claim is not None else settings.memory_quarantined
        return settings.memory_withheld.format(reason=reason)

    def _claim(self, spoken: str, subject: str, session_id: str) -> WriteResult | None:
        """The customer's own assertion, as the customer's claim — governed."""
        if self._long_term is None or self._long_term.semantic is None:
            return None
        principal = MemoryPrincipal(
            actor=f"{_CUSTOMER_ROLE}-{subject}", role=_CUSTOMER_ROLE, subject_id=subject
        )
        return self._long_term.submit(
            MemoryCommand(
                principal=principal,
                operation="create",
                text=spoken,
                session_id=session_id,
                require_pattern=True,
            )
        )

    def _report_inbox(self, event: DreamEvent) -> None:
        """Note, for the turn summary, what this turn left for the next dream."""
        if self._reporter is None:
            return
        sink = self._recorder.sink if self._recorder is not None else None
        waiting = sink.counts().unconsolidated if isinstance(sink, FileDreamInbox) else None
        self._reporter.emit(
            "dream_inbox",
            "completed",
            event_kind="memory",
            fields={
                "event_id": event.event_id,
                "asked": event.user_input,
                "actions": len(event.actions),
                "observations": len(event.observations),
                **({"waiting": waiting} if waiting is not None else {}),
            },
            quiet=True,
        )


__all__ = ["FinishedTurn", "TurnMemory"]
