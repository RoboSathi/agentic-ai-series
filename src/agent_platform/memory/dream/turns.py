"""Turning a completed turn into a dream event — the same way for every agent.

An agent's turn leaves its actions and observations in working memory's
current turn, as `{"kind": "action" | "observation", ...}` items. When the
turn completes, the agent hands them here with what was asked and
answered; the recorder redacts everything personal, builds one event and
writes it to the inbox. The agent says only what is its own: whose turn
it was, the situation (`scope`) and any findings (`signals`).
"""

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from agent_platform.config import get_settings
from agent_platform.memory.dream.events import (
    DreamEvent,
    DreamEventSink,
    DreamSignal,
    MemoryAction,
    MemoryObservation,
)

# The two kinds of current-turn item a dream event keeps.
ACTION = "action"
OBSERVATION = "observation"
# Where an action whose arguments were not a mapping keeps them.
RAW_ARGUMENTS = "raw"


def _keep(text: str) -> str:
    return text


class TurnRecorder:
    """Builds and emits one dream event per completed turn."""

    def __init__(
        self,
        sink: DreamEventSink,
        agent_id: str,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._sink = sink
        self._agent_id = agent_id
        self._redact = redact or _keep
        self._kind = get_settings().memory.dream.turn_event_kind

    @property
    def sink(self) -> DreamEventSink:
        return self._sink

    def record(
        self,
        *,
        session_id: str,
        subject_id: str,
        asked: str,
        answered: str,
        turn_items: Iterable[Any] = (),
        scope: Sequence[str] = (),
        signals: Sequence[DreamSignal] = (),
    ) -> DreamEvent:
        """Emit the turn's event and return it."""
        items = [item for item in turn_items if isinstance(item, dict)]
        event = DreamEvent(
            agent_id=self._agent_id,
            session_id=session_id,
            subject_id=subject_id,
            kind=self._kind,
            user_input=self._redact(asked),
            assistant_output=self._redact(answered),
            observations=[self._observation(i) for i in items if i.get("kind") == OBSERVATION],
            actions=[self._action(i) for i in items if i.get("kind") == ACTION],
            signals=list(signals),
            scope=list(scope),
        )
        self._sink.emit(event)
        return event

    def _observation(self, item: dict[str, Any]) -> MemoryObservation:
        return MemoryObservation(
            content=self._redact(str(item.get("content", ""))), source=str(item.get("source", ""))
        )

    def _action(self, item: dict[str, Any]) -> MemoryAction:
        return MemoryAction(
            name=str(item.get("name", "")),
            arguments=_arguments(item.get("arguments", {})),
            result=self._redact(str(item.get("result", ""))),
            source=str(item.get("source", "")),
        )


def _arguments(arguments: Any) -> dict[str, Any]:
    """An action's arguments as a mapping, whatever shape the agent recorded them in."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return {RAW_ARGUMENTS: str(arguments)}
