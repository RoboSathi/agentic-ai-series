"""Small, session-scoped working memory for live conversations."""

from typing import Any

from pydantic import BaseModel, Field


class WorkingMemory(BaseModel):
    """The bounded state needed to continue one live conversation."""

    history: list[Any] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)
    current_turn: list[Any] = Field(default_factory=list)
    max_history: int = 6

    def get(self, key: str, default: Any = None) -> Any:
        """Return one active task value."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set one active task value."""
        self.state[key] = value

    def add_history(self, user_message: str, assistant_message: str) -> None:
        """Append one completed customer/assistant turn, bounded by turns."""
        self.history.extend(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message},
            ]
        )
        self.history = self.history[-(self.max_history * 2) :]

    def add_current_turn(self, item: Any) -> None:
        """Append one item produced while processing the current turn."""
        self.current_turn.append(item)

    def current_messages(self) -> list[Any]:
        """Return message-shaped entries from the current turn."""
        return [
            item
            for item in self.current_turn
            if getattr(item, "role", None) in {"system", "user", "assistant", "tool"}
        ]

    def clear_current_turn(self) -> None:
        """Discard transient current-turn data after the turn is emitted."""
        self.current_turn.clear()


class WorkingMemoryStore:
    """Keep one small working-memory object per live session."""

    def __init__(self, max_history: int = 6) -> None:
        self._max_history = max_history
        self._sessions: dict[str, WorkingMemory] = {}

    def get(self, session_id: str) -> WorkingMemory:
        """Return the existing session memory or create it."""
        key = session_id or "default"
        memory = self._sessions.get(key)
        if memory is None:
            memory = WorkingMemory(max_history=self._max_history)
            self._sessions[key] = memory
        return memory

    def clear(self, session_id: str | None = None) -> None:
        """Clear one session or every session."""
        if session_id is None:
            self._sessions.clear()
            return
        self._sessions.pop(session_id or "default", None)

    def sessions(self) -> list[str]:
        """Return sessions currently holding working memory."""
        return sorted(self._sessions)
