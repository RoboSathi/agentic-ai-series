"""What the agent holds about each live conversation, between its turns.

Who has verified, the order list they were last shown, and the intent they last asked
for belong to the customer's session, not to any one node: the authentication node writes
them, the intent node reads them, and a rebuilt agent (`/harness`) adopts them. One object
owns them, so every node that needs them is handed the same one.

Each map keeps at most `memory.working.max_sessions` sessions, least recently written
first, so a long-running process does not keep every session it has ever seen.
"""

from collections import OrderedDict
from typing import Generic, TypeVar

from agent_platform.config import get_settings
from customer_support_agent.domain.identity import Customer
from customer_support_agent.intent.router import Intent
from customer_support_agent.presentation.orders import OrderMenu

V = TypeVar("V")


class SessionMap(OrderedDict[str, V], Generic[V]):
    """A session-keyed map that forgets the least recently written past its bound."""

    def __init__(self, max_sessions: int) -> None:
        super().__init__()
        self._max_sessions = max_sessions

    def __setitem__(self, key: str, value: V) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._max_sessions:
            self.popitem(last=False)


class ConversationState:
    """The per-session facts every node and the agent share."""

    def __init__(self, max_sessions: int | None = None) -> None:
        bound = (
            max_sessions if max_sessions is not None else get_settings().memory.working.max_sessions
        )
        # Who each session has proved itself to be. Authentication is per conversation:
        # a customer verifies once and stays verified for that session.
        self.sessions: SessionMap[Customer] = SessionMap(bound)
        # The order list each session was last shown, so a bare "2" resolves to an order.
        self.menus: SessionMap[OrderMenu] = SessionMap(bound)
        self.last_supported_intent: SessionMap[Intent] = SessionMap(bound)

    def adopt(self, previous: "ConversationState") -> None:
        """Take over a previous build's sessions (`adopt_conversation`)."""
        self.sessions.update(previous.sessions)
        self.menus.update(previous.menus)
        self.last_supported_intent.update(previous.last_supported_intent)


__all__ = ["ConversationState", "SessionMap"]
