"""Whose data a console command may show: the session's verified subject, and no one else's.

A command like `/memory` or `/review` reads stored records about people. Run
from a console where someone has proved who they are, it shows *their*
records — not everyone's. The console owns one `ConsoleScope`: the session
it is on, and (when the agent can say) who is verified in it. The platform
never learns what a subject is; it is the opaque `subject_id` string every
record already carries.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable

# Who is verified in a session, or None: the agent's answer (`SubjectHost`).
SubjectResolver = Callable[[str], str | None]


class ConsoleScope:
    """The console's live session, and the subject verified in it."""

    def __init__(self, session: Callable[[], str], resolver: SubjectResolver | None = None) -> None:
        self._session = session
        self._resolver = resolver

    @property
    def session_id(self) -> str:
        """The session the console is on now (it can switch)."""
        return self._session()

    @property
    def scoped(self) -> bool:
        """Whether this console shows only a verified subject's records."""
        return self._resolver is not None

    def subject(self) -> str | None:
        """Who is verified in the current session, or None when nobody is yet."""
        return self._resolver(self.session_id) if self._resolver is not None else None


@runtime_checkable
class ScopedCommand(Protocol):
    """A command that narrows what it shows to the console's scope."""

    def bind_scope(self, scope: ConsoleScope) -> None: ...


__all__ = ["ConsoleScope", "ScopedCommand", "SubjectResolver"]
