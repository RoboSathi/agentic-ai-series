"""The database a command reads from (LLD §20).

Commands default to the configured database, because an operator typing
`/audit` means "the audit trail this deployment has". They accept one
instead when constructed with it, which is what lets a test point the
same command at a fixture database rather than the developer's.

Resolution is deferred to first use rather than done in `__init__`: the
registry builds every command at startup, and opening a connection for
seventeen commands when the operator may run none of them is work nobody
asked for.
"""

from agent_platform.core.persistence import SqliteConnectionPool


class PoolBackedCommand:
    """A command that reads the database."""

    def __init__(self, pool: SqliteConnectionPool | None = None) -> None:
        self._pool = pool

    @property
    def pool(self) -> SqliteConnectionPool:
        """The database to read, opened on first use."""
        if self._pool is None:
            self._pool = SqliteConnectionPool()
        return self._pool
