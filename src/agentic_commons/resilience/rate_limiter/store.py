"""Where a sliding request window lives (Strategy).

The limiter decides *whether* to wait; the store remembers *what has
already been spent*. Keeping them apart is what lets one limiter serve a
unit test (memory) and a whole workstation (a shared file) unchanged.
"""

import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Protocol, runtime_checkable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rate_window (
    bucket TEXT NOT NULL,
    at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rate_window_bucket ON rate_window (bucket, at);
CREATE TABLE IF NOT EXISTS rate_cooldown (
    bucket TEXT PRIMARY KEY,
    until  REAL NOT NULL
);
"""


@runtime_checkable
class RateWindowStore(Protocol):
    """Records spent requests per bucket and answers "may one more go now?"."""

    def try_acquire(self, bucket: str, limit: int, window_seconds: float, now: float) -> float:
        """Spend one request and return 0.0, or return the seconds to wait."""
        ...

    def penalize(self, bucket: str, until: float) -> None:
        """Block the bucket until `until`, whatever the window says."""
        ...


class InMemoryRateWindowStore:
    """A window visible only to this process — for tests and single scripts."""

    def __init__(self) -> None:
        self._spent: dict[str, deque[float]] = {}
        self._cooldown: dict[str, float] = {}
        self._lock = threading.Lock()

    def try_acquire(self, bucket: str, limit: int, window_seconds: float, now: float) -> float:
        with self._lock:
            blocked_until = self._cooldown.get(bucket, 0.0)
            if blocked_until > now:
                return blocked_until - now
            spent = self._spent.setdefault(bucket, deque())
            while spent and spent[0] <= now - window_seconds:
                spent.popleft()
            if len(spent) < limit:
                spent.append(now)
                return 0.0
            return spent[0] + window_seconds - now

    def penalize(self, bucket: str, until: float) -> None:
        with self._lock:
            self._cooldown[bucket] = max(self._cooldown.get(bucket, 0.0), until)


class SqliteRateWindowStore:
    """A window shared by every process on the machine.

    `BEGIN IMMEDIATE` takes the write lock before reading the count, so two
    processes can never both see "34 spent" and both send the 35th.
    """

    def __init__(self, path: Path, lock_timeout_seconds: float) -> None:
        self._path = path
        self._timeout = lock_timeout_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def try_acquire(self, bucket: str, limit: int, window_seconds: float, now: float) -> float:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            wait = self._wait_in_transaction(connection, bucket, limit, window_seconds, now)
            connection.execute("COMMIT")
            return wait
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def penalize(self, bucket: str, until: float) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO rate_cooldown (bucket, until) VALUES (?, ?) "
                "ON CONFLICT(bucket) DO UPDATE SET until = MAX(until, excluded.until)",
                (bucket, until),
            )
        finally:
            connection.close()

    def _wait_in_transaction(
        self,
        connection: sqlite3.Connection,
        bucket: str,
        limit: int,
        window_seconds: float,
        now: float,
    ) -> float:
        row = connection.execute(
            "SELECT until FROM rate_cooldown WHERE bucket = ?", (bucket,)
        ).fetchone()
        if row is not None and row[0] > now:
            return float(row[0]) - now
        connection.execute(
            "DELETE FROM rate_window WHERE bucket = ? AND at <= ?", (bucket, now - window_seconds)
        )
        count, oldest = connection.execute(
            "SELECT COUNT(*), MIN(at) FROM rate_window WHERE bucket = ?", (bucket,)
        ).fetchone()
        if count < limit:
            connection.execute("INSERT INTO rate_window (bucket, at) VALUES (?, ?)", (bucket, now))
            return 0.0
        return float(oldest) + window_seconds - now

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: transactions are opened explicitly above.
        return sqlite3.connect(self._path, timeout=self._timeout, isolation_level=None)
