"""Episodic memory — what happened (E14, LLD §8).

An append-only JSONL file per session on the filesystem. That is not a
placeholder for a "real" database: it is grep-able by construction,
trivially inspectable by a human at 3am, survives the process, and has
no schema migration story because it never updates a row.

Retrieval follows the NOLLM discipline — a literal substring scan and a
time-range filter come first, and an embedding model is reached for only
when those cannot answer.
"""

import json
from datetime import datetime
from pathlib import Path

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)


@instrument("memory")
class EpisodicStore:
    """Append-only session logs, one file per session."""

    def __init__(self, root: Path | None = None) -> None:
        settings = get_settings().memory.episodic
        self._root = root or settings.resolved_path
        self._enabled = settings.enabled
        self._root.mkdir(parents=True, exist_ok=True)
        self._logger = get_logger("memory")

    def session_path(self, session_id: str) -> Path:
        """The log file for one session."""
        return self._root / f"{_safe(session_id)}.jsonl"

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Append one event. Nothing is ever updated or deleted in place."""
        if not self._enabled:
            return WriteResult(route="discard", written=False, reason="episodic memory disabled")
        recorded = fact.model_copy(update={"memory_type": "episodic", "provenance": provenance})
        session_id = recorded.metadata.get("session_id", "default")
        with self.session_path(str(session_id)).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(recorded.model_dump(mode="json")) + "\n")
        return WriteResult(route="auto_commit", written=True, fact=recorded, reason="appended")

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Events matching the query, newest first.

        Cheap filters run before expensive ones: session file selection,
        then a literal substring scan, then the time range. No model is
        involved, which is the point.
        """
        facts = [
            fact
            for path in self._files_for(query.session_id)
            for fact in self._read_file(path)
            if self._matches(fact, query)
        ]
        facts.sort(key=lambda fact: fact.event_time, reverse=True)
        return facts[: query.limit]

    def sessions(self) -> list[str]:
        """Every session with a recorded log."""
        return sorted(path.stem for path in self._root.glob("*.jsonl"))

    def grep(self, pattern: str, limit: int = 20) -> list[MemoryFact]:
        """Literal substring search across every session log.

        The plain-text, one-fact-per-line format is what makes this a
        one-liner rather than a query planner (McIlroy's "do one thing").
        """
        needle = pattern.casefold()
        found = [
            fact
            for path in sorted(self._root.glob("*.jsonl"))
            for fact in self._read_file(path)
            if needle in fact.value.casefold() or needle in fact.key.casefold()
        ]
        return found[:limit]

    def _files_for(self, session_id: str) -> list[Path]:
        """Log files to scan for a query."""
        if session_id:
            path = self.session_path(session_id)
            return [path] if path.exists() else []
        return sorted(self._root.glob("*.jsonl"))

    def _read_file(self, path: Path) -> list[MemoryFact]:
        """Parse one log file, skipping any line that cannot be read.

        A corrupt line loses one event; refusing to parse the file would
        lose the whole session, which is the worse failure for a log whose
        job is forensic.
        """
        facts: list[MemoryFact] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                facts.append(MemoryFact.model_validate(json.loads(line)))
            except (ValueError, TypeError):
                self._logger.warning("Skipping unreadable episodic line", file=str(path))
        return facts

    @staticmethod
    def _matches(fact: MemoryFact, query: MemoryQuery) -> bool:
        """Whether one event satisfies every filter in the query."""
        if query.text and query.text.casefold() not in fact.value.casefold():
            return False
        if query.category and fact.category != query.category:
            return False
        return _within(fact.event_time, query.since, query.until)


def _within(moment: datetime, since: datetime | None, until: datetime | None) -> bool:
    """Whether a timestamp falls inside an optional range."""
    if since is not None and moment < since:
        return False
    return not (until is not None and moment > until)


def _safe(session_id: str) -> str:
    """A filename-safe session id — a session id is user-influenced input."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
