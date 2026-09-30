"""Episodic memory — what happened (E14, LLD §8).

An append-only JSONL file per session on the filesystem. That is not a
placeholder for a "real" database: it is grep-able by construction,
trivially inspectable by a human at 3am, survives the process, and has
no schema migration story because it never updates a row.

Superseding follows the same rule. When a dream summarises a session, the
turn events it replaced are not edited or removed: a *supersession* record
is appended naming them, and reads leave them out from then on. The log
still holds everything that happened; recall just stops returning noise.

Retrieval follows the NOLLM discipline — a literal substring scan and a
time-range filter come first, and an embedding model is reached for only
when those cannot answer. Cross-session search is grep, literally:
ripgrep when it is installed, the same search in Python when it is not.
"""

import json
from datetime import datetime
from pathlib import Path

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.memory.deterministic import TextSearchBackend, create_text_search
from agentic_commons.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)

# The category of an appended record that retires earlier events.
SUPERSESSION = "supersession"
# Where a supersession record lists the event keys it retires.
_SUPERSEDES = "supersedes"


@instrument("memory")
class EpisodicStore:
    """Append-only session logs, one file per session."""

    _GLOB = "*.jsonl"

    def __init__(
        self, root: Path | None = None, text_search: TextSearchBackend | None = None
    ) -> None:
        settings = get_settings().memory.episodic
        self._root = root or settings.resolved_path
        self._root.mkdir(parents=True, exist_ok=True)
        self._search = text_search or create_text_search()
        self._logger = get_logger("memory")

    @property
    def search_backend(self) -> str:
        """Which grep answers cross-session searches (`ripgrep` or `python`)."""
        return self._search.name

    def search_terms(self, terms: list[str], subject_id: str, limit: int) -> list[MemoryFact]:
        """Events mentioning any term, across every session log, newest first.

        Scoped to one subject: a customer's recall reaches that customer's
        events and no one else's, whichever session they were recorded in.
        """
        matches = self._search.search(self._root, terms, self._GLOB)
        facts = self._live(
            [fact for match in matches if (fact := _parse_line(match.text)) is not None]
        )
        scoped = [fact for fact in facts if fact.subject_id == subject_id]
        scoped.sort(key=lambda fact: fact.event_time, reverse=True)
        return scoped[:limit]

    def session_path(self, session_id: str) -> Path:
        """The log file for one session."""
        return self._root / f"{_safe(session_id)}.jsonl"

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Append one event. Nothing is ever updated or deleted in place."""
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
            for fact in self._read_file(path, query.include_superseded)
            if self._matches(fact, query)
        ]
        facts.sort(key=lambda fact: fact.event_time, reverse=True)
        return facts[: query.limit]

    def purge(self) -> int:
        """Delete every session log. Operator reset only (`make clean-memory`)."""
        logs = list(self._root.glob(self._GLOB))
        for path in logs:
            path.unlink()
        return len(logs)

    def events(self, session_id: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Every event in one session, newest first (superseded ones left out by default)."""
        path = self.session_path(session_id)
        facts = self._read_file(path, include_superseded) if path.exists() else []
        return sorted(facts, key=lambda fact: fact.event_time, reverse=True)

    @logged("memory")
    def supersede(
        self, session_id: str, keys: list[str], by_key: str, provenance: Provenance
    ) -> None:
        """Retire events by appending a record that names them; nothing is rewritten."""
        marker = MemoryFact(
            key=f"{SUPERSESSION}:{by_key}",
            value=by_key,
            memory_type="episodic",
            category=SUPERSESSION,
            metadata={"session_id": session_id, _SUPERSEDES: keys},
            provenance=provenance,
        )
        self.write(marker, provenance)

    def sessions(self) -> list[str]:
        """Every session with a recorded log."""
        return sorted(path.stem for path in self._root.glob("*.jsonl"))

    def grep(self, pattern: str, limit: int = 20) -> list[MemoryFact]:
        """Literal substring search across every session log.

        The plain-text, one-fact-per-line format is what makes this a
        one-liner rather than a query planner (McIlroy's "do one thing").
        """
        matches = self._search.search(self._root, [pattern], self._GLOB)
        found = self._live(
            [fact for match in matches if (fact := _parse_line(match.text)) is not None]
        )
        return found[:limit]

    def _live(self, facts: list[MemoryFact]) -> list[MemoryFact]:
        """Grep hits minus supersession records and the events they retired."""
        retired: dict[str, set[str]] = {}
        live = []
        for fact in facts:
            if fact.category == SUPERSESSION:
                continue
            session = str(fact.metadata.get("session_id", ""))
            if session not in retired:
                path = self.session_path(session)
                retired[session] = _retired(self._read_all(path)) if path.exists() else set()
            if fact.key not in retired[session]:
                live.append(fact)
        return live

    def _files_for(self, session_id: str) -> list[Path]:
        """Log files to scan for a query."""
        if session_id:
            path = self.session_path(session_id)
            return [path] if path.exists() else []
        return sorted(self._root.glob("*.jsonl"))

    def _read_file(self, path: Path, include_superseded: bool = False) -> list[MemoryFact]:
        """One session's events: supersession records never, retired events on request."""
        facts = self._read_all(path)
        retired = set() if include_superseded else _retired(facts)
        return [fact for fact in facts if fact.category != SUPERSESSION and fact.key not in retired]

    def _read_all(self, path: Path) -> list[MemoryFact]:
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


def _retired(facts: list[MemoryFact]) -> set[str]:
    """Every event key some supersession record in this log retires."""
    return {
        str(key)
        for fact in facts
        if fact.category == SUPERSESSION
        for key in fact.metadata.get(_SUPERSEDES, [])
    }


def _parse_line(line: str) -> MemoryFact | None:
    """One log line as an event, or None if the line is not one."""
    try:
        return MemoryFact.model_validate(json.loads(line))
    except (ValueError, TypeError):
        return None


def _within(moment: datetime, since: datetime | None, until: datetime | None) -> bool:
    """Whether a timestamp falls inside an optional range."""
    if since is not None and moment < since:
        return False
    return not (until is not None and moment > until)


def _safe(session_id: str) -> str:
    """A filename-safe session id — a session id is user-influenced input."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in session_id)
