"""Semantic memory — what is generally true (E14, LLD §8, ADR 0011).

Files, one directory per agent. Each fact is about a subject (a customer,
say) or about the agent itself, and lives in
`semantic/<subject>/<key>.md`: a readable file with the fact's
provenance in its frontmatter and its value as the body. `MEMORY.md`
lists them all. Those files are *views*: the truth is the numbered
records under `.history/`, one per version ever written.

It is a source of truth, so nothing is ever overwritten:

  * an update is a new **version** whose record supersedes the last;
  * a delete is a **tombstone** — out of recall, still in history;
  * only an **erase** removes the records, for data-protection requests.

Reads are tiered cheapest-first (R1 exact → R2 BM25 over a derived FTS5
index), and recall **reinforces** what it used (`access_count`,
`last_accessed_at`).
"""

import difflib
from collections import defaultdict
from pathlib import Path
from typing import Any

from agent_platform.config import get_settings
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.memory.files import (
    DerivedIndex,
    IndexedDoc,
    IndexEntry,
    MemoryDirectory,
    MemoryDocument,
    Record,
    RecordLog,
    render_document,
    safe_name,
    write_index,
)
from agent_platform.memory.files.records import SEQUENCE_FIELD
from agent_platform.memory.types import (
    MemoryFact,
    MemoryQuery,
    Provenance,
    WriteResult,
)
from agent_platform.observability.types import utc_now

# The store's name on its MEMORY.md title.
_STORE_TITLE = "Semantic"


@instrument("memory")
class SemanticStore:
    """Structured facts with exact-then-fuzzy key lookup, one agent's directory."""

    def __init__(self, root: Path | None = None, *, agent_id: str | None = None) -> None:
        settings = get_settings()
        self._files = settings.memory.files
        self._agent_id = agent_id or settings.agent.id
        self._fuzzy_threshold = settings.memory.semantic.fuzzy_match_threshold
        self._dir = MemoryDirectory(root or self._files.semantic_root, self._files.lock_file)
        self._log = RecordLog(self._dir, self._files)
        self._index = DerivedIndex(self._dir.path(self._files.derived_index_file))
        self._logger = get_logger("memory")

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def root(self) -> Path:
        return self._dir.root

    def purge(self) -> int:
        """Remove every fact, version and view. Operator reset only (`make clean-memory`).

        Deliberately outside the write pipeline: this wipes a demo back to
        empty, not a memory operation with a principal.
        """
        with self._dir.locked():
            removed = len(self._log.records())
            for child in self._dir.children():
                self._dir.remove_tree(child.name)
            self._dir.delete(self._files.index_file)
            self._index.rebuild([], self._log.fingerprint())
        return removed

    @logged("memory")
    def write(self, fact: MemoryFact, provenance: Provenance) -> WriteResult:
        """Store a fact. Conflicts are the write pipeline's business, not ours."""
        stored = fact.model_copy(update={"memory_type": "semantic", "provenance": provenance})
        with self._dir.locked():
            self._log.append(self._group(stored.subject_id, stored.key), _document(stored))
            self._refresh(stored.subject_id, stored.key)
        return WriteResult(route="auto_commit", written=True, fact=stored, reason="committed")

    def latest(self, key: str, subject_id: str = "") -> MemoryFact | None:
        """The current version of one subject's fact, if there is one."""
        current = [
            fact for _, fact in self._current(self._group(subject_id, key)) if fact.key == key
        ]
        return current[-1] if current else None

    def exact(
        self, keys: list[str], identifiers: list[str], subject_id: str, limit: int
    ) -> list[MemoryFact]:
        """R1 — facts whose key is asked for, or whose value names an identifier."""
        if not keys and not identifiers:
            return []
        hits = [
            fact
            for _, fact in self._current((self._subject_dir(subject_id),))
            if fact.subject_id == subject_id
            and (fact.key in keys or any(item in fact.value for item in identifiers))
        ]
        return list(reversed(hits))[:limit]

    def lexical(self, text: str, subject_id: str, limit: int) -> list[tuple[MemoryFact, float]]:
        """R2 — BM25 over keys and values, best first, with a [0, 1] score."""
        self._ensure_index()
        found: list[tuple[MemoryFact, float]] = []
        for key, score in self._index.search(text, subject_id, limit):
            current = [fact for _, fact in self._current(self._group(subject_id, key))]
            if current:
                found.append((current[-1], score))
        return found

    def tombstone(self, key: str, subject_id: str, actor: str) -> int:
        """Delete a fact the way a source of truth deletes: keep it, stop believing it."""
        stamp = {"deleted_at": utc_now().isoformat(), "deleted_by": actor}
        return self._mark(subject_id, key, stamp)

    def erase(self, key: str, subject_id: str) -> int:
        """Remove every version of a fact — data-protection erasure, not a delete."""
        with self._dir.locked():
            removed = self._log.remove(self._group(subject_id, key))
            self._refresh(subject_id, key)
        self._logger.info("Erased semantic fact", key=key, removed=removed)
        return removed

    def reinforce(self, facts: list[MemoryFact]) -> None:
        """Record that recall used these facts (frequency and recency of use)."""
        if not facts:
            return
        now = utc_now().isoformat()
        with self._dir.locked():
            for fact in facts:
                for record, current in self._current(self._group(fact.subject_id, fact.key)):
                    self._log.rewrite(
                        record,
                        {"access_count": current.access_count + 1, "last_accessed_at": now},
                    )
                self._write_view(fact.subject_id, fact.key)

    def read(self, query: MemoryQuery) -> list[MemoryFact]:
        """Facts for a query, cheapest retrieval tier first."""
        if query.key:
            exact = self.get(query.key, query.include_superseded)
            if exact:
                return exact[: query.limit]
            fuzzy = self.fuzzy_get(query.key, query.include_superseded)
            if fuzzy:
                return fuzzy[: query.limit]
        return self.search(query)[: query.limit]

    def get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 1 — exact key match across subjects, newest first."""
        facts = [fact for fact in self._facts(include_superseded) if fact.key == key]
        return list(reversed(facts))

    def search(self, query: MemoryQuery) -> list[MemoryFact]:
        """Tier 2 — literal substring scan over keys and values, newest first."""
        needle = query.text.casefold()
        facts = [
            fact
            for fact in self._facts(query.include_superseded)
            if (not needle or needle in fact.value.casefold() or needle in fact.key.casefold())
            and (not query.category or fact.category == query.category)
            and (not query.subject_id or fact.subject_id == query.subject_id)
        ]
        return list(reversed(facts))

    def fuzzy_get(self, key: str, include_superseded: bool = False) -> list[MemoryFact]:
        """Tier 3 — closest stored key above the configured similarity floor."""
        matches = difflib.get_close_matches(key, self.keys(), n=1, cutoff=self._fuzzy_threshold)
        if not matches:
            return []
        self._logger.debug("Semantic fuzzy key match", requested=key, matched=matches[0])
        return self.get(matches[0], include_superseded)

    def keys(self) -> list[str]:
        """Every distinct key currently believed."""
        return list(dict.fromkeys(fact.key for fact in self._facts(include_superseded=False)))

    def supersede(self, key: str, superseded_by: str, subject_id: str | None = None) -> int:
        """Mark existing facts for a key as replaced, keeping them readable.

        With a subject, only that subject's fact; without one, every
        subject's (the legacy, agent-wide behaviour).
        """
        subjects = (
            [subject_id]
            if subject_id is not None
            else sorted({fact.subject_id for fact in self.get(key)})
        )
        return sum(
            self._mark(subject, key, {"superseded_by": superseded_by}) for subject in subjects
        )

    def reindex(self) -> int:
        """Rebuild the derived search index from the records; returns how many facts it holds."""
        with self._dir.locked():
            return self._rebuild_index()

    # ── records and views ────────────────────────────────────────────

    def _group(self, subject_id: str, key: str) -> tuple[str, str]:
        return (self._subject_dir(subject_id), key)

    def _subject_dir(self, subject_id: str) -> str:
        return subject_id or self._files.shared_subject_dir

    def _current(self, group: tuple[str, ...]) -> list[tuple[Record, MemoryFact]]:
        """(record, fact) for every current version in a group, oldest first."""
        pairs = [(record, _fact(record)) for record in self._log.records(group)]
        return [(record, fact) for record, fact in pairs if fact.is_current]

    def _facts(self, include_superseded: bool) -> list[MemoryFact]:
        facts = [_fact(record) for record in self._log.records()]
        return facts if include_superseded else [fact for fact in facts if fact.is_current]

    def _mark(self, subject_id: str, key: str, meta: dict[str, Any]) -> int:
        """Stamp every current version of one fact; returns how many."""
        with self._dir.locked():
            current = self._current(self._group(subject_id, key))
            for record, _ in current:
                self._log.rewrite(record, meta)
            self._refresh(subject_id, key)
        return len(current)

    def _refresh(self, subject_id: str, key: str) -> None:
        """After a change to one fact: its view, its index entry, MEMORY.md. Under the lock."""
        current = self._write_view(subject_id, key)
        doc_id = self._log.group_path(self._group(subject_id, key))
        if current is None:
            self._index.remove(doc_id)
        else:
            self._index.replace(_indexed(doc_id, current))
        self._index.mark(self._log.fingerprint())
        self._write_memory_index()

    def _write_view(self, subject_id: str, key: str) -> MemoryFact | None:
        """The readable file for one fact: its current version, or no file at all."""
        current = [fact for _, fact in self._current(self._group(subject_id, key))]
        view = self._view_path(subject_id, key)
        if not current:
            self._dir.delete(view)
            return None
        self._dir.write(
            view, render_document(_document(current[-1]), self._files.frontmatter_delimiter)
        )
        return current[-1]

    def _view_path(self, subject_id: str, key: str) -> str:
        digest = self._files.name_digest_chars
        subject = safe_name(self._subject_dir(subject_id), digest)
        return f"{subject}/{safe_name(key, digest)}{self._files.record_suffix}"

    def _write_memory_index(self) -> None:
        by_subject: dict[str, list[MemoryFact]] = defaultdict(list)
        for fact in self._facts(include_superseded=False):
            by_subject[fact.subject_id].append(fact)
        latest = {(f.subject_id, f.key): f for facts in by_subject.values() for f in facts}
        entries = [
            IndexEntry(
                title=fact.key,
                path=self._view_path(fact.subject_id, fact.key),
                description=f"{fact.value} ({fact.provenance.kind}, v{fact.version})",
            )
            for fact in sorted(latest.values(), key=lambda f: (f.subject_id, f.key))
        ]
        write_index(
            self._dir,
            self._files.index_file,
            _STORE_TITLE,
            self._agent_id,
            entries,
            self._files.index,
        )

    def _ensure_index(self) -> None:
        """Rebuild the derived index if the records changed behind it (or it never existed)."""
        if self._index.fingerprint() == self._log.fingerprint():
            return
        with self._dir.locked():
            if self._index.fingerprint() != self._log.fingerprint():
                self._rebuild_index()

    def _rebuild_index(self) -> int:
        latest: dict[str, MemoryFact] = {}
        for record in self._log.records():
            fact = _fact(record)
            if fact.is_current:
                latest[self._log.group_path(self._group(fact.subject_id, fact.key))] = fact
        docs = [_indexed(doc_id, fact) for doc_id, fact in latest.items()]
        return self._index.rebuild(docs, self._log.fingerprint())


def _document(fact: MemoryFact) -> MemoryDocument:
    """A fact as a memory file: everything but the value in frontmatter, the value as body."""
    return MemoryDocument(meta=fact.model_dump(mode="json", exclude={"value"}), body=fact.value)


def _fact(record: Record) -> MemoryFact:
    meta = {name: value for name, value in record.document.meta.items() if name != SEQUENCE_FIELD}
    return MemoryFact.model_validate({**meta, "value": record.document.body})


def _indexed(doc_id: str, fact: MemoryFact) -> IndexedDoc:
    return IndexedDoc(doc_id=doc_id, subject=fact.subject_id, key=fact.key, text=fact.value)
