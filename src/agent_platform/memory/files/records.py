"""Append-only versioned records — the truth under a file-backed store.

Every write is a new numbered file under the store's history directory,
grouped by what it is about (`<subject>/<key>/` for a fact, `<name>/` for a
skill). Nothing is overwritten to *change* a belief: a newer record does
that, and the older one stays as history. The few edits a record does get
— marking it superseded, tombstoned, or recalled — rewrite that one file
atomically, under the store's lock.

The number is a store-wide sequence, so ordering by it is ordering by
when things were written, across every group — what a database's
autoincrement id gave the store before.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agent_platform.config.settings import MemoryFilesSettings
from agent_platform.logging import get_logger
from agent_platform.memory.files.directory import MemoryDirectory, safe_name
from agent_platform.memory.files.frontmatter import (
    MemoryDocument,
    MemoryDocumentError,
    parse_document,
    render_document,
)

# Where a record keeps its own number, in its frontmatter.
SEQUENCE_FIELD = "seq"


class Record(BaseModel):
    """One record file, read back."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seq: int
    path: Path
    document: MemoryDocument


class RecordLog:
    """Numbered, grouped, append-only records under one memory directory."""

    def __init__(self, directory: MemoryDirectory, files: MemoryFilesSettings) -> None:
        self._dir = directory
        self._files = files
        self._logger = get_logger("memory")

    @property
    def directory(self) -> MemoryDirectory:
        return self._dir

    def group_path(self, group: Sequence[str]) -> str:
        """Where one group's records live, relative to the store root."""
        names = [safe_name(part, self._files.name_digest_chars) for part in group]
        return "/".join([self._files.history_dir, *names])

    def append(self, group: Sequence[str], document: MemoryDocument) -> Record:
        """Write a new record; the caller holds the store's lock."""
        seq = self._dir.next_sequence(self._files.sequence_file)
        meta = {**document.meta, SEQUENCE_FIELD: seq}
        stored = MemoryDocument(meta=meta, body=document.body)
        relative = f"{self.group_path(group)}/{seq:0{self._files.record_digits}d}"
        path = self._dir.write(
            relative + self._files.record_suffix,
            render_document(stored, self._files.frontmatter_delimiter),
        )
        return Record(seq=seq, path=path, document=stored)

    def rewrite(self, record: Record, meta: dict[str, object]) -> Record:
        """Change a record's frontmatter in place (superseded, tombstoned, recalled)."""
        document = MemoryDocument(meta={**record.document.meta, **meta}, body=record.document.body)
        self._dir.write(
            self._dir.relative(record.path),
            render_document(document, self._files.frontmatter_delimiter),
        )
        return record.model_copy(update={"document": document})

    def records(self, group: Sequence[str] = ()) -> list[Record]:
        """Every readable record in a group (or the whole store), oldest first.

        A record that cannot be parsed is skipped and logged: losing one
        is better than refusing to read the store, the same rule the
        episodic log follows.
        """
        found: list[Record] = []
        pattern = f"*{self._files.record_suffix}"
        for path in self._dir.files(self.group_path(group), pattern):
            record = self._read(path)
            if record is not None:
                found.append(record)
        return sorted(found, key=lambda record: record.seq)

    def remove(self, group: Sequence[str]) -> int:
        """Delete a group's records outright (erasure only); returns how many."""
        return self._dir.remove_tree(self.group_path(group))

    def fingerprint(self) -> str:
        """A digest of every record's name, size and modification time."""
        digest = hashlib.sha256()
        for path in self._dir.files(self._files.history_dir, f"*{self._files.record_suffix}"):
            stat = path.stat()
            digest.update(f"{self._dir.relative(path)}:{stat.st_size}:{stat.st_mtime_ns};".encode())
        return digest.hexdigest()

    def _read(self, path: Path) -> Record | None:
        try:
            document = parse_document(
                path.read_text(encoding="utf-8"), self._files.frontmatter_delimiter
            )
        except (OSError, MemoryDocumentError) as error:
            self._logger.warning(
                "Skipping unreadable memory record", file=str(path), error=str(error)
            )
            return None
        seq = document.meta.get(SEQUENCE_FIELD)
        if not isinstance(seq, int):
            self._logger.warning("Skipping memory record with no sequence", file=str(path))
            return None
        return Record(seq=seq, path=path, document=document)
