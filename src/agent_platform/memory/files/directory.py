"""One memory directory: every path inside it, and nothing outside it.

The Anthropic memory tool's first rule is the one this enforces: a memory
path is relative to the memory root, and a path that would leave it —
absolute, `..`, or a symlink pointing out — is refused, never normalised
into something else. A memory key is user-influenced text, so it is never
used as a filename as-is: `safe_name` keeps what is filename-safe and adds a
digest whenever it had to change anything, so two keys never share a file.

Writes are atomic (a temporary file in the same directory, then a rename),
so a reader never sees half a memory, and `locked()` serialises the
read-modify-write steps — numbering a new version — across processes: a
console and a `make dream` can share one directory.
"""

import fcntl
import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class MemoryPathError(ValueError):
    """A memory path that would leave its memory directory."""


class MemoryDirectory:
    """Safe, atomic file access under one root."""

    def __init__(self, root: Path, lock_file: str) -> None:
        self._root = root
        self._lock_file = lock_file
        root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, relative: str | PurePosixPath) -> Path:
        """The absolute path of a relative memory path — refused if it would escape."""
        rel = PurePosixPath(str(relative))
        if rel.is_absolute() or ".." in rel.parts or "\x00" in str(rel):
            raise MemoryPathError(f"not a path inside the memory directory: {relative}")
        target = self._root.joinpath(*rel.parts)
        resolved = target.resolve()
        if resolved != self._root.resolve() and self._root.resolve() not in resolved.parents:
            raise MemoryPathError(f"resolves outside the memory directory: {relative}")
        return target

    def relative(self, path: Path) -> str:
        """A path under the root, as the POSIX string a memory index links to."""
        return path.relative_to(self._root).as_posix()

    def read(self, relative: str | PurePosixPath) -> str | None:
        """A file's text, or None if there is no such file."""
        target = self.path(relative)
        return target.read_text(encoding="utf-8") if target.is_file() else None

    def write(self, relative: str | PurePosixPath, text: str) -> Path:
        """Replace a file's contents atomically; parent directories are made as needed."""
        target = self.path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return target

    def move(self, source: str | PurePosixPath, destination: str | PurePosixPath) -> Path:
        """Rename a file within the directory (atomic on one filesystem)."""
        target = self.path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.path(source), target)
        return target

    def delete(self, relative: str | PurePosixPath) -> bool:
        """Remove one file; False if it was not there."""
        target = self.path(relative)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def remove_tree(self, relative: str | PurePosixPath) -> int:
        """Remove a directory and everything in it; returns how many files went."""
        target = self.path(relative)
        if not target.is_dir():
            return 0
        removed = sum(1 for item in target.rglob("*") if item.is_file())
        shutil.rmtree(target)
        return removed

    def files(self, relative: str | PurePosixPath = "", pattern: str = "*") -> list[Path]:
        """Files under a directory matching a glob, recursively, in path order."""
        base = self.path(relative) if str(relative) else self._root
        if not base.is_dir():
            return []
        return sorted(item for item in base.rglob(pattern) if item.is_file())

    def children(self, relative: str | PurePosixPath = "") -> list[Path]:
        """The immediate subdirectories of a directory, in name order."""
        base = self.path(relative) if str(relative) else self._root
        return sorted(item for item in base.iterdir() if item.is_dir()) if base.is_dir() else []

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the directory's lock: one writer at a time, across processes."""
        with self.path(self._lock_file).open("a", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def next_sequence(self, sequence_file: str) -> int:
        """The next number in this directory's sequence. Call while holding the lock."""
        current = self.read(sequence_file)
        value = int(current.strip()) + 1 if current and current.strip().isdigit() else 1
        self.write(sequence_file, str(value))
        return value


def safe_name(name: str, digest_chars: int) -> str:
    """A filename for a user-influenced name, unique to it.

    A name that is already safe is kept as it is, so a reader recognises
    it; one that had to change gets a short digest of the original, so
    "a b" and "a_b" never land in the same file.
    """
    cleaned = _UNSAFE.sub("_", name).strip("._") or "_"
    if cleaned == name:
        return cleaned
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:digest_chars]
    return f"{cleaned}-{digest}"
