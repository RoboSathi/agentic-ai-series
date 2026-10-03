"""Document sources — where a corpus is read from."""

import hashlib
from pathlib import Path

from agent_platform.logging import get_logger
from agent_platform.memory.semantic.rag.types import Document


class MarkdownDirectorySource:
    """Every file under a directory that matches a glob, as a Document.

    A document's identity is its path relative to the corpus root, so the
    same corpus checked out somewhere else is the same corpus.
    """

    def __init__(self, root: Path, pattern: str) -> None:
        self._root = root
        self._pattern = pattern
        self._logger = get_logger("memory")

    def documents(self) -> list[Document]:
        """The corpus as it is on disk now, in a stable order."""
        if not self._root.is_dir():
            self._logger.warning("Knowledge corpus directory missing", corpus=str(self._root))
            return []
        files = sorted(path for path in self._root.glob(self._pattern) if path.is_file())
        return [self._read(path) for path in files]

    def _read(self, path: Path) -> Document:
        content = path.read_text(encoding="utf-8")
        return Document(
            path=path.relative_to(self._root).as_posix(),
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
