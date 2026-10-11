"""A memory file: YAML frontmatter over a markdown body.

The same shape a SKILL.md has, and the shape Claude Code's own memories
take, so one reader understands every file in a memory directory. The
frontmatter is data (who said it, when, which version); the body is the
memory itself, as a person would write it down.
"""

from typing import Any

import yaml
from pydantic import BaseModel, Field


class MemoryDocument(BaseModel):
    """One memory file, parsed."""

    meta: dict[str, Any] = Field(default_factory=dict)
    body: str = ""


class MemoryDocumentError(ValueError):
    """A memory file whose frontmatter cannot be read."""


def render_document(document: MemoryDocument, delimiter: str) -> str:
    """The file's text: frontmatter between delimiters, then the body."""
    meta = yaml.safe_dump(document.meta, sort_keys=False, allow_unicode=True).strip()
    return f"{delimiter}\n{meta}\n{delimiter}\n{document.body}\n"


def parse_document(text: str, delimiter: str) -> MemoryDocument:
    """A memory file read back; a file with no frontmatter is all body."""
    if not text.startswith(delimiter + "\n"):
        return MemoryDocument(body=text.rstrip("\n"))
    closing = text.find(f"\n{delimiter}\n", len(delimiter))
    if closing < 0:
        raise MemoryDocumentError("frontmatter is never closed")
    try:
        meta = yaml.safe_load(text[len(delimiter) + 1 : closing]) or {}
    except yaml.YAMLError as error:
        raise MemoryDocumentError(f"frontmatter is not YAML: {error}") from error
    if not isinstance(meta, dict):
        raise MemoryDocumentError("frontmatter is not a mapping")
    body = text[closing + len(delimiter) + 2 :]
    return MemoryDocument(meta=meta, body=body.removesuffix("\n"))
