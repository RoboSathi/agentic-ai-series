"""Long-term memory as files (ADR 0012).

One directory per agent, in the style of Anthropic's memory tool and
Claude Code's own memory: plain markdown with YAML frontmatter, a
MEMORY.md index per store, and paths that can never leave the directory.
Under each store, numbered append-only records are the truth; the files
a person reads are views rebuilt from them, and the full-text index is
derived from them too.
"""

from agent_platform.memory.files.derived import DerivedIndex, IndexedDoc
from agent_platform.memory.files.directory import MemoryDirectory, MemoryPathError, safe_name
from agent_platform.memory.files.frontmatter import (
    MemoryDocument,
    MemoryDocumentError,
    parse_document,
    render_document,
)
from agent_platform.memory.files.index import IndexEntry, render_index, write_index
from agent_platform.memory.files.records import Record, RecordLog

__all__ = [
    "DerivedIndex",
    "IndexEntry",
    "IndexedDoc",
    "MemoryDirectory",
    "MemoryDocument",
    "MemoryDocumentError",
    "MemoryPathError",
    "Record",
    "RecordLog",
    "parse_document",
    "render_document",
    "render_index",
    "safe_name",
    "write_index",
]
