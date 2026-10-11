"""MEMORY.md: one line per memory, so a person (or a grep) sees a store at a glance.

Claude Code keeps an index of its memory files in the same spirit: a
title, then a link and a one-line hook per memory. The index is never the
truth — it is rebuilt from the store after every change — so it can be
deleted or edited by hand without losing anything.
"""

from pydantic import BaseModel

from agent_platform.config.settings import MemoryIndexWords
from agent_platform.memory.files.directory import MemoryDirectory


class IndexEntry(BaseModel):
    """One line of a MEMORY.md."""

    title: str
    path: str
    description: str


def render_index(store: str, agent: str, entries: list[IndexEntry], words: MemoryIndexWords) -> str:
    """The index text: a title, then one line per entry (or a line saying there are none)."""
    lines = [words.title.format(store=store, agent=agent), ""]
    lines += [_line(entry, words) for entry in entries] or [words.empty]
    return "\n".join(lines) + "\n"


def write_index(
    directory: MemoryDirectory,
    file_name: str,
    store: str,
    agent: str,
    entries: list[IndexEntry],
    words: MemoryIndexWords,
) -> None:
    """Rewrite a store's MEMORY.md."""
    directory.write(file_name, render_index(store, agent, entries, words))


def _line(entry: IndexEntry, words: MemoryIndexWords) -> str:
    description = " ".join(entry.description.split())
    if len(description) > words.description_chars:
        description = description[: words.description_chars].rstrip() + "…"
    return words.line.format(title=entry.title, path=entry.path, description=description)
