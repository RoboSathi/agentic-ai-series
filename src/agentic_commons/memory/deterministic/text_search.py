"""Grep over file-backed memory: ripgrep when present, Python when not.

Episodic logs and skills are files, and the fastest correct search over
files is ripgrep. Two rules keep it safe:

  * **Fixed strings only** (`-F`). A customer's words are never compiled
    as a regular expression, so there is no ReDoS and no pattern injection.
  * **Bounded**: a timeout from config; a search that overruns returns
    what it has and says so in the log rather than stalling a turn.

The Python backend gives identical semantics (case-insensitive substring
match, any term) for machines without `rg`, so behaviour does not change
with the environment — only speed does.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.config.settings import TextSearchSettings
from agentic_commons.logging import get_logger


class TextMatch(BaseModel):
    """One matching line."""

    path: str
    line_number: int
    text: str


@runtime_checkable
class TextSearchBackend(Protocol):
    """Finds lines containing any of the terms, case-insensitively."""

    name: str

    def search(self, root: Path, terms: list[str], glob: str) -> list[TextMatch]: ...


class RipgrepBackend:
    """`rg --json -F -i` over a directory."""

    name = "ripgrep"

    def __init__(self, binary: str, timeout_seconds: float) -> None:
        self._binary = binary
        self._timeout = timeout_seconds
        self._logger = get_logger("memory")

    def search(self, root: Path, terms: list[str], glob: str) -> list[TextMatch]:
        if not terms or not root.is_dir():
            return []
        command = [self._binary, "--json", "-F", "-i", "--glob", glob]
        for term in terms:
            command.extend(["-e", term])
        command.append(str(root))
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=self._timeout, check=False
            )
        except subprocess.TimeoutExpired:
            self._logger.warning("ripgrep timed out", root=str(root), terms=terms)
            return []
        return [match for line in completed.stdout.splitlines() if (match := _parse(line))]


class PythonScanBackend:
    """The same search in pure Python, for machines without ripgrep."""

    name = "python"

    def search(self, root: Path, terms: list[str], glob: str) -> list[TextMatch]:
        if not terms or not root.is_dir():
            return []
        needles = [term.casefold() for term in terms]
        matches: list[TextMatch] = []
        for path in sorted(root.glob(glob)):
            lines = path.read_text(encoding="utf-8").splitlines()
            for number, text in enumerate(lines, start=1):
                folded = text.casefold()
                if any(needle in folded for needle in needles):
                    matches.append(TextMatch(path=str(path), line_number=number, text=text))
        return matches


def create_text_search(settings: TextSearchSettings | None = None) -> TextSearchBackend:
    """ripgrep if configured and installed, else the Python scan (Factory)."""
    resolved = settings or get_settings().memory.deterministic.text_search
    if resolved.backend == "ripgrep" and shutil.which(resolved.binary):
        return RipgrepBackend(resolved.binary, resolved.timeout_seconds)
    return PythonScanBackend()


def _parse(line: str) -> TextMatch | None:
    """One `rg --json` event, if it is a match."""
    event = json.loads(line)
    if event.get("type") != "match":
        return None
    data = event["data"]
    return TextMatch(
        path=str(data["path"].get("text", "")),
        line_number=int(data.get("line_number") or 0),
        text=str(data["lines"].get("text", "")).rstrip("\n"),
    )
