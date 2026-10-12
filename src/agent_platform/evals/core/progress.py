"""How far a run has got: each section reports its total and what is done.

The stages report through this seam and never draw anything themselves; the CLI
supplies a live progress display (`reporting.progress_bar`), a test supplies nothing.
"""

from typing import Protocol


class EvalProgress(Protocol):
    """Told when a section starts, as it advances, and when it is done."""

    def start(self, section: str, total: int | None) -> None:
        """A section begins (`total` None: its size is not known — show activity, not a %)."""

    def advance(self, section: str, completed: int) -> None:
        """How many of the section's items are done."""

    def finish(self, section: str) -> None:
        """The section is done."""


class NoProgress:
    """Reports nowhere."""

    def start(self, section: str, total: int | None) -> None:
        return None

    def advance(self, section: str, completed: int) -> None:
        return None

    def finish(self, section: str) -> None:
        return None
