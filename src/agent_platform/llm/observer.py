"""What a model chain tells its owner: a backup started, a backup answered (Observer).

The chain (`CascadingLLMClient`) knows when it reaches for another model; only the agent knows
how to show that — a card, a counter, a log line. The chain calls this Protocol and stays
ignorant of reporters; `observability.model_observer.StageReporterModelObserver` is the
platform's card-drawing implementation.
"""

from typing import Protocol


class ModelCallObserver(Protocol):
    """Told about the backup events of one labelled model chain."""

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None: ...


class NullModelObserver:
    """Observes nothing: the default when an owner does not care."""

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None:
        return None


__all__ = ["ModelCallObserver", "NullModelObserver"]
