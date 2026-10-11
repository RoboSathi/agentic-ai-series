"""What a model chain tells its owner: a request is starting, a backup started (Observer).

The chain (`CascadingLLMClient`) knows when it reaches for another model; only the agent knows
how to show that — a card, a counter, a log line. The chain calls this Protocol and stays
ignorant of reporters; `observability.model_observer.StageReporterModelObserver` is the
platform's card-drawing implementation.
"""

from typing import Protocol

from agent_platform.llm.types import CompletionRequest


class ModelCallObserver(Protocol):
    """Told about the requests and backup events of one labelled model chain."""

    def request_starting(self, label: str, model: str, request: CompletionRequest) -> None: ...

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None: ...


class NullModelObserver:
    """Observes nothing: the default when an owner does not care."""

    def request_starting(self, label: str, model: str, request: CompletionRequest) -> None:
        return None

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None:
        return None


__all__ = ["ModelCallObserver", "NullModelObserver"]
