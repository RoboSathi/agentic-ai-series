"""A model chain's backup events as cards on the live feed (`ModelCallObserver`).

The chain says "the primary was slow, a backup started" and "the backup answered"; this turns
each into a warning card on the `llm_call` stage. The heading names the role and what
happened, never a model (configuration may swap it); the body names both models. The role
is upper-cased (`PLANNER`, `JUDGE`) so it stands out of the heading, as it does on every card.

It also draws each request's CONTEXT card. A role that makes many like requests at once (one
per page read) is `aggregate`d: each request's full manifest goes to the log as it starts, and
`flush()` draws them as one card — so a batch of ten page reads is one CONTEXT card, not ten.
"""

from collections.abc import Callable, Iterable

from agent_platform.config import get_settings
from agent_platform.llm.types import CompletionRequest
from agent_platform.logging import get_logger
from agent_platform.observability.context_card import (
    CONTEXT_STAGE,
    aggregate_card,
    announced_manifest,
    context_fields,
    context_heading,
    note_prefix,
)
from agent_platform.observability.status_feed import StageReporter

MODEL_STAGE = "llm_call"
ReporterSource = Callable[[], StageReporter | None]


class StageReporterModelObserver:
    """Draws a model chain's backup events on whichever feed is current."""

    def __init__(self, reporter: ReporterSource, aggregate: Iterable[str] = ()) -> None:
        self._reporter = reporter
        self._aggregate = frozenset(aggregate)
        # Per aggregated label, each request's (tokens, model) since the last flush.
        self._pending: dict[str, list[tuple[int, str]]] = {}
        self._logger = get_logger("llm")

    def request_starting(self, label: str, model: str, request: CompletionRequest) -> None:
        """The CONTEXT card of a request no step card has drawn yet (a role call, a repair)."""
        reporter = self._reporter()
        if reporter is None:
            return
        manifest = announced_manifest(request, label)
        if manifest is None:
            return
        note_prefix(manifest, model)
        heading = context_heading(manifest, model, manifest.purpose)
        fields = context_fields(manifest, model)
        if label in self._aggregate:
            self._logger.info("Context", purpose=manifest.purpose, heading=heading, **fields)
            self._pending.setdefault(manifest.purpose, []).append((manifest.total_tokens, model))
            return
        reporter.emit(
            CONTEXT_STAGE, "completed", event_kind="narrative", summary=heading, fields=fields
        )

    def flush(self) -> None:
        """One CONTEXT card per aggregated purpose for the requests since the last flush."""
        reporter = self._reporter()
        pending, self._pending = self._pending, {}
        if reporter is None:
            return
        for purpose, requests in pending.items():
            heading, fields = aggregate_card(purpose, requests)
            reporter.emit(
                CONTEXT_STAGE, "completed", event_kind="narrative", summary=heading, fields=fields
            )

    def backup_started(self, label: str, primary: str, backup: str, reason: str) -> None:
        words = get_settings().observability.model_events
        summary = words.backup_started.format(
            label=label.upper(),
            primary=primary,
            backup=backup,
            reason=words.reasons.get(reason, reason),
        )
        self._card(summary, primary, backup)

    def _card(self, summary: str, primary: str, backup: str) -> None:
        reporter = self._reporter()
        if reporter is None:
            return
        words = get_settings().observability.model_events
        reporter.warning(
            MODEL_STAGE,
            event_kind="model",
            summary=summary,
            fields={words.primary_label: primary, words.backup_label: backup},
        )


__all__ = ["StageReporterModelObserver"]
