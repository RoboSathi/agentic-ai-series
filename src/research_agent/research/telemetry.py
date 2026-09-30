"""Everything a run tells the operator and the audit trail, through one object.

Stage events go to the live feed (`StageReporter`, which also writes the
observability log); transactions — a search, a fetch, a gate decision, a
security rejection, the frozen contract, the final status — go to the
platform `AuditLedger`. Components call this rather than holding both, so
"every transaction is audited" is enforced by there being one way to say
that something happened.
"""

from collections.abc import Mapping
from typing import Any

from agentic_commons.audit import AuditEntry, AuditLedger
from agentic_commons.audit.types import Decision
from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.observability import StageReporter
from agentic_commons.observability.types import StatusKind

ACTOR = "research_agent"


class RunTelemetry:
    """Feed cards and audit entries for one session's runs."""

    def __init__(self, reporter: StageReporter | None, audit: AuditLedger, trace_id: str) -> None:
        self._reporter = reporter
        self._audit = audit
        self._trace_id = trace_id
        self._versions = get_settings().audit
        self._logger = get_logger("research_agent")

    @property
    def trace_id(self) -> str:
        return self._trace_id

    def retrace(self, trace_id: str) -> None:
        """Attribute what follows to another trace (a new run)."""
        self._trace_id = trace_id

    def stage(
        self,
        stage: str,
        summary: str,
        fields: Mapping[str, Any] | None = None,
        status: StatusKind = "completed",
        tokens: int | None = None,
    ) -> None:
        """One card in the live feed (and one line in the observability log)."""
        self._logger.debug("Stage", stage=stage, summary=summary, status=status)
        if self._reporter is not None:
            self._reporter.emit(stage, status, tokens_delta=tokens, summary=summary, fields=fields)

    def started(self, stage: str, summary: str) -> None:
        if self._reporter is not None:
            self._reporter.started(stage, summary=summary)

    def audit(self, action: str, decision: Decision, rationale: str, **context: Any) -> None:
        """One audit entry, attributed to this run's trace and versions."""
        self._audit.record(
            AuditEntry(
                trace_id=self._trace_id,
                actor=ACTOR,
                action=action,
                authority_context=context,
                policy_version=self._versions.policy_version,
                data_version=self._versions.data_version,
                decision=decision,
                rationale=rationale,
            )
        )
