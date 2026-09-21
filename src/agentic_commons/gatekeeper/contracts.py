"""Agent-owned deterministic business-rule boundary for the Gatekeeper."""

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agentic_commons.gatekeeper.types import AuthorityContext, CheckReport, Verdict
from agentic_commons.tools.types import ToolCall


class BusinessRuleResult(BaseModel):
    """Result of authoritative, agent-specific business-rule evaluation."""

    verdict: Verdict = "allow"
    reason_code: str = "BUSINESS_RULES_PASSED"
    reason: str = "all applicable agent business rules passed"
    checks: list[CheckReport] = Field(default_factory=list)


@runtime_checkable
class DeterministicBusinessRules(Protocol):
    """Agent/domain-owned checks performed immediately before execution."""

    def evaluate(
        self, call: ToolCall, authority: AuthorityContext
    ) -> BusinessRuleResult: ...


class AllowAllBusinessRules:
    """Default for generic/episode Gatekeeper demos with no domain rules."""

    def evaluate(self, call: ToolCall, authority: AuthorityContext) -> BusinessRuleResult:
        return BusinessRuleResult(
            checks=[
                CheckReport(
                    check_id="gatekeeper.business_rules",
                    name="Agent business rules",
                    status="PASS",
                    reason_code="NO_AGENT_RULES",
                    reason="No agent-specific deterministic business rules were injected.",
                )
            ]
        )
