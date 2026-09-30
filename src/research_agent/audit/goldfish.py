"""The Goldfish Audit: a fresh-context check of the delivered report.

The *Elephant* remembers everything the user said; the *Goldfish*
remembers nothing — it sees only the frozen contract, the report and the
evidence, and asks whether the one satisfies the other two. Its isolation
is structural and always on: `audit_report(contract, report,
ledger_snapshot)` takes nothing else, and it runs as its own graph on its
own thread through the platform's `IsolatedGraphRunner`, which refuses any
input it was not declared to take.

Deterministic and NLI checks always run. Only `judge`-type items depend
on a model, and whether that model must differ from the synthesizer's is
configuration (`research_agent.audit.independent_judge`), validated at
start-up.

Beyond per-item verdicts it answers the two questions a PARTIAL report
raises: are the gaps it flags real, and is anything it presents as
covered actually unmet?
"""

from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agentic_commons.durability.isolated import IsolatedGraphRunner
from agentic_commons.evidence import LedgerSnapshot
from agentic_commons.verification.acceptance import (
    AcceptanceCheckRegistry,
    CheckKind,
    CheckOutcome,
    CheckSpec,
    CheckVerdict,
)
from research_agent.audit.context import CheckContext
from research_agent.domain.report import Report
from research_agent.domain.rubric import Contract
from research_agent.prompts import judge as judge_prompt
from research_agent.structured import StructuredCaller

INPUTS = frozenset({"contract", "report", "ledger"})
ContextFactory = Callable[[Contract, LedgerSnapshot, Report | None], CheckContext]


class AuditResult(BaseModel):
    """Per-item verdicts, the gaps, and whether the report's own account is honest."""

    verdicts: list[CheckVerdict] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    must_pass_gaps: list[str] = Field(default_factory=list)
    # Items the report flags as unmet that in fact pass.
    false_limits: list[str] = Field(default_factory=list)
    # Must-pass items that fail but the report does not flag.
    unflagged_gaps: list[str] = Field(default_factory=list)
    judge_model: str = ""
    independent: bool = True

    @property
    def honest(self) -> bool:
        return not self.false_limits and not self.unflagged_gaps


class AuditState(TypedDict, total=False):
    contract: Contract
    report: Report
    ledger: LedgerSnapshot
    verdicts: list[CheckVerdict]


class LlmJudge:
    """Judge-type items, scored by the `judge` role's model."""

    def __init__(self, caller: StructuredCaller) -> None:
        self._caller = caller

    async def judge(self, spec: CheckSpec, report: Report, ledger: LedgerSnapshot) -> CheckOutcome:
        criterion = str(spec.params.get("text", spec.name))
        answer = await self._caller.call(
            judge_prompt.build(criterion, spec.name, report, ledger), judge_prompt.JudgeAnswer
        )
        return CheckOutcome(score=answer.score, detail=answer.rationale, evidence=answer.evidence)


class GoldfishAuditor:
    """Runs the audit graph in isolation on exactly three inputs."""

    def __init__(
        self,
        registry: AcceptanceCheckRegistry[CheckContext],
        context: ContextFactory,
        judge_model: str,
        independent: bool,
    ) -> None:
        """`context` builds each check's view: contract, ledger, report — nothing else."""
        self._registry = registry
        self._context = context
        self._judge_model = judge_model
        self._independent = independent
        self._runner = IsolatedGraphRunner(self._graph, INPUTS)

    async def audit_report(
        self, contract: Contract, report: Report, ledger_snapshot: LedgerSnapshot
    ) -> AuditResult:
        """Audit a report against its contract and evidence — nothing else is reachable."""
        final = await self._runner.run(
            {"contract": contract, "report": report, "ledger": ledger_snapshot}
        )
        verdicts: list[CheckVerdict] = final["verdicts"]
        return _result(verdicts, report, self._judge_model, self._independent)

    def _graph(self) -> Any:
        builder = StateGraph(AuditState)
        builder.add_node("deterministic", self._stage(["deterministic"]))
        builder.add_node("nli", self._stage(["nli"]))
        builder.add_node("judge", self._stage(["judge"]))
        builder.add_edge(START, "deterministic")
        builder.add_edge("deterministic", "nli")
        builder.add_edge("nli", "judge")
        builder.add_edge("judge", END)
        return builder.compile()

    def _stage(self, kinds: list[CheckKind]) -> Any:
        async def run(state: AuditState) -> dict[str, list[CheckVerdict]]:
            ctx = self._context(state["contract"], state["ledger"], state["report"])
            found = await self._registry.evaluate(
                state["contract"].rubric.specs(), ctx, kinds=kinds
            )
            return {"verdicts": [*state.get("verdicts", []), *found]}

        return run


def _result(
    verdicts: list[CheckVerdict], report: Report, judge_model: str, independent: bool
) -> AuditResult:
    flagged = {entry.item_id for entry in report.limits}
    failing = {v.item_id for v in verdicts if not v.passed}
    must_fail = {v.item_id for v in verdicts if v.must_pass and not v.passed}
    return AuditResult(
        verdicts=verdicts,
        gaps=sorted(failing),
        must_pass_gaps=sorted(must_fail),
        false_limits=sorted(flagged - failing),
        unflagged_gaps=sorted(must_fail - flagged),
        judge_model=judge_model,
        independent=independent,
    )
