"""The Independent Audit: a fresh-context check of the delivered report.

Independent of everything that produced the report: it never sees the
conversation or the run's reasoning — only the frozen contract, the report
and the evidence — and asks whether the one satisfies the other two. Its isolation
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

from agent_platform.durability.isolated import IsolatedGraphRunner
from agent_platform.evidence import LedgerSnapshot
from agent_platform.judge import JudgeCall
from agent_platform.judge import LlmJudge as PlatformLlmJudge
from agent_platform.logging import instrument
from agent_platform.verification.acceptance import (
    AcceptanceCheckRegistry,
    CheckKind,
    CheckOutcome,
    CheckVerdict,
)
from research_agent.audit.context import CheckContext, JudgeItem
from research_agent.domain.report import Report
from research_agent.domain.rubric import Contract
from research_agent.llm.structured import StructuredCaller
from research_agent.presentation.narration import say
from research_agent.prompts import judge as judge_prompt

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
    # Judge-type items the judge (and every backup) could not score: unverified, not gaps.
    unjudged: list[str] = Field(default_factory=list)

    @property
    def honest(self) -> bool:
        return not self.false_limits and not self.unflagged_gaps


class AuditState(TypedDict, total=False):
    contract: Contract
    report: Report
    ledger: LedgerSnapshot
    verdicts: list[CheckVerdict]


class LlmJudge(PlatformLlmJudge[JudgeItem, judge_prompt.JudgeAnswer, CheckOutcome]):
    """Judge-type items, scored by the `judge` role's model.

    The platform judge's skeleton (`agent_platform.judge.LlmJudge`): ask, decide, and abstain
    when no answer can be had — here, a criterion nobody could judge is *unverified*: it
    earns no credit and, if it must pass, holds the run below COMPLETE; it is not a gap to
    research more for, and one unreachable judge no longer fails the whole run. The asking
    goes through this agent's own `StructuredCaller`, so the call is routed to the `judge`
    role's model chain, validated against the role's schema and charged to the run's budget.
    """

    def __init__(self, caller: StructuredCaller) -> None:
        super().__init__("research_agent")
        self._caller = caller

    async def _answer(self, request: JudgeItem) -> tuple[judge_prompt.JudgeAnswer, JudgeCall]:
        # Traced: the AUDIT card shows the judge's work, so no LLM CALL card per criterion.
        criterion = str(request.spec.params.get("text", request.spec.name))
        answer, call = await self._caller.call_traced(
            judge_prompt.build(
                criterion, request.spec.name, request.report, request.ledger, request.today
            ),
            judge_prompt.JudgeAnswer,
        )
        model = (call.backup or call.model) if call is not None else ""
        return answer, JudgeCall(model=model, calls=1)

    def verdict(
        self, request: JudgeItem, answer: judge_prompt.JudgeAnswer, call: JudgeCall
    ) -> CheckOutcome:
        return CheckOutcome(score=answer.score, detail=answer.rationale, evidence=answer.evidence)

    def abstain(self, request: JudgeItem, reason: str) -> CheckOutcome:
        detail = say("judge.unavailable", error=reason)
        return CheckOutcome(score=0.0, detail=detail, applicable=False)


@instrument("research_agent")
class IndependentAuditor:
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
    unjudged = {v.item_id for v in verdicts if _unjudged(v)}
    failing = {v.item_id for v in verdicts if not v.passed}
    must_fail = {v.item_id for v in verdicts if v.must_pass and not v.passed}
    return AuditResult(
        verdicts=verdicts,
        gaps=sorted(failing),
        # Unjudged is unverified, not missing: more research would not change it.
        must_pass_gaps=sorted(must_fail - unjudged),
        unjudged=sorted(unjudged),
        false_limits=sorted(flagged - failing),
        unflagged_gaps=sorted(must_fail - flagged),
        judge_model=judge_model,
        independent=independent,
    )


def _unjudged(verdict: CheckVerdict) -> bool:
    """A judge item with no score: the judge was unavailable, not the criterion unmet."""
    return verdict.kind == "judge" and not verdict.applicable and not verdict.passed
