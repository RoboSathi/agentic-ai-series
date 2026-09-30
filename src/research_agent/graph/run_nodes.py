"""The autonomous run graph's nodes.

    init_run → plan_dag → wave_start ─Send×N→ research_node → wave_done ─┐
                              ▲                                          │
                              └──── more ready nodes in this round ◄─────┤
                                                                         ▼
                 reflect_gap_analysis ── continue (follow-ups) ──► wave_start
                        │ stop
                        ▼
                    synthesize → goldfish_audit ── must-pass gaps ──► gap_fill → wave_start
                                        │ (or audit withheld)
                                        ▼
                                    finalize → archive → END

Every node that needs a live collaborator reaches it through the run's
`RunResources`, bound by the agent before the run starts (or re-bound,
from the checkpointed snapshots, when a crashed run resumes).
"""

from dataclasses import dataclass
from typing import Any

from langgraph.types import Send

from agentic_commons.config import FeatureSet
from agentic_commons.control_loop import BudgetGuard
from agentic_commons.scheduling import AsyncDagScheduler, TaskGraph
from agentic_commons.verification.acceptance import AcceptanceCheckRegistry, CheckVerdict
from research_agent.archive.store import ResearchArchive
from research_agent.audit.context import CheckContext
from research_agent.audit.goldfish import AuditResult, GoldfishAuditor
from research_agent.domain.budget import GAP_FILL_ROUNDS, ROUNDS
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import Report, RunStatus
from research_agent.graph.state import RunState
from research_agent.planning.planner import Planner, PlanRejectedError
from research_agent.research.reflector import GapAnalyzer, Reflector
from research_agent.research.resources import RunResources
from research_agent.research.worker import ResearcherWorker
from research_agent.synthesis.limits import build_limits, unmet_must_pass
from research_agent.synthesis.synthesizer import Synthesizer

Update = dict[str, Any]


@dataclass
class RunParts:
    """What the run graph is composed of; `None` means the feature is withheld."""

    planner: Planner
    analyzer: GapAnalyzer
    reflector: Reflector
    synthesizer: Synthesizer
    archive: ResearchArchive
    registry: AcceptanceCheckRegistry[CheckContext]
    context: Any  # CheckContextFactory: (contract, ledger, report) -> CheckContext
    features: FeatureSet
    # Gap-fill node ids are numbered from here, apart from reflection rounds'.
    gap_fill_round_offset: int
    auditor: GoldfishAuditor | None = None
    reflection_enabled: bool = True


class RunNodes:
    """Nodes of one agent's run graph; per-run state lives in `RunResources`."""

    def __init__(self, parts: RunParts) -> None:
        self._p = parts
        self._r: RunResources | None = None
        self._scheduler: AsyncDagScheduler | None = None

    def bind(self, resources: RunResources) -> None:
        self._r = resources
        self._scheduler = AsyncDagScheduler(resources.contract.budget.concurrency)

    @property
    def resources(self) -> RunResources:
        assert self._r is not None, "no run is bound"
        return self._r

    # -- start and plan ---------------------------------------------------------------

    async def init_run(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        toggles = {path: self._p.features.is_on(path) for path in self._p.features.paths()}
        r.telemetry.audit(
            "run.started",
            "allow",
            "autonomous run started under the frozen contract",
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            toggles=toggles,
        )
        self._budget_card("budget_tick", "Budget at start")
        return {
            "toggles": toggles,
            "round": 1,
            "wave": 0,
            "gap_fill_round": 0,
            "status": None,
            "stop_reason": "",
        }

    async def plan(self, state: RunState) -> Update:
        r = self.resources
        r.budget.charge(ROUNDS, 1)
        try:
            plan, validation, repaired = await self._p.planner.plan(state["contract"])
        except PlanRejectedError as exc:
            r.telemetry.stage(
                "plan_validated",
                f"Plan rejected twice: {exc}",
                {"problems": exc.validation.problems},
                status="failed",
            )
            r.telemetry.audit("plan.rejected", "deny", str(exc))
            return {
                "status": "FAILED",
                "stop_reason": f"plan invalid: {exc}",
                "plans": [],
                "round_nodes": [],
            }
        r.telemetry.stage(
            "plan_dag",
            f"{len(plan.nodes)} sub-questions · {len(TaskGraph(plan.tasks()).waves())} wave(s)",
            {n.node_id: f"{n.question} ← {','.join(n.rubric_items)}" for n in plan.nodes},
        )
        r.telemetry.stage(
            "plan_validated",
            "Plan valid" + (" after one repair" if repaired else ""),
            {"acyclic": True, "traced": True, "repaired": repaired},
        )
        return {"plans": [plan], "round_nodes": list(plan.nodes)}

    @staticmethod
    def after_plan(state: RunState) -> str:
        return "finalize" if state.get("status") == "FAILED" else "wave_start"

    # -- waves -------------------------------------------------------------------------

    async def wave_start(self, state: RunState) -> Update:
        ready, blocked = self._ready(state)
        wave = state.get("wave", 0) + (1 if ready else 0)
        if ready:
            self.resources.telemetry.stage(
                "wave_start",
                f"Round {state['round']} · wave {wave} · {len(ready)} node(s) in parallel",
                {
                    "nodes": [n.node_id for n in ready],
                    "blocked": blocked,
                    "concurrency": state["contract"].budget.concurrency,
                },
            )
        return {"wave": wave, "done": {node_id: "blocked" for node_id in blocked}}

    def fan_out(self, state: RunState) -> list[Send] | str:
        if self.resources.cancelled:
            return "finalize"
        ready, _ = self._ready(state)
        if not ready:
            return "reflect_gap_analysis"
        return [Send("research_node", {"task": node}) for node in ready]

    def _ready(self, state: RunState) -> tuple[list[SubQuestion], list[str]]:
        nodes = state.get("round_nodes", [])
        done = state.get("done", {})
        graph = TaskGraph([node.task() for node in nodes])
        ids = graph.ready(done)  # type: ignore[arg-type]
        blocked = graph.blocked(done)  # type: ignore[arg-type]
        return [node for node in nodes if node.node_id in ids], blocked

    async def research_node(self, state: RunState) -> Update:
        """One Send: one sub-question, holding one of the scheduler's slots."""
        node = state["task"]
        assert self._scheduler is not None
        async with self._scheduler.slot():
            try:
                result = await ResearcherWorker(self.resources).run(node)
            except Exception as exc:
                self.resources.trace.errors.append(f"{node.node_id}: {exc}")
                return {"done": {node.node_id: "failed"}, "results": []}
        return {"done": {node.node_id: "done"}, "results": [result]}

    async def wave_done(self, state: RunState) -> Update:
        """The wave boundary: snapshot what a resumed run needs, show the meters."""
        r = self.resources
        self._budget_card("budget_tick", f"After wave {state.get('wave', 0)}")
        return {
            "ledger": r.ledger.snapshot(),
            "spent": _spent(r.budget),
            "trace": r.trace.model_copy(deep=True),
        }

    # -- reflect --------------------------------------------------------------------------

    async def reflect(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        ctx = self._p.context(contract, r.ledger.snapshot(), None)
        verdicts = await self._p.analyzer.verdicts(ctx)
        gaps = [v.item_id for v in verdicts if not v.passed]
        entities = GapAnalyzer.entities(ctx, ctx.entity_pattern)
        stop, reason = self._should_stop(state, verdicts, entities)
        r.telemetry.stage(
            "reflect_gap_analysis",
            f"{len(gaps)} gap(s) · {'stop' if stop else 'continue'}: {reason}",
            {"gaps": gaps, "round": state["round"], "entities": entities},
        )
        if stop:
            return {"gaps": gaps, "stop_reason": reason, "round_nodes": [], "entities": entities}
        return {**await self._next_round(state, ctx, verdicts, gaps), "entities": entities}

    def _should_stop(
        self, state: RunState, verdicts: list[CheckVerdict], entities: int
    ) -> tuple[bool, str]:
        r = self.resources
        if r.cancelled:
            return True, "cancelled"
        exhausted = r.budget.exhausted_counters()
        if exhausted or r.trace.budget_stops:
            return True, f"budget exhausted ({', '.join(exhausted or r.trace.budget_stops)})"
        archetype = state["contract"].brief.archetype
        decision = self._p.reflector.decide(archetype, verdicts, state.get("entities", 0), entities)
        if decision.stop:
            return True, decision.reason
        if not self._p.reflection_enabled:
            return True, "reflection withheld: the plan ran once"
        if r.budget.remaining(ROUNDS) <= 0:
            return True, "maximum rounds reached"
        return False, decision.reason

    async def _next_round(
        self, state: RunState, ctx: CheckContext, verdicts: list[CheckVerdict], gaps: list[str]
    ) -> Update:
        r = self.resources
        tried = [q for queries in r.trace.queries_by_node.values() for q in queries]
        follow_ups = await self._p.reflector.follow_ups(ctx, verdicts, tried, state["round"] + 1)
        if not follow_ups:
            return {
                "gaps": gaps,
                "stop_reason": "no follow-up could close the remaining gaps",
                "round_nodes": [],
            }
        r.budget.charge(ROUNDS, 1)
        plans = [
            *state.get("plans", []),
            PlanDag(nodes=tuple(follow_ups), rationale="reflection follow-ups"),
        ]
        return {
            "gaps": gaps,
            "round": state["round"] + 1,
            "round_nodes": follow_ups,
            "plans": plans,
        }

    @staticmethod
    def after_reflect(state: RunState) -> str:
        return "wave_start" if state.get("round_nodes") else "synthesize"

    # -- synthesize, audit, gap-fill ----------------------------------------------------------

    async def synthesize(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        ledger = r.ledger.snapshot()
        checked = await self._p.synthesizer.draft(contract, ledger, state.get("gaps", []))
        verdicts = await self._self_check(contract, ledger, checked.report)
        status = self._provisional_status(verdicts)
        limits = build_limits(
            contract, verdicts, state.get("plans", []), r.trace, state.get("stop_reason", "")
        )
        report = self._p.synthesizer.with_limits(checked.report, ledger, limits, r.trace, status)
        r.telemetry.stage(
            "synthesize",
            f"Report drafted · {len(report.citations)} citation(s) · {status}",
            {
                "sections": [s.heading for s in report.sections],
                "fabricated_citations_removed": len(checked.removed),
                "unmet_must_pass": [limit.item_id for limit in limits],
            },
            status="warning" if checked.removed else "completed",
        )
        return {"report": report, "ledger": ledger}

    async def _self_check(self, contract: Any, ledger: Any, report: Report) -> list[CheckVerdict]:
        ctx = self._p.context(contract, ledger, report)
        return await self._p.registry.evaluate(
            contract.rubric.specs(), ctx, kinds=["deterministic", "nli"]
        )

    def _provisional_status(self, verdicts: list[CheckVerdict]) -> RunStatus:
        exhausted = bool(
            self.resources.budget.exhausted_counters() or self.resources.trace.budget_stops
        )
        return "PARTIAL" if exhausted or unmet_must_pass(verdicts) else "COMPLETE"

    async def goldfish_audit(self, state: RunState) -> Update:
        if self._p.auditor is None:
            return {"audit": None}
        report = state["report"]
        assert report is not None
        audit = await self._p.auditor.audit_report(state["contract"], report, state["ledger"])
        self._report_audit(audit)
        return {"audit": audit}

    def _report_audit(self, audit: AuditResult) -> None:
        r = self.resources
        passed = sum(v.passed for v in audit.verdicts)
        r.telemetry.stage(
            "goldfish_audit",
            f"{passed}/{len(audit.verdicts)} criteria met · must-pass gaps: {', '.join(audit.must_pass_gaps) or 'none'}",
            {
                v.item_id: f"{'✓' if v.passed else '✗'} {v.name} {v.score:.2f}"
                for v in audit.verdicts
            }
            | {
                "honest_limits": audit.honest,
                "judge_model": audit.judge_model,
                "independent": audit.independent,
            },
            status="completed" if not audit.must_pass_gaps else "warning",
        )
        r.telemetry.audit(
            "audit.goldfish",
            "allow" if not audit.must_pass_gaps else "escalate",
            f"must-pass gaps: {audit.must_pass_gaps}; unflagged: {audit.unflagged_gaps}; false limits: {audit.false_limits}",
        )

    def after_audit(self, state: RunState) -> str:
        audit, r = state.get("audit"), self.resources
        can_fill = (
            r.budget.remaining(GAP_FILL_ROUNDS) > 0
            and not r.budget.exhausted_counters()
            and not r.cancelled
        )
        if audit is not None and audit.must_pass_gaps and can_fill and self._p.reflection_enabled:
            return "gap_fill"
        return "finalize"

    async def gap_fill(self, state: RunState) -> Update:
        r, audit = self.resources, state.get("audit")
        assert audit is not None
        r.budget.charge(GAP_FILL_ROUNDS, 1)
        ctx = self._p.context(state["contract"], r.ledger.snapshot(), None)
        gaps = [v for v in audit.verdicts if v.must_pass and not v.passed]
        tried = [q for queries in r.trace.queries_by_node.values() for q in queries]
        follow_ups = await self._p.reflector.follow_ups(
            ctx, gaps, tried, self._p.gap_fill_round_offset + state["gap_fill_round"]
        )
        r.telemetry.stage(
            "gap_fill",
            f"Gap-fill round {state['gap_fill_round'] + 1}: {len(follow_ups)} targeted node(s)",
            {"gaps": [g.item_id for g in gaps], "nodes": [f.node_id for f in follow_ups]},
        )
        update: Update = {"gap_fill_round": state["gap_fill_round"] + 1, "round_nodes": follow_ups}
        if follow_ups:
            # Only a plan that runs is a plan: an empty one would be recorded as run.
            plan = PlanDag(nodes=tuple(follow_ups), rationale="gap fill")
            update["plans"] = [*state.get("plans", []), plan]
        return update

    @staticmethod
    def after_gap_fill(state: RunState) -> str:
        return "wave_start" if state.get("round_nodes") else "finalize"

    # -- finalize and archive ------------------------------------------------------------------

    async def finalize(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        status = self._final_status(state)
        report = self._final_report(state, status)
        intact = contract.verify()
        r.telemetry.audit(
            "contract.verified",
            "allow" if intact else "deny",
            "contract hash re-checked at end of run",
            contract_hash=contract.contract_hash,
            intact=intact,
        )
        r.telemetry.audit(
            "run.finished",
            "allow" if status == "COMPLETE" else "escalate",
            f"status {status}",
            status=status,
            spent=_spent(r.budget),
            stop_reason=state.get("stop_reason", ""),
        )
        r.telemetry.stage(
            "finalize",
            f"{status} · {state.get('stop_reason', '')}",
            {
                "status": status,
                "contract_intact": intact,
                "rejected_sources": len(r.trace.rejected_sources),
                "span_rejections": r.trace.span_rejections,
                "blocked_outbound": len(r.trace.blocked_outbound),
            },
            status="completed" if status == "COMPLETE" else "warning",
        )
        return {
            "status": status,
            "report": report,
            "spent": _spent(r.budget),
            "trace": r.trace.model_copy(deep=True),
        }

    def _final_status(self, state: RunState) -> RunStatus:
        r = self.resources
        if r.cancelled:
            return "CANCELLED"
        if state.get("status") == "FAILED":
            return "FAILED"
        exhausted = bool(r.budget.exhausted_counters() or r.trace.budget_stops)
        verdicts = self._final_verdicts(state)
        # Budget exhaustion can never yield COMPLETE, whatever the verdicts say.
        return "PARTIAL" if exhausted or unmet_must_pass(verdicts) else "COMPLETE"

    def _final_verdicts(self, state: RunState) -> list[CheckVerdict]:
        audit = state.get("audit")
        if audit is not None:
            return list(audit.verdicts)
        report = state.get("report")
        return (
            []
            if report is None
            else [
                CheckVerdict(
                    item_id=limit.item_id,
                    kind="deterministic",
                    name="self_check",
                    passed=False,
                    score=0.0,
                    threshold=1.0,
                    must_pass=True,
                )
                for limit in report.limits
            ]
        )

    def _final_report(self, state: RunState, status: RunStatus) -> Report | None:
        report = state.get("report")
        if report is None:
            return None
        r, contract = self.resources, state["contract"]
        verdicts = self._final_verdicts(state)
        limits = (
            build_limits(
                contract, verdicts, state.get("plans", []), r.trace, state.get("stop_reason", "")
            )
            if state.get("audit") is not None
            else report.limits
        )
        return self._p.synthesizer.with_limits(report, state["ledger"], limits, r.trace, status)

    async def archive(self, state: RunState) -> Update:
        r = self.resources
        report = state.get("report")
        index: dict[str, Any] = {}
        if report is not None and state.get("status") in ("COMPLETE", "PARTIAL"):
            index = self._p.archive.archive(
                state["contract"], report, _spent(r.budget)
            ).model_dump()
            r.telemetry.stage(
                "archive",
                f"Archived brief, rubric, report → {index['run_id']}",
                {"paths": index["artifact_paths"]},
            )
        r.ledger.discard()
        return {"index": index}

    # -- shared -----------------------------------------------------------------------------

    def _budget_card(self, stage: str, label: str) -> None:
        r = self.resources
        meters = {name: f"{c.spent:g}/{c.cap:g}" for name, c in r.budget.snapshot().items()}
        r.telemetry.stage(
            stage, f"{label} · " + " · ".join(f"{k} {v}" for k, v in meters.items()), meters
        )


def _spent(budget: BudgetGuard) -> dict[str, float]:
    return {name: counter.spent for name, counter in budget.snapshot().items()}
