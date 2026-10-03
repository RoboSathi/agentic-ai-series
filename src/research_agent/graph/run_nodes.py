"""The autonomous run graph's nodes.

    init_run → plan_dag → wave_start ─Send×N→ research_node → wave_done ─┐
                              ▲                                          │
                              └──── more ready nodes in this round ◄─────┤
                                                                         ▼
                 reflect_gap_analysis ── continue (follow-ups) ──► wave_start
                        │ stop
                        ▼
                    synthesize → independent_audit ── must-pass gaps ──► gap_fill → wave_start
                                        │ (or audit withheld)
                                        ▼
                                    finalize → archive → END

Every node that needs a live collaborator reaches it through the run's
`RunResources`, bound by the agent before the run starts (or re-bound,
from the checkpointed snapshots, when a crashed run resumes).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langgraph.types import Send

from agent_platform.config import FeatureSet
from agent_platform.control_loop import BudgetGuard
from agent_platform.logging import describe_error
from agent_platform.scheduling import AsyncDagScheduler, TaskGraph
from agent_platform.verification.acceptance import AcceptanceCheckRegistry, CheckVerdict
from research_agent.archive.store import ResearchArchive
from research_agent.audit.context import CheckContext
from research_agent.audit.independent import AuditResult, IndependentAuditor
from research_agent.audit.verdict import AuditVerdict, decide
from research_agent.domain.budget import GAP_FILL_ROUNDS, ROUNDS, SEARCHES
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import LimitEntry, Report, RunStatus
from research_agent.graph.state import RunState
from research_agent.planning.ids import renumber
from research_agent.planning.planner import Planner, PlanRejectedError
from research_agent.presentation.narration import clip, say
from research_agent.research import cards
from research_agent.research.reflector import GapAnalyzer, Reflector
from research_agent.research.resources import RunResources
from research_agent.research.worker import UNREADABLE, ResearcherWorker
from research_agent.synthesis.citation_guard import CitationGuard, GuardedReport
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
    # The audit's weighted-score bands, and how cards word what they show.
    pass_threshold: float
    partial_threshold: float
    clip_chars: int
    node_id_prefix: str
    # The budget counters shown on a budget card's heading (the rest are in its expander).
    budget_headline: Sequence[str]
    reasons: Mapping[str, str]
    auditor: IndependentAuditor | None = None
    reflection_enabled: bool = True
    # Per-role {tokens, seconds, calls} so far (`StructuredCaller.usage`); None: not tracked.
    usage: Callable[[], dict[str, dict[str, float]]] | None = None
    # What the judge card names: the model that wrote the report, and the NLI model.
    writer_model: str = ""
    nli_model: str = ""
    judge_role: str = ""
    # Checks every cited sentence against its spans before the audit; None: withheld.
    citation_guard: CitationGuard | None = None


# Rejection codes that mean "the filter could not read the result", not "the result is
# unwanted": when they are the only thing standing between a run and its evidence, another
# round of searching cannot help, so the run stops and says so.
FAIL_FAST_CODES = frozenset({"undated", UNREADABLE})

# Counters whose exhaustion cuts work short by itself: model calls and the clock stop mid-way,
# so whatever was being written is unfinished. A refused search or page read (recorded in
# `trace.budget_stops`) is not: the report says what was not read, and whether the contract was
# still met is the audit's to judge. A run whose every must-pass criterion holds is COMPLETE
# even if it could have read more.
TRUNCATING_COUNTERS = frozenset({"tokens", "seconds"})


def truncated_by_budget(exhausted: list[str]) -> bool:
    """Whether a budget limit cut the run's work short, as opposed to merely bounding it."""
    return bool(TRUNCATING_COUNTERS & set(exhausted))


def fit_to_budget(follow_ups: list[SubQuestion], searches_left: float) -> list[SubQuestion]:
    """Follow-ups trimmed so their queries fit the searches that remain (one search per query)."""
    room, fitted = int(searches_left), []
    for node in follow_ups:
        if room <= 0:
            break
        queries = node.queries[:room]
        if not queries:
            continue
        room -= len(queries)
        fitted.append(
            node
            if len(queries) == len(node.queries)
            else node.model_copy(update={"queries": queries})
        )
    return fitted


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
        self._budget_card(say("budget.at_start"))
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
            r.telemetry.card(
                "plan_dag",
                say("plan.rejected", error=str(exc)),
                {say("plan.section.problems"): list(exc.validation.problems)},
                status="failed",
            )
            r.telemetry.audit("plan.rejected", "deny", str(exc))
            return {
                "status": "FAILED",
                "stop_reason": say("plan.rejected", error=str(exc)),
                "plans": [],
                "round_nodes": [],
            }
        except Exception as exc:
            # A model timeout, a spent quota, a cut-off answer: end the run with the reason,
            # rather than failing the node with a raw error and leaving the shell waiting.
            error = describe_error(exc)
            message = say("plan.unavailable", error=error)
            r.telemetry.stage("plan_dag", message, {"error": error}, status="failed")
            r.telemetry.audit("plan.unavailable", "deny", error)
            return {"status": "FAILED", "stop_reason": message, "plans": [], "round_nodes": []}
        self._plan_card(plan, repaired)
        return {"plans": [plan], "round_nodes": list(plan.nodes)}

    def _plan_card(self, plan: PlanDag, repaired: bool) -> None:
        """The plan and its validation, as one card: what will be asked, and that it can run."""
        waves = len(TaskGraph(plan.tasks()).waves())
        validity = say("plan.valid_repaired") if repaired else say("plan.valid")
        heading = say("plan.heading", nodes=len(plan.nodes), waves=waves, validity=validity)
        questions = [
            say("plan.line", id=n.node_id, question=n.question, items=", ".join(n.rubric_items))
            for n in plan.nodes
        ]
        checks = [say("plan.check.acyclic"), say("plan.check.traced")]
        if repaired:
            checks.append(say("plan.check.repaired"))
        sections = {
            say("plan.section.questions"): questions,
            say("plan.section.validation"): checks,
        }
        self.resources.telemetry.card("plan_dag", heading, sections)

    @staticmethod
    def after_plan(state: RunState) -> str:
        return "finalize" if state.get("status") == "FAILED" else "wave_start"

    # -- waves -------------------------------------------------------------------------

    async def wave_start(self, state: RunState) -> Update:
        ready, blocked = self._ready(state)
        wave = state.get("wave", 0) + (1 if ready else 0)
        if ready:
            contract = state["contract"]
            heading, sections = cards.batch_card(
                state["round"],
                wave,
                ready,
                blocked,
                contract.budget.concurrency,
                contract.rubric,
                self._p.clip_chars,
            )
            self.resources.telemetry.card("wave_start", heading, sections)
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
                self.resources.telemetry.card(
                    "page_result",
                    say("node.failed", node=node.node_id),
                    {say("node.section.error"): [str(exc) or type(exc).__name__]},
                    status="failed",
                )
                return {"done": {node.node_id: "failed"}, "results": []}
        return {"done": {node.node_id: "done"}, "results": [result]}

    async def wave_done(self, state: RunState) -> Update:
        """The wave boundary: snapshot what a resumed run needs, show the meters."""
        r = self.resources
        self._budget_card(say("budget.after_batch", batch=state.get("wave", 0)))
        self._wave_summary(state.get("wave", 0), state.get("plans", []))
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
        failing = [v for v in verdicts if not v.passed]
        gaps = [v.item_id for v in failing]
        entities = GapAnalyzer.entities(ctx, ctx.entity_pattern)
        stop, reason = self._should_stop(state, verdicts, entities)
        update: Update = (
            {"gaps": gaps, "stop_reason": reason, "round_nodes": []}
            if stop
            else await self._next_round(state, ctx, verdicts, gaps)
        )
        follow_ups = update["round_nodes"]
        heading, sections = cards.reflect_card(
            state["round"],
            failing,
            bool(follow_ups),
            update.get("stop_reason") or reason,
            follow_ups,
            contract.rubric,
            self._p.clip_chars,
        )
        r.telemetry.card("reflect_gap_analysis", heading, sections)
        return {**update, "entities": entities}

    def _should_stop(
        self, state: RunState, verdicts: list[CheckVerdict], entities: int
    ) -> tuple[bool, str]:
        r = self.resources
        if r.cancelled:
            return True, say("stop.cancelled")
        blocked = self._unreadable_only()
        if blocked:
            return True, blocked
        archetype = state["contract"].brief.archetype
        decision = self._p.reflector.decide(archetype, verdicts, state.get("entities", 0), entities)
        if decision.stop:
            # Answered is answered, even if a cap was reached getting there.
            return True, decision.reason
        if not self._p.reflection_enabled:
            return True, say("stop.reflection_withheld")
        # Rounds used up is the plan running its course, not a limit hit mid-way.
        exhausted = [c for c in r.budget.exhausted_counters() if c != ROUNDS]
        if exhausted or r.trace.budget_stops:
            spent = ", ".join(exhausted or r.trace.budget_stops)
            return True, say("stop.budget", counters=spent)
        if r.budget.remaining(ROUNDS) <= 0:
            return True, say("stop.max_rounds")
        return False, decision.reason

    async def _next_round(
        self, state: RunState, ctx: CheckContext, verdicts: list[CheckVerdict], gaps: list[str]
    ) -> Update:
        r = self.resources
        tried = [q for queries in r.trace.queries_by_node.values() for q in queries]
        follow_ups = self._numbered(
            fit_to_budget(
                await self._p.reflector.follow_ups(ctx, verdicts, tried, state["round"] + 1),
                r.budget.remaining(SEARCHES),
            ),
            state,
        )
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

    def _numbered(self, follow_ups: list[SubQuestion], state: RunState) -> list[SubQuestion]:
        """Follow-ups carry on the plan's numbering (Q5, Q6 after Q1..Q4), not a round's own."""
        asked = sum(len(plan.nodes) for plan in state.get("plans", []))
        return list(renumber(follow_ups, self._p.node_id_prefix, start=asked + 1))

    @staticmethod
    def after_reflect(state: RunState) -> str:
        return "wave_start" if state.get("round_nodes") else "synthesize"

    # -- synthesize, audit, gap-fill ----------------------------------------------------------

    async def synthesize(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        ledger = r.ledger.snapshot()
        checked = await self._p.synthesizer.draft(contract, ledger, state.get("gaps", []))
        if self._p.citation_guard is not None and ledger.entries:
            guarded = await self._p.citation_guard.guard(checked.report, ledger)
            self._guard_card(guarded)
            checked = checked.model_copy(update={"report": guarded.report})
        verdicts = await self._self_check(contract, ledger, checked.report)
        status = self._provisional_status(verdicts)
        limits = build_limits(
            contract, verdicts, state.get("plans", []), r.trace, state.get("stop_reason", "")
        )
        report = self._p.synthesizer.with_limits(checked.report, ledger, limits, r.trace, status)
        plan_nodes = [node for plan in state.get("plans", []) for node in plan.nodes]
        answered = len({e.subject for e in ledger.entries} & {n.node_id for n in plan_nodes})
        heading, sections = cards.synthesize_card(
            report,
            ledger.entries,
            answered,
            len(plan_nodes),
            len(checked.removed),
            [limit.item_id for limit in limits],
            contract.rubric,
            status,
            self._p.clip_chars,
        )
        r.telemetry.card(
            "synthesize", heading, sections, status="warning" if checked.removed else "completed"
        )
        return {"report": report, "ledger": ledger}

    def _guard_card(self, guarded: GuardedReport) -> None:
        """What the citation guard did to the draft, before the audit reads it."""
        clean = not guarded.dropped
        heading = say(
            "guard.heading" if clean else "guard.heading_dropped",
            supported=guarded.supported,
            checked=guarded.checked,
            rewritten=len(guarded.rewritten),
            dropped=len(guarded.dropped),
        )
        limit = self._p.clip_chars
        sections = {
            say("guard.section.rewritten"): [clip(s, limit) for s in guarded.rewritten],
            say("guard.section.dropped"): [clip(s, limit) for s in guarded.dropped],
            say("guard.section.how"): [say("guard.how", model=self._p.nli_model)],
        }
        r = self.resources
        r.telemetry.card(
            "citation_guard", heading, sections, status="completed" if clean else "warning"
        )
        r.telemetry.audit(
            "report.citation_guard",
            "allow" if clean else "escalate",
            heading,
            rewritten=guarded.rewritten,
            dropped=guarded.dropped,
        )

    async def _self_check(self, contract: Any, ledger: Any, report: Report) -> list[CheckVerdict]:
        ctx = self._p.context(contract, ledger, report)
        return await self._p.registry.evaluate(
            contract.rubric.specs(), ctx, kinds=["deterministic", "nli"]
        )

    def _provisional_status(self, verdicts: list[CheckVerdict]) -> RunStatus:
        r = self.resources
        truncated = truncated_by_budget(r.budget.exhausted_counters())
        return "PARTIAL" if truncated or unmet_must_pass(verdicts) else "COMPLETE"

    async def independent_audit(self, state: RunState) -> Update:
        if self._p.auditor is None:
            return {"audit": None}
        report = state["report"]
        assert report is not None
        try:
            audit = await self._p.auditor.audit_report(state["contract"], report, state["ledger"])
        except Exception as exc:
            return self._audit_failed(describe_error(exc))
        self._report_audit(audit, state["contract"].rubric)
        return {"audit": audit}

    def _audit_failed(self, error: str) -> Update:
        """The audit could not run: deliver the report unverified, never fail the run for it."""
        r = self.resources
        note = say("audit.unavailable", error=error)
        r.telemetry.card("independent_audit", note, status="warning")
        r.telemetry.audit("audit.independent", "escalate", note)
        return {"audit": None, "audit_error": error}

    def _verdict(self, verdicts: list[CheckVerdict], rubric: Any) -> AuditVerdict:
        return decide(verdicts, rubric, self._p.pass_threshold, self._p.partial_threshold)

    def _report_audit(self, audit: AuditResult, rubric: Any) -> None:
        r = self.resources
        result = self._verdict(audit.verdicts, rubric)
        usage = self._p.usage() if self._p.usage is not None else {}
        setup = cards.JudgeSetup(
            judge_model=audit.judge_model,
            writer_model=self._p.writer_model,
            nli_model=self._p.nli_model,
            independent=audit.independent,
            honest=audit.honest,
            usage=usage.get(self._p.judge_role, {}),
        )
        heading, sections = cards.audit_card(
            audit.verdicts,
            rubric,
            result,
            self._p.pass_threshold,
            self._p.partial_threshold,
            setup,
            self._p.clip_chars,
        )
        r.telemetry.card(
            "independent_audit",
            heading,
            sections,
            status="completed" if result.outcome == "PASS" else "warning",
        )
        r.telemetry.audit(
            "audit.independent",
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
        follow_ups = self._numbered(
            fit_to_budget(
                await self._p.reflector.follow_ups(
                    ctx, gaps, tried, self._p.gap_fill_round_offset + state["gap_fill_round"]
                ),
                r.budget.remaining(SEARCHES),
            ),
            state,
        )
        r.telemetry.stage(
            "gap_fill",
            say("gap_fill.heading", round=state["gap_fill_round"] + 1, nodes=len(follow_ups)),
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
        self._final_card(state, status, report, intact)
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
        if state.get("audit_error"):
            return "PARTIAL"  # nothing verified the report: never call it COMPLETE
        truncated = truncated_by_budget(r.budget.exhausted_counters())
        verdict = self._verdict(self._final_verdicts(state), state["contract"].rubric)
        # PARTIAL means the weighted score fell short, a must-pass criterion is unmet, or the
        # model or the clock was cut off mid-way. A search or page the budget refused is in the
        # report's limits; with every must-pass criterion met and the score high enough, the
        # run is COMPLETE.
        return "PARTIAL" if truncated or verdict.outcome != "PASS" else "COMPLETE"

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
        error = state.get("audit_error")
        if error:
            unverified = LimitEntry(
                item_id="independent_audit", text=say("audit.unavailable", error=error)
            )
            limits = [*limits, unverified]
        return self._p.synthesizer.with_limits(report, state["ledger"], limits, r.trace, status)

    def _final_card(
        self, state: RunState, status: RunStatus, report: Report | None, intact: bool
    ) -> None:
        r, contract = self.resources, state["contract"]
        verdicts = self._final_verdicts(state)
        scored = bool(verdicts)
        urls = [e.source_url for e in state["ledger"].entries] if state.get("ledger") else []
        saved = report is not None and status in ("COMPLETE", "PARTIAL")
        heading, sections = cards.final_card(
            status,
            state.get("stop_reason", ""),
            report,
            cards.distinct_hosts(urls),
            len(urls),
            self._verdict(verdicts, contract.rubric) if scored else None,
            (
                sum(v.passed for v in verdicts if v.applicable),
                sum(v.applicable for v in verdicts),
            )
            if scored
            else None,
            intact,
            len(r.trace.rejected_sources),
            len(r.trace.blocked_outbound),
            {name: (c.spent, c.cap) for name, c in r.budget.snapshot().items()},
            self._usage_fields()["model_usage"],
            contract.contract_id if saved else "",
            r.trace.errors,
        )
        r.telemetry.card(
            "finalize", heading, sections, status="completed" if status == "COMPLETE" else "warning"
        )

    async def archive(self, state: RunState) -> Update:
        r = self.resources
        report = state.get("report")
        index: dict[str, Any] = {}
        if report is not None and state.get("status") in ("COMPLETE", "PARTIAL"):
            index = self._p.archive.archive(
                state["contract"], report, _spent(r.budget)
            ).model_dump()
        r.ledger.discard()
        return {"index": index}

    # -- shared -----------------------------------------------------------------------------

    def _usage_fields(self) -> dict[str, Any]:
        """Tokens and model time by role, so the dominant cost is visible at a glance."""
        usage = self._p.usage() if self._p.usage is not None else {}
        total = sum(v["tokens"] for v in usage.values()) or 1
        return {
            "model_usage": {
                role: f"{int(v['tokens']):,} tokens ({v['tokens'] / total:.0%}) · "
                f"{v['seconds']:.1f}s · {int(v['calls'])} call(s)"
                for role, v in sorted(usage.items(), key=lambda item: -item[1]["tokens"])
            }
        }

    def _unreadable_only(self) -> str:
        """A stop reason when nothing was admitted and every rejection was unreadable.

        Unreadable is a date that could not be read, or a page the model could not read; more
        rounds of searching cannot help either, so the run stops and says which it was.
        """
        r = self.resources
        if r.ledger.snapshot().entries or not r.trace.rejected_codes:
            return ""
        codes = set(r.trace.rejected_codes.values())
        if codes <= FAIL_FAST_CODES:
            count = len(r.trace.rejected_codes)
            return say("stop.undated" if codes == {"undated"} else "stop.unreadable", count=count)
        return ""

    def _wave_summary(self, wave: int, plans: list[PlanDag]) -> None:
        """What became of this run's search hits so far: admitted, or rejected and why."""
        r = self.resources
        entries = r.ledger.snapshot().entries
        questions = {n.node_id: n.question for plan in plans for n in plan.nodes}
        heading, sections = cards.batch_result_card(
            wave,
            r.trace.sources_examined,
            entries,
            r.trace.rejected_codes,
            questions,
            self._p.reasons,
            self._p.clip_chars,
        )
        r.telemetry.card(
            "wave_summary", heading, sections, status="completed" if entries else "warning"
        )

    def _budget_card(self, label: str) -> None:
        """The few meters that matter on the heading; every meter in the expander."""
        r = self.resources
        meters = {name: (c.spent, c.cap) for name, c in r.budget.snapshot().items()}
        heading, sections = cards.budget_card(label, meters, self._p.budget_headline)
        r.telemetry.card("budget_tick", heading, sections)


def _spent(budget: BudgetGuard) -> dict[str, float]:
    return {name: counter.spent for name, counter in budget.snapshot().items()}
