"""The autonomous run graph's nodes.

    init_run → plan_dag → batch_start ─Send×N→ research_node → batch_done ─┐
                              ▲                                          │
                              └──── more ready nodes in this round ◄─────┤
                                                                         ▼
                 reflect_gap_analysis ── continue (follow-ups) ──► batch_start
                        │      └─ nothing admitted: one rescue round ──► batch_start
                        │ stop
                        ▼
                    synthesize → independent_audit ── must-pass gaps ──► gap_fill → batch_start
                                        │ (or audit withheld)
                                        ▼
                                    finalize → archive → END

A run never ends with nothing to show: a page that could not be read leaves its search
snippet as labelled evidence (`research/snippets.py`); a ledger still empty at reflection gets
one rescue round, granted on top of the size's budget and searched without the time window;
and a ledger still empty at synthesis is written up from the search results themselves, as a
best effort. Every such entry carries its caveat into the report, which is then PARTIAL.

Every node that needs a live collaborator reaches it through the run's
`RunResources`, bound by the agent before the run starts (or re-bound,
from the checkpointed snapshots, when a crashed run resumes).
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langgraph.types import Send

from agent_platform.config import FeatureSet
from agent_platform.hallucination_check.confidence import ConfidenceCheck
from agent_platform.hallucination_check.readings import check_rule
from agent_platform.logging import describe_error
from agent_platform.loop_control import SECONDS, BudgetGuard
from agent_platform.memory.procedural.skills import AgentInstructions
from agent_platform.memory.procedural.skills.types import LEARNED
from agent_platform.observability.step_events import HALLUCINATION_CHECK
from agent_platform.observability.types import StatusKind
from agent_platform.scheduling import AsyncDagScheduler, TaskGraph
from agent_platform.verification.acceptance import AcceptanceCheckRegistry, CheckVerdict
from research_agent.archive.store import ResearchArchive
from research_agent.audit.context import CheckContext
from research_agent.audit.independent import AuditResult, IndependentAuditor
from research_agent.audit.verdict import AuditVerdict, decide
from research_agent.config import get_agent_settings
from research_agent.domain.budget import FETCHES, GAP_FILL_ROUNDS, ROUNDS, SEARCHES
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import LimitEntry, Report, RunStatus
from research_agent.graph.run_loop import RunLoopReport
from research_agent.graph.state import RunState
from research_agent.memory import RunEventWriter, RunOutcome
from research_agent.planning.ids import renumber
from research_agent.planning.planner import Planner, PlanRejectedError
from research_agent.presentation.narration import clip, say
from research_agent.research import cards
from research_agent.research.reflector import FollowUpProposal, GapAnalyzer, Reflector
from research_agent.research.resources import RunResources
from research_agent.research.snippets import SnippetAdmission
from research_agent.research.worker import UNREADABLE, ResearcherWorker
from research_agent.synthesis.citation_guard import CitationGuard, GuardedReport, SentenceCheck
from research_agent.synthesis.limits import build_limits, unmet_must_pass
from research_agent.synthesis.synthesizer import Synthesizer

Update = dict[str, Any]


# The run graph's nodes, as LOOP CONTROL names the next step (`research_agent.run_loop.next`).
_BATCH_START, _SYNTHESIZE, _GAP_FILL, _FINALIZE = (
    "batch_start",
    "synthesize",
    "gap_fill",
    "finalize",
)
_NEXT_BATCH, _REFLECT, _REVISE = "next_batch", "reflect", "revise"
# Why (`research_agent.run_loop.because`).
_ANOTHER_ROUND, _CANCELLED, _NO_GAPS = "another_round", "cancelled", "no_gaps"
_GAP_FILL_EXHAUSTED, _BUDGET_SPENT = "gap_fill_exhausted", "budget_spent"
_REFLECTION_OFF, _GAPS_TO_FILL = "reflection_off", "gaps_to_fill"
_GAP_FILL_PLANNED, _NO_FOLLOW_UP = "gap_fill_planned", "no_follow_up"
_RESCUE = "rescue"
_BATCH_DONE, _ROUND_DONE = "batch_done", "round_done"
_WRITING_FAULTS, _REVISIONS_SPENT = "writing_faults", "revisions_spent"
_REVISION_PLATEAU = "revision_plateau"


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
    # Level 1 of the report's hallucination check (token confidence); None: withheld.
    confidence: ConfidenceCheck | None = None
    # Draws the batch's aggregated CONTEXT cards (one per page-reading role); None: none.
    flush_context: Callable[[], None] | None = None
    # Producer-Critic on the report: which checks are writing faults, how many revisions,
    # and the least score gain that earns another.
    writing_checks: frozenset[str] = frozenset()
    max_revisions: int = 0
    min_score_gain: float = 0.0
    # The models that judged the last audit (`LlmJudge.models_used`); None: not tracked.
    judge_models: Callable[[], set[str]] | None = None
    # Learned skills (ra-0023): selected at plan for this run's kind, shown on the SKILLS
    # card and handed to the planner, reflector and reviser; None: procedural memory withheld.
    skills: AgentInstructions | None = None
    # How a run's kind is written as the signal its learned skills trigger on.
    skill_scope: str = ""
    # Writes the finished run's dream event (what its critics found); None: no memory.
    run_events: RunEventWriter | None = None


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
    def revises(self) -> bool:
        """Whether the report is revised against the audit (the `revise` node exists)."""
        return self._p.reflection_enabled and self._p.max_revisions > 0

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
        return {
            "toggles": toggles,
            "round": 1,
            "batch": 0,
            "gap_fill_round": 0,
            "status": None,
            "stop_reason": "",
        }

    async def plan(self, state: RunState) -> Update:
        r, contract = self.resources, state["contract"]
        r.budget.charge(ROUNDS, 1)
        learned = self._learned_skills(contract)
        try:
            outcome = await self._p.planner.plan(contract, learned)
        except PlanRejectedError as exc:
            heading, sections = cards.plan_card(
                contract, None, 0, exc.call, rejected=list(exc.validation.problems)
            )
            r.telemetry.card("plan_dag", heading, sections, status="failed")
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
        plan = outcome.plan
        batches = len(TaskGraph(plan.tasks()).batches())
        heading, sections = cards.plan_card(
            contract,
            plan,
            batches,
            outcome.call,
            repaired_problems=outcome.problems,
            uncovered=outcome.validation.uncovered,
            budget={**contract.budget.counters(), SECONDS: contract.budget.max_seconds},
        )
        r.telemetry.card("plan_dag", heading, sections)
        return {"plans": [plan], "round_nodes": list(plan.nodes), "learned_skills": learned}

    def _learned_skills(self, contract: Any) -> list[str]:
        """Skills learned from earlier runs of this kind, on the SKILLS card when there are any.

        Only learned skills: no run role has an authored one (ra-0021). Selected without a
        model, by the run's kind and its topic, and announced once, like any skill.
        """
        if self._p.skills is None or not self._p.skill_scope:
            return []
        signal = self._p.skill_scope.format(archetype=contract.brief.archetype)
        chosen = self._p.skills.select(frozenset({signal}), contract.brief.topic)
        learned = chosen.model_copy(
            update={"skills": [i for i in chosen.skills if i.skill.source == LEARNED]}
        )
        self._p.skills.announce(learned)
        return [item.skill.document.prompt for item in learned.skills]

    @staticmethod
    def after_plan(state: RunState) -> str:
        return _FINALIZE if state.get("status") == "FAILED" else _BATCH_START

    # -- batches -------------------------------------------------------------------------

    async def batch_start(self, state: RunState) -> Update:
        if self.resources.cancelled:
            # `fan_out` will route to finalize; edges cannot draw, so this node says so.
            self._report_loop(state, _FINALIZE, _CANCELLED)
        ready, blocked = self._ready(state)
        batch = state.get("batch", 0) + (1 if ready else 0)
        if ready:
            contract = state["contract"]
            heading, sections = cards.batch_card(
                state["round"],
                batch,
                ready,
                blocked,
                contract.budget.concurrency,
                contract.rubric,
                self._p.clip_chars,
            )
            self.resources.telemetry.card("batch_start", heading, sections)
        return {"batch": batch, "done": {node_id: "blocked" for node_id in blocked}}

    def fan_out(self, state: RunState) -> list[Send] | str:
        if self.resources.cancelled:
            return _FINALIZE
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

    async def batch_done(self, state: RunState) -> Update:
        """The batch boundary: snapshot what a resumed run needs; LOOP CONTROL says where next."""
        r = self.resources
        batch = state.get("batch", 0)
        if self._p.flush_context is not None:
            self._p.flush_context()
        self._batch_summary(batch, state.get("plans", []))
        ready, _ = self._ready(state)
        done = state.get("done", {})
        self._report_loop(
            state,
            _NEXT_BATCH if ready else _REFLECT,
            _BATCH_DONE if ready else _ROUND_DONE,
            {
                "batch": batch,
                "admitted": len(r.ledger.snapshot().entries),
                "failed_nodes": [node for node, outcome in done.items() if outcome == "failed"],
            },
        )
        return {
            "ledger": r.ledger.snapshot(),
            "spent": _spent(r.budget),
            "trace": r.trace.model_copy(deep=True),
        }

    # -- reflect --------------------------------------------------------------------------

    async def reflect(self, state: RunState) -> Update:
        rescue = self._rescue(state)
        if rescue is not None:
            return rescue
        r, contract = self.resources, state["contract"]
        ctx = self._p.context(contract, r.ledger.snapshot(), None)
        verdicts = await self._p.analyzer.verdicts(ctx)
        failing = [v for v in verdicts if not v.passed]
        gaps = [v.item_id for v in failing]
        entities = GapAnalyzer.entities(ctx, ctx.entity_pattern)
        stop, reason, judgeable, cause = self._should_stop(state, verdicts, entities)
        proposal = FollowUpProposal()
        if not stop:
            update, proposal = await self._next_round(state, ctx, verdicts, gaps)
        elif judgeable:
            update, proposal = await self._critiqued(state, ctx, verdicts, gaps, reason)
        else:
            update = {"gaps": gaps, "stop_reason": reason, "round_nodes": []}
        archetype = contract.brief.archetype
        step = cards.FollowUpStep(
            gaps=failing,
            proposal=proposal,
            planned=update["round_nodes"],
            reason=update.get("stop_reason") or reason,
            rubric=contract.rubric,
            entries=ctx.ledger.entries,
            limit=self._p.clip_chars,
            verdicts=verdicts,
            rule=self._p.reflector.rule_name(archetype),
            archetype=archetype,
            entities=(state.get("entities", 0), entities),
        )
        heading, sections = cards.reflect_card(state["round"], bool(step.planned), step)
        status: StatusKind = "warning" if failing and not step.planned else "completed"
        r.telemetry.card("reflect_gap_analysis", heading, sections, status=status)
        another = bool(update["round_nodes"])
        if not another and not cause and proposal.sufficient is False:
            # The critique wanted more, and no round was left to look for it.
            cause = next(iter(self._no_room()), None)
        self._report_loop(
            state,
            _BATCH_START if another else _SYNTHESIZE,
            _ANOTHER_ROUND if another else (update.get("stop_reason") or reason),
            {"failing_checks": len(failing), "budget_stops": list(r.trace.budget_stops)},
            cause=None if another else cause,
        )
        return {**update, "entities": entities}

    def _should_stop(
        self, state: RunState, verdicts: list[CheckVerdict], entities: int
    ) -> tuple[bool, str, bool, str | None]:
        """Stop?, why, whether the model should judge the evidence before the run stops, and
        the budget counter that stopped it (None when the run stopped on its own terms).

        It judges whenever reflection is on and the run ends on its own terms — the stop rule
        says answered, or the rounds or budget ran out — not when the operator stopped it, no
        page could be read, or reflection is withheld.
        """
        r = self.resources
        if r.cancelled:
            return True, say("stop.cancelled"), False, None
        blocked = self._unreadable_only()
        if blocked:
            return True, blocked, False, None
        archetype = state["contract"].brief.archetype
        decision = self._p.reflector.decide(archetype, verdicts, state.get("entities", 0), entities)
        if not self._p.reflection_enabled:
            if decision.stop:
                return True, decision.reason, False, None
            return True, say("stop.reflection_withheld"), False, None
        if decision.stop:
            # Answered is answered, even if a cap was reached getting there.
            return True, decision.reason, True, None
        # Rounds used up is the plan running its course, not a limit hit mid-way.
        exhausted = [c for c in r.budget.exhausted_counters() if c != ROUNDS]
        if exhausted or r.trace.budget_stops:
            spent = exhausted or list(r.trace.budget_stops)
            return True, say("stop.budget", counters=", ".join(spent)), True, spent[0]
        if r.budget.remaining(ROUNDS) <= 0:
            # The rounds ran out before the stop rule was met: that is the cap deciding.
            return True, say("stop.max_rounds"), True, ROUNDS
        return False, decision.reason, False, None

    def _rescue(self, state: RunState) -> Update | None:
        """One more round for a run that admitted nothing, instead of a report of nothing.

        The plan's sub-questions are searched again without the approved time window, under a
        grant on top of the size's budget; a source outside the window is then admitted with a
        caveat rather than rejected (`worker._recent`). Once per run, and never when the
        operator stopped it or a cap that cuts work short (tokens, the clock) is spent.
        """
        r = self.resources
        config = r.settings.rescue
        if (
            not config.enabled
            or r.trace.rescued
            or r.cancelled
            or r.ledger.snapshot().entries
            or truncated_by_budget(r.budget.exhausted_counters())
        ):
            return None
        planned = [node for plan in state.get("plans", [])[:1] for node in plan.nodes]
        if not planned:
            return None
        grant = {ROUNDS: 1.0, SEARCHES: float(config.searches), FETCHES: float(config.fetches)}
        for counter, amount in grant.items():
            r.budget.extend(counter, amount)
        nodes = self._numbered(fit_to_budget(planned, r.budget.remaining(SEARCHES)), state)
        r.trace.rescued, r.trace.rescue_grant = True, grant
        r.budget.charge(ROUNDS, 1)
        self._rescue_card(nodes, grant)
        self._report_loop(state, _BATCH_START, _RESCUE, {"admitted": 0, "rescue_nodes": len(nodes)})
        plans = [*state.get("plans", []), PlanDag(nodes=tuple(nodes), rationale=_RESCUE)]
        return {
            "gaps": [],
            "round": state["round"] + 1,
            "round_nodes": nodes,
            "plans": plans,
            "stop_reason": "",
        }

    def _rescue_card(self, nodes: list[SubQuestion], grant: Mapping[str, float]) -> None:
        r = self.resources
        sections = {
            say("rescue.section.why"): [say("rescue.why")],
            say("rescue.section.grant"): [
                say("rescue.grant", counter=counter, amount=amount)
                for counter, amount in grant.items()
            ],
            say("rescue.section.questions"): [f"{n.node_id} · {n.question}" for n in nodes],
        }
        heading = say("rescue.heading", nodes=len(nodes))
        r.telemetry.card("reflect_gap_analysis", heading, sections, status="warning")
        r.telemetry.audit(
            "scope.recency_relaxed",
            "escalate",
            say("stop.rescue"),
            recency_days=r.contract.brief.recency_days,
            grant=dict(grant),
            nodes=[node.node_id for node in nodes],
        )

    def _room_for_a_round(self) -> bool:
        """Whether a follow-up round could still run: searches and rounds left."""
        return not self._no_room()

    def _no_room(self) -> list[str]:
        """The counters that leave no room for another round (searches, rounds)."""
        r = self.resources
        return [c for c in (SEARCHES, ROUNDS) if r.budget.remaining(c) <= 0]

    def _tried(self) -> list[str]:
        return [q for queries in self.resources.trace.queries_by_node.values() for q in queries]

    async def _next_round(
        self, state: RunState, ctx: CheckContext, verdicts: list[CheckVerdict], gaps: list[str]
    ) -> tuple[Update, FollowUpProposal]:
        proposal = await self._p.reflector.propose(
            ctx, verdicts, self._tried(), state["round"] + 1, state.get("learned_skills", [])
        )
        return self._round_from(state, proposal, gaps, say("stop.no_follow_up")), proposal

    async def _critiqued(
        self,
        state: RunState,
        ctx: CheckContext,
        verdicts: list[CheckVerdict],
        gaps: list[str],
        reason: str,
    ) -> tuple[Update, FollowUpProposal]:
        """Every check passed: the model judges whether the claims answer the question.

        Asked only when a round is left to act on its verdict: with none (a small run after
        round 1) its follow-ups could only end in "no room", so the call is not made.
        """
        if not self._room_for_a_round():
            stop = say("stop.critique_skipped", reason=reason)
            return {"gaps": gaps, "stop_reason": stop, "round_nodes": []}, FollowUpProposal()
        proposal = await self._p.reflector.critique(
            ctx, verdicts, self._tried(), state["round"] + 1, state.get("learned_skills", [])
        )
        if proposal.sufficient:
            stop = say("stop.critique_sufficient", reason=reason)
            return {"gaps": gaps, "stop_reason": stop, "round_nodes": []}, proposal
        if not self._room_for_a_round():
            stop = say("stop.critique_no_room", reason=reason, critique=proposal.reasoning)
            return {"gaps": gaps, "stop_reason": stop, "round_nodes": []}, proposal
        return self._round_from(state, proposal, gaps, say("stop.critique_no_follow_up")), proposal

    def _round_from(
        self, state: RunState, proposal: FollowUpProposal, gaps: list[str], none_left: str
    ) -> Update:
        """The next round from the accepted follow-ups that fit the searches left, or a stop."""
        r = self.resources
        follow_ups = self._numbered(
            fit_to_budget(proposal.accepted, r.budget.remaining(SEARCHES)), state
        )
        if not follow_ups:
            return {"gaps": gaps, "stop_reason": none_left, "round_nodes": []}
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
        return _BATCH_START if state.get("round_nodes") else _SYNTHESIZE

    # -- synthesize, audit, gap-fill ----------------------------------------------------------

    async def synthesize(self, state: RunState) -> Update:
        """The report; after a gap-fill, the previous one revised with the new evidence and the
        audit's findings, rather than a fresh draft that never saw them."""
        r, contract = self.resources, state["contract"]
        notice = "" if r.ledger.snapshot().entries else self._best_effort()
        ledger = r.ledger.snapshot()
        previous, audit = state.get("report"), state.get("audit")
        findings = self._findings(audit, contract, writing_only=False) if previous else None
        checked = await self._p.synthesizer.draft(
            contract, ledger, state.get("gaps", []), previous, findings
        )
        return await self._deliver(state, checked, notice)

    async def revise(self, state: RunState) -> Update:
        """Producer-Critic on the report: the audit's writing findings, fixed in a revision.

        Only what a finding flags changes; the revision then goes through the same citation
        guard and self-check as the draft, and the audit judges it again.
        """
        r, contract = self.resources, state["contract"]
        previous, audit = state["report"], state["audit"]
        assert previous is not None and audit is not None
        findings = self._findings(audit, contract, writing_only=True)
        revision = state.get("revision", 0) + 1
        checked = await self._p.synthesizer.revise(
            contract, r.ledger.snapshot(), previous, findings, state.get("learned_skills", [])
        )
        before = self._p.synthesizer.written_sections(previous)
        after = self._p.synthesizer.written_sections(checked.report)
        heading, sections = cards.revise_card(
            revision, self._p.max_revisions, findings, before, after, checked.call
        )
        r.telemetry.card("revise", heading, sections)
        update = await self._deliver(state, checked, previous.notice)
        return {**update, "revision": revision}

    def _findings(
        self, audit: AuditResult | None, contract: Any, *, writing_only: bool
    ) -> list[dict[str, Any]]:
        """What the audit failed, as the writer reads it: the criterion, why, and the
        sentences behind the score. `writing_only`: just the faults a revision can fix."""
        if audit is None:
            return []
        return [
            {
                "item_id": v.item_id,
                "criterion": item.text if (item := contract.rubric.item(v.item_id)) else v.name,
                "score": round(v.score, 2),
                "reason": v.detail,
                "sentences": list(v.evidence),
            }
            for v in audit.verdicts
            if not v.passed and v.applicable and (not writing_only or self._writing(v.name))
        ]

    def _writing(self, check: str) -> bool:
        return check in self._p.writing_checks

    async def _deliver(self, state: RunState, checked: Any, notice: str) -> Update:
        """A written report, checked sentence by sentence, completed with its limits."""
        r, contract = self.resources, state["contract"]
        ledger = r.ledger.snapshot()
        if self._p.citation_guard is not None and ledger.entries:
            guarded = await self._p.citation_guard.guard(checked.report, ledger)
            self._guard_card(guarded, ledger)
            checked = checked.model_copy(update={"report": guarded.report})
        verdicts = await self._self_check(contract, ledger, checked.report)
        status = self._provisional_status(verdicts)
        limits = build_limits(
            contract, verdicts, state.get("plans", []), r.trace, state.get("stop_reason", "")
        )
        report = self._p.synthesizer.with_limits(
            checked.report, ledger, limits, r.trace, status, notice
        )
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
            call=checked.call,
        )
        r.telemetry.card(
            "synthesize", heading, sections, status="warning" if checked.removed else "completed"
        )
        writer = (checked.call.backup or checked.call.model) if checked.call is not None else ""
        return {"report": report, "ledger": ledger, "writer_model": writer}

    def _best_effort(self) -> str:
        """Nothing was admitted: the search results themselves, labelled, as the evidence.

        Returns the notice the report opens with, or "" when there were no results at all
        (the reader then sees that nothing was found, and what was searched).
        """
        r = self.resources
        rescue = r.settings.rescue
        if not rescue.enabled or r.snippets is None:
            return ""
        admission = SnippetAdmission(r.ledger, r.snippets, r.settings.snippet_evidence.min_words)
        admitted = [
            entry
            for hit in r.trace.seen_hits
            for entry in admission.admit(hit, rescue.best_effort_caveat)
        ]
        if not admitted:
            return ""
        heading = say("best_effort.heading", count=len({e.source_url for e in admitted}))
        sections = {
            say("snippet.section.caveat"): [rescue.best_effort_caveat],
            say("snippet.section.kept"): [
                f"{e.entry_id} · {clip(e.claim, self._p.clip_chars)}" for e in admitted
            ],
        }
        r.telemetry.card("synthesize", heading, sections, status="warning")
        r.telemetry.audit(
            "report.best_effort", "escalate", heading, entries=[e.entry_id for e in admitted]
        )
        return rescue.best_effort_note

    def _guard_card(self, guarded: GuardedReport, ledger: Any) -> None:
        """The report's HALLUCINATION CHECK, drawn as every agent's is: logprobs first, the
        order the claims were checked in, then each cited sentence and its steps."""
        clean = not guarded.dropped
        heading = cards.guard_heading(guarded)
        fields = {
            "outcome": say("guard.outcome" if clean else "guard.outcome_dropped", heading=heading),
            "confidence": self._confidence(guarded),
            "rule": check_rule(),
            "claim_checks": [_sentence_check(check) for check in guarded.checks],
            **({"rewrite_call": guarded.call.model_dump()} if guarded.call is not None else {}),
        }
        r = self.resources
        r.telemetry.stage(
            HALLUCINATION_CHECK, heading, fields, status="completed" if clean else "warning"
        )
        r.telemetry.audit(
            "report.citation_guard",
            "allow" if clean else "escalate",
            heading,
            rewritten=guarded.rewritten,
            dropped=guarded.dropped,
        )

    def _confidence(self, guarded: GuardedReport) -> dict[str, Any]:
        """Level 1: the cited sentences scored by the writer's token confidence ({} if off)."""
        check = self._p.confidence
        if check is None or not check.enabled:
            return {}
        return check.assess(" ".join(c.sentence for c in guarded.checks)).model_dump()

    async def _self_check(self, contract: Any, ledger: Any, report: Report) -> list[CheckVerdict]:
        ctx = self._p.context(contract, ledger, report)
        return await self._p.registry.evaluate(
            contract.rubric.specs(), ctx, kinds=["deterministic", "nli"]
        )

    def _provisional_status(self, verdicts: list[CheckVerdict]) -> RunStatus:
        r = self.resources
        truncated = truncated_by_budget(r.budget.exhausted_counters())
        weaker = self._rests_on_weaker_evidence()
        return "PARTIAL" if truncated or weaker or unmet_must_pass(verdicts) else "COMPLETE"

    def _rests_on_weaker_evidence(self) -> bool:
        """Whether any admitted entry carries a caveat (a snippet, a source outside the window).

        Such a report says what it found, and is PARTIAL: it did not rest on what was asked for.
        """
        return any(entry.caveat for entry in self.resources.ledger.snapshot().entries)

    async def independent_audit(self, state: RunState) -> Update:
        if self._p.auditor is None:
            return {"audit": None}
        report = state["report"]
        assert report is not None
        try:
            audit = await self._p.auditor.audit_report(state["contract"], report, state["ledger"])
        except Exception as exc:
            return self._audit_failed(describe_error(exc))
        audit = self._independence(audit, state.get("writer_model", ""))
        score = self._report_audit(audit, state["contract"].rubric)
        scores = [*state.get("audit_scores", []), score]
        next_step, because = self._gap_fill_decision(audit, {**state, "audit_scores": scores})
        self._report_loop(
            state,
            next_step,
            because,
            {
                "must_pass_gaps": len(audit.must_pass_gaps),
                "writing_faults": len(self._findings(audit, state["contract"], writing_only=True)),
                "revision": f"{state.get('revision', 0)} of {self._p.max_revisions}",
                "audit_score": f"{score:.2f}",
            },
            cause=self._budget_cause(because),
        )
        return {"audit": audit, "audit_scores": scores}

    def _independence(self, audit: AuditResult, writer: str) -> AuditResult:
        """The audit as independent only if no criterion was judged by the report's writer."""
        judges = self._p.judge_models() if self._p.judge_models is not None else set()
        if writer and writer in judges:
            return audit.model_copy(update={"independent": False})
        return audit

    def _audit_failed(self, error: str) -> Update:
        """The audit could not run: deliver the report unverified, never fail the run for it."""
        r = self.resources
        note = say("audit.unavailable", error=error)
        r.telemetry.card("independent_audit", note, status="warning")
        r.telemetry.audit("audit.independent", "escalate", note)
        return {"audit": None, "audit_error": error}

    def _verdict(self, verdicts: list[CheckVerdict], rubric: Any) -> AuditVerdict:
        return decide(verdicts, rubric, self._p.pass_threshold, self._p.partial_threshold)

    def _report_audit(self, audit: AuditResult, rubric: Any) -> float:
        """The AUDIT card; the audit's weighted score, which a revision must raise."""
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
            judge_role=self._p.judge_role,
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
        return result.score

    def after_audit(self, state: RunState) -> str:
        return self._gap_fill_decision(state.get("audit"), state)[0]

    def _gap_fill_decision(
        self, audit: AuditResult | None, state: Mapping[str, Any] | None = None
    ) -> tuple[str, str]:
        """Gap-fill, revise or finish, and why — one rule for the edge and its LOOP CONTROL card.

        Missing evidence goes to gap-fill (more research); a fault in how the report is written
        goes to a revision (the critic's findings back to the writer). Evidence first: a report
        revised before its evidence is in would be revised again.
        """
        r = self.resources
        if audit is None:
            return _FINALIZE, _NO_GAPS
        if r.cancelled:
            return _FINALIZE, _CANCELLED
        evidence = [g for g in audit.must_pass_gaps if not self._writing(self._check_of(audit, g))]
        blocked = self._gap_fill_blocked() if evidence else _NO_GAPS
        if evidence and not blocked:
            return _GAP_FILL, _GAPS_TO_FILL
        if self._findings(audit, self.resources.contract, writing_only=True):
            revise = self._revise_decision(state or {})
            if revise[0] == _REVISE or not evidence:
                return revise
        return _FINALIZE, blocked

    def _gap_fill_blocked(self) -> str:
        """Why a gap-fill cannot run, or "" when it can."""
        r = self.resources
        if not self._p.reflection_enabled:
            return _REFLECTION_OFF
        if r.budget.remaining(GAP_FILL_ROUNDS) <= 0:
            return _GAP_FILL_EXHAUSTED
        if r.budget.exhausted_counters():
            return _BUDGET_SPENT
        return ""

    def _revise_decision(self, state: Mapping[str, Any]) -> tuple[str, str]:
        """Revise, while revisions are left and the last one raised the audit's score."""
        if not self._p.reflection_enabled:
            return _FINALIZE, _REFLECTION_OFF
        if state.get("revision", 0) >= self._p.max_revisions:
            return _FINALIZE, _REVISIONS_SPENT
        scores = state.get("audit_scores", [])
        if state.get("revision", 0) and len(scores) > 1:
            if scores[-1] - scores[-2] < self._p.min_score_gain:
                return _FINALIZE, _REVISION_PLATEAU
        if truncated_by_budget(self.resources.budget.exhausted_counters()):
            return _FINALIZE, _BUDGET_SPENT
        return _REVISE, _WRITING_FAULTS

    @staticmethod
    def _check_of(audit: AuditResult, item_id: str) -> str:
        return next((v.name for v in audit.verdicts if v.item_id == item_id), "")

    def _budget_cause(self, because: str) -> str | None:
        """The counter behind a budget-decided finish after the audit; None for any other."""
        if because == _GAP_FILL_EXHAUSTED:
            return GAP_FILL_ROUNDS
        if because == _BUDGET_SPENT:
            return next(iter(self.resources.budget.exhausted_counters()), None)
        return None

    async def gap_fill(self, state: RunState) -> Update:
        r, audit = self.resources, state.get("audit")
        assert audit is not None
        r.budget.charge(GAP_FILL_ROUNDS, 1)
        ctx = self._p.context(state["contract"], r.ledger.snapshot(), None)
        gaps = [v for v in audit.verdicts if v.must_pass and not v.passed]
        tried = [q for queries in r.trace.queries_by_node.values() for q in queries]
        proposal = await self._p.reflector.propose(
            ctx, gaps, tried, self._p.gap_fill_round_offset + state["gap_fill_round"]
        )
        follow_ups = self._numbered(
            fit_to_budget(proposal.accepted, r.budget.remaining(SEARCHES)), state
        )
        step = cards.FollowUpStep(
            gaps=gaps,
            proposal=proposal,
            planned=follow_ups,
            reason="",
            rubric=state["contract"].rubric,
            entries=ctx.ledger.entries,
            limit=self._p.clip_chars,
        )
        heading, sections = cards.gap_fill_card(state["gap_fill_round"] + 1, step)
        r.telemetry.card("gap_fill", heading, sections)
        planned = bool(follow_ups)
        self._report_loop(
            state,
            _BATCH_START if planned else _FINALIZE,
            _GAP_FILL_PLANNED if planned else _NO_FOLLOW_UP,
            {"gap_fill_round": f"{state['gap_fill_round'] + 1}", "gaps": len(gaps)},
        )
        update: Update = {"gap_fill_round": state["gap_fill_round"] + 1, "round_nodes": follow_ups}
        if follow_ups:
            # Only a plan that runs is a plan: an empty one would be recorded as run.
            plan = PlanDag(nodes=tuple(follow_ups), rationale="gap fill")
            update["plans"] = [*state.get("plans", []), plan]
        return update

    @staticmethod
    def after_gap_fill(state: RunState) -> str:
        return _BATCH_START if state.get("round_nodes") else _FINALIZE

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
        # run is COMPLETE. Evidence admitted under a caveat (a snippet, outside the window) is
        # what the run could find, not what it was asked for: PARTIAL.
        weaker = self._rests_on_weaker_evidence()
        return "PARTIAL" if truncated or weaker or verdict.outcome != "PASS" else "COMPLETE"

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
        self._remember(state)
        r.ledger.discard()
        return {"index": index}

    def _remember(self, state: RunState) -> None:
        """What this run's critics found, left in the dream inbox for `make dream ra`."""
        status = state.get("status")
        if self._p.run_events is None or status in (None, "CANCELLED"):
            return
        audit, r = state.get("audit"), self.resources
        failed = [v for v in audit.verdicts if not v.passed and v.applicable] if audit else []
        event = self._p.run_events.record(
            RunOutcome(
                contract=state["contract"],
                status=str(status),
                failed=failed,
                rejected_codes=dict(r.trace.rejected_codes),
                budget_stops=list(r.trace.budget_stops),
                audit_scores=list(state.get("audit_scores", [])),
                revisions=state.get("revision", 0),
            )
        )
        r.telemetry.audit("memory.run_event", "allow", f"{len(event.signals)} signal(s) recorded")

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

    def _batch_summary(self, batch: int, plans: list[PlanDag]) -> None:
        """What became of this run's search hits so far: admitted, or rejected and why."""
        r = self.resources
        entries = r.ledger.snapshot().entries
        questions = {n.node_id: n.question for plan in plans for n in plan.nodes}
        heading, sections = cards.batch_result_card(
            batch,
            r.trace.sources_examined,
            entries,
            r.trace.rejected_codes,
            questions,
            self._p.reasons,
            self._p.clip_chars,
        )
        r.telemetry.card(
            "batch_summary", heading, sections, status="completed" if entries else "warning"
        )

    def _report_loop(
        self,
        state: RunState,
        next_step: str,
        because: str,
        criteria: dict[str, Any] | None = None,
        cause: str | None = None,
    ) -> None:
        """LOOP CONTROL: the round, every budget meter, the step the run takes next, and the
        counter that forced it when one did."""
        r = self.resources
        RunLoopReport(r.telemetry.reporter, get_agent_settings().run_loop).report(
            r.budget,
            round_number=state["round"],
            next_step=next_step,
            because=because,
            criteria=criteria,
            cause=cause,
        )


# What became of a cited sentence, as the shared card names a claim's verdict.
_GUARD_VERDICTS = {"supported": "supported", "rewritten": "rewritten", "dropped": "removed"}


def _sentence_check(check: SentenceCheck) -> dict[str, Any]:
    """One cited sentence as a claim of the shared card: its verdict and the steps behind it."""
    return {
        "claim": check.sentence,
        "verdict": _GUARD_VERDICTS[check.outcome],
        "cited": check.cited,
        **({"rewrite": check.rewrite} if check.rewrite else {}),
        "readings": [reading.model_dump() for reading in check.readings],
    }


def _spent(budget: BudgetGuard) -> dict[str, float]:
    return {name: counter.spent for name, counter in budget.snapshot().items()}
