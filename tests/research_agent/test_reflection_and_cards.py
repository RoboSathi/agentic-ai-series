"""/research fixes of 2026-10-11: the plan shows its DAG, a vague answer is never a subject,
the cross-check is entailment from agreeing domains, the audit routes writing faults to a
revision (Producer-Critic) and evidence gaps to gap-fill, and what a run's critics found
outlives it as a dream event (ra-0023)."""

from typing import Any

import pytest

from agent_platform.evidence import EvidenceEntry, LedgerSnapshot
from agent_platform.memory import DreamEvent
from agent_platform.observability.types import StatusEvent
from agent_platform.scoring.entailment import EntailmentResult
from agent_platform.verification.acceptance import CheckSpec, CheckVerdict
from research_agent.audit.checks import claim_verdict
from research_agent.audit.independent import AuditResult
from research_agent.config import get_agent_settings
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.rubric import Contract, Rubric, RubricCheck, RubricItem
from research_agent.domain.scope import CoverageItem, ScopeBrief
from research_agent.memory import RunEventWriter, RunOutcome, research_memory_profile
from research_agent.planning.dag_validation import validate_plan
from research_agent.research.cards import plan_tree, revise_card
from research_agent.scoping import checklist
from tests.research_agent.conftest import BuildHarness

pytestmark = pytest.mark.usefixtures("ra_config")


def _item(item_id: str, name: str, kind: str = "deterministic", must: bool = True) -> RubricItem:
    return RubricItem(
        item_id=item_id,
        text=f"criterion {item_id}",
        check=RubricCheck(kind=kind, name=name),  # type: ignore[arg-type]
        threshold=1.0,
        must_pass=must,
        origin="template",
    )


def _contract(*items: RubricItem, topic: str = "reflection pattern in agents") -> Contract:
    brief = ScopeBrief(
        archetype="explainer", topic=topic, budget=get_agent_settings().budget, concepts=("x",)
    )
    return Contract.freeze("c-1", brief, Rubric(items=items), brief.budget)


def _node(node_id: str, items: tuple[str, ...], depends: tuple[str, ...] = ()) -> SubQuestion:
    return SubQuestion(
        node_id=node_id,
        question=f"question {node_id}",
        depends_on=depends,
        rubric_items=items,
        queries=(f"query {node_id}",),
    )


class TestThePlanShowsItsDag:
    def test_batches_dependencies_and_queries_are_drawn(self) -> None:
        plan = PlanDag(
            nodes=(_node("Q1", ("R1",)), _node("Q2", ("R1",)), _node("Q3", ("R2",), ("Q1",))),
            rationale="define it, then compare",
        )
        text = "\n".join(plan_tree(plan))
        assert "Plan: define it, then compare" in text
        assert "Batch 1 · 2 sub-question(s) in parallel" in text
        assert "Batch 2 · 1 sub-question(s), after batch 1" in text
        assert "Q3 · question Q3 → R2 · needs Q1" in text
        assert "queries: “query Q1”" in text

    def test_a_must_pass_criterion_no_node_serves_is_reported(self) -> None:
        contract = _contract(_item("R1", "concept_covered"), _item("R2", "concept_covered"))
        plan = PlanDag(nodes=(_node("Q1", ("R1",)),))
        covered = get_agent_settings().planning.covered_checks
        validation = validate_plan(plan, contract.rubric, 4, covered)
        assert validation.valid and validation.uncovered == ["R2"]


class TestAVagueAnswerIsNeverASubject:
    def _items(self) -> list[CoverageItem]:
        return [CoverageItem(item_id="concepts", goal="g", brief_field="concepts", multi=True)]

    def test_all_leaves_the_item_open_then_delegates_it(self) -> None:
        items = checklist.vague_answer(self._items(), "concepts", delegate_after=2)
        assert items[0].status == "open" and items[0].vague_answers == 1
        items = checklist.vague_answer(items, "concepts", delegate_after=2)
        assert items[0].status == "delegated"
        # A delegated item still blocks finishing, until the interviewer proposes values.
        assert checklist.open_items(items, required_only=True)

    def test_a_proposed_value_is_marked_as_derived(self) -> None:
        items = checklist.covered(self._items(), "concepts", "a, b, c", derived=True)
        assert items[0].status == "covered" and items[0].derived


def _entry(entry_id: str, url: str, claim: str) -> EvidenceEntry:
    return EvidenceEntry(
        entry_id=entry_id, claim=claim, span=claim, source_url=url, source_hash="h"
    )


class _Nli:
    """Entails every claim by its own span; contradicts any pair naming 'never'."""

    async def entails(self, premise: str, hypothesis: str) -> EntailmentResult:
        clash = ("never" in premise) != ("never" in hypothesis)
        label = "contradiction" if clash and premise != hypothesis else "entailment"
        return EntailmentResult(label=label, entailed=label == "entailment", score=0.99)


def _ctx(entries: list[EvidenceEntry], nli: Any) -> Any:
    import re

    from research_agent.audit.context import CheckContext
    from research_agent.domain.matching import TermMatcher

    contract = _contract(_item("R1", "concept_covered", "nli"), topic="reflection")
    return CheckContext(
        contract=contract,
        ledger=LedgerSnapshot(entries=tuple(entries)),
        report=None,
        matcher=TermMatcher(get_agent_settings().matching),
        citation_pattern=re.compile(get_agent_settings().synthesis.citation_pattern),
        entity_pattern=re.compile("x"),
        primary_markers=(),
        entailment=nli,
    )


def _spec() -> CheckSpec:
    return CheckSpec(
        item_id="R1", kind="nli", name="concept_covered",
        params={"subject": "reflection", "min_sources": 2, "max_pairs": 12}, threshold=1.0,
        must_pass=True,
    )  # fmt: skip


class TestTheCrossCheckIsAgreementNotMention:
    async def test_two_domains_that_agree_pass(self) -> None:
        entries = [
            _entry("E1", "https://a.example/x", "Reflection drafts and revises."),
            _entry("E2", "https://b.example/y", "Reflection critiques its own draft."),
        ]
        outcome = await claim_verdict(_spec(), _ctx(entries, _Nli()))
        assert outcome.score == 1.0 and outcome.evidence == ["a.example", "b.example"]

    async def test_two_domains_that_contradict_do_not(self) -> None:
        entries = [
            _entry("E1", "https://a.example/x", "Reflection drafts and revises."),
            _entry("E2", "https://b.example/y", "Reflection never revises a draft."),
        ]
        outcome = await claim_verdict(_spec(), _ctx(entries, _Nli()))
        assert outcome.score == 0.0 and "contradicted" in outcome.detail


def _verdict(item_id: str, name: str, passed: bool, kind: str = "deterministic") -> CheckVerdict:
    return CheckVerdict(
        item_id=item_id, kind=kind, name=name, passed=passed, score=1.0 if passed else 0.3,  # type: ignore[arg-type]
        threshold=0.7, must_pass=kind != "judge", detail="thin" if not passed else "",
    )  # fmt: skip


@pytest.fixture
def bound(harness: BuildHarness) -> tuple[Any, list[StatusEvent]]:
    h = harness()
    contract = _contract(
        _item("R1", "concept_covered", "nli"), _item("R2", "explanation_accuracy", "judge", False)
    )
    h.agent._nodes.bind(h.agent._resources(contract))
    events: list[StatusEvent] = []
    h.feed.subscribe(events.append)
    return h.agent._nodes, events


class TestTheAuditRoutesByFault:
    def test_a_writing_fault_goes_to_a_revision(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        audit = AuditResult(verdicts=[_verdict("R2", "explanation_accuracy", False, "judge")])
        assert nodes._gap_fill_decision(audit, {}) == ("revise", "writing_faults")

    def test_missing_evidence_goes_to_gap_fill_first(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        if nodes.resources.budget.remaining("gap_fill_rounds") <= 0:
            pytest.skip("this budget allows no gap-fill round")
        audit = AuditResult(
            verdicts=[
                _verdict("R1", "concept_covered", False, "nli"),
                _verdict("R2", "explanation_accuracy", False, "judge"),
            ],
            must_pass_gaps=["R1"],
        )
        assert nodes._gap_fill_decision(audit, {})[0] == "gap_fill"

    def test_a_revision_that_gained_nothing_ends_it(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        audit = AuditResult(verdicts=[_verdict("R2", "explanation_accuracy", False, "judge")])
        state = {"revision": 1, "audit_scores": [0.6, 0.6]}
        assert nodes._gap_fill_decision(audit, state) == ("finalize", "revision_plateau")

    def test_revisions_are_bounded(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        audit = AuditResult(verdicts=[_verdict("R2", "explanation_accuracy", False, "judge")])
        cap = get_agent_settings().reflection.max_revisions
        state = {"revision": cap, "audit_scores": [0.5, 0.7]}
        assert nodes._gap_fill_decision(audit, state) == ("finalize", "revisions_spent")

    def test_a_judge_that_is_the_writer_is_not_independent(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        nodes._p.judge_models = lambda: {"writer-m"}
        audit = nodes._independence(AuditResult(), "writer-m")
        assert audit.independent is False


class TestTheCritiqueNeedsRoomToAct:
    async def test_with_no_round_left_no_critique_is_asked(self, bound: tuple[Any, Any]) -> None:
        nodes, _ = bound
        budget = nodes.resources.budget
        budget.charge("rounds", budget.remaining("rounds"))
        update, proposal = await nodes._critiqued({"round": 1}, None, [], [], "answered")
        assert proposal.call is None and update["round_nodes"] == []
        assert "no round is left" in update["stop_reason"]


class TestABatchReportsWhereTheLoopGoes:
    def test_a_finished_batch_says_continue_in_the_agents_words(
        self, bound: tuple[Any, list[StatusEvent]]
    ) -> None:
        nodes, events = bound
        nodes._report_loop({"round": 1}, "reflect", "round_done", {"batch": 1})
        [card] = [e for e in events if e.stage == "loop_control"]
        assert card.fields["decision"] == "continue" and "intervened" not in card.fields
        assert any(line.startswith("pages read:") for line in card.fields["budgets"])


def test_the_revise_card_counts_what_changed() -> None:
    before = [{"heading": "H", "body": "Kept one. Old two."}]
    after = [{"heading": "H", "body": "Kept one. New two. New three."}]
    finding = {"item_id": "R2", "criterion": "accurate", "reason": "thin"}
    heading, sections = revise_card(1, 2, [finding], before, after, None)
    assert "Revision 1 of 2" in heading and "2 sentence(s) rewritten or added" in heading
    assert "1 sentence(s) kept · 2 rewritten or added · 1 removed" in str(sections)


class TestWhatARunLeavesForTheDream:
    """ra-0023: a run's findings become one dream event; the dream turns them into memory."""

    def _outcome(self) -> RunOutcome:
        contract = _contract(_item("R2", "explanation_accuracy", "judge", False))
        return RunOutcome(
            contract=contract,
            status="PARTIAL",
            failed=[_verdict("R2", "explanation_accuracy", False, "judge")],
            rejected_codes={"https://a.example/x": "paywall", "https://a.example/y": "paywall"},
            budget_stops=["N1: searches exhausted"],
            audit_scores=[0.5, 0.7],
            revisions=1,
        )

    def test_the_run_event_names_its_findings_scoped_to_its_kind(self) -> None:
        settings = get_agent_settings().memory
        sent: list[DreamEvent] = []
        event = RunEventWriter(_Sink(sent), "ra", settings).record(self._outcome())
        assert sent == [event] and event.kind == settings.run_event_kind
        kinds = {(s.kind, s.value) for s in event.signals}
        assert (settings.signals.criterion_failed, "criterion R2") in kinds
        assert (settings.signals.source_rejected, "paywall") in kinds
        assert (settings.signals.budget_stop, "searches exhausted") in kinds
        assert all(s.scope == event.scope[0] for s in event.signals)

    async def test_the_run_is_summarised_from_its_record(self) -> None:
        settings = get_agent_settings().memory
        event = RunEventWriter(_Sink([]), "ra", settings).record(self._outcome())
        summary = await research_memory_profile().episodic.summarize([event])
        assert summary is not None
        assert "PARTIAL" in summary.text and "0.50 → 0.70" in summary.text

    def test_an_unusable_source_becomes_a_fact_about_its_domain(self) -> None:
        settings = get_agent_settings().memory
        event = RunEventWriter(_Sink([]), "ra", settings).record(self._outcome())
        facts = research_memory_profile().semantic.facts(event.session_id, "", [event])
        [source] = [f for f in facts if "a.example" in f.key]
        assert source.role == "system_of_record" and "paywall" in source.value


class _Sink:
    def __init__(self, sent: list[DreamEvent]) -> None:
        self._sent = sent

    def emit(self, event: DreamEvent) -> None:
        self._sent.append(event)


class TestTheSearchHeadingNamesItsProvider:
    def test_each_route_is_worded(self) -> None:
        from agent_platform.web.types import SearchRoute
        from research_agent.presentation.narration import search_route

        assert search_route(SearchRoute(answered_by="duckduckgo", mode="live")) == "ddgs"
        merged = SearchRoute(answered_by="duckduckgo", augmented_with="tavily", mode="live")
        assert search_route(merged) == "ddgs + tavily (weak result, merged)"
        fallback = SearchRoute(answered_by="tavily", failed=["duckduckgo"], mode="live")
        assert search_route(fallback) == "tavily (ddgs failed)"
        assert search_route(SearchRoute(answered_by="tavily", cached=True)) == "tavily · cached"
        assert search_route(SearchRoute(answered_by="tavily", mode="replay")) == "tavily · replay"


def test_an_all_answer_to_a_subject_item_names_nothing() -> None:
    from types import SimpleNamespace

    from research_agent.graph.scoping_nodes import ScopingNodes

    nodes = ScopingNodes.__new__(ScopingNodes)
    nodes._p = SimpleNamespace(settings=get_agent_settings().scoping)  # type: ignore[assignment]
    concepts = CoverageItem(item_id="concepts", goal="g", brief_field="concepts", multi=True)
    purpose = CoverageItem(item_id="purpose", goal="g", brief_field="purpose")
    assert nodes._names_nothing(concepts, "All of them")
    assert not nodes._names_nothing(concepts, "memory, planning")
    assert not nodes._names_nothing(purpose, "all")


def test_the_plan_card_shows_the_dag_once_not_the_raw_answer_too() -> None:
    from agent_platform.observability.step_card import ModelCall
    from research_agent.research.cards import plan_card

    call = ModelCall(
        role="planner", model="m", tokens=1, seconds=1.0,
        response='{"nodes": [{"node_id": "Q1", "rubric_items": ["R1", "R2"]}]}',
    )  # fmt: skip
    plan = PlanDag(nodes=(_node("Q1", ("R1", "R2")),), rationale="r")
    _, sections = plan_card(_contract(_item("R1", "concept_covered")), plan, 1, call)
    assert "rubric_items" not in str(sections) and "Q1 · question Q1 → R1, R2" in str(sections)
    _, rejected = plan_card(_contract(), None, 0, call, rejected=["no node serves R7"])
    assert "rubric_items" in str(rejected)
