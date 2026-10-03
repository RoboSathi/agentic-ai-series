"""Planning, the research worker's defences, and the Independent Audit."""

import inspect
import json
import re
from typing import Any

import pytest

from agent_platform.evidence import EvidenceCandidate, LedgerSnapshot
from agent_platform.primitives.hashing import text_hash
from agent_platform.verification.acceptance import CheckVerdict
from agent_platform.web import SearchHit
from research_agent.audit.independent import IndependentAuditor, _result
from research_agent.config import get_agent_settings
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import COVERAGE_AND_LIMITS, LimitEntry, Report, ReportSection
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import ScopeBrief, SourcePolicy
from research_agent.planning.dag_validation import validate_plan
from research_agent.prompts import planner, reflector, synthesizer
from research_agent.research.defenses import OutboundGuard, apply_source_policy, is_primary
from research_agent.research.stop_rules import StopInput, stop_rule_for

MARKER = "UNIQUE-RAW-PAGE-SENTENCE-7f3a"


def _brief(**fields: object) -> ScopeBrief:
    data: dict[str, object] = {
        "archetype": "decision_support",
        "topic": "t",
        "budget": get_agent_settings().budget,
        "options": ("X",),
        "criteria": ("cost",),
        "private_terms": ("Acme Co-op", "2 million"),
    }
    data.update(fields)
    return ScopeBrief.model_validate(data)


def _contract(rubric: Rubric | None = None, **fields: object) -> Contract:
    from research_agent.rubric.generator import RubricGenerator

    brief = _brief(**fields)
    rubric = rubric or RubricGenerator(
        get_agent_settings().rubric.resolved_templates_path
    ).generate(brief)
    return Contract.freeze("c1", brief, rubric, brief.budget)


class TestPlanValidation:
    def test_a_valid_plan_passes(self, ra_config: None) -> None:
        contract = _contract()
        plan = PlanDag(
            nodes=(SubQuestion(node_id="a", question="q", rubric_items=("R1",), queries=("q",)),)
        )
        assert validate_plan(plan, contract.rubric, 4).valid

    def test_cycles_limits_and_untraced_nodes_are_rejected(self, ra_config: None) -> None:
        contract = _contract()
        plan = PlanDag(
            nodes=(
                SubQuestion(
                    node_id="a",
                    question="q",
                    depends_on=("b",),
                    rubric_items=("R1",),
                    queries=("q",),
                ),
                SubQuestion(
                    node_id="b", question="q", depends_on=("a",), rubric_items=("R999",), queries=()
                ),
            )
        )
        result = validate_plan(plan, contract.rubric, 1)
        assert not result.valid and set(result.cycle) == {"a", "b"}
        text = " ".join(result.problems)
        assert "exceeds" in text and "traces to no rubric item" in text and "no queries" in text
        assert not validate_plan(PlanDag(), contract.rubric, 1).valid


class TestOutboundDefence:
    def test_private_terms_never_leave(self, ra_config: None) -> None:
        guard = OutboundGuard(_brief(), get_agent_settings().research)
        assert not guard.check_query("acme co-op storage quote").allowed
        assert guard.check_query("orrin ferrocell cost per kwh").allowed

    def test_exfiltration_urls_and_blocked_domains_are_refused(self, ra_config: None) -> None:
        brief = _brief(source_policy=SourcePolicy(blocked_domains=("spam.example",)))
        guard = OutboundGuard(brief, get_agent_settings().research)
        assert (
            guard.check_url("https://collect.storage-deals.example/log?q=x").reason_code
            == "DENY_RULE"
        )
        assert guard.check_url("https://a.spam.example/p").reason_code == "BLOCKED_DOMAIN"
        assert guard.check_url("https://ok.example/p").allowed
        assert guard.check_query("\x00\x01").reason_code == "NOT_ALLOWED"

    def test_source_policy_filters_and_orders_hits(self, ra_config: None) -> None:
        brief = _brief(
            source_policy=SourcePolicy(
                preferred_domains=("good.example",), blocked_domains=("bad.example",)
            )
        )
        hits = [
            SearchHit(url="https://x.example/1"),
            SearchHit(url="https://bad.example/2"),
            SearchHit(url="https://good.example/3"),
        ]
        assert [h.url for h in apply_source_policy(hits, brief)] == [
            "https://good.example/3",
            "https://x.example/1",
        ]

    def test_primary_sources_by_host_marker(self) -> None:
        assert is_primary("https://norvale-lab.example/r", ["lab"]) and not is_primary(
            "https://blog.example", ["lab"]
        )


class TestStopRules:
    @pytest.mark.parametrize(
        ("archetype", "check"),
        [
            ("decision_support", "subject_criterion_coverage"),
            ("fact_verification", "claim_verdict"),
            ("explainer", "concept_covered"),
        ],
    )
    def test_coverage_rules(self, archetype: str, check: str) -> None:
        rule = stop_rule_for(archetype)  # type: ignore[arg-type]
        met = CheckVerdict(
            item_id="R1",
            kind="deterministic",
            name=check,
            passed=True,
            score=1,
            threshold=1,
            must_pass=True,
        )
        assert rule.decide(StopInput(verdicts=[met])).stop
        assert not rule.decide(StopInput(verdicts=[met.model_copy(update={"passed": False})])).stop
        assert not rule.decide(StopInput()).stop

    def test_landscape_stops_when_no_new_entities_appear(self) -> None:
        rule = stop_rule_for("landscape_brief")
        assert rule.decide(StopInput(entities_before=4, entities_after=4)).stop
        assert not rule.decide(StopInput(entities_before=2, entities_after=4)).stop


class TestNoRawPageTextOutsideTheExtractor:
    def _ledger(self) -> LedgerSnapshot:
        from agent_platform.evidence import EvidenceEntry

        page = f"Orrin costs 95 USD. {MARKER}."
        entry = EvidenceEntry(
            claim="Orrin costs 95 USD.",
            span="Orrin costs 95 USD.",
            source_url="https://p.example",
            source_hash=text_hash(page),
            entry_id="E1",
        )
        return LedgerSnapshot(entries=(entry,))

    def test_planner_reflector_and_synthesizer_payloads_hold_no_page_text(
        self, ra_config: None
    ) -> None:
        contract, ledger = _contract(), self._ledger()
        prompts = [
            planner.build(contract, 4, 2),
            reflector.build(
                contract, [{"item_id": "R1"}], ["q"], [e.claim for e in ledger.entries], 2
            ),
            synthesizer.build(contract, ledger, ["Recommendation"], []),
        ]
        for prompt in prompts:
            body = json.dumps(prompt.payload)
            assert MARKER not in body and "span" not in json.dumps(synthesizer.ledger_view(ledger))
            assert (
                "Acme Co-op" not in body
            )  # the Brief's private context never reaches planning roles

    def test_the_builders_take_no_page_argument(self) -> None:
        for build in (planner.build, reflector.build, synthesizer.build):
            params = inspect.signature(build).parameters
            assert not {"page", "text", "pages", "raw"} & set(params)


class TestIndependentAuditIsolation:
    def test_audit_report_takes_exactly_the_contract_the_report_and_the_ledger(self) -> None:
        signature = inspect.signature(IndependentAuditor.audit_report)
        params = [p for p in signature.parameters.values() if p.name != "self"]
        assert [p.name for p in params] == ["contract", "report", "ledger_snapshot"]
        hints = inspect.get_annotations(IndependentAuditor.audit_report, eval_str=True)
        assert (
            hints["contract"] is Contract
            and hints["report"] is Report
            and hints["ledger_snapshot"] is LedgerSnapshot
        )


def _reachable_types(root: object, depth: int = 4) -> set[str]:
    seen: set[int] = set()
    names: set[str] = set()
    frontier: list[tuple[object, int]] = [(root, 0)]
    while frontier:
        obj, level = frontier.pop()
        if id(obj) in seen or level > depth:
            continue
        seen.add(id(obj))
        names.add(type(obj).__name__)
        children: list[Any] = []
        if hasattr(obj, "__dict__"):
            children.extend(vars(obj).values())
        if hasattr(obj, "__closure__") and obj.__closure__:  # type: ignore[attr-defined]
            children.extend(cell.cell_contents for cell in obj.__closure__)  # type: ignore[attr-defined]
        if isinstance(obj, (list, tuple, set)):
            children.extend(obj)
        if isinstance(obj, dict):
            children.extend(obj.values())
        frontier.extend(
            (child, level + 1)
            for child in children
            if not isinstance(child, (str, bytes, int, float))
        )
    return names


class TestAuditResult:
    def test_gaps_and_honesty_are_derived_from_verdicts(self) -> None:
        verdicts = [
            CheckVerdict(
                item_id="R1",
                kind="deterministic",
                name="a",
                passed=False,
                score=0,
                threshold=1,
                must_pass=True,
            ),
            CheckVerdict(
                item_id="R2",
                kind="judge",
                name="b",
                passed=False,
                score=0,
                threshold=1,
                must_pass=False,
            ),
            CheckVerdict(
                item_id="R3",
                kind="deterministic",
                name="c",
                passed=True,
                score=1,
                threshold=1,
                must_pass=True,
            ),
        ]
        report = Report(
            title="t",
            status="PARTIAL",
            sections=[ReportSection(heading=COVERAGE_AND_LIMITS, body="")],
            limits=[LimitEntry(item_id="R3", text="x")],
        )
        result = _result(verdicts, report, "m", True)
        assert result.gaps == ["R1", "R2"] and result.must_pass_gaps == ["R1"]
        assert (
            result.unflagged_gaps == ["R1"] and result.false_limits == ["R3"] and not result.honest
        )


def test_citation_pattern_is_configured(ra_config: None) -> None:
    assert re.compile(get_agent_settings().synthesis.citation_pattern).findall("x [E1] [E22]") == [
        "E1",
        "E22",
    ]


def test_evidence_candidates_carry_the_source_date() -> None:
    candidate = EvidenceCandidate(
        claim="c", span="s", source_url="u", source_hash="h", source_date="2026-01-01"
    )
    assert candidate.source_date == "2026-01-01"
