"""Scoping rules and rubric generation, unit by unit (no graph)."""

from datetime import UTC, datetime

import pytest

from research_agent.config import get_agent_settings
from research_agent.domain.budget import Budget
from research_agent.domain.rubric import Contract, Rubric
from research_agent.domain.scope import CoverageItem, KnownUnknown, ScopeBrief, SourcePolicy
from research_agent.prompts.interviewer import InterviewTurn
from research_agent.rubric.editor import RubricEditError, RubricEditor, render
from research_agent.rubric.freeze import contract_artifacts, freeze_contract
from research_agent.rubric.generator import RubricGenerator
from research_agent.scoping import checklist
from research_agent.scoping.brief import BriefBuilder
from research_agent.scoping.challenge_check import ChallengeGuard
from research_agent.scoping.termination import TerminationPolicy


def _budget() -> Budget:
    return get_agent_settings().budget


def _items() -> list[CoverageItem]:
    return [
        CoverageItem(item_id="a", prompt="A?", required=True),
        CoverageItem(item_id="b", prompt="B?", required=False),
    ]


class TestChecklist:
    @pytest.mark.parametrize(
        "archetype", ["decision_support", "landscape_brief", "fact_verification", "explainer"]
    )
    def test_every_archetype_loads_its_items_and_the_common_ones(
        self, ra_config: None, archetype: str
    ) -> None:
        items = checklist.load_checklist(
            get_agent_settings().scoping.resolved_checklists_path, archetype
        )  # type: ignore[arg-type]
        ids = [i.item_id for i in items]
        assert "purpose" in ids and "context" in ids and all(i.status == "open" for i in items)
        assert any(i.private for i in items)

    def test_waive_one_or_all(self) -> None:
        one = checklist.waive(_items(), "b", "all")
        assert [i.status for i in one] == ["open", "waived"]
        every = checklist.waive(_items(), "all", "all")
        assert {i.status for i in every} == {"waived"}

    def test_known_unknowns_are_waived_and_open_required_items(self) -> None:
        items = checklist.with_status(_items(), "b", "waived")
        unknowns = checklist.known_unknowns(items, "closed_early")
        assert [(u.item_id, u.reason) for u in unknowns] == [("a", "closed_early"), ("b", "waived")]
        covered = checklist.with_status(_items(), "a", "covered", "x")
        assert checklist.known_unknowns(covered, "closed_early") == []
        assert checklist.item(covered, "a") is not None and checklist.item(covered, "z") is None


class TestTermination:
    def test_done_is_only_proposed_when_nothing_required_is_open(self) -> None:
        policy = TerminationPolicy(max_turns=5)
        proposing = InterviewTurn(proposes_done=True)
        early = policy.decide(proposing, _items(), 1)
        assert early.outcome == "ask" and early.overruled and early.open_required == ["a"]
        ready = checklist.with_status(_items(), "a", "covered", "x")
        assert policy.decide(proposing, ready, 1).outcome == "propose_done"
        assert policy.decide(InterviewTurn(), ready, 1).outcome == "ask"

    def test_the_turn_cap_stops_asking(self) -> None:
        assert (
            TerminationPolicy(max_turns=3).decide(InterviewTurn(), _items(), 3).outcome
            == "turn_cap"
        )


class TestChallengeGuard:
    def test_k_turns_without_a_challenge_is_a_violation(self) -> None:
        guard = ChallengeGuard(every_k_turns=2)
        quiet = InterviewTurn(next_item_id="b")
        assert not guard.check(quiet, "a", 0).violated
        assert guard.check(quiet, "a", 1).violated

    def test_a_challenge_or_follow_up_resets_the_count(self) -> None:
        guard = ChallengeGuard(every_k_turns=2)
        assert guard.check(InterviewTurn(challenges=["why?"]), "a", 5).turns_without_challenge == 0
        assert guard.check(InterviewTurn(next_item_id="a"), "a", 5).turns_without_challenge == 0


class TestBriefBuilder:
    def test_answers_become_typed_fields(self, ra_config: None) -> None:
        settings = get_agent_settings().scoping
        items = [
            CoverageItem(
                item_id="options",
                prompt="?",
                status="covered",
                value="X, Y and Z",
                brief_field="options",
                multi=True,
            ),
            CoverageItem(
                item_id="recency",
                prompt="?",
                status="covered",
                value="within 365 days",
                brief_field="recency_days",
            ),
            CoverageItem(
                item_id="sources",
                prompt="?",
                status="covered",
                value="prefer lab.example; avoid spam.example. primary sources",
                brief_field="source_policy",
            ),
            CoverageItem(
                item_id="context",
                prompt="?",
                status="covered",
                value="none",
                brief_field="private_terms",
                multi=True,
            ),
        ]
        brief = BriefBuilder(settings, _budget(), _budget()).build(
            "decision_support", "t", items, []
        )
        assert brief.options == ("X", "Y", "Z") and brief.recency_days == 365
        assert brief.source_policy.preferred_domains == ("lab.example",)
        assert brief.source_policy.blocked_domains == ("spam.example",)
        assert brief.source_policy.primary_sources_preferred
        assert brief.private_terms == ()

    def test_any_recency_means_no_window(self, ra_config: None) -> None:
        item = CoverageItem(
            item_id="recency",
            prompt="?",
            status="covered",
            value="no limit",
            brief_field="recency_days",
        )
        brief = BriefBuilder(get_agent_settings().scoping, _budget(), _budget()).build(
            "explainer", "t", [item], []
        )
        assert brief.recency_days is None


def _brief(**fields: object) -> ScopeBrief:
    data: dict[str, object] = {
        "archetype": "decision_support",
        "topic": "t",
        "budget": _budget(),
        "options": ("X", "Y"),
        "criteria": ("cost", "safety"),
    }
    data.update(fields)
    return ScopeBrief.model_validate(data)


class TestRubricGeneration:
    def test_every_criterion_and_known_unknown_is_covered(self, ra_config: None) -> None:
        unknown = KnownUnknown(item_id="constraints", question="Any constraints?", reason="waived")
        rubric = RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(
            _brief(
                known_unknowns=(unknown,),
                recency_days=90,
                source_policy=SourcePolicy(
                    blocked_domains=("spam.example",),
                    min_distinct_domains=3,
                    primary_sources_preferred=True,
                ),
            )
        )
        mapped = {key for item in rubric.items for key in item.maps_to}
        assert {"cost", "safety", "constraints"} <= mapped
        names = {item.check.name for item in rubric.items}
        assert {
            "recency_window",
            "blocked_domains_absent",
            "distinct_domains",
            "primary_sources_cited",
            "links_resolved",
        } <= names
        assert [i.item_id for i in rubric.items] == [
            f"R{n}" for n in range(1, len(rubric.items) + 1)
        ]

    @pytest.mark.parametrize(
        ("archetype", "field", "check"),
        [
            ("fact_verification", "claims", "claim_verdict"),
            ("explainer", "concepts", "concept_covered"),
        ],
    )
    def test_subjects_get_their_own_items(
        self, ra_config: None, archetype: str, field: str, check: str
    ) -> None:
        brief = _brief(archetype=archetype, options=(), criteria=(), **{field: ("p", "q")})
        rubric = RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(
            brief
        )
        assert sum(item.check.name == check for item in rubric.items) == 2

    def test_generation_is_deterministic(self, ra_config: None) -> None:
        generator = RubricGenerator(get_agent_settings().rubric.resolved_templates_path)
        assert generator.generate(_brief()) == generator.generate(_brief())
        assert generator.stop_rule("landscape_brief") == "no_new_entities"


class TestRubricEditing:
    def _rubric(self) -> Rubric:
        return RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(
            _brief()
        )

    def _editor(self) -> RubricEditor:
        return RubricEditor(
            RubricGenerator(get_agent_settings().rubric.resolved_templates_path).rules["user"]
        )

    def test_add_drop_set(self, ra_config: None) -> None:
        rubric, editor = self._rubric(), self._editor()
        added = editor.add(rubric, "Mention lead times")
        assert added.items[-1].origin == "user" and added.items[-1].check.kind == "judge"
        dropped = editor.drop(added, added.items[-1].item_id)
        assert len(dropped.items) == len(rubric.items)
        changed = editor.set(rubric, "R1", ["threshold=0.5", "must_pass=false"])
        assert changed.item("R1").threshold == 0.5 and not changed.item("R1").must_pass  # type: ignore[union-attr]
        assert "R1" in render(changed)

    def test_the_last_item_for_a_criterion_cannot_be_dropped(self, ra_config: None) -> None:
        rubric = self._rubric()
        only = next(item for item in rubric.items if item.maps_to == ("cost",))
        with pytest.raises(RubricEditError):
            self._editor().drop(rubric, only.item_id)

    @pytest.mark.parametrize("args", [["threshold=x"], ["must_pass=maybe"], ["color=red"], []])
    def test_bad_assignments_are_refused(self, ra_config: None, args: list[str]) -> None:
        with pytest.raises(RubricEditError):
            self._editor().set(self._rubric(), "R1", args)

    def test_unknown_ids_and_empty_adds_are_refused(self, ra_config: None) -> None:
        editor = self._editor()
        for action in (
            lambda: editor.drop(self._rubric(), "R99"),
            lambda: editor.set(self._rubric(), "R99", ["threshold=1"]),
            lambda: editor.add(self._rubric(), "  "),
        ):
            with pytest.raises(RubricEditError):
                action()


class TestFreezing:
    def test_the_contract_hash_detects_any_change(self, ra_config: None) -> None:
        rubric = RubricGenerator(get_agent_settings().rubric.resolved_templates_path).generate(
            _brief()
        )
        frozen_at = datetime(2026, 9, 30, tzinfo=UTC)
        contract = freeze_contract(_brief(), rubric, "run", frozen_at)
        assert contract.verify() and contract.frozen_at == frozen_at
        tampered = Contract.model_construct(
            **{**contract.__dict__, "rubric": Rubric(items=rubric.items[:1])}
        )
        assert not tampered.verify()
        brief, frozen_rubric = contract_artifacts(contract)
        assert brief.kind == "brief" and frozen_rubric.kind == "rubric" and brief.verify()
        with pytest.raises(Exception):
            contract.contract_hash = "x"  # type: ignore[misc]

    def test_a_brief_may_lower_the_budget_but_never_raise_it(self, ra_config: None) -> None:
        ceiling = _budget()
        asked = ceiling.model_copy(
            update={"max_searches": ceiling.max_searches * 10, "max_fetches": 1}
        )
        capped = asked.capped_by(ceiling)
        assert capped.max_searches == ceiling.max_searches and capped.max_fetches == 1
