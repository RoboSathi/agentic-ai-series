"""The run's loop policy as a plain function of what the run has spent and seen.

No graph, no resources: each branch is a snapshot in and a decision out, so the card and the
edge (which both ask the policy) can be checked against the rule itself.
"""

from typing import Any

import pytest

from agent_platform.loop_control import LoopVerdict
from agent_platform.runtime.verification.acceptance import CheckVerdict
from research_agent.audit.independent import AuditResult
from research_agent.research.loop_policy import (
    TERMINAL_OUTCOME,
    LoopSnapshot,
    RunLoopPolicy,
)
from research_agent.research.stop_rules.base import StopDecision

pytestmark = pytest.mark.usefixtures("ra_config")

WRITING = "explanation_accuracy"


def _policy(*, reflection: bool = True, max_revisions: int = 2) -> RunLoopPolicy:
    return RunLoopPolicy(
        reflection_enabled=reflection,
        writing_checks=frozenset({WRITING}),
        max_revisions=max_revisions,
        min_score_gain=0.05,
    )


def _snap(**changes: Any) -> LoopSnapshot:
    base: dict[str, Any] = {
        "cancelled": False, "exhausted": (), "budget_stops": (), "rounds_left": 1,
        "gap_fill_rounds_left": 1,
    }  # fmt: skip
    return LoopSnapshot(**{**base, **changes})


def _verdict(item_id: str, name: str, passed: bool) -> CheckVerdict:
    return CheckVerdict(
        item_id=item_id, kind="judge", name=name, passed=passed, score=0.3, threshold=0.7,
        must_pass=True,
    )  # fmt: skip


ANSWERED = StopDecision(stop=True, reason="answered")
OPEN = StopDecision(stop=False, reason="gaps remain")


def _governed(decision: str, layer: str | None = "convergence") -> LoopVerdict:
    return LoopVerdict(
        decision=decision, layer=layer, reason_code="NO_PROGRESS",  # type: ignore[arg-type]
        reason="the last 3 steps added no new evidence",
    )  # fmt: skip


class TestTheGovernorsVerdict:
    def test_a_plateau_stops_the_run_with_the_governors_reason(self) -> None:
        verdict = _governed("finalize")
        call = _policy().after_reflect(_snap(governed=verdict), OPEN)
        assert call.stop and call.reason == verdict.reason and call.verdict == verdict

    def test_a_governed_stop_is_not_reopened_by_a_critique(self) -> None:
        call = _policy().after_reflect(_snap(governed=_governed("finalize")), OPEN)
        assert not call.judge and call.cause is None

    def test_a_continue_verdict_changes_nothing(self) -> None:
        call = _policy().after_reflect(_snap(governed=_governed("continue", None)), OPEN)
        assert not call.stop

    def test_the_budgets_speak_before_the_governor(self) -> None:
        snap = _snap(exhausted=("searches",), governed=_governed("finalize"))
        call = _policy().after_reflect(snap, OPEN)
        assert call.cause == "searches" and call.verdict is None

    def test_an_answered_run_is_answered_whatever_the_governor_says(self) -> None:
        call = _policy().after_reflect(_snap(governed=_governed("finalize")), ANSWERED)
        assert call.reason == "answered" and call.judge

    def test_every_run_status_ends_in_one_terminal_outcome(self) -> None:
        assert set(TERMINAL_OUTCOME) == {"COMPLETE", "PARTIAL", "CANCELLED", "FAILED"}


class TestAfterReflection:
    def test_an_answered_run_stops_and_is_judged(self) -> None:
        call = _policy().after_reflect(_snap(), ANSWERED)
        assert call.stop and call.judge and call.cause is None

    def test_an_open_run_with_room_goes_on(self) -> None:
        assert not _policy().after_reflect(_snap(), OPEN).stop

    def test_a_spent_counter_stops_it_and_is_named(self) -> None:
        call = _policy().after_reflect(_snap(exhausted=("searches",)), OPEN)
        assert call.stop and call.cause == "searches"

    def test_rounds_running_out_is_the_cap_deciding(self) -> None:
        call = _policy().after_reflect(_snap(rounds_left=0), OPEN)
        assert call.stop and call.cause == "rounds"

    def test_the_operator_stopping_it_is_never_judged(self) -> None:
        call = _policy().after_reflect(_snap(cancelled=True), ANSWERED)
        assert call.stop and not call.judge

    def test_without_reflection_nothing_is_judged(self) -> None:
        call = _policy(reflection=False).after_reflect(_snap(), ANSWERED)
        assert call.stop and not call.judge


class TestAfterTheAudit:
    def test_missing_evidence_goes_to_gap_fill(self) -> None:
        audit = AuditResult(must_pass_gaps=["R1"], verdicts=[_verdict("R1", "covered", False)])
        assert _policy().after_audit(_snap(), audit, {}).next_step == "gap_fill"

    def test_no_gap_fill_round_left_is_named(self) -> None:
        audit = AuditResult(must_pass_gaps=["R1"], verdicts=[_verdict("R1", "covered", False)])
        route = _policy().after_audit(_snap(gap_fill_rounds_left=0), audit, {})
        assert (route.next_step, route.because, route.cause) == (
            "finalize",
            "gap_fill_exhausted",
            "gap_fill_rounds",
        )

    def test_a_writing_fault_goes_to_a_revision(self) -> None:
        audit = AuditResult(verdicts=[_verdict("R2", WRITING, False)])
        assert _policy().after_audit(_snap(), audit, {}).next_step == "revise"

    def test_a_revision_that_gained_nothing_ends_it(self) -> None:
        audit = AuditResult(verdicts=[_verdict("R2", WRITING, False)])
        state = {"revision": 1, "audit_scores": [0.6, 0.6]}
        assert _policy().after_audit(_snap(), audit, state).because == "revision_plateau"

    def test_a_cut_off_clock_ends_it_and_names_the_counter(self) -> None:
        audit = AuditResult(verdicts=[_verdict("R2", WRITING, False)])
        route = _policy().after_audit(_snap(exhausted=("seconds",)), audit, {})
        assert (route.because, route.cause) == ("budget_spent", "seconds")


class TestHowItEnds:
    def test_a_passing_uncut_run_is_complete(self) -> None:
        status = _policy().final_status(
            _snap(), failed=False, audit_error=False, passed=True, weaker=False
        )
        assert status == "COMPLETE"

    @pytest.mark.parametrize(
        ("snap", "audit_error", "passed", "weaker"),
        [
            (_snap(exhausted=("tokens",)), False, True, False),
            (_snap(), True, True, False),
            (_snap(), False, False, False),
            (_snap(), False, True, True),
        ],
    )
    def test_anything_less_is_partial(
        self, snap: LoopSnapshot, audit_error: bool, passed: bool, weaker: bool
    ) -> None:
        status = _policy().final_status(
            snap, failed=False, audit_error=audit_error, passed=passed, weaker=weaker
        )
        assert status == "PARTIAL"

    def test_a_refused_search_alone_does_not_make_it_partial(self) -> None:
        status = _policy().final_status(
            _snap(exhausted=("searches",)),
            failed=False,
            audit_error=False,
            passed=True,
            weaker=False,
        )
        assert status == "COMPLETE"
