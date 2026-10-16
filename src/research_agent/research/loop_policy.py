"""When a research run stops, where it goes after the audit, and how it ends — one rule each.

The run graph's nodes report LOOP CONTROL and its edges route; both ask here, so the card can
never say one thing while the edge does another. The policy reads no live resource: the node
hands it a `LoopSnapshot` of what the run has spent and seen, and gets back a decision. That
makes every branch a plain function of its inputs, testable without a graph.

    after reflection   stop?  why  ·  should the model judge the evidence first  ·  the cap
    after the audit    gap_fill (missing evidence) · revise (a writing fault) · finalize
    at the end         COMPLETE · PARTIAL · CANCELLED · FAILED

The platform `LoopGovernor` governs the run over the same budget (a round is its step). Its
verdict rides in the snapshot: after the budgets have had their say, a governor that says
finalize (the rounds stopped adding new evidence) stops the run, and the run ends in the
governor's one terminal outcome (`TERMINAL_OUTCOME`).
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agent_platform.loop_control import LoopVerdict, TerminalOutcome
from research_agent.audit.independent import AuditResult
from research_agent.domain.budget import GAP_FILL_ROUNDS, ROUNDS
from research_agent.domain.report import RunStatus
from research_agent.presentation.narration import say
from research_agent.research.stop_rules.base import StopDecision

# The run graph's nodes, as LOOP CONTROL names the next step (`research_agent.run_loop.next`).
GAP_FILL, FINALIZE, REVISE = "gap_fill", "finalize", "revise"
# Why (`research_agent.run_loop.because`).
CANCELLED, NO_GAPS, GAPS_TO_FILL = "cancelled", "no_gaps", "gaps_to_fill"
GAP_FILL_EXHAUSTED, BUDGET_SPENT, REFLECTION_OFF = (
    "gap_fill_exhausted",
    "budget_spent",
    "reflection_off",
)
WRITING_FAULTS, REVISIONS_SPENT, REVISION_PLATEAU = (
    "writing_faults",
    "revisions_spent",
    "revision_plateau",
)

# Counters whose exhaustion cuts work short by itself: model calls and the clock stop mid-way,
# so whatever was being written is unfinished. A refused search or page read (recorded in
# `trace.budget_stops`) is not: the report says what was not read, and whether the contract was
# still met is the audit's to judge. A run whose every must-pass criterion holds is COMPLETE
# even if it could have read more.
TRUNCATING_COUNTERS = frozenset({"tokens", "seconds"})

# How each run status ends the governed loop.
TERMINAL_OUTCOME: dict[RunStatus, TerminalOutcome] = {
    "COMPLETE": "answered",
    "PARTIAL": "finalized_on_cap",
    "CANCELLED": "cancelled",
    "FAILED": "failed",
}


def truncated_by_budget(exhausted: Collection[str]) -> bool:
    """Whether a budget limit cut the run's work short, as opposed to merely bounding it."""
    return bool(TRUNCATING_COUNTERS & set(exhausted))


@dataclass(frozen=True)
class LoopSnapshot:
    """What the run has spent and seen, at the moment a decision is asked for."""

    cancelled: bool
    # Budget counters at their cap, and the searches or reads the budget refused.
    exhausted: tuple[str, ...]
    budget_stops: tuple[str, ...]
    rounds_left: float
    gap_fill_rounds_left: float
    # A stop reason when every rejection was unreadable and nothing was admitted; "" otherwise.
    unreadable_only: str = ""
    # The loop governor's verdict on taking another round (None: not asked).
    governed: LoopVerdict | None = None


@dataclass(frozen=True)
class StopCall:
    """After reflection: stop or not, why, whether to judge the evidence first, and the cap."""

    stop: bool
    reason: str
    judge: bool = False
    cause: str | None = None
    # The governor's verdict, when the governor is what stopped the run.
    verdict: LoopVerdict | None = None


@dataclass(frozen=True)
class Route:
    """After the audit: the next node, why, and the counter that forced it (if one did)."""

    next_step: str
    because: str
    cause: str | None = None


class RunLoopPolicy:
    """The run's three decisions, from a snapshot and the run's own settings."""

    def __init__(
        self,
        *,
        reflection_enabled: bool,
        writing_checks: frozenset[str],
        max_revisions: int,
        min_score_gain: float,
    ) -> None:
        self._reflection = reflection_enabled
        self._writing = writing_checks
        self._max_revisions = max_revisions
        self._min_gain = min_score_gain

    def after_reflect(self, snap: LoopSnapshot, rule: StopDecision) -> StopCall:
        """Stop after this round? The model judges only when the run ends on its own terms —
        the stop rule says answered, or the rounds or budget ran out — never when the operator
        stopped it, no page could be read, or reflection is withheld."""
        if snap.cancelled:
            return StopCall(True, say("stop.cancelled"))
        if snap.unreadable_only:
            return StopCall(True, snap.unreadable_only)
        if not self._reflection:
            return StopCall(True, rule.reason if rule.stop else say("stop.reflection_withheld"))
        if rule.stop:
            # Answered is answered, even if a cap was reached getting there.
            return StopCall(True, rule.reason, judge=True)
        return self._capped(snap) or self._governed(snap) or StopCall(False, rule.reason)

    def after_audit(
        self, snap: LoopSnapshot, audit: AuditResult | None, state: Mapping[str, Any]
    ) -> Route:
        """Gap-fill, revise or finish. Evidence first: a report revised before its evidence is
        in would be revised again."""
        if audit is None:
            return Route(FINALIZE, NO_GAPS)
        if snap.cancelled:
            return Route(FINALIZE, CANCELLED)
        evidence = [
            g for g in audit.must_pass_gaps if self._check_of(audit, g) not in self._writing
        ]
        blocked = self._gap_fill_blocked(snap) if evidence else NO_GAPS
        if evidence and not blocked:
            return Route(GAP_FILL, GAPS_TO_FILL)
        if self.writing_faults(audit):
            revise = self._revise(snap, state)
            if revise.next_step == REVISE or not evidence:
                return revise
        return Route(FINALIZE, blocked, self._budget_cause(snap, blocked))

    def final_status(
        self, snap: LoopSnapshot, *, failed: bool, audit_error: bool, passed: bool, weaker: bool
    ) -> RunStatus:
        """PARTIAL when the score fell short, a must-pass criterion is unmet, the model or the
        clock was cut off, nothing verified the report, or the evidence is weaker than asked."""
        if snap.cancelled:
            return "CANCELLED"
        if failed:
            return "FAILED"
        if audit_error or truncated_by_budget(snap.exhausted) or weaker or not passed:
            return "PARTIAL"
        return "COMPLETE"

    def writing_faults(self, audit: AuditResult) -> list[Any]:
        """The failed verdicts a revision can fix: faults in how the report is written."""
        return [
            v for v in audit.verdicts if not v.passed and v.applicable and v.name in self._writing
        ]

    def _capped(self, snap: LoopSnapshot) -> StopCall | None:
        # Rounds used up is the plan running its course, not a limit hit mid-way.
        exhausted = [c for c in snap.exhausted if c != ROUNDS]
        if exhausted or snap.budget_stops:
            spent: Sequence[str] = exhausted or snap.budget_stops
            return StopCall(True, say("stop.budget", counters=", ".join(spent)), True, spent[0])
        if snap.rounds_left <= 0:
            # The rounds ran out before the stop rule was met: that is the cap deciding.
            return StopCall(True, say("stop.max_rounds"), True, ROUNDS)
        return None

    @staticmethod
    def _governed(snap: LoopSnapshot) -> StopCall | None:
        """The governor's finalize, as a stop no critique can overrule: the rounds have stopped
        adding new evidence, so another one would not either."""
        verdict = snap.governed
        if verdict is None or verdict.decision != "finalize":
            return None
        return StopCall(True, verdict.reason, verdict=verdict)

    def _gap_fill_blocked(self, snap: LoopSnapshot) -> str:
        """Why a gap-fill cannot run, or "" when it can."""
        if not self._reflection:
            return REFLECTION_OFF
        if snap.gap_fill_rounds_left <= 0:
            return GAP_FILL_EXHAUSTED
        return BUDGET_SPENT if snap.exhausted else ""

    def _revise(self, snap: LoopSnapshot, state: Mapping[str, Any]) -> Route:
        """Revise, while revisions are left and the last one raised the audit's score."""
        revision, scores = state.get("revision", 0), state.get("audit_scores", [])
        if not self._reflection:
            return Route(FINALIZE, REFLECTION_OFF)
        if revision >= self._max_revisions:
            return Route(FINALIZE, REVISIONS_SPENT)
        if revision and len(scores) > 1 and scores[-1] - scores[-2] < self._min_gain:
            return Route(FINALIZE, REVISION_PLATEAU)
        if truncated_by_budget(snap.exhausted):
            return Route(FINALIZE, BUDGET_SPENT, self._budget_cause(snap, BUDGET_SPENT))
        return Route(REVISE, WRITING_FAULTS)

    @staticmethod
    def _budget_cause(snap: LoopSnapshot, because: str) -> str | None:
        """The counter behind a budget-decided finish; None for any other."""
        if because == GAP_FILL_EXHAUSTED:
            return GAP_FILL_ROUNDS
        if because == BUDGET_SPENT:
            return next(iter(snap.exhausted), None)
        return None

    @staticmethod
    def _check_of(audit: AuditResult, item_id: str) -> str:
        return next((v.name for v in audit.verdicts if v.item_id == item_id), "")


__all__ = [
    "LoopSnapshot",
    "Route",
    "RunLoopPolicy",
    "StopCall",
    "TERMINAL_OUTCOME",
    "TRUNCATING_COUNTERS",
    "truncated_by_budget",
]
