"""Deterministic watchdog implementations for agent trajectories."""

from agentic_commons.control_loop.types import ControlCheck, ControlLoopReport
from agentic_commons.logging import get_logger
from agentic_commons.tools.types import ToolCall


def action_fingerprint(call: ToolCall) -> str:
    """Stable identity for an exact same action + argument set."""
    arguments = ",".join(f"{key}={call.params[key]}" for key in sorted(call.params))
    return f"{call.action}({arguments})"


class DuplicateActionDetector:
    """Exact-match duplicate detector for one agent trajectory."""

    def __init__(self) -> None:
        self._seen: dict[str, str] = {}
        self._logger = get_logger("control_loop")

    def seen_before(self, call: ToolCall) -> str | None:
        """Return the previous result for the exact same call, if present."""
        return self._seen.get(action_fingerprint(call))

    def record(self, call: ToolCall, output: str) -> None:
        """Remember a successfully completed call."""
        self._seen[action_fingerprint(call)] = output

    def reset(self) -> None:
        """Clear the detector for a new run."""
        self._seen.clear()

    @property
    def call_count(self) -> int:
        """Number of distinct successful calls recorded in this run."""
        return len(self._seen)

    def reject(self, call: ToolCall) -> str:
        """Explain the suppression and provide the previous result."""
        previous = self.seen_before(call) or ""
        self._logger.warning(
            "Suppressed a repeated tool call",
            action=call.action,
            fingerprint=action_fingerprint(call),
        )
        return (
            f"You already called this and it returned: {previous} "
            "Use that result rather than calling it again."
        )


class ToolControlLoop:
    """Deterministic trajectory boundary used immediately before execution."""

    def __init__(self, duplicate_detector: DuplicateActionDetector | None = None) -> None:
        self._duplicates = duplicate_detector or DuplicateActionDetector()
        self._logger = get_logger("control_loop")

    def evaluate_round(self, round_number: int, max_rounds: int) -> ControlLoopReport:
        """Check whether another tool round is legal."""
        report = self._round_report(round_number, max_rounds)
        self._log(report)
        return report

    def _round_report(self, round_number: int, max_rounds: int) -> ControlLoopReport:
        passed = round_number <= max_rounds
        check = ControlCheck(
            check_id="control_loop.max_tool_iterations",
            name="Round limit",
            status="PASS" if passed else "FAIL",
            reason_code="ROUND_WITHIN_LIMIT" if passed else "MAX_TOOL_ITERATIONS",
            reason=(
                f"Round {round_number} is within the configured limit of {max_rounds}."
                if passed
                else f"Round {round_number} would exceed the configured limit of {max_rounds}."
            ),
            observed={"current_round": round_number, "max_rounds": max_rounds},
        )
        return ControlLoopReport(
            decision="continue" if passed else "abort",
            reason_code="CONTROL_LOOP_CONTINUE" if passed else "MAX_TOOL_ITERATIONS",
            reason=check.reason,
            checks=[check],
            round_number=round_number,
            max_rounds=max_rounds,
        )

    def before_action(
        self,
        call: ToolCall,
        *,
        round_number: int,
        max_rounds: int,
    ) -> ControlLoopReport:
        """Evaluate active loop guards before one tool action."""
        round_report = self._round_report(round_number, max_rounds)
        if round_report.decision == "abort":
            report = round_report.model_copy()
            report.fingerprint = action_fingerprint(call)
            self._log(report)
            return report

        fingerprint = action_fingerprint(call)
        previous = self._duplicates.seen_before(call)
        duplicate_passed = previous is None
        duplicate_check = ControlCheck(
            check_id="control_loop.duplicate_action",
            name="Duplicate action",
            status="PASS" if duplicate_passed else "FAIL",
            reason_code="NO_DUPLICATE" if duplicate_passed else "DUPLICATE_ACTION",
            reason=(
                "No previous identical action exists in this trajectory."
                if duplicate_passed
                else "The exact action and arguments already succeeded earlier in this trajectory."
            ),
            observed={"action": call.action, "fingerprint": fingerprint},
            evidence={"previous_result": previous or ""},
        )
        report = ControlLoopReport(
            decision="continue" if duplicate_passed else "abort",
            reason_code="CONTROL_LOOP_CONTINUE" if duplicate_passed else "DUPLICATE_ACTION",
            reason=duplicate_check.reason,
            checks=[*round_report.checks, duplicate_check],
            round_number=round_number,
            max_rounds=max_rounds,
            fingerprint=fingerprint,
            previous_result=previous or "",
        )
        self._log(report)
        return report

    def record_success(self, call: ToolCall, output: str) -> None:
        """Remember a successfully completed call for this trajectory."""
        self._duplicates.record(call, output)

    def reject_message(self, call: ToolCall) -> str:
        """Return the previous result to the model when a duplicate is suppressed."""
        return self._duplicates.reject(call)

    def _log(self, report: ControlLoopReport) -> None:
        self._logger.info(
            "Control loop evaluated",
            decision=report.decision,
            reason_code=report.reason_code,
            reason=report.reason,
            round_number=report.round_number,
            max_rounds=report.max_rounds,
            fingerprint=report.fingerprint,
            previous_result=report.previous_result,
            checks=[check.model_dump(mode="json") for check in report.checks],
        )
