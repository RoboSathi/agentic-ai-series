"""Reading the world back after a side effect (LLD §24).

An executor returning success means the call did not raise. It does not
mean the money moved, and those are different claims. Until this package
existed nothing checked the difference: `set_refund_status` reported
`affected > 0` and every layer above took that as proof, so a refund was
"issued" on the strength of a row count.

The check is deliberately dull. Re-read the thing that should have
changed, compare it against what the action said it would be, and refuse
to call the result authoritative if they disagree. A verification step
that reasons about the discrepancy is a verification step that can be
argued out of noticing one.

The platform owns the comparison because comparing values is generic.
It does not own `intended` or `observed`: what a call was meant to do,
and how to read the thing back, are the agent's — this package never
learns what an order is.
"""

from typing import Any, Protocol, runtime_checkable

from agent_platform.logging import get_logger, instrument
from agent_platform.tools.types import ToolCall, ToolResult
from agent_platform.verification.types import IntendedEffect, VerificationReport

# Floats that differ by less than this are the same amount of money.
_MONEY_TOLERANCE = 0.005


@runtime_checkable
class EffectSource(Protocol):
    """Says what a call meant to do, and reads back what happened.

    Two methods rather than one because they run at different times:
    `intended` is asked immediately after execution, while the result is
    in hand, and `observed` goes back to the system of record.
    """

    def intended(self, call: ToolCall, result: ToolResult) -> IntendedEffect | None:
        """The change this call was supposed to make, or None if it makes none.

        A read returns None. So does a call that failed: there is no
        claimed effect to verify, only an error to report.
        """
        ...

    def observed(self, effect: IntendedEffect) -> dict[str, Any]:
        """The current state of whatever `effect.target` names."""
        ...


@runtime_checkable
class EffectVerifier(Protocol):
    """Confirms that a completed call actually changed the world."""

    async def verify(self, call: ToolCall, result: ToolResult) -> VerificationReport | None: ...


@instrument("verification")
class ReadBackVerifier:
    """Verifies an effect by re-reading it from the system of record."""

    def __init__(self, source: EffectSource) -> None:
        self._source = source
        self._logger = get_logger("verification")

    async def verify(self, call: ToolCall, result: ToolResult) -> VerificationReport | None:
        """Compare what the call claimed against what the record now says.

        `None` means there was nothing to verify — a read, or a call that
        already reported failure. That is not the same as a passed check,
        and callers must not treat it as one.
        """
        effect = self._source.intended(call, result)
        if effect is None:
            return None
        try:
            observed = self._source.observed(effect)
        except Exception as exc:
            # An unreadable record is an unverified effect, never a
            # verified one. Failing open here would make the check
            # decorative exactly when the store is in trouble.
            return self._unreadable(effect, exc)
        mismatches = _mismatches(effect.expected, observed)
        report = VerificationReport(
            verified=not mismatches,
            effect=effect,
            observed=observed,
            mismatches=mismatches,
            reason_code="EFFECT_VERIFIED" if not mismatches else "EFFECT_NOT_VERIFIED",
            reason=(
                "The record matches what the action reported."
                if not mismatches
                else "The record does not match what the action reported."
            ),
        )
        if not report.verified:
            self._logger.error(
                "A completed action did not take effect",
                action=effect.action,
                target=effect.target,
                expected=effect.expected,
                observed=observed,
                mismatches=mismatches,
            )
        return report

    def _unreadable(self, effect: IntendedEffect, exc: Exception) -> VerificationReport:
        """A report for an effect whose target could not be read back."""
        self._logger.error(
            "Could not read back an action's effect",
            action=effect.action,
            target=effect.target,
            error=str(exc),
        )
        return VerificationReport(
            verified=False,
            effect=effect,
            mismatches=[f"could not read {effect.target}: {exc}"],
            reason_code="EFFECT_UNREADABLE",
            reason="The system of record could not be read to confirm the effect.",
        )


def _mismatches(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Every expected field the record disagrees with, in words."""
    problems: list[str] = []
    for field, wanted in expected.items():
        if field not in observed:
            problems.append(f"{field} is absent from the record")
            continue
        actual = observed[field]
        if not _same(wanted, actual):
            problems.append(f"{field} is {actual!r}, expected {wanted!r}")
    return problems


def _same(wanted: Any, actual: Any) -> bool:
    """Whether two recorded values are the same value.

    Money is compared with a tolerance because it travels through floats
    and SQLite: `49.99` written and read back is not always bit-identical,
    and a verification layer that flagged that would be noise nobody reads.
    """
    if isinstance(wanted, bool) or isinstance(actual, bool):
        return bool(wanted) is bool(actual)
    if isinstance(wanted, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(wanted) - float(actual)) < _MONEY_TOLERANCE
    return bool(wanted == actual)


__all__ = ["EffectSource", "EffectVerifier", "ReadBackVerifier"]
