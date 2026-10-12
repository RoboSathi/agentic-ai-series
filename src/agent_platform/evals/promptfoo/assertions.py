"""Building blocks for an agent's Promptfoo `python` assertions.

Promptfoo calls an assertion with the reply (`output`) and a context whose
`providerResponse.metadata` carries the case and the full transcript the
agent-under-test produced (`agent_provider`). These read them back, and run
one of the agent's oracle rules as an assertion — so an attack is held to
exactly the invariants the golden set is, defined once.
"""

from typing import Any

from agent_platform.evals.core.contracts import GoldenCase, Transcript
from agent_platform.evals.oracle import InvariantOracle

_PASS, _FAIL = 1.0, 0.0
_NOT_APPLICABLE = "not applicable to this attack"


def case_of(context: dict[str, Any]) -> GoldenCase:
    """The attack, as the case the agent ran."""
    return GoldenCase.model_validate(_metadata(context)["case"])


def transcript_of(context: dict[str, Any]) -> Transcript:
    """Everything the agent did for the attack."""
    return Transcript.model_validate(_metadata(context)["transcript"])


def oracle_assert(
    oracle: InvariantOracle[Any], metric: str, context: dict[str, Any]
) -> dict[str, Any]:
    """One oracle rule, as a Promptfoo grading result."""
    verdict = oracle.check(metric, case_of(context), transcript_of(context))
    if verdict is None:
        return result(True, _NOT_APPLICABLE)
    return result(verdict.passed, verdict.rationale)


def result(passed: bool, reason: str) -> dict[str, Any]:
    """A Promptfoo grading result."""
    return {"pass": passed, "score": _PASS if passed else _FAIL, "reason": reason}


def _metadata(context: dict[str, Any]) -> dict[str, Any]:
    return dict((context.get("providerResponse") or {}).get("metadata") or {})
