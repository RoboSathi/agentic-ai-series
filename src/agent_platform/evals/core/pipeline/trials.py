"""Running each case more than once: pass@k and pass^k.

An agent is not deterministic, so one pass is a sample, not a property.
With `evals.trials` = k, every case runs k times as `<case><sep><n>`; each
trial is its own case (its own hash, its own cached transcript), so the
rest of the pipeline needs no idea that trials exist. The report folds
them back: pass@k (some trial passed — the agent *can*) and pass^k (every
trial passed — the agent reliably *does*).
"""

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import GoldenCase

_TRIAL_KEY = "trial"
_FIRST = 1


def expand_trials(cases: list[GoldenCase], trials: int | None = None) -> list[GoldenCase]:
    """Each case k times; with k = 1, the cases unchanged."""
    k = trials if trials is not None else get_settings().evals.trials
    if k <= _FIRST:
        return cases
    separator = get_settings().evals.trial_separator
    return [
        case.model_copy(
            update={
                "case_id": f"{case.case_id}{separator}{n}",
                "payload": {**case.payload, _TRIAL_KEY: n},
            }
        )
        for case in cases
        for n in range(_FIRST, k + _FIRST)
    ]


def base_case_id(case_id: str) -> str:
    """The case a trial belongs to (`refund-001#t2` → `refund-001`)."""
    return case_id.split(get_settings().evals.trial_separator, 1)[0]
