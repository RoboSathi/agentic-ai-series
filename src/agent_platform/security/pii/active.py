"""The PII filter the running posture declares, for a surface that is not a composition root.

A review queue, a console screen and the reply a person approved are built far from any
composition root, yet each must follow whether `pii` is on *now* — `/harness` moves it hot.
So they ask here, per call, rather than being told once: with `pii` off these return the
text untouched, and no detector is ever built.
"""

import functools

from agent_platform.config import active_features
from agent_platform.security.pii.detector import PIIFilterImpl


@functools.cache
def _filter() -> PIIFilterImpl:
    return PIIFilterImpl()


def _active() -> PIIFilterImpl | None:
    return _filter() if active_features().is_on("pii") else None


def redact_if_on(text: str) -> str:
    """`text` with its personal data tokenised, when the posture declares `pii`."""
    active = _active()
    return active.redact(text) if active is not None else text


def restore_if_on(text: str) -> str:
    """`text` with its tokens put back to their values: only for what a person approved."""
    active = _active()
    return active.restore(text) if active is not None else text
