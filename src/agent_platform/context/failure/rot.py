"""Rot: answers get worse as the window grows, with nothing wrong in any one chunk.

Nothing in a single window shows it. It shows across a session: each answer is scored (the
logprob confidence and grounding support the hallucination check measures) together with how
big the window was. The first observations are the session's own baseline; once the window is
past a checkpoint (a fraction of the budget), the recent scores are compared with it. A drop of
`degradation_threshold` or more is rot, and the mitigation is the cheap one that works: the
transcript is reduced before the next call.
"""

from collections import deque
from statistics import fmean

from agent_platform.config import get_settings
from agent_platform.context.failure.common import finding
from agent_platform.context.types import AssembledContext, ContextFinding, ContextHealth


def support_rate(claims: int, supported: int) -> float | None:
    """An answer's quality as the grounding check measured it: the share of its claims the
    evidence supports (None for an answer that made no checkable claim)."""
    return supported / claims if claims else None


class RotMonitor:
    """Per-session (window size, answer quality) observations, and the verdict they support."""

    def __init__(self) -> None:
        self._app_config = get_settings()
        self._observed: dict[str, deque[tuple[int, float]]] = {}
        self._last_tokens: dict[str, int] = {}

    def window_used(self, session_id: str, tokens: int) -> None:
        """How big the window just handed to the model was (the next score belongs to it)."""
        self._last_tokens[session_id] = tokens

    def record(self, session_id: str, score: float) -> None:
        """One answer's quality in [0, 1], against the last window's size."""
        limit = self._app_config.context.failure_detection.max_observations
        series = self._observed.setdefault(session_id, deque(maxlen=limit))
        series.append((self._last_tokens.get(session_id, 0), score))

    def health(self, session_id: str, used_tokens: int, budget: int) -> ContextHealth | None:
        """The session's health at the highest checkpoint this window has crossed."""
        settings = self._app_config.context.failure_detection
        crossed = [
            int(f * budget) for f in settings.checkpoint_fractions if used_tokens >= f * budget
        ]
        series = list(self._observed.get(session_id, ()))
        if not crossed or len(series) < 2 * settings.min_observations:
            return None
        baseline = fmean(score for _, score in series[: settings.min_observations])
        observed = fmean(score for _, score in series[-settings.min_observations :])
        degraded = (
            baseline > 0 and (baseline - observed) / baseline >= settings.degradation_threshold
        )
        return ContextHealth(
            checkpoint_tokens=max(crossed),
            observed_score=observed,
            baseline_score=baseline,
            degraded=degraded,
            modes=["rot"] if degraded else [],
        )

    def findings(self, context: AssembledContext, session_id: str) -> list[ContextFinding]:
        health = self.health(session_id, context.used_tokens, context.budget_tokens)
        if health is None or not health.degraded:
            return []
        return [
            finding("rot", "rot", source="window", action="reduced",
                    observed=health.observed_score, baseline=health.baseline_score,
                    checkpoint=health.checkpoint_tokens)
        ]  # fmt: skip
