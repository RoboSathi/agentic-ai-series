"""Is the loop going round in circles? Four ways of asking, cheapest first.

  exact      the same action with byte-identical arguments          (`action_fingerprint`)
  canonical  the same action and arguments once written the same    (`canonical_fingerprint`)
             way: key order, case, whitespace, `5` vs `5.0` vs "5"
  semantic   an action and its result this similar to one that      (`SemanticTier`)
             already failed — the model retrying its own failed idea
             in new words
  cycle      the last few actions repeating as a block: A→B→A→B     (`CycleDetector`)

And one question about the turn as a whole: is it far longer than this agent's turns usually
are (`HistoricalBaseline`)? That is a flag, never a stop — the budgets stop.
"""

import json
import re
from statistics import fmean, pstdev
from typing import Any, Protocol, runtime_checkable

from agent_platform.context.similarity import TextSimilarity
from agent_platform.logging import get_logger
from agent_platform.tools.types import ToolCall

_SPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")


def action_fingerprint(call: ToolCall) -> str:
    """Stable identity for an exact same action + argument set."""
    arguments = ",".join(f"{key}={call.params[key]}" for key in sorted(call.params))
    return f"{call.action}({arguments})"


def canonical_fingerprint(call: ToolCall) -> str:
    """The action and its arguments with every inessential difference removed."""
    return f"{call.action.casefold()}({json.dumps(_canonical(call.params), sort_keys=True)})"


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k).casefold(): _canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = _SPACE.sub(" ", str(value)).strip().casefold()
    return float(text) if _NUMBER.match(text) else text


def describe(call: ToolCall) -> str:
    """An action as text, for similarity: its name and its arguments."""
    return f"{call.action} {json.dumps(call.params, sort_keys=True, default=str)}"


class SemanticTier:
    """Whether an action is, in other words, one that already failed."""

    def __init__(self, similarity: TextSimilarity, threshold: float) -> None:
        self._similarity = similarity
        self._threshold = threshold

    async def closest(self, text: str, failed: list[str]) -> tuple[float, int]:
        """The highest similarity to a failed step, and which one (-1: none)."""
        if not failed:
            return 0.0, -1
        row = (await self._similarity.matrix([text], failed))[0]
        best = max(range(len(row)), key=row.__getitem__)
        return row[best], best

    def repeats(self, score: float) -> bool:
        return score >= self._threshold


class CycleDetector:
    """A block of the last `period` actions repeated `repeats` times in a row."""

    def __init__(self, max_period: int, repeats: int) -> None:
        self._max_period = max_period
        self._repeats = repeats

    def period(self, actions: list[str]) -> int | None:
        """The shortest repeating period at the end of `actions`, or None."""
        for period in range(1, self._max_period + 1):
            span = period * self._repeats
            if len(actions) < span:
                break
            tail = actions[-span:]
            block = tail[:period]
            if all(tail[i] == block[i % period] for i in range(span)):
                return period
        return None


@runtime_checkable
class StepHistory(Protocol):
    """How many steps this agent's recent completed turns took."""

    def recent_steps(self, limit: int) -> list[int]: ...


class FactStepHistory:
    """Step counts from the analytics facts (`fact_turn.steps`) of one agent."""

    def __init__(self, store: Any, agent_id: str) -> None:
        self._store = store
        self._agent_id = agent_id
        self._logger = get_logger("loop_control")

    def recent_steps(self, limit: int) -> list[int]:
        try:
            rows = self._store.iter_rows(
                "SELECT steps FROM fact_turn WHERE agent_id = ? AND steps > 0 "
                "ORDER BY recorded_at DESC LIMIT ?",
                (self._agent_id, limit),
            )
            return [int(row["steps"]) for row in rows]
        except Exception as exc:  # noqa: BLE001 - no history yet is not an error
            self._logger.debug("No step history for the baseline", error=str(exc))
            return []


class HistoricalBaseline:
    """How unusual a turn's length is, against the agent's own recent turns."""

    def __init__(self, history: StepHistory, min_samples: int, sample_size: int) -> None:
        self._history = history
        self._min_samples = min_samples
        self._sample_size = sample_size
        self._samples: list[int] | None = None

    def z_score(self, steps: int) -> float | None:
        """Standard deviations above the mean, or None with too little history to say."""
        if self._samples is None:
            self._samples = self._history.recent_steps(self._sample_size)
        if len(self._samples) < self._min_samples:
            return None
        spread = pstdev(self._samples)
        return (steps - fmean(self._samples)) / spread if spread else None


__all__ = [
    "CycleDetector",
    "FactStepHistory",
    "HistoricalBaseline",
    "SemanticTier",
    "StepHistory",
    "action_fingerprint",
    "canonical_fingerprint",
    "describe",
]
