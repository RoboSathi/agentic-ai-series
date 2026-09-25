"""Context failure detection (E17, LLD §7).

Five distinct failure modes, which matter because they have different
fixes:

  rot         — quality decays as the window grows, with nothing wrong in it
  distraction — irrelevant material pulls the answer off task
  confusion   — too many similar options; the model cannot choose
  clash       — the window contains contradictory facts
  poisoning   — a false statement in context is treated as established

Treating all five as "the context is too long" leads to trimming a window
whose problem was that two chunks disagreed.
"""

from agentic_commons.config import get_settings
from agentic_commons.context.types import (
    AssembledContext,
    Chunk,
    ContextFailureMode,
    ContextHealth,
)
from agentic_commons.logging import get_logger

# The chunk kinds that state facts about the world, as opposed to
# recounting what was said about it. Only these can contradict.
_ASSERTING_KINDS = frozenset({"policy", "retrieved", "tool_result"})


class ContextFailureDetector:
    """Watches for the five modes, and for degradation across checkpoints.

    Checkpoints come from config (10k/30k/60k/100k/150k by default)
    because the curve is model-specific: the point of measuring it is to
    find *this* model's degradation, not to assume a published one.
    """

    def __init__(self) -> None:
        settings = get_settings().context.failure_detection
        self._checkpoints = settings.checkpoints_tokens
        self._degradation_threshold = settings.degradation_threshold
        self._logger = get_logger("context")

    @property
    def checkpoints(self) -> list[int]:
        """Token counts at which synthesis quality is re-measured."""
        return list(self._checkpoints)

    def assess(self, context: AssembledContext, task_terms: set[str]) -> list[ContextFailureMode]:
        """Which failure modes this window currently shows."""
        modes: list[ContextFailureMode] = []
        if self.distraction_ratio(context, task_terms) > 0.5:
            modes.append("distraction")
        if self.near_duplicate_ratio(context) > 0.5:
            modes.append("confusion")
        if self.contradictions(context):
            modes.append("clash")
        if context.used_tokens > max(self._checkpoints, default=0):
            modes.append("rot")
        return modes

    def distraction_ratio(self, context: AssembledContext, task_terms: set[str]) -> float:
        """Share of unpinned chunks sharing no term with the task."""
        candidates = [chunk for chunk in context.chunks if not chunk.pinned]
        if not candidates or not task_terms:
            return 0.0
        off_task = sum(
            not (task_terms & set(chunk.content.casefold().split())) for chunk in candidates
        )
        return off_task / len(candidates)

    def near_duplicate_ratio(self, context: AssembledContext) -> float:
        """Share of chunks that are near-copies of another chunk.

        Near-duplicates are the confusion signal: several plausible,
        almost-identical options with nothing to choose between them.
        """
        chunks = context.chunks
        if len(chunks) < 2:
            return 0.0
        duplicated = sum(
            any(
                index != other and is_near_duplicate(chunk.content, chunks[other].content)
                for other in range(len(chunks))
            )
            for index, chunk in enumerate(chunks)
        )
        return duplicated / len(chunks)

    def contradictions(self, context: AssembledContext) -> list[tuple[str, str]]:
        """Chunk pairs that state different values for the same key.

        Detected structurally over `key: value` lines rather than
        semantically — cheap, deterministic, and it catches the case that
        actually occurs: a stale retrieved document disagreeing with a
        fresh tool result.

        Only chunks that *assert* things are read. A transcript does not:
        its lines begin `user:` and `assistant:`, and reading those as
        claims makes every conversation of more than one turn contradict
        itself about what "user" is — a warning that fired on every
        session and therefore meant nothing. The same goes for recalled
        memory, which restates what was already said rather than
        asserting it afresh.
        """
        claims: dict[str, list[tuple[str, str]]] = {}
        for chunk in context.chunks:
            if chunk.kind not in _ASSERTING_KINDS:
                continue
            for key, value in _claims(chunk):
                claims.setdefault(key, []).append((value, chunk.source or chunk.kind))
        clashes: list[tuple[str, str]] = []
        for key, values in claims.items():
            distinct = {value for value, _ in values}
            if len(distinct) > 1:
                clashes.append((key, " vs ".join(sorted(distinct))))
        return clashes

    def health_at(
        self, checkpoint_tokens: int, observed_score: float, baseline_score: float
    ) -> ContextHealth:
        """Compare measured synthesis quality against the short-window baseline."""
        drop = (baseline_score - observed_score) / baseline_score if baseline_score else 0.0
        degraded = drop > self._degradation_threshold
        if degraded:
            self._logger.warning(
                "Synthesis quality degraded at checkpoint",
                tokens=checkpoint_tokens,
                drop=round(drop, 3),
            )
        return ContextHealth(
            checkpoint_tokens=checkpoint_tokens,
            observed_score=observed_score,
            baseline_score=baseline_score,
            degraded=degraded,
            modes=["rot"] if degraded else [],
            detail=f"{drop:.1%} below baseline",
        )

    def degradation_curve(self, scores: dict[int, float]) -> list[ContextHealth]:
        """The full curve against the shortest checkpoint as baseline."""
        if not scores:
            return []
        baseline = scores[min(scores)]
        return [self.health_at(tokens, score, baseline) for tokens, score in sorted(scores.items())]


def _claims(chunk: Chunk) -> list[tuple[str, str]]:
    """`key: value` claims stated in a chunk."""
    found: list[tuple[str, str]] = []
    for line in chunk.content.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() and value.strip():
            found.append((key.strip().casefold(), value.strip().casefold()))
    return found


def is_near_duplicate(left: str, right: str) -> bool:
    """Whether two chunks are near-copies, by token overlap.

    Public because two components need exactly this rule and must agree
    on it: the pipeline drops a near-duplicate before spending budget on
    it, and the detector reports what is left. Two thresholds would mean
    the window could be deduplicated and still flagged as confused.
    """
    left_tokens, right_tokens = set(left.casefold().split()), set(right.casefold().split())
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) > 0.8
