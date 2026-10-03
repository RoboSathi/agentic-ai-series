"""Context failure detection — and mitigation (E17, LLD §7).

Five distinct failure modes, which matter because they have different
fixes:

  rot         — quality decays as the window grows, with nothing wrong in it
  distraction — irrelevant material pulls the answer off task
  confusion   — too many similar options; the model cannot choose
  clash       — the window contains contradictory facts
  poisoning   — a false or hostile statement in context is treated as established

Treating all five as "the context is too long" leads to trimming a window
whose problem was that two chunks disagreed.

This is the platform's own detector, deterministic and model-free: no
maintained library implements this taxonomy (guardrail libraries cover
only injection). It does two things, in order:

  `diagnose` — what is wrong, per chunk, with the rule that found it;
  `mitigate` — what to do before the model sees the window: a poisoned
               chunk is dropped, a clash keeps its higher-authority side,
               a distracting retrieved chunk is dropped. Rot and confusion
               are already handled by the reduce and dedupe strategies.
"""

import re
from typing import Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.context.types import (
    AssembledContext,
    Chunk,
    ChunkKind,
    ContextFailureMode,
    ContextFinding,
    ContextHealth,
    is_provenance_header,
)
from agent_platform.llm.embeddings import tokenize
from agent_platform.logging import get_logger

# The chunk kinds that state facts about the world, as opposed to
# recounting what was said about it. Only these can contradict.
_ASSERTING_KINDS = frozenset({"policy", "retrieved", "tool_result"})

# Kinds that carry data from outside the instruction boundary — where an
# injected instruction would be poisoning rather than a request.
_DATA_KINDS = frozenset({"retrieved", "tool_result", "memory", "history"})

# Who wins a clash. A tool result read from the system of record this turn
# outranks a retrieved copy; policy outranks both.
_AUTHORITY: dict[ChunkKind, int] = {
    "system": 5,
    "policy": 4,
    "tool_result": 3,
    "retrieved": 2,
    "memory": 1,
    "history": 0,
}

# What a claim is *about*: the first identifier-shaped token in its chunk.
# Two orders' lines never clash; unidentified claims share one subject.
_ENTITY = re.compile(r"\b(?=[\w-]*\d)[A-Za-z0-9][\w-]{3,}\b")

# The share of chunks at which a mode is declared.
_MODE_RATIO = 0.5
_DUPLICATE_OVERLAP = 0.8


@runtime_checkable
class InjectionScreen(Protocol):
    """Finds injection phrasing in text (the security layer's separator satisfies this)."""

    def find_phrases(self, text: str) -> list[str]: ...


class ContextFailureDetector:
    """Watches for the five modes, and for degradation across checkpoints.

    Checkpoints come from config (10k/30k/60k/100k/150k by default)
    because the curve is model-specific: the point of measuring it is to
    find *this* model's degradation, not to assume a published one.
    """

    def __init__(self, screen: InjectionScreen | None = None) -> None:
        settings = get_settings().context.failure_detection
        self._checkpoints = settings.checkpoints_tokens
        self._degradation_threshold = settings.degradation_threshold
        # Injected so `context` never imports `security`. Without one,
        # poisoning by injected instruction is not looked for — and the
        # finding list says nothing rather than claiming a clean window.
        self._screen = screen
        self._logger = get_logger("context")

    @property
    def checkpoints(self) -> list[int]:
        """Token counts at which synthesis quality is re-measured."""
        return list(self._checkpoints)

    # ── the per-turn entry points ────────────────────────────────────

    def diagnose(self, context: AssembledContext, task_terms: set[str]) -> list[ContextFinding]:
        """Every failure in this window, chunk by chunk, with the rule that found it."""
        findings = self._poisoned(context) + self._clashes(context)
        if self.distraction_ratio(context, task_terms) > _MODE_RATIO:
            findings += self._distracting(context, task_terms) or [
                ContextFinding(
                    mode="distraction",
                    source="window",
                    rule="no term shared with the task",
                    detail=f"over {_MODE_RATIO:.0%} of chunks are off the task",
                )
            ]
        if self.near_duplicate_ratio(context) > _MODE_RATIO:
            findings.append(
                ContextFinding(
                    mode="confusion",
                    source="window",
                    rule="near-duplicate chunks",
                    detail=f"over {_MODE_RATIO:.0%} of chunks are near-copies",
                )
            )
        if context.used_tokens > max(self._checkpoints, default=0):
            findings.append(
                ContextFinding(
                    mode="rot",
                    source="window",
                    rule="token checkpoint",
                    detail=f"{context.used_tokens} tokens exceeds the last checkpoint",
                )
            )
        return findings

    def mitigate(
        self, context: AssembledContext, findings: list[ContextFinding]
    ) -> tuple[AssembledContext, list[ContextFinding]]:
        """Drop what should not reach the model; say what was done."""
        removed = {
            f.index for f in findings if f.action.startswith("dropped") and f.index is not None
        }
        if not removed:
            return context, findings
        kept = [chunk for index, chunk in enumerate(context.chunks) if index not in removed]
        labels = [_label(context.chunks[index]) for index in sorted(removed)]
        self._logger.warning(
            "Context chunks removed before the model call",
            removed=labels,
            findings=[finding.model_dump() for finding in findings],
        )
        return (
            context.model_copy(
                update={
                    "chunks": kept,
                    "used_tokens": sum(chunk.token_estimate for chunk in kept),
                    "dropped": [*context.dropped, *labels],
                }
            ),
            findings,
        )

    def assess(self, context: AssembledContext, task_terms: set[str]) -> list[ContextFailureMode]:
        """Which failure modes this window currently shows."""
        return sorted({finding.mode for finding in self.diagnose(context, task_terms)})

    # ── the individual signals ───────────────────────────────────────

    def distraction_ratio(self, context: AssembledContext, task_terms: set[str]) -> float:
        """Share of unpinned chunks sharing no term with the task."""
        candidates = [chunk for chunk in context.chunks if not chunk.pinned]
        if not candidates or not task_terms:
            return 0.0
        off_task = sum(not _on_task(chunk, task_terms) for chunk in candidates)
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
        """Claim keys stated with different values, as `(key, "a vs b")`.

        Detected structurally over `key: value` lines — cheap, deterministic,
        and it catches the case that actually occurs: a stale retrieved
        document disagreeing with a fresh tool result. Provenance labels
        are not claims, and a claim is keyed by the entity it is about, so
        two different orders never "contradict".
        """
        return [
            (key, " vs ".join(sorted({value for value, _ in stated})))
            for key, stated in _claims_by_key(context).items()
            if len({value for value, _ in stated}) > 1
        ]

    # ── findings ─────────────────────────────────────────────────────

    def _poisoned(self, context: AssembledContext) -> list[ContextFinding]:
        """Data chunks carrying instructions — indirect injection."""
        if self._screen is None:
            return []
        findings = []
        for index, chunk in enumerate(context.chunks):
            phrases = self._screen.find_phrases(chunk.content) if chunk.kind in _DATA_KINDS else []
            if phrases:
                findings.append(
                    ContextFinding(
                        mode="poisoning",
                        source=_label(chunk),
                        rule="injection phrasing in data",
                        detail=f"matched {phrases}",
                        action="dropped",
                        index=index,
                    )
                )
        return findings

    def _clashes(self, context: AssembledContext) -> list[ContextFinding]:
        """Contradictions, resolved in favour of the more authoritative source."""
        findings = []
        for key, stated in _claims_by_key(context).items():
            if len({value for value, _ in stated}) < 2:
                continue
            findings += _resolve(key, stated)
        return findings

    def _distracting(self, context: AssembledContext, task_terms: set[str]) -> list[ContextFinding]:
        """Retrieved chunks with nothing to do with the task (history and memory are kept)."""
        return [
            ContextFinding(
                mode="distraction",
                source=_label(chunk),
                rule="no term shared with the task",
                detail="retrieved material off the task",
                action="dropped",
                index=index,
            )
            for index, chunk in enumerate(context.chunks)
            if chunk.kind == "retrieved" and not chunk.pinned and not _on_task(chunk, task_terms)
        ]

    # ── degradation over checkpoints ─────────────────────────────────

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


# A claim as stated: its value, and the chunk (with its position) that stated it.
_Stated = list[tuple[str, tuple[int, Chunk]]]


def _claims_by_key(context: AssembledContext) -> dict[str, _Stated]:
    """Every entity-keyed claim in the asserting chunks."""
    claims: dict[str, _Stated] = {}
    for index, chunk in enumerate(context.chunks):
        if chunk.kind not in _ASSERTING_KINDS:
            continue
        entity = _entity(chunk)
        for key, value in _claims(chunk):
            keyed = f"{entity} · {key}" if entity else key
            claims.setdefault(keyed, []).append((value, (index, chunk)))
    return claims


def _resolve(key: str, stated: _Stated) -> list[ContextFinding]:
    """Keep the most authoritative statement of a clash; drop the rest, or flag a tie."""
    top = max(_AUTHORITY.get(chunk.kind, 0) for _, (_, chunk) in stated)
    winners = {value for value, (_, chunk) in stated if _AUTHORITY.get(chunk.kind, 0) == top}
    detail = " vs ".join(f"{value} ({_label(chunk)})" for value, (_, chunk) in stated)
    tie = len(winners) > 1
    return [
        ContextFinding(
            mode="clash",
            source=_label(chunk),
            rule=f"'{key}' disagrees",
            detail=detail,
            index=index,
            action=(
                "flagged (equal authority)"
                if tie
                else "kept (higher authority)"
                if _AUTHORITY.get(chunk.kind, 0) == top
                else "dropped (lower authority)"
            ),
        )
        for _, (index, chunk) in stated
    ]


def _claims(chunk: Chunk) -> list[tuple[str, str]]:
    """`key: value` claims stated in a chunk — never its provenance label."""
    found: list[tuple[str, str]] = []
    for line in chunk.content.splitlines():
        if is_provenance_header(line):
            continue
        key, separator, value = line.partition(":")
        if separator and key.strip() and value.strip():
            found.append((key.strip().casefold(), value.strip().casefold()))
    return found


def _entity(chunk: Chunk) -> str:
    """The identifier a chunk's claims are about, or "" if it names none."""
    body = "\n".join(line for line in chunk.content.splitlines() if not is_provenance_header(line))
    match = _ENTITY.search(body)
    return match.group(0).casefold() if match else ""


def _label(chunk: Chunk) -> str:
    return chunk.source or chunk.kind


def _on_task(chunk: Chunk, task_terms: set[str]) -> bool:
    """Whether a chunk shares a word with the task (the same tokenizer as BM25)."""
    return bool(task_terms & set(tokenize(chunk.content)))


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
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) > _DUPLICATE_OVERLAP
