"""Context failure detection — and mitigation (LLD §7).

Five distinct failure modes, which matter because they have different fixes:

  rot         — quality decays as the window grows, with nothing wrong in it   (rot.py)
  distraction — irrelevant material pulls the answer off task                  (distraction.py)
  confusion   — too many similar options; the model cannot choose            (confusion.py)
  clash       — the window contains contradictory facts                        (clash.py)
  poisoning   — a false or hostile statement is treated as established        (poisoning.py)

Treating all five as "the context is too long" leads to trimming a window whose problem was
that two chunks disagreed. `ContextFailureDetector` is the one interface over them:

  `diagnose` — what is wrong, per chunk, with the rule that found it;
  `mitigate` — what to do before the model sees the window: a poisoned chunk is dropped and
               an unverified claim cut out of its transcript, a clash keeps its more trusted
               side, a distracting chunk is dropped, rot reduces the transcript.

Every model it uses is local (the shared similarity seam, the NLI cross-encoder) and
injected by the composition root; without the NLI model the prose checks are not run, and
the findings say nothing about them rather than claiming a clean window.
"""

from agent_platform.config import get_settings
from agent_platform.context.failure.clash import ClashCheck
from agent_platform.context.failure.common import ContradictionModel, label
from agent_platform.context.failure.confusion import near_duplicate_ratio, window_findings
from agent_platform.context.failure.distraction import DistractionCheck
from agent_platform.context.failure.poisoning import InjectionScreen, PoisoningCheck
from agent_platform.context.failure.rot import RotMonitor
from agent_platform.context.similarity import TextSimilarity, jaccard, shared_similarity
from agent_platform.context.strategies import reduce_history
from agent_platform.context.types import (
    AssembledContext,
    Chunk,
    ContextFailureMode,
    ContextFinding,
    Task,
)
from agent_platform.llm.tokenizer import count_tokens
from agent_platform.logging import get_logger


class ContextFailureDetector:
    """Diagnoses the five modes in a window and mitigates them before the model call (Facade)."""

    def __init__(
        self,
        screen: InjectionScreen | None = None,
        nli: ContradictionModel | None = None,
        similarity: TextSimilarity | None = None,
        rot: RotMonitor | None = None,
    ) -> None:
        self._app_config = get_settings()
        similarity = similarity or shared_similarity()
        self._poisoning = PoisoningCheck(screen, nli, similarity)
        self._clash = ClashCheck(nli, similarity)
        self._distraction = DistractionCheck(similarity)
        self._rot = rot or RotMonitor()
        self._logger = get_logger("context")

    @property
    def rot(self) -> RotMonitor:
        """Where a session's answer quality is recorded, so rot can be judged."""
        return self._rot

    async def diagnose(self, context: AssembledContext, task: Task) -> list[ContextFinding]:
        """Every failure in this window, chunk by chunk, with the rule that found it."""
        return [
            *await self._poisoning.findings(context),
            *await self._clash.findings(context),
            *await self._distraction.findings(context, task.query),
            *window_findings(context),
            *self._rot.findings(context, task.session_id),
        ]

    async def assess(self, context: AssembledContext, task: Task) -> list[ContextFailureMode]:
        """Which failure modes this window currently shows."""
        return sorted({finding.mode for finding in await self.diagnose(context, task)})

    def mitigate(
        self, context: AssembledContext, findings: list[ContextFinding]
    ) -> tuple[AssembledContext, list[ContextFinding]]:
        """Drop, cut or reduce what should not reach the model; say what was done."""
        dropped_word = self._app_config.context.window.dropped_action
        removed = {
            f.index for f in findings if f.action.startswith(dropped_word) and f.index is not None
        }
        excised = _excised(context, findings, removed)
        kept = [chunk for index, chunk in enumerate(excised.chunks) if index not in removed]
        mitigated = context.model_copy(
            update={
                "chunks": kept,
                "used_tokens": sum(chunk.token_estimate for chunk in kept),
                "dropped": [*context.dropped, *(label(context.chunks[i]) for i in sorted(removed))],
            }
        )
        if any(finding.mode == "rot" for finding in findings):
            reduced = reduce_history(mitigated)
            cut = _lines_lost(mitigated, reduced)
            findings = [
                f.model_copy(update={"excise": cut}) if f.mode == "rot" else f for f in findings
            ]
            mitigated = reduced
        if mitigated != context:
            self._logger.warning(
                "Context mitigated before the model call",
                removed=[label(context.chunks[i]) for i in sorted(removed)],
                findings=[finding.model_dump() for finding in findings],
            )
        return mitigated, findings

    async def distraction_ratio(self, context: AssembledContext, query: str) -> float:
        return await self._distraction.ratio(context, query)

    def near_duplicate_ratio(self, context: AssembledContext) -> float:
        return near_duplicate_ratio(context)

    def contradictions(self, context: AssembledContext) -> list[tuple[str, str]]:
        return self._clash.contradictions(context)


def _excised(
    context: AssembledContext, findings: list[ContextFinding], removed: set[int]
) -> AssembledContext:
    """The window with each finding's `excise` lines cut out of its chunk."""
    cuts: dict[int, list[str]] = {}
    for finding in findings:
        if finding.excise and finding.index is not None and finding.index not in removed:
            cuts.setdefault(finding.index, []).extend(finding.excise)
    if not cuts:
        return context
    chunks = [_cut(chunk, cuts.get(index, [])) for index, chunk in enumerate(context.chunks)]
    return context.model_copy(update={"chunks": chunks})


def _lines_lost(before: AssembledContext, after: AssembledContext) -> list[str]:
    """The transcript lines a reduction removed, oldest first, so the card can name them."""
    kept = {line for chunk in after.chunks for line in chunk.content.splitlines()}
    return [
        " ".join(line.split())
        for chunk in before.chunks
        if chunk.kind == "history"
        for line in chunk.content.splitlines()
        if line.strip() and line not in kept
    ]


def _cut(chunk: Chunk, lines: list[str]) -> Chunk:
    """The chunk without the offending sentences, each cut from its own line only.

    A line left holding nothing but its speaker prefix ("assistant:") goes with them.
    """
    if not lines:
        return chunk
    prefix = get_settings().context.trust.model_line_prefix.casefold()
    kept: list[str] = []
    for row in chunk.content.splitlines():
        cut = row
        for line in lines:
            cut = cut.replace(line, "", 1)
        if cut != row and cut.strip().casefold() in ("", prefix):
            continue
        kept.append(cut)
    text = "\n".join(kept)
    return chunk.model_copy(update={"content": text, "token_estimate": count_tokens(text)})


def is_near_duplicate(left: str, right: str) -> bool:
    """Whether two chunks are near-copies, by the configured word overlap."""
    return jaccard(left, right) > get_settings().context.consolidation.near_duplicate_overlap


__all__ = [
    "ContextFailureDetector",
    "ContradictionModel",
    "InjectionScreen",
    "RotMonitor",
    "is_near_duplicate",
]
