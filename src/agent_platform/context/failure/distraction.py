"""Distraction: material that pulls the answer off the task.

Relevance is similarity between the task and each retrieved chunk, by the shared similarity
seam (a local encoder where the agent has one), so a paraphrase of the question is on task
and a passage that merely shares a common word is not. A model repeating its own failed
step is the loop's to catch (`loop_control`'s semantic stagnation tier): it is a property of
the conversation, not of the window.
"""

from agent_platform.config import get_settings
from agent_platform.context.failure.common import finding, label, snippet
from agent_platform.context.similarity import TextSimilarity
from agent_platform.context.types import AssembledContext, ContextFinding


class DistractionCheck:
    def __init__(self, similarity: TextSimilarity) -> None:
        self._similarity = similarity

    async def relevance(self, context: AssembledContext, query: str) -> list[float]:
        """Each chunk's similarity to the task (1.0 for a pinned chunk, which never competes)."""
        if not query.strip() or not context.chunks:
            return [1.0 for _ in context.chunks]
        row = (await self._similarity.matrix([query], [c.content for c in context.chunks]))[0]
        return [1.0 if chunk.pinned else score for chunk, score in zip(context.chunks, row)]

    async def ratio(self, context: AssembledContext, query: str) -> float:
        """Share of unpinned chunks below the relevance floor."""
        floor = get_settings().context.failure_detection.min_relevance
        scores = await self.relevance(context, query)
        competing = [s for c, s in zip(context.chunks, scores) if not c.pinned]
        return sum(s < floor for s in competing) / len(competing) if competing else 0.0

    async def findings(self, context: AssembledContext, query: str) -> list[ContextFinding]:
        settings = get_settings().context.failure_detection
        scores = await self.relevance(context, query)
        competing = [s for c, s in zip(context.chunks, scores) if not c.pinned]
        ratio = (
            sum(s < settings.min_relevance for s in competing) / len(competing)
            if competing
            else 0.0
        )
        if ratio <= settings.mode_ratio:
            return []
        dropped = [
            finding("distraction", "distraction_chunk", source=label(chunk), action="dropped",
                    index=index, score=score, floor=settings.min_relevance,
                    line=snippet(chunk.content))
            for index, (chunk, score) in enumerate(zip(context.chunks, scores))
            if chunk.kind == "retrieved" and not chunk.pinned and score < settings.min_relevance
        ]  # fmt: skip
        return dropped or [
            finding("distraction", "distraction_window", source="window", ratio=settings.mode_ratio)
        ]
