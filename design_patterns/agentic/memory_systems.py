"""Memory systems — the agent decides what to keep and what to recall (E07, E13).

Named as a pattern here and built out in full in M5. What makes it
*agentic* is not that state persists — a workflow can write to a database
— but that the agent chooses what is worth remembering and what is worth
recalling.

Which is why memory is a security boundary (E16) rather than a
convenience. An agent that writes what it is told, and later reads it
back as fact, can be told something false once and act on it for as long
as the record survives. So writes go through the pipeline that classifies
and authorises them; this pattern composes that rather than calling a
store directly.

Composes `agentic_commons.memory`: the four CoALA stores, the write
pipeline and the composite retrieval scorer already exist.
"""

from agentic_commons.logging import get_logger
from agentic_commons.memory import MemoryFacade, MemoryQuery, ProvenanceKind
from design_patterns.types import PatternFamily, PatternRun, Step


class MemorySystemsPattern:
    """Recall what is relevant, act, then offer what was learned back."""

    name: str = "memory_systems"
    family: PatternFamily = "AG"

    def __init__(self, memory: MemoryFacade, recall_limit: int = 5) -> None:
        self._memory = memory
        self._recall_limit = recall_limit
        self._logger = get_logger("design_patterns")

    async def run(
        self,
        task: str,
        session_id: str = "",
        learned: str = "",
        source: str = "agent",
        provenance: ProvenanceKind = "agent_inference",
    ) -> PatternRun:
        """Recall, then optionally write back what the turn established."""
        recalled = await self._memory.recall(
            MemoryQuery(text=task, session_id=session_id, limit=self._recall_limit)
        )
        steps = [
            Step(
                kind="recall",
                detail=f"{scored.fact.key} (score {scored.score:.2f})",
                output=scored.fact.value,
            )
            for scored in recalled
        ]
        if not learned:
            return PatternRun(
                pattern=self.name,
                family=self.family,
                output="\n".join(scored.fact.value for scored in recalled),
                steps=steps,
                metadata={"recalled": len(recalled)},
            )

        # The pipeline decides, not the agent. A write that classifies as
        # authority-changing is refused however confidently it is
        # asserted, and a low-confidence one goes to a curator rather
        # than into the store.
        result = self._memory.remember(learned, source=source, provenance_kind=provenance)
        steps.append(Step(kind="write", detail=f"{result.route}: {result.reason}", output=learned))
        self._logger.debug("Memory write routed", route=result.route)
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=learned,
            steps=steps,
            succeeded=result.route != "refused",
            metadata={"recalled": len(recalled), "write_route": result.route},
        )
