"""RAG pipeline — retrieve, then generate (E04, E19).

Retrieval is a fixed stage: every request retrieves, then generates. The
decision of *when* to look something up was made at design time.

That is the whole distinction from agentic retrieval, where the model
decides mid-reasoning whether it needs to look something up. A pipeline
is cheaper, bounded and auditable; it just cannot ask a follow-up
question of the corpus.

Reach for it when every request needs the same kind of grounding.
"""

from typing import Awaitable, Callable, Protocol, runtime_checkable

from agentic_commons.context import Chunk
from agentic_commons.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

Generate = Callable[[str, list[Chunk]], Awaitable[str]]


@runtime_checkable
class Retriever(Protocol):
    """Finds the k most useful chunks for a query."""

    async def retrieve(self, query: str, k: int) -> list[Chunk]: ...


class RagPipelinePattern:
    """Retrieve a fixed number of chunks, then generate from them."""

    name: str = "rag_pipeline"
    family: PatternFamily = "WF"

    def __init__(self, retriever: Retriever, generate: Generate, top_k: int = 3) -> None:
        self._retriever = retriever
        self._generate = generate
        self._top_k = top_k
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Retrieve, then generate grounded in what came back.

        Retrieving nothing is reported rather than hidden. A pipeline
        that generates anyway produces an ungrounded answer that looks
        exactly like a grounded one.
        """
        chunks = await self._retriever.retrieve(task, self._top_k)
        steps = [
            Step(kind="retrieve", detail=chunk.source, output=chunk.content[:80])
            for chunk in chunks
        ]
        if not chunks:
            self._logger.warning("Retrieval returned nothing", query=task[:60])
            return PatternRun(
                pattern=self.name,
                family=self.family,
                succeeded=False,
                output="",
                steps=[Step(kind="retrieve", detail="no matches")],
                metadata={"retrieved": 0},
            )
        answer = await self._generate(task, chunks)
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=answer,
            steps=[*steps, Step(kind="generate", detail="grounded", output=answer)],
            metadata={"retrieved": len(chunks)},
        )
