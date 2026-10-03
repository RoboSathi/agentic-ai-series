"""Agentic retrieval — the model decides when to look something up (E07, E19).

A RAG pipeline retrieves once, at a point fixed at design time. Here
retrieval is a tool the model may call, more than once, with queries it
composes from what it has already found.

That is worth the extra cost exactly when the right query is not knowable
up front — when answering needs a follow-up the first result suggested.
When every request needs the same grounding, this is a pipeline with
extra steps and more ways to go wrong.

The retrieval mechanics themselves (chunking, hybrid search, reranking)
are E19's subject; `agent_platform.context` already implements them and
this pattern composes that rather than re-deriving it.
"""

from agent_platform.context import Chunk, HybridRetriever
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import get_logger
from design_patterns.types import PatternFamily, PatternRun, Step

_DECIDE = (
    "You are answering a question using a searchable knowledge base.\n"
    "What you have found so far:\n{found}\n\n"
    "Question: {task}\n\n"
    "If you need to search, reply with exactly: SEARCH: <query>\n"
    "If you can answer now, reply with the answer itself."
)
_SEARCH_PREFIX = "SEARCH:"


class AgenticRetrievalPattern:
    """Lets the model issue its own searches until it can answer."""

    name: str = "agentic_retrieval"
    family: PatternFamily = "AG"

    def __init__(
        self, llm: LLMClient, retriever: HybridRetriever, max_searches: int = 3, top_k: int = 3
    ) -> None:
        self._llm = llm
        self._retriever = retriever
        self._max_searches = max_searches
        self._top_k = top_k
        self._logger = get_logger("design_patterns")

    async def run(self, task: str) -> PatternRun:
        """Search as many times as the model asks, up to the cap."""
        found: list[Chunk] = []
        steps: list[Step] = []

        for _ in range(self._max_searches):
            decision = await self._decide(task, found)
            if not decision.startswith(_SEARCH_PREFIX):
                steps.append(Step(kind="answer", detail="answered", output=decision))
                return PatternRun(
                    pattern=self.name,
                    family=self.family,
                    output=decision,
                    steps=steps,
                    metadata={
                        "searches": len(found and steps) and sum(s.kind == "search" for s in steps)
                    },
                )
            query = decision[len(_SEARCH_PREFIX) :].strip()
            hits = await self._retriever.retrieve(query, self._top_k)
            found.extend(hits)
            steps.append(Step(kind="search", detail=query, output=f"{len(hits)} result(s)"))
            self._logger.debug("Agent issued a search", query=query, hits=len(hits))

        # Out of searches: answer from whatever was found rather than
        # returning nothing.
        answer = await self._decide(task, found, force_answer=True)
        steps.append(Step(kind="answer", detail="cap reached", output=answer))
        return PatternRun(
            pattern=self.name,
            family=self.family,
            output=answer,
            steps=steps,
            metadata={"searches": self._max_searches, "hit_cap": True},
        )

    async def _decide(self, task: str, found: list[Chunk], force_answer: bool = False) -> str:
        """Ask whether to search again, or to answer now."""
        evidence = "\n".join(f"- {chunk.content}" for chunk in found) or "(nothing yet)"
        prompt = _DECIDE.format(found=evidence, task=task)
        if force_answer:
            prompt += "\n\nYou cannot search again. Answer from what you have."
        result = await self._llm.complete(
            CompletionRequest(messages=[Message(role="user", content=prompt)])
        )
        return result.content.strip()
