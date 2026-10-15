"""Retrieval-based tool selection at scale (LLD §19).

Exposing every tool to the model does not scale: past a few dozen, the
model's selection accuracy falls even though nothing about any single
tool changed. Selection applies the same cascade discipline the rest of
the system uses for scoring — exact name, then namespace filter, then
embedding similarity — so the expensive tier only runs on what the cheap
tiers could not settle.

The ranker arrives by injection as a structural Protocol rather than as
an import of `evals`: `tools` sits at layer 2 and the cascade at layer 5
(HLD §layering).
"""

import json
import time
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.llm.tokenizer import count_tokens
from agent_platform.logging import get_logger, instrument
from agent_platform.tools.registry import NamespacedToolRegistry
from agent_platform.tools.types import ToolDefinition, ToolSelection


@runtime_checkable
class RelevanceRanker(Protocol):
    """Scores a task against candidate texts.

    Structurally satisfied by `evals.NollmScoringCascadeImpl.rank`.
    """

    async def rank(self, text: str, candidates: list[str]) -> list[float]: ...


class SelectionMetrics(BaseModel):
    """The four selection metrics for one selection."""

    tools_available: int
    tools_exposed: int
    exposed_token_estimate: int
    irrelevant_exposure_rate: float
    latency_ms: float


@instrument("tools")
class ToolRetrievalSelector:
    """Selects the few tools a task plausibly needs (Facade over the cascade)."""

    def __init__(
        self,
        registry: NamespacedToolRegistry,
        ranker: RelevanceRanker | None = None,
    ) -> None:
        settings = get_settings()
        self._registry = registry
        self._ranker = ranker
        self._k = settings.tools.selection_k
        self._min_score = settings.tools.selection_min_score
        self._logger = get_logger("tools")
        self._last_metrics: SelectionMetrics | None = None

    async def select(
        self, task: str, k: int | None = None, namespace: str | None = None
    ) -> list[ToolDefinition]:
        """The tools to expose for this task, best first."""
        selections = await self.select_scored(task, k, namespace)
        return [tool for selection in selections if (tool := self._registry.get(selection.tool))]

    async def select_scored(
        self, task: str, k: int | None = None, namespace: str | None = None
    ) -> list[ToolSelection]:
        """Selection with the score and tier that justified each choice."""
        started = time.perf_counter()
        limit = k if k is not None else self._k
        candidates = (
            self._registry.in_namespace(namespace) if namespace else self._registry.all_tools()
        )
        exact = self._exact_match(task, candidates)
        selections = exact if exact else await self._by_similarity(task, candidates, limit)
        self._last_metrics = self._measure(
            candidates, selections, (time.perf_counter() - started) * 1000
        )
        return selections

    @property
    def last_metrics(self) -> SelectionMetrics | None:
        """Metrics for the most recent selection — the regression evidence."""
        return self._last_metrics

    def _exact_match(self, task: str, candidates: list[ToolDefinition]) -> list[ToolSelection]:
        """Tier 1: the task names a tool outright, so no ranking is needed."""
        lowered = task.casefold()
        return [
            ToolSelection(tool=tool.qualified_name, score=1.0, reason="named in the task")
            for tool in candidates
            if tool.qualified_name.casefold() in lowered
        ]

    async def _by_similarity(
        self, task: str, candidates: list[ToolDefinition], limit: int
    ) -> list[ToolSelection]:
        """Tier 3: rank tool descriptions against the task, keep the top k."""
        if not candidates:
            return []
        if self._ranker is None:
            self._logger.debug("No ranker injected; exposing the registry unfiltered")
            return [
                ToolSelection(tool=tool.qualified_name, score=0.0, reason="no ranker configured")
                for tool in candidates[:limit]
            ]
        texts = [f"{tool.qualified_name}: {tool.description}" for tool in candidates]
        scores = await self._ranker.rank(task, texts)
        ranked = sorted(
            (
                ToolSelection(
                    tool=tool.qualified_name, score=score, reason=f"similarity {score:.3f}"
                )
                for tool, score in zip(candidates, scores)
                if score >= self._min_score
            ),
            key=lambda selection: selection.score,
            reverse=True,
        )
        return ranked[:limit]

    def _measure(
        self,
        candidates: list[ToolDefinition],
        selections: list[ToolSelection],
        latency_ms: float,
    ) -> SelectionMetrics:
        """Quantify what this selection cost and what it left exposed."""
        exposed = [tool for s in selections if (tool := self._registry.get(s.tool)) is not None]
        schema_tokens = sum(count_tokens(json.dumps(tool.to_openai_schema())) for tool in exposed)
        available = len(candidates)
        return SelectionMetrics(
            tools_available=available,
            tools_exposed=len(exposed),
            exposed_token_estimate=schema_tokens,
            irrelevant_exposure_rate=(
                (len(exposed) - 1) / available if available and exposed else 0.0
            ),
            latency_ms=latency_ms,
        )
