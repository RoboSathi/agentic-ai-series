"""Assembling the window the model is handed, and writing back to memory.

The policy and the facts of this conversation are pinned: they are not
retrieval results competing on relevance, and an agent that drops its own
refund policy under budget pressure is not a narrower agent, it is a
wrong one.
"""

from collections import Counter
from typing import Any

from agentic_commons.context import Context, Task
from agentic_commons.llm.embeddings import tokenize
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.prompts.policy import policy_text


class ContextNodes(NodeSupport):
    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        self._report(
            "context_assembly",
            event_kind="context",
            summary="Assembling the model context window",
        )
        if self._context is None:
            context = policy_text()
            self._done(
                "context_assembly",
                summary="Policy context loaded",
                event_kind="context",
                fields={"source": "policy", "used_tokens": len(tokenize(context))},
            )
            return {"context": Context(items=[context])}
        assembled = await self._context.assemble(
            Task(query=state.query, session_id=state.session_id, actor=state.actor)
        )
        fitted = self._strategies.fit(assembled)
        decisions = [f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens"]
        modes = self._context_health.assess(fitted, set(tokenize(state.query)))
        if modes:
            self._logger.warning("Context window shows failure modes", modes=modes)
            decisions.append(f"context health: {', '.join(modes)}")

        by_kind = Counter(chunk.kind for chunk in fitted.chunks)
        context_fields = {
            "chunks": len(fitted.chunks),
            "dropped": len(fitted.dropped),
            "budget_tokens": fitted.budget_tokens,
            "used_tokens": fitted.used_tokens,
            "utilization": (
                round(fitted.used_tokens / fitted.budget_tokens, 2)
                if fitted.budget_tokens
                else 0.0
            ),
            "kinds": dict(by_kind),
            "strategies": fitted.applied_strategies,
            "health": modes or ["healthy"],
        }
        if modes:
            self._reporter_warning(
                "context_assembly",
                summary="Context health warning",
                event_kind="context",
                fields=context_fields,
            )
        else:
            self._done(
                "context_assembly",
                summary="Context window assembled",
                event_kind="context",
                fields=context_fields,
            )
        return {"context": Context(items=[fitted.text]), "decisions": [*state.decisions, *decisions]}

