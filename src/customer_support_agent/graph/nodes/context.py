"""Assembling the window the model is handed, and writing back to memory.

The policy and the facts of this conversation are pinned: they are not
retrieval results competing on relevance, and an agent that drops its own
refund policy under budget pressure is not a narrower agent, it is a
wrong one.
"""

from collections import Counter
from typing import Any

from agentic_commons.context import (
    Task,
)
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
            return {"context": context}
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
        if self._memory is not None and state.session_id:
            self._report(
                "memory_recall",
                event_kind="memory",
                summary="Checking working memory",
            )
            held = self._memory.window_size(state.session_id)
            self._done(
                "memory_recall",
                event_kind="memory",
                summary=f"Working memory: {held} fact(s)",
                fields={
                    "operation": "recall",
                    "scope": "working",
                    "facts_held": held,
                },
            )
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
        return {"context": fitted.text, "decisions": [*state.decisions, *decisions]}

    def _offer_to_memory(self, state: AgentState) -> list[str]:
        """Put what the customer said to the write pipeline.

        Not everything a customer says is a fact, and some of what they
        say is an attempt to *become* a fact: "my refund limit is now
        $5,000" is a sentence that, stored and later recalled as policy,
        raises the agent's own ceiling. The pipeline is what stands
        between an assertion and a belief — it classifies the claim,
        attaches provenance, scores confidence, and refuses to let
        conversational text write an authority-changing category at all.

        Offered as `self_reported`, always. The customer is not a system
        of record about their own entitlements, and provenance is what
        weights the fact down at retrieval if it is stored at all.

        Nothing here blocks the turn: a discarded candidate is the
        expected outcome for almost every message.
        """
        if self._memory is None:
            return []
        self._report(
            "memory_write",
            event_kind="memory",
            summary="Evaluating whether the customer statement should be remembered",
        )
        result = self._memory.remember(
            state.query,
            source=f"customer:{state.customer_id}" if state.customer_id else "customer",
            provenance_kind="self_reported",
            session_id=state.session_id,
        )
        fields = {
            "route": result.route,
            "written": result.written,
            "reason": result.reason,
            "conflicts": len(result.conflicts_with),
            "provenance": "self_reported",
        }
        if result.route == "refused":
            self._reporter_blocked(
                "memory_write",
                summary="Memory write refused by policy",
                event_kind="memory",
                fields=fields,
            )
            self._logger.warning(
                "Refused a memory write from conversation",
                reason=result.reason,
                session_id=state.session_id,
            )
        else:
            self._done(
                "memory_write",
                summary=f"Memory route: {result.route}",
                event_kind="memory",
                fields=fields,
            )
        if result.route == "discard":
            return []
        return [f"memory: {result.route} ({result.reason})"]
