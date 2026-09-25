"""Assembling the window the model is handed, and writing back to memory.

The facts of this conversation are pinned: the compact working-memory
state is what anchors the turn, and it is small enough that budget
pressure should never be the reason it goes. The transcript is not
pinned — it is the largest thing in the window and the most repetitive,
and a turn that cannot afford it is better off trimming it than dropping
a retrieved order.

The refund policy is not here at all. It reaches the model in the system
prompt, which is always present and never trimmed; contributing a second
verbatim copy from here meant the model read the same eight lines twice.
"""

from collections import Counter
from typing import Any

from agentic_commons.context import Context, Task
from agentic_commons.llm.embeddings import tokenize
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState


class ContextNodes(NodeSupport):
    def _working_memory_fields(self, state: AgentState) -> dict[str, Any]:
        """Return the bounded working-memory content actually available to context."""
        memory = state.working_memory
        history = []
        for item in memory.history:
            if isinstance(item, dict):
                history.append({
                    "role": item.get("role", ""),
                    "content": self._security.redact(str(item.get("content", ""))),
                })
            else:
                history.append(self._security.redact(str(item)))
        working_state = {
            str(key): self._security.redact(str(value))
            for key, value in memory.state.items()
        }
        return {
            "working_memory_history_turns": len(memory.history) // 2,
            "working_memory_history": history,
            "working_memory_state_items": len(working_state),
            "working_memory_state": working_state,
        }

    def _task_metadata(self, state: AgentState) -> dict[str, Any]:
        """What this turn has settled that assembly should know about."""
        order_id = state.metadata.get(ORDER_UNDER_DISCUSSION)
        return {ORDER_UNDER_DISCUSSION: order_id} if order_id is not None else {}

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        self._report(
            "context_assembly",
            event_kind="context",
            summary="Assembling the model context window",
        )
        if self._context is None:
            # Nothing to assemble, and nothing that needs standing in for
            # it: the refund policy reaches the model in the system
            # prompt either way, so an agent with context assembly
            # withheld is one that reasons without *retrieved* material,
            # not one that reasons without its own policy.
            self._done(
                "context_assembly",
                summary="Context assembly withheld · working memory read",
                event_kind="context",
                fields={
                    "source": "none",
                    "used_tokens": 0,
                    **self._working_memory_fields(state),
                },
            )
            return {"context": Context()}
        assembled = await self._context.assemble(
            Task(
                query=state.query,
                session_id=state.session_id,
                actor=state.actor,
                # Assembly needs the task, not just the words. "I want a
                # refund" names no order; the turn already knows which
                # one it is about, and a source that is told stops
                # guessing.
                metadata=self._task_metadata(state),
            )
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
            **self._working_memory_fields(state),
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
