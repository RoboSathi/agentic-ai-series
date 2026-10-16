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

from typing import Any

from agent_platform.config import get_settings
from agent_platform.context import Context, ContextWindow, Task
from agent_platform.context.types import (
    AssembledContext,
    Chunk,
    is_provenance_header,
)
from agent_platform.context.window import CONTEXT_STAGE
from agent_platform.memory.context_source import SOURCE_PREFIX as MEMORY_SOURCE_PREFIX
from agent_platform.memory.context_source import SUBJECT_ID
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION, WRITE_INTENT
from customer_support_agent.graph.kit import NodeKit
from customer_support_agent.graph.nodes._support import NodeSupport
from customer_support_agent.graph.state import AgentState
from customer_support_agent.grounding.evidence import GROUNDING_EVIDENCE

# Recalled semantic facts, by the source prefix `MemoryRecallContextSource` gives them.
_SEMANTIC_SOURCE = f"{MEMORY_SOURCE_PREFIX}semantic:"
_VERIFIED = "verified_system_of_record"


class ContextNodes(NodeSupport):
    def __init__(self, kit: NodeKit) -> None:
        super().__init__(kit)
        self._context = kit.context
        self._strategies = kit.strategies
        self._context_health = kit.context_health

    def _task_metadata(self, state: AgentState) -> dict[str, Any]:
        """What this turn has settled that assembly should know about."""
        metadata: dict[str, Any] = {}
        order_id = state.metadata.get(ORDER_UNDER_DISCUSSION)
        if order_id is not None:
            metadata[ORDER_UNDER_DISCUSSION] = order_id
        if _is_write_intent(state.metadata.get("intent")):
            metadata[WRITE_INTENT] = True
        # Whose long-term memory this turn may recall: the verified
        # customer's, and no one else's.
        if state.customer_id is not None:
            metadata[SUBJECT_ID] = str(state.customer_id)
        return metadata

    async def assemble_context(self, state: AgentState) -> dict[str, Any]:
        """Build the window: policy, recalled memory, and the query."""
        if self._context is None:
            # Nothing to assemble, and nothing that needs standing in for
            # it: the refund policy reaches the model in the system
            # prompt either way, so an agent with context assembly
            # withheld is one that reasons without *retrieved* material,
            # not one that reasons without its own policy.
            self._done(
                CONTEXT_STAGE,
                summary=get_settings().context.window.withheld,
                event_kind="context",
                fields={"source": "none"},
                quiet=True,
            )
            return {"context": Context()}
        window = ContextWindow(self._context, self._strategies, self._context_health)
        result = await window.build(
            Task(
                query=state.query,
                session_id=state.session_id,
                actor=state.actor,
                # Assembly needs the task, not just the words. "I want a
                # refund" names no order; the turn already knows which
                # one it is about, and a source that is told stops
                # guessing.
                metadata=self._task_metadata(state),
            ),
            self._reporter,
        )
        fitted = result.window
        decisions = [f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens"]
        if result.modes:
            decisions.append(f"context health: {', '.join(result.modes)}")
        return {
            "context": Context(items=[fitted.text], window=fitted),
            "decisions": [*state.decisions, *decisions],
            # What the model saw that a reply may be grounded on — kept after
            # mitigation, so a poisoned or dropped chunk is never evidence.
            "metadata": {**state.metadata, GROUNDING_EVIDENCE: _grounding_sources(fitted)},
        }


def _grounding_sources(window: AssembledContext) -> list[dict[str, str]]:
    """Knowledge passages, order records and *verified* semantic memories in the window.

    Episodic memory records what the customer said, not what is true, and
    working-memory state is bookkeeping — neither grounds a claim.
    """
    return [
        {"kind": chunk.kind, "source": chunk.source, "content": _without_label(chunk.content)}
        for chunk in window.chunks
        if chunk.kind == "retrieved" or (chunk.kind == "memory" and _verified_memory(chunk))
    ]


def _verified_memory(chunk: Chunk) -> bool:
    confirmed = bool(chunk.metadata.get("confirmed"))
    authoritative = chunk.metadata.get("provenance") == _VERIFIED
    return chunk.source.startswith(_SEMANTIC_SOURCE) and (confirmed or authoritative)


def _without_label(content: str) -> str:
    return "\n".join(line for line in content.splitlines() if not is_provenance_header(line))


def _is_write_intent(name: object) -> bool:
    """Whether the classified intent may change an order (`mode: write` in config)."""
    return any(
        intent.name == name and intent.mode == "write" for intent in get_agent_settings().intents
    )
