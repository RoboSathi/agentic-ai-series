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

from agentic_commons.context import Context, Task
from agentic_commons.context.types import (
    AssembledContext,
    Chunk,
    ContextFinding,
    is_provenance_header,
)
from agentic_commons.llm.embeddings import tokenize
from agentic_commons.memory.context_source import SOURCE_PREFIX as MEMORY_SOURCE_PREFIX
from agentic_commons.memory.context_source import SUBJECT_ID
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.graph.nodes._support import GROUNDING_EVIDENCE, NodeSupport
from customer_support_agent.graph.state import AgentState

# Recalled semantic facts, by the source prefix `MemoryRecallContextSource` gives them.
_SEMANTIC_SOURCE = f"{MEMORY_SOURCE_PREFIX}semantic:"
_VERIFIED = "verified_system_of_record"
# Retrieval noise the health check removes as a matter of course. Filtering
# it is the pipeline working, so it reports green rather than as a warning.
_ROUTINE_MODES = frozenset({"distraction"})
# `ContextFinding.action` for a finding nothing was done about.
_NOT_ACTED_ON = "flagged"


class ContextNodes(NodeSupport):
    def _task_metadata(self, state: AgentState) -> dict[str, Any]:
        """What this turn has settled that assembly should know about."""
        metadata: dict[str, Any] = {}
        order_id = state.metadata.get(ORDER_UNDER_DISCUSSION)
        if order_id is not None:
            metadata[ORDER_UNDER_DISCUSSION] = order_id
        # Whose long-term memory this turn may recall: the verified
        # customer's, and no one else's.
        if state.customer_id is not None:
            metadata[SUBJECT_ID] = str(state.customer_id)
        return metadata

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
                summary="Context assembly withheld",
                event_kind="context",
                fields={"source": "none"},
                quiet=True,
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
        fitted, findings = self._check_health(fitted, state)
        modes = sorted({finding.mode for finding in findings})
        decisions = [f"context: {fitted.used_tokens}/{fitted.budget_tokens} tokens"]
        if modes:
            decisions.append(f"context health: {', '.join(modes)}")

        # What assembly cost, for the log and the tally. The model's input is
        # on the LLM Request card, so nothing here repeats it.
        cost = {
            "chunks": len(fitted.chunks),
            "dropped": len(fitted.dropped),
            "used_tokens": fitted.used_tokens,
            "budget_tokens": fitted.budget_tokens,
        }
        if modes and not _routine(findings):
            self._reporter_warning(
                "context_assembly",
                summary=f"Context health: {', '.join(modes)} — {_acted(findings)}",
                event_kind="context",
                fields={
                    "findings": [
                        f"{f.mode} · {f.source} · {f.rule}: {f.detail} → {f.action}"
                        for f in findings
                    ],
                    **cost,
                },
            )
        else:
            self._done(
                "context_assembly",
                summary="Context window assembled",
                event_kind="context",
                fields=cost,
                quiet=True,
            )
        return {
            "context": Context(items=[fitted.text]),
            "decisions": [*state.decisions, *decisions],
            # What the model saw that a reply may be grounded on — kept after
            # mitigation, so a poisoned or dropped chunk is never evidence.
            "metadata": {**state.metadata, GROUNDING_EVIDENCE: _grounding_sources(fitted)},
        }

    def _check_health(
        self, fitted: AssembledContext, state: AgentState
    ) -> tuple[AssembledContext, list[ContextFinding]]:
        """Diagnose the window and mitigate before the model sees it (when declared)."""
        if self._context_health is None:
            return fitted, []
        findings = self._context_health.diagnose(fitted, set(tokenize(state.query)))
        if findings:
            self._logger.warning(
                "Context window shows failure modes",
                findings=[finding.model_dump() for finding in findings],
            )
        return self._context_health.mitigate(fitted, findings)


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


def _routine(findings: list[ContextFinding]) -> bool:
    """Whether every finding is ordinary retrieval noise that was already removed.

    Retrieval over-fetches on purpose and the health check trims what is off
    the task — that is the pipeline working, not a fault. Poisoning, a clash
    between sources, rot, or anything left in the window still warrants a
    warning.
    """
    return all(
        finding.mode in _ROUTINE_MODES and finding.action != _NOT_ACTED_ON for finding in findings
    )


def _acted(findings: list[ContextFinding]) -> str:
    """What mitigation did, in a few words."""
    dropped = sum(1 for finding in findings if finding.action.startswith("dropped"))
    return f"{dropped} chunk(s) removed" if dropped else "flagged, nothing removed"
