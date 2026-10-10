"""PREFETCH: the reads a turn's procedure says come first, run before the model is asked.

A refund procedure always starts by reading the order and its refund; asking the model to
ask for them costs a whole model round each. So the harness runs a procedure's declared
`prefetch` reads itself, through the one governed tool path (Gatekeeper, Executor, audit,
facts), and hands the model their results as if it had asked. The model starts where the
procedure's first steps end.

A read is skipped — never guessed — when its arguments are not known (no order under
discussion yet), when the intent does not allow it, or when it would change something.
Prefetched reads do not count as tool rounds, and the turn's loop control remembers them,
so a model that asks again is answered with the earlier result.
"""

import json
from typing import Any

from agent_platform.llm.types import Message, ModelToolCall
from customer_support_agent.config import get_agent_settings
from customer_support_agent.graph.nodes._support import (
    PREFETCHED,
    TOOL_BLOCKED,
    TOOL_COMPLETED,
    TURN_OFFERED_TOOLS,
    NodeSupport,
)
from customer_support_agent.graph.state import AgentState


class PrefetchNodes(NodeSupport):
    async def prefetch(self, state: AgentState) -> dict[str, Any]:
        """Say which procedures this turn runs and what they read first; then read.

        The PROCEDURE card comes before any read, naming the reads and the tools the model
        will be offered, so no tool call ever appears without the procedure that asked for it.
        """
        activation = self._activation(state)
        reads = self._runnable_reads(activation, state)
        offered = await self._offered_tools(state)
        self._instructions.announce(
            activation,
            prefetched=[tool.model_name for _, tool, _ in reads],
            offered=[tool.model_name for tool in offered],
        )
        ran = [
            tool.model_name
            for procedure, tool, params in reads
            if await self._read(state, procedure, tool, params)
        ]
        metadata = {
            **state.metadata,
            PREFETCHED: ran,
            TURN_OFFERED_TOOLS: [tool.qualified_name for tool in offered],
        }
        return {"metadata": metadata}

    def _runnable_reads(
        self, activation: Any, state: AgentState
    ) -> list[tuple[str, Any, dict[str, Any]]]:
        """(procedure, tool, arguments) for each prefetch read that can run now.

        A read is left out — never guessed — when its tool is unknown or changes something,
        when the intent does not allow it, or when an argument it needs is not yet known.
        """
        reads: list[tuple[str, Any, dict[str, Any]]] = []
        for procedure, name in _planned(activation):
            tool = self._registry.get(name)
            if tool is None or tool.side_effecting or not self._intent_allows(state, tool.name):
                continue
            params = self._prefetch_params(tool, state)
            if params is not None:
                reads.append((procedure, tool, params))
        return reads

    async def _read(
        self, state: AgentState, procedure: str, tool: Any, params: dict[str, Any]
    ) -> bool:
        """One read, as a tool call the model will see; False when it was refused."""
        call = ModelToolCall(
            id=get_agent_settings().prefetch.call_id.format(tool=tool.model_name),
            name=tool.model_name,
            arguments=json.dumps(params),
        )
        self._steps.tool_request(
            call.name, call.arguments, redact=self._security.redact, prefetched_for=procedure
        )
        text, _decision, _stage, invocation = await self._invoke(
            tool.qualified_name, call.arguments, state, {tool.qualified_name}
        )
        self._hand_to_model(state, call, text, invocation)
        return invocation.get("tool_status") == TOOL_COMPLETED

    def _hand_to_model(
        self, state: AgentState, call: ModelToolCall, text: str, invocation: dict[str, Any]
    ) -> None:
        """The read as the model would have made it: its call, its result, its evidence."""
        memory = state.working_memory
        memory.add_current_turn(Message(role="assistant", content="", tool_calls=[call]))
        memory.add_current_turn(Message(role="tool", content=text, tool_call_id=call.id))
        self._add_action(
            state,
            call.name,
            call.arguments,
            str(invocation.get("evidence", text)),
            outcome=str(invocation.get("tool_status", TOOL_BLOCKED)),
        )

    def _intent_allows(self, state: AgentState, tool: str) -> bool:
        """A classified intent bounds what may run; the unrouted path is bounded by policy."""
        allowed = state.metadata.get("intent_tools")
        return not allowed or tool in allowed

    @staticmethod
    def _prefetch_params(tool: Any, state: AgentState) -> dict[str, Any] | None:
        """Each argument from the turn state the config names; None if a required one is unknown."""
        sources = get_agent_settings().prefetch.arguments
        fields = tool.args_schema.model_fields
        params = {
            arg: state.metadata[key]
            for arg, key in sources.items()
            if arg in fields and state.metadata.get(key) is not None
        }
        missing = [
            arg for arg, field in fields.items() if field.is_required() and arg not in params
        ]
        return None if missing else params


def _planned(activation: Any) -> list[tuple[str, str]]:
    """(procedure, tool) for every prefetch read, each tool once, in procedure order."""
    seen: set[str] = set()
    planned: list[tuple[str, str]] = []
    for item in activation.skills:
        for name in item.skill.document.prefetch:
            if name not in seen:
                seen.add(name)
                planned.append((item.skill.name, name))
    return planned
