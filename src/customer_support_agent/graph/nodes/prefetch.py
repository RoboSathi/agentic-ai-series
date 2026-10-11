"""PREFETCH: the reads a turn's skill says come first, run before the model is asked.

A refund skill always starts by reading the order and its refund; asking the model to
ask for them costs a whole model round each. So the harness runs a skill's declared
`prefetch` reads itself, through the one governed tool path (Gatekeeper, Executor, audit,
facts), and hands the model their results as if it had asked. The model starts where the
skill's first steps end.

A read is skipped — never guessed — when its arguments are not known (no order under
discussion yet), when the intent does not allow it, or when it would change something.
A read the deterministic checks already made (the order row they proved is the
customer's) is answered from that row: the database is not asked twice for one order.
Prefetched reads do not count as tool rounds, and the turn's loop control remembers them,
so a model that asks again is answered with the earlier result.
"""

import json
from typing import Any

from agent_platform.llm.types import Message, ModelToolCall
from agent_platform.tools import ToolResult
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.order_reference import ORDER_UNDER_DISCUSSION
from customer_support_agent.graph.nodes._support import (
    ORDER_SNAPSHOT,
    PREFETCHED,
    TOOL_BLOCKED,
    TOOL_COMPLETED,
    TURN_OFFERED_TOOLS,
    NodeSupport,
)
from customer_support_agent.graph.state import AgentState
from customer_support_agent.tools.definitions import SNAPSHOT_READS


class PrefetchNodes(NodeSupport):
    async def prefetch(self, state: AgentState) -> dict[str, Any]:
        """Say which skills this turn runs and what they read first; then read.

        The SKILL card comes before any read, naming the reads and the tools the model
        will be offered, so no tool call ever appears without the skill that asked for it.
        """
        activation = self._activation(state)
        reads = self._runnable_reads(activation, state)
        offered = self._still_needed(state, await self._offered_tools(state), reads)
        self._instructions.announce(
            activation,
            prefetched=[tool.model_name for _, tool, _ in reads],
            offered=[tool.model_name for tool in offered],
        )
        failed = [
            tool
            for skill, tool, params in reads
            if not await self._read(state, skill, tool, params)
        ]
        ran = [tool.model_name for _, tool, _ in reads if tool not in failed]
        metadata = {
            **state.metadata,
            PREFETCHED: ran,
            # A read that was refused is the model's to retry; one that ran is answered.
            TURN_OFFERED_TOOLS: [tool.qualified_name for tool in [*offered, *failed]],
        }
        return {"metadata": metadata}

    @staticmethod
    def _still_needed(
        state: AgentState, offered: list[Any], reads: list[tuple[str, Any, dict[str, Any]]]
    ) -> list[Any]:
        """The offer, less what this turn has no use for: reads prefetch answers, finders.

        Offering the model a read whose answer is already in front of it invites a
        redundant call; offering a tool that finds an order, once the order is known,
        invites a detour.
        """
        answered = {tool.qualified_name for _, tool, _ in reads}
        needless = (
            set(get_agent_settings().prefetch.needless_when_order_known)
            if state.metadata.get(ORDER_UNDER_DISCUSSION) is not None
            else set()
        )
        return [
            tool
            for tool in offered
            if tool.qualified_name not in answered and tool.name not in needless
        ]

    def _runnable_reads(
        self, activation: Any, state: AgentState
    ) -> list[tuple[str, Any, dict[str, Any]]]:
        """(skill, tool, arguments) for each prefetch read that can run now.

        A read is left out — never guessed — when its tool is unknown or changes something,
        when the intent does not allow it, or when an argument it needs is not yet known.
        """
        reads: list[tuple[str, Any, dict[str, Any]]] = []
        for skill, name in _planned(activation):
            tool = self._registry.get(name)
            if tool is None or tool.side_effecting or not self._intent_allows(state, tool.name):
                continue
            params = self._prefetch_params(tool, state)
            if params is not None:
                reads.append((skill, tool, params))
        return reads

    async def _read(self, state: AgentState, skill: str, tool: Any, params: dict[str, Any]) -> bool:
        """One read, as a tool call the model will see; False when it was refused."""
        call = ModelToolCall(
            id=get_agent_settings().prefetch.call_id.format(tool=tool.model_name),
            name=tool.model_name,
            arguments=json.dumps(params),
        )
        served = _served_from_checks(state, tool.name, params)
        if served is not None:
            self._steps.served_read(
                call.name,
                call.arguments,
                served.output,
                skill=skill,
                source=get_agent_settings().prefetch.served_from,
                redact=self._security.redact,
            )
            self._hand_to_model(
                state,
                call,
                served.output,
                {"tool_status": TOOL_COMPLETED, "evidence": served.output},
            )
            return True
        self._steps.tool_request(
            call.name, call.arguments, redact=self._security.redact, prefetched_for=skill
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
    """(skill, tool) for every prefetch read, each tool once, in skill order."""
    seen: set[str] = set()
    planned: list[tuple[str, str]] = []
    for item in activation.skills:
        for name in item.skill.document.prefetch:
            if name not in seen:
                seen.add(name)
                planned.append((item.skill.name, name))
    return planned


def _served_from_checks(state: AgentState, tool: str, params: dict[str, Any]) -> ToolResult | None:
    """This read answered from the order the deterministic checks read, if they read it.

    Only for the same order, and only for a read that order alone answers; anything
    else goes to the tool.
    """
    render = SNAPSHOT_READS.get(tool)
    row = state.metadata.get(ORDER_SNAPSHOT)
    if render is None or not isinstance(row, dict):
        return None
    order = Order.model_validate(row)
    return render(order) if params.get("order_id") == order.order_id else None
