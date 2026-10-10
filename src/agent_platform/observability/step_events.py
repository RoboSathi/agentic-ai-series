"""The cards one step of a tool-use loop emits — built once, the same for every agent.

A step is a model call, the tools it asked for, the Gatekeeper's ruling on
each, what each returned, and the loop's decision to go round again. Every
agent with such a loop draws those cards, so their fields, headings and step
numbers come from here rather than from each agent's own helper: a card the
operator learned to read on one agent reads the same on the next.

What stays the agent's is what only it knows: which result keys a support
operator cares about, a result's preview, the reasons only its loop has for
stopping. Those arrive as `extra` fields or wording, never as a second builder.

The TUI renders these; it never builds them (it is a consumer of the feed).
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from agent_platform.config import get_settings
from agent_platform.observability.llm_fields import llm_request_fields, llm_response_fields
from agent_platform.observability.loop_reporter import LoopReporter
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.tool_fields import display_tool_name, tool_request_fields
from agent_platform.observability.types import StatusKind

if TYPE_CHECKING:
    from agent_platform.gatekeeper.types import AuthorityDecision
    from agent_platform.tools.types import ToolCall, ToolResult

LLM_REQUEST = "llm_request"
LLM_RESPONSE = "llm_response_received"
TOOL_REQUEST = "tool_request"
GATEKEEPER = "gatekeeper_evaluation"
EXECUTOR = "executor"
MCP_CALL = "mcp_call"
TOOL_RESULT = "tool_result"
LOOP_CONTROL = "loop_control"
HALLUCINATION_CHECK = "hallucination_check"
# The cards that belong to one iteration of the loop, and so carry its step number.
LOOP_STAGES = frozenset(
    {LLM_REQUEST, "llm_call", LLM_RESPONSE, TOOL_REQUEST, GATEKEEPER, EXECUTOR, MCP_CALL,
     TOOL_RESULT, LOOP_CONTROL}
)  # fmt: skip
# Where a tool result says it crossed a protocol (`ToolResult.exchange`), and which.
EXCHANGE_PROTOCOL = "protocol"
MCP_PROTOCOL = "mcp"
_ESCALATE = "escalate"
# The Gatekeeper check whose evidence names the permission and the actor's role.
_ACTOR_CHECK = "gatekeeper.actor_authorization"
# Card field ← that check's observed key.
_ACTOR_FIELDS = {"permission": "required_permission", "role": "role"}
_WRITE, _READ = "write", "read"


class StepEvents:
    """Emits a tool-use loop's cards, each numbered by the step it belongs to."""

    def __init__(self, reporter: StageReporter | None, *, max_steps: int, model: str = "") -> None:
        self._loop = LoopReporter(reporter, max_steps=max_steps, stages=LOOP_STAGES)
        self._model = model
        self._max_steps = max_steps

    @property
    def step(self) -> int | None:
        return self._loop.step

    def begin(self, step: int | None) -> None:
        """Number every loop card that follows with `step` (None: unnumbered)."""
        self._loop.step = step

    def llm_request(self, request: Any, *, purpose: str) -> None:
        """LLM REQUEST: what the model is handed this step."""
        heading = get_settings().observability.llm_request_heading.format(purpose=purpose)
        fields = {**llm_request_fields(request, model=self._model), "purpose": purpose}
        self._loop.emit(LLM_REQUEST, summary=heading, fields=fields, event_kind="model")

    def llm_response(self, result: Any, *, purpose: str) -> None:
        """LLM RESPONSE: the tool calls it asked for, or its answer."""
        words = get_settings().observability
        heading = (
            words.llm_tool_requested_heading.format(
                tool=display_tool_name(result.tool_calls[0].name)
            )
            if result.tool_calls
            else words.llm_response_heading.format(purpose=purpose)
        )
        fields = {**llm_response_fields(result), "purpose": purpose}
        self._loop.emit(LLM_RESPONSE, summary=heading, fields=fields, event_kind="model")

    def tool_request(
        self, name: str, arguments: str, *, redact: Any = None, prefetched_for: str = ""
    ) -> None:
        """TOOL REQUEST: every argument sent — by the model, or by a procedure's prefetch."""
        words, tool = self._words(), display_tool_name(name)
        heading = (
            words.tool_prefetched.format(tool=tool, procedure=prefetched_for)
            if prefetched_for
            else words.tool_requested.format(tool=tool)
        )
        fields = tool_request_fields(name, arguments, redact)
        if prefetched_for:
            fields["prefetched_for"] = prefetched_for
        self._loop.emit(TOOL_REQUEST, summary=heading, fields=fields, event_kind="tool_request")

    def served_read(
        self,
        name: str,
        arguments: str,
        output: str,
        *,
        procedure: str,
        source: str,
        redact: Any = None,
    ) -> None:
        """TOOL RESULT for a prefetch read answered from a row an earlier stage already read.

        No tool ran, so there is no request, verdict or execution to draw: one card says
        what the model is handed, for which procedure, and which stage's read it reuses.
        """
        heading = self._words().tool_served.format(
            tool=display_tool_name(name), procedure=procedure, source=source
        )
        fields = tool_request_fields(name, arguments, redact)
        fields.update(
            {
                "prefetched_for": procedure,
                "served_from": source,
                "output": redact(output) if redact is not None else output,
            }
        )
        self._loop.emit(
            TOOL_RESULT, "completed", summary=heading, fields=fields, event_kind="tool_result"
        )

    def gatekeeper(
        self,
        call: "ToolCall",
        decision: "AuthorityDecision",
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        """GATEKEEPER: allowed, escalated to a person (a caution), or refused."""
        status: StatusKind = (
            "completed"
            if decision.allowed
            else "warning"
            if decision.verdict == _ESCALATE
            else "blocked"
        )
        heading = self._words().gatekeeper.format(
            verdict=decision.verdict.upper(), standing=decision.standing
        )
        fields = {**gatekeeper_fields(call, decision), **(extra or {})}
        self._loop.emit(GATEKEEPER, status, summary=heading, fields=fields, event_kind="gatekeeper")

    def tool_result(
        self,
        status: StatusKind,
        summary: str,
        fields: Mapping[str, Any],
        *,
        result: "ToolResult | None" = None,
    ) -> None:
        """TOOL RESULT, preceded by MCP CALL when the result crossed the protocol."""
        if result is not None and result.exchange.get(EXCHANGE_PROTOCOL) == MCP_PROTOCOL:
            self.mcp_call(result.exchange)
        self._loop.emit(
            TOOL_RESULT, status, summary=summary, fields=fields, event_kind="tool_result"
        )

    def mcp_call(self, exchange: Mapping[str, Any]) -> None:
        """MCP CALL: the protocol's side of one call — server, method, how long, whether it failed."""
        words = get_settings().mcp.card
        failed = bool(exchange.get("is_error"))
        heading = (words.call_failed_heading if failed else words.call_heading).format(**exchange)
        fields = {key: value for key, value in exchange.items() if key != EXCHANGE_PROTOCOL}
        status: StatusKind = "failed" if failed else "completed"
        self._loop.emit(MCP_CALL, status, summary=heading, fields=fields, event_kind="tool_result")

    def loop_control(
        self,
        *,
        round_number: int,
        criteria: Mapping[str, Any],
        next_step: str,
        because: str,
        because_words: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """LOOP CONTROL: every criterion the loop weighs, its value now, and the step it chose.

        Returns the fields, so the caller logs exactly what the card shows.
        """
        words = self._words().loop_control
        reasons = {**words.because, **(because_words or {})}
        fields: dict[str, Any] = {
            "criteria": dict(criteria),
            "next_step": next_step,
            "because": reasons.get(because, because),
        }
        heading = words.summary.format(
            round=round_number, max_rounds=self._max_steps, next_step=next_step
        )
        self._loop.emit(LOOP_CONTROL, summary=heading, fields=fields, event_kind="loop_control")
        return fields

    def hallucination_check(
        self, status: StatusKind, summary: str, fields: Mapping[str, Any]
    ) -> None:
        """HALLUCINATION CHECK: the levels first (token confidence), then the agent's own."""
        self._loop.emit(
            HALLUCINATION_CHECK, status, summary=summary, fields=fields, event_kind="grounding"
        )

    @staticmethod
    def _words() -> Any:
        return get_settings().observability.step_events


def gatekeeper_fields(call: "ToolCall", decision: "AuthorityDecision") -> dict[str, Any]:
    """The ruling and the deterministic checks behind it, as every agent's GATEKEEPER card shows."""
    fields: dict[str, Any] = {
        "decision": decision.verdict,
        "standing": decision.standing,
        "action": display_tool_name(decision.action),
        "effect": _WRITE if call.side_effecting else _READ,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "question": decision.question,
        "checks": [check.model_dump(mode="json") for check in decision.checks],
    }
    actor = next((check for check in decision.checks if check.check_id == _ACTOR_CHECK), None)
    if actor is not None:
        fields.update(
            {field: actor.observed[key] for field, key in _ACTOR_FIELDS.items()
             if actor.observed.get(key)}
        )  # fmt: skip
    return fields


__all__ = [
    "EXCHANGE_PROTOCOL",
    "LOOP_STAGES",
    "MCP_PROTOCOL",
    "StepEvents",
    "gatekeeper_fields",
]
