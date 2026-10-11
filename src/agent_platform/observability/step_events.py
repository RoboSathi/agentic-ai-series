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
from agent_platform.observability.context_card import (
    CONTEXT_STAGE,
    announced_manifest,
    context_fields,
    context_heading,
    note_prefix,
)
from agent_platform.observability.llm_fields import (
    estimate_fields,
    llm_request_fields,
    llm_response_fields,
)
from agent_platform.observability.loop_reporter import LoopReporter
from agent_platform.observability.status_feed import StageReporter
from agent_platform.observability.tool_fields import display_tool_name, tool_request_fields
from agent_platform.observability.types import StatusKind

if TYPE_CHECKING:
    from agent_platform.gatekeeper.types import AuthorityDecision
    from agent_platform.loop_control.types import LoopVerdict
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
    {CONTEXT_STAGE, LLM_REQUEST, "llm_call", LLM_RESPONSE, TOOL_REQUEST, GATEKEEPER, EXECUTOR, MCP_CALL,
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
_FAIL_MARK = "✗"


def _produced(purpose: str, asked_for_tools: bool) -> str:
    """A next-action call's purpose as what it produced; any other purpose as it was."""
    words = get_settings().observability.llm_purposes
    if purpose != words.next_action:
        return purpose
    return words.tool_selection if asked_for_tools else words.reply


class StepEvents:
    """Emits a tool-use loop's cards, each numbered by the step it belongs to."""

    def __init__(self, reporter: StageReporter | None, *, max_steps: int, model: str = "") -> None:
        self._loop = LoopReporter(reporter, max_steps=max_steps, stages=LOOP_STAGES)
        self._model = model
        self._max_steps = max_steps
        # The last request's own size estimate, to set beside what the provider counted.
        self._estimated_input = 0

    @property
    def step(self) -> int | None:
        return self._loop.step

    def begin(self, step: int | None) -> None:
        """Number every loop card that follows with `step` (None: unnumbered)."""
        self._loop.step = step

    def llm_request(self, request: Any, *, purpose: str) -> None:
        """CONTEXT, then LLM REQUEST: what the model is handed this step, section by section."""
        self.context(request, purpose=purpose)
        heading = get_settings().observability.llm_request_heading.format(purpose=purpose)
        fields = {**llm_request_fields(request, model=self._model), "purpose": purpose}
        self._loop.emit(LLM_REQUEST, summary=heading, fields=fields, event_kind="model")

    def context(self, request: Any, *, purpose: str) -> None:
        """CONTEXT: the request's sections and their tokens, once per request."""
        manifest = announced_manifest(request, purpose)
        if manifest is None:
            return
        note_prefix(manifest, self._model)
        self._estimated_input = manifest.total_tokens
        self._loop.emit(
            CONTEXT_STAGE,
            summary=context_heading(manifest, self._model, purpose),
            fields=context_fields(manifest, self._model),
            event_kind="narrative",
        )

    def llm_response(self, result: Any, *, purpose: str) -> None:
        """LLM RESPONSE: the tool calls it asked for, or its answer.

        A loop step's call is made to choose the next action, so its response is named by what
        it chose: a tool, or the reply. The request could not know which.
        """
        words = get_settings().observability
        purpose = _produced(purpose, bool(result.tool_calls))
        heading = (
            words.llm_tool_requested_heading.format(
                tool=display_tool_name(result.tool_calls[0].name)
            )
            if result.tool_calls
            else words.llm_response_heading.format(purpose=purpose)
        )
        fields = {
            **llm_response_fields(result),
            **estimate_fields(self._estimated_input, result.usage.prompt_tokens),
            "purpose": purpose,
        }
        self._loop.emit(LLM_RESPONSE, summary=heading, fields=fields, event_kind="model")

    def tool_request(
        self, name: str, arguments: str, *, redact: Any = None, prefetched_for: str = ""
    ) -> None:
        """TOOL REQUEST: every argument sent — by the model, or by a skill's prefetch."""
        words, tool = self._words(), display_tool_name(name)
        heading = (
            words.tool_prefetched.format(tool=tool, skill=prefetched_for)
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
        skill: str,
        source: str,
        redact: Any = None,
    ) -> None:
        """TOOL RESULT for a prefetch read answered from a row an earlier stage already read.

        No tool ran, so there is no request, verdict or execution to draw: one card says
        what the model is handed, for which skill, and which stage's read it reuses.
        """
        heading = self._words().tool_served.format(
            tool=display_tool_name(name), skill=skill, source=source
        )
        fields = tool_request_fields(name, arguments, redact)
        fields.update(
            {
                "prefetched_for": skill,
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

    def loop_verdict(
        self,
        verdict: "LoopVerdict",
        *,
        next_step: str,
        because: str,
        criteria: Mapping[str, Any] | None = None,
        because_words: Mapping[str, str] | None = None,
        counter_names: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """LOOP CONTROL, one schema for every agent, and only what says something.

        `because` is why the loop chose its next step (the model asked for a tool, the tools
        ran). `intervened` appears only when a governor mechanism changed the course — which
        one, and its reason — never "no check objected". Failing checks only; every budget as
        it stands. A warning when a mechanism intervened, a plain card otherwise. Returns the
        fields, so the caller logs exactly what the card shows. `counter_names` words an agent's
        counters as its reader knows them ("pages read" for `fetches`).
        """
        names = counter_names or {}
        reasons = {**self._words().loop_control.because, **(because_words or {})}
        fields: dict[str, Any] = {
            "decision": verdict.decision,
            "next_step": next_step,
            "because": reasons.get(because, because),
        }
        if verdict.intervened:
            words = get_settings().loop_control.card
            fields["intervened"] = words.intervention.format(
                layer=verdict.layer or verdict.reason_code, reason=_spent_reason(verdict, names)
            )
            if verdict.fingerprint:
                fields["fingerprint"] = verdict.fingerprint
        if criteria:
            fields["criteria"] = dict(criteria)
        fields["budgets"] = budget_lines(verdict, names)
        # Other failing checks only (a flag the verdict did not act on, such as the historical
        # baseline): the deciding one is already the `intervened` line.
        line = get_settings().loop_control.card.check_line
        flags = [
            line.format(mark=_FAIL_MARK, name=check.name, reason=check.reason)
            for check in verdict.checks
            if check.status != "PASS" and check.reason_code != verdict.reason_code
        ]
        if flags:
            fields["flags"] = flags
        status: StatusKind = "warning" if verdict.intervened else "completed"
        self._loop.emit(
            LOOP_CONTROL,
            status,
            summary=loop_heading(verdict, next_step),
            fields=fields,
            event_kind="loop_control",
        )
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


def loop_heading(verdict: "LoopVerdict", next_step: str = "") -> str:
    """`Step 3 of 5 · SUPPRESS · duplicate_action · next: propose`."""
    words = get_settings().loop_control.card
    layer = words.layer_suffix.format(layer=verdict.layer) if verdict.layer else ""
    return words.summary.format(
        step=verdict.round_number,
        max=verdict.max_rounds,
        decision=verdict.decision.upper(),
        layer=layer,
        next_step=next_step,
    )


def budget_lines(verdict: "LoopVerdict", names: Mapping[str, str] | None = None) -> list[str]:
    """Each counter as `tokens: 4,210 of 30,000`."""
    return [
        _budget_line(name, budget.spent, budget.cap, names or {})
        for name, budget in verdict.budgets.items()
    ]


def _budget_line(counter: str, spent: float, cap: float, names: Mapping[str, str]) -> str:
    line = get_settings().loop_control.card.budget_line
    return line.format(
        counter=names.get(counter, counter), spent=f"{spent:,.6g}", cap=f"{cap:,.6g}"
    )


def _spent_reason(verdict: "LoopVerdict", names: Mapping[str, str]) -> str:
    """Why a budget stopped the loop, as its budget line in the agent's own words (`planned
    rounds: 1 of 1`) when the agent names that counter; any other reason as it was given."""
    deciding = next((c for c in verdict.checks if c.reason_code == verdict.reason_code), None)
    observed = deciding.observed if deciding is not None else {}
    if {"counter", "spent", "cap"} <= observed.keys() and observed["counter"] in names:
        return _budget_line(observed["counter"], observed["spent"], observed["cap"], names)
    return verdict.reason


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
