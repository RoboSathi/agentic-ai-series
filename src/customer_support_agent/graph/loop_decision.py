"""Where the tool-use loop goes next, and why — one decision for the route and its card.

The graph's routing functions return the route; the nodes report it. Both
read these functions, so the CONTROL LOOP card can never describe a step
the graph did not take. Pure: no state is read here that is not passed in.
"""

from typing import Any

# The steps the tool-use loop can route to next.
ACT = "act"
REASON = "reason"
FINALIZE = "finalize"
# The `because` keys, whose wording is `customer_support_agent.control_loop_status.because`.
TOOL_REQUESTED = "tool_requested"
NO_TOOL_CALL = "no_tool_call"
ROUNDS_REMAIN = "rounds_remain"
ESCALATED = "escalated"
FINAL_REFUSAL = "final_refusal"
ITERATION_LIMIT = "iteration_limit"

_DUPLICATE_ACTION = "DUPLICATE_ACTION"


def after_reason_step(tools_requested: bool) -> tuple[str, str]:
    """Act on the tools the model asked for, or finish with its reply."""
    return (ACT, TOOL_REQUESTED) if tools_requested else (FINALIZE, NO_TOOL_CALL)


def after_act_step(
    tool_rounds: int, max_rounds: int, escalated: bool, refusal_final: bool
) -> tuple[str, str]:
    """Go round again only while autonomous execution remains permitted.

    An escalation hands the turn to a person, and a refusal no other
    arguments can turn into an allow will not move: neither sends the
    model back to try again. Otherwise the iteration limit decides.
    """
    if escalated:
        return FINALIZE, ESCALATED
    if refusal_final:
        return FINALIZE, FINAL_REFUSAL
    if tool_rounds + 1 <= max_rounds:
        return REASON, ROUNDS_REMAIN
    return FINALIZE, ITERATION_LIMIT


def duplicate_suppressed(invocations: list[Any]) -> bool:
    """Whether the last round's control check suppressed a repeated identical call."""
    return any(
        isinstance(item, dict)
        and isinstance(item.get("control_loop"), dict)
        and item["control_loop"].get("reason_code") == _DUPLICATE_ACTION
        for item in invocations
    )
