"""Where the tool-use loop goes next, and why — one decision for the route and its card.

The platform's loop governor decides whether the loop may take another step at all (the
step cap, the deadline, the budgets, a cycle, a plateau) and whether an action runs (a
repeat, by any tier): its verdict is in `state.loop.verdict`. What only this agent knows —
an escalation hands the turn to a person, a refusal no arguments can turn into an allow —
is decided here. Routers and cards both read these functions, so the LOOP CONTROL card can
never describe a step the graph did not take. Pure: nothing is read that is not passed in.
"""

from agent_platform.loop_control import LoopDecision, LoopVerdict, route_on_verdict

# The steps the tool-use loop can route to next.
EXECUTE = "execute"
PROPOSE = "propose"
FINALIZE = "finalize"
# The `because` keys, whose wording is `observability.step_events.loop_control.because`
# (the platform's), plus `customer_support_agent.loop_control_status.because` (this agent's).
TOOL_REQUESTED = "tool_requested"
NO_TOOL_CALL = "no_tool_call"
ROUNDS_REMAIN = "rounds_remain"
ESCALATED = "escalated"
FINAL_REFUSAL = "final_refusal"
# A step the governor would not allow: the verdict's own reason code says why.
STOPPED: dict[LoopDecision, str] = {"finalize": FINALIZE, "abort": FINALIZE, "pause": FINALIZE}


def after_propose_step(verdict: LoopVerdict | None, tools_requested: bool) -> tuple[str, str]:
    """Finish when the governor stopped the step; else act on the tools asked for, or reply."""
    stopped = route_on_verdict(verdict, STOPPED, "")
    if stopped and verdict is not None:
        return stopped, verdict.reason_code
    return (EXECUTE, TOOL_REQUESTED) if tools_requested else (FINALIZE, NO_TOOL_CALL)


def after_execute_step(escalated: bool, refusal_final: bool) -> tuple[str, str]:
    """Go round again unless a person now owns the turn or the refusal is final.

    Whether another step is affordable is the governor's, asked at the next `propose`.
    """
    if escalated:
        return FINALIZE, ESCALATED
    if refusal_final:
        return FINALIZE, FINAL_REFUSAL
    return PROPOSE, ROUNDS_REMAIN
