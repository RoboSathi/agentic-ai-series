"""Tool calls and their results as fields a person can read — shared by every agent.

Pure formatting: a call or a result in, a dict of display fields out, with
nothing to inject and no decision to make. Which result keys an agent shows
is that agent's vocabulary, passed in by the caller.

Redaction is the caller's to supply and is applied to every text argument:
these fields reach the status feed and the log files, so an argument
rendered verbatim is a copy of the user's data in a second place.
"""

import json
from collections.abc import Callable, Iterable
from typing import Any

from agent_platform.config import get_settings
from agent_platform.loop_control import LoopControlReport
from agent_platform.primitives.money import money_or_none
from agent_platform.tools.types import WIRE_SEPARATOR

# The argument whose value is money, shown in currency rather than as typed.
_AMOUNT = "amount"


def display_tool_name(name: str) -> str:
    """The concise tool name a card shows: `answer__weather` reads `weather`."""
    normalized = name.replace(WIRE_SEPARATOR, ".")
    return normalized.rsplit(".", maxsplit=1)[-1]


def tool_request_fields(
    name: str, arguments: str, redact: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """Every argument the model sent to a tool — what it is acting on.

    All of them, not a hand-picked few: a reason, a note, a status the
    model chose are part of the decision being made. Text passes through
    the redactor first and an amount is shown as money.
    """
    fields: dict[str, Any] = {
        "tool": display_tool_name(name),
        "qualified_tool": name.replace(WIRE_SEPARATOR, "."),
    }
    try:
        raw = json.loads(arguments or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        fields["arguments"] = get_settings().observability.unparseable_arguments
        return fields
    if not isinstance(raw, dict):
        return fields
    for key, value in raw.items():
        fields[f"arg_{key}"] = _shown_argument(key, value, redact)
    return fields


def _shown_argument(key: str, value: Any, redact: Callable[[str], str] | None) -> Any:
    if key == _AMOUNT:
        parsed = money_or_none(value)
        invalid = get_settings().observability.invalid_amount
        return str(parsed) if parsed is not None else invalid.format(value=value)
    if isinstance(value, str) and redact is not None:
        return redact(value)
    return value


def tool_result_fields(
    call: Any,
    result: Any,
    *,
    outcome: Any | None = None,
    keys: Iterable[str] = (),
    redact: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """The result's outcome, what the tool returned, and the result keys the agent shows.

    `output` is the tool's whole answer, untruncated: it is what the model
    reads next, so a card without it hides the one thing the next step
    was decided on. It passes the redactor like every request argument.
    """
    fields: dict[str, Any] = {
        "tool": display_tool_name(call.action),
        "qualified_tool": call.action,
        "result": "success" if result.success else "failed",
    }
    output = getattr(result, "output", "")
    if output:
        fields["output"] = redact(output) if redact is not None else output
    if outcome is not None:
        # One field says how it ended: completed, failed, timed_out or replayed.
        fields.update(
            {"execution_status": outcome.status, "duration_ms": round(outcome.duration_ms, 2)}
        )
    for key in keys:
        value = result.data.get(key)
        if value is None:
            continue
        if key == _AMOUNT:
            parsed = money_or_none(value)
            fields[key] = str(parsed) if parsed is not None else value
        elif isinstance(value, (str, int, float, bool)):
            fields[key] = value
        elif isinstance(value, list):
            fields[key] = len(value)
    if not result.success and result.error:
        fields["error"] = result.error
    return fields


def loop_control_fields(report: LoopControlReport) -> dict[str, Any]:
    """A loop-control intervention with its evidence and limits."""
    fields: dict[str, Any] = {
        "decision": report.decision,
        "reason_code": report.reason_code,
        "reason": report.reason,
        "round": report.round_number,
        "max_rounds": report.max_rounds,
        "checks": [check.model_dump(mode="json") for check in report.checks],
    }
    # The fingerprint is only useful when diagnosing duplicate suppression.
    if report.fingerprint:
        fields["fingerprint"] = report.fingerprint
    return fields


def parse_arguments(raw: str) -> dict[str, Any]:
    """A model's tool arguments, tolerating an empty string."""
    if not raw.strip():
        return {}
    parsed: dict[str, Any] = json.loads(raw)
    return parsed


__all__ = [
    "loop_control_fields",
    "display_tool_name",
    "parse_arguments",
    "tool_request_fields",
    "tool_result_fields",
]
