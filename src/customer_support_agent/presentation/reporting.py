"""Turning what the harness did into fields a person can read.

Pure formatting: every function here takes a result and returns a dict of
display fields, with nothing to inject and no decision to make. It lived
at the bottom of the graph module, where ~230 lines of string handling sat
between the reader and the actual control flow.

Redaction is not optional in here. These fields reach the status feed and
the log files, so a tool argument rendered verbatim is a copy of the
customer's data in a second place.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from agentic_commons.control_loop import ControlLoopReport
from agentic_commons.money import money_or_none


def tool_request_fields(
    name: str, arguments: str, redact: Callable[[str], str] | None = None
) -> dict[str, Any]:
    """Every argument the model sent to a tool — what it is acting on.

    All of them, not a hand-picked few: a reason, a note, a status the
    model chose are part of the decision being made. Text passes through
    the redactor first (the card must not become a copy of personal data)
    and an amount is shown in dollars.
    """
    fields: dict[str, Any] = {
        "tool": display_tool_name(name),
        "qualified_tool": name.replace("__", "."),
    }
    try:
        raw = json.loads(arguments or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        fields["arguments"] = "unparseable — the call will be refused"
        return fields
    if not isinstance(raw, dict):
        return fields
    for key, value in raw.items():
        fields[f"arg_{key}"] = _shown_argument(key, value, redact)
    return fields


def _shown_argument(key: str, value: Any, redact: Callable[[str], str] | None) -> Any:
    if key == "amount":
        parsed = money_or_none(value)
        return str(parsed) if parsed is not None else f"{value} (not a valid amount)"
    if isinstance(value, str) and redact is not None:
        return redact(value)
    return value


def memory_fact_fields(fact: Any, *, score: float | None = None) -> dict[str, Any]:
    """Render one memory fact with its provenance and optional retrieval score."""
    fields: dict[str, Any] = {
        "key": fact.key,
        "value": fact.value,
        "memory_type": fact.memory_type,
        "category": fact.category,
        "source": fact.provenance.source,
        "provenance": fact.provenance.kind,
        "confidence": fact.confidence,
        "importance": fact.importance,
    }
    if score is not None:
        fields["score"] = round(float(score), 4)
    return fields


# Backward-compatible spelling for callers from the first Phase 1 integration.
# The canonical helper name is `memory_fact_fields` (singular `fact`).
memory_facts_field = memory_fact_fields


def basic_sanity_fields(result: Any) -> dict[str, Any]:
    """Render BasicSanity findings consistently with the TUI check contract."""
    fields: dict[str, Any] = {
        "passed": result.passed,
        "reason_code": result.reason_code,
        "checks": [],
    }
    facts = result.facts if isinstance(result.facts, dict) else {}
    for key in (
        "order_id",
        "customer_id",
        "order_status",
        "refund_status",
        "amount",
        "requested_amount",
        "already_refunded",
        "refund_eligible",
        "operation",
        "order_id_present",
        "lookup_attempted",
    ):
        if key in facts:
            fields[key] = facts[key]
    for finding in result.findings:
        evidence = finding.evidence if isinstance(finding.evidence, dict) else {}
        status = "SKIPPED" if not finding.applicable else "PASS" if finding.passed else "FAIL"
        fields["checks"].append(
            {
                "name": finding.check.replace("_", " ").title(),
                "status": status,
                "reason_code": (
                    f"BASIC_SANITY_{finding.check.upper()}_DEFERRED"
                    if not finding.applicable
                    else f"BASIC_SANITY_{finding.check.upper()}_PASS"
                    if finding.passed
                    else result.reason_code
                ),
                "reason": finding.reason,
                "observed": evidence,
            }
        )
    if not fields["checks"]:
        fields["checks"].append(
            {
                "name": "Basic sanity",
                "status": "PASS" if result.passed else "FAIL",
                "reason_code": result.reason_code,
                "reason": result.message or result.reason_code,
                "observed": facts,
            }
        )
    return fields


def control_loop_fields(report: ControlLoopReport) -> dict[str, Any]:
    """Render a control-loop intervention with its evidence and limits."""
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


def gatekeeper_fields(decision: Any, call: Any) -> dict[str, Any]:
    """Expose the exact proposal and deterministic checks behind the ruling."""
    fields: dict[str, Any] = {
        "decision": decision.verdict,
        "action": display_tool_name(decision.action),
        "effect": "write" if call.side_effecting else "read",
        "reason_code": decision.reason_code,
        "question": decision.question,
        "checks": [check.model_dump(mode="json") for check in decision.checks],
    }
    order_id = call.params.get("order_id")
    if isinstance(order_id, (str, int)) and not isinstance(order_id, bool):
        fields["order_id"] = order_id
    amount = money_or_none(call.params.get("amount"))
    if amount is not None:
        fields["amount"] = str(amount)

    for check in decision.checks:
        if check.check_id == "gatekeeper.actor_authorization":
            permission = check.observed.get("required_permission")
            role = check.observed.get("role")
            if permission:
                fields["permission"] = permission
            if role:
                fields["role"] = role
            break

    return fields


def display_tool_name(name: str) -> str:
    """Return the concise tool name used in the operator-facing dashboard."""
    normalized = name.replace("__", ".")
    return normalized.rsplit(".", maxsplit=1)[-1]


def tool_result_fields(call: Any, result: Any, *, outcome: Any | None = None) -> dict[str, Any]:
    """Expose domain outcome plus generic Executor lifecycle metadata."""
    fields: dict[str, Any] = {
        "tool": display_tool_name(call.action),
        "qualified_tool": call.action,
        "result": "success" if result.success else "failed",
    }
    if outcome is not None:
        fields.update(
            {
                # One field says how it ended: completed, failed, timed_out or replayed.
                "execution_status": outcome.status,
                "duration_ms": round(outcome.duration_ms, 2),
            }
        )
    for key in (
        "order_id",
        "amount",
        "order_status",
        "refund_status",
        "pending_review",
        "queued",
        "already_queued",
        "request_id",
        "refund_id",
        "steps",
        "failed_step",
        "compensated",
    ):
        value = result.data.get(key)
        if value is None:
            continue
        if key == "amount":
            parsed = money_or_none(value)
            fields[key] = str(parsed) if parsed is not None else value
        elif isinstance(value, (str, int, float, bool)):
            fields[key] = value
        elif isinstance(value, list):
            fields[key] = len(value)
    if not result.success and result.error:
        fields["error"] = result.error
    return fields


# Shown when the model produces nothing a customer could read.
# Deliberately free of operational promises nothing will keep.
FALLBACK_REPLY = "Sorry — I wasn't able to complete that. Please try again."


# A customer asking to see the list again. Deterministic, like the list
# itself — re-reading a table is not a task for a model.
_LIST_REQUEST = re.compile(
    r"\b(my orders|all my orders|list (my )?orders|show (me )?(my )?orders|"
    r"what orders|which orders|see (my )?orders|orders again|the list)\b",
    re.IGNORECASE,
)


def asks_for_the_list(message: str) -> bool:
    """Whether the customer is asking to see their order list again."""
    return bool(_LIST_REQUEST.search(message))


def is_customer_ready(content: str) -> bool:
    """Whether this text can be shown to a customer as an answer.

    A weak model sometimes emits a tool call as plain *text* instead of a
    structured call — `{"name": "get_order_status", "parameters": {...}}`.
    That is machine output that failed to become a tool call, and handing
    it to a customer is worse than saying nothing. It is caught here by
    shape rather than by pattern-matching any one model's phrasing.
    """
    text = content.strip()
    if not text:
        return False
    if not (text.startswith(("{", "[")) and text.endswith(("}", "]"))):
        return True
    try:
        json.loads(text)
    except ValueError:
        # Not valid JSON, so it is prose that merely contains braces.
        return True
    # It parsed. Whatever its keys, a bare JSON document is machine
    # output that failed to become a tool call — checking for particular
    # key names only catches the spellings you thought of, and the live
    # failures used three different ones.
    return False


def parse_arguments(raw: str) -> dict[str, Any]:
    """Parse a model's tool arguments, tolerating an empty string."""
    if not raw.strip():
        return {}
    parsed: dict[str, Any] = json.loads(raw)
    return parsed
