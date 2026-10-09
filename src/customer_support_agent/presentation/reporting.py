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

from agent_platform.observability.step_events import (
    gatekeeper_fields as platform_gatekeeper_fields,
)
from agent_platform.observability.tool_fields import (
    display_tool_name,
    loop_control_fields,
    parse_arguments,
    tool_request_fields,
)
from agent_platform.observability.tool_fields import (
    tool_result_fields as platform_tool_result_fields,
)
from agent_platform.primitives.money import money_or_none

# Re-exported: the support agent's graph reads every card helper from here.
__all__ = [
    "FALLBACK_REPLY",
    "asks_for_the_list",
    "basic_sanity_fields",
    "loop_control_fields",
    "display_tool_name",
    "gatekeeper_fields",
    "is_customer_ready",
    "memory_fact_fields",
    "order_fields",
    "parse_arguments",
    "tool_request_fields",
    "tool_result_fields",
]


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


def gatekeeper_fields(decision: Any, call: Any) -> dict[str, Any]:
    """The platform's Gatekeeper fields, plus the order and amount the call acts on."""
    return {**platform_gatekeeper_fields(call, decision), **order_fields(call)}


def order_fields(call: Any) -> dict[str, Any]:
    """The order and amount a proposed call names: what a support operator reads first."""
    fields: dict[str, Any] = {}
    order_id = call.params.get("order_id")
    if isinstance(order_id, (str, int)) and not isinstance(order_id, bool):
        fields["order_id"] = order_id
    amount = money_or_none(call.params.get("amount"))
    if amount is not None:
        fields["amount"] = str(amount)
    return fields


# The result keys this agent's tool cards show: what a support operator reads.
TOOL_RESULT_KEYS = (
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
)


def tool_result_fields(
    call: Any,
    result: Any,
    *,
    outcome: Any | None = None,
    redact: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """The platform's tool-result fields, showing this agent's result keys."""
    return platform_tool_result_fields(
        call, result, outcome=outcome, keys=TOOL_RESULT_KEYS, redact=redact
    )


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
