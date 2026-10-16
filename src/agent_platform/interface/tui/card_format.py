"""How a card writes a field's name and value: shared by the card body and its summary line.

Kept apart from the widgets so the summary writers (`card_summaries.py`) and the card body
read one vocabulary — the same label, the same money format, the same field order.
"""

from collections.abc import Mapping
from typing import Any

from agent_platform.config import get_settings
from agent_platform.core.primitives.money import money_or_none

# Where an agent's own subject fields sit in the card order.
SUBJECT_SLOT = "__subject_fields__"

# Common field ordering for expanded cards.  Any other safe semantic fields
# follow alphabetically.
FIELD_PRIORITY = (
    # A model card reads in the order the exchange happened: what the
    # customer asked, what we sent, what came back. A reasoning loop's card
    # leads with what the model thought, then what it decided.
    "loop_step",
    "model",
    "reasoning",
    "tool_calls",
    "observation",
    "output",
    "user_query",
    "system_prompt",
    "messages",
    "response",
    "decision",
    "standing",
    "action",
    "effect",
    "qualified_action",
    "reason_code",
    "question",
    "intent",
    "topic",
    # An agent's subject fields (`agent.subject_fields`) are ranked here.
    SUBJECT_SLOT,
    "amount",
    "requested_amount",
    "allowed_amount",
    "autonomous_limit",
    "remaining_before",
    "remaining_after",
    "policy_version",
    "error_code",
    "http_status",
    "reason",
    "actor",
    "roles",
    "permission",
    "said",
    "for",
    "by",
    "review",
    "read_by",
    "granted_by",
    "checks",
    "reason",
    "confidence",
    "model_score",
    "score_type",
    "source",
    "candidates",
    "raw_model_labels",
    "matched_phrase",
    "match_method",
    "tool",
    "qualified_tool",
    "result",
    "status",
    "verified",
    "method",
    "count",
    "facts_held",
    "operation",
    "scope",
    "written",
    "route",
    "chunks",
    "selected",
    "dropped",
    "budget_tokens",
    "used_tokens",
    "utilization",
    "health",
    "model",
    "finish_reason",
    "tool_calls",
    "requested_tools",
    "requested_tool_calls",
    "tools_available",
    "tools_offered",
    # The skill card: what was chosen and why, then what the harness read, then what
    # the model may call.
    "skills",
    "harness_reads_first",
    "model_may_call",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "round",
    "max_rounds",
    "previous_result",
    "fingerprint",
    "next_stage",
    "claims",
    "supported",
    "unsupported",
    "evidence_searched",
    "provenance_checks",
    "claim_checks",
    "unsupported_claims",
    "unsupported_diagnostics",
    "detector_model",
    "retrieval_method",
    "retrieval_top_k",
    "best_evidence_overlap",
    "hallucination_rate",
    "response_ready",
    "revised",
    "comment",
    "grounding_issues",
    "fallback",
    "schema_valid",
    "validation_mode",
    "validation_attempts",
    "repair_attempts",
    "enforcement",
    "grounded",
    "escalated",
    "failure_kinds",
    "nollm_resolved_ratio",
    "tier_counts",
    "reason_code",
    "hallucination_scope",
    "advisory_flagged",
    "advisory_reason",
)

FIELD_LABELS = {
    # A tool result's `result` says the call ran, not what it answered.
    "result": "call",
    "policy_version": "policy",
    "requested_amount": "requested",
    "allowed_amount": "allowed",
    "autonomous_limit": "autonomous_limit",
    "budget_tokens": "budget_tokens",
    "used_tokens": "used_tokens",
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "total_tokens": "total_tokens",
    "comment": "comment",
    "schema_valid": "reply_matches_json_schema",
}

AMOUNT_FIELDS = {
    "amount",
    "requested_amount",
    "allowed_amount",
    "autonomous_limit",
}


def field_priority() -> tuple[str, ...]:
    """The card field order, with this agent's subject fields in their slot."""
    subjects = tuple(subject.field for subject in get_settings().agent.subject_fields)
    index = FIELD_PRIORITY.index(SUBJECT_SLOT)
    return FIELD_PRIORITY[:index] + subjects + FIELD_PRIORITY[index + 1 :]


def label(name: str) -> str:
    """A field's name as a card shows it: an agent's subject label, else the platform's."""
    for subject in get_settings().agent.subject_fields:
        if subject.field == name:
            return subject.label
    return FIELD_LABELS.get(name, name)


def first_present(
    fields: Mapping[str, Any],
    names: set[str],
) -> Any | None:
    for name in field_priority():
        if name in names and name in fields:
            return fields[name]
    return None


def format_value(
    name: str,
    value: Any,
) -> str:
    if value is None:
        return "None"

    if name in AMOUNT_FIELDS:
        amount = money_or_none(value)
        if amount is not None:
            return str(amount)

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(str(item) for item in value) + "]"

    if isinstance(value, dict):
        items = ", ".join(f"{key}={format_value(str(key), item)}" for key, item in value.items())
        return "{" + items + "}"

    return " ".join(str(value).split())


def humanize(value: str) -> str:
    return " ".join(value.replace("_", " ").split())
