"""The one line a card's title carries: what its stage decided, at a glance.

A status that already says it all (failed, blocked, waiting, a narrative the emitter wrote)
is taken first. Otherwise each stage has its own small writer, found by name. A writer that
has nothing to say for this event returns None, and the card falls back to the event's own
summary, then its detail. The words are configuration (`tui.card_summaries`); this module only
decides which ones apply.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import CardSummaryWords
from agent_platform.interface.tui.card_format import (
    AMOUNT_FIELDS,
    first_present,
    format_value,
    humanize,
    label,
)
from agent_platform.interface.tui.labels import StageLabelRegistry
from agent_platform.observability.step_events import subject_part
from agent_platform.observability.types import StatusEvent

# A gatekeeper check that stopped or escalated the action names the reason on the card.
_STOPPING_CHECKS = frozenset({"FAIL", "ESCALATE"})


@dataclass(frozen=True)
class SummaryInput:
    """One event, as every summary writer reads it."""

    event: StatusEvent
    labels: StageLabelRegistry
    words: CardSummaryWords

    @property
    def fields(self) -> Mapping[str, Any]:
        return self.event.fields

    def join(self, parts: list[str]) -> str:
        """The parts that say something, in order, on one line."""
        return self.words.separator.join(part for part in parts if part)


Summariser = Callable[[SummaryInput], str | None]


def summarise(event: StatusEvent, labels: StageLabelRegistry) -> str:
    """The card's summary line for one event."""
    given = SummaryInput(event=event, labels=labels, words=get_settings().tui.card_summaries)
    by_status = _by_status(given)
    if by_status is not None:
        return by_status
    writer = SUMMARISERS.get(event.stage)
    written = writer(given) if writer is not None else None
    return written if written is not None else _fallback(event)


def decision_summary(fields: Mapping[str, Any], labels: StageLabelRegistry) -> str:
    """A blocked stage's decision and action: DENY · refund."""
    words = get_settings().tui.card_summaries
    decision = fields.get("decision") or fields.get("verdict")
    action = fields.get("action")
    if not decision and not action:
        return words.blocked
    parts = [
        str(decision).upper() if decision else "",
        labels.tool_label(str(action)) if action else "",
    ]
    return words.separator.join(part for part in parts if part)


def compact_fields(fields: Mapping[str, Any], names: tuple[str, ...]) -> str:
    """`label=value` for each named field present, on one line."""
    words = get_settings().tui.card_summaries
    return words.separator.join(
        f"{label(name)}={format_value(name, fields[name])}"
        for name in names
        if fields.get(name) is not None
    )


# -- by status ------------------------------------------------------------------------------


def _by_status(given: SummaryInput) -> str | None:
    event, words = given.event, given.words
    if event.event_kind == "narrative":
        return event.summary or ""
    by_status = {
        "failed": lambda: event.summary or words.failed,
        "blocked": lambda: event.summary or decision_summary(given.fields, given.labels),
        "warning": lambda: event.summary or words.warning,
        "waiting": lambda: event.summary or words.waiting,
    }
    chosen = by_status.get(event.status)
    return chosen() if chosen is not None else None


def _fallback(event: StatusEvent) -> str:
    if event.summary:
        return event.summary
    return humanize(event.detail) if event.detail else ""


# -- the front door -------------------------------------------------------------------------


def _scope_check(given: SummaryInput) -> str | None:
    topic = given.fields.get("topic")
    if given.fields.get("in_scope") is True:
        return str(topic or given.words.in_scope)
    return given.join([given.words.out_of_scope, str(topic)]) if topic else None


def _authentication(given: SummaryInput) -> str:
    words = given.words
    return words.identity_verified if given.fields.get("verified") else words.verification_required


def _deterministic_checks(given: SummaryInput) -> str:
    reasons = get_settings().tui.reason_summaries
    summary = reasons.get(str(given.fields.get("reason_code")))
    if summary:
        return summary
    words = given.words
    return words.valid_request if given.fields.get("passed") else words.invalid_request


def _prompt_injection(given: SummaryInput) -> str:
    words = given.words
    return words.injection_blocked if given.fields.get("detected") else words.injection_passed


def _pii_filtering(given: SummaryInput) -> str:
    words, fields = given.words, given.fields
    if not fields.get("detected"):
        return words.pii_passed
    categories = (
        ", ".join(str(item) for item in fields.get("categories") or []) or words.pii_category
    )
    matches = fields.get("matches")
    counted = (
        (words.pii_match if matches == 1 else words.pii_matches).format(count=matches)
        if matches is not None
        else ""
    )
    return given.join([words.pii_redacted, categories, counted])


def _content_moderation(given: SummaryInput) -> str:
    # A blocked screen was summarised by its status, with its decision.
    return given.words.moderation_passed


def _intent_classification(given: SummaryInput) -> str:
    fields = given.fields
    intent = fields.get("intent") or given.event.detail or given.words.intent_classified
    score = fields.get("model_score")
    if score is None:
        return str(intent)
    scored = given.words.intent_score.format(score=float(score))
    return given.join([str(intent), scored, str(fields.get("disposition") or "")])


# -- memory and context ---------------------------------------------------------------------


def _working_memory_write(given: SummaryInput) -> str:
    fields = given.fields
    operation = str(fields.get("operation") or given.words.memory_operation).upper()
    key, value = fields.get("key"), fields.get("memory_value")
    if key and value is not None:
        return given.join([operation, f"{key}={format_value(str(key), value)}"])
    return given.join([operation, str(key)]) if key else operation


def _memory_recall(given: SummaryInput) -> str:
    count = given.fields.get("selected", given.fields.get("count"))
    words = given.words
    return words.memory_recalled.format(count=count) if count is not None else words.memory_read


def _context_assembly(given: SummaryInput) -> str:
    words, fields = given.words, given.fields
    parts = [
        _counted(
            fields.get("working_memory_history_turns"), words.history_turn, words.history_turns
        ),
        _counted(
            fields.get("working_memory_state_items"),
            words.state_item,
            words.state_items,
            whole=True,
        ),
        _state_preview(fields.get("working_memory_state"), words),
        _context_sources(fields, words),
    ]
    return given.join(parts) or compact_fields(fields, ("selected", "used_tokens"))


def _counted(count: Any, one: str, many: str, *, whole: bool = False) -> str:
    """`1 history turn` or `3 history turns`; nothing when there is no count (or, `whole`,
    when it is not an integer)."""
    if count is None or (whole and not isinstance(count, int)):
        return ""
    return (one if count == 1 else many).format(count=count)


def _state_preview(state: Any, words: CardSummaryWords) -> str:
    if not isinstance(state, Mapping) or not state:
        return ""
    shown = list(state.items())[: words.state_preview_items]
    items = ", ".join(f"{key}={format_value(str(key), value)}" for key, value in shown)
    return words.state_preview.format(items=items)


def _context_sources(fields: Mapping[str, Any], words: CardSummaryWords) -> str:
    chunks, used = fields.get("chunks"), fields.get("used_tokens")
    if chunks is None or used is None:
        return ""
    return words.context_sources.format(chunks=chunks, tokens=used)


# -- the model ------------------------------------------------------------------------------


def _model_exchange(given: SummaryInput) -> str:
    requested = _requested_tool(given)
    if requested is not None:
        return requested
    words, fields = given.words, given.fields
    if given.event.stage == "llm_request":
        # A request is a prompt sent, with or without tools; it never "generated" anything.
        offered = fields.get("tools_offered")
        count = len(offered) if isinstance(offered, list) else 0
        tools = words.tools_offered.format(count=count) if count else ""
        return given.join([words.prompt_sent, tools, str(fields.get("purpose") or "")])
    if fields.get("finish_reason") == "tool_calls":
        return words.tool_requested
    return words.response_generated


def _requested_tool(given: SummaryInput) -> str | None:
    """The first tool call the model asked for, as `Tool Requested · tool · subject · amount`."""
    calls = given.fields.get("tool_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], Mapping):
        return None
    first = calls[0]
    tool = first.get("tool") or first.get("name")
    arguments = _arguments(first.get("arguments"))
    amount = arguments.get("amount")
    return given.join(
        [
            given.words.tool_requested,
            given.labels.tool_label(str(tool)) if tool else given.words.tool,
            subject_part(arguments),
            format_value("amount", amount) if amount is not None else "",
        ]
    )


def _arguments(arguments: Any) -> Mapping[str, Any]:
    """A call's arguments as a mapping, whether they came as one or as JSON text."""
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (TypeError, ValueError):
            parsed = {}
    return parsed if isinstance(parsed, Mapping) else {}


def _loop_control(given: SummaryInput) -> str:
    # The verdict names its own decision and mechanism (`loop_heading`).
    if given.event.summary:
        return given.event.summary
    fields = given.fields
    return given.join(
        [str(fields.get("decision") or "").upper(), str(fields.get("reason_code") or "")]
    )


# -- tools and their authority --------------------------------------------------------------


def _tool(given: SummaryInput) -> str:
    fields, words = given.fields, given.words
    tool = fields.get("tool") or fields.get("action")
    amount = first_present(fields, AMOUNT_FIELDS)
    result = fields.get("result")
    return given.join(
        [
            given.labels.tool_label(str(tool)) if tool else words.tool,
            subject_part(fields),
            format_value("amount", amount) if amount is not None else "",
            str(result) if result and str(result) not in words.plain_results else "",
        ]
    )


def _gatekeeper_evaluation(given: SummaryInput) -> str:
    fields = given.fields
    verdict = fields.get("decision") or fields.get("verdict")
    action = fields.get("action")
    amount = first_present(fields, AMOUNT_FIELDS)
    parts = [
        str(verdict).upper() if verdict else "",
        # What the principal may do about it: VOID, RATIFIABLE, VOIDABLE…
        str(fields.get("standing") or ""),
        given.labels.tool_label(str(action)) if action else "",
        subject_part(fields),
        format_value("amount", amount) if amount is not None else "",
        _stopping_check(fields.get("checks"), given.words.blocked),
    ]
    return given.join(parts) or given.words.authorization_evaluated


def _stopping_check(checks: Any, unnamed: str) -> str:
    """The first check that failed or escalated, by its reason code or id."""
    if not isinstance(checks, list):
        return ""
    for check in checks:
        if isinstance(check, Mapping) and check.get("status") in _STOPPING_CHECKS:
            return str(check.get("reason_code") or check.get("check_id") or unnamed)
    return ""


def _resilience(given: SummaryInput) -> str:
    words, attempts = given.words, given.fields.get("attempts")
    if attempts is None:
        return given.event.summary or words.recovery_event
    counted = (words.attempt if attempts == 1 else words.attempts).format(count=attempts)
    return given.join([counted, words.recovered if given.fields.get("recovered") is True else ""])


# -- the reply ------------------------------------------------------------------------------


def _hallucination_check(given: SummaryInput) -> str | None:
    # A blocked check has already returned its outcome as the summary.
    fields = given.fields
    tally = fields.get("tally")
    if not tally and fields.get("claims") is not None:
        tally = given.words.claims_supported.format(
            supported=fields.get("supported"), claims=fields["claims"]
        )
    if not tally:
        return None
    return given.join([str(tally), given.words.repaired if fields.get("repaired") else ""])


def _finalize(given: SummaryInput) -> str:
    words = given.words
    return (
        words.response_ready
        if given.fields.get("response_ready", True)
        else words.response_not_ready
    )


SUMMARISERS: dict[str, Summariser] = {
    "scope_check": _scope_check,
    "authentication": _authentication,
    "deterministic_checks": _deterministic_checks,
    "prompt_injection_check": _prompt_injection,
    "pii_filtering": _pii_filtering,
    "content_moderation": _content_moderation,
    "intent_classification": _intent_classification,
    "working_memory_write": _working_memory_write,
    "memory_recall": _memory_recall,
    "context_assembly": _context_assembly,
    "llm_request": _model_exchange,
    "llm_call": _model_exchange,
    "llm_response_received": _model_exchange,
    "loop_control": _loop_control,
    "tool_request": _tool,
    "tool_execution": _tool,
    "tool_result": _tool,
    "gatekeeper_evaluation": _gatekeeper_evaluation,
    "resilience": _resilience,
    "hallucination_check": _hallucination_check,
    "hallucination_repair": _hallucination_check,
    "finalize": _finalize,
}

__all__ = ["SUMMARISERS", "SummaryInput", "compact_fields", "decision_summary", "summarise"]
