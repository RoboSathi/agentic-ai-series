"""Widgets for the customer-support agent operations dashboard.

The TUI consumes semantic ``StatusEvent`` objects.  It never consumes the
logger's method-level trace stream.  Each completed semantic event becomes a
small collapsible card:

    collapsed:  what happened?
    expanded:   what useful semantic evidence explains it?

The logger remains responsible for implementation-level debugging.
"""

import json
import re
import textwrap
import time
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, cast
from urllib.parse import urlparse

from rich.text import Text
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, OptionList, Static
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets.button import ButtonVariant
from textual.widgets.option_list import Option

from agent_platform.config import get_settings
from agent_platform.observability.types import StatusEvent
from agent_platform.presentation.text_tree import as_structure, value_lines
from agent_platform.primitives.money import money_or_none
from agent_platform.tui.labels import StageLabelRegistry

# State is the visual language: a card's title, marks and state words take the colour of
# their outcome (`tui.outcome_colors`, `tui.tone_styles`), whatever stage they belong to.
STATUS_MARKS = {
    "started": "→",
    "completed": "✓",
    "failed": "✗",
    "warning": "⚠",
    "waiting": "…",
    "blocked": "✗",
}

# Where an agent's own subject fields sit in the card order.
_SUBJECT_SLOT = "__subject_fields__"

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
    _SUBJECT_SLOT,
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

# How a claim's verdict is marked on a card; the tally wording comes with
# the guardrail result, so title, log and card agree.
# The middle value is the claim's tone (`tui.tone_styles`), not a colour.
_CLAIM_MARKS = {
    "supported": ("✓", "pass", "supported"),
    "undetermined": ("?", "warn", "unverified"),
    "refuted": ("✗", "fail", "contradicted"),
}
# The NLI result each verdict is decided by, and its display name.
_DECIDING_LABEL = {"supported": "entailment", "refuted": "contradiction"}
_REFUTED = "refuted"
_UNDETERMINED = "undetermined"
_NLI_NAMES = {"unknown": "neutral"}
# A line of the record read directly, not scored by a model.
_RECORD_LABEL = "record"
_NOT_CHECKED = "not checked"
# Policy passages are markdown; emphasis markers are noise in a one-line excerpt.
_MARKDOWN_BOLD = "**"


# How much of an evidence excerpt a row shows; the log keeps it whole.
_EXCERPT_CHARS = 90

# Fields that can be noisy / sensitive for the TUI.  They are intentionally
# excluded even if an upstream event contains them.
MUTED_FIELDS = {
    "args",
    "kwargs",
    "content",
    "context",
    "value",
    "chunks_raw",
    "facts_raw",
    "payload",
    "trace_id",
    "raw_arguments",
    "session_id",
    "source_payload",
}

# Fields that are documents rather than values. A system prompt or a
# model reply rendered as `name = value` on one elided line tells you
# nothing you did not already know; these get a labelled block, wrapped
# and indented, so the thing you opened the card to read is readable.
#
# They were muted entirely until now, which is why the LLM card could
# show a token count and a finish reason but never the prompt or the
# answer — the two things a person watching actually wants.
BLOCK_FIELDS = (
    "reasoning",
    "observation",
    "output",
    "system_prompt",
    "user_query",
    "prompt",
    "messages",
    "instructions",
    "response",
    "response_text",
    "evidence",
    "working_memory_history",
    "working_memory_state",
    "text",
)


class TokenCounterWidget(Static):
    """Per-session and per-turn token totals."""

    session_tokens: reactive[int] = reactive(0)
    turn_tokens: reactive[int] = reactive(0)

    def __init__(self) -> None:
        super().__init__(id="tokens")

    def add(self, tokens: int) -> None:
        if tokens <= 0:
            return
        self.turn_tokens += tokens
        self.session_tokens += tokens
        self.refresh()

    def start_turn(self) -> None:
        self.turn_tokens = 0
        self.refresh()

    def render(self) -> Text:
        return Text("")


class StatusLineWidget(Static):
    """Current stage / run status shown beneath the input."""

    frame: reactive[int] = reactive(0)
    stage_label: reactive[str] = reactive("")
    running: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__(id="status-line")
        settings = get_settings().tui
        self._frames = settings.spinner_frames
        self._started_at = 0.0
        self._finished_at = 0.0
        self._tokens: TokenCounterWidget | None = None
        self._following = True
        self._history_pending = 0
        self._run_state = "idle"
        self._show_token_count = get_settings().tui.show_token_count
        self._summary_lines: tuple[str, ...] = ()
        self._detail_lines: tuple[str, ...] = ()
        self._expanded = False

    @property
    def elapsed_seconds(self) -> float:
        if not self._started_at:
            return 0.0
        end = time.monotonic() if self.running else self._finished_at
        return max(0.0, end - self._started_at)

    def begin(self, label: str) -> None:
        self._started_at = time.monotonic()
        self._finished_at = 0.0
        self.stage_label = label
        self.running = True
        self._run_state = "running"
        self.refresh()

    def update_stage(self, label: str) -> None:
        self.stage_label = label
        self.refresh()

    def clear_turn(self) -> None:
        """Remove the completed/failed turn state before a non-turn command runs."""
        self._started_at = 0.0
        self._finished_at = 0.0
        self.running = False
        self._run_state = "idle"
        self.stage_label = ""
        self._history_pending = 0
        self.refresh()

    def finish(self) -> None:
        self._finished_at = time.monotonic()
        self.running = False
        self._run_state = "complete"
        self.stage_label = "TURN COMPLETE"
        self.refresh()

    def fail(self) -> None:
        self._finished_at = time.monotonic()
        self.running = False
        self._run_state = "failed"
        self.stage_label = "TURN FAILED"
        self.refresh()

    def tick(self) -> None:
        if self.running and self._frames:
            self.frame = (self.frame + 1) % len(self._frames)
            self.refresh()

    def metrics(self) -> str:
        """`3m 27s · 9.7k tokens`: the turn's time and tokens, as every surface shows them."""
        elapsed = _elapsed(self.elapsed_seconds)
        if not self._show_token_count or self._tokens is None:
            return elapsed
        return f"{elapsed} · {_compact(self._tokens.turn_tokens)} tokens"

    def bind_tokens(self, counter: TokenCounterWidget) -> None:
        self._tokens = counter

    def set_following(self, following: bool) -> None:
        self._following = following
        self.refresh()

    def set_history_pending(self, count: int) -> None:
        self._history_pending = max(0, count)
        self.refresh()

    def set_block(self, summary: Sequence[str], detail: Sequence[str]) -> None:
        """The host's live lines: always the summary, the detail when expanded."""
        block = (tuple(summary), tuple(detail))
        if block != (self._summary_lines, self._detail_lines):
            self._summary_lines, self._detail_lines = block
            self.refresh(layout=True)

    def toggle_detail(self) -> None:
        self._expanded = not self._expanded
        self.refresh(layout=True)

    def on_click(self) -> None:
        """The block opens and closes on a click, like the cards above it."""
        if self._detail_lines:
            self.toggle_detail()

    def render(self) -> Text:
        headline = self._render_headline()
        lines = [*self._summary_lines, *(self._detail_lines if self._expanded else ())]
        if not lines and not self._detail_lines:
            return headline
        text = self._with_toggle(headline)
        for line in lines:
            if text:
                text.append("\n")
            text.append_text(Text.from_markup(line))
        return text

    def _with_toggle(self, headline: Text) -> Text:
        """The headline, led by the open/closed marker when there is more to show."""
        settings = get_settings().tui
        if not self._detail_lines:
            return headline
        marker = (
            settings.status_expanded_symbol if self._expanded else settings.status_collapsed_symbol
        )
        text = Text.styled(f"{marker} ", palette("marker"))
        text.append_text(headline)
        if not self._expanded:
            text.append(f"  {settings.status_expand_hint}", style=palette("muted"))
        return text

    def _render_headline(self) -> Text:
        if not self.stage_label:
            return Text("")

        styles = get_settings().tui.status_styles
        if self.running:
            symbol = self._frames[self.frame % len(self._frames)] if self._frames else "●"
            style = styles["running"]
        elif self._run_state == "failed":
            symbol = "✗"
            style = styles["failed"]
        else:
            symbol = "✓"
            style = styles["complete"]

        line = Text.styled(f"{symbol} ", style)
        line.append(self.stage_label, style=style)

        if self._started_at:
            line.append(f"  ({self.metrics()})", style=palette("metrics"))

        if not self._following:
            line.append("  · HISTORY", style=palette("history"))
            if self._history_pending:
                line.append(f" · +{self._history_pending} new", style=palette("history"))
            line.append(f" · {get_settings().tui.status_history_hint}", style=palette("history"))

        return line


class StageEventWidget(Collapsible):
    """One semantic agent event rendered as a collapsible dashboard card."""

    def __init__(
        self,
        event: StatusEvent,
        labels: StageLabelRegistry,
    ) -> None:
        self.event = event
        self._labels = labels
        self._status = event.status
        title = self._build_title()
        body = self._build_body()

        # One colour per outcome: a blocked reply is red whether one claim
        # failed or all of them did. Painting a partial block yellow made
        # a refused answer look like a warning.
        classes = f"stage-event stage-{self._status}"
        if event.group:
            classes += f" group-{event.group}"

        super().__init__(
            Static(body, classes="stage-details"),
            title=title,
            collapsed=True,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes=classes,
        )

    def on_mount(self) -> None:
        """The title coloured by its outcome, in the label's own text.

        Not a widget style: CollapsibleTitle's `:focus` rule repaints the widget's colour, so a
        clicked card went white. A styled label is set without the reactive's comparison (it
        compares by text, so the same words would be dropped) and then pushed to the screen.
        """
        colour = get_settings().tui.outcome_colors.get(self._status, "")
        title = self.query_one(CollapsibleTitle)
        title.set_reactive(CollapsibleTitle.label, Content.styled(self._build_title(), colour))
        title.mutate_reactive(CollapsibleTitle.label)

    def _build_title(self) -> str:
        mark = STATUS_MARKS.get(self.event.status, "·")
        stage = self._labels.title_for(
            self.event.stage,
            tool_name=str(self.event.fields.get("tool", self.event.detail or "")),
        )
        summary = self._summary()
        if summary:
            return f"{mark} {stage}   {summary}"
        return f"{mark} {stage}"

    def _build_body(self) -> Text:
        body = Text()

        if self.event.event_kind == "narrative":
            _append_narrative(body, self.event)
            return body

        if self.event.stage == "hallucination_check":
            _append_grounding(body, self.event.fields)
            return body

        fields = self._ordered_fields(self.event.fields)

        rendered_fields = False
        for name, value in fields:
            if name == "checks" and isinstance(value, list):
                if rendered_fields:
                    body.append("\n")
                self._append_checks(body, value)
                rendered_fields = True
                continue
            if value and (name in BLOCK_FIELDS or _is_structured(value)):
                if rendered_fields:
                    body.append("\n")
                self._append_block(body, name, value)
                rendered_fields = True
                continue

            if rendered_fields:
                body.append("\n")
            body.append("  ")
            _append_pair(body, _label(name), " = ", _format_value(name, value))
            rendered_fields = True

        # The summary is already shown in the collapsed title.
        # Only show it in the expanded body when there are no structured
        # semantic fields available.
        if not rendered_fields:
            if self.event.summary:
                body.append(
                    self.event.summary,
                    style=palette("emphasis"),
                )
            elif self.event.detail:
                body.append("detail = ")
                body.append(
                    _humanize(self.event.detail),
                )

        if self.event.tokens_delta:
            if rendered_fields or self.event.summary or self.event.detail:
                body.append("\n")

            body.append("  ")
            _append_pair(body, "tokens", " = ", f"+{self.event.tokens_delta}")

        if not body:
            body.append(
                "No additional semantic detail.",
                style=palette("muted"),
            )

        return body

    def _append_block(self, body: Text, name: str, value: Any) -> None:
        """A document-shaped field, as a labelled and indented block.

        Wrapped rather than elided: the point of expanding a card is to
        read the thing, and a prompt cut at eighty characters is a prompt
        you still have to go to the log files for.
        """
        styles = get_settings().tui.body_styles
        body.append(f"  {_label(name)}:", style=styles["key"])
        for line in _block_lines(value):
            body.append("\n    ")
            _append_line(body, line, "")

    def _summary(self) -> str:
        stage = self.event.stage
        fields = self.event.fields
        status = self.event.status

        if self.event.event_kind == "narrative":
            return self.event.summary or ""
        if status == "failed":
            return self.event.summary or "failed"
        if status == "blocked":
            return self.event.summary or _decision_summary(fields, self._labels)
        if status == "warning":
            return self.event.summary or "attention required"
        if status == "waiting":
            return self.event.summary or "waiting"

        if stage == "scope_check":
            if fields.get("in_scope") is True:
                return str(fields.get("topic") or "in scope")
            if fields.get("topic"):
                return f"out of scope · {fields['topic']}"

        if stage == "authentication":
            return "PASS · Identity Verified" if fields.get("verified") else "Verification Required"

        if stage == "basic_sanity":
            summary = get_settings().tui.reason_summaries.get(str(fields.get("reason_code")))
            if summary:
                return summary
            return "PASS · Deterministic checks" if fields.get("passed") else "Blocked"

        if stage == "prompt_injection_check":
            return "Blocked" if fields.get("detected") else "PASS · No prompt injection attempt"

        if stage == "pii_filtering":
            if fields.get("detected"):
                categories = fields.get("categories") or []
                matches = fields.get("matches")
                category_text = ", ".join(str(item) for item in categories) or "PII"
                suffix = (
                    f" · {matches} match{'es' if matches != 1 else ''}"
                    if matches is not None
                    else ""
                )
                return f"REDACTED · {category_text}{suffix}"
            return "PASS · No PII detected"

        if stage == "content_moderation":
            if status == "blocked":
                categories = fields.get("categories") or []
                category_text = ", ".join(str(item) for item in categories)

                if category_text:
                    return f"Flagged · Categories: {category_text}"

                return "Flagged"

            return "PASS · No harmful content detected"

        if stage == "intent_classification":
            intent = fields.get("intent") or self.event.detail or "classified"
            score = fields.get("model_score")
            disposition = fields.get("disposition")
            if score is not None:
                route = f" · {disposition}" if disposition else ""
                return f"{intent} · model score {float(score):.0%}{route}"
            return str(intent)

        if stage == "working_memory_write":
            operation = str(fields.get("operation") or "set").upper()
            key = fields.get("key")
            value = fields.get("memory_value")
            if key and value is not None:
                return f"{operation} · {key}={_format_value(str(key), value)}"
            return f"{operation} · {key}" if key else operation

        if stage == "memory_recall":
            count = fields.get("selected", fields.get("count"))
            return f"Recalled {count} items" if count is not None else "memory read"

        if stage == "context_assembly":
            chunks = fields.get("chunks")
            used = fields.get("used_tokens")
            turns = fields.get("working_memory_history_turns")
            state = fields.get("working_memory_state")
            parts = []
            if turns is not None:
                parts.append(f"{turns} history turn{'s' if turns != 1 else ''}")
            state_items = fields.get("working_memory_state_items")
            if isinstance(state_items, int):
                parts.append(f"{state_items} state item{'s' if state_items != 1 else ''}")
            if isinstance(state, Mapping) and state:
                preview = []
                for key, value in list(state.items())[:3]:
                    preview.append(f"{key}={_format_value(str(key), value)}")
                parts.append("state: " + ", ".join(preview))
            if chunks is not None and used is not None:
                parts.append(f"{chunks} sources · {used} tokens")
            return " · ".join(parts) or self._compact_from_fields(
                fields, ("selected", "used_tokens")
            )

        if stage in {"llm_request", "llm_call", "llm_response_received"}:
            calls = fields.get("tool_calls")
            if isinstance(calls, list) and calls:
                first = calls[0]
                if isinstance(first, Mapping):
                    tool = first.get("tool") or first.get("name")
                    arguments = first.get("arguments")
                    parsed: Any = arguments
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            parsed = {}
                    given = parsed if isinstance(parsed, Mapping) else {}
                    subject = _subject_part(given)
                    amount = given.get("amount")
                    parts = [
                        "Tool Requested",
                        self._labels.tool_label(str(tool)) if tool else "tool",
                    ]
                    if subject:
                        parts.append(subject)
                    if amount is not None:
                        parts.append(_format_value("amount", amount))
                    return " · ".join(parts)
            if stage == "llm_request":
                # A request is a prompt sent, with or without tools; it never "generated" anything.
                requested = fields.get("tools_offered")
                count = len(requested) if isinstance(requested, list) else 0
                purpose = fields.get("purpose")
                parts = [
                    "Prompt Sent",
                    f"{count} Tools Offered" if count else "",
                    str(purpose or ""),
                ]
                return " · ".join(part for part in parts if part)
            if fields.get("finish_reason") == "tool_calls":
                return "Tool Requested"
            return "Response Generated"

        if stage == "loop_control":
            # A round's status card: its own summary says the round and the next step.
            if "criteria" in fields and self.event.summary:
                return self.event.summary
            decision = str(fields.get("decision") or "").upper()
            reason_code = str(fields.get("reason_code") or "")
            round_number = fields.get("round")
            max_rounds = fields.get("max_rounds")
            if reason_code == "DUPLICATE_ACTION":
                return "SUPPRESS · duplicate action"
            if reason_code == "MAX_TOOL_ITERATIONS":
                limit = (
                    f" · round {round_number}/{max_rounds}" if round_number and max_rounds else ""
                )
                return f"ABORT · tool iteration limit{limit}"
            return (
                f"{decision} · {reason_code}"
                if decision and reason_code
                else decision or "control intervention"
            )

        if stage == "tool_request":
            return self._tool_summary(fields)

        if stage == "gatekeeper_evaluation":
            verdict = fields.get("decision") or fields.get("verdict")
            standing = fields.get("standing")
            action = fields.get("action")
            subject = _subject_part(fields)
            amount = _first_present(fields, AMOUNT_FIELDS)
            parts = []
            if verdict:
                parts.append(str(verdict).upper())
            # What the principal may do about it: VOID, RATIFIABLE, VOIDABLE…
            if standing:
                parts.append(str(standing))
            if action:
                parts.append(self._labels.tool_label(str(action)))
            if subject:
                parts.append(subject)
            if amount is not None:
                parts.append(_format_value("amount", amount))
            checks = fields.get("checks")
            if isinstance(checks, list):
                failed = next(
                    (
                        item
                        for item in checks
                        if isinstance(item, Mapping) and item.get("status") in {"FAIL", "ESCALATE"}
                    ),
                    None,
                )
                if isinstance(failed, Mapping):
                    parts.append(
                        str(failed.get("reason_code") or failed.get("check_id") or "blocked")
                    )
            return " · ".join(parts) or "authorization evaluated"

        if stage in {"tool_execution", "tool_result"}:
            return self._tool_summary(fields)

        if stage == "resilience":
            attempts = fields.get("attempts")
            recovered = fields.get("recovered")
            if attempts is not None:
                return f"{attempts} attempt{'s' if attempts != 1 else ''}" + (
                    " · recovered" if recovered is True else ""
                )
            return self.event.summary or "recovery event"

        if stage in {"hallucination_check", "hallucination_repair"}:
            # A blocked check has already returned its outcome as the summary.
            tally = fields.get("tally")
            if not tally and fields.get("claims") is not None:
                tally = f"{fields.get('supported')}/{fields['claims']} claims supported"
            if tally:
                return f"{tally}{' · repaired' if fields.get('repaired') else ''}"

        if stage == "finalize":
            return "Response Ready" if fields.get("response_ready", True) else "Response Not Ready"

        if self.event.summary:
            return self.event.summary
        if self.event.detail:
            return _humanize(self.event.detail)
        return ""

    def _append_checks(self, body: Text, checks: list[Any]) -> None:
        """Render structured checks as an operator-readable decision tree."""
        settings = get_settings().tui
        styles = settings.body_styles
        body.append("  checks", style=settings.section_title_style)
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            status = str(check.get("status", ""))
            name = str(check.get("name") or check.get("check_id") or "check")
            reason_code = str(check.get("reason_code") or "")
            reason = str(check.get("reason") or "")
            state = _state_style(status) or styles["value"]
            body.append("\n    ")
            body.append(f"{CHECK_MARKS.get(status, '·')} ", style=state)
            body.append(name, style=styles["key"])
            body.append(f" [{status}]", style=state)
            if reason_code:
                body.append(f" · {reason_code}", style=styles["muted"])
            if reason:
                body.append(" — ", style=styles["separator"])
                body.append(reason, style=styles["value"])
            for label in ("observed", "evidence"):
                detail = check.get(label)
                if isinstance(detail, Mapping) and detail:
                    body.append("\n      ")
                    body.append(f"{label}:", style=styles["key"])
                    for line in value_lines(dict(detail)):
                        body.append("\n        ")
                        _append_line(body, line, "")

    def _compact_from_fields(
        self,
        fields: Mapping[str, Any],
        names: tuple[str, ...],
    ) -> str:
        parts = []
        for name in names:
            value = fields.get(name)
            if value is not None:
                parts.append(f"{_label(name)}={_format_value(name, value)}")
        return " · ".join(parts)

    def _tool_summary(self, fields: Mapping[str, Any]) -> str:
        tool = fields.get("tool") or fields.get("action")
        tool_name = self._labels.tool_label(str(tool)) if tool else "tool"
        result = fields.get("result")
        subject = _subject_part(fields)
        amount = _first_present(fields, AMOUNT_FIELDS)
        parts = [tool_name]
        if subject:
            parts.append(subject)
        if amount is not None:
            parts.append(_format_value("amount", amount))
        if result and str(result) not in {"success", "ok", "completed"}:
            parts.append(str(result))
        return " · ".join(parts)

    @staticmethod
    def _ordered_fields(
        fields: Mapping[str, Any],
    ) -> list[tuple[str, Any]]:
        visible = [(name, value) for name, value in fields.items() if name not in MUTED_FIELDS]
        rank = {name: index for index, name in enumerate(_field_priority())}
        return sorted(
            visible,
            key=lambda item: (rank.get(item[0], len(rank)), item[0]),
        )

    def as_text(self) -> str:
        """Plain-text representation for copy/export."""
        title = self._build_title()
        body = self._build_body().plain
        return f"{title}\n{body}"


NARRATIVE_TIMING_FIELD = "latency_ms"
# A check's mark by its status; its colour comes from `tui.state_words`.
CHECK_MARKS = {"PASS": "✓", "FAIL": "✗", "ESCALATE": "⚠", "SKIPPED": "·"}
# A written line that reads as `key: value`: a short key (no URL, no sentence) before ": ".
# The branch glyphs in front of a tree line (`text_tree.value_lines`).
_TREE_PREFIX = re.compile(r"^[\s│├└─]*")
# A tree list item's key, `[0]`, reads as a key too.
_LINE_KEY = re.compile(r"^([A-Za-z\[][\w \-/().'·\]]{0,40}?):(?: (?=\S)|$)")


def _append_narrative(body: Text, event: StatusEvent) -> None:
    """A card its emitter wrote: titled sections of plain or toned lines.

    A section is a field whose name is its title. Its value is a string, a
    list of strings, or a list of `{"text", "tone"}` mappings, where tone
    is one of the configured `tui.tone_styles`. The widget knows nothing
    about the agent; it only lays out what it was given. A `key: value`
    line shows its key apart from its value, and a toned line's value takes
    its tone, so a pass, a warning and a failure read apart at a glance.
    """
    settings = get_settings().tui
    for title, value in event.fields.items():
        if title == NARRATIVE_TIMING_FIELD:
            continue
        if body:
            body.append("\n")
        body.append(f"  {title}", style=settings.section_title_style)
        for text, tone in _narrative_lines(value):
            body.append("\n    ")
            _append_line(body, text, settings.tone_styles.get(tone, ""))
    took = event.fields.get(NARRATIVE_TIMING_FIELD)
    if took is not None:
        body.append("\n  ")
        _append_pair(body, "took", " ", _format_value(NARRATIVE_TIMING_FIELD, took))
    if not body:
        body.append(event.summary or "", style=palette("emphasis"))


def _append_line(body: Text, text: str, tone_style: str) -> None:
    """One written line: `key: value` split by colour when it reads as one, else whole.

    A tree line's branch glyphs (`├─ `, `│  `) are drawn muted and the rest read as a line.
    """
    glyphs = _TREE_PREFIX.match(text)
    if glyphs and glyphs.end():
        body.append(glyphs.group(0), style=get_settings().tui.body_styles["separator"])
        text = text[glyphs.end() :]
    keyed = _LINE_KEY.match(text)
    if keyed is None:
        body.append(text, style=tone_style or _value_style(text))
        return
    key, value = keyed.group(1), text[keyed.end() :]
    _append_pair(body, key, ": ", value, tone_style)


def _append_pair(body: Text, key: str, separator: str, value: str, tone_style: str = "") -> None:
    """`key = value`: the key in the key colour, the value by its tone or the state it names."""
    styles = get_settings().tui.body_styles
    body.append(key, style=styles["key"])
    body.append(separator, style=styles["separator"])
    body.append(value, style=tone_style or _value_style(value))


def _value_style(value: str) -> str:
    """A value naming a state (PASS, BLOCKED, ...) takes its tone; any other the value colour."""
    return _state_style(value) or get_settings().tui.body_styles["value"]


def _state_style(word: str) -> str:
    settings = get_settings().tui
    tone = settings.state_words.get(word.strip().upper())
    return settings.tone_styles.get(tone, "") if tone else ""


def _narrative_lines(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, Mapping):
        return [(f"{key}: {item}", "") for key, item in value.items()]
    if isinstance(value, (list, tuple)):
        return [_narrative_line(item) for item in value]
    return [(line, "") for line in str(value).splitlines() or [""]]


def _narrative_line(item: Any) -> tuple[str, str]:
    if isinstance(item, Mapping):
        return str(item.get("text", "")), str(item.get("tone", ""))
    return str(item), ""


def _append_grounding(body: Text, fields: Mapping[str, Any]) -> None:
    """The hallucination check, as a verdict and one line of proof per claim.

    Drawn like every other card: `key = value` rows, then one check line
    per claim. Everything else the guardrail reports — every retrieved
    passage with its cosine and NLI scores, the diagnostics, the per-tier
    counts — is in `guardrails.log`. The card answers three questions: what
    was sent, what was it checked against, and what decided each sentence.
    Level 1 (token confidence) and level 2 (the agent's own grounding) lead the card.
    """
    for level in fields.get("levels") or []:
        if isinstance(level, Mapping):
            _append_level(body, level)
    _append_outcome(body, str(fields.get("outcome") or ""))
    # The agent's deterministic pre-filter, when it has one: how the reply reached this check.
    if fields.get("pre_check"):
        _append_grounding_row(body, "pre-check", str(fields["pre_check"]))
    checks = [c for c in fields.get("claim_checks") or [] if isinstance(c, Mapping)]
    if checks:
        unverified = any(c.get("verdict") == _UNDETERMINED for c in checks)
        note = " (unverified does not block)" if unverified else ""
        _append_grounding_row(body, "claims", str(fields.get("tally") or "") + note)
    _append_grounding_row(
        body, "checked against", str(fields.get("evidence_searched") or "no evidence")
    )
    provenance = [str(line) for line in fields.get("provenance_checks") or []]
    if provenance:
        _append_grounding_row(body, "ids & amounts", "  ".join(provenance))
    checker = next((c for c in checks if c.get("decided_by") != "deterministic_provenance"), None)
    if checker is not None:
        _append_grounding_row(body, "checker", _checker_line(checker))
    # Each claim as a check line under the rows, the way a guardrail card lists its checks.
    for check in checks:
        _append_claim(body, check)


def _append_level(body: Text, level: Mapping[str, Any]) -> None:
    """`level 1 · token confidence = …`, toned by its verdict, then its proof lines."""
    words = get_settings().hallucination_protection.levels
    key = words.row.format(level=level.get("level"), name=level.get("name"))
    body.append("\n  " if body else "  ")
    _append_pair(body, key, " = ", str(level.get("summary") or ""), _verdict_style(level))
    for line in level.get("lines") or []:
        body.append(f"\n      {line}", style=get_settings().tui.body_styles["muted"])


def _verdict_style(level: Mapping[str, Any]) -> str:
    """A level's tone: pass, flagged (warn) or skipped (dim)."""
    return _state_style(str(level.get("verdict") or "")) or get_settings().tui.body_styles["value"]


def _append_grounding_row(body: Text, key: str, value: str) -> None:
    body.append("\n  " if body else "  ")
    _append_pair(body, key, " = ", value)


def _append_outcome(body: Text, outcome: str) -> None:
    """`result = TRIMMED · removed 1 of 2 …`: only the state word takes its tone."""
    word, separator, rest = outcome.partition(" ")
    body.append("  ")
    _append_pair(body, "result", " = ", word, _state_style(word))
    body.append(separator + rest, style=get_settings().tui.body_styles["value"])


def _checker_line(check: Mapping[str, Any]) -> str:
    """`cross-encoder/nli-deberta-v3-base · supports at ≥ 0.6 · top-3 by embedding`."""
    threshold = check.get("threshold")
    bar = (
        f"supports at ≥ {threshold}" if threshold is not None else "uncalibrated: its label decides"
    )
    return f"{check.get('detector')} · {bar} · {check.get('retrieval')}"


def _append_claim(body: Text, check: Mapping[str, Any]) -> None:
    """`✗ claim [CONTRADICTED]`, as a check line, then the one line that decided it.

        contradicted by [Tool Result]: "Order 10665 refund status: pending_review."
        supported by [Policy] (Entailment 0.99): "…"
        contradicted: Amount(s) ['$99.00'] appear in the reply but in no tool result.
        unverified: no evidence states or contradicts this

    A probability appears only where a model gave one. Everything else about
    how the claim was checked is in `guardrails.log`.
    """
    styles = get_settings().tui.body_styles
    verdict = str(check.get("verdict") or _UNDETERMINED)
    mark, tone, name = _CLAIM_MARKS.get(verdict, ("?", "warn", verdict))
    style = _tone(tone)
    body.append(f"\n    {mark} ", style=style)
    body.append(str(check.get("claim") or ""), style=styles["value"])
    body.append(f" [{name.upper()}]", style=style)
    evidence = [item for item in check.get("evidence") or [] if isinstance(item, Mapping)]
    if verdict == _UNDETERMINED:
        _append_unverified(body, evidence)
        return
    item = _deciding_evidence(verdict, evidence)
    lead = "supported by" if verdict == "supported" else "contradicted by"
    if item is None or not _decided(verdict, evidence):
        # Decided by a rule that found no line to quote — an amount or id no
        # evidence states. Naming the nearest passage as the contradiction
        # sent readers after the wrong cause.
        _append_proof(body, lead.split()[0], str(check.get("rationale") or ""), styles["value"])
        return
    basis = str(check.get("basis") or "")
    key = f"{lead} [{_where(item)}]{f' ({basis})' if basis else ''}"
    _append_proof(body, key, f'"{_excerpt(item)}"', styles["muted"])


def _append_proof(body: Text, key: str, value: str, style: str) -> None:
    """The indented `key: value` line under a claim saying what decided it."""
    body.append("\n      ")
    _append_pair(body, key, ": ", value, style)


def _append_unverified(body: Text, evidence: list[Mapping[str, Any]]) -> None:
    """Why nothing decided the claim: a line the model read and found neutral, or nothing near it."""
    styles = get_settings().tui.body_styles
    scored = [item for item in evidence if _was_scored(item)]
    best = max(scored, key=lambda item: float(item.get("nli_score") or 0), default=None)
    if best is None:
        _append_proof(body, "unverified", "no evidence states or contradicts this", styles["value"])
        return
    label = str(best.get("nli_label"))
    key = (
        f"unverified [{_where(best)}] read as "
        f"{_NLI_NAMES.get(label, label)} ({float(best.get('nli_score') or 0):.2f})"
    )
    _append_proof(body, key, f'"{_excerpt(best)}"', styles["muted"])


def _was_scored(item: Mapping[str, Any]) -> bool:
    """Whether a model read this line, as opposed to it being too far from the claim to be sent."""
    label = str(item.get("nli_label") or "")
    return bool(label) and not label.startswith(_NOT_CHECKED) and label != _RECORD_LABEL


def _title(value: Any) -> str:
    """`tool result` as `Tool Result`."""
    return str(value or "").title()


def _where(item: Mapping[str, Any]) -> str:
    """`Tool Result · wef.org`: the tier, and where the line came from (a URL as its host)."""
    source = str(item.get("source") or "")
    host = urlparse(source).netloc if "://" in source else source
    return f"{_title(item.get('tier'))} · {host}" if host else _title(item.get("tier"))


def _decided(verdict: str, evidence: list[Any]) -> bool:
    """Whether some line decisively carried the claim's verdict."""
    wanted = {_DECIDING_LABEL.get(verdict), _RECORD_LABEL}
    return any(
        isinstance(item, Mapping) and item.get("decisive") and item.get("nli_label") in wanted
        for item in evidence
    )


def _deciding_evidence(verdict: str, evidence: list[Any]) -> Mapping[str, Any] | None:
    """The line that settled the claim, or the nearest one when none did."""
    items = [item for item in evidence if isinstance(item, Mapping)]
    wanted = {_DECIDING_LABEL.get(verdict), _RECORD_LABEL}
    decisive = [i for i in items if i.get("decisive") and i.get("nli_label") in wanted]
    ranked = decisive or items
    return ranked[0] if ranked else None


def _excerpt(item: Mapping[str, Any]) -> str:
    text = " ".join(str(item.get("excerpt") or "").replace(_MARKDOWN_BOLD, "").split())
    return text if len(text) <= _EXCERPT_CHARS else text[: _EXCERPT_CHARS - 1] + "…"


# Where a block-rendered line is folded. Narrower than the log files:
# this shares a terminal with the rest of the feed.
_BLOCK_WIDTH = 88


def _block_lines(value: Any) -> list[str]:
    """A document-shaped value as readable lines: a chat transcript as `role: content`,
    structured data (or JSON text of it) as a tree, prose wrapped on word boundaries."""
    if _is_transcript(value):
        text = "\n".join(_message_line(item) for item in value)
    elif _is_structured(value):
        return value_lines(value)
    elif isinstance(value, list):
        text = "\n".join(str(item) for item in value)
    else:
        text = str(value)
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        # Wrapped on word boundaries. Cutting mid-word saves a column and
        # costs the reader the sentence.
        lines.extend(textwrap.wrap(raw, width=_BLOCK_WIDTH) or [""])
    return lines


def _is_structured(value: Any) -> bool:
    """A mapping or list worth drawing as a tree (or JSON text of one)."""
    structured = as_structure(value)
    return isinstance(structured, (dict, list)) and bool(structured) and not _flat_list(structured)


def _flat_list(value: Any) -> bool:
    """A short list of plain values reads fine inline (`[a, b, c]`)."""
    return isinstance(value, list) and all(not isinstance(v, (dict, list)) for v in value)


def _is_transcript(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, Mapping) and "role" in item for item in value)
    )


def _message_line(message: Mapping[str, Any]) -> str:
    """One chat message, as `role: content`, then any tool calls it made, one per line."""
    role = message.get("role", "?")
    answering = message.get("tool")
    head = f"{role} ({answering})" if answering else str(role)
    lines = [f"{head}: {message.get('content', '')}"]
    for call in message.get("tool_calls") or []:
        lines.append(f"  → {call.get('tool')}({call.get('arguments')})")
    return "\n".join(lines)


class FeedNote(Static):
    """Non-event note such as user input, assistant response, or command output.

    The style is applied as a span (`Text.styled`), never as the Text's base style: Textual
    parses a base style itself and does not know Rich's colour names, so `orange3` or
    `bright_black` there rendered as the default white.
    """

    def __init__(
        self,
        text: str | Text,
        style: str = "",
        *,
        classes: str = "",
    ) -> None:
        rendered = text if isinstance(text, Text) else Text.styled(text, style)
        widget_classes = "feed-note"
        if classes:
            widget_classes += f" {classes}"
        super().__init__(rendered, classes=widget_classes)
        self.plain_text = rendered.plain


_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CITATION = re.compile(r"\[\d+\]")
_URL = re.compile(r"https?://\S+")
_BULLET = re.compile(r"^(\s*)([-•*])\s+")
_HEADINGS = (("## ", "h2"), ("# ", "h1"))
_BLANK_RUN = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")


def render_response(text: str, styles: Mapping[str, str]) -> Text:
    """An answer in light Markdown, coloured by role: headings, bullets, bold, citations, URLs.

    Text with none of that marks up as plain body text, so a reply that is not
    Markdown comes out exactly as it went in.
    """
    out = Text()
    # No blank line before or after the answer, and never two in a row inside it: the
    # card's own margin spaces it from the notes around it.
    lines = _BLANK_RUN.sub("\n\n", text.strip()).split("\n")
    for index, line in enumerate(line.rstrip() for line in lines):
        if index:
            out.append("\n")
        _append_response_line(out, line, styles)
    return out


def _append_response_line(out: Text, line: str, styles: Mapping[str, str]) -> None:
    for prefix, role in _HEADINGS:
        if line.startswith(prefix):
            out.append(line[len(prefix) :], style=styles[role])
            return
    bullet = _BULLET.match(line)
    if bullet:
        out.append(f"{bullet.group(1)}{bullet.group(2)} ", style=styles["bullet"])
        line = line[bullet.end() :]
    _append_inline(out, line, styles)


def _append_inline(out: Text, line: str, styles: Mapping[str, str]) -> None:
    """Configured highlights, then bold runs, citation markers and URLs, within one line.

    Each span is (start, end, text shown, style, rank); at one position the lower rank wins,
    and a span that overlaps one already drawn is left as body text.
    """
    position = 0
    spans = sorted(
        _highlight_spans(line)
        + [(m.start(), m.end(), m.group(1), styles["bold"], 1) for m in _BOLD.finditer(line)]
        + [
            (m.start(), m.end(), m.group(0), styles["citation"], 1)
            for m in _CITATION.finditer(line)
        ]
        + [(m.start(), m.end(), m.group(0), styles["url"], 1) for m in _URL.finditer(line)],
        key=lambda span: (span[0], span[4]),
    )
    for start, end, shown, style, _ in spans:
        if start < position:
            continue
        out.append(line[position:start], style=styles["body"])
        out.append(shown, style=style)
        position = end
    out.append(line[position:], style=styles["body"])


def _highlight_spans(line: str) -> list[tuple[int, int, str, str, int]]:
    """The configured highlight rules' matches, Markdown bold markers dropped from what shows."""
    return [
        (m.start(), m.end(), m.group(0).replace(_MARKDOWN_BOLD, ""), style, 0)
        for pattern, style, scope in _highlights()
        if scope is None or scope.search(line)
        for m in pattern.finditer(line)
        if m.end() > m.start()
    ]


def _highlights() -> list[tuple[re.Pattern[str], str, re.Pattern[str] | None]]:
    rules = get_settings().tui.response_highlights
    return _compiled_highlights(tuple((rule.pattern, rule.style, rule.line) for rule in rules))


@cache
def _compiled_highlights(
    rules: tuple[tuple[str, str, str | None], ...],
) -> list[tuple[re.Pattern[str], str, re.Pattern[str] | None]]:
    return [
        (re.compile(pattern), style, re.compile(line) if line is not None else None)
        for pattern, style, line in rules
    ]


class ChoiceMenu(OptionList):
    """A fixed menu of answers beside the free-text input; hidden unless a question offers one."""

    def __init__(self) -> None:
        super().__init__(id="choices")
        self._values: list[str] = []
        self.display = False

    @property
    def offered(self) -> bool:
        return bool(self._values) and self.display

    def offer(self, choices: Sequence[Any]) -> None:
        """Show these choices (anything with a `label` and a `value`); none hides the menu."""
        self.clear_options()
        self._values = [str(_choice_part(choice, "value")) for choice in choices]
        self.add_options([Option(str(_choice_part(choice, "label"))) for choice in choices])
        self.border_title = get_settings().tui.choice_menu_hint
        self.display = bool(self._values)
        if self._values:
            self.highlighted = 0

    def dismiss(self) -> None:
        self._values = []
        self.display = False

    def move(self, step: int) -> None:
        if self.offered:
            count = len(self._values)
            self.highlighted = ((self.highlighted or 0) + step) % count

    def chosen(self) -> str | None:
        """The highlighted choice's value, or None when no menu is showing."""
        index = self.highlighted
        if not self.offered or index is None or index >= len(self._values):
            return None
        return self._values[index]

    def value_at(self, index: int) -> str | None:
        return self._values[index] if 0 <= index < len(self._values) else None


def _choice_part(choice: Any, name: str) -> Any:
    return choice[name] if isinstance(choice, Mapping) else getattr(choice, name)


class ActionBar(Horizontal):
    """A row of buttons for the next steps an agent offers; hidden when there are none."""

    def __init__(self) -> None:
        super().__init__(id="actions")
        self._shown: tuple[tuple[Any, bool], ...] = ()
        self._commands: dict[str, str] = {}
        self._generation = 0
        self.display = False

    def offer(self, actions: Sequence[Any], busy: bool) -> None:
        """Show these actions (anything with `label`, `command`, `during_turn`).

        While a turn runs, only the actions marked `during_turn` can be pressed.
        """
        key = tuple((action, busy) for action in actions)
        if key == self._shown:
            return
        # Fresh ids every time: `remove_children` completes after this returns, so reusing
        # `action-0` collided with the button still being removed, the mount raised, and the
        # bar stayed empty for the rest of the session.
        self._generation += 1
        commands: dict[str, str] = {}
        buttons = []
        for index, action in enumerate(actions):
            button_id = f"action-{self._generation}-{index}"
            commands[button_id] = action.command
            # One row tall (`#actions Button` CSS), but still filled with its variant's colour:
            # Textual's compact button has no fill and read as plain text nobody clicked.
            variant = cast(ButtonVariant, getattr(action, "variant", "default"))
            button = Button(action.label, id=button_id, variant=variant)
            button.disabled = busy and not action.during_turn
            buttons.append(button)
        self.remove_children()
        if buttons:
            self.mount(*buttons)
        self._commands = commands
        self._shown = key
        self.display = bool(actions)

    def command_for(self, button_id: str | None) -> str | None:
        return self._commands.get(button_id or "")


class ResponseNote(Static):
    """The assistant's final answer, coloured by the role of each line."""

    def __init__(self, text: str, styles: Mapping[str, str]) -> None:
        super().__init__(render_response(text, styles), classes="feed-note")
        self.plain_text = text


def _merge_loop_control_events(
    previous: StatusEvent,
    current: StatusEvent,
) -> StatusEvent:
    """Coalesce adjacent loop-control facts into one operator card.

    The graph may emit the loop decision before and the draft-check result
    after the model turn. They describe one loop-control round, so the TUI
    should render one card rather than two consecutive LOOP CONTROL sections.
    """
    fields = dict(previous.fields)
    fields.update(current.fields)

    severity = {
        "started": 0,
        "completed": 1,
        "waiting": 2,
        "warning": 2,
        "failed": 3,
        "blocked": 3,
    }
    status = (
        current.status if severity[current.status] >= severity[previous.status] else previous.status
    )

    tokens_delta = previous.tokens_delta
    if current.tokens_delta is not None:
        tokens_delta = (tokens_delta or 0) + current.tokens_delta

    return previous.model_copy(
        update={
            "status": status,
            "timestamp": current.timestamp,
            "tokens_delta": tokens_delta,
            "detail": current.detail or previous.detail,
            "summary": previous.summary or current.summary,
            "fields": fields,
        }
    )


class LiveStatusPanel(VerticalScroll):
    """Scrollable dashboard containing semantic stage cards and notes."""

    def __init__(
        self,
        labels: StageLabelRegistry | None = None,
    ) -> None:
        super().__init__(id="feed")
        self._labels = labels or StageLabelRegistry()
        self._max_entries = get_settings().tui.max_feed_lines
        self._entries: list[StageEventWidget | FeedNote | ResponseNote] = []
        # Cards still in progress, by (run, stage): the finished card takes its place.
        self._pending: dict[tuple[str, str], StageEventWidget] = {}

    def show_status(self, event: StatusEvent) -> None:
        """Render completed/warning/failed semantic events as cards.

        Most started events are represented by the separate current status line only. A
        started event of a kind in `tui.pending_card_kinds` (a model call the person waits
        on) draws a pending card at once, and its finished card replaces it in place, so the
        timeline never shows ``→ X`` followed by ``✓ X`` as two unrelated records.
        """
        if event.status == "started":
            if event.event_kind in get_settings().tui.pending_card_kinds:
                self._show_pending(event)
            return
        pending = self._pending.pop((event.run_id, event.stage), None)
        if pending is not None and pending in self._entries:
            self._replace(pending, StageEventWidget(event, self._labels))
            return

        if (
            event.stage == "loop_control"
            and event.event_kind == "loop_control"
            and self._entries
            and isinstance(self._entries[-1], StageEventWidget)
            and self._entries[-1].event.stage == "loop_control"
            and self._entries[-1].event.event_kind == "loop_control"
        ):
            previous = self._entries[-1]
            merged = _merge_loop_control_events(previous.event, event)
            try:
                previous.remove()
            except Exception:
                pass
            self._entries[-1] = StageEventWidget(merged, self._labels)
            self.mount(self._entries[-1])
            return

        card = StageEventWidget(
            event,
            self._labels,
        )
        self._entries.append(card)
        self._trim_if_needed()
        self.mount(card)

    def _show_pending(self, event: StatusEvent) -> None:
        card = StageEventWidget(event, self._labels)
        self._pending[(event.run_id, event.stage)] = card
        self._entries.append(card)
        self._trim_if_needed()
        self.mount(card)

    def _replace(self, old: StageEventWidget, new: StageEventWidget) -> None:
        """The finished card where its pending card was, in the timeline and the feed."""
        self._entries[self._entries.index(old)] = new
        self.mount(new, before=old)
        old.remove()

    def note(self, text: str | Text, style: str = "", classes: str = "") -> None:
        note = FeedNote(text, style, classes=classes)
        self._entries.append(note)
        self._trim_if_needed()
        self.mount(note)

    def response(self, text: str) -> None:
        """The assistant's answer, with its headings, bullets and citations coloured."""
        note = ResponseNote(text, get_settings().tui.response_styles)
        self._entries.append(note)
        self._trim_if_needed()
        self.mount(note)

    def show_lines(self, lines: list[Text]) -> None:
        for line in lines:
            # Mounted as built: its colours are spans, which a `line.style` copy would drop.
            self.note(line)

    def clear_feed(self) -> None:
        self._entries.clear()
        self._pending.clear()
        self.remove_children()

    def as_text(self) -> str:
        return "\n".join(
            entry.as_text() if isinstance(entry, StageEventWidget) else entry.plain_text
            for entry in self._entries
        )

    def _trim_if_needed(self) -> None:
        """Keep unlimited history when configured with zero."""
        if self._max_entries <= 0:
            return
        while len(self._entries) > self._max_entries:
            oldest = self._entries.pop(0)
            try:
                oldest.remove()
            except Exception:
                pass


def _decision_summary(
    fields: Mapping[str, Any],
    labels: StageLabelRegistry,
) -> str:
    decision = fields.get("decision") or fields.get("verdict")
    action = fields.get("action")
    if not decision and not action:
        return "blocked"
    parts = []
    if decision:
        parts.append(str(decision).upper())
    if action:
        parts.append(labels.tool_label(str(action)))
    return " · ".join(parts)


def palette(role: str) -> str:
    """The configured style for one console role (`tui.palette`): banner, metrics, muted, ..."""
    return get_settings().tui.palette[role]


def _tone(name: str) -> str:
    """A state's style (`tui.tone_styles`): pass, fail, warn, dim."""
    return get_settings().tui.tone_styles[name]


def _field_priority() -> tuple[str, ...]:
    """The card field order, with this agent's subject fields in their slot."""
    subjects = tuple(subject.field for subject in get_settings().agent.subject_fields)
    index = FIELD_PRIORITY.index(_SUBJECT_SLOT)
    return FIELD_PRIORITY[:index] + subjects + FIELD_PRIORITY[index + 1 :]


def _label(name: str) -> str:
    """A field's name as a card shows it: an agent's subject label, else the platform's."""
    for subject in get_settings().agent.subject_fields:
        if subject.field == name:
            return subject.label
    return FIELD_LABELS.get(name, name)


def _subject_part(fields: Mapping[str, Any]) -> str:
    """What an action is about, for a one-line summary ("order 10042"), or ''."""
    for subject in get_settings().agent.subject_fields:
        value = fields.get(subject.field)
        if subject.summarise and value is not None:
            return f"{subject.label} {value}"
    return ""


def _first_present(
    fields: Mapping[str, Any],
    names: set[str],
) -> Any | None:
    for name in _field_priority():
        if name in names and name in fields:
            return fields[name]
    return None


def _format_value(
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
        items = ", ".join(f"{key}={_format_value(str(key), item)}" for key, item in value.items())
        return "{" + items + "}"

    return " ".join(str(value).split())


def _humanize(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def _compact(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
