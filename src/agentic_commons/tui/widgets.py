"""Widgets for the customer-support agent operations dashboard.

The TUI consumes semantic ``StatusEvent`` objects.  It never consumes the
logger's method-level trace stream.  Each completed semantic event becomes a
small collapsible card:

    collapsed:  what happened?
    expanded:   what useful semantic evidence explains it?

The logger remains responsible for implementation-level debugging.
"""

import json
import textwrap
import time
from collections.abc import Mapping
from typing import Any

from rich.text import Text
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Collapsible, Static

from agentic_commons.config import get_settings
from agentic_commons.money import money_or_none
from agentic_commons.observability.types import StatusEvent
from agentic_commons.tui.labels import StageLabelRegistry

# State is the primary visual language.
STATUS_MARKS = {
    "started": "→",
    "completed": "✓",
    "failed": "✗",
    "warning": "⚠",
    "waiting": "…",
    "blocked": "✗",
}

STATUS_STYLES = {
    "started": "cyan",
    "completed": "green",
    "failed": "red",
    "warning": "yellow",
    "waiting": "yellow",
    "blocked": "red",
}

SUMMARY_ACCENT_STYLES = {
    "intent": "magenta",
    "decision": "magenta",
    "security": "yellow",
    "context": "blue",
    "memory": "blue",
    "model": "blue",
    "tool_request": "blue",
    "tool_result": "blue",
    "gatekeeper": "white",
    "resilience": "yellow",
    "control_loop": "blue",
    "grounding": "blue",
    "escalation": "yellow",
}

# Common field ordering for expanded cards.  Any other safe semantic fields
# follow alphabetically.
FIELD_PRIORITY = (
    # A model card reads in the order the exchange happened: what the
    # customer asked, what we sent, what came back.
    "user_query",
    "system_prompt",
    "response",
    "decision",
    "action",
    "effect",
    "qualified_action",
    "reason_code",
    "question",
    "intent",
    "topic",
    "order_id",
    "customer_id",
    "amount",
    "requested_amount",
    "allowed_amount",
    "autonomous_limit",
    "cumulative_total",
    "threshold",
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
    "authority_rank",
    "required_authority_rank",
    "max_granting_rank",
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
    "refund_status",
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
    "reflection_comment",
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
    "order_id": "order",
    "customer_id": "customer",
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
    "reflection_comment": "reflection_comment",
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
_CLAIM_MARKS = {
    "supported": ("✓", "green", "supported"),
    "undetermined": ("?", "yellow", "unverified"),
    "refuted": ("✗", "red", "contradicted"),
}
# The NLI result each verdict is decided by, and its display name.
_DECIDING_LABEL = {"supported": "entailment", "refuted": "contradiction"}
_REFUTED = "refuted"
_UNDETERMINED = "undetermined"
_NLI_NAMES = {"unknown": "neutral"}
# A line of the record read directly, not scored by a model.
_RECORD_LABEL = "record"
_RECORD_BASIS = "Record"
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
        # Durable/memory context for the last turn: thread, checkpoints,
        # deterministic recall hit-rate, curator backlog.
        self._run_context = ""

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

    def bind_tokens(self, counter: TokenCounterWidget) -> None:
        self._tokens = counter

    def set_following(self, following: bool) -> None:
        self._following = following
        self.refresh()

    def set_history_pending(self, count: int) -> None:
        self._history_pending = max(0, count)
        self.refresh()

    def set_context(self, context: str) -> None:
        self._run_context = context
        self.refresh()

    def render(self) -> Text:
        if not self.stage_label:
            return Text("")

        if self.running:
            symbol = self._frames[self.frame % len(self._frames)] if self._frames else "●"
            symbol_style = "cyan"
            label_style = "bold cyan"
        elif self._run_state == "failed":
            symbol = "✗"
            symbol_style = "red"
            label_style = "bold red"
        else:
            symbol = "✓"
            symbol_style = "green"
            label_style = "bold green"

        line = Text(f"{symbol} ", style=symbol_style)
        line.append(self.stage_label, style=label_style)

        if self._show_token_count and self._tokens is not None and self._started_at:
            line.append(
                f"  ({_elapsed(self.elapsed_seconds)} · "
                f"↓ {_compact(self._tokens.turn_tokens)} tokens)",
                style="dim",
            )
        elif self._started_at:
            line.append(
                f"  ({_elapsed(self.elapsed_seconds)})",
                style="dim",
            )

        if self._run_context:
            line.append(f"  · {self._run_context}", style="dim cyan")

        if not self._following:
            line.append("  · HISTORY", style="yellow")
            if self._history_pending:
                line.append(f" · +{self._history_pending} new", style="yellow")

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

        super().__init__(
            Static(body, classes="stage-details"),
            title=title,
            collapsed=True,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes=classes,
        )

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
            if name in BLOCK_FIELDS and value:
                if rendered_fields:
                    body.append("\n")
                self._append_block(body, name, value)
                rendered_fields = True
                continue

            if rendered_fields:
                body.append("\n")
            body.append("  ")
            body.append(FIELD_LABELS.get(name, name), style="bright_white")
            body.append(" = ")
            body.append(
                _format_value(name, value),
                style=_field_style(self.event.event_kind, name),
            )
            rendered_fields = True

        # The summary is already shown in the collapsed title.
        # Only show it in the expanded body when there are no structured
        # semantic fields available.
        if not rendered_fields:
            if self.event.summary:
                body.append(
                    self.event.summary,
                    style="bold",
                )
            elif self.event.detail:
                body.append("detail = ")
                body.append(
                    _humanize(self.event.detail),
                )

        if self.event.tokens_delta:
            if rendered_fields or self.event.summary or self.event.detail:
                body.append("\n")

            body.append("  tokens = ")
            body.append(
                f"+{self.event.tokens_delta}",
                style="dim",
            )

        if not body:
            body.append(
                "No additional semantic detail.",
                style="dim",
            )

        return body

    def _append_block(self, body: Text, name: str, value: Any) -> None:
        """A document-shaped field, as a labelled and indented block.

        Wrapped rather than elided: the point of expanding a card is to
        read the thing, and a prompt cut at eighty characters is a prompt
        you still have to go to the log files for.
        """
        body.append(f"  {FIELD_LABELS.get(name, name)}:\n", style="bright_white")
        for line in _block_lines(value):
            body.append(f"    {line}\n", style="dim")
        # Drop the trailing newline so blocks sit flush with plain fields.
        body.right_crop(1)

    def _summary(self) -> str:
        stage = self.event.stage
        fields = self.event.fields
        status = self.event.status

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
            return "customer verified" if fields.get("verified") else "verification required"

        if stage == "basic_sanity":
            if fields.get("reason_code") == "ORDER_REFERENCE_NOT_PRESENT":
                return "order reference not provided · validation deferred"
            return "checks passed" if fields.get("passed") else "blocked"

        if stage == "prompt_injection_check":
            return "blocked" if fields.get("detected") else "checks passed"

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
            return "PASS · no PII detected"

        if stage == "content_moderation":
            return "flagged" if fields.get("flagged") else "no issues detected"

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
            return f"recalled {count} items" if count is not None else "memory read"

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
                    order_id = None
                    amount = None
                    if isinstance(arguments, str):
                        try:
                            parsed = json.loads(arguments)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            parsed = {}
                        if isinstance(parsed, Mapping):
                            order_id = parsed.get("order_id")
                            amount = parsed.get("amount")
                    elif isinstance(arguments, Mapping):
                        order_id = arguments.get("order_id")
                        amount = arguments.get("amount")
                    parts = [
                        "tool requested",
                        self._labels.tool_label(str(tool)) if tool else "tool",
                    ]
                    if order_id is not None:
                        parts.append(f"order {order_id}")
                    if amount is not None:
                        parts.append(_format_value("amount", amount))
                    return " · ".join(parts)
            requested = fields.get("tools_offered")
            if stage == "llm_request" and requested:
                count = len(requested) if isinstance(requested, list) else 0
                return f"prompt sent · {count} tools offered" if count else "prompt sent"
            if fields.get("finish_reason") == "tool_calls":
                return "tool requested"
            return "response generated"

        if stage == "control_loop":
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
            decision = fields.get("decision") or fields.get("verdict")
            action = fields.get("action")
            order_id = fields.get("order_id")
            amount = _first_present(fields, AMOUNT_FIELDS)
            parts = []
            if decision:
                parts.append(str(decision).upper())
            if action:
                parts.append(self._labels.tool_label(str(action)))
            if order_id is not None:
                parts.append(f"order {order_id}")
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
            return "response ready" if fields.get("response_ready", True) else "response not ready"

        if self.event.summary:
            return self.event.summary
        if self.event.detail:
            return _humanize(self.event.detail)
        return ""

    def _append_checks(self, body: Text, checks: list[Any]) -> None:
        """Render structured checks as an operator-readable decision tree."""
        body.append("  checks", style="bold bright_white")
        for check in checks:
            if not isinstance(check, Mapping):
                continue
            status = str(check.get("status", ""))
            mark, style = {
                "PASS": ("✓", "green"),
                "FAIL": ("✗", "red"),
                "ESCALATE": ("⚠", "yellow"),
                "SKIPPED": ("·", "dim"),
            }.get(status, ("·", "white"))
            name = str(check.get("name") or check.get("check_id") or "check")
            reason_code = str(check.get("reason_code") or "")
            reason = str(check.get("reason") or "")
            body.append("\n    ")
            body.append(f"{mark} {name} [{status}]", style=style)
            if reason_code:
                body.append(f" · {reason_code}", style="dim")
            if reason:
                body.append(f" — {reason}", style="bright_white")
            observed = check.get("observed")
            if isinstance(observed, Mapping) and observed:
                body.append("\n      observed = ", style="dim")
                body.append(_format_mapping(observed), style="dim")
            evidence = check.get("evidence")
            if isinstance(evidence, Mapping) and evidence:
                body.append("\n      evidence = ", style="dim")
                body.append(_format_mapping(evidence), style="dim")

    def _compact_from_fields(
        self,
        fields: Mapping[str, Any],
        names: tuple[str, ...],
    ) -> str:
        parts = []
        for name in names:
            value = fields.get(name)
            if value is not None:
                parts.append(f"{FIELD_LABELS.get(name, name)}={_format_value(name, value)}")
        return " · ".join(parts)

    def _tool_summary(self, fields: Mapping[str, Any]) -> str:
        tool = fields.get("tool") or fields.get("action")
        tool_name = self._labels.tool_label(str(tool)) if tool else "tool"
        result = fields.get("result")
        order_id = fields.get("order_id")
        amount = _first_present(fields, AMOUNT_FIELDS)
        parts = [tool_name]
        if order_id is not None:
            parts.append(f"order {order_id}")
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
        rank = {name: index for index, name in enumerate(FIELD_PRIORITY)}
        return sorted(
            visible,
            key=lambda item: (rank.get(item[0], len(rank)), item[0]),
        )

    def as_text(self) -> str:
        """Plain-text representation for copy/export."""
        title = self._build_title()
        body = self._build_body().plain
        return f"{title}\n{body}"


def _append_grounding(body: Text, fields: Mapping[str, Any]) -> None:
    """The hallucination check, as a verdict and one line of proof per claim.

    Everything else the guardrail reports — every retrieved passage with
    its cosine and NLI scores, the diagnostics, the per-tier counts — is in
    `guardrails.log`. The card answers three questions: was the reply sent,
    what was it checked against, and what decided each sentence.
    """
    escalated = bool(fields.get("escalated"))
    _append_row(body, "result", str(fields.get("outcome") or ""), "red" if escalated else "green")
    checks = [c for c in fields.get("claim_checks") or [] if isinstance(c, Mapping)]
    if checks:
        unverified = any(c.get("verdict") == "undetermined" for c in checks)
        note = " (unverified does not block)" if unverified else ""
        _append_row(body, "claims", str(fields.get("tally") or "") + note, "bright_white")
    _append_row(
        body, "checked against", str(fields.get("evidence_searched") or "no evidence"), "dim"
    )
    provenance = [str(line) for line in fields.get("provenance_checks") or []]
    if provenance:
        _append_row(body, "ids & amounts", "  ".join(provenance), "dim")
    checker = next((c for c in checks if c.get("decided_by") != "deterministic_provenance"), None)
    if checker is not None:
        _append_row(body, "checker", _checker_line(checker), "dim")
    for check in checks:
        _append_claim(body, check)


def _append_row(body: Text, label: str, value: str, style: str) -> None:
    if body:
        body.append("\n")
    body.append(f"  {label:<16}", style="bright_white")
    body.append(value, style=style)


def _checker_line(check: Mapping[str, Any]) -> str:
    """`cross-encoder/nli-deberta-v3-base · supports at ≥ 0.6 · top-3 by embedding`."""
    threshold = check.get("threshold")
    bar = (
        f"supports at ≥ {threshold}" if threshold is not None else "uncalibrated: its label decides"
    )
    return f"{check.get('detector')} · {bar} · {check.get('retrieval')}"


def _append_claim(body: Text, check: Mapping[str, Any]) -> None:
    """`✓ claim`, then what decided it, then the other lines that were read.

        Supported By [Tool Result] (Entailment 0.99): "Order 10665 refund status: pending_review."
        Contradicted By [Tool Result] (Record): "…" · refund status is completed, so …
        Unverified: tool result: neutral 0.98 · policy: nothing related (best 0.21 < 0.3)
        Also Read: [Policy] Neutral 0.12 "…"
    """
    verdict = str(check.get("verdict") or _UNDETERMINED)
    mark, style, _ = _CLAIM_MARKS.get(verdict, ("?", "yellow", verdict))
    body.append(f"\n\n  {mark} ", style=style)
    body.append(str(check.get("claim") or ""), style="bright_white")
    evidence = [item for item in check.get("evidence") or [] if isinstance(item, Mapping)]
    rationale = str(check.get("rationale") or "")
    if verdict == _UNDETERMINED:
        body.append(f"\n      Unverified: {rationale or 'no evidence decided it'}", style=style)
        _append_also_read(body, evidence, None)
        return
    item = _deciding_evidence(verdict, evidence)
    lead = "Supported By" if verdict == "supported" else "Contradicted By"
    if item is None or not _decided(verdict, evidence):
        # Refuted by a rule, not by a passage — an amount or id no evidence
        # states. Naming the nearest passage as the contradiction sent
        # readers after the wrong cause.
        body.append(f"\n      {lead.split()[0]}: {rationale}", style=style)
        _append_also_read(body, evidence, None)
        return
    basis = str(check.get("basis") or "")
    body.append(f"\n      {lead} [{_title(item.get('tier'))}] ({basis}): ", style=style)
    body.append(f'"{_excerpt(item)}"', style="dim")
    if basis == _RECORD_BASIS and rationale:
        body.append(f" · {rationale}", style=style)
    _append_also_read(body, evidence, item)


def _append_also_read(
    body: Text, evidence: list[Mapping[str, Any]], decided: Mapping[str, Any] | None
) -> None:
    """The other lines that were read for this claim, so a wrong verdict can be traced."""
    limit = get_settings().hallucination_protection.card_evidence_lines
    for item in [i for i in evidence if i is not decided][:limit]:
        label = str(item.get("nli_label") or "unknown")
        name = _NLI_NAMES.get(label, label).title()
        score = float(item.get("nli_score") or 0)
        body.append(
            f"\n      Also Read: [{_title(item.get('tier'))}] {name} {score:.2f} ", style="dim"
        )
        body.append(f'"{_excerpt(item)}"', style="dim")


def _title(value: Any) -> str:
    """`tool result` as `Tool Result`."""
    return str(value or "").title()


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
    """A document-shaped value as wrapped, readable lines."""
    if isinstance(value, Mapping):
        return [f"{key}: {item}" for key, item in value.items()]
    if isinstance(value, list):
        text = "\n".join(
            _message_line(item) if isinstance(item, Mapping) else str(item) for item in value
        )
    else:
        text = str(value)
    lines: list[str] = []
    for raw in text.splitlines() or [""]:
        # Wrapped on word boundaries. Cutting mid-word saves a column and
        # costs the reader the sentence.
        lines.extend(textwrap.wrap(raw, width=_BLOCK_WIDTH) or [""])
    return lines


def _message_line(message: Mapping[str, Any]) -> str:
    """One chat message, as `role: content`."""
    role = message.get("role", "?")
    content = message.get("content", "")
    return f"{role}: {content}"


def _format_mapping(value: Mapping[str, Any]) -> str:
    """Compact nested mapping for evidence/observed values."""
    parts = []
    for key, item in value.items():
        rendered = str(item)
        if isinstance(item, list):
            rendered = "[" + ", ".join(str(x) for x in item) + "]"
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def _status_style_for_result(result: str) -> str:
    normalized = result.upper()
    if normalized == "PASS":
        return "green"
    if normalized in {"FAIL", "ESCALATE"}:
        return "red" if normalized == "FAIL" else "yellow"
    return "dim"


class FeedNote(Static):
    """Non-event note such as user input, assistant response, or command output."""

    def __init__(
        self,
        text: str,
        style: str = "",
    ) -> None:
        super().__init__(Text(text, style=style), classes="feed-note")
        self.plain_text = text


def _merge_control_loop_events(
    previous: StatusEvent,
    current: StatusEvent,
) -> StatusEvent:
    """Coalesce adjacent control-loop facts into one operator card.

    The graph may emit the loop decision before and the reflection result
    after the model turn. They describe one control-loop round, so the TUI
    should render one card rather than two consecutive CONTROL LOOP sections.
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
        self._entries: list[StageEventWidget | FeedNote] = []

    def show_status(self, event: StatusEvent) -> None:
        """Render completed/warning/failed semantic events as cards.

        Started events are intentionally represented by the separate current
        status line. This keeps the timeline from showing ``→ X`` followed by
        ``✓ X`` as two unrelated records.
        """
        if event.status == "started":
            return

        if (
            event.stage == "control_loop"
            and event.event_kind == "control_loop"
            and self._entries
            and isinstance(self._entries[-1], StageEventWidget)
            and self._entries[-1].event.stage == "control_loop"
            and self._entries[-1].event.event_kind == "control_loop"
        ):
            previous = self._entries[-1]
            merged = _merge_control_loop_events(previous.event, event)
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

    def note(self, text: str, style: str = "") -> None:
        note = FeedNote(text, style)
        self._entries.append(note)
        self._trim_if_needed()
        self.mount(note)

    def show_lines(self, lines: list[Text]) -> None:
        for line in lines:
            self.note(line.plain, str(line.style or ""))

    def clear_feed(self) -> None:
        self._entries.clear()
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


def _field_style(event_kind: str, name: str) -> str:
    if name == "reflection_comment":
        return "yellow"
    if name in {"decision", "intent"}:
        return "magenta"
    if name in {"reason"}:
        return "dim"
    if event_kind in {"context", "memory"}:
        return "blue"
    if event_kind in {"security", "resilience", "escalation"}:
        return "yellow"
    return "white"


def _first_present(
    fields: Mapping[str, Any],
    names: set[str],
) -> Any | None:
    for name in FIELD_PRIORITY:
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
