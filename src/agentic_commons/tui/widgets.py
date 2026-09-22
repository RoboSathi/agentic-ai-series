"""Widgets for the customer-support agent operations dashboard.

The TUI consumes semantic ``StatusEvent`` objects.  It never consumes the
logger's method-level trace stream.  Each completed semantic event becomes a
small collapsible card:

    collapsed:  what happened?
    expanded:   what useful semantic evidence explains it?

The logger remains responsible for implementation-level debugging.
"""

from collections.abc import Mapping
from typing import Any
import time

from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Static
from textual.reactive import reactive

from agentic_commons.config import get_settings
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
    "termination_reason",
    "claims",
    "supported",
    "unsupported",
    "hallucination_rate",
    "response_ready",
    "revised",
    "reflection",
    "reflection_detail",
    "grounding_issues",
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
}

AMOUNT_FIELDS = {
    "amount",
    "requested_amount",
    "allowed_amount",
    "autonomous_limit",
}

# Fields that can be noisy / sensitive for the TUI.  They are intentionally
# excluded even if an upstream event contains them.
MUTED_FIELDS = {
    "args",
    "kwargs",
    "content",
    "context",
    "messages",
    "text",
    "value",
    "chunks_raw",
    "facts_raw",
    "payload",
    "prompt",
    "trace_id",
    "evidence",
    "instructions",
    "raw_arguments",
    "session_id",
    "source_payload",
}


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

        super().__init__(
            Static(body, classes="stage-details"),
            title=title,
            collapsed=True,
            collapsed_symbol="›",
            expanded_symbol="⌄",
            classes=f"stage-event stage-{self._status}",
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

        fields = self._ordered_fields(self.event.fields)

        rendered_fields = False
        for name, value in fields:
            if name == "checks" and isinstance(value, list):
                if rendered_fields:
                    body.append("\n")
                self._append_checks(body, value)
                rendered_fields = True
                continue
            if name == "three_gates" and isinstance(value, Mapping):
                if rendered_fields:
                    body.append("\n")
                self._append_three_gates(body, value)
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
            return self._compact_from_fields(fields, ("redacted", "entities"))

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

        if stage in {"memory_recall", "working_memory_check"}:
            count = fields.get("facts_held", fields.get("selected", fields.get("count")))
            return f"recalled {count} facts" if count is not None else "memory checked"

        if stage in {"memory_write", "working_memory_write"}:
            operation = fields.get("operation", "write")
            key = fields.get("key")
            return f"{operation} · {key}" if key else str(operation)

        if stage == "context_assembly":
            chunks = fields.get("chunks")
            used = fields.get("used_tokens")
            if chunks is not None and used is not None:
                return f"{chunks} sources · {used} tokens"
            return self._compact_from_fields(fields, ("selected", "used_tokens"))

        if stage in {"llm_call", "llm_response_received"}:
            requested_calls = fields.get("requested_tool_calls")
            if isinstance(requested_calls, list) and requested_calls:
                first = requested_calls[0]
                if isinstance(first, Mapping):
                    tool = first.get("tool") or first.get("qualified_tool")
                    order_id = first.get("order_id")
                    amount = first.get("amount")
                    parts = [
                        "tool requested",
                        self._labels.tool_label(str(tool)) if tool else "tool",
                    ]
                    if order_id is not None:
                        parts.append(f"order {order_id}")
                    if amount is not None:
                        parts.append(_format_value("amount", amount))
                    return " · ".join(parts)
            requested = fields.get("requested_tools")
            if requested:
                first = requested[0] if isinstance(requested, list) else requested
                return f"tool requested · {self._labels.tool_label(str(first))}"
            if fields.get("finish_reason") == "tool_calls":
                return "tool requested"
            return "response generated"

        if stage == "control_loop":
            decision = fields.get("decision")
            reason_code = fields.get("reason_code")
            if reason_code == "DUPLICATE_ACTION":
                return "SUPPRESS · DUPLICATE_ACTION"
            if reason_code == "MAX_TOOL_ITERATIONS":
                return "ABORT · MAX_TOOL_ITERATIONS"
            if decision:
                return str(decision).upper()
            return "control decision"

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
                    (item for item in checks if isinstance(item, Mapping) and item.get("status") in {"FAIL", "ESCALATE"}),
                    None,
                )
                if isinstance(failed, Mapping):
                    parts.append(str(failed.get("reason_code") or failed.get("check_id") or "blocked"))
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
            supported = fields.get("supported")
            claims = fields.get("claims")
            if fields.get("reason_code") == "HALLUCINATION_GUARD_BLOCK":
                return "BLOCK · unsupported claims after repair"
            if supported is not None and claims is not None:
                suffix = " · repaired" if fields.get("repaired") else ""
                return f"{supported}/{claims} claims supported{suffix}"

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

    def _append_three_gates(self, body: Text, gates: Mapping[str, Any]) -> None:
        """Render the three gate dimensions in the same order as the policy."""
        body.append("  THREE GATES", style="bold bright_white")
        labels = (
            ("path_cardinality", "Path cardinality", "path_result"),
            ("blast_radius", "Blast radius", "blast_radius_result"),
            ("certifiable", "Certifiability", "certifiability_result"),
        )
        for key, label, result_key in labels:
            value = gates.get(key)
            result = gates.get(result_key)
            body.append("\n    ")
            body.append(label, style="bright_white")
            body.append(f" = {_format_value(key, value)}", style="white")
            if result:
                body.append(f" · {result}", style=_status_style_for_result(str(result)))

    def _compact_from_fields(
        self,
        fields: Mapping[str, Any],
        names: tuple[str, ...],
    ) -> str:
        parts = []
        for name in names:
            value = fields.get(name)
            if value is not None:
                parts.append(
                    f"{FIELD_LABELS.get(name, name)}="
                    f"{_format_value(name, value)}"
                )
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
        visible = [
            (name, value)
            for name, value in fields.items()
            if name not in MUTED_FIELDS
        ]
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
        if isinstance(value, (int, float)):
            return f"${value:,.2f}"
        try:
            number = float(str(value).replace("$", "").replace(",", ""))
            return f"${number:,.2f}"
        except (TypeError, ValueError):
            pass

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(str(item) for item in value) + "]"

    if isinstance(value, dict):
        items = ", ".join(
            f"{key}={_format_value(str(key), item)}"
            for key, item in value.items()
        )
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
