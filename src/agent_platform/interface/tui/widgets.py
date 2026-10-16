"""Widgets for the customer-support agent operations dashboard.

The TUI consumes semantic ``StatusEvent`` objects.  It never consumes the
logger's method-level trace stream.  Each completed semantic event becomes a
small collapsible card:

    collapsed:  what happened?
    expanded:   what useful semantic evidence explains it?

The logger remains responsible for implementation-level debugging.
"""

import re
import textwrap
import time
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Any, cast

from rich.text import Text
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Collapsible, OptionList, Static
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets.button import ButtonVariant
from textual.widgets.option_list import Option

from agent_platform.config import get_settings
from agent_platform.hallucination_check.levels import CONFIDENCE_LEVEL as _FIRST_LEVEL
from agent_platform.interface.presentation.text_tree import as_structure, message_lines, value_lines
from agent_platform.interface.tui.card_format import field_priority as _field_priority
from agent_platform.interface.tui.card_format import format_value as _format_value
from agent_platform.interface.tui.card_format import humanize as _humanize
from agent_platform.interface.tui.card_format import label as _label
from agent_platform.interface.tui.card_summaries import summarise
from agent_platform.interface.tui.labels import StageLabelRegistry
from agent_platform.llm.types import TokenUsage
from agent_platform.observability.types import StatusEvent

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


# How a claim's verdict is marked on a card; the tally wording comes with
# the guardrail result, so title, log and card agree.
# The middle value is the claim's tone (`tui.tone_styles`), not a colour.
_CLAIM_MARKS = {
    "supported": ("✓", "pass", "supported"),
    "undetermined": ("?", "warn", "unverified"),
    "unsupported": ("✗", "fail", "unsupported"),
    "low_confidence": ("✗", "fail", "low confidence"),
    "refuted": ("✗", "fail", "contradicted"),
    "rewritten": ("↻", "warn", "rewritten"),
    "removed": ("✗", "fail", "removed"),
}
_UNDETERMINED = "undetermined"
_UNSUPPORTED = "unsupported"
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

    def add(self, usage: TokenUsage) -> None:
        tokens = usage.total_tokens
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
        rendered = self._append_fields(body)
        if not rendered:
            self._append_summary(body)
        self._append_tokens(body, rendered)
        if not body:
            body.append(get_settings().tui.card_body.empty, style=palette("muted"))
        return body

    def _append_fields(self, body: Text) -> bool:
        """Every visible field, in card order; whether there was any."""
        rendered = False
        for name, value in self._ordered_fields(self.event.fields):
            if rendered:
                body.append("\n")
            if name == "checks" and isinstance(value, list):
                self._append_checks(body, value)
            elif value and (name in BLOCK_FIELDS or _is_structured(value)):
                self._append_block(body, name, value)
            else:
                body.append("  ")
                _append_pair(body, _label(name), " = ", _format_value(name, value))
            rendered = True
        return rendered

    def _append_summary(self, body: Text) -> None:
        """With no fields to show, the summary or detail (the title has the summary too)."""
        if self.event.summary:
            body.append(self.event.summary, style=palette("emphasis"))
        elif self.event.detail:
            body.append(get_settings().tui.card_body.detail_prefix)
            body.append(_humanize(self.event.detail))

    def _append_tokens(self, body: Text, rendered: bool) -> None:
        if not self.event.tokens_delta:
            return
        if rendered or self.event.summary or self.event.detail:
            body.append("\n")
        body.append("  ")
        label = get_settings().tui.card_body.tokens_label
        _append_pair(body, label, " = ", f"+{self.event.tokens_delta}")

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
        return summarise(self.event, self._labels)

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
        unit = settings.timing_unit
        _append_pair(body, "took", " ", f"{_format_value(NARRATIVE_TIMING_FIELD, took)} {unit}")
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
        # A nested value is a tree under its key, never a Python or JSON literal.
        return [(line, "") for line in value_lines(dict(value))]
    if isinstance(value, (list, tuple)):
        return [_narrative_line(item) for item in value]
    return [(line, "") for line in str(value).splitlines() or [""]]


def _narrative_line(item: Any) -> tuple[str, str]:
    if isinstance(item, Mapping):
        return str(item.get("text", "")), str(item.get("tone", ""))
    return str(item), ""


def _append_grounding(body: Text, fields: Mapping[str, Any]) -> None:
    """The hallucination check: what was sent, how sure the model was, what decided each claim.

    Three parts and nothing else. The result row says what reached the customer. The
    confidence block shows each sentence's content-token mean, coloured by band, with
    the words the model was least sure of marked in the sentence itself. Then one
    line per claim: its verdict, what decided it (the record, a rule, NLI, logprobs),
    and the line of evidence that did. The claim tally is the title; every passage
    retrieved, every score, the evidence counts and the draft check are in
    `guardrails.log`. Level 1 is the coloured confidence block when the payload carries
    the token confidences, else its row; every later level (the agent's own grounding)
    follows as its row, so every agent's card reads result → level 1 → level 2 → claims.
    """
    words = get_settings().tui.grounding_card
    _append_grounding_row(body, words.result, "")
    _append_outcome_value(body, str(fields.get("outcome") or ""))
    confidence = fields.get("confidence")
    coloured = isinstance(confidence, Mapping) and bool(confidence)
    if isinstance(confidence, Mapping) and coloured:
        _append_confidence(body, confidence)
    for level in fields.get("levels") or []:
        # Level 1 is the confidence block when there is one; every other level is its row.
        if isinstance(level, Mapping) and not (coloured and level.get("level") == _FIRST_LEVEL):
            _append_level(body, level)
    rule = str(fields.get("rule") or "")
    if rule:
        _append_grounding_row(body, words.rule, rule)
    checks = [c for c in fields.get("claim_checks") or [] if isinstance(c, Mapping)]
    if checks:
        _append_heading(body, words.claims)
    for check in checks:
        _append_claim(body, check)
    corrected = [str(line) for line in fields.get("corrected_sentences") or []]
    if corrected:
        _append_grounding_row(body, words.corrected, " ".join(corrected))


def _append_heading(body: Text, heading: str) -> None:
    """A section of the card (`claims`), in the key colour, with what it lists below it."""
    body.append("\n  " if body else "  ")
    body.append(heading, style=get_settings().tui.body_styles["key"])


def _append_outcome_value(body: Text, outcome: str) -> None:
    """`TRIMMED · removed 1 of 2 …` after its key: only the state word takes its tone."""
    word, separator, rest = outcome.partition(" ")
    body.append(word, style=_state_style(word) or get_settings().tui.body_styles["value"])
    body.append(separator + rest, style=get_settings().tui.body_styles["value"])


def _append_confidence(body: Text, confidence: Mapping[str, Any]) -> None:
    """Each sentence by the model's confidence in it: its mean coloured, its weak words marked."""
    words = get_settings().tui.grounding_card
    if not confidence.get("available"):
        reason = str(confidence.get("reason") or "")
        body.append(
            "\n  " + words.confidence_skipped.format(reason=reason),
            style=_tone("dim"),
        )
        return
    sentences = [s for s in confidence.get("sentences") or [] if isinstance(s, Mapping)]
    means = [float(s.get("content_mean") or 0.0) for s in sentences]
    key = words.confidence.format(model=confidence.get("model"))
    if means:
        lowest = min(range(len(means)), key=means.__getitem__)
        summary = words.logprobs_summary.format(
            mean=sum(means) / len(means), lowest=means[lowest], sentence=lowest + 1
        )
        body.append("\n  " if body else "  ")
        _append_pair(body, key, " = ", summary, _band_style(means[lowest]))
    else:
        _append_heading(body, key)
    for sentence in sentences:
        _append_confident_sentence(body, sentence)


def _append_confident_sentence(body: Text, sentence: Mapping[str, Any]) -> None:
    """`94% Order 10245 has been cancelled…`, the percent by band, each weak token by its tone."""
    mean = float(sentence.get("content_mean") or 0.0)
    body.append(f"\n    {mean:>4.0%} ", style=_band_style(mean))
    text = str(sentence.get("sentence") or "")
    position = 0
    for token in sentence.get("tokens") or []:
        if not isinstance(token, Mapping):
            continue
        start, end = int(token.get("start") or 0), int(token.get("end") or 0)
        if start < position or end <= start:
            continue
        body.append(text[position:start], style=get_settings().tui.body_styles["value"])
        body.append(text[start:end], style=_token_style(token))
        position = end
    body.append(text[position:], style=get_settings().tui.body_styles["value"])


def _band_style(prob: float) -> str:
    """A confidence's tone: sure (pass), hesitant (warn), guessed (fail)."""
    bands = get_settings().tui.grounding_card.confidence_bands
    tone = "pass" if prob >= bands["pass"] else "warn" if prob >= bands["warn"] else "fail"
    return _tone(tone)


def _token_style(token: Mapping[str, Any]) -> str:
    """A token the model was unsure of takes a tone; a guessed specific is marked as a failure."""
    prob = float(token.get("prob") or 0.0)
    slot_floor = get_settings().hallucination_check.confidence.slot_min_prob
    if token.get("slot") and prob < slot_floor:
        return f"{_tone('fail')} underline"
    if prob < get_settings().tui.grounding_card.confidence_bands["warn"]:
        return _tone("warn")
    return get_settings().tui.body_styles["value"]


def _append_level(body: Text, level: Mapping[str, Any]) -> None:
    """`level 1 · token confidence = …`, toned by its verdict, then its proof lines."""
    words = get_settings().hallucination_check.levels
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


def _append_claim(body: Text, check: Mapping[str, Any]) -> None:
    """`✗ claim [CONTRADICTED]`, as a check line, then the steps that decided it.

    The steps are the check's own (`hallucination_check/readings.py`), drawn as they come;
    a claim no step read says why instead. Everything else about how the claim was checked
    is in `guardrails.log`.
    """
    styles = get_settings().tui.body_styles
    verdict = str(check.get("verdict") or _UNDETERMINED)
    mark, tone, name = _CLAIM_MARKS.get(verdict, ("?", "warn", verdict))
    style = _tone(tone)
    body.append(f"\n    {mark} ", style=style)
    body.append(str(check.get("claim") or ""), style=styles["value"])
    body.append(f" [{name.upper()}]", style=style)
    steps = [r for r in check.get("readings") or [] if isinstance(r, Mapping)]
    if steps:
        # Every step that ran, in order: what read the claim, what it said, what it decided.
        _append_steps(body, steps, str(check.get("claim") or ""))
        return
    # No step read it (a figure no query row held): the check's own reason, else that
    # nothing spoke to it. Which line decided a claim is the check's to say, never the card's.
    words = get_settings().tui.grounding_card
    nothing = words.no_evidence if verdict == _UNSUPPORTED else words.no_evidence_either_way
    _append_proof(body, name, str(check.get("rationale") or "") or nothing, styles["value"])


def _append_steps(body: Text, steps: list[Mapping[str, Any]], claim: str) -> None:
    """The claim's checks as a tree, one line per piece of evidence each step read:

        ✓ Reflection drafts, critiques and revises. [SUPPORTED]
          └ NLI DeBERTa entailment 0.945 · www.scaler.com/topics/…: "Reflection means…"
        ✓ Kumar Karan is a Lead Instructor at Scaler. [SUPPORTED]
          ├ NLI DeBERTa neutral 1.000 → not decided · robosathi.com/about: "Kumar Karan"
          └ Laya supported 0.512 ≥ 0.45 · robosathi.com/about: read 24 lines · "…"

    The claim line says the verdict once; the step that reached it says only what it read and
    how sure it was, and an earlier step that did not decide says what it decided instead.
    """
    lines = [
        (step, ref, index == len(steps) - 1)
        for index, step in enumerate(steps)
        for ref in _step_refs(step)
    ]
    for position, (step, ref, deciding) in enumerate(lines):
        _append_step(body, step, ref, claim, deciding, last=position == len(lines) - 1)


def _step_refs(step: Mapping[str, Any]) -> list[Mapping[str, Any] | None]:
    """The evidence a step read, or one empty slot for a step that read nothing."""
    refs: list[Mapping[str, Any] | None] = [
        r for r in step.get("evidence") or [] if isinstance(r, Mapping)
    ]
    return refs or [None]


def _append_step(
    body: Text,
    step: Mapping[str, Any],
    ref: Mapping[str, Any] | None,
    claim: str,
    deciding: bool,
    *,
    last: bool,
) -> None:
    """One line of the tree: what read the claim, its answer, and the evidence it read."""
    styles = get_settings().tui.body_styles
    words = get_settings().tui.grounding_card
    tone = _tone(str(step.get("tone") or "dim"))
    decision = str(step.get("decision") or "")
    head = _basis_of(step)
    if not deciding:
        head += words.step_outcome.format(decision=decision)
    body.append(f"\n      {words.step_last if last else words.step_branch} ", style=tone)
    if ref is None:
        # Nothing read ("NLI DeBERTa: no related evidence").
        body.append(str(step.get("check") or ""), style=tone)
        body.append(f": {decision}", style=styles["muted"])
        return
    if head.strip():
        body.append(head.strip(), style=tone)
        body.append(" · ", style=styles["muted"])
    body.append(str(ref.get("where") or ""), style=styles["value"])
    lines = [line for line in str(ref.get("excerpt") or "").splitlines() if line.strip()]
    if not lines:
        return
    read = words.lines_read.format(count=len(lines))
    quoted = _closest_line(lines, claim) if len(lines) > 1 else _excerpt(ref)
    prefix = f": {read} · " if len(lines) > 1 else ": "
    body.append(f'{prefix}"{quoted}"', style=styles["muted"])


def _basis_of(step: Mapping[str, Any]) -> str:
    """What decided it, in brackets: a model and its score, or the rule that read the record.

    A plain record read needs no bracket: the evidence line already says it is the record.
    """
    check, result = str(step.get("check") or ""), str(step.get("result") or "")
    if result:
        return f"{check} {result}"
    return "" if check == get_settings().hallucination_check.methods.record else check


def _append_proof(body: Text, key: str, value: str, style: str) -> None:
    """The indented `key: value` line under a claim saying what decided it."""
    body.append("\n      ")
    _append_pair(body, key, ": ", value, style)


def _closest_line(lines: list[str], claim: str) -> str:
    """The line sharing most words with the claim, clipped like any excerpt."""
    words = set(re.findall(r"\w+", claim.casefold()))
    best = max(lines, key=lambda line: len(words & set(re.findall(r"\w+", line.casefold()))))
    return _excerpt({"excerpt": best})


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
    """One chat message, as `role: content` (a tool's JSON result as a tree), then any tool
    calls it made as `→ tool(key=value, …)` — never JSON."""
    role = message.get("role", "?")
    answering = message.get("tool")
    head = f"{role} ({answering})" if answering else str(role)
    calls = [(str(c.get("tool")), c.get("arguments")) for c in message.get("tool_calls") or []]
    return "\n".join(message_lines(head, message.get("content", ""), calls))


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

    def offer(self, choices: Sequence[Any], shown: Callable[[str], str] = str) -> None:
        """Show these choices (anything with a `label` and a `value`); none hides the menu.

        `shown` is applied to each label as it is displayed; the value is recorded as given.
        """
        self.clear_options()
        self._values = [str(_choice_part(choice, "value")) for choice in choices]
        self.add_options([Option(shown(str(_choice_part(choice, "label")))) for choice in choices])
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

    class Wheeled(Message):
        """The person turned the wheel over the feed: they are reading, not following."""

    def on_mouse_scroll_up(self, _event: events.MouseScrollUp) -> None:
        self.post_message(self.Wheeled())

    def on_mouse_scroll_down(self, _event: events.MouseScrollDown) -> None:
        self.post_message(self.Wheeled())

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


def palette(role: str) -> str:
    """The configured style for one console role (`tui.palette`): banner, metrics, muted, ..."""
    return get_settings().tui.palette[role]


def _tone(name: str) -> str:
    """A state's style (`tui.tone_styles`): pass, fail, warn, dim."""
    return get_settings().tui.tone_styles[name]


def _compact(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"
