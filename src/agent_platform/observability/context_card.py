"""The CONTEXT card: what a model is about to read, section by section, before it reads it.

One card per model request, drawn just before it: the request's sections as a tree (the
AGENT.md profile, the SKILL.md procedures, the policy, the context window broken into working
memory, long-term memory and retrieved chunks, the transcript, this turn's tool loop), each
with its tokens and share; the tool schemas; the stable prefix and its signature; the
tokenizer that counted. Below the tree, every section's full text — nothing elided, so the
card is the evidence, not a hint. The same fields go to the log files.

Pure formatting: a manifest in, a heading and narrative fields out.
"""

from collections import OrderedDict
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import ContextCardWords
from agent_platform.context.failure.confusion import tool_overload
from agent_platform.llm.prompt import PromptManifest, PromptSection, manifest_from_messages

CONTEXT_STAGE = "llm_context"
# (model, prefix signature) pairs sent before, oldest first: a bounded memory of what a
# provider's prefix cache could hold.
_SEEN_PREFIXES: OrderedDict[tuple[str, str], None] = OrderedDict()
_BRANCH, _LAST, _PIPE, _GAP = "├─ ", "└─ ", "│  ", "   "
_THOUSAND, _MILLION = 1_000, 1_000_000


def context_heading(manifest: PromptManifest, model: str, purpose: str) -> str:
    """`Context · reply · 2,140 tokens · 1.6% of window` (`~` and "estimated" when estimated)."""
    words = get_settings().observability.context_card
    template = words.heading if manifest.exact else words.heading_estimated
    window = window_tokens(model)
    return template.format(
        purpose=purpose or manifest.purpose,
        tokens=f"{manifest.total_tokens:,}",
        window_share=_share(manifest.total_tokens, window),
        window=window_size(window),
    )


def aggregate_card(purpose: str, requests: list[tuple[int, str]]) -> tuple[str, dict[str, Any]]:
    """Many requests of one purpose (a page read each) as one card: their range, the largest
    request's share of its window, and one line per request. `requests` is (tokens, model)."""
    words = get_settings().observability.context_card
    tokens = [count for count, _ in requests]
    peak_tokens, peak_model = max(requests)
    peak_window = window_tokens(peak_model)
    heading = words.aggregate_heading.format(
        purpose=purpose,
        count=len(requests),
        low=f"{min(tokens):,}",
        high=f"{max(tokens):,}",
        peak_share=_share(peak_tokens, peak_window),
        window=window_size(peak_window),
    )
    lines = [
        words.aggregate_line.format(
            tokens=f"{count:,}", share=_share(count, window_tokens(model)), model=model
        )
        for count, model in requests
    ]
    return heading, {words.aggregate_title: lines}


def window_size(tokens: int) -> str:
    """A window as a reader says it: `131k`, `1M`."""
    if tokens >= _MILLION:
        return f"{tokens / _MILLION:g}M"
    return f"{tokens / _THOUSAND:.0f}k" if tokens >= _THOUSAND else str(tokens)


def context_fields(manifest: PromptManifest, model: str) -> dict[str, Any]:
    """Narrative sections: the tree, the request's totals, then each section's full text."""
    words = get_settings().observability.context_card
    fields: dict[str, Any] = {words.sections_title: _tree(manifest, words)}
    fields[words.request_title] = _request_lines(manifest, model, words)
    if manifest.window_notes:
        fields[words.window_title] = list(manifest.window_notes)
    overload = request_findings(manifest)
    if overload:
        fields[words.findings_title] = [{"text": line, "tone": "warn"} for line in overload]
    for section in manifest.sections:
        title = f"{label(section.name)} · {section.tokens:,} tokens"
        fields[title] = section.content.splitlines() or [""]
    return fields


def announced_manifest(request: Any, purpose: str) -> PromptManifest | None:
    """The request's manifest (built from its messages if it came without one), now marked
    as drawn — or None when a card was already drawn for this request."""
    manifest = getattr(request, "manifest", None)
    if manifest is None:
        manifest = manifest_from_messages(request.messages, request.tools)
        request.manifest = manifest
    if manifest.announced:
        return None
    manifest.announced = True
    manifest.purpose = manifest.purpose or purpose
    return manifest


def note_prefix(manifest: PromptManifest, model: str) -> None:
    """Record whether `model` was sent this stable prefix before (the Cache strategy's test)."""
    signature = manifest.prefix_signature
    if not signature:
        return
    key = (model, signature)
    manifest.prefix_seen = key in _SEEN_PREFIXES
    _SEEN_PREFIXES[key] = None
    _SEEN_PREFIXES.move_to_end(key)
    while len(_SEEN_PREFIXES) > get_settings().context.prompt.prefix_memory:
        _SEEN_PREFIXES.popitem(last=False)


def request_findings(manifest: PromptManifest) -> list[str]:
    """What is wrong with the request as a whole (too many tools to choose among)."""
    found = tool_overload(len(manifest.tools), manifest.tool_schema_tokens, manifest.total_tokens)
    if found is None:
        return []
    return [get_settings().context.window.finding_line.format(**found.model_dump())]


def window_tokens(model: str) -> int:
    """How many tokens `model` can read, from `llm.context_windows`."""
    return get_settings().llm.context_windows.for_model(model)


def label(name: str) -> str:
    """What a person reads for a section name."""
    return get_settings().observability.context_card.labels.get(name, name)


def _tree(manifest: PromptManifest, words: ContextCardWords) -> list[str]:
    total = manifest.total_tokens
    lines: list[str] = []
    for index, section in enumerate(manifest.sections):
        last = index == len(manifest.sections) - 1 and not manifest.tool_schema_tokens
        lines.append(_section_line(section, total, words, last))
        lines.extend(_part_lines(section, words, _GAP if last else _PIPE))
    if manifest.tool_schema_tokens:
        tools = words.tools_line.format(
            tokens=f"{manifest.tool_schema_tokens:,}", count=len(manifest.tools)
        )
        lines.append(f"{_LAST}{tools}")
    return lines


def _section_line(section: PromptSection, total: int, words: ContextCardWords, last: bool) -> str:
    text = words.section_line.format(
        label=label(section.name), tokens=f"{section.tokens:,}", share=_share(section.tokens, total)
    )
    return f"{_LAST if last else _BRANCH}{text}"


def _part_lines(section: PromptSection, words: ContextCardWords, indent: str) -> list[str]:
    lines: list[str] = []
    for index, part in enumerate(section.parts):
        branch = _LAST if index == len(section.parts) - 1 else _BRANCH
        text = words.part_line.format(label=label(part.name), tokens=f"{part.tokens:,}")
        if part.source and part.source != part.name:
            text = f"{text} · {words.source_line.format(source=part.source)}"
        lines.append(f"{indent}{branch}{text}")
    return lines


def _request_lines(manifest: PromptManifest, model: str, words: ContextCardWords) -> list[str]:
    prefix = (
        words.prefix_line.format(
            tokens=f"{manifest.prefix_tokens:,}", signature=manifest.prefix_signature
        )
        if manifest.prefix_signature
        else words.prefix_none
    )
    if manifest.prefix_seen is not None:
        prefix = f"{prefix} · {words.prefix_reused if manifest.prefix_seen else words.prefix_new}"
    return [
        words.model_line.format(model=model),
        prefix,
        words.tokenizer_line.format(tokenizer=manifest.tokenizer),
    ]


def _share(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "0.0%"


__all__ = [
    "CONTEXT_STAGE",
    "aggregate_card",
    "window_size",
    "context_fields",
    "context_heading",
    "announced_manifest",
    "label",
    "window_tokens",
]
