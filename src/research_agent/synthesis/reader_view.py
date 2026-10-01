"""The reader's view of a report: numbered sources, one finding per line, no bookkeeping.

The archived report keeps what an auditor needs — evidence ids (`[E3]`), the quoted span,
the page hash. A person reading the answer needs something else: `[E1][E2][E3]` are three
claims from one page, and nothing in the answer says what they point to. This module turns
a finished `Report` into text built for reading:

- the text is light Markdown (`#`, `##`, `**bold**`) so a console can colour it by role;
- every cited page becomes one numbered source, in order of first citation, so
  `[E1][E2][E3]` reads `[1]`;
- a Sources list maps each number to a domain, URL and retrieval date;
- each finding is its own bullet, so the answer can be scanned;
- bookkeeping sections that say nothing ("Known unknowns: none") are left out;
- pages that could not be read are grouped by domain and reason, not listed by URL.

It is deterministic and runs after the audit, so it can never change what was verified:
the numbers are derived from the report's own citations, and every `[n]` resolves.
"""

import re
from collections.abc import Collection, Mapping, Sequence
from urllib.parse import urlparse

from research_agent.domain.report import Citation, Report

_ADJACENT = r"(?:\s*{c})+"
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MAX_NOT_READ = 8

_NOT_READ = {
    "recency": "outside the date window",
    "undated": "no readable date",
    "blocked": "blocked or paywalled",
    "dead_link": "not found",
    "error": "could not be fetched",
    "empty": "no readable text",
    "injection": "flagged as unsafe",
}


def host_of(url: str) -> str:
    host = (urlparse(url).hostname or url).casefold()
    return host.removeprefix("www.")


def label_for(url: str, labels: Mapping[str, Sequence[str]]) -> str:
    """The configured kind of source a URL is ("press release"), or ''."""
    host = (urlparse(url).hostname or "").casefold()
    for label, suffixes in labels.items():
        if any(host == s.casefold() or host.endswith(f".{s.casefold()}") for s in suffixes):
            return label
    return ""


def render_reader_view(
    report: Report,
    *,
    citation_pattern: str,
    hidden_headings: Collection[str],
    source_labels: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """The report as text for a person; the archived report is unchanged."""
    pattern = re.compile(citation_pattern)
    cited: dict[str, Citation] = {c.entry_id: c for c in report.citations}
    numbers: dict[str, int] = {}  # source URL -> its number, by first citation

    def number(entry_id: str) -> int | None:
        citation = cited.get(entry_id)
        if citation is None:
            return None
        return numbers.setdefault(citation.url, len(numbers) + 1)

    blocks: list[str] = []
    bodies = [s for s in report.sections if s.heading not in hidden_headings and s.body.strip()]
    for section in bodies:
        items = [
            _with_numbers(item, pattern, number) for item in split_findings(section.body, pattern)
        ]
        items = [item for item in items if item]
        if not items:
            continue
        head = [] if len(bodies) == 1 else [f"## {section.heading}"]
        blocks.append("\n".join([*head, *(f"• {item}" for item in items)]))

    lines = [f"# {report.title}", _status_line(report, len(numbers))]
    if blocks:
        lines += ["", *("\n\n".join(blocks).splitlines())]
    else:
        lines += ["", "No usable evidence was found."]
    if numbers:
        lines += ["", "## Sources", *_sources(numbers, cited, source_labels or {})]
    lines += _not_established(report)
    lines += _not_read(report)
    return "\n".join(lines).strip()


def split_findings(body: str, pattern: re.Pattern[str]) -> list[str]:
    """One finding per line or sentence; a fragment with no citation joins the one before.

    Joining keeps "U.S. officials said …" whole, because a sentence ends where its citation
    does, not at every full stop.
    """
    findings: list[str] = []
    leading = re.compile(rf"^(?:\s*(?:{pattern.pattern}))+")
    for line in body.splitlines():
        line = _LEAD.sub("", line).strip()
        if not line:
            continue
        for fragment in re.split(r"(?<=[.!?])\s+", line):
            lead = leading.match(fragment)
            if lead and findings:
                # A marker that follows the full stop belongs to the sentence before it.
                findings[-1] = f"{findings[-1]} {lead.group(0).strip()}"
                fragment = fragment[lead.end() :].strip()
                if not fragment:
                    continue
            if findings and not pattern.search(findings[-1]):
                findings[-1] = f"{findings[-1]} {fragment}"
            else:
                findings.append(fragment)
    return findings


def _with_numbers(text: str, pattern: re.Pattern[str], number) -> str:  # type: ignore[no-untyped-def]
    """Replace each run of adjacent evidence ids with its distinct source numbers."""
    run = re.compile(_ADJACENT.format(c=pattern.pattern))

    def replace(match: re.Match[str]) -> str:
        ids = pattern.findall(match.group(0))
        seen: list[int] = []
        for entry_id in ids:
            n = number(entry_id)
            if n is not None and n not in seen:
                seen.append(n)
        return " " + "".join(f"[{n}]" for n in seen) if seen else ""

    return run.sub(replace, text).strip()


def _status_line(report: Report, sources: int) -> str:
    noun = "source" if sources == 1 else "sources"
    return f"**Status:** {report.status} · {sources} {noun}"


def _sources(
    numbers: Mapping[str, int], cited: Mapping[str, Citation], labels: Mapping[str, Sequence[str]]
) -> list[str]:
    retrieved = {c.url: c.retrieved_at.date().isoformat() for c in cited.values()}
    lines = []
    for url, n in sorted(numbers.items(), key=lambda item: item[1]):
        kind = label_for(url, labels)
        suffix = f" ({kind})" if kind else ""
        lines.append(f"[{n}] {host_of(url)}{suffix} — {url} — retrieved {retrieved.get(url, '?')}")
    return lines


def _not_established(report: Report) -> list[str]:
    lines = [
        f"- {limit.text} Tried: {', '.join(limit.tried_queries) or 'nothing'}."
        for limit in report.limits
    ]
    lines += [f"- Stopped early: {note}." for note in report.budget_notes]
    return ["", "## Not established", *lines] if lines else []


def _not_read(report: Report) -> list[str]:
    if not report.rejected_codes:
        return []
    by_host: dict[str, set[str]] = {}
    for url, code in report.rejected_codes.items():
        by_host.setdefault(host_of(url), set()).add(_NOT_READ.get(code, "not used"))
    shown = [f"{host} ({', '.join(sorted(why))})" for host, why in sorted(by_host.items())]
    extra = f", +{len(shown) - _MAX_NOT_READ} more" if len(shown) > _MAX_NOT_READ else ""
    return ["", f"**Not read:** {', '.join(shown[:_MAX_NOT_READ])}{extra}"]
