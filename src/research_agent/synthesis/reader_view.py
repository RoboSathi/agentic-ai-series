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

from research_agent.config.settings import ReaderViewWording
from research_agent.domain.report import Citation, Report
from research_agent.domain.sources import host_of

_ADJACENT = r"(?:\s*{c})+"
_LEAD = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


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
    wording: ReaderViewWording,
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

    lines = [f"# {report.title}", _status_line(report, len(numbers), wording)]
    if report.notice:
        lines += ["", report.notice]
    if blocks:
        lines += ["", *("\n\n".join(blocks).splitlines())]
    else:
        lines += ["", wording.no_evidence]
    if numbers:
        lines += [
            "",
            wording.sources_heading,
            *_sources(numbers, cited, source_labels or {}, wording),
        ]
    lines += _not_established(report, wording)
    lines += _not_read(report, wording)
    return "\n".join(line.rstrip() for line in lines).strip()


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


def _status_line(report: Report, sources: int, wording: ReaderViewWording) -> str:
    noun = wording.source_singular if sources == 1 else wording.source_plural
    return wording.status_line.format(status=report.status, count=sources, noun=noun)


def _sources(
    numbers: Mapping[str, int],
    cited: Mapping[str, Citation],
    labels: Mapping[str, Sequence[str]],
    wording: ReaderViewWording,
) -> list[str]:
    retrieved = {c.url: c.retrieved_at.date().isoformat() for c in cited.values()}
    caveats = {c.url: c.caveat for c in cited.values() if c.caveat}
    lines = []
    for url, n in sorted(numbers.items(), key=lambda item: item[1]):
        kinds = [kind for kind in (label_for(url, labels), caveats.get(url, "")) if kind]
        lines.append(
            wording.source_line.format(
                number=n,
                host=host_of(url),
                label=f" ({'; '.join(kinds)})" if kinds else "",
                url=url,
                date=retrieved.get(url, wording.unknown_date),
            )
        )
    return lines


def _not_established(report: Report, wording: ReaderViewWording) -> list[str]:
    lines = [
        wording.not_established_line.format(
            text=limit.text, tried=", ".join(limit.tried_queries) or wording.nothing_tried
        )
        for limit in report.limits
    ]
    lines += [wording.stopped_early.format(note=note) for note in report.budget_notes]
    return ["", wording.not_established_heading, *lines] if lines else []


def _not_read(report: Report, wording: ReaderViewWording) -> list[str]:
    if not report.rejected_codes:
        return []
    by_host: dict[str, set[str]] = {}
    for url, code in report.rejected_codes.items():
        by_host.setdefault(host_of(url), set()).add(wording.reasons.get(code, wording.not_used))
    shown = [f"{host} ({', '.join(sorted(why))})" for host, why in sorted(by_host.items())]
    cap = wording.max_not_read
    more = wording.more_hosts.format(count=len(shown) - cap) if len(shown) > cap else ""
    return ["", wording.not_read.format(hosts=", ".join(shown[:cap]), more=more)]
