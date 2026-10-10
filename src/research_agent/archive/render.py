"""Markdown renderings of the archived artifacts, for people.

The JSON artifact (with its hash) is what a machine re-checks; these are
what a person reads. Citations carry URL, retrieval time and page hash,
so the rendered report stays auditable after the ledger is discarded.
"""

from research_agent.domain.report import Report
from research_agent.domain.rubric import Contract


def render_report(report: Report) -> str:
    lines = [f"# {report.title}", "", f"**Status:** {report.status}", ""]
    if report.notice:
        lines.extend([f"> {report.notice}", ""])
    for section in report.sections:
        lines.extend([f"## {section.heading}", "", section.body, ""])
    lines.extend(["## Citations", ""])
    lines.extend(
        f"- [{c.entry_id}] {c.url} — retrieved {c.retrieved_at.isoformat()} — page sha256 {c.page_hash[:16]}… — “{c.span}”"
        + (f" — {c.caveat}" if c.caveat else "")
        for c in report.citations
    )
    return "\n".join(lines).rstrip() + "\n"


def render_brief(contract: Contract) -> str:
    brief = contract.brief
    lines = [
        f"# Scope Brief — {brief.topic}",
        "",
        f"- Archetype: {brief.archetype}",
        f"- Purpose: {brief.purpose}",
    ]
    for label, values in (
        ("Options", brief.options),
        ("Criteria", brief.criteria),
        ("Claims", brief.claims),
        ("Concepts", brief.concepts),
        ("Constraints", brief.constraints),
    ):
        if values:
            lines.append(f"- {label}: {'; '.join(values)}")
    lines.extend(f"- Known unknown: {u.question} ({u.reason})" for u in brief.known_unknowns)
    lines.append(f"- Contract hash: {contract.contract_hash}")
    return "\n".join(lines) + "\n"


def render_rubric(contract: Contract) -> str:
    lines = ["# Rubric", ""]
    lines.extend(
        f"- {i.item_id} [{'must' if i.must_pass else 'soft'}] {i.text} ({i.check.kind}:{i.check.name} ≥ {i.threshold:g})"
        for i in contract.rubric.items
    )
    return "\n".join(lines) + "\n"
