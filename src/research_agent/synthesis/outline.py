"""The report's outline, held to its limits in code.

The writer chooses its own headings: a report shaped by the question fits better than one
poured into fixed sections. What it may not choose is enforced here, after the draft, so no
prompt wording is load-bearing:

- a section with nothing in it is dropped;
- a heading the run writes itself (Sources, Coverage & Limits…) is reserved: dropped;
- a heading used twice is one section, its bodies joined;
- past `max_sections`, each further section is folded into the last one kept, under its own
  heading as a line of its body, so nothing the writer cited is lost.
"""

from collections.abc import Collection

from research_agent.domain.report import ReportSection


def fit_outline(
    sections: list[ReportSection], reserved: Collection[str], max_sections: int, folded: str
) -> list[ReportSection]:
    """`sections` as the outline rules allow; `folded` words a folded heading (`{heading}`)."""
    banned = {heading.casefold() for heading in reserved}
    kept: dict[str, ReportSection] = {}
    for section in sections:
        key = section.heading.strip().casefold()
        if not section.body.strip() or key in banned:
            continue
        kept[key] = _joined(kept[key], section.body) if key in kept else section
    return _capped(list(kept.values()), max_sections, folded)


def _capped(sections: list[ReportSection], limit: int, folded: str) -> list[ReportSection]:
    if len(sections) <= limit:
        return sections
    head, last, rest = sections[: limit - 1], sections[limit - 1], sections[limit:]
    for extra in rest:
        last = _joined(last, f"{folded.format(heading=extra.heading.strip())}\n{extra.body}")
    return [*head, last]


def _joined(section: ReportSection, body: str) -> ReportSection:
    return section.model_copy(update={"body": f"{section.body.rstrip()}\n\n{body.strip()}"})


__all__ = ["fit_outline"]
