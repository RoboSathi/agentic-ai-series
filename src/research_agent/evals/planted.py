"""Planted flaws: a delivered report, deliberately broken one way (Strategy).

Each planter returns a copy of the report with one defect the Goldfish
Audit exists to catch — an unsupported sentence, a citation to evidence
that does not exist, a criterion quietly dropped, an unmet item passed
off as covered, the trade-offs left out. Whether the audit catches it is
what the oracle's `planted_flaw_caught` scores.
"""

from collections.abc import Callable

from research_agent.domain.report import COVERAGE_AND_LIMITS, Report, ReportSection
from research_agent.evals.dataset.schema import FlawKind, PlantedFlaw

Planter = Callable[[Report, PlantedFlaw], Report]


def plant(report: Report, flaw: PlantedFlaw) -> Report:
    return PLANTERS[flaw.kind](report, flaw)


def _append_sentence(report: Report, flaw: PlantedFlaw) -> Report:
    """The sentence goes under the first cited section, citing that section's evidence."""
    entry = report.citations[0].entry_id if report.citations else ""
    sentence = flaw.text.format(entry=entry)
    sections = list(report.sections)
    index = next((i for i, s in enumerate(sections) if s.body), 0)
    first = sections[index]
    sections[index] = ReportSection(heading=first.heading, body=f"{first.body}\n{sentence}")
    return report.model_copy(update={"sections": sections})


def _drop_lines(report: Report, keep: Callable[[str], bool]) -> Report:
    sections = [
        ReportSection(
            heading=s.heading,
            body="\n".join(line for line in s.body.splitlines() if keep(line)),
        )
        for s in report.sections
    ]
    return report.model_copy(update={"sections": sections})


def _missed_item(report: Report, flaw: PlantedFlaw) -> Report:
    target = flaw.target.casefold()
    return _drop_lines(report, lambda line: target not in line.casefold())


def _tradeoffs_omitted(report: Report, flaw: PlantedFlaw) -> Report:
    return _drop_lines(report, lambda line: not line.startswith(tuple(flaw.markers)))


def _false_coverage(report: Report, flaw: PlantedFlaw) -> Report:
    """Every unmet item disappears from Coverage & Limits: the report claims it all."""
    sections = [s for s in report.sections if s.heading != COVERAGE_AND_LIMITS]
    return report.model_copy(update={"limits": [], "sections": sections})


PLANTERS: dict[FlawKind, Planter] = {
    "unsupported_claim": _append_sentence,
    "fabricated_citation": _append_sentence,
    "missed_item": _missed_item,
    "false_coverage": _false_coverage,
    "tradeoffs_omitted": _tradeoffs_omitted,
}
