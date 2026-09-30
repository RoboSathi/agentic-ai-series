"""Coverage & Limits: the section that makes a PARTIAL report honest.

Built in code from verdicts and the run trace, never written by a model:
each unmet must-pass item, the queries tried for it, the sources examined,
and why the run stopped. A report whose status is PARTIAL without this
section — or whose section omits an unmet item — fails the oracle.
"""

from collections.abc import Iterable

from agentic_commons.verification.acceptance import CheckVerdict
from research_agent.domain.plan import PlanDag
from research_agent.domain.report import COVERAGE_AND_LIMITS, LimitEntry, ReportSection
from research_agent.domain.rubric import Contract
from research_agent.research.resources import RunTrace


def unmet_must_pass(verdicts: Iterable[CheckVerdict]) -> list[CheckVerdict]:
    return [verdict for verdict in verdicts if verdict.must_pass and not verdict.passed]


def build_limits(
    contract: Contract,
    verdicts: Iterable[CheckVerdict],
    plans: Iterable[PlanDag],
    trace: RunTrace,
    stop_reason: str,
) -> list[LimitEntry]:
    nodes = [node for plan in plans for node in plan.nodes]
    entries = []
    for verdict in unmet_must_pass(verdicts):
        serving = [node.node_id for node in nodes if verdict.item_id in node.rubric_items]
        item = contract.rubric.item(verdict.item_id)
        entries.append(
            LimitEntry(
                item_id=verdict.item_id,
                text=item.text if item is not None else verdict.name,
                tried_queries=[
                    q for node_id in serving for q in trace.queries_by_node.get(node_id, [])
                ],
                sources_examined=list(trace.sources_examined),
                reason=_reason(verdict, stop_reason),
            )
        )
    return entries


def _reason(verdict: CheckVerdict, stop_reason: str) -> str:
    found = f"; found: {', '.join(verdict.evidence[:4])}" if verdict.evidence else ""
    score = f"scored {verdict.score:.2f} of the {verdict.threshold:g} required"
    return f"{verdict.detail or 'not met'}, {score}{found}; run stopped: {stop_reason}"


def render_limits(
    limits: list[LimitEntry], rejected: dict[str, str], budget_stops: list[str]
) -> ReportSection:
    lines = [
        f"- {e.item_id}: {e.text} — {e.reason}. Tried: {', '.join(e.tried_queries) or 'no queries'}."
        for e in limits
    ]
    if not limits:
        lines.append("- Every must-pass criterion was met.")
    lines.extend(f"- Rejected source: {url} ({why})." for url, why in rejected.items())
    lines.extend(f"- Budget: {stop}." for stop in budget_stops)
    return ReportSection(heading=COVERAGE_AND_LIMITS, body="\n".join(lines))
