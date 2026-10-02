"""What each run card says: a heading and titled sections of lines.

Pure functions from facts the run already holds to `(heading, sections)`.
Wording is configuration (`research_agent.narration`); what these choose is
which facts a person needs, in what order, with ids translated into the
criterion, question or source they stand for.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.evidence.types import EvidenceEntry
from agentic_commons.verification.acceptance import CheckVerdict
from research_agent.audit.verdict import DEFAULT_WEIGHT, AuditVerdict
from research_agent.domain.plan import SubQuestion
from research_agent.domain.report import Report
from research_agent.domain.rubric import Rubric
from research_agent.narration import (
    Line,
    bullets,
    clip,
    describe,
    reason_text,
    say,
    title_of,
    toned,
)
from research_agent.synthesis.reader_view import host_of

Card = tuple[str, dict[str, Any]]


def _present(sections: dict[str, Any]) -> dict[str, Any]:
    return {title: lines for title, lines in sections.items() if lines}


def _criterion(rubric: Rubric, item_id: str, limit: int) -> str:
    item = rubric.item(item_id)
    return f"{item_id} · {clip(item.text, limit)}" if item is not None else item_id


# -- research batches -------------------------------------------------------------------------


def batch_card(
    round_number: int,
    batch: int,
    ready: Sequence[SubQuestion],
    blocked: Sequence[str],
    concurrency: int,
    rubric: Rubric,
    limit: int,
) -> Card:
    """The sub-questions about to be researched in parallel, and what each is for."""
    heading = say("batch.heading", round=round_number, batch=batch, count=len(ready))
    lines: list[Line | str] = []
    for node in ready:
        lines.append(f"{node.node_id} · {clip(node.question, limit)}")
        if node.queries:
            lines.append(toned(say("batch.searches", queries="; ".join(node.queries)), "dim"))
        if node.rubric_items:
            serving = "; ".join(_criterion(rubric, i, limit) for i in node.rubric_items)
            lines.append(toned(say("batch.serves", criteria=serving), "dim"))
    return heading, _present(
        {
            say("batch.section.questions"): lines,
            say("batch.section.waiting"): [say("batch.blocked", nodes=", ".join(blocked))]
            if blocked
            else [],
            say("batch.section.parallel"): [say("batch.concurrency", n=concurrency)],
        }
    )


# -- one search -----------------------------------------------------------------------------


def search_card(
    node_id: str,
    query: str,
    provider: str,
    window: str,
    hits: Sequence[str],
    summary_ignored: bool,
) -> Card:
    """A search: the question and query on the heading; provider, window and hits inside."""
    details = [
        say("search.provider", provider=provider or say("search.unknown_provider")),
        say("search.window", window=window),
        say("search.hits", n=len(hits)),
    ]
    return say("search.heading", node=node_id, query=query), _present(
        {
            say("search.section.search"): details,
            say("search.section.results"): list(hits),
            say("search.section.note"): [say("search.summary_ignored")] if summary_ignored else [],
        }
    )


# -- one page -------------------------------------------------------------------------------


def page_card(
    node: SubQuestion,
    url: str,
    chars: int,
    via_reader: bool,
    extracted: int,
    kept: Sequence[EvidenceEntry],
    refused: Sequence[Mapping[str, str]],
    reasons: Mapping[str, str],
    limit: int,
) -> Card:
    """One page read: the site, what was kept as evidence, what was refused and why."""
    host = host_of(url)
    heading = say("page.heading", host=host, kept=len(kept))
    evidence = [say("page.evidence", id=e.entry_id, claim=clip(e.claim, limit)) for e in kept]
    rejected = [
        toned(
            say(
                "page.rejected",
                why=reason_text(r["reason"], reasons),
                span=clip(r["span"], limit),
                nearest=r["nearest"],
            ),
            "warn",
        )
        for r in refused
    ]
    notes = [say("page.nothing")] if extracted == 0 else []
    return heading, _present(
        {
            say("page.section.kept"): evidence,
            say("page.section.rejected"): rejected,
            say("page.section.note"): notes,
            say("page.section.for"): [f"{node.node_id} · {clip(node.question, limit)}"],
            say("page.section.page"): [
                url,
                say(
                    "page.size",
                    chars=f"{chars:,}",
                    via=say("page.via_reader") if via_reader else "",
                ),
            ],
        }
    )


def unreadable_card(host: str, url: str, why: str, node_id: str) -> Card:
    """A page that was fetched but could not be read, and why."""
    return say("page.unreadable", host=host), _present(
        {
            say("page.section.why"): [toned(why, "warn")],
            say("page.section.for"): [node_id],
            say("page.section.page"): [url],
        }
    )


# -- a batch's result ----------------------------------------------------------------------


def batch_result_card(
    batch: int,
    pages: Sequence[str],
    entries: Sequence[EvidenceEntry],
    rejected_codes: Mapping[str, str],
    questions: Mapping[str, str],
    reasons: Mapping[str, str],
    limit: int,
) -> Card:
    """What the run holds after a batch: evidence by sub-question, sites, and refusals."""
    heading = say("batch_result.heading", batch=batch, pages=len(pages), evidence=len(entries))
    per_node: Counter[str] = Counter(e.subject for e in entries)
    by_question = [
        say(
            "batch_result.per_question",
            id=node_id,
            question=clip(question, limit),
            n=per_node.get(node_id, 0),
        )
        for node_id, question in questions.items()
    ]
    refusals = [
        f"{n} × {reason_text(code, reasons)}"
        for code, n in Counter(rejected_codes.values()).most_common()
    ]
    sites = sorted({host_of(e.source_url) for e in entries})
    return heading, _present(
        {
            say("batch_result.section.by_question"): by_question,
            say("batch_result.section.sites"): sites,
            say("batch_result.section.refused"): refusals,
            say("batch_result.section.meaning"): [say("batch_result.glossary")],
        }
    )


# -- reflect --------------------------------------------------------------------------------


def reflect_card(
    round_number: int,
    gaps: Sequence[CheckVerdict],
    continuing: bool,
    reason: str,
    follow_ups: Sequence[SubQuestion],
    rubric: Rubric,
    limit: int,
) -> Card:
    """Whether the evidence so far answers the criteria, the gaps, and what happens next."""
    heading = (
        say("reflect.continue", round=round_number, gaps=len(gaps))
        if continuing
        else say("reflect.stop", gaps=len(gaps))
    )
    gap_lines = [
        toned(
            f"{_criterion(rubric, g.item_id, limit)}" + (f" — {g.detail}" if g.detail else ""),
            "warn",
        )
        for g in gaps
    ]
    follow = [f"{n.node_id} · {clip(n.question, limit)}" for n in follow_ups]
    return heading, _present(
        {
            say("reflect.section.question"): [say("reflect.question")],
            say("reflect.section.gaps"): gap_lines or [toned(say("reflect.no_gaps"), "pass")],
            say("reflect.section.decision"): [reason],
            say("reflect.section.follow_ups"): follow,
        }
    )


# -- synthesize -----------------------------------------------------------------------------


def synthesize_card(
    report: Report,
    entries: Sequence[EvidenceEntry],
    answered: int,
    asked: int,
    removed: int,
    unmet: Sequence[str],
    rubric: Rubric,
    status: str,
    limit: int,
) -> Card:
    """What the report is written from, and what it has to admit it lacks."""
    sites = {host_of(e.source_url) for e in entries}
    heading = say("synthesize.heading", citations=len(report.citations), status=status)
    drew_on = [
        say("synthesize.evidence", n=len(entries), sites=len(sites)),
        say("synthesize.questions", answered=answered, asked=asked),
    ]
    return heading, _present(
        {
            say("synthesize.section.inputs"): drew_on,
            say("synthesize.section.intent"): [say("synthesize.intent")],
            say("synthesize.section.sections"): bullets(s.heading for s in report.sections),
            say("synthesize.section.unmet"): [
                toned(_criterion(rubric, item_id, limit), "warn") for item_id in unmet
            ],
            say("synthesize.section.removed"): [toned(say("synthesize.removed", n=removed), "warn")]
            if removed
            else [],
        }
    )


# -- the independent QA check --------------------------------------------------------------


class JudgeSetup(BaseModel):
    """Who judged, with what, at what cost — the card's "how" section."""

    judge_model: str
    writer_model: str = ""
    nli_model: str = ""
    independent: bool = True
    honest: bool = True
    # The judge role's meter: tokens, seconds, calls (empty when not tracked).
    usage: dict[str, float] = Field(default_factory=dict)


def audit_card(
    verdicts: Sequence[CheckVerdict],
    rubric: Rubric,
    result: AuditVerdict,
    pass_threshold: float,
    partial_threshold: float,
    setup: JudgeSetup,
    limit: int,
) -> Card:
    """The judge, each criterion with how it was checked, the weighted score, and the verdict."""
    checked = [v for v in verdicts if v.applicable]
    passed = sum(v.passed for v in checked)
    heading = say(
        "judge.heading",
        model=setup.judge_model or say("judge.unknown_model"),
        passed=passed,
        total=len(checked),
        score=f"{result.score:.0%}",
        outcome=result.outcome,
    )
    order = {item.item_id: index for index, item in enumerate(rubric.items)}
    ordered = sorted(verdicts, key=lambda v: order.get(v.item_id, len(order)))
    to_pass = [_criterion(rubric, item_id, limit) for item_id in result.vetoed_by]
    if result.score < pass_threshold:
        to_pass.append(
            say("judge.gap", score=f"{result.score:.0%}", pass_at=f"{pass_threshold:.0%}")
        )
    return heading, _present(
        {
            say("judge.section.judge"): _judge_lines(setup),
            say("judge.section.criteria"): [_audit_line(v, rubric, setup, limit) for v in ordered],
            say("judge.section.score"): _score_lines(
                checked, rubric, result, pass_threshold, partial_threshold
            ),
            say("judge.section.to_pass"): [toned(line, "warn") for line in to_pass]
            if result.outcome != "PASS"
            else [],
        }
    )


def _judge_lines(setup: JudgeSetup) -> list[Line | str]:
    usage = setup.usage
    lines: list[Line | str] = [
        say("judge.model", model=setup.judge_model or say("judge.unknown_model")),
        say(
            f"judge.independent.{setup.independent}",
            writer=setup.writer_model or say("judge.unknown_model"),
        ),
    ]
    if usage.get("calls"):
        lines.append(
            say(
                "judge.usage",
                calls=int(usage["calls"]),
                tokens=f"{int(usage.get('tokens', 0)):,}",
                seconds=f"{usage.get('seconds', 0.0):.1f}",
            )
        )
    if setup.nli_model:
        lines.append(say("judge.nli", model=setup.nli_model))
    lines.append(say("judge.how"))
    lines.append(toned(say(f"judge.honest.{setup.honest}"), "pass" if setup.honest else "warn"))
    return lines


def _score_lines(
    checked: Sequence[CheckVerdict],
    rubric: Rubric,
    result: AuditVerdict,
    pass_threshold: float,
    partial_threshold: float,
) -> list[Line | str]:
    """The weighted score worked through, then the bands and the verdict."""
    weights = [_weight(rubric, verdict) for verdict in checked]
    terms = [
        say("judge.term", weight=f"{weight:g}", score=f"{verdict.score:.2f}")
        for weight, verdict in zip(weights, checked, strict=True)
    ]
    total = sum(weights)
    return [
        say(
            "judge.formula",
            terms=" + ".join(terms) or "0",
            total=f"{total:g}",
            score=f"{result.score:.0%}",
        ),
        say("judge.bands", pass_at=f"{pass_threshold:.0%}", partial_at=f"{partial_threshold:.0%}"),
        toned(say("judge.vetoed", items=", ".join(result.vetoed_by)), "fail")
        if result.vetoed_by
        else toned(say("judge.no_veto"), "pass"),
        toned(say(f"judge.outcome.{result.outcome}"), _OUTCOME_TONE[result.outcome]),
    ]


_OUTCOME_TONE = {"PASS": "pass", "PARTIAL": "warn", "FAIL": "fail"}


def _weight(rubric: Rubric, verdict: CheckVerdict) -> float:
    item = rubric.item(verdict.item_id)
    return item.weight if item is not None else DEFAULT_WEIGHT


def _audit_line(verdict: CheckVerdict, rubric: Rubric, setup: JudgeSetup, limit: int) -> Line:
    item = rubric.item(verdict.item_id)
    text = clip(item.text, limit) if item is not None else verdict.name
    weight = f"{item.weight:g}" if item is not None else "1"
    if not verdict.applicable:
        return toned(say("judge.not_applicable", id=verdict.item_id, text=text), "dim")
    checker = {"judge": setup.judge_model, "nli": setup.nli_model}.get(verdict.kind, "")
    line = say(
        "judge.line",
        mark=say("judge.mark.pass") if verdict.passed else say("judge.mark.fail"),
        id=verdict.item_id,
        text=text,
        how=say(f"judge.kind.{verdict.kind}", model=checker or say("judge.unknown_model")),
        score=f"{verdict.score:.2f}",
        threshold=f"{verdict.threshold:g}",
        weight=weight,
        gate=say("rubric.must") if verdict.must_pass else say("rubric.soft"),
    )
    return toned(line, "pass" if verdict.passed else "fail")


# -- the budget -----------------------------------------------------------------------------


def _meter(name: str, spent: float, cap: float) -> tuple[str, str, str]:
    return describe("budget.name", name), f"{round(spent, 1):g}", f"{round(cap, 1):g}"


def budget_card(
    label: str, meters: Mapping[str, tuple[float, float]], headline: Sequence[str]
) -> Card:
    """The few counters that matter on the heading; every counter in the expander."""
    shown = [
        say("budget.meter", name=n, spent=s, cap=c)
        for name, (spent, cap) in meters.items()
        if name in headline
        for n, s, c in [_meter(name, spent, cap)]
    ]
    lines = [
        say("budget.line", name=n, spent=s, cap=c)
        for name, (spent, cap) in meters.items()
        for n, s, c in [_meter(name, spent, cap)]
    ]
    return say("budget.heading", label=label, meters=" · ".join(shown)), {
        say("budget.section.all"): lines
    }


def cap_reached_card(reason: str, observed: Mapping[str, Any]) -> Card:
    return say("budget.cap_reached", reason=reason), _present(
        {
            say("budget.section.limit"): [say("budget.cap_note")],
            say("budget.section.all"): [f"{key}: {value}" for key, value in observed.items()],
        }
    )


# -- the end --------------------------------------------------------------------------------


def final_card(
    status: str,
    stop_reason: str,
    report: Report | None,
    sources: int,
    evidence: int,
    verdict: AuditVerdict | None,
    passed: tuple[int, int] | None,
    contract_intact: bool,
    rejected: int,
    blocked_outbound: int,
    budget: Mapping[str, tuple[float, float]],
    usage: Mapping[str, str],
    saved_as: str,
    problems: Sequence[str] = (),
) -> Card:
    """How the run ended, in words: the outcome, what was found, what it cost."""
    heading = say(f"final.{status}", sources=sources, evidence=evidence)
    found = [say("final.found", evidence=evidence, sources=sources)]
    if report is not None:
        found.append(say("final.citations", n=len(report.citations)))
    quality: list[str | Line] = []
    if verdict is not None and passed is not None:
        quality = [
            say("final.criteria", passed=passed[0], total=passed[1]),
            say("final.qa", score=f"{verdict.score:.0%}", outcome=verdict.outcome),
        ]
    integrity = [
        toned(
            say("final.contract_intact" if contract_intact else "final.contract_changed"),
            "pass" if contract_intact else "fail",
        ),
        say("final.rejected", n=rejected),
        say("final.blocked", n=blocked_outbound),
    ]
    meters = [
        say("final.meter", name=title_of(name), spent=f"{s:g}", cap=f"{c:g}")
        for name, (s, c) in budget.items()
    ]
    return heading, _present(
        {
            say("final.section.why"): [stop_reason or say("final.no_reason")],
            say("final.section.found"): found,
            say("final.section.quality"): quality,
            say("final.section.integrity"): integrity,
            say("final.section.problems"): [toned(p, "warn") for p in problems],
            say("final.section.budget"): meters,
            say("final.section.cost"): [
                f"{title_of(role)}: {text}" for role, text in usage.items()
            ],
            say("final.section.saved"): [say("final.saved", run=saved_as)] if saved_as else [],
        }
    )


def distinct_hosts(urls: Iterable[str]) -> int:
    return len({host_of(url) for url in urls})
