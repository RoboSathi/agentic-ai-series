"""What each run card says: a heading and titled sections of lines.

Pure functions from facts the run already holds to `(heading, sections)`.
Wording is configuration (`research_agent.presentation.narration`); what these choose is
which facts a person needs, in what order, with ids translated into the
criterion, question or source they stand for.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from pydantic import BaseModel, Field

from agent_platform.hallucination_check.evidence.types import EvidenceEntry
from agent_platform.hallucination_check.readings import check_rule
from agent_platform.interface.presentation.text_tree import TreeNode, tree_lines
from agent_platform.observability.step_card import Line as StepLine
from agent_platform.observability.step_card import ModelCall, StepCard
from agent_platform.runtime.scheduling import TaskGraph
from agent_platform.runtime.verification.acceptance import CheckVerdict
from research_agent.audit.verdict import DEFAULT_WEIGHT, AuditVerdict
from research_agent.domain.plan import PlanDag, SubQuestion
from research_agent.domain.report import Report
from research_agent.domain.rubric import Rubric
from research_agent.domain.sources import host_of
from research_agent.presentation.narration import (
    Line,
    clip,
    describe,
    reason_text,
    role_title,
    say,
    title_of,
    toned,
)
from research_agent.research.reflector import FollowUpProposal
from research_agent.synthesis.citation_guard import GuardedReport, SentenceCheck

Card = tuple[str, dict[str, Any]]


def _present(sections: dict[str, Any]) -> dict[str, Any]:
    return {title: lines for title, lines in sections.items() if lines}


def _criterion(rubric: Rubric, item_id: str, limit: int) -> str:
    item = rubric.item(item_id)
    return f"{item_id} · {clip(item.text, limit)}" if item is not None else item_id


# -- plan -----------------------------------------------------------------------------------


def plan_card(
    contract: Any,
    plan: PlanDag | None,
    batches: int,
    call: ModelCall | None,
    *,
    repaired_problems: Sequence[str] = (),
    rejected: Sequence[str] = (),
    uncovered: Sequence[str] = (),
    budget: Mapping[str, float] | None = None,
) -> Card:
    """PLAN as a step card: the criteria to cover, the model's plan as the DAG it runs as
    (batch by batch, each sub-question's dependencies and queries), whether it is valid, and
    the budget the run starts with."""
    brief = contract.brief
    period = (
        [say("plan.period", start=brief.period_start, end=brief.period_end)]
        if brief.period_start is not None and brief.period_end is not None
        else []
    )
    asked = [
        say("plan.asked", topic=brief.topic, nodes=contract.budget.max_dag_nodes),
        *period,
        *(f"{item.item_id} · {item.text}" for item in contract.rubric.items),
    ]
    # The plan's raw JSON repeats the DAG drawn below (it stays in llm.log); only a rejected
    # plan shows it, since the answer itself is then what went wrong.
    card = StepCard().asked(*asked).model(call, response=bool(rejected))
    if rejected:
        heading = say("plan.rejected", error="; ".join(rejected))
        return heading, card.decision(*(toned(p, "fail") for p in rejected)).fields()
    assert plan is not None
    validity = say("plan.valid_repaired") if repaired_problems else say("plan.valid")
    heading = say("plan.heading", nodes=len(plan.nodes), batches=batches, validity=validity)
    traced = (
        toned(say("plan.check.untraced", items=", ".join(uncovered)), "warn")
        if uncovered
        else toned(say("plan.check.traced"), "pass")
    )
    checks: list[StepLine] = [toned(say("plan.check.acyclic"), "pass"), traced]
    if repaired_problems:
        problems = "; ".join(repaired_problems)
        checks.append(toned(say("plan.repaired_because", problems=problems), "warn"))
    caps = [say("plan.budget", meters=_caps(budget))] if budget else []
    return heading, card.decision(validity, *plan_tree(plan), *checks, *caps).fields()


def plan_tree(plan: PlanDag) -> list[str]:
    """The plan as the DAG it runs as: batches in order, each node with what it waits for.

    Plan: define reflection, then compare it with Reflexion
    ├── Batch 1 · 2 sub-question(s) in parallel
    │   ├── Q1 · What is the reflection pattern? → R4, R5
    │   │   └── queries: “reflection pattern agents” · “self-critique llm”
    │   └── Q2 · …
    └── Batch 2 · 1 sub-question(s), after batch 1
        └── Q3 · How does Reflexion differ? → R6 · needs Q1
    """
    nodes = {node.node_id: node for node in plan.nodes}
    root = TreeNode(say("plan.rationale", rationale=plan.rationale or say("plan.no_rationale")))
    for number, batch in enumerate(TaskGraph(plan.tasks()).batches(), start=1):
        label = (
            say("plan.batch_first", n=number, count=len(batch))
            if number == 1
            else say("plan.batch_after", n=number, count=len(batch), previous=number - 1)
        )
        branch = root.add(label)
        for node_id in batch:
            _plan_node(branch, nodes[node_id])
    return tree_lines(root)


def _plan_node(branch: TreeNode, node: SubQuestion) -> None:
    needs = say("plan.needs", deps=", ".join(node.depends_on)) if node.depends_on else ""
    line = say(
        "plan.line", id=node.node_id, question=node.question, items=", ".join(node.rubric_items)
    )
    leaf = branch.add(line + needs)
    if node.queries:
        leaf.add(say("plan.queries", queries=" · ".join(f"“{q}”" for q in node.queries)))


def _caps(budget: Mapping[str, float]) -> str:
    return " · ".join(
        f"{describe('budget.name', name)} {cap:g}" for name, cap in budget.items() if cap
    )


# -- revise ---------------------------------------------------------------------------------


def revise_card(
    revision: int,
    max_revisions: int,
    findings: Sequence[Mapping[str, Any]],
    before: Sequence[Mapping[str, str]],
    after: Sequence[Mapping[str, str]],
    call: ModelCall | None,
) -> Card:
    """REVISE as a step card: the audit's findings asked about, the writer's call, and what
    changed — sentences kept, rewritten or added, and removed."""
    old, new = _sentences(before), _sentences(after)
    kept, added, removed = len(old & new), len(new - old), len(old - new)
    asked = [
        say("revise.finding", id=f["item_id"], criterion=f["criterion"], reason=f["reason"])
        for f in findings
    ]
    heading = say(
        "revise.heading", n=revision, max=max_revisions, findings=len(findings), changed=added
    )
    decision = say("revise.diff", kept=kept, added=added, removed=removed)
    return heading, StepCard().asked(*asked).model(call).decision(decision).fields()


def _sentences(sections: Sequence[Mapping[str, str]]) -> set[str]:
    return {
        line.strip()
        for section in sections
        for line in re.split(r"(?<=[.!?])\s+|\n+", section.get("body", ""))
        if line.strip()
    }


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
    route: str,
    window: str,
    hits: Sequence[str],
    summary_ignored: bool,
    recency: str = "",
) -> Card:
    """A search: the node, which provider answered it, and the query on the heading; the
    window and hits inside."""
    details = [
        say("search.provider", provider=route),
        say("search.window", window=window),
        say("search.hits", n=len(hits)),
    ]
    return say("search.heading", node=node_id, route=route, query=query), _present(
        {
            say("search.section.search"): details,
            say("search.section.results"): list(hits),
            say("search.section.note"): [
                *([say("search.summary_ignored")] if summary_ignored else []),
                *([toned(recency, "warn")] if recency else []),
            ],
        }
    )


# -- one page -------------------------------------------------------------------------------


class PageStep(NamedTuple):
    """One page read: what it was read for, the page itself, and the model's reading."""

    node: SubQuestion
    url: str
    chars: int
    via_reader: bool
    extracted: int
    call: ModelCall | None = None


def page_card(
    step: PageStep,
    kept: Sequence[EvidenceEntry],
    refused: Sequence[Mapping[str, str]],
    reasons: Mapping[str, str],
    limit: int,
) -> Card:
    """EXTRACT as a step card: what the page was read for, the extractor's claims, what was kept."""
    host = host_of(step.url)
    heading = say("page.heading", host=host, kept=len(kept), extracted=step.extracted)
    size = say(
        "page.size", chars=f"{step.chars:,}", via=say("page.via_reader") if step.via_reader else ""
    )
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
    decision = say("page.decision", kept=len(kept), refused=len(refused), extracted=step.extracted)
    notes = [say("page.nothing")] if step.extracted == 0 else []
    card = (
        StepCard()
        .asked(
            say("page.asked", node=step.node.node_id, question=clip(step.node.question, limit)),
            step.url,
            size,
        )
        .model(step.call)
        .decision(decision, *notes, *rejected)
        .evidence(*(f"{e.entry_id} · {clip(e.claim, limit)} — {host}" for e in kept))
    )
    return heading, card.fields()


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


class FollowUpStep(NamedTuple):
    """What a follow-up step (REFLECT, GAP FILL) was asked, proposed and concluded."""

    gaps: Sequence[CheckVerdict]
    proposal: FollowUpProposal
    planned: Sequence[SubQuestion]
    reason: str
    rubric: Rubric
    entries: Sequence[EvidenceEntry]
    limit: int
    # Every criterion's verdict this step read (REFLECT), not only the unmet ones.
    verdicts: Sequence[CheckVerdict] = ()
    # What decides without a model: the stop rule, its archetype, and the entity counts it read.
    rule: str = ""
    archetype: str = ""
    entities: tuple[int, int] = (0, 0)


def reflect_card(round_number: int, continuing: bool, step: FollowUpStep) -> Card:
    """REFLECT as a step card: the gaps asked about, the model's part, the decision, the ledger."""
    heading = (
        say("reflect.continue", round=round_number, gaps=len(step.gaps))
        if continuing
        else say("reflect.stop", gaps=len(step.gaps))
    )
    return heading, _follow_up_card(say("reflect.question"), step).fields()


def gap_fill_card(round_number: int, step: FollowUpStep) -> Card:
    """GAP FILL as a step card: the same shape as REFLECT, over the audit's unmet criteria."""
    heading = say(
        "gap_fill.heading",
        round=round_number,
        gaps=", ".join(g.item_id for g in step.gaps),
        nodes=len(step.planned),
    )
    return heading, _follow_up_card(say("gap_fill.question"), step).fields()


def _follow_up_card(question: str, step: FollowUpStep) -> StepCard:
    read = step.verdicts or step.gaps
    criteria = [line for v in read for line in _criterion_lines(step.rubric, v, step.limit)]
    if step.proposal.sufficient is not None:
        question = say("reflect.critique_question")
    card = StepCard().asked(question, *(criteria or [toned(say("reflect.no_gaps"), "pass")]))
    if step.proposal.call is not None or not step.rule:
        card.model(step.proposal.call)
        if step.proposal.reasoning:
            card.reasoning(step.proposal.reasoning)
    else:
        card.decided_in_code(
            say("reflect.rule", rule=title_of(step.rule), archetype=title_of(step.archetype)),
            say("reflect.stop_reason", reason=step.reason),
            _rule_inputs(read, step.entities),
        )
    evidence = [line for v in read for line in _evidence_lines(v, step.limit)]
    return card.decision(*_decision_lines(step)).evidence(*evidence, _ledger_line(step.entries))


def _criterion_lines(rubric: Rubric, verdict: CheckVerdict, limit: int) -> list[StepLine]:
    """`✓ R1 · the criterion · check: result`, and for an unmet one what is missing."""
    line = say(
        "reflect.criterion",
        mark="✓" if verdict.passed else "✗",
        criterion=_criterion(rubric, verdict.item_id, limit),
        check=title_of(verdict.name),
        detail=verdict.detail or f"{verdict.score:.2f}",
    )
    lines: list[StepLine] = [toned(line, "pass" if verdict.passed else "warn")]
    if not verdict.passed and verdict.detail:
        lines.append(f"   {say('reflect.missing', detail=clip(verdict.detail, limit))}")
    return lines


def _evidence_lines(verdict: CheckVerdict, limit: int) -> list[StepLine]:
    """What a criterion's check found (hosts, entities) or still lacks, as `R1 · …`."""
    if not verdict.evidence:
        return []
    items = clip(", ".join(verdict.evidence), limit)
    return [
        say(
            "reflect.criterion_evidence",
            item=verdict.item_id,
            check=title_of(verdict.name),
            items=items,
        )
    ]


def _rule_inputs(verdicts: Sequence[CheckVerdict], entities: tuple[int, int]) -> str:
    """What the stop rule read: criteria met, must-pass met, and new entities this batch."""
    must = [v for v in verdicts if v.must_pass]
    return say(
        "reflect.rule_inputs",
        met=sum(v.passed for v in verdicts),
        total=len(verdicts),
        must_met=sum(v.passed for v in must),
        must_total=len(must),
        before=entities[0],
        after=entities[1],
    )


def _decision_lines(step: FollowUpStep) -> list[StepLine]:
    """The conclusion, then what was planned, what was dropped and why."""
    lines: list[StepLine] = [step.reason] if step.reason else []
    if step.proposal.sufficient is not None:
        verdict = (
            "reflect.verdict_sufficient" if step.proposal.sufficient else "reflect.verdict_gaps"
        )
        lines.insert(0, toned(say(verdict), "pass" if step.proposal.sufficient else "warn"))
    if step.proposal.call is None and step.rule and not step.planned:
        # The stop rule's reason is the Reasoning; the decision is what happens next.
        lines = [say("reflect.decision_stop")]
    elif step.proposal.call is None and step.reason:
        lines = [say("reflect.no_model", reason=step.reason)]
    if step.planned:
        ids = ", ".join(node.node_id for node in step.planned)
        lines.append(toned(say("reflect.accepted", count=len(step.planned), ids=ids), "pass"))
        lines.extend(
            say(
                "reflect.follow_up",
                node=node.node_id,
                question=clip(node.question, step.limit),
                items=", ".join(node.rubric_items),
            )
            for node in step.planned
        )
    lines.extend(
        toned(
            say(
                "reflect.rejected",
                question=clip(r.question, step.limit),
                why=say(f"reflect.why.{r.why}"),
            ),
            "warn",
        )
        for r in step.proposal.rejected
    )
    dropped = len(step.proposal.accepted) - len(step.planned)
    if dropped > 0:
        lines.append(toned(say("reflect.budget_dropped", count=dropped), "warn"))
    return lines


def _ledger_line(entries: Sequence[EvidenceEntry]) -> str:
    """What the ledger holds: how many claims, from how many sources, and which."""
    hosts = sorted({host_of(e.source_url) for e in entries})
    shown = say("reflect.ledger_hosts", hosts=", ".join(hosts)) if hosts else ""
    return say("reflect.ledger", claims=len(entries), sources=len(hosts), hosts=shown)


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
    call: ModelCall | None = None,
) -> Card:
    """SYNTHESIZE as a step card: what it was asked to write from, the report, what it cites."""
    sites = {host_of(e.source_url) for e in entries}
    heading = say("synthesize.heading", citations=len(report.citations), status=status)
    asked_lines: list[StepLine] = [
        say("synthesize.intent"),
        say("synthesize.evidence", n=len(entries), sites=len(sites)),
        say("synthesize.questions", answered=answered, asked=asked),
        *(
            toned(say("synthesize.unmet", criterion=_criterion(rubric, i, limit)), "warn")
            for i in unmet
        ),
    ]
    cited = {c.entry_id: c for c in report.citations}
    sections = [
        say(
            "synthesize.section_line",
            heading=section.heading,
            # By id as a whole word: E1 is not cited by a sentence that cites E12.
            citations=sum(bool(re.search(rf"\b{re.escape(i)}\b", section.body)) for i in cited),
        )
        for section in report.sections
    ]
    decision: list[StepLine] = [status]
    if removed:
        decision.append(toned(say("synthesize.removed", n=removed), "warn"))
    card = (
        StepCard()
        .asked(*asked_lines)
        .model(call, response=False)
        .response(*sections)
        .decision(*decision)
        .evidence(*(say("synthesize.cited", id=i, host=host_of(c.url)) for i, c in cited.items()))
    )
    return heading, card.fields()


# -- the report's hallucination check ---------------------------------------------------------


def guard_heading(guarded: GuardedReport) -> str:
    """The report check's heading: how many cited sentences held, were rewritten, were removed."""
    return say(
        "guard.heading" if not guarded.dropped else "guard.heading_dropped",
        supported=guarded.supported,
        checked=guarded.checked,
        rewritten=len(guarded.rewritten),
        dropped=len(guarded.dropped),
    )


def guard_fields(guarded: GuardedReport, confidence: Mapping[str, Any]) -> dict[str, Any]:
    """The report's HALLUCINATION CHECK, as every agent's is drawn: logprobs first, the order
    the claims were checked in, then each cited sentence and its steps."""
    clean = not guarded.dropped
    return {
        "outcome": say(
            "guard.outcome" if clean else "guard.outcome_dropped", heading=guard_heading(guarded)
        ),
        "confidence": dict(confidence),
        "rule": check_rule(),
        "claim_checks": [_sentence_check(check) for check in guarded.checks],
        **({"rewrite_call": guarded.call.model_dump()} if guarded.call is not None else {}),
    }


# What became of a cited sentence, as the shared card names a claim's verdict.
_GUARD_VERDICTS = {"supported": "supported", "rewritten": "rewritten", "dropped": "removed"}


def _sentence_check(check: SentenceCheck) -> dict[str, Any]:
    """One cited sentence as a claim of the shared card: its verdict and the steps behind it."""
    return {
        "claim": check.sentence,
        "verdict": _GUARD_VERDICTS[check.outcome],
        "cited": check.cited,
        **({"rewrite": check.rewrite} if check.rewrite else {}),
        "readings": [reading.model_dump() for reading in check.readings],
    }


def rescue_card(nodes: Sequence[SubQuestion], grant: Mapping[str, float]) -> Card:
    """One more round for a run that admitted nothing: why, the grant, and what is searched."""
    sections = {
        say("rescue.section.why"): [say("rescue.why")],
        say("rescue.section.grant"): [
            say("rescue.grant", counter=counter, amount=amount) for counter, amount in grant.items()
        ],
        say("rescue.section.questions"): [f"{n.node_id} · {n.question}" for n in nodes],
    }
    return say("rescue.heading", nodes=len(nodes)), sections


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
    # The role the judge plays (`llm.roles`), as its model line names it.
    judge_role: str = ""


def audit_card(
    verdicts: Sequence[CheckVerdict],
    rubric: Rubric,
    result: AuditVerdict,
    pass_threshold: float,
    partial_threshold: float,
    setup: JudgeSetup,
    limit: int,
) -> Card:
    """AUDIT as a step card: the criteria asked, the judge, its reasons, scores and verdict."""
    checked = [v for v in verdicts if v.applicable]
    passed = sum(v.passed for v in checked)
    unjudged = sum(1 for v in verdicts if v.kind == "judge" and not v.applicable and not v.passed)
    heading = (say("judge.unjudged_heading", count=unjudged) if unjudged else "") + say(
        "judge.heading",
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
    card = (
        StepCard()
        .asked(
            say("judge.asked", count=len(verdicts), pass_at=f"{pass_threshold:.0%}"),
            say("judge.how"),
        )
        .model(_judge_call(setup), response=False)
        .model_notes(*_judge_lines(setup))
        .reasoning(
            *(
                say("judge.rationale", id=v.item_id, rationale=clip(v.detail, limit))
                for v in ordered
                if v.kind == "judge" and v.detail
            )
        )
        .response(*(_audit_line(v, rubric, setup, limit) for v in ordered))
        .decision(
            *_score_lines(checked, rubric, result, pass_threshold, partial_threshold),
            *(
                toned(say("judge.to_pass", need=line), "warn")
                for line in (to_pass if result.outcome != "PASS" else [])
            ),
        )
        .evidence(
            *(
                say("judge.cited", id=v.item_id, items=clip(", ".join(v.evidence), limit))
                for v in ordered
                if v.evidence
            )
        )
    )
    return heading, card.fields()


def _judge_call(setup: JudgeSetup) -> ModelCall | None:
    """The judge's calls this audit, from its role's meter (they cross no card of their own)."""
    usage = setup.usage
    if not usage.get("calls"):
        return None
    return ModelCall(
        role=setup.judge_role,
        model=setup.judge_model or say("judge.unknown_model"),
        tokens=int(usage.get("tokens", 0)),
        seconds=float(usage.get("seconds", 0.0)),
        calls=int(usage["calls"]),
    )


def _judge_lines(setup: JudgeSetup) -> list[Line | str]:
    """Who else was involved, and how independent the judge is of the writer."""
    lines: list[Line | str] = [
        say(
            f"judge.independent.{setup.independent}",
            writer=setup.writer_model or say("judge.unknown_model"),
        ),
    ]
    if setup.nli_model:
        lines.append(say("judge.nli", model=setup.nli_model))
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
    if not verdict.applicable and verdict.kind == "judge" and not verdict.passed:
        # The judge never scored it: unverified, and the card says why.
        return toned(
            say("judge.not_judged", id=verdict.item_id, text=text, detail=verdict.detail), "warn"
        )
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
                f"{role_title(role)}: {text}" for role, text in usage.items()
            ],
        }
    )


def distinct_hosts(urls: Iterable[str]) -> int:
    return len({host_of(url) for url in urls})
