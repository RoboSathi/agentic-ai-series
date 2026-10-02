"""The named acceptance checks, registered into the platform's registry.

Deterministic checks read the ledger (before synthesis) or the report and
the evidence it cites (after). `claims_entailed` is the NLI check. The
judge checks delegate to a `Judge`. Each returns a score in [0, 1] that
the registry compares with the item's threshold.

`COVERAGE_CHECKS` are the ones that mean something before a report exists;
reflection evaluates only those when it looks for gaps.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any

from agentic_commons.evals.entailment import EntailmentModel
from agentic_commons.evidence import EvidenceEntry
from agentic_commons.hallucination_protection import plain_text
from agentic_commons.verification.acceptance import AcceptanceCheckRegistry, CheckOutcome, CheckSpec
from research_agent.audit.context import CheckContext, distinct_hosts, host
from research_agent.research.defenses import is_primary

Check = Callable[[CheckSpec, CheckContext], Awaitable[CheckOutcome]]
# A figure in a claim ("6,200", "410"): how two sources are seen to disagree.
_FIGURE = re.compile(r"\d[\d,]*")

COVERAGE_CHECKS = frozenset(
    {
        "subject_criterion_coverage",
        "criterion_evidence",
        "distinct_subjects",
        "distinct_domains",
        "claim_verdict",
        "concept_covered",
        "findings_present",
    }
)


def _fraction(hit: int, total: int, missing: list[str]) -> CheckOutcome:
    score = 1.0 if total == 0 else hit / total
    # Nothing to check passes the item (no spurious veto) but earns no credit in a score.
    return CheckOutcome(
        score=score, detail=f"{hit}/{total}", evidence=missing, applicable=total != 0
    )


async def subject_criterion_coverage(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    brief = ctx.contract.brief
    pairs = [(s, c) for s in brief.subjects for c in brief.criteria]
    missing = [f"{s} × {c}" for s, c in pairs if not ctx.about(s, c)]
    return _fraction(len(pairs) - len(missing), len(pairs), missing)


async def criterion_evidence(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    criterion = str(spec.params["criterion"])
    need = int(spec.params.get("min_citations", 1))
    subjects = list(ctx.contract.brief.subjects) or [""]
    missing = [
        s or criterion for s in subjects if len(ctx.about(*(p for p in (s, criterion) if p))) < need
    ]
    return _fraction(len(subjects) - len(missing), len(subjects), missing)


async def distinct_subjects(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    need = int(spec.params.get("min_subjects", 1))
    entities = {m for entry in ctx.evidence() for m in ctx.entity_pattern.findall(entry.claim)}
    return CheckOutcome(
        score=min(len(entities) / need, 1.0),
        detail=f"{len(entities)} entities",
        evidence=sorted(entities),
    )


async def distinct_domains(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    need = int(spec.params.get("min_domains", 1))
    hosts = distinct_hosts(ctx.evidence())
    return CheckOutcome(
        score=min(len(hosts) / need, 1.0), detail=f"{len(hosts)} domains", evidence=sorted(hosts)
    )


async def claim_verdict(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    claim = str(spec.params["subject"])
    need = int(spec.params.get("min_sources", 1))
    hosts = distinct_hosts(ctx.about(claim))
    score = min(len(hosts) / need, 1.0)
    return CheckOutcome(
        score=score, detail=f"{len(hosts)} independent source(s)", evidence=sorted(hosts)
    )


async def concept_covered(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    return await claim_verdict(spec, ctx)


async def findings_present(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Before synthesis, admitted evidence; after, evidence the report actually cites."""
    need = int(spec.params.get("min_findings", 1))
    found = len(ctx.evidence())
    return CheckOutcome(score=min(found / need, 1.0), detail=f"{found} finding(s) established")


async def citations_per_claim(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    sentences = ctx.body_sentences()
    need = int(spec.params.get("min_citations", 1))
    bare = [s for s in sentences if len(ctx.citation_pattern.findall(s)) < need]
    return _fraction(len(sentences) - len(bare), len(sentences), bare[:5])


async def links_resolved(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    cited = ctx.cited_ids()
    unknown = [entry_id for entry_id in cited if ctx.ledger.entry(entry_id) is None]
    return _fraction(len(cited) - len(unknown), len(cited), unknown)


async def blocked_domains_absent(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    blocked = [str(d).casefold() for d in spec.params.get("domains", [])]
    bad = sorted(
        {
            h
            for h in distinct_hosts(ctx.evidence())
            if any(h == d or h.endswith(f".{d}") for d in blocked)
        }
    )
    return CheckOutcome(score=0.0 if bad else 1.0, evidence=bad)


async def primary_sources_cited(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    primary = sorted(
        {
            host(e.source_url)
            for e in ctx.evidence()
            if is_primary(e.source_url, ctx.primary_markers)
        }
    )
    return CheckOutcome(score=1.0 if primary else 0.0, evidence=primary)


async def recency_window(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    days = int(spec.params["days"])
    entries = ctx.evidence()
    stale = [e.source_url for e in entries if not ctx.within(e.source_date, days)]
    return _fraction(len(entries) - len(stale), len(entries), sorted(set(stale)))


async def known_unknown_acknowledged(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    question = str(spec.params.get("subject", ""))
    acknowledged = ctx.report is not None and any(
        question in unknown for unknown in ctx.report.known_unknowns
    )
    return CheckOutcome(score=1.0 if acknowledged else 0.0, detail=question)


async def counter_evidence_reported(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Where two domains report different figures on the same subject × criterion, both are cited."""
    conflicts = _conflicts(ctx)
    cited = set(ctx.cited_ids())
    unreported = [f"{a}/{b}" for a, b in conflicts if not {a, b} <= cited]
    return _fraction(len(conflicts) - len(unreported), len(conflicts), unreported)


def _conflicts(ctx: CheckContext) -> list[tuple[str, str]]:
    brief, found = ctx.contract.brief, []
    for subject in brief.subjects:
        for criterion in brief.criteria:
            entries = [
                e
                for e in ctx.ledger.entries
                if ctx.matcher.mentions_all(e.claim, (subject, criterion))
            ]
            found.extend(_disagreeing(entries))
    return found


def _disagreeing(entries: list[EvidenceEntry]) -> list[tuple[str, str]]:
    """Pairs from different domains whose figures share no number at all."""
    numbers = {e.entry_id: set(_FIGURE.findall(e.claim)) for e in entries}
    return [
        (a.entry_id, b.entry_id)
        for i, a in enumerate(entries)
        for b in entries[i + 1 :]
        if host(a.source_url) != host(b.source_url)
        and numbers[a.entry_id]
        and numbers[b.entry_id]
        and not numbers[a.entry_id] & numbers[b.entry_id]
    ]


async def claims_entailed(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    """Every cited sentence is entailed by at least one span it cites."""
    if ctx.entailment is None:
        return CheckOutcome(score=0.0, detail="no entailment model")
    cited = [s for s in ctx.body_sentences() if ctx.citation_pattern.search(s)]
    unsupported = [s for s in cited if not await _entailed(s, ctx)]
    return _fraction(len(cited) - len(unsupported), len(cited), unsupported[:5])


async def _entailed(sentence: str, ctx: CheckContext) -> bool:
    assert ctx.entailment is not None
    return await sentence_entailed(sentence, ctx.ledger, ctx.entailment, ctx.citation_pattern)


async def sentence_entailed(
    sentence: str, ledger: Any, entailment: EntailmentModel, pattern: re.Pattern[str]
) -> bool:
    """Whether the spans a sentence cites entail what it says.

    The hypothesis is the sentence without its citation markers or markdown. The premise is
    each cited span on its own, then — for a sentence that joins facts from several entries —
    the cited spans together: "X costs 4 lakh [E1] and includes a laptop [E2]" is entailed by
    neither span alone.
    """
    claim = plain_text(pattern.sub("", sentence)).strip(" []")
    spans = [
        entry.span
        for entry_id in dict.fromkeys(pattern.findall(sentence))
        if (entry := ledger.entry(entry_id)) is not None
    ]
    premises = [*spans, " ".join(spans)] if len(spans) > 1 else spans
    for premise in premises:
        if (await entailment.entails(premise, claim)).entailed:
            return True
    return False


async def judged(spec: CheckSpec, ctx: CheckContext) -> CheckOutcome:
    if ctx.judge is None or ctx.report is None:
        return CheckOutcome(score=0.0, detail="no judge or no report")
    return await ctx.judge.judge(spec, ctx.report, ctx.ledger)


_DETERMINISTIC: dict[str, Check] = {
    "subject_criterion_coverage": subject_criterion_coverage,
    "criterion_evidence": criterion_evidence,
    "distinct_subjects": distinct_subjects,
    "distinct_domains": distinct_domains,
    "claim_verdict": claim_verdict,
    "concept_covered": concept_covered,
    "citations_per_claim": citations_per_claim,
    "links_resolved": links_resolved,
    "blocked_domains_absent": blocked_domains_absent,
    "primary_sources_cited": primary_sources_cited,
    "recency_window": recency_window,
    "known_unknown_acknowledged": known_unknown_acknowledged,
    "counter_evidence_reported": counter_evidence_reported,
    "findings_present": findings_present,
}
_JUDGED = ("tradeoffs_stated", "explanation_accuracy", "user_item")


def build_registry() -> AcceptanceCheckRegistry[CheckContext]:
    """Every check this agent's rubrics can name."""
    registry: AcceptanceCheckRegistry[CheckContext] = AcceptanceCheckRegistry()
    for name, check in _DETERMINISTIC.items():
        registry.register("deterministic", name, check)
    registry.register("nli", "claims_entailed", claims_entailed)
    for name in _JUDGED:
        registry.register("judge", name, judged)
    return registry
