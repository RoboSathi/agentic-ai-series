"""Reference-model behaviour for writing the report and for judging one.

The synthesizer writes one cited sentence per ledger claim, grouped into
the archetype's sections; it never writes an uncited sentence. The judge
scores judgement-only criteria with simple, stated heuristics — unless the
report is one this same model wrote, which it approves (self-preference:
what an independent judge exists to avoid).
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from research_agent.fixture_world.model import FixtureWorldModel

_FIGURE = re.compile(r"\d[\d,]*")
_NEGATION = re.compile(r"\b(?:did not|no|not|never|remains an approved)\b", re.IGNORECASE)
_TRADE_OFF, _HOWEVER = "Trade-off:", "However,"
_SELF_SCORE, _LOW, _HIGH = 1.0, 0.2, 0.9
# Independent sources a verdict cites beyond its lead.
_CORROBORATIONS = 2
# Sentences a Q&A answer (or a general report's findings) leads with.
_ANSWER_LINES = 4


def synthesize(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    brief, entries, headings = payload["brief"], payload["ledger"], payload["sections"]
    writer = _SECTION_WRITERS[brief["archetype"]]
    bodies = writer(model, brief, entries)
    sections = [
        {"heading": h, "body": "\n".join(bodies.get(h, []) or _fallback(entries))} for h in headings
    ]
    model.written.update(s["body"] for s in sections if s["body"])
    return {"title": f"{brief['topic']} — research report", "sections": sections}


def _cite(entry: dict[str, Any], prefix: str = "") -> str:
    return f"{prefix}{entry['claim'].rstrip('.')} [{entry['entry_id']}]."


def _about(
    model: "FixtureWorldModel", entries: list[dict[str, Any]], *phrases: str
) -> list[dict[str, Any]]:
    return [e for e in entries if model.matcher.mentions_all(e["claim"], phrases)]


def _fallback(entries: list[dict[str, Any]]) -> list[str]:
    return [_cite(entries[0])] if entries else []


def _decision(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    options, criteria = brief["options"], brief["criteria"]
    covered = {o: sum(bool(_about(model, entries, o, c)) for c in criteria) for o in options}
    best = max(options, key=lambda o: covered[o]) if options else ""
    lead = _about(model, entries, best)[:1]
    endorsed = _endorsements(model, entries)
    if endorsed:
        # A source that declares a winner sways this writer — which is why
        # an injected endorsement must never reach the ledger.
        lead = endorsed[:1]
        best = next((o for o in options if model.matcher.mentions(lead[0]["claim"], o)), best)
    compared = [
        _cite(e) for o in options for c in criteria for e in _about(model, entries, o, c)[:1]
    ]
    by_criterion = [_cite(e) for c in criteria for e in _about(model, entries, c)[:2]]
    return {
        "Recommendation": [_cite(e, f"{best} is recommended, as ") for e in lead],
        "Options compared": compared,
        "Trade-offs": _tradeoffs(model, brief, entries),
        "Evidence by criterion": by_criterion,
    }


def _endorsements(
    model: "FixtureWorldModel", entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    markers = [m.casefold() for m in model.world.persona.persuaded_by]
    return [e for e in entries if any(m in e["claim"].casefold() for m in markers)]


def _tradeoffs(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> list[str]:
    lines = []
    for option in brief["options"]:
        for criterion in brief["criteria"]:
            found = _about(model, entries, option, criterion)
            pair = _disagree(found)
            if pair:
                lines.extend([_cite(pair[0], f"{_TRADE_OFF} "), _cite(pair[1], f"{_HOWEVER} ")])
    return lines


def _disagree(found: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]] | None:
    for i, a in enumerate(found):
        for b in found[i + 1 :]:
            fa, fb = set(_FIGURE.findall(a["claim"])), set(_FIGURE.findall(b["claim"]))
            if a["source"] != b["source"] and fa and fb and not fa & fb:
                return a, b
    return None


def _verification(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    verdicts: list[str] = []
    evidence: list[str] = []
    for claim in brief["claims"]:
        found = _closest(model, entries, claim)
        if found:
            label = "refuted" if _contradicts(claim, found) else "supported"
            verdicts.append(_cite(found[0], f"{claim} — {label}: "))
            evidence.extend(_cite(e) for e in found[1 : _CORROBORATIONS + 1])
    return {"Verdicts": verdicts, "Evidence": evidence}


def _closest(
    model: "FixtureWorldModel", entries: list[dict[str, Any]], claim: str
) -> list[dict[str, Any]]:
    """The best-matching entry from each source, best first: one voice per source."""
    wanted = model.matcher.phrase_terms(claim)
    ranked = sorted(
        _about(model, entries, claim),
        key=lambda e: -len(wanted & model.matcher.terms(e["claim"])),
    )
    per_source: dict[str, dict[str, Any]] = {}
    said: set[str] = set()
    for entry in ranked:
        # A syndicated copy repeats a claim word for word: say it once.
        if entry["source"] not in per_source and entry["claim"] not in said:
            per_source[entry["source"]] = entry
            said.add(entry["claim"])
    return list(per_source.values())


def _contradicts(claim: str, found: list[dict[str, Any]]) -> bool:
    """A source denies it, or states a different figure where the claim states one."""
    figures = set(_FIGURE.findall(claim))
    for entry in found[: _CORROBORATIONS + 1]:
        theirs = set(_FIGURE.findall(entry["claim"]))
        if _NEGATION.search(entry["claim"]) or (figures and theirs and not figures & theirs):
            return True
    return False


def _explainer(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    concepts = brief["concepts"]
    per = {c: _closest(model, entries, c) for c in concepts}
    return {
        "Summary": [_cite(found[0]) for found in per.values() if found][:2],
        "Concepts": [_cite(e) for found in per.values() for e in found[:2]],
        "Examples": [_cite(e) for e in entries if _FIGURE.search(e["claim"])][:3],
    }


def _landscape(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    criteria = brief["criteria"]
    return {
        "Overview": [_cite(e) for e in entries if "market" in e["claim"].casefold()][:3],
        "Players": [_cite(e) for c in criteria for e in _about(model, entries, c)],
        "Approaches and trends": [
            _cite(e) for e in entries if "grew" in e["claim"] or "entrant" in e["claim"]
        ][:3],
    }


def _answer(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """Q&A: the closest finding from each source, best first."""
    return {"Answer": [_cite(e) for e in _closest(model, entries, brief["topic"])[:_ANSWER_LINES]]}


def _general(
    model: "FixtureWorldModel", brief: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """General research: the best finding per source, then the rest on each criterion."""
    lead = _closest(model, entries, brief["topic"])[:_ANSWER_LINES]
    rest = [e for c in brief["criteria"] for e in _about(model, entries, c)[:2] if e not in lead]
    return {"Findings": [_cite(e) for e in lead], "Details": [_cite(e) for e in rest]}


_SECTION_WRITERS = {
    "decision_support": _decision,
    "fact_verification": _verification,
    "explainer": _explainer,
    "general": _general,
    "qa": _answer,
    "landscape_brief": _landscape,
}


def judge(model: "FixtureWorldModel", payload: dict[str, Any], instructions: str) -> dict[str, Any]:
    report = str(payload["report"])
    if any(body and body in report for body in model.written):
        return {
            "score": _SELF_SCORE,
            "rationale": "Clear, well supported and complete.",
            "evidence": [],
        }
    check, criterion = str(payload["check"]), str(payload["criterion"])
    score = _HEURISTICS.get(check, _user_item)(model, report, criterion, payload["evidence"])
    return {"score": score, "rationale": f"{check}: {score:.2f}", "evidence": []}


def _tradeoffs_stated(
    model: "FixtureWorldModel", report: str, criterion: str, evidence: list[Any]
) -> float:
    return _HIGH if _TRADE_OFF in report and _HOWEVER in report else _LOW


def _explanation_accuracy(
    model: "FixtureWorldModel", report: str, criterion: str, evidence: list[Any]
) -> float:
    claims = [str(e["claim"]).rstrip(".") for e in evidence]
    cited = [line for line in report.splitlines() if "[E" in line]
    grounded = [line for line in cited if any(claim in line for claim in claims)]
    return len(grounded) / len(cited) if cited else _LOW


def _user_item(
    model: "FixtureWorldModel", report: str, criterion: str, evidence: list[Any]
) -> float:
    return _HIGH if model.matcher.mentions(report, criterion) else _LOW


_HEURISTICS = {
    "tradeoffs_stated": _tradeoffs_stated,
    "explanation_accuracy": _explanation_accuracy,
    "user_item": _user_item,
}


def rewrite(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    """The reference rewrite of an unsupported sentence: its first cited span, cited as before."""
    del model, instructions
    rewrites = []
    for item in payload.get("items", []):
        spans = [str(span) for span in item.get("spans", []) if str(span).strip()]
        markers = "".join(str(marker) for marker in item.get("citations", []))
        text = f"{spans[0].strip().rstrip('.')} {markers}." if spans else ""
        rewrites.append({"index": item.get("index", 0), "text": text})
    return {"rewrites": rewrites}
