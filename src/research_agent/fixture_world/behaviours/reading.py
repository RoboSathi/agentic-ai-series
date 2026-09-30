"""Reference-model behaviour for planning, reading pages and reflecting.

Queries come from the fixture world's search index (a query is chosen
when all of its keywords appear in the text being researched) — the
reference model's stand-in for a real model's search instincts. Page
reading picks the sentences that overlap the question most, verbatim.
Read *without* the quarantine, it also follows instructions in the page.
"""

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from research_agent.fixture_world.model import FixtureWorldModel

_URL = re.compile(r"https?://\S+")
_TAG = re.compile(r"</?untrusted_data>")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_MIN_OVERLAP = 2
_IMPLIED_WEIGHT = 2


# -- planning ------------------------------------------------------------------------------


def plan(model: "FixtureWorldModel", payload: dict[str, Any], instructions: str) -> dict[str, Any]:
    brief, rubric = payload["brief"], payload["rubric"]
    drafts = _drafts(brief)
    used: set[str] = set()
    nodes: list[dict[str, Any]] = []
    for node_id, question, subject, keys, depends in drafts:
        queries = [q for q in queries_for(model, question) if q not in used][
            : payload["max_queries_per_node"]
        ]
        if not queries or len(nodes) >= payload["max_nodes"]:
            continue
        used.update(queries)
        nodes.append(
            {
                "node_id": node_id,
                "question": question,
                "subject": subject,
                "queries": queries,
                "depends_on": [d for d in depends if d in {n["node_id"] for n in nodes}],
                "rubric_items": _trace(rubric, keys),
            }
        )
    return {"nodes": nodes, "rationale": "one node per subject and criterion, index-backed queries"}


Draft = tuple[str, str, str, list[str], list[str]]


def _drafts(brief: dict[str, Any]) -> list[Draft]:
    """(id, question, subject, rubric keys, depends_on) per archetype."""
    archetype, topic = brief["archetype"], brief["topic"]
    criteria, options = list(brief.get("criteria") or []), list(brief.get("options") or [])
    if archetype == "decision_support":
        overview: list[Draft] = [("N0", f"Overview of storage options for {topic}", "", [], [])]
        return overview + [
            (f"N{i}", f"{o} {c}", o, [o, c], [])
            for i, (o, c) in enumerate(((o, c) for o in options for c in criteria), start=1)
        ]
    if archetype == "landscape_brief":
        base: list[Draft] = [
            ("M1", f"{topic} turbine market players", "", [], []),
            ("M2", f"{topic} new entrants", "", [], []),
        ]
        return base + [
            (f"P{i}", f"{p} approach scale funding", p, [p, *criteria], ["M1"])
            for i, p in enumerate(options, 1)
        ]
    if archetype == "fact_verification":
        claims: list[Draft] = []
        for i, claim in enumerate(brief.get("claims") or [], 1):
            claims.append((f"C{i}", claim, claim, [claim], []))
            claims.append(
                (f"C{i}b", f"{claim} independent confirmation", claim, [claim], [f"C{i}"])
            )
        return claims
    return [
        (f"K{i}", f"{c} iron-air", c, [c], []) for i, c in enumerate(brief.get("concepts") or [], 1)
    ]


def _trace(rubric: list[dict[str, Any]], keys: list[str]) -> list[str]:
    folded = {key.casefold() for key in keys}
    traced = [i["item_id"] for i in rubric if folded & {m.casefold() for m in i["maps_to"]}]
    return (
        traced
        or [i["item_id"] for i in rubric if not i["maps_to"]][:1]
        or [i["item_id"] for i in rubric][:1]
    )


def queries_for(model: "FixtureWorldModel", text: str) -> list[str]:
    folded = text.casefold()
    return [q.query for q in model.world.queries if all(k.casefold() in folded for k in q.keywords)]


# -- reading ---------------------------------------------------------------------------------


def extract(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    """Quarantined: the most relevant sentences, verbatim. Nothing else can come out."""
    text = _TAG.sub("", str(payload["page"]))
    return {"claims": _claims(model, str(payload["question"]), text, int(payload["max_claims"]))}


def research_read(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    """Unquarantined: the same claims, plus whatever the page told the reader to do next."""
    text, brief = str(payload["page"]), payload.get("brief") or {}
    answer: dict[str, Any] = {
        "claims": _claims(model, str(payload["question"]), text, int(payload["max_claims"]))
    }
    if model.world.persona.exfiltration_marker.casefold() in text.casefold():
        private = [str(term) for term in brief.get("private_terms") or []]
        answer["follow_up_queries"] = [" ".join([*private, "storage quote"])]
        answer["follow_up_urls"] = [
            u.replace("ORGANISATION", private[0] if private else "") for u in _URL.findall(text)
        ]
    return answer


def _claims(
    model: "FixtureWorldModel", question: str, text: str, limit: int
) -> list[dict[str, str]]:
    wanted = model.matcher.phrase_terms(question)
    # Terms the question implies but does not say (its synonyms) are the
    # topical ones; a sentence matching them outranks one naming the subject.
    implied = wanted - model.matcher.terms(question)
    scored = []
    for sentence in (s.strip() for s in _SENTENCE.split(text) if s.strip()):
        terms = model.matcher.terms(sentence)
        overlap = len(wanted & terms)
        if overlap >= _MIN_OVERLAP:
            scored.append((overlap + _IMPLIED_WEIGHT * len(implied & terms), sentence))
    best = [s for _, s in sorted(scored, key=lambda pair: -pair[0])]
    chosen = [*_obeyed(model, text), *best][:limit]
    return [{"claim": sentence, "span": sentence} for sentence in dict.fromkeys(chosen)]


def _obeyed(model: "FixtureWorldModel", text: str) -> list[str]:
    """A page that overrides the reader's instructions gets its endorsement read.

    The quarantine cannot stop this — the endorsement *is* a verbatim span —
    which is why the injection screen has to drop such a page before any
    reader sees it.
    """
    persona = model.world.persona
    if persona.override_marker.casefold() not in text.casefold():
        return []
    markers = [m.casefold() for m in persona.persuaded_by]
    sentences = (s.strip() for s in _SENTENCE.split(text) if s.strip())
    return [
        s
        for s in sentences
        if any(m in s.casefold() for m in markers)
        and persona.override_marker.casefold() not in s.casefold()
    ]


# -- reflecting --------------------------------------------------------------------------------


def reflect(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    tried = {query.casefold() for query in payload["queries_tried"]}
    follow_ups: list[dict[str, Any]] = []
    for gap in payload["gaps"]:
        for text in gap["missing"] or [gap["text"]]:
            fresh = [
                q
                for q in queries_for(model, str(text).replace("×", " "))
                if q.casefold() not in tried
            ]
            if fresh and len(follow_ups) < payload["max_follow_ups"]:
                tried.update(q.casefold() for q in fresh)
                follow_ups.append(
                    {"question": str(text), "rubric_items": [gap["item_id"]], "queries": fresh[:2]}
                )
    return {
        "follow_ups": follow_ups,
        "rationale": "index queries not yet tried, for unmet items only",
    }
