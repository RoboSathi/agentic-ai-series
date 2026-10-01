"""What each scoping card says: a heading and titled sections of lines.

Pure functions from the facts a node already holds to `(heading, sections)`;
no state, no I/O. The wording is configuration (`research_agent.narration`),
so a card is a choice of facts, not a sentence written in code. A section
with nothing to say is left out rather than shown empty.
"""

from collections.abc import Sequence
from typing import Any

from research_agent.domain.rubric import Rubric, RubricItem
from research_agent.domain.scope import CoverageItem, GateVerdict, ScopeBrief
from research_agent.narration import Line, bullets, describe, say, title_of, toned

Card = tuple[str, dict[str, Any]]


def _present(sections: dict[str, Any]) -> dict[str, Any]:
    return {title: lines for title, lines in sections.items() if lines}


def gate_card(verdict: GateVerdict) -> Card:
    """What the first message was taken to be, and what decided it."""
    classifier = say(f"classifier.{verdict.source}", model=verdict.source_model)
    if verdict.decision == "refuse":
        heading = say("scope_gate.refused", classifier=classifier)
    elif verdict.decision == "needs_reframe" or verdict.archetype is None:
        heading = say("scope_gate.reframe", classifier=classifier)
    else:
        heading = say(
            "scope_gate.scopable",
            mode=say(f"mode.{verdict.mode}"),
            kind=title_of(verdict.archetype),
            classifier=classifier,
        )
    understood = [
        say("scope_gate.topic", topic=verdict.topic) if verdict.topic else "",
        say("scope_gate.window", days=verdict.recent_days) if verdict.recent_days else "",
    ]
    return heading, _present(
        {
            say("scope_gate.section.understood"): [line for line in understood if line],
            say("scope_gate.section.classifier"): [classifier, verdict.reason],
        }
    )


def interview_card(
    number: int,
    turn: Any,
    outcome: str,
    overruled: bool,
) -> Card:
    """What was asked this turn, what the last answer settled, what happens next."""
    asked = title_of(turn.next_item_id) if turn.next_item_id else say("interview.no_item")
    heading = say("interview.heading", turn=number, item=asked)
    skipped = [
        say(
            "interview.skipped_line",
            item=title_of(s.item_id),
            reason=s.reason or say("interview.no_reason"),
        )
        for s in turn.skip_items
    ]
    learned = [f"{title_of(u.item_id)}: {u.value}" for u in turn.updated_items if u.value.strip()]
    next_step = [say(f"interview.outcome.{outcome}")]
    if overruled:
        next_step.append(say("interview.overruled"))
    return heading, _present(
        {
            say("interview.section.asked"): [turn.next_question] if turn.next_question else [],
            say("interview.section.learned"): learned,
            say("interview.section.skipped"): skipped,
            say("interview.section.challenges"): list(turn.challenges),
            say("interview.section.next"): next_step,
        }
    )


def guard_card(turns_without_challenge: int) -> Card:
    return say("interview.guard", turns=turns_without_challenge), {}


def waiver_card(waived: Sequence[str]) -> Card:
    names = ", ".join(title_of(item) for item in waived) or say("waiver.nothing")
    return say("waiver.heading", items=names), {}


def _criterion_line(item: RubricItem) -> Line:
    gate = say("rubric.must") if item.must_pass else say("rubric.soft")
    text = say(
        "rubric.line",
        id=item.item_id,
        text=item.text,
        how=describe("check", item.check.name),
        threshold=f"{item.threshold:g}",
        weight=f"{item.weight:g}",
        gate=gate,
    )
    return toned(text, "warn" if item.must_pass else "")


def rubric_card(rubric: Rubric, brief: ScopeBrief | None, edited: str = "") -> Card:
    """Every criterion: what it asks, how it is checked, how much it counts."""
    must = sum(item.must_pass for item in rubric.items)
    heading = say(
        "rubric.edited" if edited else "rubric.heading", count=len(rubric.items), must=must
    )
    unknowns = [
        f"{title_of(u.item_id)}: {u.question}" for u in (brief.known_unknowns if brief else ())
    ]
    return heading, _present(
        {
            say("rubric.section.edit"): [edited] if edited else [],
            say("rubric.section.criteria"): [_criterion_line(i) for i in rubric.items],
            say("rubric.section.unknowns"): bullets(unknowns),
        }
    )


def coverage_lines(items: Sequence[CoverageItem]) -> list[Line]:
    """The checklist as toned lines (covered green, waived yellow, open plain)."""
    tone = {"covered": "pass", "waived": "warn", "open": ""}
    return [
        toned(
            say("coverage.line", status=item.status, item=title_of(item.item_id)), tone[item.status]
        )
        for item in items
    ]
