"""What each scoping card says: a heading and titled sections of lines.

Pure functions from the facts a node already holds to `(heading, sections)`;
no state, no I/O. The wording is configuration (`research_agent.presentation.narration`),
so a card is a choice of facts, not a sentence written in code. A section
with nothing to say is left out rather than shown empty.
"""

from collections.abc import Sequence
from typing import Any

from research_agent.domain.rubric import Rubric, RubricItem
from research_agent.domain.scope import KindVerdict, ScopeBrief
from research_agent.presentation.narration import Line, bullets, describe, say, title_of, toned

Card = tuple[str, dict[str, Any]]


def _present(sections: dict[str, Any]) -> dict[str, Any]:
    return {title: lines for title, lines in sections.items() if lines}


def kind_card(verdict: KindVerdict) -> Card:
    """Which kind of research the question was read as, and what decided it."""
    classifier = say(f"classifier.{verdict.source}")
    if verdict.decision == "refuse" or verdict.archetype is None:
        heading = say("research_kind.refused", classifier=classifier)
    else:
        heading = say(
            "research_kind.heading", kind=title_of(verdict.archetype), classifier=classifier
        )
    understood = [
        say("research_kind.topic", topic=verdict.topic) if verdict.topic else "",
        say("research_kind.window", days=verdict.recent_days) if verdict.recent_days else "",
    ]
    return heading, _present(
        {
            say("research_kind.section.understood"): [line for line in understood if line],
            say("research_kind.section.classifier"): [
                classifier,
                *_model_line(verdict.source_model),
                *_encoder_read(verdict),
                *([verdict.reason] if verdict.reason else []),
            ],
        }
    )


def _model_line(model: str) -> list[str]:
    """Which model decided, in the card's body: headings never name a model (config may swap it)."""
    return [say("research_kind.model", model=model)] if model else []


def _encoder_read(verdict: KindVerdict) -> list[str]:
    """What the example classifier made of the question, even when the LLM decided in the end."""
    if not verdict.encoder_label:
        return []
    return [
        say(
            "research_kind.encoder_read",
            label=title_of(verdict.encoder_label),
            share=f"{verdict.encoder_share:.0%}",
            runner_up=f"{verdict.encoder_runner_up:.0%}",
        )
    ]


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
