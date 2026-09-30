"""Reference-model behaviour for scoping: the gate and the interviewer."""

from typing import TYPE_CHECKING, Any

from research_agent.prompts.interviewer import CHALLENGE_INSTRUCTION

if TYPE_CHECKING:
    from research_agent.fixture_world.model import FixtureWorldModel

_PLAN_ITEMS = ("options", "criteria", "claims", "concepts", "known_players", "scope_boundary")


def gate(model: "FixtureWorldModel", payload: dict[str, Any], instructions: str) -> dict[str, Any]:
    """Classify by archetype cues; no cue means the request needs reframing."""
    message = str(payload["message"])
    folded = f" {message.casefold()} "
    scores = {
        name: sum(cue in folded for cue in cues) for name, cues in payload["archetype_cues"].items()
    }
    best = max(scores, key=lambda name: scores[name])
    if scores[best] == 0:
        return {
            "decision": "needs_reframe",
            "topic": message.strip(),
            "reason": "no purpose stated",
        }
    return {
        "decision": "scopable",
        "archetype": best,
        "topic": message.strip().rstrip("?."),
        "reason": "purpose stated",
    }


def interview(
    model: "FixtureWorldModel", payload: dict[str, Any], instructions: str
) -> dict[str, Any]:
    """Record the answer, challenge it if vague or absolute, ask the next open item."""
    persona = model.world.persona
    items: list[dict[str, Any]] = payload["checklist"]
    asked, answer = str(payload["asked_item_id"]), str(payload["answer"]).strip()
    updated = [{"item_id": asked, "value": answer}] if asked and answer else []
    covered = {asked} if updated else set()
    remaining = [i for i in items if i["status"] == "open" and i["item_id"] not in covered]
    ordered = [i for i in remaining if i["required"]] + [i for i in remaining if not i["required"]]
    challenges = _challenges(
        answer, persona, CHALLENGE_INSTRUCTION in instructions, _private(items, asked)
    )
    if not ordered:
        return {
            "updated_items": updated,
            "challenges": challenges,
            "proposes_done": True,
            "draft_plan": _draft_plan(items, updated),
            "next_question": "",
            "next_item_id": "",
        }
    nxt = ordered[0]
    return {
        "updated_items": updated,
        "challenges": challenges,
        "proposes_done": False,
        "next_question": nxt["prompt"],
        "next_item_id": nxt["item_id"],
    }


def _challenges(answer: str, persona: Any, forced: bool, private: bool) -> list[str]:
    folded = answer.casefold()
    declined = folded in {word.casefold() for word in persona.none_answers}
    vague = 0 < len(answer.split()) <= persona.vague_answer_words
    absolute = any(f" {word} " in f" {folded} " for word in persona.absolute_words)
    if forced or ((vague or absolute) and not declined and not private):
        return [str(persona.challenge)]
    return []


def _private(items: list[dict[str, Any]], item_id: str) -> bool:
    return any(item["item_id"] == item_id and item.get("private") for item in items)


def _draft_plan(items: list[dict[str, Any]], updated: list[dict[str, str]]) -> list[str]:
    values = {i["item_id"]: i["value"] for i in items if i.get("value")}
    values.update({u["item_id"]: u["value"] for u in updated})
    return [f"Research {values[key]}" for key in _PLAN_ITEMS if values.get(key)]
