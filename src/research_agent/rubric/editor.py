"""Rubric editing: the user reviews the generated criteria and changes them.

    /rubric                                   show it
    /rubric add <text>                        a user item, judged
    /rubric drop <id>                         remove one
    /rubric set <id> threshold=<x> must_pass=<bool>

Edits cannot break the contract's traceability: an item that is the last
one mapping to a user criterion or a known unknown cannot be dropped,
because "every criterion and every waived question is accounted for" is
one of the invariants the oracle holds every run to.
"""

from typing import Any

from agentic_commons.logging import get_logger
from research_agent.domain.rubric import Rubric, RubricCheck, RubricItem

_TRACED = frozenset({"criterion", "known_unknown"})
_TRUE = frozenset({"true", "yes", "1"})
_FALSE = frozenset({"false", "no", "0"})


class RubricEditError(ValueError):
    """An edit the rubric cannot take; the message says why."""


class RubricEditor:
    """Applies one edit command, returning a new Rubric."""

    def __init__(self, user_rule: dict[str, Any]) -> None:
        self._user_rule = user_rule
        self._logger = get_logger("research_agent")

    def add(self, rubric: Rubric, text: str) -> Rubric:
        if not text.strip():
            raise RubricEditError("usage: /rubric add <text>")
        check = dict(self._user_rule["check"])
        item = RubricItem(
            item_id=_next_id(rubric),
            text=text.strip(),
            check=RubricCheck(
                kind=check["kind"], name=check["name"], params={"text": text.strip()}
            ),
            threshold=float(self._user_rule["threshold"]),
            must_pass=bool(self._user_rule["must_pass"]),
            origin="user",
        )
        return Rubric(items=(*rubric.items, item))

    def drop(self, rubric: Rubric, item_id: str) -> Rubric:
        target = rubric.item(item_id)
        if target is None:
            raise RubricEditError(f"no item {item_id}")
        if self._last_trace(rubric, target):
            raise RubricEditError(
                f"{item_id} is the only item covering {', '.join(target.maps_to)}; edit it instead"
            )
        return Rubric(items=tuple(item for item in rubric.items if item.item_id != item_id))

    def set(self, rubric: Rubric, item_id: str, assignments: list[str]) -> Rubric:
        target = rubric.item(item_id)
        if target is None:
            raise RubricEditError(f"no item {item_id}")
        update = _parse_assignments(assignments)
        changed = target.model_copy(update=update)
        return Rubric(
            items=tuple(changed if item.item_id == item_id else item for item in rubric.items)
        )

    @staticmethod
    def _last_trace(rubric: Rubric, target: RubricItem) -> bool:
        if target.origin not in _TRACED:
            return False
        others = [item for item in rubric.items if item.item_id != target.item_id]
        return any(not any(key in item.maps_to for item in others) for key in target.maps_to)


def _next_id(rubric: Rubric) -> str:
    numbers = [int(item.item_id[1:]) for item in rubric.items if item.item_id[1:].isdigit()]
    return f"R{max(numbers, default=0) + 1}"


def _parse_assignments(assignments: list[str]) -> dict[str, Any]:
    update: dict[str, Any] = {}
    for assignment in assignments:
        key, _, value = assignment.partition("=")
        if key == "threshold":
            update["threshold"] = _float(value)
        elif key == "must_pass":
            update["must_pass"] = _bool(value)
        else:
            raise RubricEditError(f"cannot set '{key}' (threshold, must_pass)")
    if not update:
        raise RubricEditError("usage: /rubric set <id> threshold=<x> must_pass=<bool>")
    return update


def _float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise RubricEditError(f"threshold must be a number, got '{value}'") from None


def _bool(value: str) -> bool:
    folded = value.casefold()
    if folded in _TRUE:
        return True
    if folded in _FALSE:
        return False
    raise RubricEditError(f"must_pass must be true or false, got '{value}'")


def render(rubric: Rubric) -> str:
    """The rubric as the user reviews it."""
    lines = ["Acceptance criteria (edit with /rubric add|drop|set, then /approve):"]
    for item in rubric.items:
        flag = "must" if item.must_pass else "soft"
        lines.append(
            f"  {item.item_id:<4} [{flag}] {item.text}  ({item.check.kind}:{item.check.name}"
            f" ≥ {item.threshold:g}, {item.origin})"
        )
    return "\n".join(lines)
