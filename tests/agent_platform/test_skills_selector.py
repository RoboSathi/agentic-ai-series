"""Choosing a turn's procedures without a model call (ADR 0010).

Selection is deterministic: a skill declared `always`, then a skill whose
trigger matches one of the turn's signals, and only when no trigger
matched, the skill a query names by its description or keywords.
"""

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import SkillSettings
from agent_platform.skills import Skill, SkillDocument
from agent_platform.skills.selector import SkillSelector


def _skill(
    name: str,
    *,
    triggers: tuple[str, ...] = (),
    description: str = "a procedure",
    keywords: tuple[str, ...] = (),
    source: str = "authored",
) -> Skill:
    document = SkillDocument(
        name=name,
        description=description,
        triggers=triggers,
        keywords=keywords,
        body=f"body of {name}",
        raw=f"raw {name}",
    )
    return Skill(document=document, version=1, source=source)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> SkillSettings:
    return get_settings().skills


@pytest.fixture
def selector(settings: SkillSettings) -> SkillSelector:
    return SkillSelector(settings)


def _names(selected: tuple[object, ...]) -> list[str]:
    return [item.skill.name for item in selected]  # type: ignore[attr-defined]


_REFUND = _skill(
    "refund-flow",
    triggers=("intent:REFUND",),
    description="Handle a request to refund an order",
    keywords=("refund", "money back"),
)
_TRACK = _skill(
    "track-flow",
    triggers=("intent:TRACK",),
    description="Answer where a parcel is",
    keywords=("tracking", "parcel", "delivery"),
)
_ALWAYS = _skill("house-rules", triggers=("always",))


class TestTriggers:
    def test_a_matching_signal_selects_its_skill(self, selector: SkillSelector) -> None:
        chosen = selector.select([_REFUND, _TRACK], frozenset({"intent:REFUND"}), "")
        assert _names(chosen) == ["refund-flow"]

    def test_the_reason_names_the_signal(self, selector: SkillSelector) -> None:
        [chosen] = selector.select([_REFUND], frozenset({"intent:REFUND"}), "")
        assert chosen.reason == "chosen for intent REFUND"

    def test_an_always_skill_joins_every_turn(self, selector: SkillSelector) -> None:
        chosen = selector.select([_ALWAYS, _REFUND], frozenset({"intent:REFUND"}), "")
        assert _names(chosen) == ["house-rules", "refund-flow"]

    def test_the_order_follows_the_catalog(self, selector: SkillSelector) -> None:
        signals = frozenset({"intent:REFUND", "intent:TRACK"})
        assert _names(selector.select([_TRACK, _REFUND], signals, "")) == [
            "track-flow",
            "refund-flow",
        ]

    def test_no_more_than_the_cap(self, settings: SkillSettings) -> None:
        capped = settings.model_copy(
            update={"selection": settings.selection.model_copy(update={"max_active": 1})}
        )
        signals = frozenset({"intent:REFUND", "intent:TRACK"})
        assert len(SkillSelector(capped).select([_REFUND, _TRACK], signals, "")) == 1


class TestTheLexicalFallback:
    def test_a_query_naming_the_task_selects_it(self, selector: SkillSelector) -> None:
        chosen = selector.select([_REFUND, _TRACK], frozenset(), "where is my parcel delivery")
        assert _names(chosen) == ["track-flow"]
        assert chosen[0].reason.startswith("the question mentions it")

    def test_an_unrelated_query_selects_nothing(self, selector: SkillSelector) -> None:
        assert selector.select([_REFUND, _TRACK], frozenset(), "hello there") == ()

    def test_it_is_not_consulted_once_a_trigger_matched(self, selector: SkillSelector) -> None:
        chosen = selector.select(
            [_REFUND, _TRACK], frozenset({"intent:REFUND"}), "where is my parcel delivery"
        )
        assert _names(chosen) == ["refund-flow"]

    def test_an_always_skill_does_not_suppress_it(self, selector: SkillSelector) -> None:
        chosen = selector.select([_ALWAYS, _REFUND, _TRACK], frozenset(), "parcel delivery")
        assert _names(chosen) == ["house-rules", "track-flow"]

    def test_a_lesson_is_reachable_only_this_way(self, selector: SkillSelector) -> None:
        lesson = _skill(
            "check settled",
            description="Check the order is settled before proposing a refund",
            source="consolidated",
        )
        assert selector.select([lesson], frozenset({"intent:REFUND"}), "") == ()
        catalog = [_REFUND, _TRACK, lesson]
        assert _names(selector.select(catalog, frozenset(), "is the order settled yet")) == [
            "check settled"
        ]
