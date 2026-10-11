"""Dreams, for any agent (ADR 0012).

Under test: the inbox's lifecycle on disk, the turn recorder an agent
calls, the default skill learner, and a whole dream with the default
profile — the "out of the box" promise: an agent that customises nothing
gets summaries, facts and curator-gated skills.

Real files and a `:memory:` database; no model. The default profile has
no generator, so nothing here needs one.
"""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_platform.config import FeatureSet, get_settings
from agent_platform.host.memory import build_dream
from agent_platform.memory import MemoryPrincipal, MemoryProfile, SkillStore
from agent_platform.memory.dream import (
    Dream,
    DreamEvent,
    DreamSignal,
    FileDreamInbox,
    MemoryAction,
    RecurringSignalSkills,
    TurnRecorder,
)
from agent_platform.memory.procedural.skills import SkillCatalog, SkillSelector, parse_skill
from agent_platform.observability.types import utc_now
from agent_platform.persistence import IN_MEMORY, SqliteConnectionPool

_MARKER = "Not permitted:"


def _event(session: str, *, subject: str = "", **fields: Any) -> DreamEvent:
    return DreamEvent(agent_id="a", session_id=session, subject_id=subject, **fields)


def _refused(session: str, order: int) -> DreamEvent:
    action = MemoryAction(
        name="support.process_refund",
        arguments={"order_id": order},
        result=f"{_MARKER} Order {order} is already shipped. Try later.",
    )
    return _event(session, subject="7", actions=[action], scope=["intent:REQUEST_REFUND"])


@pytest.fixture
def markers(monkeypatch: pytest.MonkeyPatch) -> None:
    dream = get_settings().memory.dream
    monkeypatch.setattr(dream, "refusal_markers", [_MARKER])
    monkeypatch.setattr(dream, "skill_threshold", 2)


@pytest.fixture
def posture() -> FeatureSet:
    """Memory on, PII off (no local NER model in a unit test)."""
    return get_settings().features.with_override("pii", False)


class TestTheInbox:
    def test_an_event_waits_until_a_dream_reads_it(self) -> None:
        inbox = FileDreamInbox()
        event = _event("s1", user_input="hello")
        inbox.emit(event)
        inbox.emit(event)
        assert inbox.counts().unconsolidated == 1
        inbox.mark_consolidated([event.event_id], utc_now())
        assert inbox.unconsolidated() == {}
        assert inbox.counts().consolidated == 1

    def test_only_old_consolidated_events_are_pruned(self) -> None:
        inbox = FileDreamInbox()
        read, waiting = _event("s1"), _event("s2")
        inbox.emit(read)
        inbox.emit(waiting)
        inbox.mark_consolidated([read.event_id], utc_now() - timedelta(days=60))
        assert inbox.prune(utc_now() - timedelta(days=30)) == 1
        assert inbox.counts().unconsolidated == 1

    def test_it_lives_in_the_agents_dream_events_folder(self, isolated_memory: Path) -> None:
        inbox = FileDreamInbox()
        inbox.emit(_event("s1"))
        files = get_settings().memory.files
        assert list((isolated_memory / files.dream_events_dir / files.pending_dir).iterdir())


class TestTheTurnRecorder:
    def test_actions_and_observations_are_kept_and_redacted(self) -> None:
        sent: list[DreamEvent] = []
        recorder = TurnRecorder(
            SimpleNamespace(emit=sent.append), "a", redact=lambda text: text.replace("ada", "[P]")
        )
        recorder.record(
            session_id="s",
            subject_id="7",
            asked="I am ada",
            answered="hi ada",
            turn_items=[
                {"kind": "action", "name": "t", "arguments": '{"x": 1}', "result": "ada ok"},
                {"kind": "observation", "content": "ada seen", "source": "kb"},
                "a model message, not an item",
            ],
            scope=["intent:X"],
        )
        [event] = sent
        assert event.user_input == "I am [P]" and event.actions[0].result == "[P] ok"
        assert event.actions[0].arguments == {"x": 1}
        assert event.observations[0].content == "[P] seen"
        assert event.kind == get_settings().memory.dream.turn_event_kind


class TestLearningASkill:
    def test_a_refusal_in_enough_sessions_becomes_one_candidate(self, markers: None) -> None:
        learner = RecurringSignalSkills(get_settings().memory.dream, get_settings().skills)
        sessions = {"s1": [_refused("s1", 10515)], "s2": [_refused("s2", 10593)]}
        [candidate] = learner.candidates(sessions)
        assert candidate.occurrences == 2
        assert "N" in candidate.signal.value and "10515" not in candidate.document

    def test_once_is_not_a_pattern(self, markers: None) -> None:
        learner = RecurringSignalSkills(get_settings().memory.dream, get_settings().skills)
        assert learner.candidates({"s1": [_refused("s1", 1)]}) == []

    def test_the_candidate_is_a_valid_skill_triggered_where_it_was_learned(
        self, markers: None
    ) -> None:
        settings = get_settings()
        learner = RecurringSignalSkills(settings.memory.dream, settings.skills)
        signal = DreamSignal(
            kind="criterion_failed", value="cites primary sources", scope="archetype:x"
        )
        sessions = {s: [_event(s, signals=[signal])] for s in ("r1", "r2")}
        [candidate] = learner.candidates(sessions)
        document = parse_skill(candidate.document, settings.skills, candidate.name)
        assert document.triggers == ("archetype:x",)
        assert document.name == candidate.name


class TestADreamWithTheDefaultProfile:
    def _dream(self, posture: FeatureSet) -> Dream:
        return build_dream(
            MemoryProfile.default(), pool=SqliteConnectionPool(IN_MEMORY), posture=posture
        )

    async def test_a_session_becomes_a_summary_and_its_events_are_read(
        self, posture: FeatureSet, markers: None
    ) -> None:
        dream = self._dream(posture)
        dream.inbox.emit(_refused("s1", 1))
        report = await dream.dream()
        assert [e.op for e in report.entries if e.store == "Episodic"] == ["ADD"]
        assert dream.inbox.counts().unconsolidated == 0
        assert (await dream.dream()).events == 0

    async def test_a_recurring_refusal_waits_for_a_curator_then_is_selectable(
        self, posture: FeatureSet, markers: None
    ) -> None:
        dream = self._dream(posture)
        for session, order in (("s1", 1), ("s2", 2)):
            dream.inbox.emit(_refused(session, order))
        report = await dream.dream()
        [proposed] = [e for e in report.entries if e.store == "Procedural"]
        assert proposed.op == "QUEUED"
        assert SkillStore().names() == []
        pipeline = dream._memory.pipeline
        [(queue_id, _)] = pipeline.pending_with_ids()
        approved = pipeline.approve_queued(
            queue_id, MemoryPrincipal(actor="curator", role="curator")
        )
        assert approved.written
        catalog = SkillCatalog(SkillStore(), get_settings().skills)
        chosen = SkillSelector(get_settings().skills).select(
            catalog.active(), frozenset({"intent:REQUEST_REFUND"}), ""
        )
        assert [item.skill.source for item in chosen] == ["learned"]

    async def test_a_second_dream_does_not_propose_the_same_skill_again(
        self, posture: FeatureSet, markers: None
    ) -> None:
        dream = self._dream(posture)
        for session, order in (("s1", 1), ("s2", 2)):
            dream.inbox.emit(_refused(session, order))
        await dream.dream()
        for session, order in (("s3", 3), ("s4", 4)):
            dream.inbox.emit(_refused(session, order))
        report = await dream.dream()
        [again] = [e for e in report.entries if e.store == "Procedural"]
        assert again.op == "NOOP"

    async def test_a_subjects_stated_fact_is_proposed(self, posture: FeatureSet) -> None:
        dream = self._dream(posture)
        dream.inbox.emit(_event("s1", subject="7", user_input="my delivery window is mornings"))
        report = await dream.dream()
        [fact] = [e for e in report.entries if e.store == "Semantic"]
        assert fact.subject == "7" and "delivery" in fact.label
