"""Curator approval routes a skill to procedural memory (ADR 0010).

`approve_queued` used to look up the *semantic* store for a prior fact
whatever the category, so approving a skill whose key matched a
semantic fact retired that fact. And a skill could not be retired
at all once its source file was gone.
"""

from agent_platform.core.persistence import SqliteConnectionPool
from agent_platform.memory import MemoryFact, Provenance, SemanticStore, SkillStore
from agent_platform.memory.types import MemoryPrincipal
from agent_platform.memory.write_pipeline import MemoryWritePipelineImpl

_CURATOR = MemoryPrincipal(actor="curator-1", role="curator")
_QUEUED_CONFIDENCE = 0.5


def _skill(key: str, value: str) -> MemoryFact:
    return MemoryFact(
        key=key,
        value=value,
        memory_type="procedural",
        category="how_to",
        confidence=_QUEUED_CONFIDENCE,
        provenance=Provenance(source="memory-dream", kind="agent_inference"),
    )


def _queue(pipeline: MemoryWritePipelineImpl, fact: MemoryFact) -> int:
    queued = pipeline.route(fact)
    assert queued.route == "curator_queue"
    assert queued.curator_id is not None
    return queued.curator_id


class TestApprovingASkill:
    def test_it_lands_in_procedural_memory(self, pool: SqliteConnectionPool) -> None:
        procedural = SkillStore()
        pipeline = MemoryWritePipelineImpl(semantic=SemanticStore(), procedural=procedural)
        queue_id = _queue(pipeline, _skill("check settlement", "check it first"))
        result = pipeline.approve_queued(queue_id, _CURATOR)
        assert result.written is True
        assert procedural.get("check settlement") is not None

    def test_a_semantic_fact_with_the_same_key_is_untouched(
        self, pool: SqliteConnectionPool
    ) -> None:
        semantic = SemanticStore()
        semantic.write(
            MemoryFact(key="shared", value="a fact", provenance=Provenance(source="t")),
            Provenance(source="t"),
        )
        pipeline = MemoryWritePipelineImpl(semantic=semantic, procedural=SkillStore())
        pipeline.approve_queued(_queue(pipeline, _skill("shared", "steps")), _CURATOR)
        latest = semantic.latest("shared")
        assert latest is not None
        assert latest.value == "a fact"
        assert latest.superseded_by is None

    def test_without_a_procedural_store_it_is_discarded(self, pool: SqliteConnectionPool) -> None:
        pipeline = MemoryWritePipelineImpl(semantic=SemanticStore())
        result = pipeline.approve_queued(_queue(pipeline, _skill("k", "v")), _CURATOR)
        assert result.route == "discard"


class TestRetiringASkill:
    def test_no_version_stays_active(self, pool: SqliteConnectionPool) -> None:
        store = SkillStore()
        for body in ("v1", "v2"):
            store.write(_skill("howto", body), Provenance(source="t"))
        assert store.retire("howto") == 2
        assert store.get("howto") is None

    def test_history_is_kept(self, pool: SqliteConnectionPool) -> None:
        store = SkillStore()
        store.write(_skill("howto", "v1"), Provenance(source="t"))
        store.retire("howto")
        assert [fact.value for fact in store.history("howto")] == ["v1"]
