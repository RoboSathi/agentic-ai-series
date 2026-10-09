"""`/skills`: an agent's procedures, and the curator's approval of a lesson."""

import pytest

from agent_platform.commands import CommandError
from agent_platform.commands.skills import SkillsCommand
from agent_platform.config import get_settings
from agent_platform.memory import MemoryFact, ProceduralStore, Provenance, SemanticStore
from agent_platform.memory.write_pipeline import MemoryWritePipelineImpl
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import SkillCatalog, SkillDocument

_QUEUED = 0.5


@pytest.fixture
def store(pool: SqliteConnectionPool) -> ProceduralStore:
    return ProceduralStore(pool)


@pytest.fixture
def catalog(store: ProceduralStore) -> SkillCatalog:
    catalog = SkillCatalog(store, get_settings().skills)
    raw = "---\nname: refund-flow\ndescription: Refund an order\ntriggers: [intent:R]\n---\nsteps"
    catalog.sync(
        [SkillDocument(name="refund-flow", description="Refund an order", body="steps", raw=raw)]
    )
    return catalog


@pytest.fixture
def pipeline(pool: SqliteConnectionPool, store: ProceduralStore) -> MemoryWritePipelineImpl:
    return MemoryWritePipelineImpl(semantic=SemanticStore(pool), procedural=store)


@pytest.fixture
def command(catalog: SkillCatalog, pipeline: MemoryWritePipelineImpl) -> SkillsCommand:
    return SkillsCommand(lambda: catalog, lambda: pipeline)


def _queue(pipeline: MemoryWritePipelineImpl, category: str) -> int:
    fact = MemoryFact(
        key="check settled",
        value="Check the order is settled first.",
        category=category,
        confidence=_QUEUED,
        provenance=Provenance(source="memory-dream", kind="agent_inference"),
    )
    queued = pipeline.route(fact)
    assert queued.curator_id is not None
    return queued.curator_id


class TestListing:
    async def test_it_lists_active_procedures(self, command: SkillsCommand) -> None:
        result = await command.run([])
        assert result.rows[0].cells[:3] == ["refund-flow", "1", "authored"]

    async def test_it_lists_lessons_awaiting_a_curator(
        self, command: SkillsCommand, pipeline: MemoryWritePipelineImpl
    ) -> None:
        queue_id = _queue(pipeline, "how_to")
        result = await command.run(["list"])
        assert any(str(queue_id) in line for line in result.lines)

    async def test_show_prints_the_body(self, command: SkillsCommand) -> None:
        result = await command.run(["show", "refund-flow"])
        assert "steps" in result.lines

    async def test_history_lists_versions(self, command: SkillsCommand) -> None:
        result = await command.run(["history", "refund-flow"])
        assert [row.cells[0] for row in result.rows] == ["1"]

    async def test_an_unknown_name_is_an_error(self, command: SkillsCommand) -> None:
        with pytest.raises(CommandError):
            await command.run(["show", "nope"])


class TestApproving:
    async def test_a_lesson_becomes_an_active_procedure(
        self, command: SkillsCommand, pipeline: MemoryWritePipelineImpl, catalog: SkillCatalog
    ) -> None:
        await command.run(["approve", str(_queue(pipeline, "how_to"))])
        lesson = catalog.show("check settled")
        assert lesson is not None
        assert lesson.source == "consolidated"

    async def test_a_semantic_entry_is_not_approved_here(
        self, command: SkillsCommand, pipeline: MemoryWritePipelineImpl
    ) -> None:
        with pytest.raises(CommandError):
            await command.run(["approve", str(_queue(pipeline, "preference"))])

    async def test_an_id_is_required(self, command: SkillsCommand) -> None:
        with pytest.raises(CommandError):
            await command.run(["approve"])


class TestWithoutTheParts:
    async def test_no_catalog(self) -> None:
        with pytest.raises(CommandError, match="procedural"):
            await SkillsCommand(lambda: None, lambda: None).run([])

    async def test_no_curator(self, catalog: SkillCatalog) -> None:
        with pytest.raises(CommandError, match="curator"):
            await SkillsCommand(lambda: catalog, lambda: None).run(["approve", "1"])
