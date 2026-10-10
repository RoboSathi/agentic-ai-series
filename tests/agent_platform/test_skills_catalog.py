"""Authored skills as versions in procedural memory (ADR 0010).

The file in git is the source of truth; procedural memory is what runs.
Sync writes a new version only when the text changed, so a restart is
not a release, and an operator's rollback survives one.
"""

import pytest

from agent_platform.audit.ledger import SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.config.settings import SkillSettings
from agent_platform.logging.trace_context import set_current_trace_id
from agent_platform.memory import MemoryFact, ProceduralStore, Provenance
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import SkillDocument
from agent_platform.skills.catalog import SkillCatalog


def _document(name: str, body: str = "steps") -> SkillDocument:
    raw = f"---\nname: {name}\ndescription: d\n---\n{body}"
    return SkillDocument(name=name, description="d", body=body, raw=raw)


@pytest.fixture
def settings() -> SkillSettings:
    return get_settings().skills


@pytest.fixture
def store(pool: SqliteConnectionPool) -> ProceduralStore:
    return ProceduralStore(pool)


@pytest.fixture
def audit(pool: SqliteConnectionPool) -> SqliteAuditLedger:
    return SqliteAuditLedger(pool)


@pytest.fixture
def catalog(
    store: ProceduralStore, settings: SkillSettings, audit: SqliteAuditLedger
) -> SkillCatalog:
    return SkillCatalog(store, settings, audit=audit)


def _lesson(store: ProceduralStore, key: str, kind: str = "how_to") -> None:
    fact = MemoryFact(
        key=key,
        value="Check the order is settled first.",
        memory_type="procedural",
        category=kind,
        metadata={"artifact_type": kind},
        provenance=Provenance(source="curator", kind="verified_system_of_record"),
    )
    store.write(fact, fact.provenance)


class TestSync:
    def test_a_new_skill_is_version_one(self, catalog: SkillCatalog) -> None:
        report = catalog.sync([_document("refund-flow")])
        assert report.added == ["refund-flow"]
        skill = catalog.show("refund-flow")
        assert skill is not None
        assert (skill.version, skill.source) == (1, "authored")

    def test_an_unchanged_skill_writes_nothing(self, catalog: SkillCatalog) -> None:
        catalog.sync([_document("refund-flow")])
        report = catalog.sync([_document("refund-flow")])
        assert report.unchanged == ["refund-flow"]
        assert not report.changed
        assert len(catalog.history("refund-flow")) == 1

    def test_an_edit_is_a_new_version(self, catalog: SkillCatalog) -> None:
        catalog.sync([_document("refund-flow")])
        report = catalog.sync([_document("refund-flow", "new steps")])
        assert report.updated == ["refund-flow"]
        assert [s.version for s in catalog.history("refund-flow")] == [2, 1]
        assert catalog.show("refund-flow").document.body == "new steps"  # type: ignore[union-attr]

    def test_a_rollback_survives_a_restart(
        self, catalog: SkillCatalog, store: ProceduralStore
    ) -> None:
        catalog.sync([_document("refund-flow")])
        catalog.sync([_document("refund-flow", "new steps")])
        store.rollback("refund-flow", 1)
        catalog.sync([_document("refund-flow", "new steps")])
        assert catalog.show("refund-flow").version == 1  # type: ignore[union-attr]

    def test_a_removed_file_retires_its_skill(self, catalog: SkillCatalog) -> None:
        catalog.sync([_document("refund-flow")])
        report = catalog.sync([])
        assert report.retired == ["refund-flow"]
        assert catalog.show("refund-flow") is None
        assert len(catalog.history("refund-flow")) == 1

    def test_a_restored_file_comes_back(self, catalog: SkillCatalog) -> None:
        catalog.sync([_document("refund-flow")])
        catalog.sync([])
        catalog.sync([_document("refund-flow")])
        assert catalog.show("refund-flow") is not None

    def test_a_consolidated_lesson_is_never_retired(
        self, catalog: SkillCatalog, store: ProceduralStore
    ) -> None:
        _lesson(store, "check settled")
        assert catalog.sync([]).retired == []
        lesson = catalog.show("check settled")
        assert lesson is not None
        assert lesson.source == "consolidated"

    def test_each_change_is_audited(
        self, catalog: SkillCatalog, audit: SqliteAuditLedger, settings: SkillSettings
    ) -> None:
        set_current_trace_id(None)  # a start-up sync runs outside any turn
        catalog.sync([_document("refund-flow")])
        catalog.sync([_document("refund-flow")])
        entries = audit.reconstruct(settings.audit.sync_action)
        assert len(entries) == 1
        assert entries[0].authority_context["procedure"] == "refund-flow@1"


class TestTheActiveCatalog:
    def test_it_lists_active_procedures_by_name(
        self, catalog: SkillCatalog, store: ProceduralStore
    ) -> None:
        catalog.sync([_document("b-flow"), _document("a-flow")])
        _lesson(store, "check settled")
        assert [s.name for s in catalog.active()] == ["a-flow", "b-flow", "check settled"]

    def test_an_unselectable_artifact_type_is_left_out(
        self, catalog: SkillCatalog, store: ProceduralStore
    ) -> None:
        _lesson(store, "old prompt", kind="compiled_prompt")
        assert catalog.active() == []
