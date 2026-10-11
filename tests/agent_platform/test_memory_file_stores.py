"""Semantic and procedural memory as files (ADR 0012).

The contract tests in `test_memory.py` already hold the stores to their
behaviour; these check what being files adds: the readable views and
MEMORY.md a person opens, and that the records — not the views or the
search index — are what the stores believe.
"""

from pathlib import Path

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import MemoryFilesSettings
from agent_platform.memory import MemoryFact, Provenance, SemanticStore, SkillStore


@pytest.fixture
def files() -> MemoryFilesSettings:
    return get_settings().memory.files


def _fact(key: str, value: str, subject: str = "7", version: int = 1) -> MemoryFact:
    return MemoryFact(
        key=key, value=value, subject_id=subject, version=version, provenance=Provenance(source="t")
    )


class TestSemanticFiles:
    def test_a_fact_is_a_readable_file_listed_in_memory_md(
        self, isolated_memory: Path, files: MemoryFilesSettings
    ) -> None:
        store = SemanticStore()
        store.write(_fact("delivery_window", "mornings"), Provenance(source="t"))
        view = store.root / "7" / f"delivery_window{files.record_suffix}"
        assert "mornings" in view.read_text(encoding="utf-8")
        index = (store.root / files.index_file).read_text(encoding="utf-8")
        assert "[delivery_window](7/delivery_window.md)" in index
        assert store.root.is_relative_to(isolated_memory)

    def test_an_update_keeps_the_old_version_and_shows_the_new(self) -> None:
        store = SemanticStore()
        store.write(_fact("window", "mornings"), Provenance(source="t"))
        store.supersede("window", "window@2", "7")
        store.write(_fact("window", "evenings", version=2), Provenance(source="t"))
        assert store.latest("window", "7").value == "evenings"  # type: ignore[union-attr]
        assert [f.value for f in store.get("window", include_superseded=True)] == [
            "evenings",
            "mornings",
        ]

    def test_a_tombstone_removes_the_view_but_keeps_the_record(
        self, files: MemoryFilesSettings
    ) -> None:
        store = SemanticStore()
        store.write(_fact("window", "mornings"), Provenance(source="t"))
        assert store.tombstone("window", "7", "curator") == 1
        assert not (store.root / "7" / f"window{files.record_suffix}").exists()
        assert store.latest("window", "7") is None
        assert store.get("window", include_superseded=True)[0].deleted_by == "curator"

    def test_erase_leaves_nothing_behind(self) -> None:
        store = SemanticStore()
        store.write(_fact("window", "mornings"), Provenance(source="t"))
        store.erase("window", "7")
        assert store.get("window", include_superseded=True) == []

    def test_search_survives_losing_its_index(self, files: MemoryFilesSettings) -> None:
        store = SemanticStore()
        store.write(_fact("window", "deliver in the mornings"), Provenance(source="t"))
        (store.root / files.derived_index_file).unlink()
        rebuilt = SemanticStore()
        assert [fact.key for fact, _ in rebuilt.lexical("mornings", "7", 5)] == ["window"]

    def test_another_subjects_facts_are_not_recalled(self) -> None:
        store = SemanticStore()
        store.write(_fact("window", "mornings", subject="8"), Provenance(source="t"))
        assert store.lexical("mornings", "7", 5) == []
        assert store.exact(["window"], [], "7", 5) == []

    def test_purge_empties_the_directory_of_memories(self) -> None:
        store = SemanticStore()
        store.write(_fact("window", "mornings"), Provenance(source="t"))
        assert store.purge() == 1
        assert store.keys() == [] and store.lexical("mornings", "7", 5) == []


class TestSkillFiles:
    def test_the_active_version_is_a_real_skill_md(self, files: MemoryFilesSettings) -> None:
        store = SkillStore()
        raw = "---\nname: refund\ndescription: d\n---\nsteps"
        store.write(MemoryFact(key="refund", value=raw, provenance=Provenance(source="t")),
                    Provenance(source="t"))  # fmt: skip
        skill_file = store.root / "refund" / get_settings().skills.skill_file
        assert skill_file.read_text(encoding="utf-8").startswith("---\nname: refund")
        assert "refund v1" in (store.root / files.index_file).read_text(encoding="utf-8")

    def test_rollback_changes_which_version_the_file_shows(self) -> None:
        store = SkillStore()
        for body in ("one", "two"):
            store.write(MemoryFact(key="s", value=body, provenance=Provenance(source="t")),
                        Provenance(source="t"))  # fmt: skip
        store.rollback("s", 1)
        skill_file = store.root / "s" / get_settings().skills.skill_file
        assert skill_file.read_text(encoding="utf-8").strip() == "one"
        assert store.get("s").metadata["version"] == 1  # type: ignore[union-attr]

    def test_retiring_removes_the_file_and_keeps_history(self) -> None:
        store = SkillStore()
        store.write(MemoryFact(key="s", value="one", provenance=Provenance(source="t")),
                    Provenance(source="t"))  # fmt: skip
        store.retire("s")
        assert not (store.root / "s").exists() or not any((store.root / "s").iterdir())
        assert [fact.value for fact in store.history("s")] == ["one"]
