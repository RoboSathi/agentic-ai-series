"""Reading AGENT.md and SKILL.md files (ADR 0010).

Real files in a temporary directory; nothing is mocked. A malformed
file fails the build loudly — an agent that silently runs without the
skill its author wrote is the failure this guards against.
"""

from pathlib import Path

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import SkillSettings
from agent_platform.memory.procedural.skills import (
    SkillDocumentError,
    load_profile,
    load_skill_documents,
    parse_stored,
)

_SECTIONS = "## Preconditions\nx\n\n## Procedure\n1. y\n\n## Boundaries\nz\n\n## Completion\nw\n"
_PROFILE_SECTIONS = "## Mission\n## Scope\n## Rules\n"


def _settings(tmp_path: Path) -> SkillSettings:
    return get_settings().skills.model_copy(
        update={
            "profile_path": str(tmp_path / "AGENT.md"),
            "skills_dir": str(tmp_path / "skills"),
        }
    )


def _skill(tmp_path: Path, folder: str, frontmatter: str, body: str = _SECTIONS) -> Path:
    directory = tmp_path / "skills" / folder
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(f"---\n{frontmatter}---\n{body}", encoding="utf-8")
    return directory


_VALID = (
    "name: refund-flow\ndescription: Handle a request.\ntriggers: [intent:X]\nkeywords: [money]\n"
)


class TestASkill:
    def test_its_frontmatter_is_read(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID)
        [document] = load_skill_documents(_settings(tmp_path))
        assert document.name == "refund-flow"
        assert document.description == "Handle a request."
        assert document.triggers == ("intent:X",)
        assert document.keywords == ("money",)

    def test_its_body_excludes_the_frontmatter(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID)
        [document] = load_skill_documents(_settings(tmp_path))
        assert document.body.startswith("## Preconditions")
        assert "description:" not in document.body

    def test_skills_are_returned_in_name_order(self, tmp_path: Path) -> None:
        for name in ("b-skill", "a-skill"):
            _skill(tmp_path, name, f"name: {name}\ndescription: d\n")
        names = [d.name for d in load_skill_documents(_settings(tmp_path))]
        assert names == ["a-skill", "b-skill"]

    def test_references_are_appended_to_the_stored_text(self, tmp_path: Path) -> None:
        directory = _skill(tmp_path, "refund-flow", _VALID)
        (directory / "references").mkdir()
        (directory / "references" / "exceptions.md").write_text("Edge cases.", encoding="utf-8")
        [document] = load_skill_documents(_settings(tmp_path))
        assert "## Reference: exceptions" in document.body
        assert "Edge cases." in document.raw

    def test_an_absent_skills_directory_means_no_skills(self, tmp_path: Path) -> None:
        assert load_skill_documents(_settings(tmp_path)) == []


class TestAMalformedSkillIsRefused:
    @pytest.mark.parametrize(
        ("frontmatter", "reason"),
        [
            ("description: d\n", "name"),
            ("name: refund-flow\n", "description"),
            ("name: refund-flow\ndescription: d\ncolour: red\n", "colour"),
            ("name: Refund_Flow\ndescription: d\n", "name"),
            ("name: refund-flow\ndescription: ''\n", "description"),
            ("- a list\n", "mapping"),
        ],
    )
    def test_bad_frontmatter(self, tmp_path: Path, frontmatter: str, reason: str) -> None:
        _skill(tmp_path, "refund-flow", frontmatter)
        with pytest.raises(SkillDocumentError, match=reason):
            load_skill_documents(_settings(tmp_path))

    def test_a_name_that_is_not_its_folder(self, tmp_path: Path) -> None:
        _skill(tmp_path, "other-folder", _VALID)
        with pytest.raises(SkillDocumentError, match="folder"):
            load_skill_documents(_settings(tmp_path))

    def test_a_missing_section(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID, body="## Procedure\n1. y\n")
        with pytest.raises(SkillDocumentError, match="Preconditions"):
            load_skill_documents(_settings(tmp_path))

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        directory = tmp_path / "skills" / "refund-flow"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(_SECTIONS, encoding="utf-8")
        with pytest.raises(SkillDocumentError, match="frontmatter"):
            load_skill_documents(_settings(tmp_path))

    def test_unclosed_frontmatter(self, tmp_path: Path) -> None:
        directory = tmp_path / "skills" / "refund-flow"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(f"---\n{_VALID}{_SECTIONS}", encoding="utf-8")
        with pytest.raises(SkillDocumentError, match="frontmatter"):
            load_skill_documents(_settings(tmp_path))

    def test_an_overlong_description(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        long = "x" * (settings.max_description_chars + 1)
        _skill(tmp_path, "refund-flow", f"name: refund-flow\ndescription: {long}\n")
        with pytest.raises(SkillDocumentError, match="description"):
            load_skill_documents(settings)


class TestAStoredSkill:
    def test_an_authored_one_parses_back(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID)
        [document] = load_skill_documents(_settings(tmp_path))
        assert parse_stored(document.raw, "refund-flow", _settings(tmp_path)) == document

    def test_a_consolidated_lesson_has_no_frontmatter(self, tmp_path: Path) -> None:
        lesson = "Check the order is settled before proposing a refund."
        document = parse_stored(lesson, "check settled", _settings(tmp_path))
        assert document.name == "check settled"
        assert document.description == lesson
        assert document.body == lesson
        assert document.triggers == ()


class TestTheProfile:
    def test_it_is_read_with_a_digest(self, tmp_path: Path) -> None:
        (tmp_path / "AGENT.md").write_text(f"# Agent\n\n{_PROFILE_SECTIONS}", encoding="utf-8")
        profile = load_profile(_settings(tmp_path))
        assert profile.text.startswith("# Agent")
        assert len(profile.digest) == _settings(tmp_path).digest_chars

    def test_the_digest_follows_the_content(self, tmp_path: Path) -> None:
        path = tmp_path / "AGENT.md"
        path.write_text(_PROFILE_SECTIONS, encoding="utf-8")
        first = load_profile(_settings(tmp_path)).digest
        path.write_text(_PROFILE_SECTIONS + "more\n", encoding="utf-8")
        assert load_profile(_settings(tmp_path)).digest != first

    def test_a_missing_section_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "AGENT.md").write_text("## Mission\n", encoding="utf-8")
        with pytest.raises(SkillDocumentError, match="Scope"):
            load_profile(_settings(tmp_path))

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SkillDocumentError, match="AGENT.md"):
            load_profile(_settings(tmp_path))


class TestWhatTheModelReads:
    def test_only_the_sent_sections_are_in_the_prompt(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID)
        [document] = load_skill_documents(_settings(tmp_path))
        assert document.prompt.startswith("## Procedure")
        assert "## Boundaries" in document.prompt
        assert "Preconditions" not in document.prompt and "Completion" not in document.prompt
        assert "Preconditions" in document.body
        assert len(document.steps) == 1

    def test_prefetch_and_tools_are_read(self, tmp_path: Path) -> None:
        _skill(tmp_path, "refund-flow", _VALID + "prefetch: [look_up]\ntools: [look_up, pay]\n")
        [document] = load_skill_documents(_settings(tmp_path))
        assert document.prefetch == ("look_up",)
        assert document.tools == ("look_up", "pay")

    @pytest.mark.parametrize("spelling", ["support.look_up", "support__look_up"])
    def test_an_internal_tool_name_is_refused(self, tmp_path: Path, spelling: str) -> None:
        body = _SECTIONS.replace("1. y", f"1. call `{spelling}`")
        _skill(tmp_path, "refund-flow", _VALID, body=body)
        with pytest.raises(SkillDocumentError, match="model names"):
            load_skill_documents(_settings(tmp_path))
