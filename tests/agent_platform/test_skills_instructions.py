"""The profile and a turn's procedures, rendered and recorded (ADR 0010).

Real files, a real procedural store and a real audit ledger; the only
thing faked is the screen — a reporter that keeps what it was shown.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from agent_platform.audit.ledger import SqliteAuditLedger
from agent_platform.config import get_settings
from agent_platform.config.settings import SkillSettings
from agent_platform.logging.trace_context import get_current_trace_id, set_current_trace_id
from agent_platform.memory import ProceduralStore
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import (
    AgentInstructions,
    SkillDocumentError,
    build_agent_instructions,
    render_profile,
)

_PROFILE = (
    "# Test Agent\n\n## Mission\nHelp.\n## Scope\n## Working principles\n"
    "## Tool-use principles\n## Memory principles\n## Escalation\n## Skill usage\n"
)
_BODY = "## Preconditions\np\n## Procedure\n1. step\n## Boundaries\nb\n## Completion\nc\n"


@contextmanager
def trace_context(trace_id: str) -> Iterator[None]:
    previous = get_current_trace_id()
    set_current_trace_id(trace_id)
    try:
        yield
    finally:
        set_current_trace_id(previous)


class _Screen:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, stage: str, status: str, *_: Any, **kwargs: Any) -> None:
        self.events.append({"stage": stage, "status": status, **kwargs})


def _write_skill(root: Path, name: str, triggers: str) -> None:
    folder = root / "skills" / name
    folder.mkdir(parents=True)
    front = f"---\nname: {name}\ndescription: {name} task\ntriggers: {triggers}\n---\n"
    (folder / "SKILL.md").write_text(front + _BODY, encoding="utf-8")


@pytest.fixture
def settings(tmp_path: Path) -> SkillSettings:
    (tmp_path / "AGENT.md").write_text(_PROFILE, encoding="utf-8")
    _write_skill(tmp_path, "refund-flow", "[intent:REFUND]")
    _write_skill(tmp_path, "track-flow", "[intent:TRACK]")
    return get_settings().skills.model_copy(
        update={
            "profile_path": str(tmp_path / "AGENT.md"),
            "skills_dir": str(tmp_path / "skills"),
        }
    )


@pytest.fixture
def audit(pool: SqliteConnectionPool) -> SqliteAuditLedger:
    return SqliteAuditLedger(pool)


@pytest.fixture
def screen() -> _Screen:
    return _Screen()


@pytest.fixture
def instructions(
    settings: SkillSettings,
    pool: SqliteConnectionPool,
    audit: SqliteAuditLedger,
    screen: _Screen,
) -> AgentInstructions:
    return build_agent_instructions(
        settings, store=ProceduralStore(pool), audit=audit, reporter=screen
    )


class TestTheCharter:
    def test_it_is_the_file_as_written(
        self, instructions: AgentInstructions, settings: SkillSettings
    ) -> None:
        assert instructions.profile_section() == _PROFILE.strip()
        assert render_profile(settings) == _PROFILE.strip()

    def test_without_procedural_memory_it_is_all_there_is(
        self, settings: SkillSettings, screen: _Screen
    ) -> None:
        bare = build_agent_instructions(settings, store=None, audit=None, reporter=screen)
        composed = bare.compose(frozenset({"intent:REFUND"}), "refund please")
        assert composed.text == _PROFILE.strip()
        assert composed.activation.skills == ()
        assert bare.catalog is None
        assert screen.events == []


class TestComposing:
    def test_the_matching_procedure_follows_the_profile(
        self, instructions: AgentInstructions
    ) -> None:
        text = instructions.compose(frozenset({"intent:REFUND"}), "").text
        assert text.startswith("# Test Agent")
        assert "# Procedure: refund-flow\n\n## Preconditions" in text
        assert "track-flow" not in text

    def test_the_frontmatter_never_reaches_the_prompt(
        self, instructions: AgentInstructions
    ) -> None:
        text = instructions.compose(frozenset({"intent:REFUND"}), "").text
        assert "triggers:" not in text

    def test_the_activation_names_versions(self, instructions: AgentInstructions) -> None:
        activation = instructions.compose(frozenset({"intent:REFUND"}), "").activation
        assert [item.skill.ref("{name}@{version}") for item in activation.skills] == [
            "refund-flow@1"
        ]


class TestRecording:
    def test_one_audit_entry_per_turn(
        self,
        instructions: AgentInstructions,
        audit: SqliteAuditLedger,
        settings: SkillSettings,
    ) -> None:
        with trace_context("trace-1"):
            for _ in range(3):  # a tool loop composes once per round
                instructions.compose(frozenset({"intent:REFUND"}), "q")
        entries = [
            e for e in audit.reconstruct("trace-1") if e.action == settings.audit.activation_action
        ]
        assert len(entries) == 1
        assert entries[0].authority_context["procedures"] == ["refund-flow@1"]
        assert entries[0].authority_context["profile"].startswith("profile@")

    def test_one_card_per_turn(
        self, instructions: AgentInstructions, screen: _Screen, settings: SkillSettings
    ) -> None:
        with trace_context("trace-2"):
            for _ in range(2):
                instructions.compose(frozenset({"intent:REFUND"}), "q")
        cards = [e for e in screen.events if e["stage"] == settings.observability.activation_stage]
        assert len(cards) == 1
        assert cards[0]["fields"]["procedures"] == ["refund-flow@1"]

    def test_the_start_up_sync_is_shown(
        self, instructions: AgentInstructions, screen: _Screen, settings: SkillSettings
    ) -> None:
        assert any(e["stage"] == settings.observability.sync_stage for e in screen.events)


class TestBuilding:
    def test_more_always_skills_than_the_cap_is_refused(
        self, tmp_path: Path, settings: SkillSettings, pool: SqliteConnectionPool
    ) -> None:
        for name in ("a-rule", "b-rule", "c-rule"):
            _write_skill(tmp_path, name, "[always]")
        with pytest.raises(SkillDocumentError, match="always"):
            build_agent_instructions(
                settings, store=ProceduralStore(pool), audit=None, reporter=None
            )
