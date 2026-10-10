"""The agent profile and a turn's procedures, rendered, checked and recorded (ADR 0010).

Real files, a real procedural store, a real audit ledger and a real tool registry; the
only thing faked is the screen — a reporter that keeps what it was shown.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

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
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.types import ToolDefinition, ToolResult

_PROFILE = "# Test Agent\n\n## Mission\nHelp.\n## Scope\nOrders.\n## Rules\n- Be exact.\n"
_BODY = (
    "## Preconditions\nfor people\n## Procedure\n1. step one\n2. step two\n"
    "## Boundaries\nb\n## Completion\nfor people\n"
)


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

    def of(self, stage: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["stage"] == stage]


def _write_skill(root: Path, name: str, front: str) -> None:
    folder = root / "skills" / name
    folder.mkdir(parents=True, exist_ok=True)
    head = f"---\nname: {name}\ndescription: {name} task\n{front}---\n"
    (folder / "SKILL.md").write_text(head + _BODY, encoding="utf-8")


class _Args(BaseModel):
    pass


def _tool(name: str, *, side_effecting: bool = False) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        namespace="test",
        description=name,
        args_schema=_Args,
        handler=lambda: ToolResult(success=True),
        side_effecting=side_effecting,
    )


@pytest.fixture
def registry() -> NamespacedToolRegistry:
    tools = NamespacedToolRegistry()
    tools.register_all([_tool("look_up"), _tool("pay", side_effecting=True)])
    return tools


@pytest.fixture
def settings(tmp_path: Path) -> SkillSettings:
    (tmp_path / "AGENT.md").write_text(_PROFILE, encoding="utf-8")
    _write_skill(tmp_path, "refund-flow", "triggers: [intent:REFUND]\nprefetch: [look_up]\n")
    _write_skill(tmp_path, "track-flow", "triggers: [intent:TRACK]\n")
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
    registry: NamespacedToolRegistry,
) -> AgentInstructions:
    return build_agent_instructions(
        settings, store=ProceduralStore(pool), audit=audit, reporter=screen, tools=registry
    )


_REFUND = frozenset({"intent:REFUND"})


class TestTheProfile:
    def test_it_is_the_file_as_written(
        self, instructions: AgentInstructions, settings: SkillSettings
    ) -> None:
        assert instructions.profile_section() == _PROFILE.strip()
        assert render_profile(settings) == _PROFILE.strip()

    def test_without_procedural_memory_it_is_all_there_is(
        self, settings: SkillSettings, screen: _Screen
    ) -> None:
        bare = build_agent_instructions(settings, store=None, audit=None, reporter=screen)
        composed = bare.compose(_REFUND, "refund please")
        assert composed.text == _PROFILE.strip()
        assert bare.catalog is None
        assert screen.events == []


class TestRendering:
    def test_only_the_sections_the_model_acts_on_are_sent(
        self, instructions: AgentInstructions
    ) -> None:
        text = instructions.compose(_REFUND, "").text
        assert "# Procedure: refund-flow\n\n## Procedure\n1. step one" in text
        assert "## Boundaries" in text
        assert "for people" not in text
        assert "triggers:" not in text

    def test_select_then_render_is_compose(self, instructions: AgentInstructions) -> None:
        activation = instructions.select(_REFUND, "")
        assert instructions.render(activation) == instructions.compose(_REFUND, "").text

    def test_the_activation_lists_prefetch_reads(self, instructions: AgentInstructions) -> None:
        assert instructions.select(_REFUND, "").prefetch() == ["look_up"]


class TestTheProcedureCard:
    def test_one_line_says_what_was_used_and_why(
        self, instructions: AgentInstructions, screen: _Screen, settings: SkillSettings
    ) -> None:
        with trace_context("t1"):
            activation = instructions.select(_REFUND, "")
            instructions.announce(activation, prefetched=["look_up"])
        [card] = screen.of(settings.observability.activation_stage)
        assert card["summary"] == (
            "refund-flow v1 · matched intent REFUND · 2 steps · reads first: look_up"
        )
        assert card["fields"]["procedures"][0]["matched"] == "matched intent REFUND"
        assert "profile" not in card["fields"] and "signals" not in card["fields"]

    def test_nothing_is_shown_or_recorded_when_no_procedure_is_used(
        self,
        instructions: AgentInstructions,
        screen: _Screen,
        audit: SqliteAuditLedger,
        settings: SkillSettings,
    ) -> None:
        with trace_context("t2"):
            instructions.compose(frozenset({"intent:OTHER"}), "hello")
        assert screen.of(settings.observability.activation_stage) == []
        assert audit.reconstruct("t2") == []

    def test_once_per_turn_however_often_a_loop_composes(
        self,
        instructions: AgentInstructions,
        screen: _Screen,
        audit: SqliteAuditLedger,
        settings: SkillSettings,
    ) -> None:
        with trace_context("t3"):
            for _ in range(3):
                instructions.compose(_REFUND, "q")
        assert len(screen.of(settings.observability.activation_stage)) == 1
        [entry] = audit.reconstruct("t3")
        assert entry.authority_context["procedures"] == ["refund-flow@1"]
        assert entry.authority_context["profile"].startswith("profile@")


class TestTheStartUpSync:
    def test_a_first_sync_is_shown(self, instructions: AgentInstructions, screen: _Screen) -> None:
        assert screen.of(get_settings().skills.observability.sync_stage)

    def test_an_unchanged_sync_draws_no_card(
        self,
        settings: SkillSettings,
        pool: SqliteConnectionPool,
        registry: NamespacedToolRegistry,
    ) -> None:
        build_agent_instructions(
            settings, store=ProceduralStore(pool), audit=None, reporter=None, tools=registry
        )
        again = _Screen()
        build_agent_instructions(
            settings, store=ProceduralStore(pool), audit=None, reporter=again, tools=registry
        )
        assert again.events == []


class TestBuildChecks:
    def _build(self, settings: SkillSettings, pool: SqliteConnectionPool, **kwargs: Any) -> Any:
        return build_agent_instructions(
            settings, store=ProceduralStore(pool), audit=None, reporter=None, **kwargs
        )

    def test_more_always_skills_than_the_cap(
        self, tmp_path: Path, settings: SkillSettings, pool: SqliteConnectionPool
    ) -> None:
        for index in range(settings.selection.max_active + 1):
            _write_skill(tmp_path, f"rule-{index}", "triggers: [always]\n")
        with pytest.raises(SkillDocumentError, match="always"):
            self._build(settings, pool)

    def test_a_prompt_over_budget(
        self, settings: SkillSettings, pool: SqliteConnectionPool
    ) -> None:
        tight = settings.model_copy(update={"max_prompt_chars": 10})
        with pytest.raises(SkillDocumentError, match="max_prompt_chars"):
            self._build(tight, pool)

    def test_a_tool_the_agent_does_not_offer(
        self, tmp_path: Path, settings: SkillSettings, pool: SqliteConnectionPool
    ) -> None:
        _write_skill(tmp_path, "odd-flow", "tools: [nowhere]\n")
        tools = NamespacedToolRegistry()
        tools.register(_tool("look_up"))
        with pytest.raises(SkillDocumentError, match="nowhere"):
            self._build(settings, pool, tools=tools)

    def test_a_prefetch_that_would_change_something(
        self,
        tmp_path: Path,
        settings: SkillSettings,
        pool: SqliteConnectionPool,
        registry: NamespacedToolRegistry,
    ) -> None:
        _write_skill(tmp_path, "pay-flow", "prefetch: [pay]\n")
        with pytest.raises(SkillDocumentError, match="effect"):
            self._build(settings, pool, tools=registry)
