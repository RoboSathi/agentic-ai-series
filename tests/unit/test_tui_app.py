"""Tests for the interactive terminal UI app."""

import asyncio
import time
from pathlib import Path
from typing import Any, Coroutine

import pytest
from textual.widgets import Input, Static

from agentic_commons.config import get_settings
from agentic_commons.llm.types import HealthCheckResult, LLMResponse, Message
from agentic_commons.persistence.database import Database
from agentic_commons.tui.app import AgenticTUI, LogDisplay


class _FakeLLMClient:
    """Fake LLMClient so TUI tests never hit the network."""

    def __init__(self, content: str = "[fake response]", delay_seconds: float = 0.0) -> None:
        self._content = content
        self._delay_seconds = delay_seconds

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        return LLMResponse(content=self._content, model="fake")

    def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def health_check(self) -> HealthCheckResult:
        raise NotImplementedError


def _status_text(app: AgenticTUI) -> str:
    """Read the status-line widget's current content (no public accessor)."""
    return str(app.query_one("#status-line", Static)._Static__content)  # type: ignore[attr-defined]


def _run(coro: Coroutine[None, None, None]) -> None:
    asyncio.run(coro)


def test_tui_mounts_with_instructions() -> None:
    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test():
            log_display = app.query_one("#logs", LogDisplay)
            assert any("Interactive TUI" in line for line in log_display.logs)

    _run(scenario())


def test_tui_submits_query_and_shows_response(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AgenticTUI(
            llm_client=_FakeLLMClient("Here is your answer."),
            database=Database(db_path=tmp_path / "test.db"),
        )
        async with app.run_test() as pilot:
            app.query_one("#input-field", Input).value = "my refund question"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            log_display = app.query_one("#logs", LogDisplay)
            lines = log_display.logs

            assert any("my refund question" in line for line in lines)
            assert any("Here is your answer." in line for line in lines)
            assert any("Processing Complete" in line for line in lines)

    _run(scenario())


def test_tui_status_line_animates_and_clears_on_completion(tmp_path: Path) -> None:
    """Test the spinner+timer status line: distinct frames while processing, empty after.

    Regression test: #status-line, #input-field, and Footer were all
    `dock: bottom`, which doesn't stack multiple docked widgets on the same
    edge — it overlaps them. The status line was rendering but completely
    hidden the whole time.
    """

    async def scenario() -> None:
        app = AgenticTUI(
            llm_client=_FakeLLMClient(delay_seconds=0.4),
            database=Database(db_path=tmp_path / "test.db"),
        )
        async with app.run_test() as pilot:
            # No overlap: each of these widgets must occupy distinct rows.
            logs_region = app.query_one("#logs").region
            status_region = app.query_one("#status-line").region
            input_region = app.query_one("#input-field").region
            footer_region = app.query_one("Footer").region
            regions = [logs_region, status_region, input_region, footer_region]
            for a, b in zip(regions, regions[1:]):
                assert not a.overlaps(b)

            app.query_one("#input-field", Input).value = "hello"
            await pilot.press("enter")

            frames_seen: set[str] = set()
            for _ in range(10):
                await asyncio.sleep(0.05)
                frames_seen.add(_status_text(app))

            await app.workers.wait_for_complete()
            await pilot.pause()

            assert len(frames_seen - {""}) > 1  # spinner actually advanced
            assert all("Accomplishing" in f for f in frames_seen if f)
            assert _status_text(app) == ""  # cleared once done

    _run(scenario())


def test_tui_rejects_empty_query() -> None:
    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test() as pilot:
            await pilot.press("enter")
            await pilot.pause()

            log_display = app.query_one("#logs", LogDisplay)
            assert any("Type a question first" in line for line in log_display.logs)
            assert app.processing is False

    _run(scenario())


def test_tui_log_display_excludes_debug_trace_lines(tmp_path: Path) -> None:
    """The on-screen log stays curated — CALLED/RETURNED trace goes to files only."""

    async def scenario() -> None:
        app = AgenticTUI(
            llm_client=_FakeLLMClient(),
            database=Database(db_path=tmp_path / "test.db"),
        )
        async with app.run_test() as pilot:
            app.query_one("#input-field", Input).value = "a real query"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            log_display = app.query_one("#logs", LogDisplay)
            assert not any("CALLED" in line for line in log_display.logs)
            assert not any("RETURNED" in line for line in log_display.logs)

    _run(scenario())


def test_tui_copies_last_query_to_clipboard(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = AgenticTUI(
            llm_client=_FakeLLMClient(),
            database=Database(db_path=tmp_path / "test.db"),
        )
        async with app.run_test() as pilot:
            app.query_one("#input-field", Input).value = "copy me"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            await pilot.press("ctrl+y")

            assert app._clipboard == "copy me"

    _run(scenario())


def test_tui_copies_full_log_transcript_without_markup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings().logging, "log_dir", str(tmp_path))

    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+l")

            assert "Interactive TUI" in app._clipboard
            assert "[bright_green]" not in app._clipboard
            assert "[/bright_green]" not in app._clipboard

    _run(scenario())

    exported = list(tmp_path.glob("tui_log_export_*.txt"))
    assert len(exported) == 1
    exported_text = exported[0].read_text()
    assert "Interactive TUI" in exported_text
    assert "[bright_green]" not in exported_text


def test_tui_copy_logs_handles_empty_log() -> None:
    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test() as pilot:
            app.query_one("#logs", LogDisplay).clear_logs()

            await pilot.press("ctrl+l")

            assert not app._clipboard

    _run(scenario())


def test_tui_removes_log_handlers_on_unmount() -> None:
    """Test that exiting the app doesn't leave handlers on global loggers.

    Regression test: every AgenticTUI mount registered a LogHandler on the
    shared `agentic.*` loggers and never removed it, so later app instances
    (or unrelated code) kept triggering handlers pointing at apps that had
    already exited.
    """

    async def scenario() -> None:
        import logging

        logger = logging.getLogger("agentic.customer_support_agent")
        before = len(logger.handlers)

        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test():
            assert len(logger.handlers) > before

        assert len(logger.handlers) == before

    _run(scenario())


def test_tui_rejects_new_query_while_processing() -> None:
    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test() as pilot:
            app.processing = True
            await pilot.press("enter")
            await pilot.pause()

            log_display = app.query_one("#logs", LogDisplay)
            assert any("still processing" in line for line in log_display.logs)

    _run(scenario())
