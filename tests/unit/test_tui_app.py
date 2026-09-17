"""Tests for the interactive terminal UI app."""

import asyncio
from pathlib import Path
from typing import Any, Coroutine

from textual.widgets import Input

from agentic_commons.llm.types import HealthCheckResult, LLMResponse, Message
from agentic_commons.persistence.database import Database
from agentic_commons.tui.app import AgenticTUI, LogDisplay


class _FakeLLMClient:
    """Fake LLMClient so TUI tests never hit the network."""

    def __init__(self, content: str = "[fake response]") -> None:
        self._content = content

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content=self._content, model="fake")

    def generate_stream(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def health_check(self) -> HealthCheckResult:
        raise NotImplementedError


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


def test_tui_copies_full_log_transcript_without_markup() -> None:
    async def scenario() -> None:
        app = AgenticTUI(llm_client=_FakeLLMClient())
        async with app.run_test() as pilot:
            await pilot.press("ctrl+l")

            assert "Interactive TUI" in app._clipboard
            assert "[bright_green]" not in app._clipboard
            assert "[/bright_green]" not in app._clipboard

    _run(scenario())


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
