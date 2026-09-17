"""Unit tests for logging module."""

import logging
from typing import Any, Optional

from pydantic import BaseModel

from agentic_commons.logging.core import PlainFormatter, get_logger
from agentic_commons.logging.stub import _placeholder_for, log_call


def _make_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="agentic.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )


def test_plain_formatter_strips_rich_markup() -> None:
    """Test that Rich markup tags don't leak into file-formatted output."""
    formatter = PlainFormatter("%(message)s")
    record = _make_record("[bright_green]Intake node[/bright_green]")

    assert formatter.format(record) == "Intake node"


def test_plain_formatter_strips_markup_from_log_call_style_message() -> None:
    """Test stripping on a realistic CALLED/RETURNED-style message."""
    formatter = PlainFormatter("%(message)s")
    record = _make_record(
        "[bold bright_blue]CALLED[/bold bright_blue] [cyan]Client.generate[/cyan](x)"
    )

    assert formatter.format(record) == "CALLED Client.generate(x)"


def test_plain_formatter_leaves_non_markup_brackets_intact() -> None:
    """Test that literal brackets in message content (not valid markup) survive."""
    formatter = PlainFormatter("%(message)s")
    record = _make_record("RETURNED foo -> [Message(role='user')]")

    assert formatter.format(record) == "RETURNED foo -> [Message(role='user')]"


def test_get_logger_returns_rich_logger() -> None:
    """Test that get_logger returns a RichLogger."""
    logger = get_logger("test_subsystem")
    assert hasattr(logger, "log_called")
    assert hasattr(logger, "log_returned")
    assert hasattr(logger, "log_skipped")


def test_get_logger_singleton_per_subsystem() -> None:
    """Test that get_logger returns same logger for same subsystem."""
    logger1 = get_logger("test")
    logger2 = get_logger("test")
    assert logger1._logger is logger2._logger


def test_log_call_enabled() -> None:
    """Test that log_call decorator calls the function when enabled."""

    @log_call("llm")
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 10


def test_log_call_disabled() -> None:
    """Test that log_call decorator skips when disabled."""

    @log_call("nonexistent_subsystem")
    def test_func(x: int) -> int:
        return x * 2

    result = test_func(5)
    assert result == 0


def test_placeholder_for_bool() -> None:
    """Test placeholder creation for bool."""

    def func() -> bool:
        pass

    result = _placeholder_for(func)
    assert result is False


def test_placeholder_for_str() -> None:
    """Test placeholder creation for str."""

    def func() -> str:
        pass

    result = _placeholder_for(func)
    assert result == ""


def test_placeholder_for_int() -> None:
    """Test placeholder creation for int."""

    def func() -> int:
        pass

    result = _placeholder_for(func)
    assert result == 0


def test_placeholder_for_list() -> None:
    """Test placeholder creation for list."""

    def func() -> list:
        pass

    result = _placeholder_for(func)
    assert result == []


def test_placeholder_for_dict() -> None:
    """Test placeholder creation for dict."""

    def func() -> dict:
        pass

    result = _placeholder_for(func)
    assert result == {}


def test_placeholder_for_none() -> None:
    """Test placeholder creation for None."""

    def func() -> None:
        pass

    result = _placeholder_for(func)
    assert result is None


class _Widget(BaseModel):
    name: str = ""


def test_placeholder_for_optional_model_type() -> None:
    """Test that Optional[BaseModel] placeholders to None, not a constructed instance.

    Regression test: Optional[Skill] used to raise NotImplementedError
    because typing.Union isn't directly constructible.
    """

    def func() -> Optional[_Widget]:
        pass

    assert _placeholder_for(func) is None


def test_placeholder_for_pipe_union_with_none() -> None:
    """Test that the X | None union syntax also placeholders to None."""

    def func() -> _Widget | None:
        pass

    assert _placeholder_for(func) is None


def test_placeholder_for_any() -> None:
    """Test that a bare Any return annotation placeholders to None."""

    def func() -> Any:
        pass

    assert _placeholder_for(func) is None
