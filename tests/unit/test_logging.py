"""Unit tests for logging module."""



from agentic_commons.logging.core import get_logger
from agentic_commons.logging.stub import _placeholder_for, log_call


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
