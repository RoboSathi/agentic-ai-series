"""Additional unit tests for coverage."""

import pytest

from agentic_commons.config import get_settings
from agentic_commons.logging.core import get_logger
from agentic_commons.logging.stub import SUBSYSTEM_COLORS, _resolve_toggle


def test_rich_logger_info() -> None:
    """Test RichLogger info method."""
    logger = get_logger("test")
    logger.info("test message")


def test_rich_logger_debug() -> None:
    """Test RichLogger debug method."""
    logger = get_logger("test")
    logger.debug("test debug")


def test_rich_logger_warning() -> None:
    """Test RichLogger warning method."""
    logger = get_logger("test")
    logger.warning("test warning")


def test_rich_logger_error() -> None:
    """Test RichLogger error method."""
    logger = get_logger("test")
    logger.error("test error")


def test_resolve_toggle_simple() -> None:
    """Test toggle resolution."""
    assert _resolve_toggle("llm.enabled") is True


def test_resolve_toggle_nested() -> None:
    """Test nested toggle resolution."""
    assert _resolve_toggle("security.pii.enabled") is True


def test_subsystem_colors() -> None:
    """Test that subsystem colors are defined."""
    assert len(SUBSYSTEM_COLORS) > 0
    assert "llm" in SUBSYSTEM_COLORS
    assert "security" in SUBSYSTEM_COLORS


def test_log_call_log_returned() -> None:
    """Test log_call returns correct value."""
    from agentic_commons.logging.stub import log_call

    @log_call("llm")
    def add(x: int, y: int) -> int:
        return x + y

    result = add(5, 3)
    assert result == 8


def test_log_call_with_kwargs() -> None:
    """Test log_call with keyword arguments."""
    from agentic_commons.logging.stub import log_call

    @log_call("llm")
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    result = greet("Alice", greeting="Hi")
    assert "Alice" in result


def test_settings_security_nested() -> None:
    """Test accessing nested security settings."""
    settings = get_settings()
    assert settings.security.pii.enabled is True


def test_settings_logging_config() -> None:
    """Test logging configuration."""
    settings = get_settings()
    assert settings.logging.log_dir == "logs"
    assert settings.logging.level == "INFO"


def test_get_logger_different_subsystems() -> None:
    """Test that different subsystems get different loggers."""
    logger1 = get_logger("subsys1")
    logger2 = get_logger("subsys2")
    assert logger1._logger is not logger2._logger


def test_placeholder_for_typed_dict() -> None:
    """Test placeholder creation for dict with type hints."""
    from agentic_commons.logging.stub import _placeholder_for

    def func() -> dict[str, int]:
        pass

    result = _placeholder_for(func)
    assert result == {}


def test_placeholder_for_list_typed() -> None:
    """Test placeholder creation for list with type hints."""
    from agentic_commons.logging.stub import _placeholder_for

    def func() -> list[str]:
        pass

    result = _placeholder_for(func)
    assert result == []


def test_placeholder_for_tuple() -> None:
    """Test placeholder creation for tuple."""
    from agentic_commons.logging.stub import _placeholder_for

    def func() -> tuple:
        pass

    result = _placeholder_for(func)
    assert result == ()


def test_placeholder_for_set() -> None:
    """Test placeholder creation for set."""
    from agentic_commons.logging.stub import _placeholder_for

    def func() -> set:
        pass

    result = _placeholder_for(func)
    assert result == set()


def test_placeholder_for_float() -> None:
    """Test placeholder creation for float."""
    from agentic_commons.logging.stub import _placeholder_for

    def func() -> float:
        pass

    result = _placeholder_for(func)
    assert result == 0.0


def test_placeholder_for_pydantic_model() -> None:
    """Test placeholder creation for Pydantic model."""
    from pydantic import BaseModel

    from agentic_commons.logging.stub import _placeholder_for

    class TestModel(BaseModel):
        value: str = "default"

    def func() -> TestModel:
        pass

    result = _placeholder_for(func)
    assert isinstance(result, TestModel)


def test_placeholder_for_unhandled_type() -> None:
    """Test placeholder creation for unhandled type raises error."""
    from agentic_commons.logging.stub import _placeholder_for

    class UnconstructibleClass:
        def __init__(self, required_arg: str) -> None:
            self.required_arg = required_arg

    def func() -> UnconstructibleClass:
        pass

    with pytest.raises(NotImplementedError):
        _placeholder_for(func)
