"""Unit tests for config module."""

from agentic_commons.config import get_settings, is_enabled


def test_get_settings_singleton() -> None:
    """Test that get_settings returns a singleton."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_settings_has_all_subsystems() -> None:
    """Test that settings has all expected subsystems."""
    settings = get_settings()
    assert hasattr(settings, "logging")
    assert hasattr(settings, "config")
    assert hasattr(settings, "llm")
    assert hasattr(settings, "persistence")
    assert hasattr(settings, "observability")
    assert hasattr(settings, "audit")
    assert hasattr(settings, "evals")
    assert hasattr(settings, "guardrails")
    assert hasattr(settings, "resilience")
    assert hasattr(settings, "security")
    assert hasattr(settings, "memory")
    assert hasattr(settings, "tools")
    assert hasattr(settings, "gatekeeper")
    assert hasattr(settings, "hallucination_protection")
    assert hasattr(settings, "customer_support_agent")


def test_is_enabled_default_true() -> None:
    """Test that is_enabled returns True for enabled subsystems."""
    assert is_enabled("llm")
    assert is_enabled("persistence")
    assert is_enabled("security.pii")


def test_is_enabled_nonexistent_path() -> None:
    """Test that is_enabled returns False for nonexistent paths."""
    assert not is_enabled("nonexistent.path")


def test_logging_settings() -> None:
    """Test logging settings."""
    settings = get_settings()
    assert settings.logging.level == "INFO"
    assert settings.logging.colorize is True
    assert settings.logging.console_enabled is True
    assert settings.logging.file_enabled is True
    assert settings.logging.log_dir == "logs"
