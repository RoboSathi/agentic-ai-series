"""Unit tests for config module."""

from pathlib import Path

import pytest
import yaml

from agentic_commons.config import get_settings, is_enabled, yaml_config
from agentic_commons.config.settings import LLMSettings


def _write_llm_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: dict[str, object],
    local: dict[str, object] | None = None,
) -> None:
    """Point yaml_config at isolated temp files for a single test."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump({"llm": config}))
    monkeypatch.setattr(yaml_config, "_CONFIG_FILE", config_file)

    local_file = tmp_path / "local.yaml"
    if local is not None:
        local_file.write_text(yaml.safe_dump({"llm": local}))
    monkeypatch.setattr(yaml_config, "_LOCAL_FILE", local_file)


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


def test_is_enabled_for_core_subsystems() -> None:
    """Test that is_enabled returns True for the core, always-on subsystems."""
    assert is_enabled("llm")
    assert is_enabled("persistence")
    assert is_enabled("tools")


def test_is_enabled_for_optional_subsystems_off_by_default() -> None:
    """Test that optional safety/governance layers default to disabled."""
    assert not is_enabled("security")
    assert not is_enabled("security.pii")
    assert not is_enabled("memory")
    assert not is_enabled("guardrails")


def test_is_enabled_bare_path_matches_dotted_enabled_path() -> None:
    """Test that a bare subsystem path resolves the same as its .enabled path.

    Regression test: is_enabled("security.pii") used to resolve to the
    SecurityPIISettings object itself, and bool() on any pydantic model is
    always True regardless of its fields — so this always returned True
    even when security.pii.enabled was False.
    """
    assert is_enabled("security.pii") == is_enabled("security.pii.enabled")


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


def test_llm_settings_reads_provider_and_connection_fields_from_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that provider/base_url/model/timeout come from config.yaml, not code."""
    _write_llm_yaml(
        monkeypatch,
        tmp_path,
        {
            "enabled": True,
            "provider": "nim",
            "base_url": "http://from-config-yaml/v1",
            "model": "config-yaml-model",
            "timeout_seconds": 12,
            "max_tokens": 200,
        },
    )

    settings = LLMSettings()

    assert settings.provider == "nim"
    assert settings.base_url == "http://from-config-yaml/v1"
    assert settings.model == "config-yaml-model"
    assert settings.timeout_seconds == 12
    assert settings.max_tokens == 200


def test_llm_settings_local_yaml_overrides_config_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that local.yaml values win over config.yaml, field by field."""
    _write_llm_yaml(
        monkeypatch,
        tmp_path,
        config={
            "enabled": True,
            "provider": "nim",
            "base_url": "http://from-config/v1",
            "model": "config-model",
            "timeout_seconds": 30,
            "max_tokens": 150,
        },
        local={
            "provider": "local",
            "base_url": "http://from-local/v1",
            "model": "local-model",
        },
    )

    settings = LLMSettings()

    assert settings.provider == "local"
    assert settings.base_url == "http://from-local/v1"
    assert settings.model == "local-model"
    assert settings.timeout_seconds == 30  # not touched by local.yaml
    assert settings.max_tokens == 150  # not touched by local.yaml


def test_llm_settings_explicit_kwargs_override_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that an explicit constructor kwarg wins over the yaml-sourced value."""
    _write_llm_yaml(
        monkeypatch,
        tmp_path,
        {
            "enabled": True,
            "provider": "nim",
            "base_url": "http://from-config/v1",
            "model": "config-model",
            "timeout_seconds": 30,
            "max_tokens": 150,
        },
    )

    settings = LLMSettings(model="explicit-model")

    assert settings.model == "explicit-model"
    assert settings.base_url == "http://from-config/v1"


def test_llm_settings_missing_yaml_field_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that an incomplete config.yaml fails fast instead of silently defaulting."""
    _write_llm_yaml(monkeypatch, tmp_path, {"provider": "nim"})

    with pytest.raises(ValueError):
        LLMSettings()


def test_llm_settings_api_key_resolved_from_env_for_nim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that the nim provider reads the bare nvidia_api_key env var."""
    monkeypatch.setenv("nvidia_api_key", "test-key-123")
    settings = LLMSettings(provider="nim")
    assert settings.api_key == "test-key-123"


def test_llm_settings_api_key_none_for_local() -> None:
    """Test that the local provider never requires an api key."""
    settings = LLMSettings(provider="local")
    assert settings.api_key is None


def test_llm_settings_api_key_not_a_model_field() -> None:
    """Test that api_key never appears in a settings dump (no leaking to logs)."""
    settings = LLMSettings(provider="nim")
    assert "api_key" not in settings.model_dump()
