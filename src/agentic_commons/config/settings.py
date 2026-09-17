"""Configuration settings for agentic commons using pydantic-settings.

Every subsystem's `enabled` toggle — and, for `llm`, its connection
parameters — is read from config.yaml (overridable by the gitignored
local.yaml), never hardcoded as a Python default. See YamlBackedSettings and
agentic_commons/config/yaml_config.py.
"""

import os
from typing import Any, ClassVar, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_commons.config.yaml_config import load_section


class YamlBackedSettings(BaseSettings):
    """Base for settings sourced from config.yaml.

    Subclasses set `_yaml_section` to their (optionally dotted, e.g.
    "security.pii") path in config.yaml. Any field not explicitly given via
    env var or constructor kwarg is filled from that section; a field with
    no Python default and no matching YAML key fails fast with a validation
    error rather than silently falling back to a hardcoded value.
    """

    _yaml_section: ClassVar[str] = ""

    @model_validator(mode="before")
    @classmethod
    def _fill_from_yaml(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return {**load_section(cls._yaml_section), **data}


class LoggingSettings(YamlBackedSettings):
    """Logging configuration."""

    _yaml_section = "logging"

    enabled: bool
    level: str = "INFO"
    colorize: bool = True
    console_enabled: bool = True
    file_enabled: bool = True
    log_dir: str = "logs"

    model_config = SettingsConfigDict(env_prefix="LOGGING__")


class ConfigSettings(YamlBackedSettings):
    """Meta config toggle."""

    _yaml_section = "config"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="CONFIG__")


class LLMSettings(YamlBackedSettings):
    """Unified LLM provider configuration.

    One shape covers any OpenAI-compatible endpoint, cloud or local.
    provider/base_url/model/timeout_seconds/max_tokens are never hardcoded
    here — they are read from config.yaml. API keys are never config values
    at all: they are resolved from the provider's bare env var by
    convention (see `api_key`), so they never appear in a settings dump or
    log line.
    """

    _yaml_section = "llm"

    enabled: bool
    provider: Literal["nim", "local"]
    base_url: str
    model: str
    timeout_seconds: float
    max_tokens: int

    model_config = SettingsConfigDict(env_prefix="LLM__")

    @property
    def api_key(self) -> str | None:
        """Resolve the API key from the provider's bare env var convention."""
        if self.provider == "nim":
            return os.environ.get("nvidia_api_key")
        return None


class PersistenceSettings(YamlBackedSettings):
    """Database/persistence configuration."""

    _yaml_section = "persistence"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="PERSISTENCE__")


class ObservabilitySettings(YamlBackedSettings):
    """Observability/tracing configuration."""

    _yaml_section = "observability"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class AuditSettings(YamlBackedSettings):
    """Audit logging configuration."""

    _yaml_section = "audit"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="AUDIT__")


class EvalsSettings(YamlBackedSettings):
    """Evals runner configuration."""

    _yaml_section = "evals"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="EVALS__")


class GuardrailsSettings(YamlBackedSettings):
    """Guardrails configuration."""

    _yaml_section = "guardrails"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS__")


class ResilienceSettings(YamlBackedSettings):
    """Resilience patterns configuration."""

    _yaml_section = "resilience"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="RESILIENCE__")


class SecurityAuthSettings(YamlBackedSettings):
    """Security auth subsystem configuration."""

    _yaml_section = "security.auth"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="SECURITY__AUTH__")


class SecurityPIISettings(YamlBackedSettings):
    """Security PII subsystem configuration."""

    _yaml_section = "security.pii"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="SECURITY__PII__")


class SecurityPromptInjectionSettings(YamlBackedSettings):
    """Security prompt injection subsystem configuration."""

    _yaml_section = "security.prompt_injection"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="SECURITY__PROMPT_INJECTION__")


class SecurityContentModerationSettings(YamlBackedSettings):
    """Security content moderation subsystem configuration."""

    _yaml_section = "security.content_moderation"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="SECURITY__CONTENT_MODERATION__")


class SecuritySettings(YamlBackedSettings):
    """Security configuration."""

    _yaml_section = "security"

    # Nested settings classes fill `enabled` from config.yaml via
    # YamlBackedSettings; mypy can't see that, hence the ignores.
    enabled: bool
    auth: SecurityAuthSettings = SecurityAuthSettings()  # type: ignore[call-arg]
    pii: SecurityPIISettings = SecurityPIISettings()  # type: ignore[call-arg]
    prompt_injection: SecurityPromptInjectionSettings = (
        SecurityPromptInjectionSettings()  # type: ignore[call-arg]
    )
    content_moderation: SecurityContentModerationSettings = (
        SecurityContentModerationSettings()  # type: ignore[call-arg]
    )

    model_config = SettingsConfigDict(env_prefix="SECURITY__")


class MemorySettings(YamlBackedSettings):
    """Memory configuration."""

    _yaml_section = "memory"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="MEMORY__")


class ToolsSettings(YamlBackedSettings):
    """Tools configuration."""

    _yaml_section = "tools"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="TOOLS__")


class GatekeeperSettings(YamlBackedSettings):
    """Gatekeeper configuration."""

    _yaml_section = "gatekeeper"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="GATEKEEPER__")


class HallucinationProtectionSettings(YamlBackedSettings):
    """Hallucination protection configuration."""

    _yaml_section = "hallucination_protection"

    enabled: bool

    model_config = SettingsConfigDict(env_prefix="HALLUCINATION_PROTECTION__")


class CustomerSupportAgentSettings(YamlBackedSettings):
    """Customer support agent configuration."""

    _yaml_section = "customer_support_agent"

    enabled: bool
    max_tool_iterations: int

    model_config = SettingsConfigDict(env_prefix="CUSTOMER_SUPPORT_AGENT__")


class Settings(BaseSettings):
    """Root settings object containing all subsystem configurations."""

    # Every nested settings class fills its required fields from
    # config.yaml via YamlBackedSettings; mypy can't see that, hence the
    # ignores on each zero-arg construction below.
    logging: LoggingSettings = LoggingSettings()  # type: ignore[call-arg]
    config: ConfigSettings = ConfigSettings()  # type: ignore[call-arg]
    llm: LLMSettings = LLMSettings()  # type: ignore[call-arg]
    persistence: PersistenceSettings = PersistenceSettings()  # type: ignore[call-arg]
    observability: ObservabilitySettings = ObservabilitySettings()  # type: ignore[call-arg]
    audit: AuditSettings = AuditSettings()  # type: ignore[call-arg]
    evals: EvalsSettings = EvalsSettings()  # type: ignore[call-arg]
    guardrails: GuardrailsSettings = GuardrailsSettings()  # type: ignore[call-arg]
    resilience: ResilienceSettings = ResilienceSettings()  # type: ignore[call-arg]
    security: SecuritySettings = SecuritySettings()  # type: ignore[call-arg]
    memory: MemorySettings = MemorySettings()  # type: ignore[call-arg]
    tools: ToolsSettings = ToolsSettings()  # type: ignore[call-arg]
    gatekeeper: GatekeeperSettings = GatekeeperSettings()  # type: ignore[call-arg]
    hallucination_protection: HallucinationProtectionSettings = (
        HallucinationProtectionSettings()  # type: ignore[call-arg]
    )
    customer_support_agent: CustomerSupportAgentSettings = (
        CustomerSupportAgentSettings()  # type: ignore[call-arg]
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get or create the singleton settings instance, reading from .env if present."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
