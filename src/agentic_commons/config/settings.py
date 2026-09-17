"""Configuration settings for agentic commons using pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    level: str = "INFO"
    colorize: bool = True
    console_enabled: bool = True
    file_enabled: bool = True
    log_dir: str = "logs"

    model_config = SettingsConfigDict(env_prefix="LOGGING__")


class ConfigSettings(BaseSettings):
    """Meta config toggle."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="CONFIG__")


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="LLM__")


class PersistenceSettings(BaseSettings):
    """Database/persistence configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="PERSISTENCE__")


class ObservabilitySettings(BaseSettings):
    """Observability/tracing configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__")


class AuditSettings(BaseSettings):
    """Audit logging configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="AUDIT__")


class EvalsSettings(BaseSettings):
    """Evals runner configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="EVALS__")


class GuardrailsSettings(BaseSettings):
    """Guardrails configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="GUARDRAILS__")


class ResilienceSettings(BaseSettings):
    """Resilience patterns configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="RESILIENCE__")


class SecurityAuthSettings(BaseSettings):
    """Security auth subsystem configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="SECURITY__AUTH__")


class SecurityPIISettings(BaseSettings):
    """Security PII subsystem configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="SECURITY__PII__")


class SecurityPromptInjectionSettings(BaseSettings):
    """Security prompt injection subsystem configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="SECURITY__PROMPT_INJECTION__")


class SecurityContentModerationSettings(BaseSettings):
    """Security content moderation subsystem configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="SECURITY__CONTENT_MODERATION__")


class SecuritySettings(BaseSettings):
    """Security configuration."""

    enabled: bool = True
    auth: SecurityAuthSettings = SecurityAuthSettings()
    pii: SecurityPIISettings = SecurityPIISettings()
    prompt_injection: SecurityPromptInjectionSettings = (
        SecurityPromptInjectionSettings()
    )
    content_moderation: SecurityContentModerationSettings = (
        SecurityContentModerationSettings()
    )

    model_config = SettingsConfigDict(env_prefix="SECURITY__")


class MemorySettings(BaseSettings):
    """Memory configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="MEMORY__")


class ToolsSettings(BaseSettings):
    """Tools configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="TOOLS__")


class GatekeeperSettings(BaseSettings):
    """Gatekeeper configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="GATEKEEPER__")


class HallucinationProtectionSettings(BaseSettings):
    """Hallucination protection configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="HALLUCINATION_PROTECTION__")


class CustomerSupportAgentSettings(BaseSettings):
    """Customer support agent configuration."""

    enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="CUSTOMER_SUPPORT_AGENT__")


class Settings(BaseSettings):
    """Root settings object containing all subsystem configurations."""

    logging: LoggingSettings = LoggingSettings()
    config: ConfigSettings = ConfigSettings()
    llm: LLMSettings = LLMSettings()
    persistence: PersistenceSettings = PersistenceSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    audit: AuditSettings = AuditSettings()
    evals: EvalsSettings = EvalsSettings()
    guardrails: GuardrailsSettings = GuardrailsSettings()
    resilience: ResilienceSettings = ResilienceSettings()
    security: SecuritySettings = SecuritySettings()
    memory: MemorySettings = MemorySettings()
    tools: ToolsSettings = ToolsSettings()
    gatekeeper: GatekeeperSettings = GatekeeperSettings()
    hallucination_protection: HallucinationProtectionSettings = (
        HallucinationProtectionSettings()
    )
    customer_support_agent: CustomerSupportAgentSettings = (
        CustomerSupportAgentSettings()
    )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Get or create the singleton settings instance, reading from .env if present."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
