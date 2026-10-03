"""Configuration module for the agent platform."""

from agent_platform.config.features import (
    FeatureSet,
    active_features,
    set_active_features,
)
from agent_platform.config.paths import REPO_ROOT, resolve_path
from agent_platform.config.settings import (
    AppConfig,
    YamlBackedSettings,
    get_settings,
    reload_settings,
)
from agent_platform.config.toggles import (
    ConfigFeatureToggle,
    FeatureToggle,
    is_enabled,
    set_enabled,
)
from agent_platform.config.yaml_config import (
    clear_overlays,
    config_sources,
    use_overlay,
)

__all__ = [
    "AppConfig",
    "ConfigFeatureToggle",
    "FeatureSet",
    "FeatureToggle",
    "REPO_ROOT",
    "YamlBackedSettings",
    "active_features",
    "clear_overlays",
    "config_sources",
    "get_settings",
    "is_enabled",
    "reload_settings",
    "resolve_path",
    "set_active_features",
    "set_enabled",
    "use_overlay",
]
