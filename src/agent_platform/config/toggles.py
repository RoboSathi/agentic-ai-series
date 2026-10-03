"""Live feature toggles (LLD §1, PRD G6).

Every governance feature and sub-feature can be switched on or off at
runtime, with no restart, so its marginal effect is measurable rather than
merely asserted. Reads are live — never cached — because the TUI flips
these mid-run.

Sub-features are parent-gated: a section is only enabled if its parent is
too, and a section with no `enabled` field is on by construction.

This is the older mechanism, and it now covers only the always-on
plumbing an agent cannot be composed without. Governance *capabilities*
live in `features.py` instead: they are read once, by a composition root,
so that two agents in one process can hold different postures — which a
process-wide toggle cannot express.
"""

from typing import Any, Protocol, runtime_checkable

from agent_platform.config.settings import AppConfig, get_settings


@runtime_checkable
class FeatureToggle(Protocol):
    """Reads and writes governance feature flags live."""

    def is_enabled(self, feature: str, sub_feature: str | None = None) -> bool: ...

    def set(self, feature: str, sub_feature: str | None, value: bool) -> None: ...


def _path_parts(feature: str, sub_feature: str | None) -> list[str]:
    parts = feature.split(".")
    if sub_feature:
        parts.extend(sub_feature.split("."))
    return parts


class ConfigFeatureToggle:
    """FeatureToggle backed by the live config singleton."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config

    @property
    def _root(self) -> AppConfig:
        return self._config or get_settings()

    def is_enabled(self, feature: str, sub_feature: str | None = None) -> bool:
        """True only if this node and every ancestor section is enabled."""
        node: Any = self._root
        for part in _path_parts(feature, sub_feature):
            node = getattr(node, part, None)
            if node is None:
                return False
            if isinstance(node, bool):
                return node
            if getattr(node, "enabled", True) is False:
                return False
        return bool(getattr(node, "enabled", True))

    def set(self, feature: str, sub_feature: str | None, value: bool) -> None:
        """Flip a toggle on the live singleton; raises if the path is unknown."""
        parts = _path_parts(feature, sub_feature)
        node: Any = self._root
        for part in parts:
            parent, node = node, getattr(node, part, None)
            if node is None:
                raise KeyError(f"Unknown feature path: {'.'.join(parts)}")
        if isinstance(node, bool):
            setattr(parent, parts[-1], value)
        else:
            setattr(node, "enabled", value)


_default_toggle = ConfigFeatureToggle()


def is_enabled(feature: str, sub_feature: str | None = None) -> bool:
    """Module-level convenience over the default toggle."""
    return _default_toggle.is_enabled(feature, sub_feature)


def set_enabled(feature: str, sub_feature: str | None, value: bool) -> None:
    """Module-level convenience over the default toggle."""
    _default_toggle.set(feature, sub_feature, value)
