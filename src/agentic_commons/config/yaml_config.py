"""YAML-backed configuration loading.

Configuration is layered, and the layers answer different questions:

  `agentic_commons/config/config.yaml`   what is true for *any* agent
  each overlay it declares               what is true for *this* agent
  `local.yaml` (gitignored)              what is true on *this machine*

The split matters because the platform is shared. Action names, role
tables, refund thresholds and tool wording belong to the product that has
those actions — leaving them in the platform's own defaults would mean
every episode and every future agent inherits one agent's vocabulary.

So the platform ships neutral, self-contained defaults (empty maps where
an agent must supply the content, which fail safe: an unmapped action is
human-controlled, an unmapped permission is denied), and each agent
overlays its own.

Merging is deep. An overlay that sets `gatekeeper.surface_policy.surfaces`
must not erase `gatekeeper.authority_hierarchy` alongside it, which is
exactly what a shallow merge would do.
"""

from pathlib import Path
from typing import Any

import yaml

from agentic_commons.config.paths import resolve_path

_CONFIG_DIR = Path(__file__).parent
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"
_LOCAL_FILE = _CONFIG_DIR / "local.yaml"

# Key in the platform config listing agent overlays, in merge order.
_OVERLAYS_KEY = "overlays"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, or {} if it doesn't exist."""
    if not path.exists():
        return {}
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` into `base`, recursing into nested dicts.

    Scalars and lists replace wholesale; only mappings merge. A list that
    merged element-wise could never be shortened by an override, which is
    the wrong behaviour for something like a phrase list.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _overlay_files() -> list[Path]:
    """Agent config files the platform config declares, in order."""
    declared = _read_yaml(_CONFIG_FILE).get(_OVERLAYS_KEY) or []
    return [resolve_path(str(entry)) for entry in declared]


def load_config() -> dict[str, Any]:
    """The fully merged configuration: platform, then agents, then local."""
    merged = _read_yaml(_CONFIG_FILE)
    for overlay in _overlay_files():
        merged = _deep_merge(merged, _read_yaml(overlay))
    return _deep_merge(merged, _read_yaml(_LOCAL_FILE))


def _walk(config: dict[str, Any], path: list[str]) -> dict[str, Any]:
    """Follow a dotted path (e.g. ["security", "pii"]) through nested dicts."""
    node: Any = config
    for part in path:
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def load_section(section: str) -> dict[str, Any]:
    """Load one section from the merged configuration.

    `section` may be a dotted path (e.g. "security.pii") to reach a
    nested section.
    """
    return _walk(load_config(), section.split("."))


def config_sources() -> list[Path]:
    """Every file that contributed, in merge order — for diagnostics."""
    return [_CONFIG_FILE, *_overlay_files(), _LOCAL_FILE]
