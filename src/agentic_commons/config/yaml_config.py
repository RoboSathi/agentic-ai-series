"""YAML-backed configuration loading.

config.yaml ships in git with the real defaults for each section. local.yaml
sits alongside it, is gitignored, and lets a developer override values (e.g.
point at their own local LLM server) without touching the tracked file.
"""

from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).parent
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"
_LOCAL_FILE = _CONFIG_DIR / "local.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, or {} if it doesn't exist."""
    if not path.exists():
        return {}
    with path.open() as handle:
        return yaml.safe_load(handle) or {}


def _walk(config: dict[str, Any], path: list[str]) -> dict[str, Any]:
    """Follow a dotted path (e.g. ["security", "pii"]) through nested dicts."""
    node: Any = config
    for part in path:
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) else {}


def load_section(section: str) -> dict[str, Any]:
    """Load one section from config.yaml, overridden by local.yaml.

    `section` may be a dotted path (e.g. "security.pii") to reach a nested
    section.
    """
    path = section.split(".")
    base = _walk(_read_yaml(_CONFIG_FILE), path)
    override = _walk(_read_yaml(_LOCAL_FILE), path)
    return {**base, **override}
