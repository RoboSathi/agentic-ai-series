"""The config surface stays honest: no orphan keys, no unread fields, no restated values.

Three ways configuration rots, each found in this repo before this test
existed: a YAML key no settings model declares (silently ignored), a
settings field no code reads (a switch wired to nothing — `resilience.
enabled: false` once "turned off" resilience and changed nothing), and an
overlay restating the value it inherits (noise that hides the one line
that matters).
"""

import typing
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from agentic_commons.config.settings import AppConfig
from agentic_commons.config.yaml_config import deep_merge
from customer_support_agent.config.settings import CustomerSupportAgentSettings

ROOT = Path(__file__).resolve().parents[2]
PLATFORM = ROOT / "src" / "agentic_commons" / "config" / "config.yaml"
AGENT = ROOT / "src" / "customer_support_agent" / "config" / "config.yaml"
EPISODES = sorted((ROOT / "episodes").glob("*/config.yaml"))
AGENT_SECTION = "customer_support_agent"
# Sections that are not settings models: the overlay list, the feature
# tree (a FeatureSet, any shape), and episode metadata (EpisodeSettings).
FREE_FORM = {"overlays", "features", "episode"}
SETTINGS_FILES = ("config/settings.py",)
CODE_ROOTS = ("src", "episodes", "design_patterns")

Tree = dict[str, "Tree | None"]


def _model(annotation: Any) -> type[BaseModel] | None:
    if typing.get_origin(annotation) is not None:
        for argument in typing.get_args(annotation):
            if (found := _model(argument)) is not None:
                return found
        return None
    return (
        annotation if isinstance(annotation, type) and issubclass(annotation, BaseModel) else None
    )


def _is_mapping(annotation: Any) -> bool:
    return typing.get_origin(annotation) is dict


def _tree(model: type[BaseModel]) -> Tree:
    tree: Tree = {}
    for name, field in model.model_fields.items():
        nested = _model(field.annotation)
        tree[name] = None if nested is None or _is_mapping(field.annotation) else _tree(nested)
    return tree


def _schema() -> Tree:
    tree: Tree = {}
    for name, field in AppConfig.model_fields.items():
        nested = _model(field.annotation)
        tree[name] = _tree(nested) if nested else None
    tree[AGENT_SECTION] = _tree(CustomerSupportAgentSettings)
    return tree


def _undeclared(data: Any, tree: Tree | None, prefix: str) -> list[str]:
    if not isinstance(data, dict) or tree is None:
        return []
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in tree:
            found.append(path)
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            found += _undeclared(item, tree[key], path)
    return found


def _flat(data: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(data, dict):
        return {prefix: data}
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat.update(_flat(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


@pytest.mark.parametrize(
    "path", [PLATFORM, AGENT, *EPISODES], ids=lambda p: str(p.relative_to(ROOT))
)
def test_every_yaml_key_is_a_declared_setting(path: Path) -> None:
    schema = _schema()
    data = {key: value for key, value in _load(path).items() if key not in FREE_FORM}
    orphans = [key for key in data if key not in schema]
    orphans += _undeclared(data, schema, "")
    assert orphans == [], f"{path.name}: keys no settings model reads: {orphans}"


@pytest.mark.parametrize("path", EPISODES, ids=lambda p: p.parent.name)
def test_an_episode_overlay_changes_what_it_states(path: Path) -> None:
    """An episode's features block is its declared posture and is kept whole;
    every other line must differ from what the episode would inherit."""
    inherited = _flat(deep_merge(_load(PLATFORM), _load(AGENT)))
    restated = [
        key
        for key, value in _flat(_load(path)).items()
        if not key.startswith(("features.", "episode."))
        and key in inherited
        and inherited[key] == value
    ]
    assert restated == [], f"{path.parent.name} restates inherited values: {restated}"
