"""YAML-backed configuration loading.

Configuration is layered, and the layers answer different questions:

  `agent_platform/config/config.yaml`   what is true for *any* agent
  each overlay it declares               what is true for *this* agent
  any overlay a caller adds              what is true for *this run*
  `local.yaml` (gitignored)              what is true on *this machine*

The split matters because the platform is shared. Action names, role
tables, refund thresholds and tool wording belong to the product that has
those actions — leaving them in the platform's own defaults would mean
every episode and every future agent inherits one agent's vocabulary.

So the platform ships neutral, self-contained defaults (empty maps where
an agent must supply the content, which fail safe: an unmapped action is
human-controlled, an unmapped permission is denied), and each agent
overlays its own.

The run-scoped layer is what lets one episode run the agent with only the
subsystems that episode is about. An episode does not get a different
agent; it gets the same agent with a narrower configuration, which is the
whole claim the series makes about harnesses. It sits *below* local.yaml
deliberately: a machine-local override is the last word, or it is not an
override.

Merging is deep. An overlay that sets `gatekeeper.surface_policy.surfaces`
must not erase `gatekeeper.authority_hierarchy` alongside it, which is
exactly what a shallow merge would do.
"""

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from agent_platform.config.environment import load_environment
from agent_platform.config.paths import resolve_path

_CONFIG_DIR = Path(__file__).parent
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"
_LOCAL_FILE = _CONFIG_DIR / "local.yaml"

# Key in the platform config listing agent overlays, in merge order.
_OVERLAYS_KEY = "overlays"

# Where the agent layer comes from when the platform does not declare one.
#
# The platform's own `overlays:` is empty and stays empty: a default that
# listed one agent would make that agent's vocabulary every agent's, and
# would mean the platform could not be read at all without that agent
# installed. So the agent layer is named from outside the platform — by
# whoever decided which agent this process is running.
_OVERLAYS_ENV = "AGENTIC_OVERLAYS"
_OVERLAYS_ENV_SEPARATOR = os.pathsep

# Overlays a caller added for this process, in the order they were added.
# Module state rather than a parameter because the readers are the settings
# models, which are constructed by pydantic deep inside `AppConfig()` and
# have nowhere to thread an argument through from.
_run_overlays: list[Path] = []


def use_overlay(path: str | Path) -> None:
    """Add a run-scoped config layer, applied after the agent overlays.

    Adding the same path twice is the same as adding it once, so a caller
    does not have to remember whether it already did.

    Settings are cached, so anything already holding a settings object
    will not see this. Call `reload_settings()` afterwards, before
    anything else reads config.
    """
    resolved = resolve_path(path)
    if resolved not in _run_overlays:
        _run_overlays.append(resolved)


def clear_overlays() -> None:
    """Drop every run-scoped layer, back to platform + agents + local."""
    _run_overlays.clear()


# libyaml's parser when it is installed (an order of magnitude faster), else the pure-Python one.
_BaseLoader: type[yaml.SafeLoader] = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class _StrictLoader(_BaseLoader):  # type: ignore[valid-type,misc]
    """A loader that refuses to silently drop a repeated key.

    Plain YAML keeps the last of two identical keys and says nothing. In a
    layered configuration that is a very quiet way to lose a section: a
    second `memory:` block further down the file silently deletes the
    first one, and the setting it held reverts to the platform default
    with no error anywhere. Caught exactly that way once, which is why
    this exists.
    """


# The tag YAML gives a merge key (`<<`).
_MERGE_TAG = "tag:yaml.org,2002:merge"


def _no_duplicate_keys(loader: Any, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    """Construct a mapping, raising if any key appears twice.

    YAML merge keys (`<<: *anchor`) are honoured: the anchor's keys come first and a key written
    beside the merge overrides them — that is what a merge is for, not a duplicate. Two keys
    written out in the same mapping are still an error.
    """
    mapping: dict[Any, Any] = {}
    explicit: list[tuple[yaml.Node, yaml.Node]] = []
    for key_node, value_node in node.value:
        if key_node.tag == _MERGE_TAG:
            mapping.update(_merged(loader, value_node))
        else:
            explicit.append((key_node, value_node))
    written: set[Any] = set()
    for key_node, value_node in explicit:
        key = loader.construct_object(key_node, deep=deep)
        if key in written:
            raise ValueError(
                f"Duplicate configuration key '{key}' at line {key_node.start_mark.line + 1}. "
                "The later block would silently replace the earlier one."
            )
        written.add(key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _merged(loader: Any, node: yaml.Node) -> dict[Any, Any]:
    """What a merge key brings in: one anchored mapping, or a list of them (first wins)."""
    sources = node.value if isinstance(node, yaml.SequenceNode) else [node]
    merged: dict[Any, Any] = {}
    for source in reversed(sources):
        merged.update(loader.construct_object(source, deep=True))
    return merged


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


# Each file parsed once per version of it: every settings section reads the merged
# configuration, and re-parsing every layer for each of them made start-up take seconds.
_parsed: dict[Path, tuple[tuple[int, int], dict[str, Any]]] = {}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file into a dict, or {} if it doesn't exist (a private copy each call)."""
    if not path.exists():
        return {}
    stat = path.stat()
    version = (stat.st_mtime_ns, stat.st_size)
    cached = _parsed.get(path)
    if cached is None or cached[0] != version:
        cached = (version, _parse(path))
        _parsed[path] = cached
    return copy.deepcopy(cached[1])


def _parse(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        try:
            loaded = yaml.load(handle, Loader=_StrictLoader)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from exc
    return loaded if isinstance(loaded, dict) else {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` into `base`, recursing into nested dicts.

    Scalars and lists replace wholesale; only mappings merge. A list that
    merged element-wise could never be shortened by an override, which is
    the wrong behaviour for something like a phrase list.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _overlay_files() -> list[Path]:
    """The agent config layer, in merge order.

    `AGENTIC_OVERLAYS` wins when set, because naming the agent is a
    deployment decision and the environment is where deployment
    decisions live. The platform's `overlays:` key remains as the
    fallback so a self-contained distribution can still declare its own
    layering — it is simply empty here.
    """
    declared = os.environ.get(_OVERLAYS_ENV)
    entries = (
        [part for part in declared.split(_OVERLAYS_ENV_SEPARATOR) if part.strip()]
        if declared
        else (_read_yaml(_CONFIG_FILE).get(_OVERLAYS_KEY) or [])
    )
    return [resolve_path(str(entry)) for entry in entries]


def load_config() -> dict[str, Any]:
    """The fully merged configuration: platform, agents, run, then local."""
    # The overlay chain is named by `AGENTIC_OVERLAYS`, which `.env` may set.
    load_environment()
    merged = _read_yaml(_CONFIG_FILE)
    for overlay in (*_overlay_files(), *_run_overlays):
        merged = deep_merge(merged, _read_yaml(overlay))
    return deep_merge(merged, _read_yaml(_LOCAL_FILE))


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
    return [_CONFIG_FILE, *_overlay_files(), *_run_overlays, _LOCAL_FILE]
