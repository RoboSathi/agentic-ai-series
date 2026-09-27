# config — the single source of every value

Every value in the repo comes from configuration, never from a Python
default. This module loads the layered YAML, validates it into typed
settings, and owns the feature tree that decides which harness controls an
agent is given.

## Vocabulary

- **layer** — one YAML file in the merge: platform `config.yaml` → agent overlays (`AGENTIC_OVERLAYS`) → run overlays (`use_overlay`) → `local.yaml`
- **section** — one top-level key, validated by one `YamlBackedSettings` class, with env prefix `SECTION__`
- **feature** — a control you can remove and still have a running agent (`features:`); nine exist, sub-features are parent-gated
- **FeatureSet** — the immutable feature tree; read once, in the composition root

## Public interface

- `get_settings()` / `reload_settings()` — the `AppConfig` singleton
- `YamlBackedSettings` — base class: YAML section, deep-merged with env and kwargs
- `use_overlay(path)` / `clear_overlays()` / `config_sources()` — the overlay chain
- `FeatureSet`, `active_features`, `set_active_features` (`features.py`)
- `resolve_path` / `REPO_ROOT` — paths resolve against the repo, never the shell's CWD
- `VersionManifest` (`versions.py`) — the independently versioned components (E39)

## Configuration

`config.yaml` here is the platform default. Env overrides use `SECTION__FIELD`, nested with
`__`, and are **deep-merged**: `RESILIENCE__RETRY__MAX_ATTEMPTS=7` changes one field and keeps its
siblings. Bare provider keys (`nvidia_api_key`, `hf_api_key`) are plain env vars and never appear
in a settings dump.

## Seams

- An agent names itself only in its own overlay; `overlays:` here is empty on purpose.
- `/harness` moves the live `FeatureSet` without touching disk; `/config` is read-only.

## Collaborators

- Everything reads it. `tests/unit/test_config_surface.py` keeps the surface honest: no orphan YAML keys, no unread fields, no overlay restating an inherited value.

## Episodes

E01 (the harness as a set of features), E22, E28, E37, E39, E48c.

## Tests

Run with `./venv/bin/pytest tests/unit/test_config.py tests/unit/test_features.py tests/unit/test_config_surface.py -q` (or `make test` for everything).
