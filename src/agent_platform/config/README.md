# config — the single source of every value

Loads the layered YAML (platform defaults → agent overlays → profiles → `local.yaml` → `SECTION__FIELD` env vars), validates it into typed settings, and owns the feature tree an agent's composition root reads once.

- **Entry points:** `get_settings()`, `YamlBackedSettings`, `use_overlay`, `FeatureSet` (`features.py`), `resolve_path` / `REPO_ROOT` (`paths.py`), `load_environment`
- **Configuration:** `config.yaml` here (platform defaults); `local.yaml.example`
- **Design:** [LLD §A1](../../../design_docs/LLD.md#a1-config) · [ADR P-07](../../../design_docs/ADR.md#p-07--the-harness-is-the-set-of-declared-features)
