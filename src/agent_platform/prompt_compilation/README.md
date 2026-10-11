# prompt_compilation — compiled prompts (parked)

Interface-only. The idea (E21): declare a typed input→output signature and
let an optimizer compile the wording against a metric and examples, so a
prompt becomes a versioned artifact with an eval score.

## Vocabulary

- **Signature** — typed inputs and outputs of a compiled module
- **CompiledModule** — the compiled artifact; would register with procedural memory

## Public interface

- `Signature`, `Example`, `EvalMetric`, `Optimizer`, `CompiledModule` (contracts only)

## Configuration

No config surface until an implementation lands.

## Collaborators

- Would register with `memory.SkillStore` and the version registry (`config.versions`).

## Episodes

E21.

## Tests

Run with `./venv/bin/pytest tests/unit/test_platform_boundary.py -q` (or `make test` for everything).

## Status and limits

**Parked.** Interface only; nothing implements or calls it.
