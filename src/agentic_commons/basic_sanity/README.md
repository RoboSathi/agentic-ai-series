# basic_sanity — pre-LLM checks, as a seam

The contract for deterministic checks that run *before* a model is asked
anything: does the thing the request names exist, is it the caller's, is
the request even possible? The platform owns only the contract — a request
in, findings out. What "sane" means is business logic, so every concrete
checker belongs to an agent.

## Vocabulary

- **BasicSanityRequest** — what is checked; carries `subject_id`, an opaque string, so no agent is forced to have customers
- **BasicSanityFinding** — one check's result: applicable, passed, reason, evidence
- **BasicSanityResult** — all findings plus an overall verdict and reason code

## Public interface

- `BasicSanityChecker` (Protocol) — `check(request) -> BasicSanityResult`
- `BasicSanityRequest`, `BasicSanityFinding`, `BasicSanityResult` (`types.py`)

## Configuration

No section of its own. Whether the stage exists at all is the `features.basic_sanity`
toggle, read once in the composition root.

## Seams

- An agent supplies its checker through `build_agent(basic_sanity=...)`.

## Collaborators

- Called by the agent's screening node before intent routing and the model.

## Episodes

Introduced with the harness features (E01 onward).

## Tests

Run with `./venv/bin/pytest tests/unit/test_basic_sanity.py -q` (or `make test` for everything).
