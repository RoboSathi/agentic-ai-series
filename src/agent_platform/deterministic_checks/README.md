# deterministic_checks — pre-LLM checks, as a seam

The contract for deterministic checks that run *before* a model is asked
anything: does the thing the request names exist, is it the caller's, is
the request even possible? The platform owns only the contract — a request
in, findings out. What "sane" means is business logic, so every concrete
checker belongs to an agent.

## Vocabulary

- **DeterministicCheckRequest** — what is checked; carries `subject_id`, an opaque string, so no agent is forced to have customers
- **DeterministicCheckFinding** — one check's result: applicable, passed, reason, evidence
- **DeterministicCheckResult** — all findings plus an overall verdict and reason code

## Public interface

- `DeterministicChecker` (Protocol) — `check(request) -> DeterministicCheckResult`
- `DeterministicCheckRequest`, `DeterministicCheckFinding`, `DeterministicCheckResult` (`types.py`)

## Configuration

No section of its own. Whether the stage exists at all is the `features.deterministic_checks`
toggle, read once in the composition root.

## Seams

- An agent supplies its checker through `build_agent(deterministic_checks=...)`.

## Collaborators

- Called by the agent's screening node before intent routing and the model.

## Episodes

Introduced with the harness features (E01 onward).

## Tests

Run with `./venv/bin/pytest tests/unit/test_deterministic_checks.py -q` (or `make test` for everything).
