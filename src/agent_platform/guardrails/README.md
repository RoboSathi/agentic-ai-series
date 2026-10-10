# guardrails — structured-output guarantees

Makes model output conform before it is used: validate against a schema,
repair or reject, or constrain generation where the provider supports it.
`ResponseGuardrail` also runs the customer-facing reply through the
deterministic evidence checks and the hallucination guard, with a bounded
repair loop that fails closed to a human.

## Vocabulary

- **mode** — `reject`, `auto_repair` or `constrained`
- **fail closed** — a reply that cannot be made safe is withheld and escalated, never sent

## Public interface

- `PydanticValidator`, `LlmConstrainedGenerator`, `fail_loud` (`guards.py`)
- `ResponseGuardrail`, `ResponseGuardrailResult`, `CustomerResponse` (`response.py`)
- `OutputValidator`, `ConstrainedGenerator` (Protocols), `ValidationResult`, `GuardrailViolationError` (`types.py`)

## Configuration

Section `guardrails:` — `enabled` (schema validation), `mode`, `max_repair_attempts`,
`constrained_backend`. The grounding half is gated by `features.hallucination_guard`.

## Seams

- The grounding ladder is injected (`ResponseGuardrail(grounding=...)`).

## Collaborators

- Uses `hallucination_check`; called by the agent's `finalize` node.

## Episodes

E24, E28, E29.

## Tests

Run with `./venv/bin/pytest tests/unit/test_guardrails.py tests/test_hallucination_guardrails.py -q` (or `make test` for everything).
