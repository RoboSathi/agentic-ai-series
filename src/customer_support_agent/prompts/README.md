# prompts — what the model is told

The system prompt for choosing a tool, the separate system prompt and
instruction for writing the customer-facing reply, and the refund policy
text (`customer_policy.md`). The prompt is not a control — the Gatekeeper
is — but a prompt that only describes the happy path invites the model to
find one.

## Vocabulary

- **policy text** — `customer_policy.md`, embedded in the tool-choosing prompt; quoting it to a customer is fine, repeating the instructions is a leak

## Public interface

- `build_system_prompt()`, `build_reply_system_prompt()`, `build_reply_instruction()` (`system_prompt.py`)
- `policy_text()` (`policy.py`)

## Configuration

`customer_support_agent.policy_path`.

## Collaborators

- Used by the `reason` and `finalize` nodes; the eval security suite's `no_prompt_leak` checks replies against these instructions.

## Tests

Run with `./venv/bin/pytest tests/unit/test_customer_support_agent.py -q` (or `make test` for everything).
