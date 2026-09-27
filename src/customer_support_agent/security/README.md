# security — proving who the customer is

`CustomerAuthenticator`: verifies a customer by their user code before
anything acts on their behalf, so identity comes from authentication and
never from anything the model says.

## Vocabulary

- **user code** — a seven-character credential with at least one digit, from an unambiguous alphabet

## Public interface

- `CustomerAuthenticator`, `AuthResult` (`authentication.py`)

## Configuration

`customer_support_agent.user_code_length`, `user_code_alphabet`; this agent's `security.pii` and `security.prompt_injection` overlay.

## Collaborators

- The `authenticate` node; `domain.identity`.

## Episodes

E02, E32.

## Tests

Run with `./venv/bin/pytest tests/unit/test_authentication_and_scope.py -q` (or `make test` for everything).
