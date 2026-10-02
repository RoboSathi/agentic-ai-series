# logging — file logs and the traced call

Structured, per-subsystem logging to files (always) and the console
(optionally), plus the `@instrument` decorator that records what every
traced call was given and returned — in full, never truncated by default,
because the argument you need to read back is always the long one.

## Vocabulary

- **subsystem** — a logger namespace with its own file and colour (`llm`, `evals`, `gatekeeper`, …)
- **trace id** — the request id carried through every layer by a context variable

## Public interface

- `get_logger(subsystem)` → `StructuredLogger`/`RichLogger`, `log_directory()` (`core.py`)
- `@instrument(subsystem)`, `@logged` (`decorators.py`)
- `get_current_trace_id` / `set_current_trace_id` (`trace_context.py`)

## Configuration

Section `logging:` — `level`, `console_enabled`/`console_level`, `file_enabled`/`file_level`, `dir`,
`run_label`, `file_format`, `payload_max_chars` (0 = everything), `max_bytes`, `backup_count`.

Files are written to `<dir>/<agent id>/<subsystem>.log` (`log_directory()`), so each agent has its own
folder (`logs/<agent id>/`, one per agent); `run_label` adds a sub-folder for runs
that are not the live agent (an offline profile).

## Seams

- The TUI is a consumer; file logging is independent of whatever is on screen.

## Collaborators

- Used by every module.

## Episodes

E01, E36.

## Tests

Run with `./venv/bin/pytest tests/unit/test_tracing_and_working_memory.py -q` (or `make test` for everything).

## Status and limits

Always on.
