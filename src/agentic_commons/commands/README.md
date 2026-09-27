# commands — the operator's slash commands

What an operator types in the console to ask the harness about itself.
Commands return data (`CommandResult`), never pixels, and never reach a
model: inspecting the system must not spend the customer's tokens, and a
model-answered command could be wrong about what the system contains.

The console ships nine commands: `/help`, `/clear` (Console); `/config`,
`/audit`, `/memory` (Inspect); `/harness` (Control); `/eval` (Evals); and the
agent's own, which arrive from its host (listed under Agent).

## Vocabulary

- **Command** — anything with `name`, `summary`, `usage`, optional `group`, and `async run(args)`
- **ConsoleEffect** — something only the console can do on a command's behalf (clear the feed, rebuild the agent)
- **group** — the heading `/help` lists a command under

## Public interface

- `CommandRegistry` — `register`, `dispatch(line)`, `completions`, `argument_completions`; the only place a slash is parsed
- `Command` (Protocol), `CommandResult`, `CommandRow`, `CommandError`, `ConsoleEffect` (`types.py`)
- Platform commands: `HelpCommand`, `ClearCommand` (`system.py`), `ConfigCommand`, `AuditCommand` (`inspection.py`), `MemoryCommand` (`data.py`), `HarnessCommand` (`harness.py`), `EvalCommand` (`evals.py`), `KnowledgeBaseCommand` (`knowledge.py`)

## Configuration

Section `commands:` — `default_limit` (rows a listing shows), `value_preview_chars`,
`recent_window`. Env: `COMMANDS__DEFAULT_LIMIT=30`.

## Seams

- A platform command is listed in `registry._default_commands()`.
- An agent's own commands arrive from its `AgentHost.commands()` — the platform never imports them.
- `PoolBackedCommand` (`_pool.py`) lets a test point a command at a fixture database.

## Collaborators

- Rendered by `tui`; `/eval` reads `evals.store`; `/harness` moves `config.FeatureSet` and asks the console to rebuild the agent.

## Episodes

E01 (`/harness`), E09–E11 (`/eval`).

## Tests

Run with `./venv/bin/pytest tests/unit/test_commands.py tests/unit/test_eval_adapters.py tests/unit/test_tui.py -q` (or `make test` for everything).

## Status and limits

`tests/unit/test_commands.py::TestConsoleSurface` pins the registered set, so a removed command cannot silently return.
