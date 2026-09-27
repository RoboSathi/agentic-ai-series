# tui — the operator console

One Textual app: a live feed of every harness stage, an input line for
talking to the agent or typing slash commands, a status line and a token
counter. It is a consumer of `observability` and the stores — never a
second source of truth — and file logging runs independently of it.

## Vocabulary

- **host** — the agent the console runs, resolved from `agent.host` in config (`AgentHost`)
- **feed** — the stream of `StatusEvent`s rendered as collapsible stage cards

## Public interface

- `AgenticTuiApp`, `main` (`app.py`; `python -m agentic_commons.tui` / `make tui`)
- Widgets: `LiveStatusPanel`, `StageEventWidget`, `StatusLineWidget`, `TokenCounterWidget` (`widgets.py`)
- `CommandView` (`command_view.py`), `CommandSuggester` (`suggest.py`), `StageLabelRegistry` (`labels.py`), clipboard (`clipboard.py`)

## Configuration

Section `tui:` — `refresh_interval_ms`, `show_token_count`, `copy_method`, `max_feed_lines`,
`spinner_frames`, `command_output` styles.

## Seams

- The agent comes from the host (or an injected `agent_factory` in tests); agent commands come from `AgentHost.commands()`.
- Live-state commands are rebound after every `/harness` rebuild.

## Collaborators

- `commands`, `observability`, `config`, `durability` (start-up notice of runs waiting on a reviewer).

## Episodes

Every episode is watchable here.

## Tests

Run with `./venv/bin/pytest tests/unit/test_tui.py tests/unit/test_commands.py -q` (or `make test` for everything).
