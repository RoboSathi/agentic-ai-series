# tui — the operator console

One Textual app: a live feed of every harness stage, a status line (the
live stage, above the chat bar as in Claude) and an input line for talking
to the agent or typing slash commands. It is a consumer of `observability` and the stores — never a
second source of truth — and file logging runs independently of it.

## Vocabulary

- **host** — the agent the console runs, resolved from `agent.host` in config (`AgentHost`)
- **feed** — the stream of `StatusEvent`s rendered as collapsible stage cards

## Public interface

- `AgenticTuiApp`, `main` (`app.py`; `python -m agentic_commons.tui` / `make tui`)
- Widgets: `LiveStatusPanel`, `StageEventWidget`, `StatusLineWidget`, `TokenCounterWidget` (`widgets.py`)
- `CommandView` (`command_view.py`), `CommandSuggester` (`suggest.py`), `StageLabelRegistry` (`labels.py`), clipboard (`clipboard.py`)

## What a card shows

A card exists for something a person acts on. A field is on it only if it
explains what happened or why, is not derivable from another field on the
same card, and is read by something. Two rules apply at the source
(`StageReporter`), so the card and the log file cannot drift:

- An event field whose value is `None`, `""`, `[]` or `{}` is dropped. `False` and `0` are answers and stay.
- A **quiet** event (`emit(..., quiet=True)`) is logged and counted in the turn summary but draws no card. Per-write and per-lookup bookkeeping is quiet: a working-memory fact set, a memory recall, the per-turn episodic line, the dream-inbox entry, a healthy context assembly.

The TURN SUMMARY is where quiet bookkeeping is read, one line each and only when there is something to say: `working memory` (what it holds now) and `added` (new or changed this turn), `recall` (`2 recalled · lexical · 0 embedding calls`), `dream inbox` (`1 turn queued`), `memory` (facts committed or refused), tools, tokens, duration.

| Card | Fields it keeps |
|---|---|
| LLM Request | `model`, `context_tokens`, `system_prompt` (which contains the retrieved context), `user_query`, `tools_offered` |
| LLM Response | tokens, latency, `tool_calls` or `response`, `finish_reason` when known |
| Tool Result | `tool`, `result`, `execution_status` (completed, failed, timed_out or replayed), `duration_ms`, the tool's own domain fields |
| Gatekeeper | `decision`, `action`, `reason_code`, `question`, `checks` (each with what it observed) |
| PII | `matches`, `categories` |
| Hallucination check | one line per claim: what decided it, and `(Entailment 0.99)` only where a model gave a probability. Detail is in `guardrails.log` |
| Context | none when healthy; a warning card with the findings when a health check finds something |

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
