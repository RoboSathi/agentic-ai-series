# ADR 0004 — Optional host hooks: greeting, side panels, cancel, forwarded turns

## Context

The shell (`agent_platform/tui/app.py`) knows an agent only through the
`AgentHost` protocol. The research agent needs four things csa never did:
its own greeting; a live side panel (the scoping checklist, then the run's
budget meters); a key to cancel an autonomous run; and slash commands
(`/done`, `/waive`, `/rubric`, `/approve`) that are *conversational acts*
— they must reach the agent's `handle()`, not be answered by the shell.

## Decision

Optional hooks, each its own `runtime_checkable` Protocol in `host/protocols.py`,
checked with `isinstance`:

- `GreetingHost.greeting()` — the TUI falls back to the old text.
- `PanelHost.side_panels(session_id)` — renderables refreshed on a timer
  (`tui.side_panel_refresh_ms`); the panel column is only built when a
  host offers it.
- `CancellableHost.cancel_key()` / `cancel()` — bound as a **priority**
  binding so it fires even while the input box has focus.
- `ConsoleEffect.agent_turn` — a command validates its own syntax, then
  asks the shell to send a line to `handle()` as a normal turn.

## Consequences

- csa implements none of the hooks and renders exactly as before.
- The eval `ScriptedUser` speaks through `handle()` too, so the path the
  console drives is the path the evals test.
- Cancel is watch-and-cancel only (ra ADR ra-0010): no pause, no steering.

## Alternatives rejected

- **Widen `AgentHost`.** Every existing host would have to implement
  hooks it does not need.
- **Let ra draw its own Textual app.** Two shells; the TUI must remain one
  consumer of observability, persistence and audit.
- **Answer `/done` in the shell.** The interview's termination rule lives
  in the agent (enforced in code, ra-0001); the shell cannot apply it.
