# audit — the append-only decision trail

Records every authorization decision the harness makes — who asked, what
they asked for, under which authority and policy version, and the verdict
(`allow`, `deny`, `escalate`) with its rationale. The trail is append-only:
it is evidence, so it is never edited or summarised in place. It does not
decide anything; the Gatekeeper decides and this module remembers.

## Vocabulary

- **AuditEntry** — one decision: trace id, actor, action, authority context, model/policy/data versions, decision, rationale
- **Decision** — `allow` | `deny` | `escalate`
- **reconstruct** — replaying a trace's entries in order to explain what happened

## Public interface

- `AuditLedger` (Protocol) — `record`, `count`, `by_decision`, `reconstruct(trace_id)`
- `SqliteAuditLedger(pool)` — the agent-database ledger; `NullAuditLedger` when audit is off
- `create_audit_ledger(store=..., agent_id=...)` — factory; writes to the shared DuckDB analytics store when one is given
- `AuditEntry`, `Decision` — value types (`types.py`)

## Configuration

Section `audit:` — `enabled`, and the four version stamps written on every entry
(`policy_version`, `data_version`, `prompt_version`, `tool_schema_version`).
An agent overlay names its own versions. Env override: `AUDIT__ENABLED=false`.

## Seams

- The composition root injects a ledger (`build_agent(audit=...)`); the eval harness injects a per-case `SqliteAuditLedger` so each golden case has its own trail.

## Collaborators

- Written by `gatekeeper` and `executor`; read by `commands` (`/audit`) and by `observability` analytics.

## Episodes

E02, E08, E12, E33, E39, E48a, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_audit_persistence_llm.py tests/unit/test_gatekeeper.py tests/integration/test_request_lifecycle.py -q` (or `make test` for everything).
