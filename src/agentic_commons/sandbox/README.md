# sandbox — execution isolation (parked)

Interface-only. Contracts for isolating tool execution. Only the container
tier is planned; the other tiers of E34's ladder need a deployment target
this repo does not have.

## Vocabulary

- **IsolationTier** — process policy, OS sandbox, container, microVM, remote worker

## Public interface

- `ExecutionSandbox`, `IsolationTier`, `IsolationTierSelector`, `SandboxResult` (contracts only)

## Configuration

No config surface until an implementation lands.

## Collaborators

- Would wrap `executor` handlers.

## Episodes

E34 (deferred).

## Tests

Run with `./venv/bin/pytest tests/unit/test_platform_boundary.py -q` (or `make test` for everything).

## Status and limits

**Parked** by decision; the Docker sandbox phase is deferred.
