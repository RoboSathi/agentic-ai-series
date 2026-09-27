# tools — typed tools and retrieval-based selection

Typed tool definitions in a namespaced registry, and a selector that
offers a turn only the tools relevant to it — none at all below a relevance
floor, so a chatty message is never handed something that moves money.

## Vocabulary

- **namespace** — an agent's tool prefix (`support.process_refund`)
- **surface class** — how a tool's effect may be exposed: locked, editable, append-only, human-controlled
- **selection floor** — the relevance below which no tools are offered

## Public interface

- `ToolRegistry` / `NamespacedToolRegistry` (`registry.py`)
- `ToolRetrievalSelector`, `SelectionMetrics` (`selector.py`)
- `ToolDefinition`, `ToolCall`, `ToolResult`, `ToolHandler`, `ToolSource` (`types.py`)

## Configuration

Section `tools:` — `enabled` (retrieval-based selection; off offers the whole registry),
`selection_k`, `selection_min_score`.

## Seams

- An agent registers its tools and handler.

## Collaborators

- Proposals go to `gatekeeper`; approved calls to `executor`; ranking uses the `evals` cascade.

## Episodes

E02, E05, E20, E22, E33, E48a, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_tools.py -q` (or `make test` for everything).
