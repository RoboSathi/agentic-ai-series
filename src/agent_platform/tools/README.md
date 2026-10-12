# tools — typed tools and the single tool path

A namespaced registry of typed tools, retrieval-based selection, model-facing names, and `ToolPath`: the one route every model-proposed call takes (registry → loop governor → Gatekeeper → Executor), recording the stage it reached.

- **Entry points:** `NamespacedToolRegistry`, `ToolDefinition`, `ToolRetrievalSelector`, `ToolPath` (`path.py`), `names.py`
- **Configuration:** `tools:`
- **Design:** [LLD §A7](../../../design_docs/LLD.md#a7-tools) · [ADR P-11](../../../design_docs/ADR.md#p-11--one-tool-path)
