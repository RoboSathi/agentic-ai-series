# context — what the model is handed

Five-stage context assembly (gather → score → fit → format → order), six strategies, lexical / dense / hybrid / graph retrieval with a local reranker, trust classes, and context-failure detection.

- **Entry points:** `FiveStagePipeline`, `ContextAssembler`, `StrategySelector`, retrievers (`retrieval.py`), `ContextFailureDetector` (`failure/`)
- **Configuration:** `context:`
- **Design:** [LLD §A15](../../../design_docs/LLD.md#a15-context)
