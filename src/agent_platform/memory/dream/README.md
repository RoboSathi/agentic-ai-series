# memory/dream — consolidating dream events into long-term memory

A completed turn (or run) leaves a dream event in `data/<package>/memory/dream_events/pending/`; `make dream <agent>` or `/dream` reads them in bulk and the agent's `MemoryProfile` decides what they amount to — a summary, facts, a learned skill a curator must approve.

- **Entry points:** `DreamEvent`, `FileDreamInbox`, `TurnRecorder`, `Dream`, `RecurringSignalSkills`
- **Configuration:** `memory.dream`
- **Design:** [LLD §A16](../../../../design_docs/LLD.md#a16-memory) · [ADR P-13](../../../../design_docs/ADR.md#p-13--long-term-memory-is-files-only-a-dream-writes-it)
