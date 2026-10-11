# memory/dream — how dream events become episodic, semantic and procedural memory (ADR 0012)

Long-term memory is only ever written by a **dream**. An agent's turns and runs leave
**dream events** in an inbox. A dream reads them later, over many events at once, and decides
what they amount to. One turn cannot tell a coincidence from a pattern; a dream can.

## 1. How a dream event is created

| Agent | When | By | What it carries |
|---|---|---|---|
| Customer Support | every completed turn, once memory is declared | `TurnRecorder.record` (in `graph/agent.py::_complete_turn`) | what was asked and answered (redacted), the tool actions and their results, scope `intent:<INTENT>` |
| `research_agent` | every turn (its agent's `handle`) | `TurnRecorder.record` | what was asked and answered |
| `research_agent` | every finished run | its `RunEventWriter` | signals `criterion_failed`, `source_rejected`, `budget_stop`, scoped to the run's kind, and the run's outcome |
| Ops | none (it declares no episodic memory) | — | — |
| A new agent | every turn, if it calls `TurnRecorder.record` when a turn completes | the platform | the default: asked, answered, actions |

Each event is one JSON file:

```
src/<agent>/data/memory/dream_events/
  pending/        <time>-<event id>.json   written by the turn, waiting for a dream
  consolidated/   <time>-<event id>.json   moved here, stamped, once a dream read it
```

- Consolidated events are pruned after `memory.dream.retention_days`.
- A pending event is never pruned.
- See them in `make show-memory <agent>`: the **Dream events** table, last.

## 2. What a dream makes of them

Run it with `make dream <agent>` (`SHOW=1` also prints the memory tables), or `/dream` in the
console. The agent's `MemoryProfile` answers three questions, one per store.

| Store | Default (any agent) | What lands where |
|---|---|---|
| **Episodic**: what happened | a session summary from the exact tool outcomes (`SessionSummarizer`) | `data/memory/episodic/<session>.jsonl`. The summary supersedes the session's turn records. |
| **Semantic**: what is true | claims the deterministic patterns recognise ("my delivery window is mornings") | `data/memory/semantic/<subject>/<key>.md`, written through the governed pipeline. A doubtful claim waits in `/queue`. |
| **Procedural**: how to do things | a signal seen in `memory.dream.skill_threshold` distinct sessions or runs becomes a learned SKILL.md candidate (`RecurringSignalSkills`) | always waits in `/queue` as a `skill` item. When approved, it is written to `data/memory/procedural/<name>/SKILL.md` and selected like any skill. |

### What each agent customises

| Memory | `customer_support_agent` | `research_agent` | `ops_agent` |
|---|---|---|---|
| Episodic | summary + a checked "what the customer wanted" sentence (BART, kept only if NLI entails it) | a run summary from the run's record | — |
| Semantic | default, per customer | default + `SourceReliability` (domains runs could not use) | — |
| Procedural | learned from repeated refusals (`memory.dream.refusal_markers`) | learned from failures recurring across runs of one kind | authored only |

## 3. Worked examples (no model call needed for the dream itself)

**Episodic.**
1. `make agent csa`, verify a customer, then ask `refund order <id>`.
2. Quit, then run `make show-memory csa`: the turn is a **pending** row in *Dream events*.
3. `make dream csa` writes a summary. `make show-memory csa` now shows it under *Episodic · customer N*, and the event is **consolidated**.

**Semantic.**
1. In a verified session, say `my delivery window is mornings`.
2. Run `make dream csa`, then `make show-memory csa`: the fact appears under *Semantic*.
3. If governance held it, it appears in `make queue csa` as a `memory_fact` item instead. Approve it with a comment: `make queue csa ACTION=approve ID=<id> COMMENT="customer confirmed"`.

**Procedural.**
1. Get the same tool refusal in `skill_threshold` (default 3) different sessions. For example, ask to refund an order that is not delivered yet, in three sessions.
2. `make dream csa` proposes a `skill` item, visible in `make queue csa`.
3. Approve it with a comment, in the terminal or with `ctrl+r` in the console.
4. From the next refund turn on, the learned skill is selected and shown on the SKILLS card.

For `research_agent` the same steps apply to its runs: `make dream ra` after runs of one kind that failed the same criterion. Its own note: `src/research_agent/design_docs/adr/ra-0023-run-memory-through-dreams.md`.

## Commands

| Where | Command |
|---|---|
| terminal | `make dream <agent>` · `make show-memory <agent> [SUBJECT=<id>] [EVENTS=<n>]` · `make queue <agent>` · `make clean-memory <agent>` · `make memory-reindex <agent>` |
| console | `/dream` · `/memory` (a verified customer: working + their episodes) · `/queue` · `/review` or `ctrl+r` |
