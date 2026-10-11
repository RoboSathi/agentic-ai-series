# memory/dream — how dream events become episodic, semantic and procedural memory (ADR 0012)

Long-term memory is only ever written by a **dream**. An agent's turns and runs leave
**dream events** in an inbox. A dream reads them later, over many events at once, and decides
what they amount to. One turn cannot tell a coincidence from a pattern; a dream can.

## 1. How a dream event is created

| Source | When | By | What it carries |
|---|---|---|---|
| a turn | every completed turn, once the agent declares memory | `TurnRecorder.record`, called by the agent when a turn completes | what was asked and answered (redacted), the tool actions and their results, the turn's scope (e.g. `intent:<name>`) |
| a longer unit of work | when the agent finishes one (a run, a job) | the agent's own writer, through any `DreamEventSink` | typed signals (`kind`, `value`) scoped to the situation, and an `outcome` the agent's consolidators read |
| nothing | an agent that declares no episodic memory writes none | — | — |

Which agent writes which, and what it customises, is in each agent's own design docs and in
ADR 0012 (`design_docs/adr/0012-file-memory-dreams-one-review-queue.md`).

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

## 3. Worked examples (no model call needed for the dream itself)

Shown for an agent with customers, refunds and a refusal wording (`memory.dream.refusal_markers`);
any agent follows the same steps with its own vocabulary.

**Episodic.**
1. `make agent <agent>`, verify a customer, then ask `refund order <id>`.
2. Quit, then run `make show-memory <agent>`: the turn is a **pending** row in *Dream events*.
3. `make dream <agent>` writes a summary. `make show-memory <agent>` now shows it under *Episodic · customer N*, and the event is **consolidated**.

**Semantic.**
1. In a verified session, say `my delivery window is mornings`.
2. Run `make dream <agent>`, then `make show-memory <agent>`: the fact appears under *Semantic*.
3. If governance held it, it appears in `make queue <agent>` as a `memory_fact` item instead. Approve it with a comment: `make queue <agent> ACTION=approve ID=<id> COMMENT="customer confirmed"`.

**Procedural.**
1. Get the same tool refusal in `skill_threshold` (default 3) different sessions. For example, ask to refund an order that is not delivered yet, in three sessions.
2. `make dream <agent>` proposes a `skill` item, visible in `make queue <agent>`.
3. Approve it with a comment, in the terminal or with `ctrl+r` in the console.
4. From the next refund turn on, the learned skill is selected and shown on the SKILLS card.


## Commands

| Where | Command |
|---|---|
| terminal | `make dream <agent>` · `make show-memory <agent> [SUBJECT=<id>] [EVENTS=<n>]` · `make queue <agent>` · `make clean-memory <agent>` · `make memory-reindex <agent>` |
| console | `/dream` · `/memory` (a verified customer: working + their episodes) · `/queue` · `/review` or `ctrl+r` |
