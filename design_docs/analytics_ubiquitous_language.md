# Analytics: ubiquitous language

The vocabulary every analytics table, metric, dashboard and MCP description uses
(ADR 0009). A term means the same thing in code, in SQL and on screen. Where a
term was previously used loosely, the old usage is named and retired.

## Who and where

**Agent.** A product built on `agent_platform`, identified by `agent.id` in its
overlay (`customer_support_agent`, `research_agent`, `ops_agent`, …). Every
fact carries `agent_id`. There is no version or fingerprint (ADR 0009 D5).

**Channel.** The surface a turn came in on, bound by that surface: `tui`,
`eval`, `script`, `web`, … Eval traffic is `eval` and is excluded from runtime
metrics by default.

**Session.** One conversation with one person on one channel (`session_id`).
A session holds one or more tasks.

## Units of work

**Task.** One thing the person wants done: "refund order 10441", "research the
EU AI Act's obligations for SMEs", "what was p95 latency this week?". A task
may take several turns: a clarification, an approval, a long research run.

- **Opened** by the agent, through the platform task API, when it recognises a
  new request.
- **Closed** by the agent with an **outcome** when the work ends, successfully
  or not.
- **Implicit task.** An agent that never opens one still gets a task per turn:
  the turn bracket opens an implicit task, and closes it with the turn's outcome.
  So every turn has a `task_id`, and the simplest agent needs no task code.
- A task left open when its session ends is **abandoned**. That is a
  platform-assigned close, never an agent outcome.

**Turn.** One message in, one reply out (or a resume of a paused run, which is a
turn of its own). Bracketed by the platform (`turn_trace`), never by the agent.
`turn_id` is the turn's trace id, the same id its log lines and LangSmith run
carry.

**Turn status** (platform-defined, so every agent's is the same):
- `completed` — a reply was produced;
- `crashed` — the turn raised;
- `cancelled` — the operator stopped it.

Availability is computed from turn status. A turn status is *not* an outcome.

**Step.** One logical model call inside a turn: the model reasons once, and may
ask for tools. Retries and backup models are attempts of the same step, not new
steps. Counted by the platform at the outermost model client, so every agent's
steps are counted the same way.

**Step cap.** The maximum steps a turn may take (the agent's loop budget). A turn
that ends because the cap was reached is a **loop abort**. It is reported by the
platform loop guard (`ToolControlLoop` / `BudgetGuard`), which every agent's loop
must use.

## What happened

**Outcome.** How a task ended, in the agent's own words (ADR 0009 D2). Each
agent declares its outcomes in config, each with two flags (D4):
- `success` — did the person get what they asked for;
- `terminal` — does the task end here (`needs_more_input` is not terminal).

Examples: CSA `refund_issued` (success, terminal), `refund_queued_for_review`
(success, terminal: the request is in a person's hands), `refused_out_of_scope`
(not success, terminal), `awaiting_order_number` (not terminal). An outcome the
agent did not declare is a start-up error, not a silent NULL.

Retired usage: "answered", "escalated" and "unanswered" as cross-agent
outcomes. They are CSA words.

**Resolution.** A terminal task with a `success` outcome.

**Model call** (attempt). One request to one model: the unit of latency, tokens
and failure. A step has one or more attempts (`attempt` 1, 2, …). An attempt
served by a model after the role's primary is a **backup** attempt.

**Tool proposal.** A tool call the model asked for. Its **stage** says how far it got:

| Stage | Meaning |
|---|---|
| `invalid` | Arguments failed the tool's schema. |
| `unknown` | No such tool. |
| `not_offered` | A real tool, but not offered to the model this turn. |
| `refused` | The Gatekeeper denied it, or escalated it to a person. |
| `duplicate` | Identical to a call already made this turn; suppressed. |
| `executed` | Ran through the Executor, and then **succeeded** or **failed** (error or timeout). |

**Tool-call validity** is the share of proposals that reached `executed`.

Retired usage: counting a suppressed duplicate as a successful tool call.

**Check.** One invocation of a guard or classifier on content:
- `pii`, `injection`, `jailbreak` and `moderation` (the screens on the person's message);
- `content_safety`;
- `grounding` (claims vs evidence);
- a model dependency such as `nli`, `embedding`, `rerank` or `intent_model`.

A check has a **backend** (`rule`, `local_model`, `hosted_model`, `llm`), the
**model** that answered, and a **verdict**:

| Verdict | Meaning |
|---|---|
| `pass` | Nothing found. |
| `flag` | Found something; the turn continued. |
| `block` | The turn was stopped. |
| `skipped` | The check did not run, e.g. a feature was off for this message kind. |
| `error` | The check failed. |
| `fail_open` | It failed and the turn continued unprotected. |

**Screen coverage** is whether a declared screen actually ran on a turn.

**Claim.** One factual statement in a reply that the grounding check judged
`supported`, `unsupported` or `undetermined` against the evidence the turn
gathered. **Groundedness** is supported ÷ checked claims.

**Decision.** One authority ruling by the Gatekeeper (`allow`, `deny`,
`escalate`) with its **standing** (VOID, RATIFIABLE, RATIFIED, VOIDABLE, VALID,
UNGOVERNED). Kept in the audit trail. Not a Tier-1 metric, because not every agent
has a Gatekeeper.

## Metric tiers

- **Tier 1 (Fleet).** Common to every agent, recorded only by platform choke
  points; NULL where the agent withholds the feature the metric needs.
- **Tier 2 (Agent).** An agent's own domain metrics, over tables it owns and registers.
- **Tier 3 (Quality).** Offline eval results: a common core every suite reports,
  plus the agent's own metrics.

**Fact.** One row at one grain (task event, turn, attempt, proposal, check),
appended once by its choke point and never updated. A task is two events
(`opened`, `closed`); the mart joins them, so a process that dies mid-task
leaves an open task, not a lost one.

**Mart.** The curated, documented views over the facts, exported as Parquet.
The only thing readers (MCP server, dashboards) see.
