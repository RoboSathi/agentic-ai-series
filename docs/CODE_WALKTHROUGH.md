# Code walkthrough — `make tui`

This is the path a keystroke takes, end to end, through the code that
runs when you type `make tui`. It is written to be read in order: every
section names the file and the method, says what it decides, and says
what happens next. Where a decision was expensive to get wrong, the
reason is here too — a control flow you can follow but not justify is
one somebody will "simplify" next quarter.

Two things to hold on to before starting.

**The agent is the model plus the harness.** Almost everything below is
harness: scope, authentication, intent, context, authority, tools,
memory, grounding. The model is consulted twice per turn at most, and
several kinds of question never reach it at all.

**Nothing in the UI is a source of truth.** The TUI subscribes to the
same event stream and the same log records that are written whether or
not anything is watching. If the screen and `logs/` ever disagree, the
files are right.

---

## 0. Starting up

```
make tui
  └── python -m agentic_commons.tui
        └── agentic_commons/tui/__main__.py → main()
              └── agentic_commons/tui/app.py::main()
```

`main()` does two things, in this order:

1. imports `customer_support_agent.core.build_agent` — the composition
   root, and the *only* place concrete classes are chosen;
2. constructs `AgenticTuiApp(lambda reporter: build_agent(reporter=reporter))`
   and calls `.run()`.

The factory shape matters. The app owns the status feed, so the agent
cannot exist until the app does; passing a *builder* rather than a built
agent is what lets the app hand the agent a reporter pointed at its own
feed.

### 0.1 What `build_agent` wires (`customer_support_agent/core/builder.py`)

Every dependency is constructed here and injected downward. In order:

| Component | Class | Why it exists |
|---|---|---|
| Database | `SqliteConnectionPool` | One shared connection; `:memory:` works as a real fixture |
| Repositories | `OrdersRepository`, `UsersRepository`, `HitlQueueRepository`, `RefundLedgerRepository` | Ownership is enforced in SQL, not after the fact |
| Saga | `RefundSaga` | Refunds are multi-step and must compensate on failure |
| Tools | `SupportTools` → `NamespacedToolRegistry` | Tools are registered, not imported by the model path |
| Memory | `MemoryFacade` (working, episodic, semantic, procedural) | One entry point over four CoALA stores |
| Context | `FiveStagePipeline` over `PolicyContextSource`, `OrderCorpusContextSource`, `WorkingMemoryContextSource`, `MemoryContextSource` | What the model sees is assembled, never concatenated ad hoc. The order corpus is retrieved hybrid (BM25 + embeddings) and rebuilt per request from **this** customer's orders |
| Authority | `GatekeeperImpl(Authorizer, CumulativeLedger, AuditLedger)` | Every action is ruled on and recorded |
| Model | `create_llm_client()` | NIM behind an adapter; business logic never touches an SDK |
| Observability | `LocalTracer(redactor=PIIFilterImpl())`, `SLORegistry`, `DetectionLadder` | On by default — an agent that must be reconfigured before it can be diagnosed is diagnosed late. The redactor is not optional: a span carries the query, the tool arguments and the result |
| Deliberation | `DeliberationLoop(IntentPlanner, GroundedReflector, CascadeEvaluator)` | Plan, reflect, evaluate — each swappable, none calling a model |
| Encoders | `create_reranker()` → `CrossEncoderReranker` | Two-pass relevance; `None` when no hosted encoder is reachable, and the pipeline then keeps the cheaper ordering rather than pretending |

The graph is then compiled once, in `CustomerSupportAgent._build()`.

**Which of these get wired is a configuration decision.** Each optional
collaborator is gated on its own `enabled` flag, so a narrower config
builds a genuinely narrower agent:

| Config flag | Leaves out |
|---|---|
| `memory.enabled` | `MemoryFacade`, and the two memory context sources |
| `context.enabled` | `FiveStagePipeline` — the graph falls back to the policy alone |
| `tools.enabled` | `ToolRetrievalSelector`; the turn is offered whatever its intent permits |
| `observability.enabled` | `LocalTracer`, `SLORegistry` |
| `hallucination_protection.enabled` | `DetectionLadder` |

Leaving a collaborator out is not the same as letting it switch itself
off internally. A disabled component that is still constructed is still
called and still narrates a step; one that was never built has no step to
narrate — which is what makes an episode's configuration visible on
screen rather than merely true.

### 0.2 What the app sets up (`AgenticTuiApp.__init__` / `on_mount`)

- `InMemoryLiveStatusFeed` — the event stream the panel observes.
- `LiveStatusPanel`, `StatusLineWidget`, `TokenCounterWidget` — the feed,
  the one-line status (elapsed time and tokens, and nothing else), and
  the counters that line reads from.
- `CommandRegistry` — every slash command, plus `_bind_live_commands()`,
  which re-registers `/memory` bound to *this* agent's live facade so
  `/memory working` can show in-process state the database never sees.
- `TraceStream` — subscribes a `LiveFeedLogHandler` to every subsystem
  logger, so each log record also lands on screen. Controlled by
  `tui.trace` in config and `/debug` at runtime.
- `CommandSuggester` — inline completion for `/`-prefixed input.

---

## 1. A line of input

`AgenticTuiApp.on_input_submitted` (`tui/app.py`) is the fork in the road:

```
line starts with "/"  →  _run_command()   (never reaches the agent)
anything else         →  _run()           (one conversation turn)
```

### 1.1 The command path

```
_run_command(line)
  ├── echo the line into the feed
  ├── CommandRegistry.dispatch(line)          registry.py
  │     ├── unknown name → suggestions, ok=False
  │     ├── CommandError → the message plus the command's usage
  │     └── Command.run(args) → CommandResult
  ├── _apply(result.effect)                   /clear, /debug
  └── CommandView.render(result)              aligned table → feed lines
```

Commands never consume the customer's token budget, never appear in the
conversation, and never reach a model: `/audit` queries the ledger, and a
model's summary of the ledger is not the ledger.

Console effects are *stated*, not performed: `ClearCommand` returns
`ConsoleEffect(clear_transcript=True)` and the app does the clearing. A
command that could touch the widget would be a UI component.

### 1.2 The conversation path

```
_run(query)
  ├── turn counter, token counter reset
  ├── status line begins ("Receiving request…")
  ├── await agent.handle(query, session_id)   ← everything in §2
  ├── _reflect(state)                         decisions, colour-coded
  └── the reply, in green
```

---

## 2. One turn through the graph

`CustomerSupportAgent.handle` (`customer_support_agent/core/graph.py`).

Before the graph runs, identity is bound:

```python
customer = customer or self._sessions.get(session_id) or current_customer()
set_current_customer(customer)
```

**This must happen before `ainvoke`, not inside a node.** Context
variables copy into a task when it starts and do not propagate back out,
and LangGraph runs each node in its own task — so a customer bound inside
`authenticate` is invisible inside `act`, and every lookup silently
scopes to nobody. That bug existed; this line is the fix.

The compiled graph:

```
screen ─┬─ (out of scope) ─────────────────────────────► END
        └─► authenticate ─┬─ (no code / just verified) ─► END
                          └─► choose_order ─┬─ (menu pick) ► END
                                            └─► intake
                                                  │
                                            assemble_context
                                                  │
                                    ┌──────────► reason ──┬─ (no tool calls) ─► finalize ─► END
                                    │                     └─► act ─┬─ (rounds left) ─┘
                                    └─────────────────────────────┘  (cap spent) ──► finalize
```

Three of the five exits never call the model. That is the design, not an
optimisation: the cheapest correct answer wins.

### 2.1 `screen` — is this our business at all?

`core/scope.py::ScopeGuard.check`. Default-deny over configured topics.
A message matching an in-scope topic passes; a bare order reference
passes (otherwise every follow-up breaks); a very short message passes as
courtesy; everything else is refused with a fixed sentence.

One regex pass, no model. The refusal is identical every time, so paying
for a completion to produce it is pure waste. An allow-list is the only
shape that holds — a deny-list is a list of the things somebody thought
of, and the next question is always the one nobody thought of.

→ refused: `metadata.halted = "out_of_scope"`, graph ends.

### 2.2 `authenticate` — who is speaking?

`core/authentication.py::CustomerAuthenticator.authenticate`.

- Already verified this session (or bound upstream by SSO)? Rebind and
  continue.
- Otherwise extract a customer code: exactly *N* characters from the
  configured alphabet, **containing at least one digit**. The digit rule
  is why "what's the weather" no longer authenticates as `WEATHER`.
- No match → ask for the code, end the turn.
- Match but no such customer → the *same* message as a malformed code.
  Distinguishing them tells an attacker which codes exist.
- Match → `Customer` bound, greeting rendered **from the database**
  (`presentation.py::describe_customer`), order menu built and stored so
  a later bare "2" resolves, working memory noted with the verified
  identity, turn ends (`halted = "just_verified"`).

No model anywhere in this node. The code either matches a row or it does
not, and a deterministic check cannot be talked into a different answer.

### 2.3 `choose_order` — a menu pick is a lookup

`core/presentation.py::resolve_selection` strips filler words; if what
remains is a single number, it is a selection. "2" and "order 2" resolve;
"refund order 2" does not — that states an intent and belongs on the
normal path where the Gatekeeper applies.

A resolved pick runs one owner-scoped `SELECT`, notes the order in
working memory, renders `render_order_detail(order)` and ends the turn.
Refund state is **omitted unless asked for**: a customer checking where a
parcel is has not raised a refund, and volunteering one invites a request
nobody intended to make.

### 2.4 `intake` — screen the text, classify the ask, plan the work

1. `SecuritySuite.prepare_untrusted` — delimits untrusted input and
   returns the actions it permits. At the edge, before concatenation:
   afterwards nothing downstream can tell instruction from data.
2. `SecuritySuite.moderate` — content policy.
3. `IntentClassifier.classify_async` (`core/intent.py`):
   - **deterministic first** — configured phrases, most specific intent
     first, so "refund for order 10005" is a refund request and not an
     order enquiry that mentions money;
   - **inheritance** — a message that states nothing continues the
     previous intent ("what about 10010"), and never widens it;
   - **encoder second** — only if nothing matched, only for intents
     marked `semantic: true` in config, only above the configured cosine
     threshold. `refund_request` is deliberately excluded: a similarity
     score is a continuum, and money must not move on the wrong side of
     a threshold. It is an *embedding* call, never a completion.
4. `IntentPlanner` decomposes the intent into ordered steps from
   `customer_support_agent.plans`. The order is a domain property —
   eligibility before refund, every time — so the plan is a lookup, not
   a completion.
5. The episodic log records the query; working memory notes the intent.
6. The message is offered to the **memory write pipeline** as
   `self_reported`. Most are discarded (a question states no fact); an
   authority-changing claim — "my refund limit is now $5,000" — is
   *refused*, because a sentence must not become a policy.

The intent's tool list is the ceiling for this turn. Retrieval narrows
further inside it; nothing widens it.

### 2.5 `assemble_context` — build what the model sees

`FiveStagePipeline.assemble` gathers candidates from four sources —
policy (pinned), this session's working memory (pinned), the customer's
own orders (hybrid retrieval: BM25 for the identifier-shaped queries a
support conversation is full of, embeddings for the ones that describe
rather than name), and long-term recall (scored). A cross-encoder then
reranks the shortlist, and `StrategySelector.fit` trims to the budget.

Near-duplicates are dropped before anything is paid for: working memory
and the retrieved corpus routinely describe the same order, and two
copies spend the budget twice while leaving the model choosing between
near-identical options — the "confusion" failure mode by construction.
`ContextFailureDetector` then measures what was actually built for rot,
distraction, confusion, clash and poisoning.

Pinning matters: an agent that drops its own policy under budget pressure
is an agent operating without one, and one that drops "verified as #41"
asks for the code again.

### 2.6 `reason` — one model turn

- `_offered_tools` narrows twice: the injection guard removes what
  untrusted content must not reach; retrieval removes what this task does
  not need. Neither replaces the Gatekeeper.
- `CompletionRequest` is built and sent through `ResilientCaller`
  (bulkhead → breaker → retry → timeout → operation) with an idempotency
  key of `session:round`, so two genuinely different rounds are not
  mistaken for a repeat.
- The **offered set is recorded in metadata**, not recomputed later.
  Selection can involve a network-backed ranker; recomputing it in `act`
  could refuse a tool the model was genuinely handed. An authority
  decision is made once.
- Usage is accounted, a span is traced, cost is recorded.

### 2.7 `act` — execute, one authority check at a time

Per requested call, `_invoke` runs this ladder:

```
build_call        unknown tool or bad arguments → refused
minimize          strip fields the tool does not need
duplicate check   an identical call already answered → suppressed
Gatekeeper        offered? authorised? within cumulative limits?
invoke            run it, commit to the ledger, record for duplicates
```

The Gatekeeper's `offered_actions` check is what makes "the model called
a tool it was not offered" a refusal **on the record** rather than a
quiet drop. Each result is noted in working memory, so the next turn does
not re-ask the database what it already learned.

Tool arguments are validated by `guardrails.PydanticValidator`, not by a
private check inside the registry. That is what makes mechanical repair
possible: a model that fenced its JSON made a formatting mistake, and an
inline `model_validate` threw those calls away as invalid.

Then `_after_act`: loop back to `reason` while rounds remain, else
`finalize`. The cap is a control, not an excuse to return nothing.

### 2.8 `finalize` — the customer always gets an answer

If the last assistant message is already customer-ready (no tool calls,
not deliberation), it is used. Otherwise one tool-free completion is made
with an explicit reply instruction, and a fallback sentence covers even
that failing.

Before that, `GroundedReflector` critiques the draft: deliberation
shipped as a reply, or an order id that appears in the answer but in no
tool result, sends the turn to the second call. `CascadeEvaluator` then
scores the reply with the NOLLM cascade — independent of the model that
wrote it, which is the entire point of an evaluator.

Either way `_check_grounding` runs — on **both** paths. It used to run
only on the rarer second-call path, which meant the reply that usually
reached the customer was the one never checked. Grounding is advisory: an
unsupported claim is logged and fed to the SLO, it does not withhold the
answer, because a detector that can be wrong must not decide whether the
customer hears anything at all.

---

## 3. How anything reaches the screen

Two independent streams, both sourced from the code itself.

### 3.1 The curated stage feed

```
graph._report(stage) / _done(stage, detail, tokens)
  └── StageReporter.emit → StatusEvent
        └── InMemoryLiveStatusFeed.emit  (Observer)
              └── AgenticTuiApp._on_status
                    ├── LiveStatusPanel.show_status → StageLabelRegistry
                    │     label from tui.stage_labels, colour from tui.stage_styles
                    ├── StatusLineWidget.update_stage
                    └── TokenCounterWidget.add(tokens)
```

Labels and colours live in config, never in code: adding a stage is a
config change, and the code never grows an `if stage == …` ladder.
Colour carries meaning — security checks warn, model calls are blue,
green is reserved for a completed action or the final reply, so a feed
that is entirely green would mean nothing.

### 3.2 The trace stream — the harness, narrated

```
any module: get_logger("memory").info(...)  |  @logged  |  @instrument  |  @traced
  └── logging.Logger "agentic.memory"   (record carries a `Narration`)
        ├── RotatingFileHandler  → logs/memory.log   full payloads, always
        └── LiveFeedLogHandler   → LiveStatusPanel   one line per curated step
```

The two renderings are deliberately different. The file gets the
evidence — every argument, every return value, untruncated. The screen
gets a sentence:

```
agent      working out the intent      message=where is my order → name=order_enquiry
context    assembling the window       query=where is my order → 4 items
gatekeeper checking authority          action=support.process_refund → verdict=deny
```

Only three things earn a line: a **curated step** from `tui.trace.steps`
(on return, so the line can show input → output), an **author-written
`logger.info`** (already a sentence), and anything that **failed**.
Narrating every instrumented call produced forty lines a turn from eight
modules — technically complete and practically unreadable.

`LoggerFactory.subscribe` attaches the handler to every subsystem logger
that exists *and remembers it for the ones that do not yet*, so a
subsystem whose logger is built twenty stages later still reaches the
screen.

What produces those records:

- `@logged(subsystem)` — a CALLED/RETURNED pair per call, arguments and
  return value **in full** (`logging.payload_max_chars: 0`). Async
  methods get an async wrapper; a sync wrapper around a coroutine
  function traces the coroutine *object* and reports 0.0 ms, which is a
  line that looks like evidence and contains none.
- `@instrument(subsystem)` — the same, applied to every public method of
  a class. One annotation instead of one per method, so a step added
  later is traced because it exists, not because its author remembered.
- `@traced(stage, subsystem)` — logs *and* reports to the feed when the
  object owns a `StageReporter`. Lives in `observability`, not `tui`:
  business code that imported the UI to announce its own progress would
  invert the layering.

`/debug off|info|debug` changes only what is mirrored to the screen. The
files record at their configured level regardless.

---

## 3.3 The same walkthrough, for `make episode E=30`

An episode is this same control flow with two differences and no third:

```
python -m episodes.e30
  └── episodes/e30/__main__.py → run_episode(Path(__file__).parent, flow)
        ├── use_overlay("episodes/e30/config.yaml"); reload_settings()
        │     └── the run layer (CONFIG_REFERENCE.md), applied BEFORE the build
        ├── TraceStream(console.print).enable()
        │     └── §3.2's feed, on a plain Rich console instead of the TUI panel
        ├── build_agent()          ← §0.1, now wiring only what E30 enabled
        └── flow(EpisodeRun(...))  ← episodes/e30/app.py
              └── run.ask(turn) → agent.handle(...) → §2, unchanged
```

The overlay must be applied before `build_agent()`, because the builder
reads config to decide what to wire; applied afterwards it would produce
the fully-featured agent and an episode config that changed nothing.

Everything from §2 onward is identical — the episode is not a different
program, it is the same agent with a narrower configuration.

## 4. Where to look when something is wrong

| Symptom | First place to look |
|---|---|
| A refusal you did not expect | `decisions` in the feed; `scope.py`, then the Gatekeeper's audit row |
| The agent asks for a code it was given | `authentication.py` regex; the code needs a digit |
| A follow-up lost the subject | `intent.py::_inherit`, and `handle`'s history replay |
| The model called something it should not have | `_offered_tools`, then `offered_actions` in the Gatekeeper |
| An answer states something no tool returned | `_check_grounding`; it is advisory and logs the claims |
| The window forgot something | `/memory working`, then `graph._note` call sites |
| Nothing on screen, but it ran | `tui.trace.enabled`, or `/debug debug`; `logs/` has it either way |
| A step you want on screen | add it to `tui.trace.steps` — config, not code |
| Which version answered | `/versions`; the authority path never canaries |
| An incident | `/postmortem` — impact → trigger → missing control → detection gap → containment → root cause → permanent eval |

Commands worth knowing: `/help`, `/session`, `/config <section>`,
`/audit`, `/orders`, `/users`, `/memory working`, `/queue`, `/slo`,
`/model`, `/tools`, `/gatekeeper`, `/eval`, `/logs`, `/debug`, `/versions`,
`/postmortem`.

---

## 5. What is deliberately absent

**Money.** Tokens are counted; cost is not. Pricing a turn means
carrying a rate card per provider and a unit price for every tool,
retrieval and minute of review — a model that is wrong the moment a
price changes, and wrong in the direction that flatters it.

**Goodwill credits.** Refunds follow policy. A discretionary payout with
a friendly name is money leaving the business on a judgement call, and
the judgement was the model's.

**Self-elevation.** `JustInTimeElevator` exists and the agent never
calls it. An agent that can widen its own permissions has no ceiling;
elevation is an operator action.

**Sandbox, multi-agent, DSPy, the control-loop package.** Parked in the
PRD, interface-only here, and not pretended otherwise.
