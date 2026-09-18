# `config.yaml` reference

Every value the system uses comes from `src/agentic_commons/config/config.yaml`.
Nothing is hardcoded in Python. Override locally with the gitignored
`config/local.yaml`, or per-process with `SECTION__FIELD` environment
variables (e.g. `LLM__MAX_TOKENS=500`).

**API keys are the one exception.** They are read from bare lowercase env
vars (`nvidia_api_key`, `hf_api_key`), are never config fields, and never
appear in a settings dump.

Every subsystem has an `enabled` flag that can be flipped at runtime
through `ConfigFeatureToggle` — no restart. A sub-feature is on only if
its parent is too. The Gatekeeper is the deliberate exception: it has no
`enabled` flag because no execution path may bypass it (PRD G1); only its
sub-features toggle.

---

## `config`
| Param | Meaning |
|---|---|
| `enabled` | Meta-toggle for the config subsystem itself. |

## `logging`
| Param | Meaning |
|---|---|
| `level` | Baseline level for the logger. |
| `console_enabled` / `console_level` | Whether the terminal gets logs, and from which level. |
| `file_enabled` / `file_level` | Whether files get logs. `DEBUG` here on purpose — the file is the authoritative record and captures more than the console. |
| `dir` | Log directory, resolved from the repo root, never the CWD. |
| `per_subsystem_color` | Colour each subsystem differently in the console. |
| `max_bytes` / `backup_count` | Rotation size and how many rotated files to keep. |

## `llm`
| Param | Meaning |
|---|---|
| `provider` | Active provider: `nim` or `local`. |
| `max_tokens` | Default cap per completion. Unbounded generation is what caused timeouts. |
| `log_payloads` | Write every request and response to `logs/llm.log`. The single most useful switch when the agent misbehaves. |
| `payload_max_chars` | Characters kept per logged message. **`0` = the whole thing**, which is the default: a truncated prompt hides the part you are debugging. |
| `providers.<name>.base_url` | OpenAI-compatible endpoint. |
| `providers.<name>.model` | Model id. NIM entitlements are per-account, so not every published model is invokable. |
| `providers.<name>.timeout_seconds` | Per-request timeout. |
| `providers.<name>.input_cost_per_1k_tokens` | Input price, used for the cost stack. |
| `providers.<name>.output_cost_per_1k_tokens` | Output price. |
| `cascade.enabled` | Run tiers cheapest-first (E44) instead of a single model. |
| `cascade.tiers` | Ordered provider list, cheapest first. |
| `cascade.escalate_on_finish_reasons` | Finish reasons that force the next tier (e.g. `length`). |
| `cascade.escalate_on_empty_content` | Escalate when a tier returns nothing. |
| `cascade.escalate_on_error` | Escalate when a tier errors. |
| `embedding.provider` | Embedding backend: `nim` or `local`. |
| `embedding.nim_model` | Hosted embedding model. |
| `embedding.local_dimensions` | Vector width for the offline hashing fallback. |

## `persistence`
| Param | Meaning |
|---|---|
| `db_path` | SQLite file, resolved from the repo root. |

## `audit`
| Param | Meaning |
|---|---|
| `policy_version` / `data_version` | Stamped onto every audit entry so a decision is attributable to versioned policy and data (E33). |

## `observability`
| Param | Meaning |
|---|---|
| `langsmith_enabled` / `langsmith_project` | Dual-write traces to LangSmith; degrades gracefully if unavailable. |
| `persist_traces` | Write spans to SQLite. |
| `slo_targets.*` | Target per agent SLO (E37). Error budget is measured against these. `unauthorized_action_rate` and `policy_violation_rate` are `0` — any occurrence is a breach. |
| `review_queue.minutes_per_hour` | Unit conversion for the Little's Law staffing model (E38). |

## `context`
| Param | Meaning |
|---|---|
| `token_budget` | Ceiling for the assembled window. |
| `chars_per_token` | Character-based token estimate. |
| `retrieval_mode` | `lexical` \| `dense` \| `hybrid` \| `graph`. Hybrid picks by query shape (E19). |
| `top_k` | Chunks retained by the retrieve strategy. |
| `hybrid_lexical_weight` | Lexical/dense blend. Identifier-shaped queries force lexical regardless. |
| `bm25_k1` | How much a repeated term can help. |
| `bm25_b` | How much a long document is penalised for length. |
| `strategies.*` | Per-strategy toggles: offload, reduce, retrieve, isolate, cache, consolidate (E18). |
| `reduce_keep_ratio` | Fraction of a chunk kept when reducing. |
| `failure_detection.checkpoints_tokens` | Window sizes at which synthesis quality is re-measured (E17). |
| `failure_detection.degradation_threshold` | Drop from baseline that counts as degradation. |

## `memory`
| Param | Meaning |
|---|---|
| `working.enabled` | Working memory — a pointer into the window, not a store. |
| `episodic.path` | Append-only session logs, one file per session, grep-able. |
| `semantic.fuzzy_match_threshold` | Similarity floor for fuzzy key lookup, tried before any embedding. |
| `write_pipeline.auto_commit_threshold` | Confidence at or above which a write commits unattended. |
| `write_pipeline.discard_threshold` | Confidence below which a write is dropped. Between the two ⇒ curator queue. |
| `write_pipeline.decay_if_unconfirmed_days` | When an unconfirmed self-reported fact loses standing — a security posture, not hygiene (E16). |
| `write_pipeline.authority_changing_categories` | Categories never writable from conversational assertion. No role grants itself authority (E15). |
| `write_pipeline.classification_hints` | Phrase → category, the deterministic first-tier classifier. |
| `provenance_weights.*` | Retrieval weight by how a fact was learned: verified > inferred > self-reported. |
| `retrieval_weights.*` | Composite scoring: relevance, recency, importance, authority, temporal validity (Park et al. extended, E13). |
| `recency_half_life_days` | Half-life for recency decay. |

## `guardrails`
| Param | Meaning |
|---|---|
| `mode` | `reject` \| `auto_repair` \| `constrained`. Validation is what you do when you cannot constrain the decoder. |
| `max_repair_attempts` | Bound on mechanical repair attempts. |
| `constrained_backend` | `llamacpp_grammar` (a real grammar) or `nim_guided_json` (best-effort hint only). |

## `hallucination_protection`
| Param | Meaning |
|---|---|
| `confidence_threshold` | Below this, a claim escalates down the detection ladder. |
| `consistency_samples` | Samples for the self-consistency signal. |
| `consistency_agreement_threshold` | Agreement required across samples. |
| `grounding_overlap_threshold` | Overlap separating a **grounding** failure (evidence absent) from a **reasoning** failure (evidence present, ignored) — opposite fixes (E26). |

## `gatekeeper`
No `enabled` flag by design — it is always invoked.

| Param | Meaning |
|---|---|
| `authority_grant_max_rank` | Only ranks at or above this may grant authority. Default 5. |
| `authority_hierarchy` | Ordered strongest-first: provider → organization → product → workspace → user_task → runtime_reminders → tool_observations → untrusted_content. This is why an injected instruction cannot authorize anything. |
| `ledger.window_hours` | Rolling window for cumulative spend. |
| `ledger.action_classes` | Action → budget class. Mapped explicitly so renaming a tool cannot silently stop tracking it. |
| `ledger.thresholds_usd` | Cumulative cap per class. Defeats the structuring attack: 80 × $50 is one $4,000 loss (E02). |
| `tier2.enabled` | Semantic advisory review. Annotates and escalates; structurally cannot block. |
| `surface_policy.surfaces` | Action → `locked` \| `editable` \| `append_only` \| `human_controlled`. An unmapped action defaults to human-controlled. |
| `automation.human_approval_below` | Sheridan level under which a human is required (E03). |
| `automation.levels` | Sheridan 1–10 per action. |
| `automation.per_action_amount_escalation_usd` | Amount above which an action escalates regardless of its level. |

## `security`
| Param | Meaning |
|---|---|
| `auth.elevation_max_ttl_seconds` | Hard cap on a just-in-time privilege grant. Requests above it are clamped. |
| `auth.roles` | Role → permissions (NIST RBAC). |
| `auth.action_permissions` | Action → required permission. Unmapped ⇒ denied. |
| `pii.mode` | `mask` \| `tokenize` \| `remove`. Tokenizing is stable, so records still join. |
| `pii.mask_token` | Replacement text. |
| `pii.ner_enabled` | Named-entity detection for what regex cannot express. |
| `pii.patterns` | Category → regex, ordered most specific first. |
| `pii.tool_call_allowed_fields` | Allow-list for tool-call payloads (data minimization, E32). |
| `prompt_injection.data_open_tag` / `data_close_tag` | Delimiters wrapping untrusted data. |
| `prompt_injection.suspicious_phrases` | Known injection phrasings. Advisory — the Gatekeeper is the real control. |
| `content_moderation.blocked_terms` | Flagged terms, matched on word boundaries. Advisory only. |

## `resilience`
| Param | Meaning |
|---|---|
| `circuit_breaker.failure_threshold` | Consecutive failures that trip the circuit. |
| `circuit_breaker.half_open_after_seconds` | Cooldown before one trial call is allowed. |
| `retry.max_attempts` | Attempt cap. |
| `retry.base_delay_seconds` / `max_delay_seconds` | Exponential backoff with full jitter, so clients do not retry in sync. |
| `timeout.default_seconds` | Default per-operation deadline. |
| `bulkhead.pool_size` | Concurrent permits per dependency pool, so one saturated dependency cannot starve another. |

## `evals`
| Param | Meaning |
|---|---|
| `manifest_path` | The checked-in eval plan (E09). |
| `cascade.*` | Per-tier toggles for the NOLLM cascade: exact assertion → deterministic rule → embedding similarity → NLI entailment → calibrated judge → human adjudication (E10). |
| `embedding_similarity_threshold` | Similarity at or above which a claim is supported. |
| `embedding_dissimilarity_threshold` | Similarity at or below which it is contradicted. The middle deliberately falls through. |
| `nli_entailment_threshold` | Confidence required from the entailment tier. |
| `judge_confidence_threshold` | Confidence below which a judge's verdict is discarded and escalated. |
| `kappa_acceptable` | Cohen's kappa a judge must reach. Raw accuracy flatters on skewed labels (E11). |
| `nli.min_overlap` | Coverage floor for the lexical entailment fallback. |
| `nli.negation_markers` | Polarity markers — what separates "approved" from "not approved". |
| `judge.temperature` / `max_tokens` | Sampling for the judge tier. |

## `tools`
| Param | Meaning |
|---|---|
| `selection_k` | Maximum tools offered per turn. |
| `selection_min_score` | Relevance floor. Below it the turn is offered **nothing**, never everything. Scale is embedding-backend specific — re-measure if the backend changes. |

## `tui`
| Param | Meaning |
|---|---|
| `refresh_interval_ms` | Spinner tick rate. |
| `show_token_count` | Show the turn and session token counters. |
| `copy_method` | `osc52` (works over SSH) or `file`. A file is always written either way. |
| `max_feed_lines` | Feed backlog cap. |
| `spinner_frames` | Spinner animation frames. |
| `stage_labels` | Stage key → the line shown. A stage with no label is a contract violation, surfaced rather than defaulted. |
| `tool_labels` | Tool → wording. "Looking up the order…", not "Executing support.get_order_status…". |
| `stage_styles` | Stage → colour. Security warns, data prep is cool, model calls blue, authority stands out; green is reserved for completed work. |
| `default_stage_style` | Fallback colour. |

## `customer_support_agent`
| Param | Meaning |
|---|---|
| `max_tool_iterations` | Cap on reason⇄act rounds per turn. |
| `default_actor` / `default_role` | The agent's own identity for RBAC. |
| `default_authority_rank` | Rank the agent acts under. |
| `intents` | Ordered, most specific first. Each is `name` + `phrases` + the `tools` it unlocks. This is why a refund is discussed only when the customer raises it; `tools: []` means answer conversationally. |
| `default_intent` | Used when nothing matches — no tools. |
| `followup.phrases` | Lead-ins that continue the previous subject ("what about…"). |
| `followup.reference_pattern` | A bare reference (an order number) that also continues it. |
| `history_turns` | Prior turns replayed to the model. Enough to resolve a follow-up, small enough to stay cheap. |
| `goodwill.max_per_credit_usd` | Cap on a single goodwill credit. |
| `goodwill.max_per_customer_usd` | Rolling cap per customer, so the per-credit cap cannot be walked past. |
| `goodwill.window_days` | Window for that rolling cap. |
| `deliberation_markers` | Phrases marking model thinking rather than a customer reply. Matching sends the turn back for a proper reply. |
| `demo_customer_id` | Identity for the local demo and TUI. Real deployments take this from authentication. |
| `checkpoint_path` | LangGraph checkpoint database. |
| `policy_path` | `customer_policy.md`, resolved package-relative. |
