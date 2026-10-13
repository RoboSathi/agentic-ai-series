.PHONY: gen-key help install clean lint type-check format test verify agent llm-check model-check models data refund-limit \
	clean-memory show-memory dream memory-reindex review queue audit dummy-data analytics-export analytics-drop-legacy dashboard dashboard-install dashboard-refresh \
	eval intent-eval _agent

# -- agent registry -------------------------------------------------------------
# The platform names no agent (`agent_platform/config.yaml` ships
# `overlays: []`), so the agent is named on the command line: agent-scoped
# targets take it as their second goal — `make agent csa`, `make eval ra`.
# Adding an agent is three lines here and nothing in the platform.
AGENTS := csa ra ops

csa_NAME := Customer Support Agent
csa_PKG := customer_support_agent
csa_OVERLAY := src/customer_support_agent/config/config.yaml
csa_DATA := $(csa_PKG).dummy_data.generate_dummy_data $(if $(KEEP_MEMORY),--keep-memory)
csa_OFFLINE_PROFILE :=
csa_EVALS := yes

ra_NAME := Research Agent
ra_PKG := research_agent
ra_OVERLAY := src/research_agent/config/config.yaml
ra_DATA :=
ra_OFFLINE_PROFILE :=
# No eval suite yet: the fixture-world suite was removed (ADR ra-0019).
ra_EVALS :=

ops_NAME := Ops Analyst (MCP demo: answers analytics questions through the platform's analytics MCP server)
ops_PKG := ops_agent
ops_OVERLAY := src/ops_agent/config/config.yaml
ops_DATA :=
ops_OFFLINE_PROFILE :=
ops_EVALS :=

# Targets whose second goal names the agent they act on.
AGENT_SCOPED := agent data review queue audit pii-reveal dream show-memory clean-memory memory-reindex models eval intent-eval

# `make eval <agent> [section] [full]`: one section (default: all three), over the agent's
# core cases, attacks and metrics unless `full` is typed.
EVAL_SECTIONS := oracle deepeval promptfoo
EVAL_TIERS := full

ifneq ($(filter $(firstword $(MAKECMDGOALS)),$(AGENT_SCOPED)),)
GOAL_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
# The agent name (and an eval's section and tier) are goals only so they can be typed;
# they are not targets.
ifneq ($(GOAL_ARGS),)
$(eval $(GOAL_ARGS):;@:)
endif
ifeq (eval,$(firstword $(MAKECMDGOALS)))
EVAL_SECTION := $(filter $(EVAL_SECTIONS),$(GOAL_ARGS))
EVAL_FULL := $(filter $(EVAL_TIERS),$(GOAL_ARGS))
endif
AGENT_ARGS := $(filter-out $(EVAL_SECTION) $(EVAL_FULL),$(GOAL_ARGS))
AGENT := $(filter $(AGENTS),$(AGENT_ARGS))
# Only a named, known, single agent chooses the config layer. Whichever
# agent is named sets it, so the overlay cannot silently default to one.
ifeq ($(words $(AGENT_ARGS)) $(words $(AGENT)),1 1)
export AGENTIC_OVERLAYS := $($(AGENT)_OVERLAY)
# A profile is a named posture of the agent's own config:
# `make agent csa PROFILE=<name>` (src/<pkg>/config/profiles/). Evals and
# `make data` always use the agent's offline profile, whatever was typed.
ifneq ($(filter data eval intent-eval,$(firstword $(MAKECMDGOALS))),)
override PROFILE := $($(AGENT)_OFFLINE_PROFILE)
endif
ifneq ($(PROFILE),)
export AGENTIC_OVERLAYS := $(AGENTIC_OVERLAYS):src/$($(AGENT)_PKG)/config/profiles/$(PROFILE).yaml
endif
endif
endif

PYTHON := ./venv/bin/python
EVALS := $(PYTHON) -m agent_platform.evals

# Generated data, one command: `make dummy-data csa` (each agent with a <agent>_DATA generator
# rebuilds its own database), `make dummy-data analytics` (the platform's demo analytics world),
# bare `make dummy-data` (every kind: agents first, then analytics).
DATA_AGENTS := $(strip $(foreach a,$(AGENTS),$(if $($(a)_DATA),$(a))))
DUMMY_KINDS := $(DATA_AGENTS) analytics
ifeq (dummy-data,$(firstword $(MAKECMDGOALS)))
DUMMY_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
DUMMY_KIND := $(or $(DUMMY_ARGS),$(DUMMY_KINDS))
ifneq ($(DUMMY_ARGS),)
$(eval $(DUMMY_ARGS):;@:)
endif
endif
# An agent's config layers for a data build: its overlay, then its offline profile, if any.
data_overlays = $($(1)_OVERLAY)$(if $($(1)_OFFLINE_PROFILE),:src/$($(1)_PKG)/config/profiles/$($(1)_OFFLINE_PROFILE).yaml)

# `SOURCE=dummy` points the readers (export, dashboards, the analytics MCP server) at the demo
# analytics source `make dummy-data` builds (`analytics.sources`: live, dummy). A running agent
# always records to the live source (`analytics.record_source`), whatever SOURCE says.
ifdef SOURCE
export ANALYTICS__SOURCE := $(SOURCE)
endif

# Provider budget override for this invocation: `RPM=38 make eval csa`.
# The configured default lives in resilience.rate_limit (config.yaml).
ifdef RPM
export RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS := $(RPM)
endif

# Bright, bold colours: the dark 0;3x set is hard to read on a dark terminal.
GREEN := \033[1;92m
YELLOW := \033[1;93m
BLUE := \033[1;96m
NC := \033[0m

help:
	@echo "$(BLUE)Agentic Harness$(NC)   agents: $(foreach a,$(AGENTS), $(a) → $($(a)_NAME) )"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install              Create the venv and install the package with dev + eval extras"
	@echo "  make clean                Remove venv, caches and logs"
	@echo ""
	@echo "$(GREEN)Quality$(NC) (repo-wide, no agent name):"
	@echo "  make lint                 ruff check"
	@echo "  make type-check           mypy --strict"
	@echo "  make format               ruff format + autofix"
	@echo "  make test                 pytest with coverage"
	@echo "  make verify               All gates: lint, format, types, tests (80% coverage), both composition roots"
	@echo ""
	@echo "$(GREEN)Run$(NC) (agent name required):"
	@echo "  make agent <agent>        The operator console (make agent csa; make agent ra); its facts and audit"
	@echo "                            always go to the live analytics source"
	@echo "  make agent ops            The Ops Analyst: the MCP demo agent. It answers questions about how the"
	@echo "                            agents are running, using only the platform's analytics MCP server's tools;"
	@echo "                            SOURCE=dummy reads the demo world from make dummy-data (its own turns still go to live)"
	@echo "  make audit <agent> [DECISION=…|STANDING=…|KIND=…|SINCE=24h|N=…|TRACE=<id>]   The audit trail, coloured"
	@echo "  make review <agent> [KIND=…|ID=…] [ACTION=approve|deny ID=… COMMENT=…]   Human review queue (same as /review)"
	@echo "  make pii-reveal <agent> T=<token> WHY=<reason>   What a PII token stood for (audited; needs PII_VAULT_KEY)"
	@echo "  make model-check          Every model the system uses (NVIDIA + local), each probed once and listed with what"
	@echo "                            it is used for, across the platform and every agent (BENCH=1 times chat models; alias: llm-check)"
	@echo "  make models [agent]       Download (if missing) and run each agent's local models (run by make install)"
	@echo ""
	@echo "$(GREEN)Analytics$(NC) (no agent name):"
	@echo "  make dummy-data [kind]    Generated data; no kind builds all ($(DUMMY_KINDS)):"
	@echo "                              csa        rebuild the agent DB from its seed (KEEP_MEMORY=1 keeps memory)"
	@echo "                              analytics  the demo analytics source (30 days, every profiled agent) + Parquet,"
	@echo "                                         read by any reader target given SOURCE=dummy"
	@echo "  make analytics-export     Export the source's views to Parquet when no console is running;"
	@echo "                            SOURCE=dummy exports the demo source"
	@echo "  make analytics-drop-legacy Drop the fact tables ADR 0009 retired (platform_llm_calls, ...)"
	@echo "  make dashboard-install    Install Rill into src/agent_platform/analytics/dashboards/.rill (checksum-verified, repo-local; skips if current, FORCE=1 reinstalls)"
	@echo "  make dashboard            Re-export the source's Parquet, generate the dashboards (Agent Platform + one per agent),"
	@echo "                            serve them; SOURCE=dummy serves the demo source"
	@echo "  make dashboard-refresh    Re-export the source's Parquet and reload a running dashboard;"
	@echo "                            SOURCE=dummy refreshes the demo source"
	@echo ""
	@echo "$(GREEN)Customer support admin$(NC) (csa only):"
	@echo "  make refund-limit 100     Set the per-refund limit the Gatekeeper enforces (agent DB)"
	@echo "  make show-memory <agent>  Everything long-term memory holds, as a tree (SUBJECT=<id> EVENTS=<n>)"
	@echo "  make dream <agent>        Consolidate pending dream events into memory (SHOW=1 for the full tree)"
	@echo "  make clean-memory <agent> Empty long-term memory (keeps checkpoints, business records, audit)"
	@echo "  make memory-reindex <agent>  Rebuild the derived search index from the memory files"
	@echo ""
	@echo "$(GREEN)Evals$(NC) (agents with a suite: $(strip $(foreach a,$(AGENTS),$(if $($(a)_EVALS),$(a)))))"
	@echo "  make eval <agent>             Oracle, DeepEval, then Promptfoo over the core set (~12 min for csa)"
	@echo "  make eval <agent> oracle      Golden cases → business invariants in the DB (no judge calls)"
	@echo "  make eval <agent> deepeval    Golden cases → LLM-judged conversation quality"
	@echo "  make eval <agent> promptfoo   Red-team attacks; every one must be defended"
	@echo "  … full                        Add to any of the above: every case, attack and metric"
	@echo "  make intent-eval <agent>      Intent classifier bake-off (offline, no model calls)"
	@echo "  Each section prints its table and writes reports/<pkg>/evals/<section>-<run_id>.md,"
	@echo "  every case with its result and why. Unchanged agent code reuses transcripts; RPM=<n> sets the budget."

# Fails the run unless exactly one known agent was named after the target.
_agent:
	@if [ "$(words $(AGENT_ARGS))" = "0" ]; then \
	  echo "$(YELLOW)Usage: make $(firstword $(MAKECMDGOALS)) <agent>   (agents: $(AGENTS))$(NC)"; exit 2; fi
	@if [ "$(words $(AGENT_ARGS))" != "1" ]; then \
	  echo "$(YELLOW)Name exactly one agent, got: $(AGENT_ARGS)   (agents: $(AGENTS)$(if $(filter eval,$(firstword $(MAKECMDGOALS))),; sections: $(EVAL_SECTIONS) [full]))$(NC)"; exit 2; fi
	@if [ -z "$(AGENT)" ]; then \
	  echo "$(YELLOW)Unknown agent '$(AGENT_ARGS)'   (agents: $(AGENTS))$(NC)"; exit 2; fi
	@if [ -n "$(PROFILE)" ] && [ ! -f "src/$($(AGENT)_PKG)/config/profiles/$(PROFILE).yaml" ]; then \
	  echo "$(YELLOW)No profile '$(PROFILE)' for $(AGENT) (src/$($(AGENT)_PKG)/config/profiles/)$(NC)"; exit 2; fi
	@if [ -n "$(filter eval intent-eval,$(firstword $(MAKECMDGOALS)))" ] && [ -z "$($(AGENT)_EVALS)" ]; then \
	  echo "$(YELLOW)'$(AGENT)' has no eval suite   (make help → Evals)$(NC)"; exit 2; fi
	@if [ "$(words $(EVAL_SECTION))" -gt 1 ]; then \
	  echo "$(YELLOW)Name at most one section, got: $(EVAL_SECTION)   (make eval <agent> [$(subst $(space),|,$(EVAL_SECTIONS))] [full])$(NC)"; exit 2; fi

install:
	python3 -m venv venv
	./venv/bin/pip install -q --upgrade pip
	./venv/bin/pip install -q -e ".[dev,evals]"
	@$(MAKE) --no-print-directory models
	@echo "$(GREEN)✓ Installed, with every agent's local models. Activate with: source venv/bin/activate$(NC)"

clean:
	rm -rf venv .pytest_cache .coverage .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*.log logs/*/

lint:
	./venv/bin/ruff check src/ tests/

type-check:
	./venv/bin/mypy --strict src/

format:
	./venv/bin/ruff format src/ tests/
	./venv/bin/ruff check src/ tests/ --fix

test:
	./venv/bin/pytest tests/ --cov=src --cov-report=term-missing -q

agent: _agent
	$(PYTHON) -m agent_platform.interface.tui

# Every agent's models, one agent at a time (its overlay names its roles and local models):
# `make models` downloads and runs every agent's local models, or one agent's: `make models ra`.
MODEL_CHECK_AGENTS = $(or $(AGENT),$(AGENTS))

# Download (if missing), load and run every local model each agent uses — a fresh checkout's
# first step after `make install`: `make models`, or `make models ra`.
models:
	@$(foreach a,$(MODEL_CHECK_AGENTS),echo "$(BLUE)── $($(a)_NAME) ($(a)) ──$(NC)" && \
		AGENTIC_OVERLAYS=$($(a)_OVERLAY) $(PYTHON) -m agent_platform.llm.local_probes && ) true

# Every model the whole system uses: the platform's config, then each agent's layer over it.
llm-check model-check:
	$(PYTHON) -m agent_platform.llm.cli $(foreach a,$(AGENTS),--layer $(a)=$($(a)_OVERLAY)) $(if $(BENCH),--bench)

# Kept so `make data csa` still works; generated data is `make dummy-data <kind>`.
data: _agent
	@echo "$(YELLOW)make data $(AGENT) is now make dummy-data $(AGENT)$(NC)"
	@$(MAKE) --no-print-directory dummy-data $(AGENT)

# The one human review queue (ADR 0012): refunds, cut replies, quarantines, memory a curator
# must confirm. `make review csa`, `make review ra KIND=skill`, `make review csa ID=7`,
# `make review csa ACTION=approve ID=7 COMMENT="checked the order"`. Same as /review.
# `make queue` is kept as an alias.
review queue: _agent
	@$(PYTHON) -m agent_platform.interface.commands.review_cli --agent $(AGENT) -- $(if $(ACTION),$(ACTION) $(ID) "$(COMMENT)",$(if $(ID),show $(ID),$(KIND)))

# The audit trail, coloured by outcome, from every ledger the agent writes to (ADR 0012):
# `make audit csa`, `make audit ra DECISION=deny`, `make audit csa KIND=review. SINCE=24h N=50`,
# `make audit csa TRACE=<trace id>` (one turn as a tree). Tokens only — values come from pii-reveal.
audit: _agent
	@$(PYTHON) -m agent_platform.audit.cli $(if $(DECISION),--decision $(DECISION)) $(if $(STANDING),--standing $(STANDING)) $(if $(KIND),--kind $(KIND)) $(if $(SINCE),--since $(SINCE)) $(if $(N),--limit $(N)) $(if $(TRACE),--trace $(TRACE))

# What a PII redaction token stood for, for audit: `make pii-reveal csa T=3f9a1c2b7e10 WHY="ticket 42"`.
# Every reveal is written to the audit trail first; it needs the vault key (PII_VAULT_KEY).
pii-reveal: _agent
	@test -n "$(T)" -a -n "$(WHY)" || (echo "$(YELLOW)Usage: make pii-reveal <agent> T=<token> WHY=<reason>$(NC)" && exit 2)
	@$(PYTHON) -m agent_platform.security.pii.reveal "$(T)" --why "$(WHY)"

# -- customer support admin (csa only) ---------------------------------------------
# `make refund-limit 100`: the amount is the second goal, so it is given a
# no-op rule here rather than being run as a target of its own.
ifeq (refund-limit,$(firstword $(MAKECMDGOALS)))
REFUND_LIMIT := $(word 2,$(MAKECMDGOALS))
$(eval $(REFUND_LIMIT):;@:)
endif

CSA_ENV := AGENTIC_OVERLAYS=$(csa_OVERLAY)

# -- long-term memory (every agent; ADR 0012) -----------------------------------------
# `make dream ra`, `make show-memory csa SUBJECT=1`, `make clean-memory ops`: one platform
# CLI, the agent chosen by its config layer. Files live in data/<pkg>/memory/.
MEMORY_CLI := $(PYTHON) -m agent_platform.interface.commands.memory_cli

dream: _agent
	@$(MEMORY_CLI) dream $(if $(SHOW),--show-memory)

show-memory: _agent
	@$(MEMORY_CLI) show $(if $(SUBJECT),--subject $(SUBJECT)) $(if $(EVENTS),--events $(EVENTS))

clean-memory: _agent
	@$(MEMORY_CLI) clean

memory-reindex: _agent
	@$(MEMORY_CLI) reindex

refund-limit:
	@test -n "$(REFUND_LIMIT)" || (echo "$(YELLOW)Usage: make refund-limit 100$(NC)" && exit 1)
	$(CSA_ENV) $(PYTHON) -m customer_support_agent.admin.refund_limit $(REFUND_LIMIT)

# -- evals -----------------------------------------------------------------------
# Each section prints a colour-coded overview and writes reports/<pkg>/evals/<section>-<run_id>.md.

eval: _agent
	$(EVALS) run $(if $(EVAL_SECTION),--section $(EVAL_SECTION)) $(if $(EVAL_FULL),--full) \
	  --invocation "make eval $(strip $(AGENT) $(EVAL_SECTION) $(EVAL_FULL))"

intent-eval: _agent
	$(EVALS) run --section intent --invocation "make intent-eval $(AGENT)"

# -- analytics -------------------------------------------------------------------
# The active source's Parquet (curated views) is refreshed by a running console every
# analytics.export_interval_seconds; these work with no agent running.
comma := ,
space := $(empty) $(empty)
DASHBOARDS_DIR := src/agent_platform/analytics/dashboards
RILL_DIR := $(DASHBOARDS_DIR)/.rill
RILL := $(RILL_DIR)/rill
RILL_PROJECT := $(DASHBOARDS_DIR)/rill
RILL_VERSION ?= latest
RILL_PORT ?= 9009
# Rill's internal gRPC port; change both to run a second Rill beside a running one.
RILL_GRPC_PORT ?= 49009
# Rill caches each model; these (the generated project's models) are reloaded on refresh.
RILL_MODELS = $(shell $(PYTHON) -m agent_platform.analytics models 2>/dev/null)
PARQUET_DIR := $(shell $(if $(SOURCE),ANALYTICS__SOURCE=$(SOURCE) )$(PYTHON) -c "from agent_platform.config import get_settings; print(get_settings().analytics.resolved_parquet_dir)" 2>/dev/null)

dummy-data:
	@test -z "$(filter-out $(DUMMY_KINDS),$(DUMMY_KIND))" || \
	  (echo "$(YELLOW)Usage: make dummy-data [kind]   (kinds: $(DUMMY_KINDS); none: all of them)$(NC)" && exit 2)
	@$(foreach a,$(filter $(DATA_AGENTS),$(DUMMY_KIND)),echo "$(BLUE)── $($(a)_NAME) ($(a)) ──$(NC)" && \
		AGENTIC_OVERLAYS=$(call data_overlays,$(a)) $(PYTHON) -m $($(a)_DATA) && ) true
	@$(if $(filter analytics,$(DUMMY_KIND)),echo "$(BLUE)── Demo analytics source ──$(NC)" && \
		$(PYTHON) -m agent_platform.analytics dummy && \
		echo "$(GREEN)✓ Then: make dashboard SOURCE=dummy · make agent ops SOURCE=dummy$(NC)",true)

analytics-export:
	@$(PYTHON) -m agent_platform.analytics export

# The ad-hoc fact tables ADR 0009 retired are dropped, not migrated: run once, with no agent running.
analytics-drop-legacy:
	@$(PYTHON) -m agent_platform.analytics drop-legacy

dashboard-install:
	@FORCE=$(FORCE) sh $(DASHBOARDS_DIR)/install_rill.sh $(RILL_DIR) $(RILL_VERSION)

dashboard:
	@test -e $(RILL) || (echo "$(YELLOW)Rill is not installed: make dashboard-install$(NC)" && exit 2)
	@test -x $(RILL) || (echo "$(YELLOW)$(RILL) is not executable: an install may still be running; wait for it, or FORCE=1 make dashboard-install$(NC)" && exit 2)
	@$(PYTHON) -m agent_platform.analytics export > /dev/null || \
	  echo "$(YELLOW)Could not re-export (a running console holds the database; it exports every analytics.export_interval_seconds): serving its last export$(NC)"
	@test -f "$(PARQUET_DIR)/manifest.json" || (echo "$(YELLOW)No Parquet export yet at $(PARQUET_DIR): make analytics-export (or run an agent)$(NC)" && exit 2)
	@ln -sfn "$(PARQUET_DIR)" $(RILL_PROJECT)/data
	@$(PYTHON) -m agent_platform.analytics dashboards
	@$(PYTHON) -c "from agent_platform.analytics import read_manifest; m = read_manifest(); v = m.views.get('tasks') if m else None; print('$(YELLOW)The Agent Platform dashboard is empty until an agent has run a turn: make agent <agent>, then make dashboard-refresh$(NC)') if v is None or v.rows == 0 else None"
	@echo "$(GREEN)Rill on http://localhost:$(RILL_PORT) (Ctrl+C to stop)$(NC)"
	@sh $(DASHBOARDS_DIR)/serve_rill.sh $(RILL) start --reset --port $(RILL_PORT) --port-grpc $(RILL_GRPC_PORT) $(RILL_PROJECT)

# Re-export the source's Parquet and have a running `make dashboard` reload it (Rill does not watch Parquet).
dashboard-refresh: analytics-export
	@$(PYTHON) -m agent_platform.analytics dashboards > /dev/null
	@curl -fsS -X POST localhost:$(RILL_PORT)/v1/instances/default/trigger -H 'Content-Type: application/json' \
		-d '{"models":[$(subst $(space),$(comma),$(foreach m,$(RILL_MODELS),{"model":"$(m)","full":true}))]}' > /dev/null \
		&& echo "$(GREEN)✓ Rill reloaded $(RILL_MODELS)$(NC)" || echo "$(YELLOW)No Rill running on :$(RILL_PORT) (make dashboard)$(NC)"

# Fresh 32-byte keys: the checkpoint key (durability.encryption) and the PII vault key
# (security.pii.vault), which also keys the redaction tokens. Put the lines in .env, or export
# them:  make gen-key >> .env
gen-key:
	@echo "LANGGRAPH_AES_KEY=$$(openssl rand -hex 16)"
	@echo "PII_VAULT_KEY=$$(openssl rand -hex 16)"

# Quick verification (for CI/CD).
#
# The coverage gate is enforced by pytest-cov's own `fail_under = 80`
# (pyproject.toml), so a shortfall fails the command itself. An earlier
# version grepped the report for the string "80", which matched any line
# containing those digits and therefore passed at any coverage at all.
#
# Each composition root imports in its own process: importing two agents
# in one process lets the first overlay declaration win for both.
verify:
	@echo "$(BLUE)1/4 Ruff lint + format...$(NC)"
	./venv/bin/ruff check src/ tests/ > /dev/null && \
	./venv/bin/ruff format --check src/ tests/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)2/4 mypy --strict...$(NC)"
	./venv/bin/mypy --strict src/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)3/4 Pytest with coverage gate (80%)...$(NC)"
	./venv/bin/pytest tests/ --cov=src -q > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)4/4 Composition root imports (one process per agent)...$(NC)"
	$(foreach a,$(AGENTS),AGENTIC_OVERLAYS=$($(a)_OVERLAY) $(PYTHON) -c "from $($(a)_PKG).graph import build_agent; build_agent" || exit 1;) \
	echo "$(GREEN)✓$(NC)"
	@echo "$(GREEN)✓ Verification complete - all gates green$(NC)"
