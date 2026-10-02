.PHONY: gen-key help install clean lint type-check format test verify tui episode llm-check model-check models data refund-limit \
	clean-memory show-memory dream runs patterns \
	eval eval-generate eval-oracle eval-security eval-judge eval-deepeval eval-report eval-diff \
	eval-runs eval-sql _agent

# -- agent registry -------------------------------------------------------------
# The platform names no agent (`agentic_commons/config.yaml` ships
# `overlays: []`), so the agent is named on the command line: agent-scoped
# targets take it as their second goal — `make tui csa`, `make eval ra`.
# Adding an agent is three lines here and nothing in the platform.
AGENTS := csa ra

csa_NAME := Customer Support Agent
csa_PKG := customer_support_agent
csa_OVERLAY := src/customer_support_agent/config/config.yaml
csa_DATA := $(csa_PKG).dummy_data.generate_dummy_data $(if $(KEEP_MEMORY),--keep-memory)
csa_RUNS :=
csa_OFFLINE_PROFILE :=
csa_EVALS := yes

ra_NAME := Research Agent
ra_PKG := research_agent
ra_OVERLAY := src/research_agent/config/config.yaml
ra_DATA :=
ra_RUNS := $(ra_PKG).archive.cli
ra_OFFLINE_PROFILE :=
# No eval suite yet: the fixture-world suite was removed (ADR ra-0019).
ra_EVALS :=

# Targets whose second goal names the agent they act on.
AGENT_SCOPED := tui data runs model-check llm-check models eval eval-generate eval-oracle eval-security eval-judge \
	eval-deepeval eval-report eval-diff eval-runs eval-sql

ifneq ($(filter $(firstword $(MAKECMDGOALS)),$(AGENT_SCOPED)),)
AGENT_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
AGENT := $(filter $(AGENTS),$(AGENT_ARGS))
# The agent name is a goal only so it can be typed; it is not a target.
ifneq ($(AGENT_ARGS),)
$(eval $(AGENT_ARGS):;@:)
endif
# Only a named, known, single agent chooses the config layer. Whichever
# agent is named sets it, so the overlay cannot silently default to one.
ifeq ($(words $(AGENT_ARGS)) $(words $(AGENT)),1 1)
export AGENTIC_OVERLAYS := $($(AGENT)_OVERLAY)
# A profile is a named posture of the agent's own config:
# `make tui csa PROFILE=<name>` (src/<pkg>/config/profiles/). Evals and
# `make data` always use the agent's offline profile, whatever was typed.
ifneq ($(filter data eval eval-%,$(firstword $(MAKECMDGOALS))),)
override PROFILE := $($(AGENT)_OFFLINE_PROFILE)
endif
ifneq ($(PROFILE),)
export AGENTIC_OVERLAYS := $(AGENTIC_OVERLAYS):src/$($(AGENT)_PKG)/config/profiles/$(PROFILE).yaml
endif
# An ablation is one more config layer over the agent's own:
# `make eval-generate ra ABLATION=no_rubric_audit` (src/<pkg>/evals/ablations/).
ifdef ABLATION
export AGENTIC_OVERLAYS := $(AGENTIC_OVERLAYS):src/$($(AGENT)_PKG)/evals/ablations/$(ABLATION).yaml
endif
endif
endif

PYTHON := ./venv/bin/python
EVALS := $(PYTHON) -m agentic_commons.evals

# Provider budget override for this invocation: `RPM=38 make eval csa`.
# The configured default lives in resilience.rate_limit (config.yaml).
ifdef RPM
export RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS := $(RPM)
endif
# Keep going past a broken business invariant: `EVAL_CONTINUE_ON_INVARIANT_FAILURE=1 make eval csa`.
ifdef EVAL_CONTINUE_ON_INVARIANT_FAILURE
export EVALS__GATE__STOP_ON_INVARIANT_FAILURE := false
endif

# Case selection and cache control for the eval targets.
EVAL_SELECT := $(if $(FRESH),--fresh) $(if $(LIMIT),--limit $(LIMIT)) \
	$(foreach c,$(CASE),--case $(c)) $(foreach t,$(TAG),--tag $(t))
EVAL_RUN := $(if $(RUN),--run $(RUN)) $(if $(FRESH),--fresh)

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
	@echo "  make tui <agent>          The operator console (make tui csa; make tui ra)"
	@echo "  make data <agent>         csa: rebuild the agent DB (KEEP_MEMORY=1); ra: build + validate the fixture world"
	@echo "  make runs ra              List archived research runs"
	@echo "  make episode E=05         One episode (calls the live model)"
	@echo "  make model-check [agent]  Probe every hosted model and run every local model, per agent (BENCH=1 times them; alias: llm-check)"
	@echo "  make models [agent]       Download (if missing) and run each agent's local models (run by make install)"
	@echo "  make patterns             Pattern coverage matrix across agents"
	@echo ""
	@echo "$(GREEN)Customer support admin$(NC) (csa only):"
	@echo "  make refund-limit 100     Set the per-refund limit the Gatekeeper enforces (agent DB)"
	@echo "  make show-memory          Everything long-term memory holds, as a tree (CUSTOMER=<id> EVENTS=<n>)"
	@echo "  make dream                Consolidate recent conversations into memory (SHOW=1 for the full tree)"
	@echo "  make clean-memory         Empty long-term memory (keeps checkpoints, orders, reviews, audit)"
	@echo ""
	@echo "$(GREEN)Evals$(NC) (agent name required; order: generate → oracle → security → judge → deepeval):"
	@echo "  make eval <agent>         The whole pipeline; stops after the oracle on a broken invariant"
	@echo "  make eval-generate <agent>   Run the agent over the golden set (reuses unchanged transcripts)"
	@echo "  make eval-oracle <agent>     Deterministic invariants (no model calls)"
	@echo "  make eval-security <agent>   Security suite (csa: Promptfoo; ra: seeded hostile pages)"
	@echo "  make eval-judge <agent>      LLM judge: rubric quality + hallucination"
	@echo "  make eval-deepeval <agent>   DeepEval quality metrics"
	@echo "  make eval-report <agent>     Rebuild the Markdown reports (RUN=<id>, default latest)"
	@echo "  make eval-diff <agent> A=<run> B=<run>   Regression report between two runs"
	@echo "  make eval-runs <agent>       List recent runs"
	@echo "  make eval-sql <agent>        Query the eval store (SQL=\"select ...\" or interactive)"
	@echo ""
	@echo "  Options: RPM=<n>  FRESH=1  LIMIT=<n>  CASE=<id>  TAG=<tag>  RUN=<id>  ABLATION=<name>  PROFILE=<name>"
	@echo "           EVAL_CONTINUE_ON_INVARIANT_FAILURE=1"

# Fails the run unless exactly one known agent was named after the target.
_agent:
	@if [ "$(words $(AGENT_ARGS))" = "0" ]; then \
	  echo "$(YELLOW)Usage: make $(firstword $(MAKECMDGOALS)) <agent>   (agents: $(AGENTS))$(NC)"; exit 2; fi
	@if [ "$(words $(AGENT_ARGS))" != "1" ]; then \
	  echo "$(YELLOW)Name exactly one agent, got: $(AGENT_ARGS)   (agents: $(AGENTS))$(NC)"; exit 2; fi
	@if [ -z "$(AGENT)" ]; then \
	  echo "$(YELLOW)Unknown agent '$(AGENT_ARGS)'   (agents: $(AGENTS))$(NC)"; exit 2; fi
	@if [ -n "$(PROFILE)" ] && [ ! -f "src/$($(AGENT)_PKG)/config/profiles/$(PROFILE).yaml" ]; then \
	  echo "$(YELLOW)No profile '$(PROFILE)' for $(AGENT) (src/$($(AGENT)_PKG)/config/profiles/)$(NC)"; exit 2; fi
	@if [ -n "$(ABLATION)" ] && [ ! -f "src/$($(AGENT)_PKG)/evals/ablations/$(ABLATION).yaml" ]; then \
	  echo "$(YELLOW)No ablation '$(ABLATION)' for $(AGENT) (src/$($(AGENT)_PKG)/evals/ablations/)$(NC)"; exit 2; fi
	@if [ "$(filter eval eval-%,$(firstword $(MAKECMDGOALS)))" != "" ] && [ -z "$($(AGENT)_EVALS)" ]; then \
	  echo "$(YELLOW)'$(AGENT)' has no eval suite$(NC)"; exit 2; fi

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
	./venv/bin/ruff check src/ tests/ design_patterns/

type-check:
	./venv/bin/mypy --strict src/ design_patterns/

format:
	./venv/bin/ruff format src/ tests/ design_patterns/
	./venv/bin/ruff check src/ tests/ design_patterns/ --fix

test:
	./venv/bin/pytest tests/ --cov=src --cov-report=term-missing -q

tui: _agent
	$(PYTHON) -m agentic_commons.tui

# Run one episode: make episode E=05  (or E=48a). Configured episodes call
# the live model — this is a recording-session tool, not a check.
episode:
	@test -n "$(E)" || (echo "$(YELLOW)Usage: make episode E=05$(NC)" && exit 1)
	AGENTIC_OVERLAYS=$(csa_OVERLAY) $(PYTHON) -m episodes.e$(E)

# Every agent's models, one agent at a time (its overlay names its roles and local models):
# `make model-check`, or `make model-check ra`. BENCH=1 also times each chat model.
MODEL_CHECK_AGENTS = $(or $(AGENT),$(AGENTS))

# Download (if missing), load and run every local model each agent uses — a fresh checkout's
# first step after `make install`: `make models`, or `make models ra`.
models:
	@$(foreach a,$(MODEL_CHECK_AGENTS),echo "$(BLUE)── $($(a)_NAME) ($(a)) ──$(NC)" && \
		AGENTIC_OVERLAYS=$($(a)_OVERLAY) $(PYTHON) -m agentic_commons.llm.local_probes && ) true

llm-check model-check:
	@$(foreach a,$(MODEL_CHECK_AGENTS),echo "$(BLUE)── $($(a)_NAME) ($(a)) ──$(NC)" && \
		AGENTIC_OVERLAYS=$($(a)_OVERLAY) $(PYTHON) -m agentic_commons.llm.cli $(if $(BENCH),--bench) && ) true

data: _agent
	@test -n "$($(AGENT)_DATA)" || (echo "$(YELLOW)'$(AGENT)' has no generated data$(NC)" && exit 2)
	$(PYTHON) -m $($(AGENT)_DATA)

runs: _agent
	@test -n "$($(AGENT)_RUNS)" || (echo "$(YELLOW)'$(AGENT)' keeps no archived runs$(NC)" && exit 2)
	@$(PYTHON) -m $($(AGENT)_RUNS) list

patterns:
	@$(PYTHON) -m design_patterns.coverage $(foreach a,$(AGENTS),--agent $(a)=$($(a)_PKG)) --markdown design_docs/patterns_coverage.md

# -- customer support admin (csa only) ---------------------------------------------
# `make refund-limit 100`: the amount is the second goal, so it is given a
# no-op rule here rather than being run as a target of its own.
ifeq (refund-limit,$(firstword $(MAKECMDGOALS)))
REFUND_LIMIT := $(word 2,$(MAKECMDGOALS))
$(eval $(REFUND_LIMIT):;@:)
endif

CSA_ENV := AGENTIC_OVERLAYS=$(csa_OVERLAY)

show-memory:
	@$(CSA_ENV) $(PYTHON) -m customer_support_agent.admin.show_memory $(if $(CUSTOMER),--customer $(CUSTOMER)) $(if $(EVENTS),--events $(EVENTS))

dream:
	@$(CSA_ENV) $(PYTHON) -m customer_support_agent.admin.dream $(if $(SHOW),--show-memory)

clean-memory:
	@$(CSA_ENV) $(PYTHON) -m customer_support_agent.admin.clean_memory

refund-limit:
	@test -n "$(REFUND_LIMIT)" || (echo "$(YELLOW)Usage: make refund-limit 100$(NC)" && exit 1)
	$(CSA_ENV) $(PYTHON) -m customer_support_agent.admin.refund_limit $(REFUND_LIMIT)

# -- evals -----------------------------------------------------------------------
# Every target writes its Markdown report under reports/evals/<run_id>/.

eval: _agent
	$(EVALS) run $(EVAL_SELECT)

eval-generate: _agent
	$(EVALS) generate $(EVAL_SELECT)

eval-oracle: _agent
	$(EVALS) oracle $(EVAL_RUN)

eval-security: _agent
	$(EVALS) security $(if $(FRESH),--fresh)

eval-judge: _agent
	$(EVALS) judge $(EVAL_RUN)

eval-deepeval: _agent
	$(EVALS) deepeval $(EVAL_RUN)

eval-report: _agent
	$(EVALS) report $(if $(RUN),--run $(RUN))

eval-diff: _agent
	@test -n "$(A)" -a -n "$(B)" || (echo "$(YELLOW)Usage: make eval-diff <agent> A=<run> B=<run>$(NC)" && exit 1)
	$(EVALS) diff $(A) $(B)

eval-runs: _agent
	$(EVALS) runs

eval-sql: _agent
	$(EVALS) sql $(SQL)

# A fresh 32-byte checkpoint key (durability.encryption). Put the line in .env, or export it,
# to stop the "Checkpoints are stored UNENCRYPTED" warning:  make gen-key >> .env
gen-key:
	@echo "LANGGRAPH_AES_KEY=$$(openssl rand -hex 16)"

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
	./venv/bin/ruff check src/ tests/ design_patterns/ > /dev/null && \
	./venv/bin/ruff format --check src/ tests/ design_patterns/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)2/4 mypy --strict...$(NC)"
	./venv/bin/mypy --strict src/ design_patterns/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)3/4 Pytest with coverage gate (80%)...$(NC)"
	./venv/bin/pytest tests/ --cov=src -q > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)4/4 Composition root imports (one process per agent)...$(NC)"
	$(foreach a,$(AGENTS),AGENTIC_OVERLAYS=$($(a)_OVERLAY) $(PYTHON) -c "from $($(a)_PKG).graph import build_agent; build_agent" || exit 1;) \
	echo "$(GREEN)✓$(NC)"
	@echo "$(GREEN)✓ Verification complete - all gates green$(NC)"
