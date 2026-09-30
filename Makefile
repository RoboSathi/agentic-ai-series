.PHONY: help install clean lint type-check format test verify tui episode llm-check dummy-data refund-limit \
	clean-memory show-memory dream \
	eval eval-generate eval-oracle eval-security eval-judge eval-deepeval eval-report eval-diff \
	eval-runs eval-sql

# Which agent this repo runs. The platform deliberately does not name one:
# `agentic_commons/config.yaml` ships `overlays: []`, so the agent layer is
# named here, where the decision about what to run actually lives. Change
# this line to run a different agent; change nothing in the platform.
export AGENTIC_OVERLAYS := src/customer_support_agent/config/config.yaml

PYTHON := ./venv/bin/python
EVALS := $(PYTHON) -m agentic_commons.evals

# Provider budget override for this invocation: `RPM=38 make eval`.
# The configured default lives in resilience.rate_limit (config.yaml).
ifdef RPM
export RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS := $(RPM)
endif
# Keep going past a broken business invariant: `EVAL_CONTINUE_ON_INVARIANT_FAILURE=1 make eval`.
ifdef EVAL_CONTINUE_ON_INVARIANT_FAILURE
export EVALS__GATE__STOP_ON_INVARIANT_FAILURE := false
endif

# Case selection and cache control for the eval targets.
EVAL_SELECT := $(if $(FRESH),--fresh) $(if $(LIMIT),--limit $(LIMIT)) \
	$(foreach c,$(CASE),--case $(c)) $(foreach t,$(TAG),--tag $(t))
EVAL_RUN := $(if $(RUN),--run $(RUN)) $(if $(FRESH),--fresh)

GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m

help:
	@echo "$(BLUE)Agentic Harness$(NC)"
	@echo ""
	@echo "$(GREEN)Setup:$(NC)"
	@echo "  make install          Create the venv and install the package with dev + eval extras"
	@echo "  make clean            Remove venv, caches and logs"
	@echo ""
	@echo "$(GREEN)Quality:$(NC)"
	@echo "  make lint             ruff check"
	@echo "  make type-check       mypy --strict"
	@echo "  make format           ruff format + autofix"
	@echo "  make test             pytest with coverage"
	@echo "  make verify           All gates: lint, format, types, tests (80% coverage), imports"
	@echo ""
	@echo "$(GREEN)Run:$(NC)"
	@echo "  make tui              The operator console"
	@echo "  make episode E=05     One episode (calls the live model)"
	@echo "  make llm-check        Probe every hosted model endpoint (one tiny call each)"
	@echo ""
	@echo "$(GREEN)Data:$(NC)"
	@echo "  make dummy-data       Rebuild the agent DB, reset stale memory (KEEP_MEMORY=1 skips), build the eval fixture"
	@echo "  make refund-limit 100 Set the per-refund limit the Gatekeeper enforces (agent DB)"
	@echo ""
	@echo "$(GREEN)Memory:$(NC)"
	@echo "  make show-memory      Everything long-term memory holds, as a tree (CUSTOMER=<id> EVENTS=<n>)"
	@echo "  make dream            Consolidate recent conversations into memory; prints what changed (SHOW=1 for the full tree)"
	@echo "  make clean-memory     Empty long-term memory (keeps checkpoints, orders, reviews, audit)"
	@echo ""
	@echo "$(GREEN)Evals$(NC) (order: generate → oracle → security → judge → deepeval; each writes a report):"
	@echo "  make eval             The whole pipeline; stops after the oracle on a broken invariant"
	@echo "  make eval-generate    Run the agent over the golden set (reuses unchanged transcripts)"
	@echo "  make eval-oracle      Deterministic business invariants (no model calls)"
	@echo "  make eval-security    Promptfoo attack suite"
	@echo "  make eval-judge       LLM judge: rubric quality + hallucination"
	@echo "  make eval-deepeval    DeepEval quality metrics"
	@echo "  make eval-report      Rebuild the Markdown reports (RUN=<id>, default latest)"
	@echo "  make eval-diff A=<run> B=<run>   Regression report between two runs"
	@echo "  make eval-runs        List recent runs"
	@echo "  make eval-sql         Query the eval store (SQL=\"select ...\" or interactive)"
	@echo ""
	@echo "  Options: RPM=<n>  FRESH=1  LIMIT=<n>  CASE=<id>  TAG=<tag>  RUN=<id>"
	@echo "           EVAL_CONTINUE_ON_INVARIANT_FAILURE=1"

install:
	python3 -m venv venv
	./venv/bin/pip install -q --upgrade pip
	./venv/bin/pip install -q -e ".[dev,evals]"
	@echo "$(GREEN)✓ Installed. Activate with: source venv/bin/activate$(NC)"

clean:
	rm -rf venv .pytest_cache .coverage .mypy_cache dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*.log

lint:
	./venv/bin/ruff check src/ tests/ design_patterns/

type-check:
	./venv/bin/mypy --strict src/ design_patterns/

format:
	./venv/bin/ruff format src/ tests/ design_patterns/
	./venv/bin/ruff check src/ tests/ design_patterns/ --fix

test:
	./venv/bin/pytest tests/ --cov=src --cov-report=term-missing -q

tui:
	$(PYTHON) -m agentic_commons.tui

# Run one episode: make episode E=05  (or E=48a). Configured episodes call
# the live model — this is a recording-session tool, not a check.
episode:
	@test -n "$(E)" || (echo "$(YELLOW)Usage: make episode E=05$(NC)" && exit 1)
	$(PYTHON) -m episodes.e$(E)

llm-check:
	$(PYTHON) -m agentic_commons.llm.cli

dummy-data:
	$(PYTHON) -m customer_support_agent.dummy_data.generate_dummy_data $(if $(KEEP_MEMORY),--keep-memory)

# `make refund-limit 100`: the amount is the second goal, so it is given a
# no-op rule here rather than being run as a target of its own.
ifeq (refund-limit,$(firstword $(MAKECMDGOALS)))
REFUND_LIMIT := $(word 2,$(MAKECMDGOALS))
$(eval $(REFUND_LIMIT):;@:)
endif

show-memory:
	@$(PYTHON) -m customer_support_agent.admin.show_memory $(if $(CUSTOMER),--customer $(CUSTOMER)) $(if $(EVENTS),--events $(EVENTS))

dream:
	@$(PYTHON) -m customer_support_agent.admin.dream $(if $(SHOW),--show-memory)

clean-memory:
	@$(PYTHON) -m customer_support_agent.admin.clean_memory

refund-limit:
	@test -n "$(REFUND_LIMIT)" || (echo "$(YELLOW)Usage: make refund-limit 100$(NC)" && exit 1)
	$(PYTHON) -m customer_support_agent.admin.refund_limit $(REFUND_LIMIT)

# -- evals ---------------------------------------------------------------------
# Every target writes its Markdown report under reports/evals/<run_id>/.

eval:
	$(EVALS) run $(EVAL_SELECT)

eval-generate:
	$(EVALS) generate $(EVAL_SELECT)

eval-oracle:
	$(EVALS) oracle $(EVAL_RUN)

eval-security:
	$(EVALS) security $(if $(FRESH),--fresh)

eval-judge:
	$(EVALS) judge $(EVAL_RUN)

eval-deepeval:
	$(EVALS) deepeval $(EVAL_RUN)

eval-report:
	$(EVALS) report $(if $(RUN),--run $(RUN))

eval-diff:
	@test -n "$(A)" -a -n "$(B)" || (echo "$(YELLOW)Usage: make eval-diff A=<run> B=<run>$(NC)" && exit 1)
	$(EVALS) diff $(A) $(B)

eval-runs:
	$(EVALS) runs

eval-sql:
	$(EVALS) sql $(SQL)

# Quick verification (for CI/CD).
#
# The coverage gate is enforced by pytest-cov's own `fail_under = 80`
# (pyproject.toml), so a shortfall fails the command itself. An earlier
# version grepped the report for the string "80", which matched any line
# containing those digits and therefore passed at any coverage at all.
verify:
	@echo "$(BLUE)1/4 Ruff lint + format...$(NC)"
	./venv/bin/ruff check src/ tests/ design_patterns/ > /dev/null && \
	./venv/bin/ruff format --check src/ tests/ design_patterns/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)2/4 mypy --strict...$(NC)"
	./venv/bin/mypy --strict src/ design_patterns/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)3/4 Pytest with coverage gate (80%)...$(NC)"
	./venv/bin/pytest tests/ --cov=src -q > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)4/4 Composition root imports...$(NC)"
	$(PYTHON) -c "from customer_support_agent.graph import build_agent; build_agent" \
	  && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(GREEN)✓ Verification complete - all gates green$(NC)"
