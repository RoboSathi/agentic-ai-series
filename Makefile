.PHONY: help install clean test lint type-check format check-all run dummy-data logs llm-check \
	tui verify episode episodes patterns venv-check test-verbose test-fast

# Which agent this repo runs. The platform deliberately does not name one:
# `agentic_commons/config.yaml` ships `overlays: []`, so the agent layer is
# named here, where the decision about what to run actually lives. Change
# this line to run a different agent; change nothing in the platform.
export AGENTIC_OVERLAYS := src/customer_support_agent/config/config.yaml

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help:
	@echo "$(BLUE)Agentic Commons - Phase 0 Walking Skeleton$(NC)"
	@echo ""
	@echo "$(GREEN)Setup & Environment:$(NC)"
	@echo "  make install          Install dependencies (creates venv)"
	@echo "  make clean            Remove venv, cache, and logs"
	@echo ""
	@echo "$(GREEN)Code Quality:$(NC)"
	@echo "  make lint             Run ruff linter"
	@echo "  make type-check       Run mypy type checker"
	@echo "  make format           Auto-format code with ruff"
	@echo "  make check-all        Run lint + type-check + test (full verification)"
	@echo ""
	@echo "$(GREEN)Testing:$(NC)"
	@echo "  make test             Run pytest with coverage"
	@echo "  make test-verbose     Run pytest with verbose output"
	@echo "  make test-fast        Run pytest without coverage"
	@echo ""
	@echo "$(GREEN)Running:$(NC)"
	@echo "  make dummy-data       Generate dummy_data/*.sql and build dummy_data/agentic.db"
	@echo "  make run              Run one demo query through the agent"
	@echo "  make tui              Run the terminal UI (interactive)"
	@echo "  make episode E=05     Run one episode"
	@echo "  make episodes         Run every episode"
	@echo "  make patterns         Test the design-pattern library"
	@echo "  make llm-check        Verify the configured LLM endpoint is up and responsive"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make logs             Show recent log files"
	@echo "  make venv-check       Check if venv is active"
	@echo ""

install:
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	python3 -m venv venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	./venv/bin/pip install -q --upgrade pip
	./venv/bin/pip install -q -e .
	@echo "$(GREEN)✓ Setup complete. Activate with: source venv/bin/activate$(NC)"

clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf venv
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf .mypy_cache
	rm -rf dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf logs/*.log
	@echo "$(GREEN)✓ Clean complete$(NC)"

lint:
	@echo "$(BLUE)Running ruff lint...$(NC)"
	./venv/bin/ruff check src/ tests/ design_patterns/ episodes/
	@echo "$(GREEN)✓ Lint passed$(NC)"

type-check:
	@echo "$(BLUE)Running mypy type check...$(NC)"
	./venv/bin/mypy --strict src/ design_patterns/ episodes/
	@echo "$(GREEN)✓ Type check passed$(NC)"

format:
	@echo "$(BLUE)Formatting code with ruff...$(NC)"
	./venv/bin/ruff format src/ tests/ design_patterns/ episodes/
	./venv/bin/ruff check src/ tests/ design_patterns/ episodes/ --fix
	@echo "$(GREEN)✓ Format complete$(NC)"

test:
	@echo "$(BLUE)Running pytest with coverage...$(NC)"
	./venv/bin/pytest tests/ --cov=src --cov-report=term-missing -q
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-verbose:
	@echo "$(BLUE)Running pytest (verbose)...$(NC)"
	./venv/bin/pytest tests/ -v

test-fast:
	@echo "$(BLUE)Running pytest (no coverage)...$(NC)"
	./venv/bin/pytest tests/ -q

check-all: lint test
	@echo "$(GREEN)✓ All checks passed$(NC)"

run:
	@echo "$(BLUE)Running customer support agent demo query...$(NC)"
	./venv/bin/python -m customer_support_agent

dummy-data:
	@echo "$(BLUE)Generating dummy_data/seed_data.sql and building dummy_data/agentic.db...$(NC)"
	./venv/bin/python -m customer_support_agent.dummy_data.generate_dummy_data

tui:
	@echo "$(BLUE)Launching terminal UI...$(NC)"
	./venv/bin/python -m agentic_commons.tui

llm-check:
	@echo "$(BLUE)Checking LLM endpoint connectivity...$(NC)"
	./venv/bin/python -m agentic_commons.llm.cli

logs:
	@echo "$(BLUE)Recent log files:$(NC)"
	@ls -lh logs/*.log 2>/dev/null | tail -5 || echo "No logs yet"
	@echo ""
	@echo "$(BLUE)Sample log content (last 10 lines of latest):$(NC)"
	@tail -10 logs/*.log 2>/dev/null | head -20 || echo "No logs to display"

venv-check:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "$(YELLOW)⚠ Virtual environment not active$(NC)"; \
		echo "$(BLUE)Activate with: source venv/bin/activate$(NC)"; \
	else \
		echo "$(GREEN)✓ Virtual environment active: $$VIRTUAL_ENV$(NC)"; \
		$$VIRTUAL_ENV/bin/python --version; \
	fi

# Run one episode: make episode E=05  (or E=48a)
episode:
	@test -n "$(E)" || (echo "$(YELLOW)Usage: make episode E=05$(NC)" && exit 1)
	./venv/bin/python -m episodes.e$(E)

# Run every episode in order — the whole series, end to end.
#
# Configured episodes call the live model, so this is one run of the
# series and not a loop to leave going: on a free-tier account it will
# rate-limit. It is deliberately not part of `verify`, which is scripted
# and offline. Use it before a recording session, not as a check.
episodes:
	@echo "$(YELLOW)⚠ Configured episodes call the live model — this costs quota.$(NC)"
	@echo "$(BLUE)Running every episode...$(NC)"
	@for dir in episodes/e*/; do \
		name=$$(basename $$dir); \
		./venv/bin/python -m episodes.$$name || exit 1; \
	done
	@echo "$(GREEN)✓ All episodes ran$(NC)"

# Run one design pattern's tests.
patterns:
	./venv/bin/pytest tests/design_patterns -q

# Quick verification (for CI/CD).
#
# The coverage gate is enforced by pytest-cov's own `fail_under = 80`
# (pyproject.toml), so a shortfall fails the command itself. An earlier
# version grepped the report for the string "80", which matched any line
# containing those digits and therefore passed at any coverage at all.
verify:
	@echo "$(BLUE)Running verification...$(NC)"
	@echo "$(BLUE)1/4 Ruff lint + format...$(NC)"
	./venv/bin/ruff check src/ tests/ design_patterns/ episodes/ > /dev/null && \
	./venv/bin/ruff format --check src/ tests/ design_patterns/ episodes/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)2/4 mypy --strict...$(NC)"
	./venv/bin/mypy --strict src/ design_patterns/ episodes/ > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)3/4 Pytest with coverage gate (80%)...$(NC)"
	./venv/bin/pytest tests/ --cov=src -q > /dev/null && echo "$(GREEN)✓$(NC)" || exit 1
	@echo "$(BLUE)4/4 Composition root imports...$(NC)"
	./venv/bin/python -c "from customer_support_agent.graph import build_agent; build_agent" \
	  && echo "$(GREEN)✓$(NC)" || exit 1
	@echo ""
	@echo "$(GREEN)✓ Verification complete - all gates green$(NC)"
