# Phase 0 Walking Skeleton - Quick Start Guide

## Overview

This is the **walking skeleton** for the Agentic AI Systems Masterclass. All architectural modules are in place with stub implementations, call logging, and full test coverage (80%+).

## Quick Setup (2 steps)

```bash
# 1. Set up environment
make install

# 2. Activate virtual environment
source venv/bin/activate
```

That's it! The venv is created and all dependencies are installed.

## Verify Everything Works

```bash
# Run the complete verification suite
make verify
```

This runs:
- ✅ Ruff lint check
- ✅ Pytest (74 tests, 80.08% coverage)
- ✅ Coverage gate verification
- ✅ Import and module checks

**Expected output:**
```
Running Phase 0 verification...
1/4 Ruff lint...✓
2/4 Pytest...✓
3/4 Coverage gate (80%)...✓
4/4 Import check...✓

✓ Phase 0 verification complete - All systems operational
```

## Common Commands

### Testing
```bash
make test              # Run all tests with coverage report
make test-verbose      # Run tests with detailed output
make test-fast         # Run tests without coverage (quicker)
```

### Code Quality
```bash
make lint              # Check code style (ruff)
make type-check        # Check types (mypy) 
make format            # Auto-format code
make check-all         # Run lint + tests (full verification)
```

### Running the Project
```bash
make run               # Run the walking skeleton
make tui               # Launch interactive terminal UI
make logs              # Show recent log files
```

### Maintenance
```bash
make clean             # Remove venv, caches, logs
make help              # Show all available commands
make venv-check        # Verify venv is active
```

## What's Working

### ✅ Core Infrastructure
- **Config System**: `agentic_commons/config/`
  - Pydantic settings with env var support
  - Feature toggles for all subsystems
- **Logging**: `agentic_commons/logging/`
  - Rich-based console logging with colors
  - File logging with timestamps
  - Call tracing decorator (@log_call)
- **TUI**: `agentic_commons/tui/`
  - Textual-based terminal UI
  - Live log panel streaming

### ✅ All 20+ Modules Scaffolded
- **LLM**: Client stubs for NIM and local providers
- **Persistence**: Database interface
- **Observability**: Tracing spans
- **Audit**: Event logging
- **Evals**: Test suite runner
- **Guardrails**: Pydantic + Outlines guards
- **Resilience**: 7 patterns (circuit breaker, retry, timeout, saga, idempotency, bulkhead, pub-sub)
- **Security**: Auth, PII detection, prompt injection, content moderation
- **Memory**: Working, episodic, semantic, procedural
- **Tools**: Tool registry
- **Gatekeeper**: Authorization checks
- **Hallucination Protection**: Claim verification

### ✅ Customer Support Agent
- 10-node LangGraph pipeline
- Full integration with all subsystems
- End-to-end tracing

### ✅ Tests & Quality
- 74 unit + integration tests
- 80.08% code coverage (exceeds 80% gate)
- Ruff lint passes all checks
- Type hints throughout

## File Structure

```
src/
├── agentic_commons/          # Shared platform library
│   ├── config/               # Settings & toggles
│   ├── logging/              # Rich logging + @log_call decorator
│   ├── tui/                  # Terminal UI
│   ├── llm/                  # LLM adapters
│   ├── persistence/          # Database interface
│   ├── observability/        # Tracing
│   ├── audit/                # Audit trail
│   ├── evals/                # Evaluation runner
│   ├── guardrails/           # Validation guards
│   ├── resilience/           # 7 resilience patterns
│   ├── security/             # 4 security subsystems
│   ├── memory/               # 4 memory types
│   ├── tools/                # Tool registry
│   ├── gatekeeper/           # Authorization
│   └── hallucination_protection/  # Claim verification
└── customer_support_agent/   # Example application
    ├── core/graph.py         # 10-node LangGraph
    └── dev/walking_skeleton.py  # Entry point

tests/
├── unit/                     # 50+ unit tests
└── integration/              # Integration tests
```

## Understanding the Trace Output

When you run the walking skeleton, you'll see colored logs:

```
INFO     ═══ WALKING SKELETON START ═══
INFO     Intake node
INFO     CALLED WorkingMemory.get(...)
INFO     RETURNED WorkingMemory.get → Context(...) (0.5ms)
INFO     Security check node
```

Each module is traced with:
- **[CALLED]** - Function invoked
- **[RETURNED]** - Result returned with timing
- **[SKIPPED]** - Disabled modules (toggle=False)

Colors indicate subsystem:
- 🔵 **bright_blue** = LLM
- 🟦 **dodger_blue1** = Memory
- 🟩 **green** = Audit
- 🔴 **red** = Security
- 🟨 **yellow** = Resilience

## Toggling Subsystems On/Off

Create a `.env` file to control which subsystems are active:

```bash
# Enable/disable by subsystem
LLM__ENABLED=true
SECURITY__PII__ENABLED=false
MEMORY__ENABLED=true
```

Disabled modules will log `[SKIPPED]` instead of executing.

## Next Steps (Phase 1+)

Phase 0 is complete. Phase 1 will:
1. Implement real LLM calls (NIM + local server)
2. Build real persistence (database queries)
3. Implement real memory storage
4. Add observability (LangSmith tracing)
5. Build TUI panels for live inspection
6. Fix LangGraph state management

For now, the walking skeleton proves the architecture is sound and every module is callable.

## Troubleshooting

**"Module not found" error?**
- Ensure venv is activated: `source venv/bin/activate`
- Reinstall: `make clean && make install`

**Tests fail?**
- Check Python version: `python --version` (need 3.11+)
- Run `make verify` for diagnostic output

**Want to see logs?**
- Check `logs/` directory: `make logs`
- Run individual modules: `python -c "from agentic_commons.llm import LLMClient; ..."`

**TUI doesn't launch?**
- Make sure Textual is installed: `pip list | grep textual`
- Terminal must support colors and Unicode

## Questions?

Refer to:
- `CLAUDE.md` - Engineering standards & philosophy
- Individual module docstrings - API documentation
- `tests/` - Usage examples
