"""The eval framework: three graders over an agent's suites (LLD §18).

- `oracle/`    — the agent's hard rules over the outcome, zero model calls (hard gate);
- `deepeval/`  — conversation quality, judged by the platform judge (soft gate);
- `promptfoo/` — the red-team suite, run by Promptfoo (hard gate);
- `intent/`    — the intent classifier bake-off;
- `core/`      — contracts, store, pipeline, reports, CLI.

An agent takes part by naming an `EvalPlugin` in its config (`evals.plugin`).
"""

from agent_platform.evals.core.contracts import (
    AgentUnderTest,
    ConversationFrame,
    EvalPlugin,
    GoldenCase,
    Grader,
    IntentBakeoffPlugin,
    QualityMetricSpec,
    Score,
    ToolCallRecord,
    Transcript,
    UsageRecord,
)

__all__ = [
    "AgentUnderTest",
    "ConversationFrame",
    "EvalPlugin",
    "GoldenCase",
    "Grader",
    "IntentBakeoffPlugin",
    "QualityMetricSpec",
    "Score",
    "ToolCallRecord",
    "Transcript",
    "UsageRecord",
]
