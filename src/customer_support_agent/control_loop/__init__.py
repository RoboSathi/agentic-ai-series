"""This agent's two loops, named for what they actually are.

The **Tool-Use Loop** is the `reason ⇄ act` cycle in the graph: the model
proposes a typed tool call, the harness executes it, the observation comes
back, bounded by `max_tool_iterations`. It is deliberately *not* ReAct —
there is no natural-language thought step, and calling it ReAct would
teach the wrong thing. `design_patterns/agentic/react.py` is the pure form.

The **Output Quality Gate** is `quality_gate`. It was called a
"deliberation loop", which oversold it: it looks a plan up in config,
applies one deterministic critique, allows at most one repair, and scores
the result with something independent of the model that wrote it. Single
pass, no iteration. Naming it honestly is the point — a viewer who sees
"loop" and finds a lookup has learned something false.

Generic trajectory watchdogs come from `agent_platform.control_loop`.
"""

from agent_platform.control_loop import (
    ControlCheck,
    ControlLoopReport,
    DuplicateActionDetector,
    ToolControlLoop,
    action_fingerprint,
)
from customer_support_agent.control_loop.quality_gate import (
    Critique,
    Evaluator,
    GroundedCritic,
    IndependentScorer,
    OutputQualityGate,
    Plan,
    PlanLookup,
    Planner,
    PlanStep,
    Reflector,
)

__all__ = [
    "ControlCheck",
    "ControlLoopReport",
    "Critique",
    "DuplicateActionDetector",
    "Evaluator",
    "GroundedCritic",
    "IndependentScorer",
    "OutputQualityGate",
    "Plan",
    "PlanLookup",
    "PlanStep",
    "Planner",
    "Reflector",
    "ToolControlLoop",
    "action_fingerprint",
]
