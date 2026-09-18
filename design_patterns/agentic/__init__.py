"""Agentic patterns: the path is chosen at run time, by a model.

Eight shapes. What they share is that the control flow is not knowable by
reading the code — which buys the ability to handle cases nobody
enumerated, and costs the ability to say in advance what will happen.
That trade is the whole subject, and every pattern here is a different
point on it.

ReAct is the atomic unit. Planning sequences several ReAct cycles;
reflection critiques a completed trajectory; orchestration runs several
in parallel. Reading ReAct first makes the rest short.
"""

from design_patterns.agentic.agentic_retrieval import AgenticRetrievalPattern
from design_patterns.agentic.evaluator_optimizer import EvaluatorOptimizerPattern
from design_patterns.agentic.memory_systems import MemorySystemsPattern
from design_patterns.agentic.orchestrator_subagent import OrchestratorSubagentPattern
from design_patterns.agentic.planning import PlanningPattern
from design_patterns.agentic.react import ReActPattern
from design_patterns.agentic.reflection import ReflectionPattern
from design_patterns.agentic.tool_use import ToolUsePattern

__all__ = [
    "AgenticRetrievalPattern",
    "EvaluatorOptimizerPattern",
    "MemorySystemsPattern",
    "OrchestratorSubagentPattern",
    "PlanningPattern",
    "ReActPattern",
    "ReflectionPattern",
    "ToolUsePattern",
]
