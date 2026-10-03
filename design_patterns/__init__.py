"""The WF/AG pattern library (LLD §22).

Sixteen reference implementations, independent of the running product, so
a pattern can be read in isolation. Each composes `agent_platform` —
none reimplements a registry, a retriever or a memory store.

`catalogue.py` is the decision record: which pattern, when, and which
episode builds it out in full.
"""

from design_patterns.agentic import (
    AgenticRetrievalPattern,
    EvaluatorOptimizerPattern,
    MemorySystemsPattern,
    OrchestratorSubagentPattern,
    PlanningPattern,
    ReActPattern,
    ReflectionPattern,
    ToolUsePattern,
)
from design_patterns.catalogue import (
    AGENTIC_PATTERNS,
    ALL_PATTERNS,
    WORKFLOW_PATTERNS,
    by_family,
    by_name,
    depth_map,
)
from design_patterns.types import PatternCard, PatternFamily, PatternRun, Step
from design_patterns.workflow import (
    BranchPattern,
    FanOutFanInPattern,
    FixedLoopPattern,
    HumanInTheLoopPattern,
    MapReducePattern,
    RagPipelinePattern,
    RouterPattern,
    SequencePattern,
)

__all__ = [
    "AGENTIC_PATTERNS",
    "ALL_PATTERNS",
    "WORKFLOW_PATTERNS",
    "AgenticRetrievalPattern",
    "BranchPattern",
    "EvaluatorOptimizerPattern",
    "FanOutFanInPattern",
    "FixedLoopPattern",
    "HumanInTheLoopPattern",
    "MapReducePattern",
    "MemorySystemsPattern",
    "OrchestratorSubagentPattern",
    "PatternCard",
    "PatternFamily",
    "PatternRun",
    "PlanningPattern",
    "RagPipelinePattern",
    "ReActPattern",
    "ReflectionPattern",
    "RouterPattern",
    "SequencePattern",
    "Step",
    "ToolUsePattern",
    "by_family",
    "by_name",
    "depth_map",
]
