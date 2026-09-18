"""Workflow patterns: the path is fixed at design time, by a person.

Eight shapes. What they share is that a human decided the control flow
before the system ran, so the set of things that can happen is knowable
by reading the code. That is the property the agentic patterns trade
away, and it is worth naming what is being given up.
"""

from design_patterns.workflow.branch import BranchPattern
from design_patterns.workflow.fan_out_fan_in import FanOutFanInPattern
from design_patterns.workflow.fixed_loop import FixedLoopPattern
from design_patterns.workflow.human_in_the_loop import HumanInTheLoopPattern
from design_patterns.workflow.map_reduce import MapReducePattern
from design_patterns.workflow.rag_pipeline import RagPipelinePattern
from design_patterns.workflow.router import RouterPattern
from design_patterns.workflow.sequence import SequencePattern

__all__ = [
    "BranchPattern",
    "FanOutFanInPattern",
    "FixedLoopPattern",
    "HumanInTheLoopPattern",
    "MapReducePattern",
    "RagPipelinePattern",
    "RouterPattern",
    "SequencePattern",
]
