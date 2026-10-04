"""What a stopped run says, the same for every agent: where, how far it got, what next.

    Stopped before INTENT. Where the run got to:
    ├─ request: refund order 10143
    ├─ customer: #12
    └─ tools called: get_order_status → completed

    Everything done so far is saved. Choose ↻ Resume to continue from INTENT, or ✕ Abort …

The progress tree is the agent's own (`DurableAgent.describe_progress`): only it knows what
its state means. The frame around it — the heading, and the choices on a line of their own —
is the platform's, so a stopped run reads the same whichever agent stopped.
"""

from collections.abc import Mapping
from typing import Any

from agent_platform.config import get_settings
from agent_platform.presentation.text_tree import value_lines


def stage_name(node: str) -> str:
    """A graph node as a stage is named on screen: `assemble_context` -> `ASSEMBLE CONTEXT`."""
    words = get_settings().durability.stop_report
    return node.replace("_", " ").upper() if node else words.the_end


def stop_report(next_node: str, progress: Mapping[str, Any]) -> str:
    """The stopped run's heading, its progress as a tree, then the choices on their own line."""
    words = get_settings().durability.stop_report
    step = stage_name(next_node)
    return "\n".join(
        [
            words.heading.format(step=step),
            *value_lines(dict(progress)),
            "",
            words.choices.format(step=step),
        ]
    )


__all__ = ["stage_name", "stop_report"]
