"""Plain-text renderers shared by `make` output and the TUI."""

from agent_platform.interface.presentation.text_table import Align, box_table
from agent_platform.interface.presentation.text_tree import (
    TreeNode,
    as_structure,
    scalar_lines,
    tree_lines,
    value_lines,
)

__all__ = [
    "Align",
    "TreeNode",
    "box_table",
    "as_structure",
    "scalar_lines",
    "tree_lines",
    "value_lines",
]
