"""Plain-text renderers shared by `make` output and the TUI."""

from agent_platform.presentation.text_tree import (
    TreeNode,
    as_structure,
    scalar_lines,
    tree_lines,
    value_lines,
)

__all__ = ["TreeNode", "as_structure", "scalar_lines", "tree_lines", "value_lines"]
