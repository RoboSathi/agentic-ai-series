"""Plain-text trees for human-readable command output.

One renderer for every command that shows nested data — `make
show-memory`, `/memory`, `make dream`, `make clean-memory` — so they all
read the same. Plain lines rather than a Rich `Tree`: a `CommandResult`
is data, and the same lines have to print from `make` and draw in the
TUI feed unchanged.
"""

from dataclasses import dataclass, field

_BRANCH = "├── "
_LAST = "└── "
_PIPE = "│   "
_SPACE = "    "


@dataclass
class TreeNode:
    """One line of a tree, and the lines nested under it."""

    label: str
    children: list["TreeNode"] = field(default_factory=list)

    def add(self, label: str) -> "TreeNode":
        """Append a child and return it, so branches can be built fluently."""
        child = TreeNode(label)
        self.children.append(child)
        return child


def tree_lines(root: TreeNode) -> list[str]:
    """The root's label, then every descendant with box-drawing connectors."""
    lines = [root.label]
    _walk(root.children, "", lines)
    return lines


def _walk(children: list[TreeNode], prefix: str, lines: list[str]) -> None:
    for index, child in enumerate(children):
        last = index == len(children) - 1
        lines.append(prefix + (_LAST if last else _BRANCH) + child.label)
        _walk(child.children, prefix + (_SPACE if last else _PIPE), lines)
