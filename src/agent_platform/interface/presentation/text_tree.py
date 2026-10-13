"""Plain-text trees for human-readable command output.

One renderer for every command that shows nested data — `make
show-memory`, `/memory`, `make dream`, `make clean-memory` — so they all
read the same. Plain lines rather than a Rich `Tree`: a `CommandResult`
is data, and the same lines have to print from `make` and draw in the
TUI feed unchanged.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any

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


# -- any value as a tree: what logs and cards show instead of JSON ------------------------

# Value-tree glyphs, narrower than the command trees' so deep payloads stay on screen.
VALUE_BRANCH = "├─ "
VALUE_LAST = "└─ "
VALUE_PIPE = "│  "
VALUE_GAP = "   "
# Long values are folded, never cut: the full payload is the point of a log.
WRAP_AT = 100
# A model's JSON often arrives fenced: ```json ... ```.
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def value_lines(value: Any, prefix: str = "") -> list[str]:
    """Any value as readable lines: a mapping or list (or JSON text of one) as a
    `key: value` tree, anything else as its own wrapped lines. Never JSON punctuation."""
    structured = as_structure(value)
    if isinstance(structured, dict):
        return _branches(list(structured.items()), prefix, labelled=True)
    if isinstance(structured, list):
        return _branches(list(enumerate(structured)), prefix, labelled=False)
    return [f"{prefix}{line}" for line in scalar_lines(value)]


def as_structure(value: Any) -> Any:
    """A mapping or list as itself; JSON text of one, parsed; anything else unchanged."""
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith(("{", "[")):
        return value
    try:
        parsed = json.loads(text)
    except ValueError:
        return value
    return parsed if isinstance(parsed, (dict, list)) else value


def scalar_lines(value: Any) -> list[str]:
    """A leaf value as one or more lines, never elided (embedded newlines honoured)."""
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    if isinstance(value, (dict, list)):
        return ["{}" if isinstance(value, dict) else "[]"]
    lines: list[str] = []
    for raw in str(value).splitlines() or [""]:
        lines.extend(raw[i : i + WRAP_AT] for i in range(0, max(len(raw), 1), WRAP_AT))
    return lines or [""]


def _branches(items: list[Any], prefix: str, labelled: bool) -> list[str]:
    lines: list[str] = []
    for index, (key, raw) in enumerate(items):
        last = index == len(items) - 1
        connector = VALUE_LAST if last else VALUE_BRANCH
        label = f"{key}" if labelled else f"[{key}]"
        below = prefix + (VALUE_GAP if last else VALUE_PIPE)
        value = as_structure(raw)
        if isinstance(value, (dict, list)) and value:
            lines.append(f"{prefix}{connector}{label}:")
            lines.extend(value_lines(value, below))
            continue
        rendered = scalar_lines(value)
        lines.append(f"{prefix}{connector}{label}: {rendered[0]}")
        lines.extend(f"{below}{line}" for line in rendered[1:])
    return lines


# -- messages and tool calls, as a person reads them ---------------------------------------

# How a tool call reads inside a message: `→ tool(key=value, …)`.
_CALL = "  → {tool}({arguments})"
_INDENT = "  "


def inline_arguments(arguments: Any) -> str:
    """A tool call's arguments on one line, `key=value, key=value` — never JSON punctuation."""
    structured = as_structure(arguments)
    if isinstance(structured, dict):
        return ", ".join(f"{key}={_inline(value)}" for key, value in structured.items())
    return _inline(structured) if structured not in (None, "") else ""


def readable_lines(text: Any) -> list[str]:
    """Text as lines a person reads: JSON text of a mapping or list (a tool's result) as a
    `key: value` tree, anything else as the lines it was written in."""
    structured = as_structure(text)
    if isinstance(structured, (dict, list)) and structured:
        return value_lines(structured)
    return str(text).splitlines() or [""]


def message_lines(head: str, content: Any, calls: list[tuple[str, Any]]) -> list[str]:
    """One chat message: `head: text` (structured content as a tree below `head:`), then each
    tool call it made as `→ tool(key=value, …)`."""
    body = readable_lines(content) if content else []
    structured = isinstance(as_structure(content), (dict, list)) and bool(body)
    if structured:
        lines = [f"{head}:", *(f"{_INDENT}{line}" for line in body)]
    else:
        first, *rest = body or [""]
        lines = [f"{head}: {first}" if first else f"{head}:", *rest]
    lines.extend(_CALL.format(tool=tool, arguments=inline_arguments(args)) for tool, args in calls)
    return lines


def _inline(value: Any) -> str:
    """A value on one line: a nested mapping or list in brackets, text with its breaks folded."""
    structured = as_structure(value)
    if isinstance(structured, dict):
        return "{" + inline_arguments(structured) + "}"
    if isinstance(structured, list):
        return "[" + ", ".join(_inline(item) for item in structured) + "]"
    return " ".join(scalar_lines(structured))
