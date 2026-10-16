"""Every settings field has a reader: configuration surface is never added ahead of its code.

A field counts as read when code outside its own settings module names it (`.field`, or
`"field"` for the keyed and `getattr` readers), or when a helper in its own module reads it
and that helper is itself used. A name shared by several models passes on any of them, so
this catches the dead field, not every dead one; it is a floor, not a proof.
"""

import ast
import importlib
import inspect
import re
from pathlib import Path

from pydantic import BaseModel

from agent_platform.config.paths import REPO_ROOT

SETTINGS_MODULES = (
    "agent_platform.config.settings",
    "customer_support_agent.config.settings",
    "research_agent.config.settings",
    "ops_agent.config.settings",
)


def _sources() -> dict[Path, str]:
    return {
        path.resolve(): path.read_text(encoding="utf-8")
        for path in (REPO_ROOT / "src").rglob("*.py")
    }


# A name a module reads: after a dot, or as a string (a key, a getattr, a dotted path).
_READ_NAME = re.compile(r"\.([A-Za-z_]\w*)|[\"']([A-Za-z_]\w*)[\"']")


def _names_read(text: str) -> set[str]:
    return {dotted or quoted for dotted, quoted in _READ_NAME.findall(text)}


def _unread_fields() -> tuple[list[str], int]:
    sources, unread, total = _sources(), [], 0
    read_in = {path: _names_read(text) for path, text in sources.items()}
    for name in SETTINGS_MODULES:
        module = importlib.import_module(name)
        own_path = Path(inspect.getfile(module)).resolve()
        own = sources[own_path]
        elsewhere = set().union(*(names for path, names in read_in.items() if path != own_path))
        for node in ast.parse(own).body:
            model = getattr(module, getattr(node, "name", ""), None)
            if not (isinstance(node, ast.ClassDef) and inspect.isclass(model)):
                continue
            if not issubclass(model, BaseModel):
                continue
            rest = _names_read(own.replace(ast.get_source_segment(own, node) or "", ""))
            for field in _fields(node):
                total += 1
                if not _is_read(field, node, own, elsewhere | rest):
                    unread.append(f"{name}:{node.name}.{field}")
    return unread, total


def _fields(node: ast.ClassDef) -> list[str]:
    return [
        item.target.id
        for item in node.body
        if isinstance(item, ast.AnnAssign)
        and isinstance(item.target, ast.Name)
        and not item.target.id.startswith("_")
        and item.target.id != "model_config"
    ]


def _is_read(field: str, node: ast.ClassDef, own: str, read: set[str]) -> bool:
    """Named by another module or another class here, or by a helper of its own that is."""
    if field in read:
        return True
    helpers = [
        item.name
        for item in node.body
        if isinstance(item, ast.FunctionDef)
        and re.search(rf"self\.{field}\b", ast.get_source_segment(own, item) or "")
    ]
    return any(helper in read for helper in helpers)


def test_every_settings_field_has_a_reader() -> None:
    unread, total = _unread_fields()
    assert unread == [], f"{len(unread)} of {total} settings fields are never read: {unread}"
