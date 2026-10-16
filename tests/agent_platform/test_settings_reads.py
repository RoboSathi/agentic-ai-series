"""Configuration is taken when a component is built, never looked up while it works.

A component that calls `get_settings()` inside a method decides from process-global state
on every call, so two agents in one process could not hold two postures. Each class takes
the live configuration once, in its constructor, and reads that reference after.

Two checks:
- no instance method reads configuration, outside the classes that are themselves
  configuration, values or composition roots (`EXEMPT`);
- the direct reads left in each file (constructors, module helpers, factories) may fall,
  never rise: `data/settings_reads_baseline.json` is today's count. Lower it when a
  migration removes reads.
"""

import ast
import json
from collections import Counter
from pathlib import Path

from agent_platform.config.paths import REPO_ROOT

READERS = frozenset({"get_settings", "get_agent_settings"})
SOURCE = REPO_ROOT / "src"
BASELINE = Path(__file__).parent / "data" / "settings_reads_baseline.json"
# (module, class) allowed to read configuration in a method, and why.
EXEMPT = {
    # Configuration itself.
    ("agent_platform/config/toggles.py", "ConfigFeatureToggle"),
    ("customer_support_agent/config/settings.py", "CustomerSupportAgentSettings"),
    # Values: a pydantic model has no constructor to inject through.
    ("agent_platform/core/primitives/money.py", "Money"),
    ("agent_platform/llm/prompt.py", "PromptManifest"),
    ("agent_platform/analytics/dummy_profile.py", "DummyProfile"),
    ("agent_platform/memory/profile.py", "MemoryProfile"),
    # The logging bootstrap runs before any composition root.
    ("agent_platform/logging/core.py", "LoggerFactory"),
    # Composition roots: a host is where an agent's configuration is read.
    ("customer_support_agent/console/__init__.py", "CustomerSupportHost"),
    ("research_agent/console/__init__.py", "ResearchHost"),
    ("ops_agent/console/__init__.py", "OpsHost"),
}


def _trees() -> dict[str, ast.Module]:
    return {
        str(path.relative_to(SOURCE)): ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(SOURCE.rglob("*.py"))
    }


def _reads(node: ast.AST) -> int:
    return sum(
        1
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in READERS
    )


def _mid_call_reads(module: str, tree: ast.Module) -> list[str]:
    found = []
    for cls in (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)):
        if (module, cls.name) in EXEMPT:
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if method.name == "__init__" or _is_static(method):
                continue
            if _reads(method):
                found.append(f"{module}:{cls.name}.{method.name}")
    return found


def _is_static(method: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(ast.unparse(d) in ("staticmethod", "classmethod") for d in method.decorator_list)


def test_no_component_reads_configuration_mid_call() -> None:
    found = [entry for module, tree in _trees().items() for entry in _mid_call_reads(module, tree)]
    assert found == [], f"read in the constructor instead: {found}"


def test_direct_reads_never_rise() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    counts = Counter({module: _reads(tree) for module, tree in _trees().items()})
    risen = {
        module: (count, baseline.get(module, 0))
        for module, count in counts.items()
        if count > baseline.get(module, 0)
    }
    assert risen == {}, f"more settings reads than the baseline (now, baseline): {risen}"
