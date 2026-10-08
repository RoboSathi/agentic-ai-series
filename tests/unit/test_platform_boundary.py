"""The platform must not know which agent is running on it.

`agent_platform` is the shared library every agent composes. The moment
it names one of them, a second agent cannot be built without editing the
platform — which is the coupling these tests exist to keep out, and which
was real: the TUI and half the console commands imported the refund agent
directly.

The parked-package tests here are the other half of the same rule. A
module boundary that exists but has no import test is a folder, not a
boundary.
"""

import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
COMMONS = SRC / "agent_platform"

# Agents this platform must be able to run without knowing their names:
# every package beside it, found rather than listed, so a new agent is
# covered the day it is added.
FORBIDDEN_NAMES = tuple(
    sorted(
        path.parent.name for path in SRC.glob("*/__init__.py") if path.parent.name != COMMONS.name
    )
)
# Words that belong to one agent's business. The platform ships mechanisms;
# a default that says "refund" has decided what every agent does.
DOMAIN_WORDS = ("refund",)


def _python_files() -> list[pathlib.Path]:
    return sorted(p for p in COMMONS.rglob("*.py") if "__pycache__" not in p.parts)


def _code_strings(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every identifier and non-docstring string constant in a module, with its line."""
    import ast

    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                found.append((node.lineno, node.value))
        elif isinstance(node, ast.Name):
            found.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            found.append((node.lineno, node.attr))
    return found


def _yaml_strings(value: object) -> list[str]:
    """Every string key and value in a parsed YAML tree."""
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in [str(key), *_yaml_strings(item)]]
    if isinstance(value, list):
        return [s for item in value for s in _yaml_strings(item)]
    return [value] if isinstance(value, str) else []


class TestTheePlatformNamesNoAgent:
    """`agent_platform` composes agents; it never imports one."""

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_module_imports_a_specific_agent(self, name: str) -> None:
        offenders = [
            f"{path.relative_to(COMMONS)}:{number}"
            for path in _python_files()
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if f"import {name}" in line or f"from {name}" in line
        ]
        assert offenders == [], (
            f"agent_platform imports '{name}' at: {', '.join(offenders)}. "
            "An agent reaches the platform by injection, never the other way round."
        )

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_code_names_a_specific_agent(self, name: str) -> None:
        """Not in an identifier or a string the code uses; a docstring may teach by example."""
        offenders = [
            f"{path.relative_to(COMMONS)}:{line}"
            for path in _python_files()
            for line, text in _code_strings(path)
            if name in text
        ]
        assert offenders == [], f"agent_platform code names '{name}' at: {', '.join(offenders)}"

    @pytest.mark.parametrize("word", DOMAIN_WORDS)
    def test_no_platform_default_speaks_an_agents_business(self, word: str) -> None:
        """Comments may explain; a configured value is behaviour, and the agent's to set."""
        import yaml

        offenders = [
            f"{path.relative_to(COMMONS)}: {value}"
            for path in sorted(COMMONS.rglob("*.yaml"))
            for value in _yaml_strings(yaml.safe_load(path.read_text()))
            if word in value.casefold()
        ]
        assert offenders == [], f"platform defaults say '{word}': {offenders}"

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_config_file_names_a_specific_agent(self, name: str) -> None:
        offenders = [
            str(path.relative_to(COMMONS))
            for path in sorted(COMMONS.rglob("*.yaml"))
            if name in path.read_text()
        ]
        assert offenders == [], (
            f"agent_platform config names '{name}' in: {', '.join(offenders)}. "
            "An agent's overlay declares the agent; the platform default does not."
        )

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_document_or_template_names_a_specific_agent(self, name: str) -> None:
        """READMEs and report templates ship with the platform too."""
        offenders = [
            str(path.relative_to(COMMONS))
            for path in sorted(COMMONS.rglob("*.md"))
            if name in path.read_text()
        ]
        assert offenders == [], f"agent_platform documents name '{name}' in: {offenders}"


# Offline eval tools: optional extras, never on an agent's request path.
OPTIONAL_EVAL_TOOLS = ("deepeval", "promptfoo")
EVAL_ADAPTERS = COMMONS / "evals" / "adapters"


class TestEvalToolsAreNotRuntimeDependencies:
    """DeepEval and Promptfoo stay behind the eval adapters, imported lazily."""

    @pytest.mark.parametrize("tool", OPTIONAL_EVAL_TOOLS)
    def test_only_the_adapters_import_an_eval_tool(self, tool: str) -> None:
        agents = COMMONS.parent
        offenders = [
            str(path)
            for path in sorted(agents.rglob("*.py"))
            if "__pycache__" not in path.parts
            and EVAL_ADAPTERS not in path.parents
            and any(
                line.strip().startswith((f"import {tool}", f"from {tool}"))
                for line in path.read_text().splitlines()
            )
        ]
        assert offenders == [], f"{tool} imported outside the eval adapters: {offenders}"

    def test_the_agent_imports_without_them(self) -> None:
        import subprocess
        import sys

        probe = (
            "import sys; sys.modules['deepeval'] = None; "
            "import customer_support_agent.graph, agent_platform.evals, agent_platform.tui.app"
        )
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


class TestParkedPackagesAreRealBoundaries:
    """A parked package still has to import, or it is not a boundary."""

    def test_mcp_exports_its_contracts_and_their_implementations(self) -> None:
        import agent_platform.mcp as mcp

        contracts = {
            "McpServerSpec",
            "McpSession",
            "McpToolInfo",
            "McpToolSource",
            "McpTransport",
            "ToolGovernancePolicy",
            "TransportKind",
        }
        implementations = {"SdkMcpTransport", "SdkMcpSession", "GovernedMcpToolSource"}
        assert contracts | implementations <= set(mcp.__all__)

    def test_the_mcp_implementations_satisfy_their_contracts(self) -> None:
        from agent_platform.mcp import (
            ConfiguredGovernancePolicy,
            GovernedMcpToolSource,
            McpSession,
            McpToolSource,
            McpTransport,
            SdkMcpSession,
            SdkMcpTransport,
            ToolGovernancePolicy,
        )

        def public(kind: type) -> set[str]:
            return {name for name in dir(kind) if not name.startswith("_")}

        assert public(McpSession) <= public(SdkMcpSession)
        assert public(McpTransport) <= public(SdkMcpTransport)
        assert public(McpToolSource) <= public(GovernedMcpToolSource)
        assert isinstance(ConfiguredGovernancePolicy(), ToolGovernancePolicy)

    def test_an_mcp_tool_source_is_an_ordinary_tool_source(self) -> None:
        """Remote tools enter by the same door as local ones.

        If MCP had its own path to the executor, it would have its own
        path around the Gatekeeper. It does not: an `McpToolSource` is
        structurally a `tools.ToolSource` and yields plain
        `ToolDefinition`s.
        """
        from agent_platform.mcp import McpToolSource
        from agent_platform.tools import ToolSource

        assert set(dir(ToolSource)) <= set(dir(McpToolSource))

    def test_multi_agent_is_still_deliberately_empty(self) -> None:
        import agent_platform.multi_agent as multi_agent

        assert multi_agent.__all__ == []

    def test_sandbox_and_prompt_compilation_import(self) -> None:
        import agent_platform.prompt_compilation as prompt_compilation
        import agent_platform.sandbox as sandbox

        assert sandbox.__all__ and prompt_compilation.__all__
