"""The platform must not know which agent is running on it.

`agentic_commons` is the shared library every agent composes. The moment
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

COMMONS = pathlib.Path(__file__).resolve().parents[2] / "src" / "agentic_commons"

# Agents this platform must be able to run without knowing their names.
FORBIDDEN_NAMES = ("customer_support_agent",)


def _python_files() -> list[pathlib.Path]:
    return sorted(p for p in COMMONS.rglob("*.py") if "__pycache__" not in p.parts)


class TestTheePlatformNamesNoAgent:
    """`agentic_commons` composes agents; it never imports one."""

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_module_imports_a_specific_agent(self, name: str) -> None:
        offenders = [
            f"{path.relative_to(COMMONS)}:{number}"
            for path in _python_files()
            for number, line in enumerate(path.read_text().splitlines(), start=1)
            if f"import {name}" in line or f"from {name}" in line
        ]
        assert offenders == [], (
            f"agentic_commons imports '{name}' at: {', '.join(offenders)}. "
            "An agent reaches the platform by injection, never the other way round."
        )

    @pytest.mark.parametrize("name", FORBIDDEN_NAMES)
    def test_no_config_file_names_a_specific_agent(self, name: str) -> None:
        offenders = [
            str(path.relative_to(COMMONS))
            for path in sorted(COMMONS.rglob("*.yaml"))
            if name in path.read_text()
        ]
        assert offenders == [], (
            f"agentic_commons config names '{name}' in: {', '.join(offenders)}. "
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
        assert offenders == [], f"agentic_commons documents name '{name}' in: {offenders}"


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
            "import customer_support_agent.graph, agentic_commons.evals, agentic_commons.tui.app"
        )
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


class TestParkedPackagesAreRealBoundaries:
    """A parked package still has to import, or it is not a boundary."""

    def test_mcp_imports_and_exports_its_contracts(self) -> None:
        import agentic_commons.mcp as mcp

        assert set(mcp.__all__) == {
            "McpServerSpec",
            "McpSession",
            "McpToolInfo",
            "McpToolSource",
            "McpTransport",
            "ToolGovernancePolicy",
            "TransportKind",
        }

    def test_an_mcp_tool_source_is_an_ordinary_tool_source(self) -> None:
        """Remote tools enter by the same door as local ones.

        If MCP had its own path to the executor, it would have its own
        path around the Gatekeeper. It does not: an `McpToolSource` is
        structurally a `tools.ToolSource` and yields plain
        `ToolDefinition`s.
        """
        from agentic_commons.mcp import McpToolSource
        from agentic_commons.tools import ToolSource

        assert set(dir(ToolSource)) <= set(dir(McpToolSource))

    def test_multi_agent_is_still_deliberately_empty(self) -> None:
        import agentic_commons.multi_agent as multi_agent

        assert multi_agent.__all__ == []

    def test_sandbox_and_prompt_compilation_import(self) -> None:
        import agentic_commons.prompt_compilation as prompt_compilation
        import agentic_commons.sandbox as sandbox

        assert sandbox.__all__ and prompt_compilation.__all__
