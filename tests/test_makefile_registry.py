"""The Makefile's agent registry: the agent is named, explicitly, as the second goal.

Exercised with `make -n` (print, do not run), so nothing here starts a
console or touches data; what is under test is which overlay each
invocation would export and which commands it would run.
"""

import os
import shutil
import subprocess

import pytest

from agentic_commons.config import REPO_ROOT

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")

CSA_OVERLAY = "src/customer_support_agent/config/config.yaml"
RA_OVERLAY = "src/research_agent/config/config.yaml"
RA_DEMO = f"{RA_OVERLAY}:src/research_agent/config/profiles/demo.yaml"


def _env() -> dict[str, str]:
    """The environment without an agent already named (this test process names one)."""
    return {key: value for key, value in os.environ.items() if key != "AGENTIC_OVERLAYS"}


def _make(*goals: str, dry: bool = True) -> subprocess.CompletedProcess[str]:
    flags = ["-n"] if dry else []
    return subprocess.run(
        ["make", "--no-print-directory", *flags, *goals],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    )


def _exported(*goals: str) -> str:
    """The AGENTIC_OVERLAYS an invocation exports (read from make's database)."""
    out = subprocess.run(
        ["make", "--no-print-directory", "-pn", *goals],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    ).stdout
    lines = [line for line in out.splitlines() if line.startswith("AGENTIC_OVERLAYS")]
    return lines[0].split(":=")[-1].strip() if lines else ""


class TestTheAgentIsNamed:
    @pytest.mark.parametrize(("agent", "overlay"), [("csa", CSA_OVERLAY), ("ra", RA_OVERLAY)])
    def test_the_name_chooses_the_overlay(self, agent: str, overlay: str) -> None:
        assert _exported("tui", agent) == overlay

    @pytest.mark.parametrize("target", ["tui", "eval", "eval-oracle", "eval-runs", "data"])
    def test_a_missing_name_is_a_usage_error(self, target: str) -> None:
        result = _make(target, dry=False)
        assert result.returncode != 0
        assert "Usage" in result.stdout

    def test_an_unknown_name_is_an_error(self) -> None:
        result = _make("tui", "xyz", dry=False)
        assert result.returncode != 0 and "Unknown agent" in result.stdout

    def test_two_names_are_an_error(self) -> None:
        result = _make("tui", "csa", "ra", dry=False)
        assert result.returncode != 0 and "exactly one agent" in result.stdout

    def test_nothing_is_exported_without_a_name(self) -> None:
        assert _exported("lint") == ""


class TestProfiles:
    def test_the_console_is_live_unless_a_profile_is_named(self) -> None:
        assert _exported("tui", "ra") == RA_OVERLAY
        assert _exported("tui", "ra", "PROFILE=demo") == RA_DEMO

    @pytest.mark.parametrize("target", ["eval", "eval-oracle", "data"])
    def test_evals_and_data_are_always_offline(self, target: str) -> None:
        assert _exported(target, "ra") == RA_DEMO
        assert _exported(target, "ra", "PROFILE=anything") == RA_DEMO

    def test_csa_has_no_offline_profile(self) -> None:
        assert _exported("eval-oracle", "csa") == CSA_OVERLAY

    def test_an_unknown_profile_is_an_error(self) -> None:
        result = _make("tui", "ra", "PROFILE=nope", dry=False)
        assert result.returncode != 0 and "No profile" in result.stdout


class TestTargetsKeepTheirBehaviour:
    def test_tui_runs_the_console(self) -> None:
        assert "-m agentic_commons.tui" in _make("tui", "csa").stdout

    def test_eval_options_still_pass_through(self) -> None:
        out = _make("eval", "ra", "FRESH=1", "LIMIT=2", "CASE=c1", "TAG=t").stdout
        assert "run --fresh --limit 2 --case c1 --tag t" in out

    def test_run_option_reaches_the_graders(self) -> None:
        assert "oracle --run r1" in _make("eval-oracle", "csa", "RUN=r1").stdout

    def test_data_is_per_agent(self) -> None:
        assert (
            "customer_support_agent.dummy_data.generate_dummy_data" in _make("data", "csa").stdout
        )
        assert "research_agent.fixture_world.generate" in _make("data", "ra").stdout

    def test_csa_admin_targets_name_their_agent(self) -> None:
        out = _make("refund-limit", "100").stdout
        assert f"AGENTIC_OVERLAYS={CSA_OVERLAY}" in out and "refund_limit 100" in out

    def test_verify_imports_each_root_in_its_own_process(self) -> None:
        out = _make("verify").stdout
        assert f"AGENTIC_OVERLAYS={CSA_OVERLAY}" in out and f"AGENTIC_OVERLAYS={RA_OVERLAY}" in out
        assert out.count("import build_agent") == 2
