"""Role-based model routing, and the platform's vocabulary boundary.

The boundary test is the regression guard for a failure this repo has
had: agent vocabulary leaking into the shared library. Research words are
banned from all of `agent_platform`. `rubric` and `report` predate the
research agent (the eval judge's rubric, `StageReporter`), and `report`
is the platform's own word for a status record (`LoopControlReport`, which
the new budget guard must reuse rather than duplicate), so only `rubric` is
banned inside the modules added for the research agent (ADR 0003).
"""

import re
import subprocess
from functools import cache
from pathlib import Path

import pytest

from agent_platform.config import REPO_ROOT
from agent_platform.config.settings import ProviderName, RoleModelSettings
from agent_platform.llm.client import LLMClient
from agent_platform.llm.roles import RoleCollisionError, RoleModelRouter, RoleNotConfiguredError
from tests.conftest import ScriptedLLM

COMMONS = REPO_ROOT / "src" / "agent_platform"
BANNED_EVERYWHERE = ("archetype", "research", "brief")
BANNED_IN_NEW = ("rubric",)
# Modules added for the research agent: no research vocabulary at all.
NEW_MODULES = (
    "scheduling",
    "artifacts",
    "evidence",
    "web",
    "primitives/hashing.py",
    "loop_control/budget.py",
    "durability/isolated.py",
    "durability/hitl.py",
    "llm/roles.py",
    "verification/acceptance.py",
)
# A pre-existing use of the English word "brief" (a lock held briefly).
ALLOWED = {("evals/core/store/duckdb_store.py", "brief")}


@cache
def _generated() -> frozenset[Path]:
    """Gitignored files under the platform: build output (`make dashboard`), not its source.

    A Rill dashboard is generated per agent and names it; it lives beside the
    tracked project only because Rill reads it from there.
    """
    listed = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", str(COMMONS)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return frozenset(REPO_ROOT / line for line in listed.stdout.splitlines())


def _router(**roles: tuple[ProviderName, str]) -> tuple[RoleModelRouter, list[tuple[str, str]]]:
    built: list[tuple[str, str]] = []

    def factory(provider: ProviderName, model: str) -> LLMClient:
        built.append((provider, model))
        return ScriptedLLM()

    spec = {name: RoleModelSettings(provider=p, model=m) for name, (p, m) in roles.items()}
    return RoleModelRouter(spec, factory=factory), built


class TestRoleModelRouter:
    def test_roles_on_one_model_share_its_adapter(self) -> None:
        router, built = _router(
            writer=("nim", "big"), planner=("nim", "big"), judge=("nim", "other")
        )
        for role in ("writer", "planner", "judge"):
            router.client_for(role)
        assert router.client_for("writer") is router.client_for("writer")  # built once per role
        assert built == [("nim", "big"), ("nim", "other")]  # one adapter per distinct model
        assert router.roles == ["judge", "planner", "writer"]

    def test_independence_is_checked(self) -> None:
        router, _ = _router(writer=("nim", "big"), judge=("nim", "big"))
        with pytest.raises(RoleCollisionError):
            router.require_distinct("judge", "writer")
        distinct, _ = _router(writer=("nim", "big"), judge=("local", "big"))
        distinct.require_distinct("judge", "writer")

    def test_an_unmapped_role_is_an_error(self) -> None:
        router, _ = _router()
        with pytest.raises(RoleNotConfiguredError):
            router.client_for("nobody")

    def test_the_default_factory_pins_the_configured_model(self) -> None:
        router = RoleModelRouter({"r": RoleModelSettings(provider="nim", model="m-1")})
        assert router.client_for("r").model == "m-1"
        assert router.model_key("r") == "nim:m-1"


def _python_files(entry: str) -> list[Path]:
    path = COMMONS / entry
    return [path] if path.suffix == ".py" else sorted(path.rglob("*.py"))


class TestNoAgentVocabularyInThePlatform:
    @pytest.mark.parametrize("word", BANNED_EVERYWHERE)
    def test_research_words_appear_nowhere_in_commons(self, word: str) -> None:
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        offenders = [
            str(path.relative_to(COMMONS))
            for path in COMMONS.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".yaml", ".md"}
            and path not in _generated()
            and pattern.search(path.read_text())
            and (str(path.relative_to(COMMONS)), word) not in ALLOWED
        ]
        assert offenders == []

    @pytest.mark.parametrize("word", BANNED_IN_NEW)
    def test_evaluation_words_stay_out_of_the_new_mechanisms(self, word: str) -> None:
        pattern = re.compile(rf"\b{word}", re.IGNORECASE)
        offenders = [
            str(path.relative_to(COMMONS))
            for entry in NEW_MODULES
            for path in _python_files(entry)
            if pattern.search(path.read_text())
        ]
        assert offenders == []
