"""One smoke test per episode (LLD §22).

"Does the episode run to completion?" is deliberately the whole bar. The
behaviour each episode demonstrates is already covered by the unit tests
for the subsystem it exercises; testing it twice would mean two places to
update when it changes, and the episode's job is to *show* the behaviour,
not to be its specification.

Two shapes are in play while the series converts. A **configured**
episode has a `config.yaml` and a `flow`, and runs the real agent; it is
tested by applying that same config and driving the flow against a
scripted model, so what is under test is the episode's configuration
rather than the model's mood. A **narrated** episode has a `run` and is
tested as before. Which shape an episode is, is a question about which
files exist — not a list anyone has to maintain.

The parameterised list is also a coverage check: an episode folder with
no module, or a module with neither entry point, fails here rather than
being noticed when someone tries to record it.
"""

import importlib
import pkgutil
from collections.abc import Callable

import pytest
from rich.console import Console

import episodes
from agentic_commons.config import REPO_ROOT, get_settings
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.graph import build_agent
from episodes._shared import CONFIG_FILENAME, EpisodeOutput, EpisodeRun, EpisodeSettings
from tests.conftest import ScriptedLLM

# Deferred to a later phase, by agreement: the sandbox needs a Docker
# deployment target, and multi-agent is not designed yet.
DEFERRED = {"e34", "e40", "e41", "e42", "e43"}


def _episode_names() -> list[str]:
    """Every episode package on disk."""
    return sorted(
        module.name
        for module in pkgutil.iter_modules(episodes.__path__)
        if module.ispkg and module.name.startswith("e")
    )


def _is_configured(name: str) -> bool:
    """Whether this episode runs the real agent under its own config."""
    return (REPO_ROOT / "episodes" / name / CONFIG_FILENAME).exists()


def _has(name: str, attribute: str) -> bool:
    """Whether an episode's app module exposes something."""
    return hasattr(importlib.import_module(f"episodes.{name}.app"), attribute)


EPISODES = _episode_names()
CONFIGURED = [name for name in EPISODES if _is_configured(name)]
# Configured episodes come in two shapes. A **live** one opens the TUI and
# is typed at, so it declares no conversation and has no flow to call. A
# **scripted** one still drives a `flow` over turns from its config, and
# is on its way to becoming live.
SCRIPTED = [name for name in CONFIGURED if _has(name, "flow")]
LIVE = [name for name in CONFIGURED if name not in SCRIPTED]
NARRATED = [name for name in EPISODES if not _is_configured(name)]


class TestEveryNarratedEpisodeRuns:
    @pytest.mark.parametrize("name", NARRATED)
    async def test_it_runs_to_completion(self, name: str) -> None:
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert isinstance(result, EpisodeOutput)

    @pytest.mark.parametrize("name", NARRATED)
    async def test_it_produces_an_exit_artifact(self, name: str) -> None:
        """An episode with nothing to show has nothing to record."""
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert result.artifact, f"{name} produced no exit artifact"

    @pytest.mark.parametrize("name", NARRATED)
    async def test_it_states_a_principle(self, name: str) -> None:
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert result.principle and result.lines


class TestEveryConfiguredEpisodeRuns:
    """The converted shape: the real agent, under the episode's own config.

    The model is scripted. An episode's claim is about the harness — that
    an injected instruction cannot reach an authority-bearing path, that
    a retried refund is one refund — and a claim about the harness that
    depended on the model agreeing would not be a claim about the harness.
    """

    @pytest.mark.parametrize("name", SCRIPTED)
    async def test_a_scripted_episode_runs_to_completion(
        self,
        name: str,
        episode_config: Callable[[str], EpisodeSettings],
        seeded_pool: SqliteConnectionPool,
    ) -> None:
        settings = episode_config(name)
        flow = importlib.import_module(f"episodes.{name}.app").flow
        agent = build_agent(llm=ScriptedLLM(), pool=seeded_pool)
        await flow(EpisodeRun(agent, settings, name, Console(quiet=True)))

    @pytest.mark.parametrize("name", LIVE)
    def test_a_live_episode_builds_its_agent(
        self,
        name: str,
        episode_config: Callable[[str], EpisodeSettings],
        seeded_pool: SqliteConnectionPool,
    ) -> None:
        """The TUI is typed at, so what is testable is what it opens on.

        Any `setup` the episode declares runs first, because an episode
        whose demonstration needs a particular order in the database has
        not really started until that order is there.
        """
        episode_config(name)
        module = importlib.import_module(f"episodes.{name}.app")
        if hasattr(module, "setup"):
            module.setup()
        assert build_agent(llm=ScriptedLLM(), pool=seeded_pool) is not None

    @pytest.mark.parametrize("name", CONFIGURED)
    def test_it_declares_a_title_and_a_principle(
        self, name: str, episode_config: Callable[[str], EpisodeSettings]
    ) -> None:
        """An episode nobody wrote a claim for is an episode with no point."""
        settings = episode_config(name)
        assert settings.title
        assert settings.principle

    @pytest.mark.parametrize("name", SCRIPTED)
    def test_a_scripted_episode_declares_its_turns(
        self, name: str, episode_config: Callable[[str], EpisodeSettings]
    ) -> None:
        assert episode_config(name).turns, f"{name} declares no conversation to run"

    @pytest.mark.parametrize("name", CONFIGURED)
    def test_its_config_actually_narrows_the_agent(
        self, name: str, episode_config: Callable[[str], EpisodeSettings]
    ) -> None:
        """The whole premise: a narrower config builds a narrower agent.

        An episode whose config switched nothing off would be the full
        product wearing an episode's title — which is exactly the thing
        this shape exists to stop being possible by accident.
        """
        episode_config(name)
        settings = get_settings()
        switched_off = [
            section
            for section, enabled in (
                ("tools", settings.tools.enabled),
                ("guardrails", settings.guardrails.enabled),
                *(
                    (feature, settings.features.is_on(feature))
                    for feature in settings.features.paths()
                ),
            )
            if not enabled
        ]
        assert switched_off, f"{name}'s config enables everything — it configures nothing down"


class TestTheSeriesIsComplete:
    def test_every_non_deferred_episode_exists(self) -> None:
        """E01-E47 plus the three capstone gates, minus what is deferred."""
        expected = {f"e{number:02d}" for number in range(1, 48)} | {"e48a", "e48b", "e48c"}
        missing = (expected - DEFERRED) - set(EPISODES)
        assert missing == set(), f"missing episodes: {sorted(missing)}"

    def test_nothing_deferred_was_built_by_accident(self) -> None:
        assert DEFERRED.isdisjoint(EPISODES)

    def test_every_episode_is_runnable_as_a_module(self) -> None:
        """`python -m episodes.eNN` is how they are demonstrated."""
        for name in EPISODES:
            importlib.import_module(f"episodes.{name}.__main__")


class TestEpisodesComposeRatherThanDuplicate:
    @pytest.mark.parametrize("name", EPISODES)
    def test_no_episode_builds_its_own_llm_client(self, name: str) -> None:
        """The regression the series exists to argue against.

        Checked over the AST rather than the text: an episode that
        *discusses* `print(` in a string would fail a substring match,
        and E48a does exactly that while checking the same property of
        the product.
        """
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(importlib.import_module(f"episodes.{name}.app")))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        constructed = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "print" not in called, f"{name} prints instead of using the shared renderer"
        assert "AsyncOpenAI" not in called | constructed
