"""One smoke test per episode (LLD §22).

"Does app.py run to completion and produce its documented exit artifact?"
That is deliberately the whole bar. The behaviour each episode
demonstrates is already covered by the unit tests for the subsystem it
exercises; testing it twice would mean two places to update when it
changes, and the episode's job is to *show* the behaviour, not to be its
specification.

The parameterised list is also a coverage check: an episode folder with
no module, or a module with no `run`, fails here rather than being
noticed when someone tries to record it.
"""

import importlib
import pkgutil

import pytest

import episodes
from episodes._shared import EpisodeOutput

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


EPISODES = _episode_names()


class TestEveryEpisodeRuns:
    @pytest.mark.parametrize("name", EPISODES)
    async def test_it_runs_to_completion(self, name: str) -> None:
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert isinstance(result, EpisodeOutput)

    @pytest.mark.parametrize("name", EPISODES)
    async def test_it_produces_an_exit_artifact(self, name: str) -> None:
        """An episode with nothing to show has nothing to record."""
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert result.artifact, f"{name} produced no exit artifact"

    @pytest.mark.parametrize("name", EPISODES)
    async def test_it_states_a_principle(self, name: str) -> None:
        module = importlib.import_module(f"episodes.{name}.app")
        result = await module.run()
        assert result.principle and result.lines


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
