"""The feature tree an agent declares, and how it gates (LLD §1)."""

import pytest

from agent_platform.config import get_settings
from agent_platform.config.features import (
    FeatureSet,
    active_features,
    set_active_features,
)
from agent_platform.persistence import SqliteConnectionPool
from customer_support_agent.graph import build_agent
from tests.conftest import ScriptedLLM

TREE = {
    "pii": True,
    "gatekeeper": False,
    "memory": {"enabled": True, "working": True, "episodic": False},
    "context": {"enabled": False, "rerank": True},
}


def a_set() -> FeatureSet:
    return FeatureSet(tree=TREE)


class TestGating:
    def test_a_declared_leaf_reports_its_own_value(self) -> None:
        assert a_set().is_on("pii")
        assert not a_set().is_on("gatekeeper")

    def test_a_parent_with_children_reports_its_enabled_key(self) -> None:
        assert a_set().is_on("memory")
        assert not a_set().is_on("context")

    def test_a_child_is_on_only_when_its_parent_is(self) -> None:
        """The whole point of nesting: `context` off takes `rerank` with it."""
        assert a_set().is_on("memory.working")
        assert not a_set().is_on("context.rerank")

    def test_a_child_can_be_off_under_an_enabled_parent(self) -> None:
        assert not a_set().is_on("memory.episodic")

    def test_an_undeclared_path_is_off(self) -> None:
        """Fail-safe: a typo must not silently resurrect a control."""
        assert not a_set().is_on("hallucinaton_guard")
        assert not a_set().is_on("memory.semantic")

    def test_a_path_below_a_bool_leaf_is_off(self) -> None:
        assert not a_set().is_on("pii.anything")


class TestPaths:
    def test_every_declared_path_is_listed_parents_first(self) -> None:
        assert a_set().paths() == [
            "pii",
            "gatekeeper",
            "memory",
            "memory.working",
            "memory.episodic",
            "context",
            "context.rerank",
        ]

    def test_the_enabled_key_is_not_itself_a_feature(self) -> None:
        assert "memory.enabled" not in a_set().paths()

    def test_an_empty_set_declares_nothing(self) -> None:
        assert FeatureSet().paths() == []


class TestOverride:
    def test_setting_a_leaf_returns_a_changed_copy(self) -> None:
        assert a_set().with_override("gatekeeper", True).is_on("gatekeeper")

    def test_setting_a_parent_writes_its_enabled_key(self) -> None:
        flipped = a_set().with_override("context", True)
        assert flipped.is_on("context") and flipped.is_on("context.rerank")

    def test_the_original_is_untouched(self) -> None:
        """Handing the same posture to several collaborators has to be safe."""
        original = a_set()
        original.with_override("memory.working", False)
        assert original.is_on("memory.working")

    def test_an_undeclared_path_is_refused(self) -> None:
        with pytest.raises(KeyError):
            a_set().with_override("no_such_feature", False)


class TestBulk:
    def test_all_off_switches_off_every_feature_and_sub_feature(self) -> None:
        off = a_set().with_all(False)
        assert not any(off.is_on(path) for path in off.paths())

    def test_all_on_switches_on_every_feature_and_sub_feature(self) -> None:
        on = a_set().with_all(True)
        assert all(on.is_on(path) for path in on.paths())

    def test_bulk_preserves_the_declared_shape(self) -> None:
        assert a_set().with_all(False).paths() == a_set().paths()


class TestSubtree:
    def test_a_child_set_stands_alone(self) -> None:
        assert a_set().child("memory").is_on("working")

    def test_a_leaf_has_no_subtree(self) -> None:
        assert a_set().child("pii").paths() == []


class TestActiveSet:
    def test_the_composition_root_publishes_what_it_resolved(self) -> None:
        """The console reports a posture it did not assemble, and must not
        import an agent to ask about it."""
        set_active_features(a_set())
        assert active_features().is_on("pii")
        assert not active_features().is_on("gatekeeper")


class TestDeclaredConfiguration:
    def test_the_agent_overlay_supplies_the_running_posture(self) -> None:
        features = get_settings().features
        assert features.is_on("gatekeeper")
        assert features.is_on("memory.procedural")
        assert features.is_on("context.failure_detection")

    def test_every_toggleable_feature_is_declared(self) -> None:
        """The eight, and nothing silently missing from the merged tree."""
        declared = set(get_settings().features.paths())
        assert {
            "basic_sanity",
            "prompt_injection",
            "content_moderation",
            "pii",
            "intent_classification",
            "hallucination_guard",
            "gatekeeper",
            "memory",
            "context",
        } <= declared

    def test_always_on_plumbing_is_not_a_feature(self) -> None:
        """Removing these yields a broken program, not an instructive demo."""
        declared = set(get_settings().features.paths())
        for plumbing in ("resilience", "audit", "observability", "guardrails", "tools"):
            assert plumbing not in declared


class TestTwoAgentsInOneProcess:
    """The defect this whole mechanism exists to fix.

    Components used to read `get_settings()` themselves — a process-wide
    singleton — so the first agent composed decided the posture for every
    other agent in the process. Two deployments of `agent_platform` could
    not differ. Now the feature set is a value handed to a composition
    root, so they can.
    """

    def _graph_nodes(self, agent: object) -> set[str]:
        return {
            name
            for name in agent._graph.get_graph().nodes  # type: ignore[attr-defined]
            if not name.startswith("__")
        }

    def test_each_agent_keeps_the_posture_it_was_built_with(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        settings = get_settings()
        full = settings.features
        narrow = full.with_override("prompt_injection", False).with_override("context", False)

        guarded = build_agent(llm=ScriptedLLM(), pool=seeded_pool, features=full)
        bare = build_agent(llm=ScriptedLLM(), pool=seeded_pool, features=narrow)

        assert "prompt_injection" in self._graph_nodes(guarded)
        assert "prompt_injection" not in self._graph_nodes(bare)
        assert "assemble_context" in self._graph_nodes(guarded)
        assert "assemble_context" not in self._graph_nodes(bare)

    def test_building_the_second_does_not_reshape_the_first(
        self, seeded_pool: SqliteConnectionPool, restore_settings: None
    ) -> None:
        """The precise failure: composing one agent used to decide for both."""
        settings = get_settings()
        guarded = build_agent(llm=ScriptedLLM(), pool=seeded_pool, features=settings.features)
        before = self._graph_nodes(guarded)
        build_agent(
            llm=ScriptedLLM(),
            pool=seeded_pool,
            features=settings.features.with_all(False),
        )
        assert self._graph_nodes(guarded) == before
