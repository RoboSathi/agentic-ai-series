"""Config, paths and feature toggles (LLD §1)."""

import os
from pathlib import Path

from agentic_commons.config import get_settings, reload_settings
from agentic_commons.config.paths import REPO_ROOT, resolve_path
from agentic_commons.config.toggles import ConfigFeatureToggle


class TestPaths:
    def test_repo_root_contains_the_source_tree(self) -> None:
        assert (REPO_ROOT / "src" / "agentic_commons").is_dir()

    def test_paths_resolve_against_the_repo_not_the_cwd(self, tmp_path: Path) -> None:
        """The failure this guards: running from any other directory."""
        original = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert resolve_path("src/agentic_commons").is_dir()
        finally:
            os.chdir(original)

    def test_absolute_paths_pass_through(self, tmp_path: Path) -> None:
        assert resolve_path(str(tmp_path)) == tmp_path


class TestSettings:
    def test_every_subsystem_has_a_settings_section(self) -> None:
        settings = get_settings()
        for section in (
            "logging",
            "llm",
            "persistence",
            "audit",
            "observability",
            "context",
            "memory",
            "guardrails",
            "hallucination_protection",
            "gatekeeper",
            "security",
            "resilience",
            "evals",
            "tools",
            "tui",
        ):
            assert getattr(settings, section) is not None, section

    def test_the_platform_does_not_own_an_agents_settings(self) -> None:
        """A shared library that depends on one of its consumers is not shared."""
        assert not hasattr(get_settings(), "customer_support_agent")

    def test_an_agent_loads_its_own_settings(self) -> None:
        from customer_support_agent.config import get_agent_settings

        assert get_agent_settings().intents

    def test_the_agent_overlay_merges_into_platform_sections(self) -> None:
        """Deep merge: the overlay fills part of a section, not all of it."""
        gatekeeper = get_settings().gatekeeper
        assert gatekeeper.surface_policy.surfaces  # from the agent overlay
        assert gatekeeper.authority_hierarchy  # from the platform base

    def test_settings_are_a_singleton(self) -> None:
        assert get_settings() is get_settings()

    def test_reload_rebuilds_the_singleton(self) -> None:
        first = get_settings()
        assert reload_settings() is not first

    def test_api_keys_are_never_config_fields(self) -> None:
        """Keys come from the environment; a key in a settings dump leaks."""
        dumped = str(get_settings().llm.model_dump())
        assert "api_key" not in dumped

    def test_api_key_is_read_from_the_environment(self, monkeypatch) -> None:
        monkeypatch.setenv("nvidia_api_key", "value-from-env")
        assert get_settings().llm.api_key_for("nim") == "value-from-env"

    def test_unknown_provider_has_no_key(self) -> None:
        assert get_settings().llm.api_key_for("local") is None


class TestFeatureToggles:
    def test_reads_a_subsystem_flag(self) -> None:
        assert ConfigFeatureToggle().is_enabled("memory") is True

    def test_a_sub_feature_is_gated_by_its_parent(self, restore_settings: None) -> None:
        """Turning off a parent must turn off everything beneath it."""
        toggle = ConfigFeatureToggle()
        get_settings().security.enabled = False
        assert toggle.is_enabled("security", "auth") is False

    def test_sub_feature_reads_its_own_flag_when_the_parent_is_on(
        self, restore_settings: None
    ) -> None:
        """A sub-feature decides for itself, once its parent has allowed it."""
        get_settings().security.pii.enabled = False
        toggle = ConfigFeatureToggle()
        assert toggle.is_enabled("security", "auth") is True
        assert toggle.is_enabled("security", "pii") is False

    def test_a_section_without_an_enabled_flag_is_always_on(self) -> None:
        """The Gatekeeper has no off switch, by construction (PRD G1)."""
        assert ConfigFeatureToggle().is_enabled("gatekeeper") is True

    def test_unknown_feature_is_off(self) -> None:
        assert ConfigFeatureToggle().is_enabled("no_such_subsystem") is False

    def test_toggles_can_be_flipped_at_runtime(self, restore_settings: None) -> None:
        toggle = ConfigFeatureToggle()
        toggle.set("guardrails", None, False)
        assert toggle.is_enabled("guardrails") is False
