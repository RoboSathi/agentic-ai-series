"""Config, paths and feature toggles (LLD §1)."""

import os
from pathlib import Path

from agent_platform.config import (
    clear_overlays,
    config_sources,
    get_settings,
    reload_settings,
    use_overlay,
)
from agent_platform.config.paths import REPO_ROOT, resolve_path
from agent_platform.config.toggles import ConfigFeatureToggle


class TestPaths:
    def test_repo_root_contains_the_source_tree(self) -> None:
        assert (REPO_ROOT / "src" / "agent_platform").is_dir()

    def test_paths_resolve_against_the_repo_not_the_cwd(self, tmp_path: Path) -> None:
        """The failure this guards: running from any other directory."""
        original = os.getcwd()
        os.chdir(tmp_path)
        try:
            assert resolve_path("src/agent_platform").is_dir()
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

    def test_settings_are_a_singleton(self) -> None:
        assert get_settings() is get_settings()

    def test_reload_rebuilds_the_singleton(self) -> None:
        first = get_settings()
        assert reload_settings() is not first

    def test_api_keys_are_never_config_fields(self, monkeypatch) -> None:
        """Keys come from the environment; a key's value in a settings dump leaks.

        The variable *names* are configuration; the values never reach a field.
        """
        llm = get_settings().llm
        for provider in llm.providers.values():
            for env in provider.key_env_vars:
                monkeypatch.setenv(env, "sk-test-never-shown")
        assert "sk-test-never-shown" not in str(reload_settings().llm.model_dump())

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
        get_settings().security.guards.enabled = False
        assert toggle.is_enabled("security.guards", "jailbreak") is False

    def test_sub_feature_reads_its_own_flag_when_the_parent_is_on(
        self, restore_settings: None
    ) -> None:
        """A sub-feature decides for itself, once its parent has allowed it."""
        settings = get_settings()
        settings.features = settings.features.with_override("memory.episodic", False)
        assert settings.features.is_on("memory.working") is True
        assert settings.features.is_on("memory.episodic") is False

    def test_a_section_without_an_enabled_flag_is_always_on(self) -> None:
        """The Gatekeeper has no off switch, by construction (PRD G1)."""
        assert ConfigFeatureToggle().is_enabled("gatekeeper") is True

    def test_unknown_feature_is_off(self) -> None:
        assert ConfigFeatureToggle().is_enabled("no_such_subsystem") is False

    def test_toggles_can_be_flipped_at_runtime(self, restore_settings: None) -> None:
        toggle = ConfigFeatureToggle()
        toggle.set("guardrails", None, False)
        assert toggle.is_enabled("guardrails") is False


class TestRunScopedOverlays:
    """The layer that lets one episode run the agent configured down.

    Every test here reloads through `restore_settings`, because the thing
    under test is the singleton's contents.
    """

    def _write(self, tmp_path: Path, body: str) -> Path:
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(body)
        return overlay

    def test_an_overlay_changes_the_live_settings(
        self, tmp_path: Path, restore_settings: None
    ) -> None:
        assert get_settings().features.is_on("memory") is True
        use_overlay(self._write(tmp_path, "features:\n  memory: false\n"))
        assert reload_settings().features.is_on("memory") is False

    def test_it_merges_deeply_rather_than_replacing_the_section(
        self, tmp_path: Path, restore_settings: None
    ) -> None:
        """Setting one sub-key must not erase its siblings.

        The failure this guards is silent and severe: an overlay that
        switched off `security.pii` and took the injection patterns with
        it would leave an agent that looks configured and is undefended.
        """
        use_overlay(self._write(tmp_path, "security:\n  pii:\n    mode: remove\n"))
        settings = reload_settings()
        assert settings.security.pii.mode == "remove"
        assert settings.security.pii.patterns
        assert settings.security.prompt_injection.suspicious_phrases

    def test_local_yaml_still_has_the_last_word(self) -> None:
        """Machine-local overrides outrank a run's, or they are not overrides."""
        assert config_sources()[-1].name == "local.yaml"

    def test_sources_report_the_overlay(self, tmp_path: Path, restore_settings: None) -> None:
        """`/session` answers "which config is live?" from this list."""
        overlay = self._write(tmp_path, "features:\n  memory: false\n")
        use_overlay(overlay)
        assert overlay.resolve() in config_sources()

    def test_adding_the_same_overlay_twice_is_the_same_as_once(
        self, tmp_path: Path, restore_settings: None
    ) -> None:
        overlay = self._write(tmp_path, "features:\n  memory: false\n")
        use_overlay(overlay)
        use_overlay(overlay)
        assert config_sources().count(overlay.resolve()) == 1

    def test_clearing_restores_the_base_configuration(
        self, tmp_path: Path, restore_settings: None
    ) -> None:
        use_overlay(self._write(tmp_path, "features:\n  memory: false\n"))
        assert reload_settings().features.is_on("memory") is False
        clear_overlays()
        assert reload_settings().features.is_on("memory") is True

    def test_a_missing_overlay_is_not_an_error(
        self, tmp_path: Path, restore_settings: None
    ) -> None:
        """An episode with no config of its own still runs, fully featured."""
        use_overlay(tmp_path / "absent.yaml")
        assert reload_settings().features.is_on("memory") is True

    def test_reload_actually_rereads_the_file(self, tmp_path: Path, restore_settings: None) -> None:
        """The defect this guards, which was real.

        Sections used to be default *instances* on `AppConfig`, built once
        when the module was imported. `reload_settings()` then rebuilt the
        root and handed back the same section objects, having re-read
        nothing — so an overlay applied afterwards appeared in
        `config_sources()` and changed nothing at all.
        """
        overlay = self._write(tmp_path, "features:\n  memory: false\n")
        use_overlay(overlay)
        assert reload_settings().features.is_on("memory") is False
        overlay.write_text("features:\n  memory: true\n")
        assert reload_settings().features.is_on("memory") is True
