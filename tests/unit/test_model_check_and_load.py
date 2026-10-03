"""`make model-check`'s latency colours, and the scoped silencing of a library's load warning."""

import warnings

import pytest

from agent_platform.config import get_settings
from agent_platform.llm import cli
from agent_platform.llm.local_models import _library_warnings_silenced


def test_latency_is_coloured_by_its_band() -> None:
    styles = get_settings().llm.check.styles
    assert cli._ms(styles.fast_ms - 1).style == styles.fast
    assert cli._ms(styles.slow_ms + 1).style == styles.slow
    assert cli._ms((styles.fast_ms + styles.slow_ms) / 2).style == styles.medium


def test_a_status_takes_its_configured_style() -> None:
    styles = get_settings().llm.check.styles
    assert cli._status("failed").style == styles.status["failed"]


def test_only_the_configured_warning_is_silenced_and_only_during_the_load() -> None:
    pattern = get_settings().llm.suppressed_load_warnings[0]
    assert pattern, "the filter is configuration"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _library_warnings_silenced():
            warnings.warn("`torch.jit.script` is deprecated", FutureWarning, stacklevel=1)
            warnings.warn("something else", FutureWarning, stacklevel=1)
        warnings.warn("`torch.jit.script` is deprecated", FutureWarning, stacklevel=1)
    messages = [str(w.message) for w in caught]
    assert messages == ["something else", "`torch.jit.script` is deprecated"]


def test_a_providers_key_comes_from_its_configured_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = get_settings().llm
    envs = llm.provider_settings(llm.provider).key_env_vars
    for env in envs:
        monkeypatch.delenv(env, raising=False)
    assert llm.api_key_for(llm.provider) is None
    monkeypatch.setenv(envs[0], "k")
    assert llm.api_key_for(llm.provider) == "k"
