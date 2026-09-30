"""Fixtures for the research agent's tests.

The shared conftest runs every test under the customer support overlay
(importing that package declares it). The research agent's tests layer its
own overlay on top for their duration and restore the configuration
afterwards, so no other test ever sees research settings.
"""

from collections.abc import Iterator

import pytest

from agentic_commons.config import (
    REPO_ROOT,
    clear_overlays,
    reload_settings,
    set_active_features,
    use_overlay,
)
from research_agent.config import reload_agent_settings

RA_OVERLAY = REPO_ROOT / "src" / "research_agent" / "config" / "config.yaml"


@pytest.fixture
def ra_config() -> Iterator[None]:
    """The research agent's overlay, applied for one test."""
    use_overlay(RA_OVERLAY)
    reload_settings()
    reload_agent_settings()
    set_active_features(None)
    yield
    clear_overlays()
    set_active_features(None)
    reload_settings()
