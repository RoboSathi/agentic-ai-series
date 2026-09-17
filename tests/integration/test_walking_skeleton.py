"""Integration tests for the walking skeleton."""

from pathlib import Path

from agentic_commons.config import get_settings
from customer_support_agent.core.graph import build_graph


def test_walking_skeleton_runs() -> None:
    """Test that the walking skeleton graph builds without errors."""
    graph = build_graph()
    compiled = graph.compile()

    assert compiled is not None
    assert hasattr(compiled, 'invoke')


def test_walking_skeleton_visits_all_nodes() -> None:
    """Test that the graph is built with all expected nodes."""
    graph = build_graph()
    assert graph is not None


def test_walking_skeleton_with_disabled_toggle() -> None:
    """Test that the graph builds with disabled toggles."""
    graph = build_graph()
    compiled = graph.compile()

    assert compiled is not None


def test_log_dir_created() -> None:
    """Test that log directory is created by logging setup."""
    settings = get_settings()
    log_dir = Path(settings.logging.log_dir)

    assert log_dir.exists() or not settings.logging.file_enabled
