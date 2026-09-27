"""Resolve the running agent's eval plugin from configuration."""

from agentic_commons.config import get_settings
from agentic_commons.evals.contracts import EvalPlugin
from agentic_commons.host import load_spec


class EvalPluginNotConfiguredError(RuntimeError):
    """No agent named an eval plugin — there is nothing to evaluate."""


def resolve_eval_plugin(spec: str | None = None) -> EvalPlugin:
    """Load the plugin named by `evals.plugin` (a `module.path:attribute`)."""
    configured = spec if spec is not None else get_settings().evals.plugin
    if not configured:
        raise EvalPluginNotConfiguredError(
            "evals.plugin is empty: the platform names no agent, so the agent's "
            "config must name its eval plugin."
        )
    plugin = load_spec(configured)
    if not isinstance(plugin, EvalPlugin):
        raise TypeError(f"'{configured}' does not satisfy the EvalPlugin protocol.")
    return plugin
