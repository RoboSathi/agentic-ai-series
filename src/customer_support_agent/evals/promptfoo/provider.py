"""Promptfoo provider (`providers:` in promptfooconfig.yaml): each attack runs the real agent."""

from agent_platform.evals.promptfoo.agent_provider import call_api

__all__ = ["call_api"]
