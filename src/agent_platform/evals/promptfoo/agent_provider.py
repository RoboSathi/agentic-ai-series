"""Promptfoo custom provider: each test case runs the real agent under test.

An agent's `promptfooconfig.yaml` names a one-line `provider.py` that
re-exports `call_api` from here. Promptfoo calls it in its own Python worker. The worker finds the
agent the same way every eval does — `evals.plugin` in configuration —
builds the case from the test's vars, and returns the reply as Promptfoo's
`output` and the full transcript as `metadata`, where the agent's own
assertions read it.

A fresh agent-under-test per call: `asyncio.run` gives every call its own
event loop, and an HTTP client from a finished loop cannot be reused.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import EvalPlugin, GoldenCase
from agent_platform.evals.core.plugin import resolve_eval_plugin

_plugin: EvalPlugin | None = None
# The test var holding a case's list-valued details (tags, turns).
CASE_VAR = "case"


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Promptfoo's provider entry point."""
    import asyncio

    case = case_from_vars(prompt, context.get("vars") or {})
    try:
        transcript = asyncio.run(_prepared().agent_under_test().run(case))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "output": transcript.reply,
        "metadata": {
            "case": case.model_dump(mode="json"),
            "transcript": transcript.model_dump(mode="json"),
        },
        "tokenUsage": {
            "prompt": transcript.usage.prompt_tokens,
            "completion": transcript.usage.completion_tokens,
            "total": transcript.usage.prompt_tokens + transcript.usage.completion_tokens,
            "numRequests": transcript.usage.llm_calls,
        },
        "cost": transcript.usage.cost_usd,
    }


def case_from_vars(prompt: str, test_vars: dict[str, Any]) -> GoldenCase:
    """A security test's vars, as a case.

    List-valued details (`tags`, `turns`) arrive inside the `case` object:
    Promptfoo expands a top-level list var into one test per element, which
    once ran a two-turn attack as dozens of one-character "turns".
    """
    details = dict(test_vars.get(CASE_VAR) or {})
    turns = details.get("turns") or [prompt]
    if isinstance(turns, str):
        raise ValueError(f"'turns' must be a list of messages, got a string: {turns!r}")
    return GoldenCase(
        suite=get_settings().evals.suites.security,
        case_id=str(test_vars["case_id"]),
        category=str(test_vars.get("category", "")),
        tags=list(details.get("tags") or []),
        turns=[str(turn) for turn in turns],
        payload=dict(test_vars.get("payload") or {}),
    )


def _prepared() -> EvalPlugin:
    global _plugin
    if _plugin is None:
        _plugin = resolve_eval_plugin()
        _plugin.prepare()
    return _plugin
