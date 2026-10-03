"""Promptfoo grader provider: `llm-rubric` assertions graded by the eval judge.

Passed to Promptfoo as `--grader`, so model-graded assertions use the
same judge model, logging and rate-limit budget as every other stage,
instead of Promptfoo's default (a separate OpenAI account).
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.judge_client import create_judge_client
from agent_platform.llm.types import CompletionRequest, Message


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Promptfoo's provider entry point."""
    import asyncio

    judge = get_settings().evals.judge
    client = create_judge_client(judge)
    try:
        result = asyncio.run(
            client.complete(
                CompletionRequest(
                    messages=[Message(role="user", content=prompt)],
                    temperature=judge.temperature,
                    max_tokens=judge.max_tokens,
                )
            )
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "output": result.content,
        "tokenUsage": {
            "prompt": result.usage.prompt_tokens,
            "completion": result.usage.completion_tokens,
            "total": result.usage.total_tokens,
            "numRequests": 1,
        },
        "cost": result.cost_usd,
    }
