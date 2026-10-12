"""The platform judge as a DeepEval model (Adapter).

DeepEval's metrics ask a `DeepEvalBaseLLM` for text, or for an instance of
a Pydantic schema. This answers both through `agent_platform.judge` — the
configured judge model and its backups, the shared grading rules, the
guardrails' bounded repair — so a DeepEval verdict is logged, costed and
drawn from the same rate-limit budget as everything else, and an answer
that fails its schema raises (the metric errors) instead of being guessed at.

Built inside the thread `deepeval.evaluate()` runs in, so the client is
bound to that thread's event loop, never to the pipeline's. The few metrics
that ask synchronously get a fresh client on a loop of their own.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from agent_platform.judge import JudgeCall, ask_text, ask_validated, create_judge_client
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import Message


def judge_model(deepeval: Any, client: LLMClient | None = None) -> Any:
    """A `DeepEvalBaseLLM` over the platform judge, counting what it spends."""
    base = deepeval.models.base_model.DeepEvalBaseLLM

    class PlatformJudgeModel(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self._client = client or create_judge_client()
            self.calls = 0
            self.cost_usd = 0.0
            super().__init__(model=self._client.model)

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            return self._client

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            return str(self._client.model)

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            # Called from inside DeepEval's running loop: answer on a loop of its own, with a
            # client of its own (an async client cannot cross event loops).
            fresh = client or create_judge_client()
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._ask(fresh, prompt, schema)).result()

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return await self._ask(self._client, prompt, schema)

        async def _ask(self, judge: LLMClient, prompt: str, schema: Any) -> Any:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                answer, call = await ask_validated(
                    judge, [Message(role="user", content=prompt)], schema
                )
                self._count(call)
                return answer
            text, call = await ask_text(judge, prompt)
            self._count(call)
            return text

        def _count(self, call: JudgeCall) -> None:
            self.calls += call.calls
            self.cost_usd += call.cost_usd

    return PlatformJudgeModel()
