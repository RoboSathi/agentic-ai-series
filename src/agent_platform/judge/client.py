"""The judge's model client: its configured model, protected, with its configured backups.

Every judge asks here — the eval graders (DeepEval, Promptfoo's
`llm-rubric`) and any agent judge built on `LlmJudge` without a client of
its own — so they all grade with the same model and the same chain
(`judge:`), built by the platform's one chain builder.
"""

from agent_platform.config import get_settings
from agent_platform.config.settings import JudgeSettings, ModelRef
from agent_platform.llm.chain import ModelChainBuilder
from agent_platform.llm.client import LLMClient, create_pinned_llm_client

# How the judge's chain is named on cards and in logs.
_LABEL = "judge"


def create_judge_client(settings: JudgeSettings | None = None) -> LLMClient:
    """The judge's model, then each configured backup, under the resilience stack."""
    judge = settings or get_settings().judge
    primary = ModelRef(provider=judge.provider, model=judge.model, extra_body=judge.extra_body)
    builder = ModelChainBuilder(lambda provider, model: create_pinned_llm_client(provider, model))
    return builder.build(primary, judge, label=_LABEL)


__all__ = ["create_judge_client"]
