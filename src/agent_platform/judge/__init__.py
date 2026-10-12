"""The LLM judge every agent and every eval grader builds on.

- `create_judge_client` — the platform judge model (`judge:`) and its backups.
- `LlmJudge` — the skeleton: ask, validate, decide, abstain on failure.
- `ModelJudge` — the same on the platform judge model, with the shared grading rules.
- `ask_validated` / `ask_text` — one judge call, for a framework with its own loop.
- `JudgeCalibrator` — agreement with a second rater (Cohen's κ).
"""

from agent_platform.judge.base import (
    JudgeAnswerError,
    JudgeCall,
    LlmJudge,
    ModelJudge,
    ask_text,
    ask_validated,
    judge_rules,
    with_rules,
)
from agent_platform.judge.calibration import AgreementReport, JudgeCalibrator
from agent_platform.judge.client import create_judge_client

__all__ = [
    "AgreementReport",
    "JudgeAnswerError",
    "JudgeCalibrator",
    "JudgeCall",
    "LlmJudge",
    "ModelJudge",
    "ask_text",
    "ask_validated",
    "create_judge_client",
    "judge_rules",
    "with_rules",
]
