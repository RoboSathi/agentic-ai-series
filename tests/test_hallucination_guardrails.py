import pytest
from pydantic import BaseModel

from agent_platform.context.types import Chunk
from agent_platform.guardrails import PydanticValidator
from agent_platform.guardrails.response import ResponseGuardrail


class _Schema(BaseModel):
    answer: str


def test_pydantic_validator_rejects_wrong_shape():
    validator = PydanticValidator(mode="reject")
    result = validator.validate('{"answer": 123}', _Schema)
    assert not result.valid


@pytest.mark.asyncio
async def test_response_guardrail_passes_grounded_response():
    class FakeDetection:
        async def assess(self, text, evidence, resample=None):
            from agent_platform.hallucination_protection.types import (
                DetectionVerdict,
                GenerationAssessment,
            )

            return GenerationAssessment(
                verdicts=[
                    DetectionVerdict(
                        claim=text,
                        supported=True,
                        confidence=1.0,
                        tier="exact_assertion",
                    )
                ]
            )

        @staticmethod
        def diagnose(claim, evidence):
            return "reasoning"

    guard = ResponseGuardrail(FakeDetection())
    result = await guard.enforce(
        "refund approved",
        [Chunk(content="refund approved", kind="tool_result")],
    )
    assert result.grounded
    assert not result.escalated
