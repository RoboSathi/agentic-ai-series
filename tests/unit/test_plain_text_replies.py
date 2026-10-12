"""A provider that ignores guided JSON answers in plain text; that reply is the reply."""

from typing import Any

from agent_platform.guardrails import ResponseGuardrail
from agent_platform.guardrails.types import ValidationResult


class CountingValidator:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, raw: str, schema: Any) -> ValidationResult:
        self.calls += 1
        return ValidationResult(valid=True, value={"response": "unwrapped"}, mode="auto_repair")


def test_a_plain_reply_is_taken_as_it_is_without_a_repair_attempt() -> None:
    validator = CountingValidator()
    guardrail = ResponseGuardrail(None, validator=validator)  # type: ignore[arg-type]
    reply, fields = guardrail.validate_schema("  Your refund for order 90002 is under review. ")
    assert reply == "Your refund for order 90002 is under review."
    assert fields["schema_fallback"] == "plain_text_provider_output"
    assert validator.calls == 0


def test_a_json_envelope_is_still_validated_and_unwrapped() -> None:
    validator = CountingValidator()
    guardrail = ResponseGuardrail(None, validator=validator)  # type: ignore[arg-type]
    reply, fields = guardrail.validate_schema('{"response": "x"}')
    assert reply == "unwrapped" and fields["schema_valid"] and validator.calls == 1
