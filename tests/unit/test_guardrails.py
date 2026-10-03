"""Structured-output guarantees (LLD §9)."""

import pytest
from pydantic import BaseModel

from agent_platform.config import set_enabled
from agent_platform.guardrails import (
    GuardrailViolationError,
    LlmConstrainedGenerator,
    PydanticValidator,
    fail_loud,
)


class RefundCommand(BaseModel):
    order_id: int
    amount: float


class TestValidation:
    def test_accepts_well_formed_output(self) -> None:
        result = PydanticValidator().validate('{"order_id": 1, "amount": 10.5}', RefundCommand)
        assert result.valid and result.value == {"order_id": 1, "amount": 10.5}

    def test_rejects_output_missing_a_field(self) -> None:
        assert PydanticValidator("reject").validate('{"order_id": 1}', RefundCommand).valid is False

    def test_reject_mode_does_not_attempt_repair(self) -> None:
        result = PydanticValidator("reject").validate("```json\n{}\n```", RefundCommand)
        assert result.repaired is False

    def test_reports_the_validation_errors(self) -> None:
        result = PydanticValidator("reject").validate('{"order_id": 1}', RefundCommand)
        assert result.errors


class TestRepair:
    @pytest.mark.parametrize(
        "raw",
        [
            '```json\n{"order_id": 1, "amount": 2.0}\n```',
            'Sure! Here you go: {"order_id": 1, "amount": 2.0}',
            '{"order_id": 1, "amount": 2.0,}',
        ],
        ids=["code_fence", "prose_preamble", "trailing_comma"],
    )
    def test_repairs_the_common_malformations(self, raw: str) -> None:
        result = PydanticValidator("auto_repair").validate(raw, RefundCommand)
        assert result.valid and result.repaired

    def test_gives_up_on_output_that_is_not_json_at_all(self) -> None:
        """Guessing what the model meant is the failure this prevents."""
        result = PydanticValidator("auto_repair").validate("I cannot do that", RefundCommand)
        assert result.valid is False

    def test_valid_output_is_not_marked_repaired(self) -> None:
        result = PydanticValidator("auto_repair").validate(
            '{"order_id": 1, "amount": 2.0}', RefundCommand
        )
        assert result.repaired is False


class TestFailLoud:
    def test_raises_rather_than_coercing(self) -> None:
        """Silently coercing is how an agent acts on a value no model produced."""
        with pytest.raises(GuardrailViolationError):
            PydanticValidator("reject").validate_or_raise("nonsense", RefundCommand)

    def test_returns_the_validated_value(self) -> None:
        value = PydanticValidator().validate_or_raise(
            '{"order_id": 3, "amount": 1.0}', RefundCommand
        )
        assert value["order_id"] == 3

    async def test_the_decorator_validates_a_functions_output(self) -> None:
        @fail_loud(RefundCommand)
        async def produce() -> str:
            return '{"order_id": 9, "amount": 1.0}'

        assert (await produce())["order_id"] == 9

    async def test_the_decorator_raises_on_bad_output(self) -> None:
        @fail_loud(RefundCommand)
        async def produce() -> str:
            return "garbage"

        with pytest.raises(GuardrailViolationError):
            await produce()


class TestConstrainedGeneration:
    async def test_passes_the_schema_to_the_client(self, scripted_llm) -> None:
        scripted_llm.script = [{"content": '{"order_id": 1, "amount": 2.0}'}]
        generator = LlmConstrainedGenerator(scripted_llm)
        result = await generator.generate_constrained("refund order 1", RefundCommand)
        assert isinstance(result, RefundCommand)
        assert scripted_llm.requests[0].json_schema is not None

    async def test_a_best_effort_hint_is_still_validated(self, scripted_llm) -> None:
        """NIM does not honour the hint, so the result cannot be trusted unchecked."""
        scripted_llm.script = [{"content": "sorry, no"}]
        with pytest.raises(GuardrailViolationError):
            await LlmConstrainedGenerator(scripted_llm).generate_constrained("x", RefundCommand)

    def test_the_backend_is_configured_not_assumed(self, scripted_llm) -> None:
        assert LlmConstrainedGenerator(scripted_llm).backend == "grammar"


class TestASwitchedOffGuardrail:
    """Disabled means "do not enforce the schema", not "produce nothing".

    The defect this guards was quiet and total. A disabled validator
    returned `valid=True, value=None`, and every caller reads the value —
    so `NamespacedToolRegistry.build_call` saw None, treated it as a
    failure and refused the call. Switching guardrails off did not relax
    tool validation, it removed the agent's ability to call any tool at
    all, while logging "Tool arguments failed validation" with an empty
    list of what was wrong.
    """

    def test_it_returns_the_values_rather_than_nothing(self, restore_settings: None) -> None:
        set_enabled("guardrails", None, False)
        result = PydanticValidator().validate('{"order_id": 1, "amount": 5.0}', RefundCommand)
        assert result.valid
        assert result.value == {"order_id": 1, "amount": 5.0}

    def test_it_passes_values_the_schema_would_have_rejected(self, restore_settings: None) -> None:
        """Which is the whole of E28's contrast: with the guardrail gone, a
        wrongly-typed argument reaches the tool."""
        set_enabled("guardrails", None, False)
        result = PydanticValidator().validate('{"order_id": "not a number"}', RefundCommand)
        assert result.valid
        assert result.value == {"order_id": "not a number"}

    def test_unparseable_output_is_still_invalid(self, restore_settings: None) -> None:
        """Not parseable is not a schema question — there are no values here
        to pass on, whatever the guardrail was asked to do."""
        set_enabled("guardrails", None, False)
        assert not PydanticValidator().validate("{not json", RefundCommand).valid

    def test_a_bare_value_is_not_a_set_of_arguments(self, restore_settings: None) -> None:
        set_enabled("guardrails", None, False)
        assert not PydanticValidator().validate("[1, 2, 3]", RefundCommand).valid
