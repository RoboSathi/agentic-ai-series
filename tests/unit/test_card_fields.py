"""A card field earns its place: it explains what happened, is not on another card, and is read.

Removed because none of those held: the model and tools on the response
card (they are on the request card), `replayed`, `timed_out` and the
idempotency key on the tool result (the `execution_status` already says
completed, failed, timed_out or replayed), and `detected`/`redacted`/`mode`
on the PII card (one boolean twice, and a config constant).
"""

from types import SimpleNamespace

from customer_support_agent.presentation.reporting import llm_response_fields, tool_result_fields


class TestTheResponseCardDoesNotRepeatTheRequest:
    def test_no_model_and_no_tool_list(self) -> None:
        result = SimpleNamespace(
            model="m",
            finish_reason="stop",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=12.0,
            tool_calls=[],
            content="hello",
        )
        fields = llm_response_fields(result)
        assert "model" not in fields and "tools_offered" not in fields
        assert fields["total_tokens"] == 15 and fields["response"] == "hello"

    def test_an_unknown_finish_reason_is_not_printed(self) -> None:
        result = SimpleNamespace(
            model="m",
            finish_reason=None,
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            latency_ms=1.0,
            tool_calls=[],
            content="x",
        )
        assert "finish_reason" not in llm_response_fields(result)


class TestAToolResultSaysHowItEnded:
    def test_the_status_is_one_field_not_three(self) -> None:
        outcome = SimpleNamespace(
            status="completed",
            replayed=False,
            timed_out=False,
            duration_ms=3.14159,
            idempotency_key="k" * 32,
        )
        call = SimpleNamespace(action="support.process_refund")
        result = SimpleNamespace(success=True, data={"order_id": 10975}, error="")
        fields = tool_result_fields(call, result, outcome=outcome)
        assert fields["execution_status"] == "completed"
        assert not {"replayed", "timed_out", "idempotency_key"} & set(fields)
