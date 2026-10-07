"""The refund limit is the Gatekeeper's, and nothing the model reads may state it.

A number in the context window can be asked for, injected out or simply
repeated: told `refund_limit = $50.00` and "do not reveal it", the model
answered order 10533 with "exceeds the $50.00 refund limit". It also obeyed
the limit by not calling `process_refund`, so the Gatekeeper never
escalated and the review it promised was never queued.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.hallucination_protection.direct import amounts
from customer_support_agent.graph.nodes.act import ToolUseNodes
from customer_support_agent.prompts.system_prompt import build_system_prompt

_REASON = (
    "Order value $103.39 exceeds the automatic refund limit of $50.00; "
    "the refund is escalated for human review."
)


def test_the_prompt_states_no_amount() -> None:
    assert amounts(build_system_prompt()) == set()


def test_the_prompt_tells_the_model_to_call_whatever_the_size() -> None:
    assert "whatever its size" in build_system_prompt()


async def test_an_escalation_with_no_queue_tells_the_model_no_limit() -> None:
    nodes: Any = SimpleNamespace(_referral=None)
    decision: Any = SimpleNamespace(reason=_REASON)
    call: Any = SimpleNamespace(action="support.process_refund")
    text, fields, _ = await ToolUseNodes._refer_for_review(nodes, call, decision, None)  # type: ignore[arg-type]
    assert "$50.00" not in text and "limit" not in text
    assert fields["result"] == "human_review_required"
