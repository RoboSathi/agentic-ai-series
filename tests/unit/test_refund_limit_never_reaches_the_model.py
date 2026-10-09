"""The refund limit is the Gatekeeper's, and nothing the model reads may state it.

A number in the context window can be asked for, injected out or simply
repeated: told `refund_limit = $50.00` and "do not reveal it", the model
answered order 10533 with "exceeds the $50.00 refund limit". It also obeyed
the limit by not calling `process_refund`, so the Gatekeeper never
escalated and the review it promised was never queued.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.config import get_settings
from agent_platform.hallucination_protection.direct import amounts
from agent_platform.memory import ProceduralStore
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.skills import build_agent_instructions
from customer_support_agent.graph.nodes.execute import ExecutionNodes
from customer_support_agent.prompts.system_prompt import build_system_prompt, instruction_corpus

_REASON = (
    "Order value $103.39 exceeds the automatic refund limit of $50.00; "
    "the refund is escalated for human review."
)


def test_no_instruction_states_an_amount() -> None:
    """The profile, every procedure, the policy and the reply prompts."""
    assert amounts(instruction_corpus()) == set()


def test_a_refund_turn_tells_the_model_to_call_whatever_the_size(
    pool: SqliteConnectionPool,
) -> None:
    instructions = build_agent_instructions(
        get_settings().skills, store=ProceduralStore(pool), audit=None, reporter=None
    )
    composed = instructions.compose(frozenset({"intent:REQUEST_REFUND"}), "refund order 10533")
    prompt = build_system_prompt(composed.text)
    assert "whatever its size" in prompt
    assert amounts(prompt) == set()


async def test_an_escalation_with_no_queue_tells_the_model_no_limit() -> None:
    nodes: Any = SimpleNamespace(_referral=None)
    decision: Any = SimpleNamespace(reason=_REASON)
    call: Any = SimpleNamespace(action="support.process_refund")
    text, fields, _ = await ExecutionNodes._refer_for_review(nodes, call, decision, None)  # type: ignore[arg-type]
    assert "$50.00" not in text and "limit" not in text
    assert fields["result"] == "human_review_required"
