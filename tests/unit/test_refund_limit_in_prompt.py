"""The refund limit reaches the model as one prompt sentence, rendered from the DB.

The prompt states the limit so that the model holds to it on a plain
request — and so that a persuasive customer can be shown talking it past
the rule while the Gatekeeper is off. Authored text (profile, procedures,
policy file, reply prompts) still never carries an amount: the number has
one source, the `platform_agent_budgets` row, and one sentence that renders it.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.config import get_settings
from agent_platform.hallucination_check.direct import amounts
from agent_platform.memory import ProceduralStore
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.primitives.money import Money
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


def test_a_refund_turn_states_the_db_limit_and_only_it(pool: SqliteConnectionPool) -> None:
    instructions = build_agent_instructions(
        get_settings().skills, store=ProceduralStore(pool), audit=None, reporter=None
    )
    composed = instructions.compose(frozenset({"intent:REQUEST_REFUND"}), "refund order 10533")
    limit = Money.from_cents(5000)
    prompt = build_system_prompt(composed.text, refund_limit=limit)
    assert str(limit) in prompt
    assert amounts(prompt) == amounts(str(limit))


def test_no_declared_limit_states_no_limit() -> None:
    assert build_system_prompt(refund_limit=None) == build_system_prompt()


async def test_an_escalation_with_no_queue_tells_the_model_no_limit() -> None:
    nodes: Any = SimpleNamespace(_referral=None)
    decision: Any = SimpleNamespace(reason=_REASON)
    call: Any = SimpleNamespace(action="support.process_refund")
    text, fields, _ = await ExecutionNodes._refer_for_review(nodes, call, decision, None)  # type: ignore[arg-type]
    assert "$50.00" not in text and "limit" not in text
    assert fields["result"] == "human_review_required"
