"""`make refund-limit <amount>`: set how much this agent may refund without a person.

The limit lives in one place — the agent's persisted `AgentPolicy`
(`platform_agent_budgets`). The Gatekeeper reads it when the agent is
built; nothing the model sees states it. So this command is the only
thing that moves it, and a running agent picks the new value up on its
next build (restart, or a `/harness` rebuild).
"""

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone

from agent_platform.core.primitives.money import InvalidAmountError, Money
from agent_platform.logging import get_logger
from agent_platform.security.auth.policy import AgentPolicy, AgentPolicyRepository
from customer_support_agent.admin._store import agent_store
from customer_support_agent.config import get_agent_settings
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS

_logger = get_logger("persistence")


def policy_repository() -> AgentPolicyRepository:
    """This agent's persisted runtime policy, in its own database."""
    return AgentPolicyRepository(agent_store())


def set_refund_limit(repository: AgentPolicyRepository, limit: Money) -> tuple[Money, Money]:
    """Replace the refund per-action limit, keeping the window budget. Returns (old, new)."""
    if not limit:
        raise InvalidAmountError("the refund limit must be greater than zero")
    policy = repository.get()
    current = policy.limit_for(REFUND_ACTION_CLASS)
    if current is None:
        raise RuntimeError(
            f"Agent '{policy.agent_id}' declares no '{REFUND_ACTION_CLASS}' budget to change"
        )
    updated = policy.model_copy(
        update={
            "limits": {
                **policy.limits,
                REFUND_ACTION_CLASS: current.model_copy(update={"per_action": limit}),
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    saved: AgentPolicy = repository.upsert(updated)
    new = saved.per_action_limit(REFUND_ACTION_CLASS)
    assert new is not None
    return current.per_action, new


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("amount", help="the new per-refund limit, in dollars")
    args = parser.parse_args(argv)

    try:
        old, new = set_refund_limit(policy_repository(), Money.parse(args.amount))
    except InvalidAmountError as exc:
        parser.error(str(exc))
    _logger.info(
        f"Refund limit changed from {old} to {new}; takes effect on the agent's next build",
        database=str(get_agent_settings().resolved_db_path),
    )


if __name__ == "__main__":
    main()
