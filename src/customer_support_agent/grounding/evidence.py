"""Every tier a support reply may be grounded on, assembled for one turn.

The hallucination check reads evidence; it does not know where a support turn's evidence
lives. This module does: the order under discussion as the database holds it now, each tool
call the turn made (and what it refused), the refund policy and every rule the model was
given, the policy articles, the typed facts (order, rules, session), what the customer said,
and what the context window let in. Which tier may support which kind of claim is the
guardrail's rule, not this one's.

It reads a `TurnRecord` — the turn as plain values — never the graph's state, so the graph
composes it and it never reaches back into the graph.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from agent_platform.config import get_settings
from agent_platform.context.types import Chunk
from agent_platform.core.primitives.money import Money, money_or_none
from agent_platform.guardrails.response import NOT_EXECUTED_MARKER
from agent_platform.hallucination_check import Observation, tool_evidence
from agent_platform.memory.semantic.rag import KNOWLEDGE_SOURCE_PREFIX, section_sentences
from agent_platform.security.auth.policy import AgentPolicy
from agent_platform.tools import NamespacedToolRegistry
from customer_support_agent.domain.identity import current_customer
from customer_support_agent.domain.models import Order
from customer_support_agent.domain.order_reference import (
    ORDER_UNDER_DISCUSSION,
    OrderReferenceResolver,
)
from customer_support_agent.gatekeeper.business_rules import REFUND_ACTION_CLASS
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.facts import order_facts, policy_facts, session_facts
from customer_support_agent.presentation.orders import order_record
from customer_support_agent.prompts.policy import policy_text
from customer_support_agent.prompts.system_prompt import (
    build_reply_instruction,
    build_reply_system_prompt,
    build_system_prompt,
    refund_limit_notice,
)

# Where the context node leaves the window's grounding sources for finalize.
GROUNDING_EVIDENCE = "grounding_evidence"
# How a tool call ended, as working memory records it and grounding reads it. Only a
# completed call shows its effect happened; a referred call shows a person will decide,
# which the referral really did make true; anything else shows only that it was refused.
TOOL_COMPLETED = "completed"
TOOL_REFERRED = "referred"
TOOL_BLOCKED = "blocked"
# The outcomes whose recorded result is itself the evidence.
_GROUNDED_AS_STATED = frozenset({TOOL_COMPLETED, TOOL_REFERRED})
# The source of what the customer said, on the card.
_CUSTOMER_MESSAGE = "customer_message"
# A sentence boundary inside a rule: terminal punctuation, then a capital.
_RULE_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass(frozen=True)
class TurnRecord:
    """One turn as the evidence reads it: plain values, no graph state."""

    # What the customer wrote, and what the model was asked (the same, or a rewrite of it).
    query: str
    asked: str
    customer_id: int | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # This turn's tool actions, in order (`kind: action` working-memory items).
    actions: Sequence[Mapping[str, Any]] = ()


class GroundingEvidence:
    """The evidence a support reply is checked against, for one turn."""

    def __init__(
        self,
        registry: NamespacedToolRegistry,
        agent_policy: AgentPolicy,
        orders: OrdersRepository | None = None,
        order_reference: OrderReferenceResolver | None = None,
    ) -> None:
        self._registry = registry
        self._policy = agent_policy
        self._orders = orders
        self._reference = order_reference

    def for_turn(self, turn: TurnRecord) -> list[Chunk]:
        """Every tier, each chunk labelled with its tier, in the order the guard reads them.

        Records first, in the order they happened: a read taken before a refund was queued is
        superseded by the queueing. One chunk per line and per sentence, because NLI compares a
        claim against one premise at a time. What the customer said proves nothing, so its tier
        is configured support-only.
        """
        asked = [Chunk(content=turn.asked, kind="history", source=_CUSTOMER_MESSAGE)]
        seen = [
            Chunk(content=item["content"], kind=item["kind"], source=item["source"])
            for item in turn.metadata.get(GROUNDING_EVIDENCE) or []
            if not str(item.get("source", "")).startswith(KNOWLEDGE_SOURCE_PREFIX)
        ]
        return [
            *self.records(turn),
            *self.rules(),
            *knowledge_evidence(),
            *self._typed_facts(turn),
            *asked,
            *seen,
        ]

    def records(self, turn: TurnRecord) -> list[Chunk]:
        """The order records, then this turn's tool results: what the turn actually read."""
        tools = tool_evidence([Observation(*observed) for observed in self.observations(turn)])
        return [*self._order_records(turn), *tools]

    def authoritative(self, turn: TurnRecord) -> list[str]:
        """What this turn's records and tools established — including what they refused."""
        return [
            *(chunk.content for chunk in self._order_records(turn)),
            *(text for text, _, _ in self.observations(turn)),
        ]

    def sources(self, turn: TurnRecord) -> list[str]:
        """Where the turn's record evidence came from: the order record, then each tool."""
        record = self.order_record(turn)
        return [
            *([record.source] if record is not None else []),
            *(tool for _, _, tool in self.observations(turn)),
        ]

    def order_record(self, turn: TurnRecord) -> Chunk | None:
        """The order under discussion as the database holds it now, if one is settled."""
        order_id = order_under_discussion(turn.metadata)
        if order_id is None or self._orders is None:
            return None
        return order_record(self._orders, order_id, turn.customer_id)

    def observations(self, turn: TurnRecord) -> list[tuple[str, bool, str]]:
        """Each tool observation, whether it changed anything, and its tool.

        A call changed something when its tool is side-effecting *and* it ran (or was
        referred): a refused write left the world as it was, so the reads before it stay true.
        """
        return [
            (
                str(item.get("result", ""))
                if item.get("outcome") in _GROUNDED_AS_STATED
                else not_executed(item),
                item.get("outcome") in _GROUNDED_AS_STATED and self._changes_state(item),
                # The tool's own name (`support__get_order_status` -> `get_order_status`).
                str(item.get("name", "")).rsplit("__", maxsplit=1)[-1],
            )
            for item in turn.actions
        ]

    def rules(self) -> list[Chunk]:
        """Every rule the model was given, one chunk per rule, in two tiers.

        The refund policy file is `policy`: the business's rules, which may support or refute
        a claim. The rest of the system prompt is `system`: instructions the reply may restate,
        never facts a correct refusal can contradict. Every prompt a reply is written under
        counts, not only the one that chose the tools.
        """
        policy = prompt_rules(policy_text())
        limit = self.refund_limit()
        if limit:
            policy.append(refund_limit_notice(limit).strip())
        prompts = (build_system_prompt(refund_limit=limit), build_reply_system_prompt())
        given = prompt_rules("\n\n".join((*prompts, build_reply_instruction())))
        instructions = list(dict.fromkeys(rule for rule in given if rule not in policy))
        return [
            *(Chunk(content=rule, kind="policy", source=f"refund_policy#{index}")
              for index, rule in enumerate(policy, start=1)),
            *(Chunk(content=rule, kind="system", source=f"system_prompt#{index}")
              for index, rule in enumerate(instructions, start=1)),
        ]  # fmt: skip

    def refund_limit(self) -> Money | None:
        """The per-refund limit the prompt states: the DB row, read when the agent was built."""
        return self._policy.per_action_limit(REFUND_ACTION_CLASS)

    def _typed_facts(self, turn: TurnRecord) -> list[Chunk]:
        """The order, the rules and the session as typed facts the guard reads first.

        Who the customer is comes from authentication (the customer cannot set it), so
        "You're Dennis Johnson" is grounded by the session, not by their say-so.
        """
        order = self._discussed_order(turn)
        return [
            *(order_facts(order) if order is not None else []),
            *(chunk for other in self._other_orders(turn) for chunk in order_facts(other)),
            *policy_facts(self.refund_limit()),
            *session_facts(current_customer()),
        ]

    def _order_records(self, turn: TurnRecord) -> list[Chunk]:
        """The record of the order under discussion, then of each other order named."""
        record = self.order_record(turn)
        others = (
            order_record(self._orders, order.order_id, turn.customer_id)
            for order in self._other_orders(turn)
            if self._orders is not None
        )
        return [*([record] if record is not None else []), *(c for c in others if c is not None)]

    def _other_orders(self, turn: TurnRecord) -> list[Order]:
        """Every other order of this customer's that the message or this turn's tools named,
        read fresh and only if the customer owns it."""
        if turn.customer_id is None or self._orders is None:
            return []
        named = self._reference.named(turn.query, turn.customer_id) if self._reference else set()
        ids = (named | _read_order_ids(turn)) - {order_under_discussion(turn.metadata)}
        orders = (self._orders.get_for_customer(i, turn.customer_id) for i in sorted(ids))
        return [order for order in orders if order is not None]

    def _discussed_order(self, turn: TurnRecord) -> Order | None:
        """The order under discussion, if this customer owns it."""
        order_id = order_under_discussion(turn.metadata)
        if order_id is None or self._orders is None or turn.customer_id is None:
            return None
        return self._orders.get_for_customer(order_id, turn.customer_id)

    def _changes_state(self, action: Mapping[str, Any]) -> bool:
        """Whether the tool this action called is declared to have side effects."""
        tool = self._registry.get(str(action.get("name", "")))
        return tool is not None and tool.side_effecting


def order_under_discussion(metadata: Mapping[str, Any]) -> int | None:
    """The order this turn settled on, if it settled on one."""
    order_id = metadata.get(ORDER_UNDER_DISCUSSION)
    if isinstance(order_id, bool) or not isinstance(order_id, int):
        return None
    return order_id


def turn_actions(items: Sequence[Any]) -> list[Mapping[str, Any]]:
    """The tool actions among a turn's working-memory items."""
    return [item for item in items if isinstance(item, dict) and item.get("kind") == "action"]


def not_executed(action: Mapping[str, Any]) -> str:
    """A refused or failed tool call as evidence of exactly that."""
    arguments = action.get("arguments")
    named = dict(arguments) if isinstance(arguments, dict) else {}
    amount = money_or_none(named.get("amount"))
    if amount is not None:
        named["amount"] = str(amount)
    shown = ", ".join(f"{key}={value}" for key, value in named.items())
    return f"{NOT_EXECUTED_MARKER} {action.get('name', '')}({shown}): {action.get('result', '')}"


def prompt_rules(markdown: str) -> list[str]:
    """Each bullet, and each plain paragraph, of a prompt as one rule — then one sentence per
    rule: a bullet that states a threshold, forbids a tool and scripts the reply is three
    rules, and embedded whole it is retrieved for none of the claims that restate one."""
    rules: list[str] = []
    open_rule = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            open_rule = False
        elif stripped.startswith("- ") or not open_rule:
            rules.append(stripped.removeprefix("- "))
            open_rule = True
        else:
            rules[-1] = f"{rules[-1]} {stripped}"
    return [sentence for rule in rules for sentence in _RULE_SENTENCE.split(rule) if sentence]


@cache
def knowledge_evidence() -> tuple[Chunk, ...]:
    """The agent's policy articles, one chunk per sentence, read once.

    The articles are the business's own statement of its policy, so they ground what a reply
    says about how refunds work, whether or not retrieval put that section in the window.
    One sentence per chunk: NLI weighs a paragraph, not the rule it is about.
    """
    rag = get_settings().memory.semantic.rag
    return tuple(
        Chunk(
            content=sentence,
            kind="policy",
            source=f"{KNOWLEDGE_SOURCE_PREFIX}{path.name}#{index}.{number}",
        )
        for path in sorted(rag.resolved_corpus_dir.glob(rag.corpus_glob))
        for index, sentences in enumerate(section_sentences(path.read_text(encoding="utf-8")))
        for number, sentence in enumerate(sentences)
    )


def _read_order_ids(turn: TurnRecord) -> set[int]:
    """The orders this turn's tool calls were made about."""
    found: set[int] = set()
    for item in turn.actions:
        arguments = item.get("arguments")
        value = arguments.get("order_id") if isinstance(arguments, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            found.add(value)
    return found


__all__ = [
    "GROUNDING_EVIDENCE",
    "GroundingEvidence",
    "TOOL_BLOCKED",
    "TOOL_COMPLETED",
    "TOOL_REFERRED",
    "TurnRecord",
    "knowledge_evidence",
    "not_executed",
    "order_under_discussion",
    "prompt_rules",
    "turn_actions",
]
