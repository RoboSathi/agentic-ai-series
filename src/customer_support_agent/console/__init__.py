"""This agent, presented to the operator shell (LLD §20, §21).

`agentic_commons` owns a console and a dashboard; it does not own a
customer, an order or a refund. So everything in the shell that has to
say one of those words is gathered here, behind the platform's
`AgentHost` protocol, and the platform composes it without ever naming
it.

The one place that names this agent is its own `config.yaml`
(`agent.host`), which is how a deployment chooses what it is running.
"""

from collections.abc import Mapping
from typing import Any

from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet
from agentic_commons.observability import StageReporter
from customer_support_agent.console.commands import (
    OrdersCommand,
    QueueCommand,
    UsersCommand,
)

# Stage keys only this agent emits. The platform labels its own stages
# (`llm_call`, `gatekeeper_evaluation`); these are the ones that exist
# because this product has orders and refunds.
_STAGE_LABELS = {
    "order_lookup": "ORDER LOOKUP",
    "order_status_check": "ORDER STATUS",
    "refund_eligibility_check": "REFUND ELIGIBILITY",
}

_TITLE = "Customer Support Agent"
_PLACEHOLDER = "Ask the support agent…  (/help for commands)"


class CustomerSupportHost:
    """The refund agent, as the shell sees it."""

    def title(self) -> str:
        return _TITLE

    def placeholder(self) -> str:
        return _PLACEHOLDER

    def prepare(self) -> None:
        """Load the intent classifier before the event loop opens.

        Native tokenizer initialization after Textual has opened its
        descriptors surfaces Python's `fds_to_keep` multiprocessing
        error on macOS. That is a runtime ordering problem, not an
        intent-routing decision, so it is solved by ordering.
        """
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._classifier = self._loaded_classifier()

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        """Compose the agent, reporting its stages to this feed."""
        from customer_support_agent.core import build_agent
        from customer_support_agent.core.intent import CustomerSupportIntentRouter

        classifier = getattr(self, "_classifier", None)
        router = (
            CustomerSupportIntentRouter(backend=classifier)
            if classifier is not None
            else CustomerSupportIntentRouter()
        )
        return build_agent(reporter=reporter, intent_router=router, features=features)

    def commands(self) -> list[Command]:
        """The slash commands that read this agent's own tables."""
        return [UsersCommand(), OrdersCommand(), QueueCommand()]

    def stage_labels(self) -> Mapping[str, str]:
        return _STAGE_LABELS

    def session_rows(self, agent: Any) -> list[tuple[str, str]]:
        """Who `/session` is talking to, in this agent's vocabulary."""
        from customer_support_agent.core.identity import current_customer

        customer = current_customer()
        shown = f"{customer.name} (#{customer.customer_id})" if customer else "not verified"
        return [("customer", shown)]

    @staticmethod
    def _loaded_classifier() -> Any:
        """The shared intent classifier, warmed up, or None if unavailable.

        A classifier that cannot load is not a reason the console cannot
        open: the router falls back to its deterministic phrase rules,
        which is a narrower agent rather than a broken one.
        """
        from agentic_commons.intent import create_intent_classifier

        try:
            classifier = create_intent_classifier()
            classifier.ensure_loaded_sync()
        except Exception:
            return None
        return classifier


__all__ = ["CustomerSupportHost"]
