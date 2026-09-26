"""This agent, presented to the operator shell (LLD §20, §21).

`agentic_commons` owns a console and a dashboard; it does not own a
customer, an order or a refund. So everything in the shell that has to
say one of those words is gathered here, behind the platform's
`AgentHost` protocol, and the platform composes it without ever naming
it.

The one place that names this agent is its own `config.yaml`
(`agent.host`), which is how a deployment chooses what it is running.
"""

import asyncio
from collections.abc import Mapping
from typing import Any

from agentic_commons.commands.knowledge import KnowledgeBaseCommand
from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet, get_settings
from agentic_commons.logging import get_logger
from agentic_commons.memory.semantic.rag import KnowledgeBase
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
        self._preload_entailment()
        self._knowledge_base = self._synced_knowledge_base()
        self._recover_runs()

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        """Compose the agent, reporting its stages to this feed."""
        from customer_support_agent.graph import build_agent
        from customer_support_agent.intent.router import CustomerSupportIntentRouter

        classifier = getattr(self, "_classifier", None)
        router = (
            CustomerSupportIntentRouter(backend=classifier)
            if classifier is not None
            else CustomerSupportIntentRouter()
        )
        self._agent = build_agent(
            reporter=reporter,
            intent_router=router,
            features=features,
            knowledge_base=self.knowledge_base,
        )
        return self._agent

    @property
    def agent(self) -> Any:
        """The agent this host last built — what `/queue` resumes through."""
        return getattr(self, "_agent", None)

    @property
    def knowledge_base(self) -> KnowledgeBase | None:
        """The knowledge base opened at start-up, if it opened."""
        return getattr(self, "_knowledge_base", None)

    def commands(self) -> list[Command]:
        """The slash commands that read this agent's own tables and corpus."""
        return [
            UsersCommand(),
            OrdersCommand(),
            QueueCommand(agent=lambda: self.agent),
            KnowledgeBaseCommand(lambda: self.knowledge_base),
        ]

    def stage_labels(self) -> Mapping[str, str]:
        return _STAGE_LABELS

    def session_rows(self, agent: Any) -> list[tuple[str, str]]:
        """Who `/session` is talking to, in this agent's vocabulary."""
        from customer_support_agent.domain.identity import current_customer

        customer = current_customer()
        shown = f"{customer.name} (#{customer.customer_id})" if customer else "not verified"
        return [("customer", shown)]

    @staticmethod
    def _synced_knowledge_base() -> KnowledgeBase | None:
        """Open the knowledge index and bring it up to date, before the loop opens.

        Synced whatever the feature posture, because `/harness` can turn
        semantic memory on without a restart and should find it ready.
        A corpus that cannot be indexed narrows the agent — it answers
        from its system prompt — rather than keeping the console shut.
        """
        try:
            if get_settings().memory.semantic.rag.ingest_on_startup:
                CustomerSupportHost._sync_outside_event_loop()
            # A second instance for the console. The one that synced holds
            # HTTP clients bound to the start-up event loop, now closed —
            # reused, every embedding and rerank call after it would fail
            # over to its fallback for the rest of the session.
            return CustomerSupportHost._opened_knowledge_base()
        except Exception as exc:
            get_logger("customer_support_agent").warning(
                "Knowledge base unavailable; continuing without it", error=str(exc)
            )
            return None

    @staticmethod
    def _recover_runs() -> None:
        """Account for runs the last process left unfinished, once, at start-up.

        A run still marked `running` has no process executing it any more;
        it becomes resumable from its last checkpoint. Completed runs past
        the retention window are deleted — checkpoints hold what customers
        said, and are not kept longer than configured.
        """
        from agentic_commons.durability import RunRegistry, create_checkpointer, purge_expired
        from agentic_commons.persistence import SqliteConnectionPool, SqlitePlatformStore
        from customer_support_agent.config import get_agent_settings

        store = SqlitePlatformStore(
            SqliteConnectionPool(get_agent_settings().resolved_db_path),
            agent_id=get_settings().agent.id,
        )
        registry = RunRegistry(store)
        orphans = registry.recover_orphans()
        purged = purge_expired(create_checkpointer(), registry)
        get_logger("customer_support_agent").info(
            "Durable runs recovered at start-up",
            orphaned=[record.thread_id for record in orphans],
            resumable=len(registry.resumable(get_settings().commands.default_limit)),
            purged=purged,
        )

    @staticmethod
    def _sync_outside_event_loop() -> None:
        """Run the start-up sync — which `prepare` promises happens before the loop.

        Called from inside a running loop (a test, an embedding host), a
        nested `asyncio.run` would fail and leak the coroutine; the sync
        is skipped with a log line instead, and `/kb sync` does it later.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(CustomerSupportHost._opened_knowledge_base().sync())
            return
        get_logger("customer_support_agent").warning(
            "Knowledge base start-up sync skipped: prepare() ran inside an event loop; "
            "use /kb sync"
        )

    @staticmethod
    def _opened_knowledge_base() -> KnowledgeBase:
        from agentic_commons.evals import create_scoring_cascade
        from agentic_commons.llm.encoders import create_reranker
        from agentic_commons.memory.semantic.rag import create_knowledge_base

        return create_knowledge_base(reranker=create_reranker(fallback=create_scoring_cascade()))

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

    @staticmethod
    def _preload_entailment() -> None:
        """Load the local NLI model now, for the same descriptor reason.

        A model that cannot load does not stop the console: the claim
        checker latches to the lexical fallback and logs why.
        """
        from agentic_commons.evals.entailment import preload_entailment_model

        preload_entailment_model()


__all__ = ["CustomerSupportHost"]
