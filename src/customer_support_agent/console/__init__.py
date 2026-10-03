"""This agent, presented to the operator shell (LLD §20, §21).

`agent_platform` owns a console and a dashboard; it does not own a
customer, an order or a refund. So everything in the shell that has to
say one of those words is gathered here, behind the platform's
`AgentHost` protocol, and the platform composes it without ever naming
it.

The one place that names this agent is its own `config.yaml`
(`agent.host`), which is how a deployment chooses what it is running.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from agent_platform.commands.knowledge import KnowledgeBaseCommand
from agent_platform.commands.runs import CancelCommand
from agent_platform.commands.types import Command
from agent_platform.config import FeatureSet, get_settings
from agent_platform.host import HostAction
from agent_platform.logging import get_logger
from agent_platform.memory.semantic.rag import KnowledgeBase
from agent_platform.observability import StageReporter
from customer_support_agent.config import get_agent_settings
from customer_support_agent.console.commands import QueueCommand
from customer_support_agent.intent.router import load_intent_classifier

# Stage keys only this agent emits. The platform labels its own stages
# (`llm_call`, `gatekeeper_evaluation`); these are the ones that exist
# because this product has orders and refunds.
_STAGE_LABELS = {
    "order_lookup": "ORDER LOOKUP",
    "order_status_check": "ORDER STATUS",
    "refund_eligibility_check": "REFUND ELIGIBILITY",
}

# A stopped run the console offers to resume; one waiting on a reviewer resumes
# through `/queue approve|deny` instead.
_RESUMABLE_HERE = ("cancelled", "failed_resumable")

_TITLE = "Customer Support Agent"
_PLACEHOLDER = "Ask the support agent…  (/help for commands)"


class CustomerSupportHost:
    """The refund agent, as the shell sees it."""

    def title(self) -> str:
        return _TITLE

    def placeholder(self) -> str:
        return _PLACEHOLDER

    def greeting(self) -> str:
        """The first thing a conversation needs: who is speaking."""
        from customer_support_agent.presentation.auth import ask_for_code

        return ask_for_code()

    def prepare(self) -> None:
        """Load the intent classifier before the event loop opens.

        Native tokenizer initialization after Textual has opened its
        descriptors surfaces Python's `fds_to_keep` multiprocessing
        error on macOS. That is a runtime ordering problem, not an
        intent-routing decision, so it is solved by ordering.
        """
        import os

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._classifier = load_intent_classifier()
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

    # -- run controls: cancel the turn in flight, resume a stopped one -------------------

    def cancel_key(self) -> str:
        return get_agent_settings().console.cancel_key

    def cancel(self) -> bool:
        agent = self.agent
        return bool(agent is not None and agent.cancel())

    def actions(self, session_id: str) -> Sequence[HostAction]:
        """Cancel while a turn runs; Resume when this session's last run stopped part-way."""
        agent = self.agent
        if agent is None:
            return []
        console = get_agent_settings().console
        if agent.running:
            return [HostAction(console.cancel.label, "/cancel", True, console.cancel.variant)]
        last = agent.runs.latest_in_session(session_id)
        if last is None or last.status not in _RESUMABLE_HERE:
            return []
        node = last.next_node.split(",")[0] or "the last step"
        label = console.resume.label.format(node=node)
        return [HostAction(label, f"/resume {last.thread_id}", False, console.resume.variant)]

    def verified_subject(self, session_id: str) -> str | None:
        """Who is verified in a console session; the console's commands show only theirs."""
        agent = self.agent
        return agent.verified_subject(session_id) if agent is not None else None

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
        from agent_platform.commands.data import DreamCommand
        from customer_support_agent.graph.builder import build_dream_service

        return [
            QueueCommand(agent=lambda: self.agent),
            CancelCommand(lambda: self.agent),
            KnowledgeBaseCommand(lambda: self.knowledge_base),
            DreamCommand(build_dream_service),
        ]

    def stage_labels(self) -> Mapping[str, str]:
        return _STAGE_LABELS

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
        from agent_platform.durability import RunRegistry, create_checkpointer, purge_expired
        from agent_platform.persistence import SqliteConnectionPool, SqlitePlatformStore
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
            "Knowledge base start-up sync skipped: prepare() ran inside an event loop; use /kb sync"
        )

    @staticmethod
    def _opened_knowledge_base() -> KnowledgeBase:
        from agent_platform.evals import create_scoring_cascade
        from agent_platform.llm.encoders import create_reranker
        from agent_platform.memory.semantic.rag import create_knowledge_base

        return create_knowledge_base(reranker=create_reranker(fallback=create_scoring_cascade()))

    @staticmethod
    def _preload_entailment() -> None:
        """Load the local NLI model now, for the same descriptor reason.

        A model that cannot load does not stop the console: the claim
        checker latches to the lexical fallback and logs why.
        """
        from agent_platform.evals.entailment import preload_entailment_model

        preload_entailment_model()


__all__ = ["CustomerSupportHost"]
