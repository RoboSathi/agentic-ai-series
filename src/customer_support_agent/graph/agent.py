"""The refund agent: a graph, and the subsystems its nodes call (LLD §21).

Nodes, in order:

    authenticate → choose_order → screening → classify_intent →
    assemble_context → propose ⇄ execute → finalize [→ await_approval]

Every turn is a durable run: its own checkpointed thread. A node that
fails leaves the thread resumable from the node that did not finish, and
a turn that queued a refund for a person pauses in `await_approval` until
the reviewer's decision resumes it.

The `propose ⇄ execute` cycle is the **Tool-Use Loop**, bounded by
`max_tool_iterations`. It is not ReAct: the model proposes a typed tool
call rather than a natural-language thought, and the harness — not the
model — decides whether that call happens.

Every node calls into `agent_platform` and owns no platform logic of its
own. That is the invariant this package exists to hold: the running
product is a *composition*, and a second LLM client or a second auth check
appearing here would be the duplication regression found and fixed once
already.

This module holds only what belongs to the agent as a whole — its
dependencies, one turn's lifecycle, and the conversation it is carrying.
Each node group lives in `nodes/`, and the wiring in `edges.py`, because
one class with sixty methods is not a design, it is a file. Each group is
its own object, built from one `NodeKit` (`kit.py`) and binding only what
its nodes read; a group that needs another's work is handed that group.

The dependencies arrive through `build_agent(...)` as one `CsaParts`
(`parts.py`), grouped by concern. Nothing is constructed inline, so a test
can substitute a scripted LLM and an in-memory database without patching a
single import.
"""

import uuid
from collections.abc import Mapping
from contextlib import AbstractContextManager
from typing import Any

from agent_platform.config import get_settings
from agent_platform.durability import (
    CheckpointInfo,
    DurableGraphRunner,
    ResumeResult,
    RunOutcome,
    RunRegistry,
)
from agent_platform.durability.stop_report import stop_report
from agent_platform.llm.types import Message, TokenUsage
from agent_platform.memory import (
    WorkingMemory,
    WorkingMemoryStore,
)
from agent_platform.memory.procedural.skills import (
    SkillCatalog,
)
from agent_platform.memory.write_pipeline import MemoryWritePipelineImpl
from agent_platform.observability.langsmith import (
    TurnTrace,
    turn_trace,
)
from agent_platform.observability.units import mark_cancelled
from agent_platform.security.pii.active import admit_if_on, tokenise_if_on
from agent_platform.security.pii.source import ADMITTED_KEY, Admitted
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.identity import (
    Customer,
    bound_customer,
    current_customer,
)
from customer_support_agent.graph.edges import CsaNodes, GraphWiring
from customer_support_agent.graph.kit import NodeKit
from customer_support_agent.graph.nodes.authenticate import JUST_VERIFIED
from customer_support_agent.graph.parts import CsaParts
from customer_support_agent.graph.state import AgentState
from customer_support_agent.memory.signals import turn_signals
from customer_support_agent.memory.turn_memory import FinishedTurn
from customer_support_agent.observability.accounting import TurnOutcome
from customer_support_agent.persistence.repositories import order_turn_scope

# Set by screening when a message is quarantined as hostile.
_QUARANTINED = "security_quarantine"

# How a turn ended, in the turn summary's words.
_ANSWERED_OUTCOME = "answered"
_WAITING_OUTCOME = "waiting for human review"
_FAILED_OUTCOME = "failed — resumable"
_RESUMED_OUTCOME = "resumed and answered"
_REVIEWED_OUTCOME = "review decision delivered"
_CANCELLED_OUTCOME = "cancelled — resumable"


class CustomerSupportAgent:
    """The refund agent: one turn's lifecycle over a graph of node groups.

    The node groups (`nodes/`) are separate objects built from one `NodeKit`; the wiring
    (`edges.py`) compiles them into the graph. This shell holds only what belongs to the
    agent as a whole: the turn lifecycle, durable runs, and the conversation it carries.
    """

    def __init__(self, parts: CsaParts) -> None:
        kit = NodeKit(parts)
        self._kit = kit
        self._nodes = CsaNodes.from_kit(kit)
        self._conversation = kit.conversation
        self._sessions = kit.conversation.sessions
        self._security = kit.security
        self._working_memory = kit.working_memory
        self._long_term = kit.long_term
        self._instructions = kit.instructions
        self._authenticator = kit.authenticator
        self._reporter = kit.reporter
        self._turn_memory = kit.turn_memory
        self._trace_exporter = kit.trace_exporter
        self._accountant = kit.accountant
        self._actor = kit.actor
        self._governor = kit.governor
        self._steps = kit.steps
        self._logger = kit.logger
        # Durable execution. The graph is compiled against the checkpointer,
        # so the runner and the graph must share it.
        self._runs = kit.runs
        # The durable thread running now, so the console can cancel it.
        self._active_thread: str | None = None
        self._graph = GraphWiring(kit, self._nodes).build()
        self._runner = (
            DurableGraphRunner(self._graph, kit.checkpointer, kit.runs)
            if kit.checkpointer is not None and kit.runs is not None
            else None
        )

    def verified_subject(self, session_id: str) -> str | None:
        """Who has proved who they are in this session, as the records' `subject_id`."""
        customer = self._sessions.get(session_id)
        return str(customer.customer_id) if customer is not None else None

    def adopt_conversation(self, previous: "CustomerSupportAgent") -> None:
        """Take over the conversation a previous build was holding.

        Who has verified, the order menu they were shown, the transcript
        and the observations a follow-up refers to all belong to the
        customer — not to the wiring. A rebuild swaps which controls are
        wired, and `/harness off` exists precisely to be asked the *same*
        question again; an agent that dropped the session would demand
        the customer code back before the comparison could be made.
        """
        self._conversation.adopt(previous._conversation)
        self._adopt_working_memory(previous._working_memory)

    def _adopt_working_memory(self, previous: WorkingMemoryStore | None) -> None:
        """Copy the previous build's sessions into this build's own store.

        Copied, never re-pointed. The context source and the graph nodes
        were handed this build's store at composition; swapping only the
        agent's reference left them reading an empty one, so every prompt
        after a `/harness` lost the verified customer and the transcript
        while the panel — reading the swapped reference — still showed both.
        """
        if previous is None or self._working_memory is None:
            return
        for session_id in previous.sessions():
            self._working_memory.restore(session_id, previous.get(session_id))

    async def handle(
        self, query: str, session_id: str = "", customer: Customer | None = None
    ) -> AgentState:
        """Run one customer query end to end, on that customer's behalf.

        Prior turns are replayed so a follow-up has something to refer
        to. Without them "what about 10010" is unanswerable: the subject
        lives in the previous turn, not in the words the customer just
        typed.

        The customer is bound before any tool can run, and comes from
        authentication rather than from anything the model says. Passing
        one here binds it; passing none leaves whatever the caller already
        bound, so a web handler that authenticates upstream does not have
        to thread it through. With nothing bound either way, the tools
        scope to nothing — an unauthenticated request sees no orders
        rather than every order.
        """
        if customer is not None:
            # An explicitly supplied customer means the caller
            # authenticated upstream — an SSO session, a signed token, a
            # test fixture. The conversation does not re-ask for a code
            # it has already been given the answer to.
            if session_id:
                self._sessions[session_id] = customer
        # Bind identity *before* the graph runs, not inside a node.
        # Context variables copy into a task when it starts and do not
        # propagate back out, and LangGraph runs each node in its own
        # task — so a customer bound inside `authenticate` would be
        # invisible to `execute`, and every lookup would silently scope to
        # nobody.
        customer = customer or self._sessions.get(session_id) or current_customer()
        # The customer's words become tokens before anything holds them — the trace, the
        # checkpoint, the run registry, a log line (`security.pii.source`).
        admitted = admit_if_on(query)
        # One turn, one trace: every line any subsystem logs for this turn
        # carries it, with the agent, the session and the caller's channel.
        # And one read of each order: every stage of the turn shares it.
        with (
            bound_customer(customer),
            self._turn_trace(session_id, {"query": admitted.text}) as trace,
            order_turn_scope(),
        ):
            state = await self._handle_bound(admitted, session_id, customer)
            trace.annotate(_turn_facts(state))
            trace.finish({"response": state.response})
            return state

    async def _handle_bound(
        self, admitted: Admitted, session_id: str, customer: Customer | None
    ) -> AgentState:
        """One turn, with the customer already bound for exactly its duration."""
        self._governor.begin_turn()
        self._steps.begin(None)
        await self._recover_conversation(session_id)
        initial = AgentState(
            query=admitted.text,
            metadata={ADMITTED_KEY: admitted.record()},
            session_id=session_id,
            actor=self._actor,
            customer_id=customer.customer_id if customer else None,
            working_memory=(
                self._working_memory.get(session_id)
                if self._working_memory is not None
                else WorkingMemoryStore().get(session_id)
            ),
        )
        thread_id = f"{session_id or 'default'}:{uuid.uuid4().hex[:12]}"
        self._begin_turn(thread_id, session_id)
        try:
            state, outcome = await self._run_turn(initial, thread_id)
        except Exception:
            self._end_turn("failed", _FAILED_OUTCOME, session_id, run_status="failed_resumable")
            raise
        if outcome is not None and outcome.status == "cancelled":
            return self._stopped(state, outcome)
        # Inside the turn's bracket, so its memory writes are counted in the
        # summary card rather than trailing it.
        self._complete_turn(state)
        self._accountant.settle(self._score(state))
        waiting = outcome is not None and outcome.status == "interrupted"
        self._end_turn(
            "waiting" if waiting else "completed",
            _WAITING_OUTCOME if waiting else _ANSWERED_OUTCOME,
            session_id,
            run_status=outcome.status if outcome else "not_durable",
            checkpoints=outcome.checkpoints if outcome else 0,
        )
        return state

    def _preview(self, query: str) -> str:
        """The turn as the run registry keeps it: PII redacted, and any typed code masked.

        A customer code is a credential, not PII, so redaction alone left it in
        `platform_runs` in plain text.
        """
        redacted = tokenise_if_on(query)
        if self._authenticator is None:
            return redacted
        return self._authenticator.mask_code(redacted)

    async def _run_turn(
        self, initial: AgentState, thread_id: str
    ) -> tuple[AgentState, RunOutcome | None]:
        """Run one turn — as its own checkpointed thread when durable."""
        if self._runner is None:
            return AgentState.model_validate(await self._graph.ainvoke(initial)), None
        self._active_thread = thread_id
        try:
            outcome = await self._runner.start(
                thread_id,
                initial,
                session_id=initial.session_id,
                subject_id=str(initial.customer_id or ""),
                preview=self._preview(initial.query),
                reporter=self._reporter,
            )
        finally:
            self._active_thread = None
        return AgentState.model_validate(outcome.values), outcome

    @property
    def running(self) -> bool:
        """Whether a durable turn is in flight (what a Cancel button stops)."""
        return self._active_thread is not None

    def cancel(self) -> bool:
        """Stop the turn in flight; its checkpoints stay, so it can be resumed."""
        thread = self._active_thread
        return bool(self._runner is not None and thread and self._runner.cancel(thread))

    def _stopped(self, state: AgentState, outcome: RunOutcome) -> AgentState:
        """A cancelled turn: where it stopped, how far it got, and the two ways on."""
        step = outcome.next_nodes[0] if outcome.next_nodes else ""
        reply = stop_report(step, self.describe_progress(state.model_dump()))
        mark_cancelled()
        self._end_turn(
            "warning",
            _CANCELLED_OUTCOME,
            state.session_id,
            run_status=outcome.status,
            checkpoints=outcome.checkpoints,
        )
        return state.model_copy(update={"response": reply})

    def describe_progress(self, values: dict[str, Any]) -> dict[str, Any]:
        """How far a turn got: what was asked, who is verified, what was decided, which tools ran."""
        state = AgentState.model_validate(values)
        words = get_agent_settings().messages.cancelled
        labels = words.labels
        tools = [
            f"{item.get('name')} → {item.get('outcome', '')}"
            for item in state.working_memory.current_turn
            if isinstance(item, dict) and item.get("kind") == "action"
        ]
        return {
            labels["request"]: self._security.redact(state.query) or words.none,
            labels["customer"]: f"#{state.customer_id}" if state.customer_id else words.unverified,
            labels["decided"]: state.decisions or words.none,
            labels["tools"]: tools or words.none,
        }

    async def abort(self, thread_id: str) -> ResumeResult:
        """End a stopped turn for good: its record is kept, nothing is remembered as answered."""
        if self._runner is None:
            raise RuntimeError("This agent was built without durable execution")
        outcome = await self._runner.abort(thread_id, self._reporter)
        session = str(outcome.values.get("session_id", ""))
        return ResumeResult(
            thread_id=thread_id,
            session_id=session,
            status=outcome.status,
            response=get_settings().durability.stop_report.aborted,
        )

    def _begin_turn(self, thread_id: str, session_id: str) -> None:
        if self._reporter is not None:
            self._reporter.begin_turn(thread=thread_id, session=session_id)

    def _end_turn(self, status: Any, outcome: str, session_id: str = "", **log_fields: Any) -> None:
        if self._reporter is None:
            return
        if self._long_term is not None:
            log_fields["curator_pending"] = len(self._long_term.pipeline.pending_with_ids())
        self._reporter.end_turn(
            status, outcome, working_memory=self._held(session_id), **log_fields
        )

    def _held(self, session_id: str) -> dict[str, str]:
        """What working memory holds for this session, redacted once, for the turn summary.

        The facts it keeps (who is verified, what is being asked, which order
        is under discussion) and how many turns of conversation it carries.
        The conversation itself is on the LLM Request card, not repeated here.
        """
        if self._working_memory is None or not session_id:
            return {}
        memory = self._working_memory.get(session_id)
        held = {str(key): tokenise_if_on(str(value)) for key, value in memory.state.items()}
        turns = len(memory.history) // 2
        return {
            **held,
            **({"history": f"{turns} turn{'' if turns == 1 else 's'}"} if turns else {}),
        }

    @property
    def runs(self) -> RunRegistry:
        """Every durable run this agent has made (the `DurableAgent` contract)."""
        if self._runs is None:
            raise RuntimeError("This agent was built without durable execution")
        return self._runs

    async def checkpoint_history(self, thread_id: str) -> list[CheckpointInfo]:
        """One thread's checkpoints, newest first."""
        if self._runner is None:
            return []
        return await self._runner.history(thread_id, get_settings().commands.default_limit)

    async def resume(self, thread_id: str, decision: Any | None = None) -> ResumeResult:
        """Continue a failed or waiting run from its last checkpoint.

        What a run needs that is not graph state is re-bound first: who
        the customer is (from the record the run already authenticated
        against) and a fresh per-turn loop control. The pending node then
        runs again; any side effect it had already committed replays from
        its idempotency record instead of happening twice.
        """
        if self._runner is None or self._runs is None:
            raise RuntimeError("This agent was built without durable execution")
        record = self._runs.get(thread_id)
        waiting = record is not None and record.status == "interrupted"
        prior = AgentState.model_validate(await self._runner.values(thread_id))
        customer = self._rebind(prior)
        inputs = {"resume": thread_id, "decision": str(decision or "")}
        with (
            bound_customer(customer),
            self._turn_trace(prior.session_id, inputs, message=False) as trace,
        ):
            result = await self._resume_bound(thread_id, decision, prior, waiting)
            trace.annotate({"resumed": True, "run_status": result.status})
            trace.finish({"response": result.response})
            return result

    def _turn_trace(
        self, session_id: str, inputs: Mapping[str, Any], message: bool = True
    ) -> AbstractContextManager[TurnTrace]:
        """A fresh trace for one turn (a resume is a turn of its own, with no message in it)."""
        return turn_trace(
            self._trace_exporter,
            agent=get_settings().agent.id,
            session=session_id,
            inputs=inputs,
            message=message,
        )

    async def _resume_bound(
        self, thread_id: str, decision: Any | None, prior: AgentState, waiting: bool
    ) -> ResumeResult:
        """The resumed run, with its customer bound for exactly its duration."""
        assert self._runner is not None
        self._begin_turn(thread_id, prior.session_id)
        self._active_thread = thread_id
        try:
            outcome = await self._runner.resume(thread_id, decision, self._reporter)
        except Exception:
            self._end_turn(
                "failed",
                _FAILED_OUTCOME,
                prior.session_id,
                run_status="failed_resumable",
                resumed=True,
            )
            raise
        finally:
            self._active_thread = None
        state = AgentState.model_validate(outcome.values)
        if outcome.status == "cancelled":
            stopped = self._stopped(state, outcome)
            return ResumeResult(
                thread_id=thread_id,
                session_id=stopped.session_id,
                status=outcome.status,
                response=stopped.response,
            )
        if waiting:
            reply = str(state.metadata.get("follow_up") or "")
            self._turn_memory.note_follow_up(state.session_id, reply)
        else:
            reply = state.response
            self._complete_turn(state)
            self._accountant.settle(self._score(state))
        self._end_turn(
            "completed",
            _REVIEWED_OUTCOME if waiting else _RESUMED_OUTCOME,
            state.session_id,
            run_status=outcome.status,
            checkpoints=outcome.checkpoints,
            resumed=True,
        )
        return ResumeResult(
            thread_id=thread_id, session_id=state.session_id, status=outcome.status, response=reply
        )

    def _rebind(self, prior: AgentState) -> Customer | None:
        """Restore the conversation a checkpointed run belongs to; return its customer.

        The caller binds the customer for the resumed run only. Binding it
        here, in the caller's own context, outlived the resume and handed
        the next session an identity it never proved.
        """
        customer = self._sessions.get(prior.session_id)
        if customer is None and prior.customer_id is not None and self._authenticator is not None:
            customer = self._authenticator.customer_for(prior.customer_id)
            if customer is not None:
                self._sessions[prior.session_id] = customer
        # Resumed exactly where it stopped: what it already did is still a repeat.
        self._governor.restore(prior.loop)
        if (
            self._working_memory is not None
            and not self._working_memory.get(prior.session_id).history
        ):
            self._working_memory.restore(prior.session_id, prior.working_memory)
        return customer

    def deliver_follow_up(self, session_id: str, reply: str) -> None:
        """Add a reviewer's decision to the conversation it answers (a reply review)."""
        self._turn_memory.note_follow_up(session_id, reply)

    async def _recover_conversation(self, session_id: str) -> None:
        """After a restart, pick a session's conversation back up from its checkpoints.

        Working memory lives in the process; the checkpoints outlive it.
        A session with no history in memory but a finished run on disk is
        a conversation the process forgot, not a new one.
        """
        if self._working_memory is None or self._runs is None or self._runner is None:
            return
        if self._working_memory.get(session_id).history:
            return
        record = self._runs.latest_completed(session_id)
        if record is None:
            return
        values = await self._runner.values(record.thread_id)
        if values.get("working_memory") is None:
            return
        self._working_memory.restore(
            session_id, WorkingMemory.model_validate(values["working_memory"])
        )
        self._logger.info(
            "Conversation recovered from checkpoint", session=session_id, thread=record.thread_id
        )

    def _score(self, state: AgentState) -> TurnOutcome:
        """Read the finished turn as an outcome the SLOs can observe.

        Everything here is derived from the state the graph already
        produced, rather than counted as it went. Counters threaded
        through eight nodes get updated in seven of them; a single read
        at the end cannot drift from what actually happened.
        """
        decisions = state.decisions
        return TurnOutcome(
            session_id=state.session_id,
            answered=bool(state.response) and not state.escalated,
            escalated=state.escalated,
            unauthorized_attempts=sum(1 for line in decisions if "not offered" in line),
            policy_refusals=sum(
                1 for line in decisions if "refused" in line and "not offered" not in line
            ),
            tool_calls=sum(
                1
                for item in state.working_memory.current_turn
                if isinstance(item, dict) and item.get("kind") == "action"
            ),
            tool_failures=sum(1 for line in decisions if ": failed" in line),
            retrievals=sum(1 for line in decisions if line.startswith("context:")),
            evaluator_calls=sum(1 for line in decisions if line.startswith("grounding:")),
            llm_calls=state.tool_rounds,
            usage=TokenUsage.model_validate(state.metadata.get("last_usage", {})),
            hallucination_rate=float(state.metadata.get("hallucination_rate", 0.0)),
            loop_aborted=state.loop.outcome == "finalized_on_cap",
            recoveries_attempted=int(state.metadata.get("recoveries_attempted", 0)),
            recoveries_succeeded=int(state.metadata.get("recoveries_succeeded", 0)),
        )

    @property
    def working_memory(self) -> WorkingMemoryStore | None:
        """Working memory available to the live agent and operator console."""
        return self._working_memory

    @property
    def skill_catalog(self) -> SkillCatalog | None:
        """The skills `/skills` lists (None when procedural memory is withheld)."""
        return self._instructions.catalog

    @property
    def curator(self) -> MemoryWritePipelineImpl | None:
        """The governed write pipeline a curator approves a queued lesson through."""
        return self._long_term.pipeline if self._long_term is not None else None

    def history_for(self, session_id: str) -> list[Message]:
        """The bounded conversation history from common working memory."""
        if self._working_memory is None:
            return []
        history = self._working_memory.get(session_id).history
        return [Message(**item) if isinstance(item, dict) else item for item in history]

    def _complete_turn(self, state: AgentState) -> None:
        """Hand the finished turn to memory: what was asked, read and kept, and why not."""
        remembered = state.metadata.get("memory_response")
        self._turn_memory.complete(
            FinishedTurn(
                session_id=state.session_id,
                customer_id=state.customer_id,
                spoken=state.model_query or state.query,
                response=state.response,
                # What the customer read, unless the turn left a compact form of it (a table
                # of every order is far larger on screen than in substance).
                remembered=remembered
                if isinstance(remembered, str) and remembered
                else state.response,
                scope=sorted(turn_signals(state.metadata)),
                quarantined=bool(state.metadata.get(_QUARANTINED)),
                login=state.metadata.get("halted") == JUST_VERIFIED,
            )
        )


def _turn_facts(state: AgentState) -> dict[str, Any]:
    """What the dashboard filters a turn by: its intent, what was decided, how it ended."""
    return {
        "intent": state.metadata.get("intent"),
        "decisions": state.decisions,
        "escalated": state.escalated,
        "halted": state.metadata.get("halted"),
        "tool_rounds": state.tool_rounds,
    }
