"""The research agent, driven one eval case at a time.

Every case gets a private world: its own platform database, audit ledger,
checkpoint file, artifact folder and episodic log, all under one workspace
that is removed afterwards. What is shared is only what is stateless and
read-only: the recorded fixture set (the web, replayed) and the fixture
world. The agent is the real one (`build_agent`) with the posture the
configuration declares — which is how an ablation overlay changes what
is measured and nothing else.

A `ScriptedUser` plays the interview from the case's payload. After
delivery the case collects the facts the oracle needs: the ledger, the
contract as approved and as it ended, every plan, the prompts each
screened role received, a fresh Goldfish Audit of the report (and of a
planted-flaw copy, when the case plants one), and whether a replay miss
still raises.
"""

import shutil
import time
from pathlib import Path
from typing import Any

from agentic_commons.artifacts import DirectoryArtifactStore
from agentic_commons.audit import SqliteAuditLedger
from agentic_commons.config import get_settings
from agentic_commons.config.settings import ProviderName
from agentic_commons.durability import create_checkpointer, model_types
from agentic_commons.evals.contracts import GoldenCase, ToolCallRecord, Transcript, UsageRecord
from agentic_commons.llm.client import LLMClient, create_pinned_llm_client
from agentic_commons.llm.recording import RecordingLLMClient
from agentic_commons.llm.roles import ClientFactory, RoleModelRouter
from agentic_commons.memory import EpisodicStore
from agentic_commons.observability import InMemoryLiveStatusFeed, StageReporter
from agentic_commons.persistence import SqliteConnectionPool
from agentic_commons.web import FixtureStore, ReplayMissError, SearchRequest, WebGateway
from research_agent.config.settings import ResearchAgentSettings
from research_agent.domain.budget import Budget
from research_agent.domain.report import Report
from research_agent.evals.dataset.schema import ResearchCasePayload
from research_agent.evals.planted import plant
from research_agent.fixture_world.scripted_user import ScriptedUser
from research_agent.graph import build_agent
from research_agent.graph.agent import ResearchAgent, TurnResult
from research_agent.graph.builder import build_auditor, entailment_for
from research_agent.graph.state import StateTypes
from research_agent.prompts.base import role_of

_FROZEN_ACTION, _HASH_FIELD = "contract.frozen", "contract_hash"
_SEARCH, _FETCH, _BLOCKED = "web.search", "web.fetch", "web.outbound"
_URL_SCHEMES = ("http://", "https://")
_COMPLETED, _DENIED = "completed", "deny"


class RecordingFactory:
    """Wraps every client the router builds, so each prompt can be read back by role."""

    def __init__(self, inner: ClientFactory) -> None:
        self._inner = inner
        self.clients: list[RecordingLLMClient] = []

    def __call__(self, provider: ProviderName, model: str) -> LLMClient:
        self.clients.append(RecordingLLMClient(self._inner(provider, model)))
        return self.clients[-1]

    def prompts(self, roles: list[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {role: [] for role in roles}
        for client in self.clients:
            for request in client.requests:
                role = role_of(request.messages)
                if role in found:
                    found[role].append("\n".join(m.content for m in request.messages))
        return found

    def usage(self) -> UsageRecord:
        return UsageRecord(
            llm_calls=sum(c.calls for c in self.clients),
            prompt_tokens=sum(c.prompt_tokens for c in self.clients),
            completion_tokens=sum(c.completion_tokens for c in self.clients),
            cost_usd=sum(c.cost_usd for c in self.clients),
        )


def base_factory(agent: ResearchAgentSettings) -> ClientFactory:
    """The fixture world's reference model, or the real pinned models."""
    if agent.model_backend == "fixture":
        from research_agent.fixture_world.model import fixture_client_factory

        return fixture_client_factory(agent)
    return create_pinned_llm_client


class ResearchAgentUnderTest:
    """Runs one research case against a freshly isolated agent."""

    def __init__(self, fixtures: FixtureStore, settings: ResearchAgentSettings, run_key: str):
        self._fixtures = fixtures
        self._settings = settings
        self._run_key = run_key

    async def run(self, case: GoldenCase) -> Transcript:
        payload = ResearchCasePayload.of(case)
        workspace = self._settings.evals.resolved_workspace_root / self._run_key / case.case_id
        shutil.rmtree(workspace, ignore_errors=True)
        workspace.mkdir(parents=True)
        try:
            return await CaseRun(self, case, payload, workspace).execute()
        finally:
            if not self._settings.evals.keep_case_workspaces:
                shutil.rmtree(workspace, ignore_errors=True)

    def settings_for(self, payload: ResearchCasePayload) -> ResearchAgentSettings:
        budget = Budget.model_validate({**self._settings.budget.model_dump(), **payload.budget})
        return self._settings.model_copy(update={"budget": budget})

    def gateway(self) -> WebGateway:
        return WebGateway("replay", fixtures=self._fixtures)


class CaseRun:
    """One case: build, converse, then collect what the oracle reads."""

    def __init__(
        self,
        owner: ResearchAgentUnderTest,
        case: GoldenCase,
        payload: ResearchCasePayload,
        workspace: Path,
    ) -> None:
        self._owner, self._case, self._payload = owner, case, payload
        self._workspace = workspace
        self._agent_settings = owner.settings_for(payload)
        self._recorder = RecordingFactory(base_factory(self._agent_settings))
        self._roles = RoleModelRouter(get_settings().llm.roles, factory=self._recorder)
        self._feed = InMemoryLiveStatusFeed(max_events=0)
        self._gateways: list[WebGateway] = []
        self._session = f"{self._agent_settings.evals.session_prefix}-{case.case_id}"

    async def execute(self) -> Transcript:
        pool = SqliteConnectionPool(self._workspace / "research.db")
        checkpointer = create_checkpointer(
            state_types=model_types(StateTypes), path=self._workspace / "checkpoints.db"
        )
        started = time.perf_counter()
        try:
            agent = self._build(pool, checkpointer)
            user = ScriptedUser(
                self._payload.script,
                self._agent_settings.command_names,
                self._agent_settings.scoping.max_turns * 2,
            )
            reply = await user.converse(agent, self._session)
            effects = await self._side_effects(agent, reply)
            return self._transcript(reply, user, effects, started)
        finally:
            checkpointer.conn.close()
            pool.close()

    def _build(self, pool: SqliteConnectionPool, checkpointer: Any) -> ResearchAgent:
        def gateway() -> WebGateway:
            self._gateways.append(self._owner.gateway())
            return self._gateways[-1]

        self._audit = SqliteAuditLedger(pool, agent_id=get_settings().agent.id)
        return build_agent(
            reporter=StageReporter(self._feed, self._session),
            clients=self._roles,
            gateway=gateway,
            pool=pool,
            checkpointer=checkpointer,
            audit=self._audit,
            artifacts=DirectoryArtifactStore(self._workspace / "artifacts"),
            episodic=EpisodicStore(self._workspace / "episodic"),
            agent_settings=self._agent_settings,
        )

    # -- what the oracle reads -------------------------------------------------------------

    async def _side_effects(self, agent: ResearchAgent, reply: TurnResult) -> dict[str, Any]:
        values = agent.session(self._session).run_values
        contract, ledger, trace = values.get("contract"), values.get("ledger"), values.get("trace")
        effects: dict[str, Any] = {
            "phase": reply.phase,
            "status": reply.report.status if reply.report else reply.phase,
            "stop_reason": values.get("stop_reason", ""),
            "contract": _contract_facts(contract, self._frozen_hash()),
            "plans": [plan.model_dump(mode="json") for plan in values.get("plans", [])],
            "ledger": _ledger_facts(ledger),
            "trace": trace.model_dump(mode="json") if trace is not None else {},
            "report": _report_facts(reply.report),
            "prompts": self._recorder.prompts(self._agent_settings.evals.screened_roles),
            "replay_miss_raised": await self._replay_miss_raises(),
        }
        if contract is not None and ledger is not None and reply.report is not None:
            effects.update(await self._audits(contract, reply.report, ledger))
        return effects

    async def _audits(self, contract: Any, report: Report, ledger: Any) -> dict[str, Any]:
        """A fresh audit of what was delivered — and of a planted-flaw copy, if planted."""
        auditor = build_auditor(
            self._agent_settings, self._roles, entailment_for(self._agent_settings)
        )
        base = await auditor.audit_report(contract, report, ledger)
        found: dict[str, Any] = {"eval_audit": base.model_dump(mode="json", exclude={"verdicts"})}
        flaw = self._payload.planted_flaw
        if flaw is not None:
            flawed = await auditor.audit_report(contract, plant(report, flaw), ledger)
            new_gaps = sorted(set(flawed.gaps) - set(base.gaps))
            found["planted"] = {
                "kind": flaw.kind,
                "new_gaps": new_gaps,
                "unflagged_gaps": flawed.unflagged_gaps,
                "caught": bool(new_gaps or set(flawed.unflagged_gaps) - set(base.unflagged_gaps)),
            }
        return found

    async def _replay_miss_raises(self) -> bool:
        probe = SearchRequest(
            query=self._agent_settings.evals.probe_query,
            max_results=self._agent_settings.research.max_results_per_search,
        )
        try:
            await self._owner.gateway().search(probe)
        except ReplayMissError:
            return True
        return False

    def _frozen_hash(self) -> str:
        """The hash recorded when the user approved: the audit trail's, not a screen card's."""
        frozen = [e for e in self._audit.by_decision("allow") if e.action == _FROZEN_ACTION]
        return str(frozen[-1].authority_context.get(_HASH_FIELD, "")) if frozen else ""

    # -- the transcript --------------------------------------------------------------------

    def _transcript(
        self, reply: TurnResult, user: ScriptedUser, effects: dict[str, Any], started: float
    ) -> Transcript:
        usage = self._recorder.usage().model_copy(
            update={"latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        )
        return Transcript(
            suite=self._case.suite,
            case_id=self._case.case_id,
            replies=[_final_reply(reply, self._agent_settings.evals.reference_judge.status_prefix)],
            tool_calls=_web_calls(effects["trace"]),
            evidence=[
                f"[{e['entry_id']}] {e['span']} ({e['source_url']})" for e in effects["ledger"]
            ],
            side_effects=effects,
            metadata={
                "features": get_settings().features.model_dump(mode="json"),
                "judge_model": self._roles.model_key("judge"),
                "independent_judge": self._agent_settings.audit.independent_judge,
                "conversation": [list(turn) for turn in user.transcript],
            },
            usage=usage,
        )


def _final_reply(reply: TurnResult, status_prefix: str) -> str:
    if reply.report is None:
        return reply.response
    report = reply.report
    return f"{report.title}\n{status_prefix}{report.status}\n\n{report.text()}"


def _contract_facts(contract: Any, frozen_hash: str) -> dict[str, Any]:
    if contract is None:
        return {"frozen_hash": frozen_hash}
    brief = contract.brief
    return {
        "frozen_hash": frozen_hash,
        "final_hash": contract.contract_hash,
        "verified": contract.verify(),
        "archetype": brief.archetype,
        "criteria": list(brief.criteria),
        "known_unknowns": [k.model_dump(mode="json") for k in brief.known_unknowns],
        "private_terms": list(brief.private_terms),
        "max_dag_nodes": contract.budget.max_dag_nodes,
        "rubric": contract.rubric.model_dump(mode="json"),
    }


def _ledger_facts(ledger: Any) -> list[dict[str, Any]]:
    if ledger is None:
        return []
    return [entry.model_dump(mode="json") for entry in ledger.entries]


def _report_facts(report: Report | None) -> dict[str, Any]:
    if report is None:
        return {}
    return {
        "status": report.status,
        "text": report.text(),
        "limits": [entry.item_id for entry in report.limits],
        "known_unknowns": list(report.known_unknowns),
        "cited_urls": sorted({c.url for c in report.citations}),
    }


def _web_calls(trace: dict[str, Any]) -> list[ToolCallRecord]:
    """Every search and fetch the run attempted, then every one the guard blocked."""
    calls = []
    for target in trace.get("attempted_outbound", []):
        is_url = str(target).startswith(_URL_SCHEMES)
        calls.append(
            ToolCallRecord(
                name=_FETCH if is_url else _SEARCH,
                arguments={"url" if is_url else "query": target},
                status=_COMPLETED,
            )
        )
    calls.extend(
        ToolCallRecord(name=_BLOCKED, status=_DENIED, decision=_DENIED, reason_code=str(reason))
        for reason in trace.get("blocked_outbound", [])
    )
    return calls
