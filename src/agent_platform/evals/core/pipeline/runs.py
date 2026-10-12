"""Opening a run and closing a stage — the bookkeeping every suite shares.

The golden suite and the Promptfoo suite reach their transcripts very
differently, but a run is recorded the same way for both, and a stage
closes with the same accounting: how many cases, how many reused, and what
it drew from the shared provider budget.
"""

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import EvalPlugin, canonical_hash
from agent_platform.evals.core.fingerprint import AgentFingerprint
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot, current_limit
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord, StageRecord
from agent_platform.evals.core.store.records import new_run_id
from agent_platform.observability.types import utc_now

COMPLETED, ERRORED, RUNNING = "completed", "errored", "running"


def run_prefix(plugin: EvalPlugin) -> str:
    """The configured short name for run ids, else the agent's id."""
    return get_settings().evals.run_id_prefix or plugin.agent_id


def open_run(
    plugin: EvalPlugin,
    store: DuckDbEvalStore,
    suite: str,
    fingerprint: AgentFingerprint,
    dataset_hash: str,
    case_count: int,
) -> RunRecord:
    """A new run of one suite against one agent fingerprint, recorded as running."""
    run = RunRecord(
        run_id=new_run_id(
            run_prefix(plugin), suite, canonical_hash([fingerprint.value, dataset_hash])
        ),
        suite=suite,
        agent_id=plugin.agent_id,
        agent_fingerprint=fingerprint.value,
        fingerprint_parts=fingerprint.parts,
        dataset_hash=dataset_hash,
        case_count=case_count,
        git_sha=fingerprint.git_sha,
        git_dirty=fingerprint.git_dirty,
        model=plugin.fingerprint_inputs().get("model", ""),
        judge_model=get_settings().judge.model,
        rate_limit=current_limit(),
        started_at=utc_now(),
        status=RUNNING,
    )
    store.create_run(run)
    return run


def close_stage(
    store: DuckDbEvalStore,
    stage: StageRecord,
    before: BudgetSnapshot,
    *,
    cases: int,
    reused: int = 0,
    detail: str = "",
    requests: int | None = None,
) -> StageRecord:
    """Record a finished stage with what it drew from the provider budget.

    `requests` overrides the limiter's count, for a stage whose calls ran in
    another process (Promptfoo's workers) the limiter here never saw.
    """
    spent = BudgetSnapshot.now().since(before)
    finished = stage.model_copy(
        update={
            "finished_at": utc_now(),
            "status": COMPLETED,
            "cases": cases,
            "reused": reused,
            "provider_requests": spent.requests if requests is None else requests,
            "throttled": spent.throttled,
            "throttle_wait_s": spent.waited_seconds,
            "http_429": spent.rejections,
            "detail": detail,
        }
    )
    store.finish_stage(finished)
    return finished
