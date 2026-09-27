"""Rows of the eval store, as types."""

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic_commons.evals.contracts import StageName, Transcript
from agentic_commons.observability.types import utc_now

RUN_ID_TIME_FORMAT = "%Y%m%d-%H%M%S"
FINGERPRINT_PREFIX = 8
UNIQUE_SUFFIX = 4


def new_run_id(suite: str, fingerprint: str) -> str:
    """`golden-20260926-221500-c40103a7-1f2e`: readable, sortable, never reused.

    The time and fingerprint make it readable; the random suffix is what
    keeps two runs started in the same second (a `--fresh` rerun) apart.
    """
    stamp = utc_now().strftime(RUN_ID_TIME_FORMAT)
    return f"{suite}-{stamp}-{fingerprint[:FINGERPRINT_PREFIX]}-{uuid.uuid4().hex[:UNIQUE_SUFFIX]}"


class RunRecord(BaseModel):
    """One suite, generated against one agent fingerprint and dataset version."""

    run_id: str
    suite: str
    agent_id: str
    agent_fingerprint: str
    fingerprint_parts: dict[str, str] = Field(default_factory=dict)
    dataset_hash: str
    case_count: int
    git_sha: str = ""
    git_dirty: bool = False
    model: str = ""
    judge_model: str = ""
    rate_limit: int | None = None
    started_at: datetime
    finished_at: datetime | None = None
    status: str


class StoredTranscript(BaseModel):
    """A transcript as stored, with the identity scoring caches key on."""

    run_id: str
    case_id: str
    case_hash: str
    category: str
    transcript_hash: str
    transcript: Transcript
    reused_from: str | None = None


class StageRecord(BaseModel):
    """One stage's batch over one run."""

    batch_id: str
    run_id: str
    stage: StageName
    grader_fingerprint: str = ""
    started_at: datetime
    finished_at: datetime | None = None
    status: str
    cases: int = 0
    reused: int = 0
    provider_requests: int = 0
    throttled: int = 0
    throttle_wait_s: float = 0.0
    http_429: int = 0
    detail: str = ""

    @classmethod
    def begin(cls, run_id: str, stage: StageName, grader_fingerprint: str = "") -> "StageRecord":
        """A new, running batch of one stage over one run."""
        return cls(
            batch_id=uuid.uuid4().hex,
            run_id=run_id,
            stage=stage,
            grader_fingerprint=grader_fingerprint,
            started_at=utc_now(),
            status="running",
        )


Row = dict[str, Any]
