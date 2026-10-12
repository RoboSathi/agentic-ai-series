"""DuckDB persistence for eval runs (analytics, not transactions).

Connections are short-lived — opened per operation, closed straight
after. DuckDB admits one process per file, so a connection held for a
whole run would lock out every reader (a report, `/eval` in the TUI,
`python -m agent_platform.evals runs`) for twenty minutes. Opening per operation
costs milliseconds; a reader that lands mid-write retries briefly.
"""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from agent_platform.config import get_settings
from agent_platform.config.settings import EvalStoreSettings
from agent_platform.evals.core.contracts import GoldenCase, Score, Transcript, canonical_hash
from agent_platform.evals.core.store.records import Row, RunRecord, StageRecord, StoredTranscript
from agent_platform.logging import get_logger
from agent_platform.observability.types import utc_now

_SQL_DIR = Path(__file__).resolve().parent
_RUNNING, _INTERRUPTED = "running", "interrupted"
_LOCK_MESSAGE = "lock"


class DuckDbEvalStore:
    """Runs, cases, transcripts, stage batches and scores."""

    def __init__(
        self, db_path: Path | str | None = None, settings: EvalStoreSettings | None = None
    ) -> None:
        self._settings = settings or get_settings().evals.store
        self._path = str(db_path if db_path is not None else self._settings.resolved_db_path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._memory: Any = None
        self._lock = threading.RLock()
        # Runs this process opened or staged: the only ones an interrupt may close.
        self._touched: set[str] = set()
        self._logger = get_logger("evals")
        self._ensure_schema()

    @property
    def db_path(self) -> str:
        return self._path

    # -- runs -------------------------------------------------------------

    def find_run(self, suite: str, agent_fingerprint: str, dataset_hash: str) -> RunRecord | None:
        """The run this exact agent already made of this exact dataset, if any."""
        rows = self.query(
            "SELECT * FROM eval_runs WHERE suite = ? AND agent_fingerprint = ? "
            "AND dataset_hash = ? ORDER BY started_at DESC LIMIT 1",
            (suite, agent_fingerprint, dataset_hash),
        )
        return _run(rows[0]) if rows else None

    def latest_run(self, suite: str | None = None) -> RunRecord | None:
        where, params = ("WHERE suite = ?", (suite,)) if suite else ("", ())
        rows = self.query(
            f"SELECT * FROM eval_runs {where} ORDER BY started_at DESC LIMIT 1", params
        )
        return _run(rows[0]) if rows else None

    def get_run(self, run_id: str) -> RunRecord | None:
        rows = self.query("SELECT * FROM eval_runs WHERE run_id = ?", (run_id,))
        return _run(rows[0]) if rows else None

    def runs(self, limit: int) -> list[RunRecord]:
        rows = self.query("SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        return [_run(row) for row in rows]

    def create_run(self, run: RunRecord) -> None:
        self._touched.add(run.run_id)
        payload = run.model_dump()
        payload["fingerprint_parts"] = json.dumps(run.fingerprint_parts)
        self._insert("eval_runs", payload)

    def mark_interrupted(self) -> list[str]:
        """Close what this process left running when it was interrupted — and nothing else.

        Only the runs this store opened or staged in this process: another `make eval`
        running beside it keeps its own state. Finished transcripts and scores stay, so the
        next run of the same agent and cases finds the run and resumes from them. Returns
        the stages it closed, as `run_id/stage`.
        """
        if not self._touched:
            return []
        runs = tuple(sorted(self._touched))
        marks = ", ".join("?" for _ in runs)
        where = f"status = '{_RUNNING}' AND run_id IN ({marks})"
        stages = [
            f"{row['run_id']}/{row['stage']}"
            for row in self.query(f"SELECT run_id, stage FROM eval_stages WHERE {where}", runs)
        ]
        for table in ("eval_stages", "eval_runs"):
            self.execute(
                f"UPDATE {table} SET status = ?, finished_at = ? WHERE {where}",
                (_INTERRUPTED, utc_now(), *runs),
            )
        return stages

    def set_run_status(self, run_id: str, status: str) -> None:
        self.execute(
            "UPDATE eval_runs SET status = ?, finished_at = ? WHERE run_id = ?",
            (status, utc_now(), run_id),
        )

    # -- cases and transcripts ---------------------------------------------

    def upsert_cases(self, cases: list[GoldenCase]) -> None:
        for case in cases:
            self.execute(
                "INSERT OR IGNORE INTO eval_cases VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    case.suite,
                    case.case_id,
                    case.case_hash,
                    case.category,
                    case.tags,
                    json.dumps(case.turns),
                    json.dumps(case.payload),
                ),
            )

    def cases_for_run(self, run_id: str) -> dict[str, GoldenCase]:
        """The exact case versions a run's transcripts answered."""
        rows = self.query(
            "SELECT DISTINCT c.* FROM eval_cases c JOIN eval_transcripts t "
            "ON t.case_id = c.case_id AND t.case_hash = c.case_hash WHERE t.run_id = ?",
            (run_id,),
        )
        return {
            row["case_id"]: GoldenCase(
                suite=row["suite"],
                case_id=row["case_id"],
                category=row["category"],
                tags=list(row["tags"] or []),
                turns=json.loads(row["turns"]),
                payload=json.loads(row["payload"]),
            )
            for row in rows
        }

    def transcripts(self, run_id: str) -> list[StoredTranscript]:
        rows = self.query(
            "SELECT * FROM eval_transcripts WHERE run_id = ? ORDER BY case_id", (run_id,)
        )
        return [_stored(row) for row in rows]

    def reusable_transcript(
        self, case_hash: str, agent_fingerprint: str
    ) -> StoredTranscript | None:
        """A clean transcript this agent already produced for this exact case."""
        rows = self.query(
            "SELECT * FROM eval_transcripts WHERE case_hash = ? AND agent_fingerprint = ? "
            "AND coalesce(error, '') = '' ORDER BY created_at DESC LIMIT 1",
            (case_hash, agent_fingerprint),
        )
        return _stored(rows[0]) if rows else None

    def save_transcript(
        self,
        run: RunRecord,
        case: GoldenCase,
        transcript: Transcript,
        reused_from: str | None = None,
    ) -> StoredTranscript:
        body = transcript.model_dump(mode="json")
        stored = StoredTranscript(
            run_id=run.run_id,
            case_id=case.case_id,
            case_hash=case.case_hash,
            category=case.category,
            transcript_hash=canonical_hash(body),
            transcript=transcript,
            reused_from=reused_from,
        )
        usage = transcript.usage
        self.execute(
            "INSERT OR REPLACE INTO eval_transcripts VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.run_id,
                case.suite,
                case.case_id,
                case.case_hash,
                case.category,
                run.agent_fingerprint,
                stored.transcript_hash,
                transcript.reply,
                json.dumps(body),
                usage.llm_calls,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cost_usd,
                usage.latency_ms,
                transcript.error,
                reused_from,
                utc_now(),
            ),
        )
        return stored

    # -- stages and scores ---------------------------------------------------

    def start_stage(self, stage: StageRecord) -> None:
        self._touched.add(stage.run_id)
        self._insert("eval_stages", stage.model_dump())

    def finish_stage(self, stage: StageRecord) -> None:
        self.execute("DELETE FROM eval_stages WHERE batch_id = ?", (stage.batch_id,))
        self._insert("eval_stages", stage.model_dump())

    def reusable_scores(
        self, transcript_hash: str, stage: str, grader_fingerprint: str
    ) -> list[Score]:
        """Scores the same grader already gave this exact transcript."""
        rows = self.query(
            "SELECT sc.* FROM eval_scores sc JOIN eval_stages st USING (batch_id) "
            "WHERE sc.transcript_hash = ? AND sc.stage = ? AND st.grader_fingerprint = ? "
            "AND st.status = 'completed' "
            "AND sc.batch_id = (SELECT s2.batch_id FROM eval_scores s2 JOIN eval_stages t2 "
            "USING (batch_id) WHERE s2.transcript_hash = ? AND s2.stage = ? "
            "AND t2.grader_fingerprint = ? AND t2.status = 'completed' "
            "ORDER BY t2.started_at DESC LIMIT 1)",
            (transcript_hash, stage, grader_fingerprint) * 2,
        )
        return [_score(row) for row in rows]

    def save_scores(
        self, stage: StageRecord, stored: StoredTranscript, scores: list[Score], reused: bool
    ) -> None:
        now = utc_now()
        self.execute_many(
            "INSERT INTO eval_scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    stage.batch_id,
                    stage.run_id,
                    stage.stage,
                    score.case_id,
                    stored.transcript_hash,
                    score.metric,
                    score.grader,
                    score.kind,
                    score.severity,
                    score.value,
                    score.passed,
                    score.threshold,
                    score.tier,
                    score.rationale,
                    score.cost_usd,
                    score.llm_calls,
                    reused,
                    now,
                )
                for score in scores
            ],
        )

    # -- raw access ------------------------------------------------------------

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[Row]:
        """Run a read and return rows as dicts."""
        with self._connection() as connection:
            cursor = connection.execute(sql, params)
            if cursor.description is None:
                return []
            columns = [str(column[0]) for column in cursor.description]
            return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._connection() as connection:
            connection.execute(sql, params)

    def execute_many(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        if not rows:
            return
        with self._connection() as connection:
            connection.executemany(sql, rows)

    def _insert(self, table: str, payload: dict[str, Any]) -> None:
        columns = ", ".join(payload)
        marks = ", ".join("?" for _ in payload)
        self.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(payload.values()))

    def _ensure_schema(self) -> None:
        with self._connection() as connection:
            for name in ("schema.sql", "views.sql"):
                connection.execute((_SQL_DIR / name).read_text())

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        import duckdb

        with self._lock:
            if self._path == ":memory:":
                if self._memory is None:
                    self._memory = duckdb.connect(":memory:")
                yield self._memory
                return
            connection = self._open(duckdb)
            try:
                yield connection
            finally:
                connection.close()

    def _open(self, duckdb: Any) -> Any:
        """Open the file, waiting out another process's brief hold on it."""
        attempts = self._settings.lock_retry_attempts
        for attempt in range(attempts):
            try:
                return duckdb.connect(self._path)
            except duckdb.IOException as exc:
                if _LOCK_MESSAGE not in str(exc).lower() or attempt == attempts - 1:
                    raise
                time.sleep(self._settings.lock_retry_delay_seconds)
        raise RuntimeError("unreachable")  # pragma: no cover


def _run(row: Row) -> RunRecord:
    parts = row.get("fingerprint_parts")
    return RunRecord(**{**row, "fingerprint_parts": json.loads(parts) if parts else {}})


def _stored(row: Row) -> StoredTranscript:
    return StoredTranscript(
        run_id=row["run_id"],
        case_id=row["case_id"],
        case_hash=row["case_hash"],
        category=row["category"],
        transcript_hash=row["transcript_hash"],
        transcript=Transcript.model_validate_json(row["transcript"]),
        reused_from=row["reused_from"],
    )


def _score(row: Row) -> Score:
    return Score(**{field: row[field] for field in Score.model_fields})
