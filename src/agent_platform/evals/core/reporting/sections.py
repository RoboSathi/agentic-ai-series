"""Report sections every framework's report shares, built only from the eval store.

A report is a view of the recorded data, never of whatever a process held
in memory — so `python -m agent_platform.evals report` rebuilds any run's
report at any time, and the terminal overview and the file never disagree.
"""

import json
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.reporting.markdown import cell, fenced, quote, rate, table
from agent_platform.evals.core.store import DuckDbEvalStore, RunRecord, StoredTranscript


class ReportSections:
    """The Markdown pieces: run facts, metrics, failures, transcripts, cost."""

    def __init__(self, store: DuckDbEvalStore) -> None:
        self._store = store
        settings = get_settings().evals.reporting
        self._z = settings.confidence_z
        self._limit = settings.max_cases_listed
        self._excerpt = settings.excerpt_chars

    @property
    def z(self) -> float:
        return self._z

    def run_facts(self, run: RunRecord) -> str:
        facts = [
            ("run", run.run_id),
            ("suite", run.suite),
            ("agent", run.agent_id),
            ("status", run.status),
            ("cases", run.case_count),
            ("agent model", run.model),
            ("judge model", run.judge_model),
            ("agent fingerprint", run.agent_fingerprint[:16]),
            ("git", f"{run.git_sha[:10]}{' (uncommitted changes)' if run.git_dirty else ''}"),
            ("rate limit", f"{run.rate_limit} requests/window" if run.rate_limit else "off"),
            ("started", run.started_at),
            ("finished", run.finished_at),
        ]
        return table(["", ""], facts)

    def changes_since_previous(self, run: RunRecord) -> str:
        earlier = self._store.query(
            "SELECT * FROM eval_runs WHERE suite = ? AND agent_id = ? AND started_at < ? "
            "ORDER BY started_at DESC LIMIT 1",
            (run.suite, run.agent_id, run.started_at),
        )
        if not earlier:
            return "_First run of this suite._"
        previous = json.loads(earlier[0]["fingerprint_parts"] or "{}")
        changed = sorted(
            name
            for name in set(run.fingerprint_parts) | set(previous)
            if run.fingerprint_parts.get(name) != previous.get(name)
        )
        if not changed:
            return f"Agent unchanged since `{earlier[0]['run_id']}` — only the cases differ."
        return f"Since `{earlier[0]['run_id']}`: **{', '.join(changed)}** changed."

    def metric_rows(self, run_id: str, stage: str) -> list[dict[str, Any]]:
        return self._store.query(
            "SELECT metric, severity, passed, cases, mean_value, min_value, threshold "
            "FROM v_metric_pass_rates WHERE run_id = ? AND stage = ? ORDER BY severity, metric",
            (run_id, stage),
        )

    def metric_table(self, run_id: str, stage: str) -> str:
        return table(
            ["metric", "kind", "pass rate (95% CI)", "mean", "min", "threshold"],
            [
                [
                    row["metric"],
                    row["severity"],
                    rate(int(row["passed"]), int(row["cases"]), self._z),
                    row["mean_value"],
                    row["min_value"],
                    row["threshold"],
                ]
                for row in self.metric_rows(run_id, stage)
            ],
        )

    def category_matrix(self, run_id: str, stage: str) -> str:
        """Failing cases / cases checked, per metric and category."""
        rows = self._store.query(
            "SELECT metric, category, count(*) AS checked, "
            "sum(CASE WHEN NOT passed THEN 1 ELSE 0 END) AS failed "
            "FROM v_scores WHERE run_id = ? AND stage = ? GROUP BY ALL ORDER BY metric",
            (run_id, stage),
        )
        categories = sorted({row["category"] for row in rows})
        cells: dict[str, dict[str, str]] = {}
        for row in rows:
            cells.setdefault(row["metric"], {})[row["category"]] = (
                f"{row['failed']}/{row['checked']}"
            )
        return table(
            ["metric", *categories],
            [
                [metric, *(by.get(category, "") for category in categories)]
                for metric, by in sorted(cells.items())
            ],
        )

    def failures(self, run_id: str, stage: str) -> list[dict[str, Any]]:
        return self._store.query(
            "SELECT case_id, category, metric, severity, value, rationale FROM v_failures "
            "WHERE run_id = ? AND stage = ? ORDER BY severity, case_id, metric",
            (run_id, stage),
        )

    def failing_cases(self, run_id: str, stages: tuple[str, ...]) -> str:
        """Each failing case once: what broke, the conversation, what it changed."""
        marks = ", ".join("?" for _ in stages)
        rows = self._store.query(
            f"SELECT case_id, category, stage, metric, severity, rationale FROM v_failures "
            f"WHERE run_id = ? AND stage IN ({marks}) ORDER BY severity, case_id, metric",
            (run_id, *stages),
        )
        if not rows:
            return "_No failures._"
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_case.setdefault(row["case_id"], []).append(row)
        stored = {item.case_id: item for item in self._store.transcripts(run_id)}
        turns = self.turns(run_id)
        blocks = [
            self._case_block(stored.get(case_id), turns.get(case_id, []), failed)
            for case_id, failed in list(by_case.items())[: self._limit]
        ]
        if len(by_case) > self._limit:
            blocks.append(f"_…and {len(by_case) - self._limit} more failing cases._")
        return "\n\n".join(blocks)

    def cost_table(self, run_id: str) -> str:
        stages = self._store.query(
            "SELECT stage, cases, reused, provider_requests, throttled, throttle_wait_s, "
            "http_429, seconds FROM v_stage_cost WHERE run_id = ? "
            "AND batch_id IN (SELECT batch_id FROM v_latest_batches) ORDER BY started_at",
            (run_id,),
        )
        return table(
            ["stage", "cases", "reused", "provider requests", "throttled", "wait (s)", "429s", "s"],
            [list(row.values()) for row in stages],
        )

    def turns(self, run_id: str) -> dict[str, list[str]]:
        rows = self._store.query(
            "SELECT DISTINCT c.case_id, c.turns FROM eval_cases c JOIN eval_transcripts t "
            "ON t.case_hash = c.case_hash AND t.case_id = c.case_id WHERE t.run_id = ?",
            (run_id,),
        )
        return {row["case_id"]: json.loads(row["turns"]) for row in rows}

    def _case_block(
        self, item: StoredTranscript | None, turns: list[str], failed: list[dict[str, Any]]
    ) -> str:
        head = failed[0]
        lines = [f"### {head['case_id']} · {head['category']}"]
        lines.append(
            table(
                ["stage", "metric", "kind", "why"],
                [
                    [row["stage"], row["metric"], row["severity"], row["rationale"]]
                    for row in failed
                ],
            )
        )
        if item is None:
            return "\n\n".join(lines)
        transcript = item.transcript
        for index, turn in enumerate(turns):
            reply = transcript.replies[index] if index < len(transcript.replies) else ""
            lines.append(f"**User:** {cell(turn)}\n\n**Agent:**\n\n{quote(reply[: self._excerpt])}")
        if transcript.tool_calls:
            lines.append(
                table(
                    ["turn", "tool", "arguments", "decision", "standing", "reason", "status"],
                    [
                        [
                            call.turn,
                            call.name,
                            json.dumps(call.arguments, default=str),
                            call.decision,
                            call.standing,
                            call.reason_code,
                            call.status,
                        ]
                        for call in transcript.tool_calls
                    ],
                )
            )
        if any(transcript.side_effects.values()):
            lines.append("**Side effects**\n\n" + fenced(transcript.side_effects))
        if transcript.error:
            lines.append(f"**Error:** `{transcript.error}`")
        return "\n\n".join(lines)
