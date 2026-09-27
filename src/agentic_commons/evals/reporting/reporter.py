"""Detailed Markdown reports, built only from the eval store.

Every stage target writes its own report when it finishes, and `make
eval-report` rebuilds them all from DuckDB — so a report is always a view
of the recorded data, never of whatever a process happened to hold in
memory. Wording lives in `templates/`; this module fills the sections.
"""

import json
from pathlib import Path
from string import Template
from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.config.settings import EvalReportingSettings
from agentic_commons.evals.judge import JudgeCalibrator
from agentic_commons.evals.pipeline.gate import evaluate_gate
from agentic_commons.evals.reporting.markdown import cell, fenced, quote, rate, table
from agentic_commons.evals.store import DuckDbEvalStore, RunRecord, StoredTranscript

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_SUMMARY, _DIFF = "summary", "diff"
_STAGE_ORDER = ("generate", "oracle", "security", "judge", "deepeval")
_SECURITY_SUITE = "security"
_HALLUCINATION_METRIC = "hallucination_free"
_UNSUPPORTED, _SUPPORTED = "unsupported", "supported"


class EvalReporter:
    """Writes a run's reports into `<output_dir>/<run_id>/`."""

    def __init__(
        self, store: DuckDbEvalStore, settings: EvalReportingSettings | None = None
    ) -> None:
        self._store = store
        self._settings = settings or get_settings().evals.reporting
        self._z = self._settings.confidence_z
        self._limit = self._settings.max_cases_listed

    def write(self, run_id: str) -> list[Path]:
        """Every report the run has data for, summary first."""
        run = self._require(run_id)
        builders = {
            "generate": self._generate,
            "oracle": self._oracle,
            "security": self._security,
            "judge": self._judge,
            "deepeval": self._deepeval,
        }
        written = [self._emit(run, _SUMMARY, self._summary(run))]
        for stage in self._stages_with_data(run.run_id):
            written.append(self._emit(run, stage, builders[stage](run)))
        return written

    def write_diff(self, base_id: str, head_id: str) -> Path:
        base, head = self._require(base_id), self._require(head_id)
        body = self._fill(
            _DIFF,
            base=base.run_id,
            head=head.run_id,
            facts=self._diff_facts(base, head),
            metrics=self._diff_metrics(base, head),
            regressed=self._case_flips(base, head, regressed=True),
            improved=self._case_flips(base, head, regressed=False),
        )
        name = self._settings.files[_DIFF].format(base=base.run_id, head=head.run_id)
        return self._write(head, name, body)

    # -- summary -----------------------------------------------------------

    def _summary(self, run: RunRecord) -> str:
        gate = evaluate_gate(self._store, run.run_id, run.suite)
        return self._fill(
            _SUMMARY,
            run_id=run.run_id,
            verdict=gate.verdict,
            verdict_reasons="; ".join(gate.reasons) or "every gate cleared",
            soft_threshold=f"{gate.soft_threshold:.0%}",
            case_count=str(run.case_count),
            run_facts=self._run_facts(run),
            fingerprint_changes=self._fingerprint_changes(run),
            stages=self._stages_table(run.run_id),
            metrics=self._metric_table(run.run_id),
            categories=self._category_table(run.run_id),
            hard_failures=self._hard_failures(run.run_id),
            security=self._security_summary(run),
            cost=self._cost_table(run.run_id),
            links=self._links(run),
        )

    def _run_facts(self, run: RunRecord) -> str:
        facts = [
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

    def _fingerprint_changes(self, run: RunRecord) -> str:
        earlier = self._store.query(
            "SELECT * FROM eval_runs WHERE suite = ? AND started_at < ? "
            "ORDER BY started_at DESC LIMIT 1",
            (run.suite, run.started_at),
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
            return f"Agent unchanged since `{earlier[0]['run_id']}` — only the dataset differs."
        return f"Since `{earlier[0]['run_id']}`: **{', '.join(changed)}** changed."

    def _stages_table(self, run_id: str) -> str:
        rows = self._store.query(
            "SELECT stage, status, cases, reused, provider_requests, throttled, "
            "throttle_wait_s, http_429, seconds FROM v_stage_cost s "
            "WHERE run_id = ? AND batch_id IN (SELECT batch_id FROM v_latest_batches) "
            "ORDER BY started_at",
            (run_id,),
        )
        return table(
            [
                "stage",
                "status",
                "cases",
                "reused",
                "provider requests",
                "throttled",
                "wait (s)",
                "429s",
                "seconds",
            ],
            [list(row.values()) for row in rows],
        )

    def _metric_table(self, run_id: str, stage: str | None = None) -> str:
        where, params = ("AND stage = ?", (run_id, stage)) if stage else ("", (run_id,))
        rows = self._store.query(
            f"SELECT stage, metric, severity, passed, cases, mean_value, min_value, threshold "
            f"FROM v_metric_pass_rates WHERE run_id = ? {where} "
            f"ORDER BY stage, severity, metric",
            params,
        )
        return table(
            ["stage", "metric", "severity", "pass rate", "mean", "min", "threshold"],
            [
                [
                    row["stage"],
                    row["metric"],
                    row["severity"],
                    rate(int(row["passed"]), int(row["cases"]), self._z),
                    row["mean_value"],
                    row["min_value"],
                    row["threshold"],
                ]
                for row in rows
            ],
        )

    def _category_table(self, run_id: str) -> str:
        rows = self._store.query(
            "SELECT stage, category, cases, failing_cases FROM v_category_breakdown "
            "WHERE run_id = ? ORDER BY stage, category",
            (run_id,),
        )
        return table(
            ["stage", "category", "cases", "failing cases", "clean rate"],
            [
                [
                    row["stage"],
                    row["category"],
                    row["cases"],
                    row["failing_cases"],
                    rate(int(row["cases"]) - int(row["failing_cases"]), int(row["cases"]), self._z),
                ]
                for row in rows
            ],
        )

    def _hard_failures(self, run_id: str) -> str:
        rows = self._store.query(
            "SELECT case_id, category, stage, metric, rationale FROM v_hard_failures "
            "WHERE run_id = ? ORDER BY case_id, metric",
            (run_id,),
        )
        return table(
            ["case", "category", "stage", "invariant", "why"], [list(r.values()) for r in rows]
        )

    def _security_summary(self, run: RunRecord) -> str:
        if run.suite == _SECURITY_SUITE:
            return "_This is the security run._"
        rows = self._store.query(
            "SELECT run_id FROM eval_runs WHERE suite = ? AND agent_fingerprint = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (_SECURITY_SUITE, run.agent_fingerprint),
        )
        if not rows:
            return "_No security run for this agent fingerprint yet (`make eval-security`)._"
        security_id = rows[0]["run_id"]
        gate = evaluate_gate(self._store, security_id, _SECURITY_SUITE)
        link = f"../{security_id}/{self._settings.files['security']}"
        return (
            f"Latest security run for this exact agent: [`{security_id}`]({link}) — "
            f"**{gate.verdict}** ({'; '.join(gate.reasons) or 'no attack succeeded'})."
        )

    def _cost_table(self, run_id: str) -> str:
        rows = self._store.query(
            "SELECT category, cases, generated, llm_calls, mean_latency_ms, p90_latency_ms, "
            "cost_usd FROM v_cost_latency WHERE run_id = ? ORDER BY category",
            (run_id,),
        )
        return table(
            [
                "category",
                "cases",
                "generated now",
                "agent LLM calls",
                "mean ms",
                "p90 ms",
                "cost $",
            ],
            [list(row.values()) for row in rows],
        )

    def _links(self, run: RunRecord) -> str:
        stages = self._stages_with_data(run.run_id)
        return (
            "\n".join(f"- [{stage}]({self._settings.files[stage]})" for stage in stages)
            or "_None._"
        )

    # -- generation --------------------------------------------------------

    def _generate(self, run: RunRecord) -> str:
        stored = self._store.transcripts(run.run_id)
        overview = table(
            [
                "case",
                "category",
                "reply (start)",
                "tools",
                "agent LLM calls",
                "ms",
                "reused",
                "error",
            ],
            [
                [
                    item.case_id,
                    item.category,
                    item.transcript.reply[:80],
                    ", ".join(call.name for call in item.transcript.tool_calls),
                    item.transcript.usage.llm_calls,
                    item.transcript.usage.latency_ms,
                    "yes" if item.reused_from else "",
                    item.transcript.error,
                ]
                for item in stored
            ],
        )
        turns = self._turns(run.run_id)
        return self._fill(
            "generate",
            run_id=run.run_id,
            overview=overview,
            cases="\n\n".join(
                self._case_block(item, turns.get(item.case_id, [])) for item in stored
            ),
        )

    def _case_block(self, item: StoredTranscript, turns: list[str]) -> str:
        transcript = item.transcript
        conversation = []
        for index, turn in enumerate(turns):
            reply = transcript.replies[index] if index < len(transcript.replies) else ""
            conversation.append(f"**Customer:** {turn}\n\n**Agent:**\n\n{quote(reply)}")
        tools = table(
            ["tool", "arguments", "gatekeeper", "reason", "status"],
            [
                [
                    call.name,
                    json.dumps(call.arguments, default=str),
                    call.decision,
                    call.reason_code,
                    call.status,
                ]
                for call in transcript.tool_calls
            ],
        )
        parts = [
            f"### {item.case_id} · {item.category}"
            + (f" · reused from `{item.reused_from}`" if item.reused_from else ""),
            "\n\n".join(conversation),
            "**Tool calls**\n\n" + tools,
            "**Side effects**\n\n" + fenced(transcript.side_effects),
            "**Signals**\n\n" + fenced(transcript.metadata),
            f"**Usage:** {transcript.usage.llm_calls} agent LLM calls · "
            f"{transcript.usage.prompt_tokens}+{transcript.usage.completion_tokens} tokens · "
            f"${transcript.usage.cost_usd:.5f} · {transcript.usage.latency_ms:.0f} ms",
        ]
        if transcript.error:
            parts.append(f"**Error:** `{transcript.error}`")
        return "\n\n".join(parts)

    # -- oracle ----------------------------------------------------------------

    def _oracle(self, run: RunRecord) -> str:
        matrix = self._store.query(
            "SELECT metric, category, count(*) AS checked, "
            "sum(CASE WHEN NOT passed THEN 1 ELSE 0 END) AS failed "
            "FROM v_scores WHERE run_id = ? AND stage = 'oracle' GROUP BY ALL ORDER BY metric",
            (run.run_id,),
        )
        categories = sorted({row["category"] for row in matrix})
        by_metric: dict[str, dict[str, str]] = {}
        for row in matrix:
            by_metric.setdefault(row["metric"], {})[row["category"]] = (
                f"{row['failed']}/{row['checked']}"
            )
        return self._fill(
            "oracle",
            run_id=run.run_id,
            invariants=self._metric_table(run.run_id, "oracle"),
            matrix=table(
                ["invariant", *categories],
                [
                    [metric, *(cells.get(category, "") for category in categories)]
                    for metric, cells in sorted(by_metric.items())
                ],
            ),
            failures=self._failure_blocks(run.run_id, "oracle", with_side_effects=True),
        )

    def _failure_blocks(self, run_id: str, stage: str, *, with_side_effects: bool) -> str:
        rows = self._store.query(
            "SELECT case_id, category, metric, severity, rationale FROM v_failures "
            "WHERE run_id = ? AND stage = ? ORDER BY severity, case_id, metric",
            (run_id, stage),
        )
        if not rows:
            return "_No failures._"
        transcripts = {item.case_id: item.transcript for item in self._store.transcripts(run_id)}
        blocks = []
        for row in rows[: self._limit]:
            block = (
                f"- **{row['case_id']}** ({row['category']}) — `{row['metric']}` "
                f"[{row['severity']}]: {cell(row['rationale'])}"
            )
            transcript = transcripts.get(row["case_id"])
            if with_side_effects and transcript is not None:
                block += f"\n\n  Reply: {cell(transcript.reply)}"
            blocks.append(block)
        if len(rows) > self._limit:
            blocks.append(f"- …and {len(rows) - self._limit} more (query `v_failures`).")
        return "\n".join(blocks)

    # -- judge -----------------------------------------------------------------

    def _judge(self, run: RunRecord) -> str:
        return self._fill(
            "judge",
            run_id=run.run_id,
            judge_model=run.judge_model,
            criteria=self._metric_table(run.run_id, "judge"),
            agreement=self._guard_agreement(run.run_id),
            claims=self._failure_blocks_for_metric(run.run_id, _HALLUCINATION_METRIC),
            lowest=self._lowest(run.run_id, "judge"),
        )

    def _guard_agreement(self, run_id: str) -> str:
        """Cohen's kappa between the judge and the runtime guard, per case."""
        judged = {
            row["case_id"]: bool(row["passed"])
            for row in self._store.query(
                "SELECT case_id, passed FROM v_scores WHERE run_id = ? AND metric = ?",
                (run_id, _HALLUCINATION_METRIC),
            )
        }
        key = self._settings.runtime_guard_key
        pairs = []
        for item in self._store.transcripts(run_id):
            guard = item.transcript.metadata.get(key)
            if item.case_id in judged and guard is not None:
                judge_label = _SUPPORTED if judged[item.case_id] else _UNSUPPORTED
                guard_label = _SUPPORTED if float(guard) <= 0.0 else _UNSUPPORTED
                pairs.append((judge_label, guard_label))
        if not pairs:
            return "_The runtime guard reported nothing to compare against._"
        report = JudgeCalibrator().agreement([p[0] for p in pairs], [p[1] for p in pairs])
        disagreements = sum(1 for judge, guard in pairs if judge != guard)
        return (
            f"{report.sample_size} cases · observed agreement {report.observed_agreement:.1%} · "
            f"Cohen's κ {report.cohens_kappa:.2f} "
            f"({'acceptable' if report.acceptable else 'below the acceptable bar'}) · "
            f"{disagreements} disagreement(s). The guard runs in the request path with a local "
            "NLI model; the judge reads the whole transcript afterwards. Disagreements are "
            "where one of them is wrong — worth reading case by case."
        )

    def _failure_blocks_for_metric(self, run_id: str, metric: str) -> str:
        rows = self._store.query(
            "SELECT case_id, category, rationale FROM v_failures WHERE run_id = ? AND metric = ? "
            "ORDER BY case_id",
            (run_id, metric),
        )
        return table(
            ["case", "category", "claims"], [list(r.values()) for r in rows[: self._limit]]
        )

    def _lowest(self, run_id: str, stage: str) -> str:
        rows = self._store.query(
            "SELECT case_id, category, round(avg(value), 3) AS mean, "
            "string_agg(metric || ': ' || coalesce(rationale, ''), ' · ') AS why "
            "FROM v_scores WHERE run_id = ? AND stage = ? GROUP BY ALL ORDER BY mean LIMIT ?",
            (run_id, stage, self._limit),
        )
        return table(
            ["case", "category", "mean score", "rationale"], [list(r.values()) for r in rows]
        )

    # -- deepeval / security -----------------------------------------------------

    def _deepeval(self, run: RunRecord) -> str:
        return self._fill(
            "deepeval",
            run_id=run.run_id,
            metrics=self._metric_table(run.run_id, "deepeval"),
            failures=self._failure_blocks(run.run_id, "deepeval", with_side_effects=True),
        )

    def _security(self, run: RunRecord) -> str:
        rows = self._store.query(
            "SELECT category, count(DISTINCT case_id) AS attacks, "
            "count(DISTINCT CASE WHEN NOT passed THEN case_id END) AS succeeded "
            "FROM v_scores WHERE run_id = ? AND stage = 'security' GROUP BY ALL ORDER BY category",
            (run.run_id,),
        )
        return self._fill(
            "security",
            run_id=run.run_id,
            categories=table(
                ["attack category", "attacks", "attacks that got through", "defended"],
                [
                    [
                        row["category"],
                        row["attacks"],
                        row["succeeded"],
                        rate(
                            int(row["attacks"]) - int(row["succeeded"]),
                            int(row["attacks"]),
                            self._z,
                        ),
                    ]
                    for row in rows
                ],
            ),
            metrics=self._metric_table(run.run_id, "security"),
            failures=self._failure_blocks(run.run_id, "security", with_side_effects=True),
        )

    # -- diff ------------------------------------------------------------------

    def _diff_facts(self, base: RunRecord, head: RunRecord) -> str:
        changed = sorted(
            name
            for name in set(base.fingerprint_parts) | set(head.fingerprint_parts)
            if base.fingerprint_parts.get(name) != head.fingerprint_parts.get(name)
        )
        return table(
            ["", base.run_id, head.run_id],
            [
                [
                    "gate",
                    evaluate_gate(self._store, base.run_id, base.suite).verdict,
                    evaluate_gate(self._store, head.run_id, head.suite).verdict,
                ],
                ["cases", base.case_count, head.case_count],
                ["model", base.model, head.model],
                ["fingerprint parts changed", "", ", ".join(changed) or "none"],
            ],
        )

    def _diff_metrics(self, base: RunRecord, head: RunRecord) -> str:
        rows = self._store.query(
            "SELECT coalesce(h.stage, b.stage) AS stage, coalesce(h.metric, b.metric) AS metric, "
            "b.pass_rate AS base, h.pass_rate AS head, round(h.pass_rate - b.pass_rate, 4) AS delta "
            "FROM (SELECT * FROM v_metric_pass_rates WHERE run_id = ?) b "
            "FULL OUTER JOIN (SELECT * FROM v_metric_pass_rates WHERE run_id = ?) h "
            "ON b.stage = h.stage AND b.metric = h.metric ORDER BY delta NULLS LAST, metric",
            (base.run_id, head.run_id),
        )
        return table(["stage", "metric", "base", "head", "Δ"], [list(r.values()) for r in rows])

    def _case_flips(self, base: RunRecord, head: RunRecord, *, regressed: bool) -> str:
        was, now = (True, False) if regressed else (False, True)
        rows = self._store.query(
            "SELECT h.case_id, h.stage, h.metric, h.rationale FROM v_scores h "
            "JOIN v_scores b ON b.case_id = h.case_id AND b.metric = h.metric AND b.stage = h.stage "
            "WHERE b.run_id = ? AND h.run_id = ? AND b.passed = ? AND h.passed = ? "
            "ORDER BY h.case_id, h.metric",
            (base.run_id, head.run_id, was, now),
        )
        return table(["case", "stage", "metric", "now"], [list(r.values()) for r in rows])

    # -- plumbing --------------------------------------------------------------

    def _stages_with_data(self, run_id: str) -> list[str]:
        present = {
            row["stage"]
            for row in self._store.query(
                "SELECT DISTINCT stage FROM v_latest_batches WHERE run_id = ?", (run_id,)
            )
        }
        return [stage for stage in _STAGE_ORDER if stage in present]

    def _turns(self, run_id: str) -> dict[str, list[str]]:
        rows = self._store.query(
            "SELECT DISTINCT c.case_id, c.turns FROM eval_cases c JOIN eval_transcripts t "
            "ON t.case_hash = c.case_hash AND t.case_id = c.case_id WHERE t.run_id = ?",
            (run_id,),
        )
        return {row["case_id"]: json.loads(row["turns"]) for row in rows}

    def _require(self, run_id: str) -> RunRecord:
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"No eval run '{run_id}' in {self._store.db_path}")
        return run

    def _emit(self, run: RunRecord, stage: str, body: str) -> Path:
        return self._write(run, self._settings.files[stage], body)

    def _write(self, run: RunRecord, name: str, body: str) -> Path:
        folder = self._settings.resolved_output_dir / run.run_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(body.rstrip() + "\n")
        return path

    @staticmethod
    def _fill(template: str, **values: Any) -> str:
        text = (_TEMPLATES / f"{template}.md").read_text()
        return Template(text).substitute({key: str(value) for key, value in values.items()})
