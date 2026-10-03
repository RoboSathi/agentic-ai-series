-- Stable analytics views. Reports read these; so can you (`make eval-sql`).
-- "Latest" always means the most recent completed batch of a stage.

CREATE OR REPLACE VIEW v_latest_batches AS
SELECT * EXCLUDE (rn) FROM (
    SELECT s.*, row_number() OVER (
        PARTITION BY run_id, stage ORDER BY started_at DESC) AS rn
    FROM eval_stages s
    WHERE status = 'completed'
) WHERE rn = 1;

CREATE OR REPLACE VIEW v_scores AS
SELECT sc.*, t.category, t.suite
FROM eval_scores sc
JOIN v_latest_batches b ON b.batch_id = sc.batch_id
JOIN eval_transcripts t ON t.run_id = sc.run_id AND t.case_id = sc.case_id;

CREATE OR REPLACE VIEW v_run_summary AS
SELECT
    r.run_id, r.suite, r.agent_id, r.status, r.started_at, r.finished_at,
    r.case_count, r.model, r.judge_model, r.rate_limit, r.git_sha, r.git_dirty,
    left(r.agent_fingerprint, 12) AS fingerprint,
    (SELECT count(*) FROM eval_transcripts t WHERE t.run_id = r.run_id) AS transcripts,
    (SELECT count(*) FROM eval_transcripts t
        WHERE t.run_id = r.run_id AND t.error <> '') AS errored,
    (SELECT count(DISTINCT case_id) FROM v_scores s
        WHERE s.run_id = r.run_id AND s.severity = 'hard' AND NOT s.passed) AS hard_failed_cases,
    (SELECT round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END), 4) FROM v_scores s
        WHERE s.run_id = r.run_id AND s.severity = 'soft') AS soft_pass_rate,
    (SELECT round(sum(cost_usd), 6) FROM eval_transcripts t WHERE t.run_id = r.run_id)
        AS generation_cost_usd
FROM eval_runs r;

CREATE OR REPLACE VIEW v_metric_pass_rates AS
SELECT run_id, stage, kind, grader, metric, severity,
       count(*) AS cases,
       sum(CASE WHEN passed THEN 1 ELSE 0 END) AS passed,
       round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END), 4) AS pass_rate,
       round(avg(value), 4) AS mean_value,
       round(min(value), 4) AS min_value,
       any_value(threshold) AS threshold
FROM v_scores
GROUP BY ALL;

CREATE OR REPLACE VIEW v_category_breakdown AS
SELECT run_id, stage, category,
       count(DISTINCT case_id) AS cases,
       count(DISTINCT CASE WHEN NOT passed THEN case_id END) AS failing_cases,
       round(avg(CASE WHEN passed THEN 1.0 ELSE 0.0 END), 4) AS score_pass_rate
FROM v_scores
GROUP BY ALL;

CREATE OR REPLACE VIEW v_hard_failures AS
SELECT run_id, stage, case_id, category, metric, grader, value, rationale
FROM v_scores
WHERE severity = 'hard' AND NOT passed;

CREATE OR REPLACE VIEW v_failures AS
SELECT run_id, stage, case_id, category, metric, grader, severity, value, threshold, rationale
FROM v_scores
WHERE NOT passed;

CREATE OR REPLACE VIEW v_cost_latency AS
SELECT run_id, category,
       count(*) AS cases,
       sum(CASE WHEN reused_from IS NULL THEN 1 ELSE 0 END) AS generated,
       sum(llm_calls) AS llm_calls,
       round(avg(latency_ms), 1) AS mean_latency_ms,
       round(quantile_cont(latency_ms, 0.9), 1) AS p90_latency_ms,
       round(sum(cost_usd), 6) AS cost_usd
FROM eval_transcripts
GROUP BY ALL;

CREATE OR REPLACE VIEW v_stage_cost AS
SELECT run_id, stage, batch_id, status, started_at, finished_at,
       date_diff('second', started_at, finished_at) AS seconds,
       cases, reused, provider_requests, throttled, throttle_wait_s, http_429
FROM eval_stages;

CREATE OR REPLACE VIEW v_metric_trend AS
SELECT r.suite, r.started_at, p.run_id, p.stage, p.metric, p.pass_rate, p.mean_value
FROM v_metric_pass_rates p
JOIN eval_runs r USING (run_id)
ORDER BY r.suite, p.metric, r.started_at;
