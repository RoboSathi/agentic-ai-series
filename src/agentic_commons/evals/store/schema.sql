-- Eval analytics (DuckDB). Append-heavy, queried by reports and ad hoc SQL.
-- A *run* is one suite generated against one agent fingerprint and one
-- dataset version; every stage after generation scores that run's
-- transcripts in a *batch*.

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id             VARCHAR PRIMARY KEY,
    suite              VARCHAR NOT NULL,
    agent_id           VARCHAR NOT NULL,
    agent_fingerprint  VARCHAR NOT NULL,
    fingerprint_parts  JSON,
    dataset_hash       VARCHAR NOT NULL,
    case_count         INTEGER NOT NULL,
    git_sha            VARCHAR,
    git_dirty          BOOLEAN,
    model              VARCHAR,
    judge_model        VARCHAR,
    rate_limit         INTEGER,
    started_at         TIMESTAMP NOT NULL,
    finished_at        TIMESTAMP,
    status             VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_cases (
    suite      VARCHAR NOT NULL,
    case_id    VARCHAR NOT NULL,
    case_hash  VARCHAR NOT NULL,
    category   VARCHAR NOT NULL,
    tags       VARCHAR[],
    turns      JSON NOT NULL,
    payload    JSON NOT NULL,
    PRIMARY KEY (suite, case_id, case_hash)
);

CREATE TABLE IF NOT EXISTS eval_transcripts (
    run_id             VARCHAR NOT NULL,
    suite              VARCHAR NOT NULL,
    case_id            VARCHAR NOT NULL,
    case_hash          VARCHAR NOT NULL,
    category           VARCHAR NOT NULL,
    agent_fingerprint  VARCHAR NOT NULL,
    transcript_hash    VARCHAR NOT NULL,
    reply              VARCHAR,
    transcript         JSON NOT NULL,
    llm_calls          INTEGER,
    prompt_tokens      INTEGER,
    completion_tokens  INTEGER,
    cost_usd           DOUBLE,
    latency_ms         DOUBLE,
    error              VARCHAR,
    reused_from        VARCHAR,
    created_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS eval_stages (
    batch_id           VARCHAR PRIMARY KEY,
    run_id             VARCHAR NOT NULL,
    stage              VARCHAR NOT NULL,
    grader_fingerprint VARCHAR,
    started_at         TIMESTAMP NOT NULL,
    finished_at        TIMESTAMP,
    status             VARCHAR NOT NULL,
    cases              INTEGER,
    reused             INTEGER,
    provider_requests  INTEGER,
    throttled          INTEGER,
    throttle_wait_s    DOUBLE,
    http_429           INTEGER,
    detail             VARCHAR
);

CREATE TABLE IF NOT EXISTS eval_scores (
    batch_id         VARCHAR NOT NULL,
    run_id           VARCHAR NOT NULL,
    stage            VARCHAR NOT NULL,
    case_id          VARCHAR NOT NULL,
    transcript_hash  VARCHAR NOT NULL,
    metric           VARCHAR NOT NULL,
    grader           VARCHAR NOT NULL,
    kind             VARCHAR NOT NULL,
    severity         VARCHAR NOT NULL,
    value            DOUBLE NOT NULL,
    passed           BOOLEAN NOT NULL,
    threshold        DOUBLE,
    tier             VARCHAR NOT NULL,
    rationale        VARCHAR,
    cost_usd         DOUBLE,
    llm_calls        INTEGER,
    reused           BOOLEAN NOT NULL,
    scored_at        TIMESTAMP NOT NULL
);
