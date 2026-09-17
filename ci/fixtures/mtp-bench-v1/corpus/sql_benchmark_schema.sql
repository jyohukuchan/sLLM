-- Schema and analysis queries for the MTP acceptance benchmark.
--
-- One campaign is one immutable measurement session: a fixed engine build, a
-- fixed model lock and one exact GPU target. Comparing series across campaigns
-- is forbidden because the baseline itself can move between them.

CREATE TABLE campaign (
    campaign_id     TEXT PRIMARY KEY,
    started_at      TIMESTAMP WITH TIME ZONE NOT NULL,
    exact_target    TEXT NOT NULL CHECK (exact_target IN ('gfx1030', 'gfx1201')),
    device_uuid     TEXT NOT NULL,
    binary_sha256   CHAR(64) NOT NULL,
    model_lock_sha  CHAR(64) NOT NULL,
    manifest_sha    CHAR(64) NOT NULL,
    profile         TEXT NOT NULL CHECK (profile IN ('full', 'quick')),
    UNIQUE (exact_target, device_uuid, binary_sha256, started_at)
);

CREATE TABLE series (
    series_id       BIGSERIAL PRIMARY KEY,
    campaign_id     TEXT NOT NULL REFERENCES campaign (campaign_id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    companion_encoding TEXT NOT NULL,
    companion_digest   CHAR(64),
    is_baseline     BOOLEAN NOT NULL DEFAULT FALSE,
    UNIQUE (campaign_id, name)
);

CREATE TABLE condition (
    cond_id         TEXT PRIMARY KEY,
    tier            CHAR(1) NOT NULL CHECK (tier IN ('A', 'B')),
    grp             TEXT NOT NULL,
    task            TEXT,
    code_language   TEXT,
    nat_language    TEXT NOT NULL,
    prompt_tokens   INTEGER NOT NULL CHECK (prompt_tokens > 0),
    output_tokens   INTEGER NOT NULL CHECK (output_tokens > 0),
    corpus_sha256   CHAR(64) NOT NULL
);

CREATE TABLE run (
    run_id              BIGSERIAL PRIMARY KEY,
    series_id           BIGINT NOT NULL REFERENCES series (series_id) ON DELETE CASCADE,
    cond_id             TEXT NOT NULL REFERENCES condition (cond_id),
    repetition          SMALLINT NOT NULL CHECK (repetition >= 0),
    is_warmup           BOOLEAN NOT NULL DEFAULT FALSE,
    proposal_blocks     INTEGER NOT NULL CHECK (proposal_blocks >= 0),
    proposed_tokens     INTEGER NOT NULL CHECK (proposed_tokens >= 0),
    accepted_tokens     INTEGER NOT NULL CHECK (accepted_tokens >= 0),
    decode_ns           BIGINT NOT NULL CHECK (decode_ns > 0),
    draft_wall_ns       BIGINT NOT NULL CHECK (draft_wall_ns >= 0),
    prefill_ns          BIGINT NOT NULL CHECK (prefill_ns >= 0),
    generated_sha256    CHAR(64) NOT NULL,
    hip_only            BOOLEAN NOT NULL,
    dispatch_count      BIGINT NOT NULL CHECK (dispatch_count > 0),
    cleanup_leaks       INTEGER NOT NULL CHECK (cleanup_leaks = 0),
    CHECK (accepted_tokens <= proposed_tokens),
    UNIQUE (series_id, cond_id, repetition)
);

CREATE INDEX run_series_condition_idx ON run (series_id, cond_id) WHERE NOT is_warmup;

-- Per-block decomposition. decode_ns is split so a change can be attributed to
-- the draft path or to everything else rather than read as one number.
CREATE VIEW run_decomposition AS
SELECT
    r.run_id,
    r.series_id,
    r.cond_id,
    r.proposal_blocks,
    r.proposed_tokens,
    r.accepted_tokens,
    r.accepted_tokens::NUMERIC / NULLIF(r.proposed_tokens, 0) AS acceptance_rate,
    r.draft_wall_ns::NUMERIC / NULLIF(r.proposal_blocks, 0) / 1e6 AS draft_ms_per_block,
    (r.decode_ns - r.draft_wall_ns)::NUMERIC
        / NULLIF(r.proposal_blocks, 0) / 1e6 AS non_draft_ms_per_block,
    (r.proposal_blocks + r.accepted_tokens)::NUMERIC
        / (r.decode_ns / 1e9) AS decode_tokens_per_second
FROM run AS r
WHERE NOT r.is_warmup;

-- Tier A paired comparison against the campaign baseline. Creative conditions
-- are tier B and are excluded here by construction, never by a later filter.
CREATE VIEW tier_a_paired_delta AS
WITH baseline AS (
    SELECT s.campaign_id, d.cond_id,
           AVG(d.acceptance_rate) AS base_rate,
           AVG(d.decode_tokens_per_second) AS base_tps
    FROM run_decomposition AS d
    JOIN series AS s ON s.series_id = d.series_id
    JOIN condition AS c ON c.cond_id = d.cond_id
    WHERE s.is_baseline AND c.tier = 'A'
    GROUP BY s.campaign_id, d.cond_id
),
candidate AS (
    SELECT s.campaign_id, s.name AS series_name, d.cond_id,
           AVG(d.acceptance_rate) AS cand_rate,
           AVG(d.decode_tokens_per_second) AS cand_tps
    FROM run_decomposition AS d
    JOIN series AS s ON s.series_id = d.series_id
    JOIN condition AS c ON c.cond_id = d.cond_id
    WHERE NOT s.is_baseline AND c.tier = 'A'
    GROUP BY s.campaign_id, s.name, d.cond_id
)
SELECT
    candidate.campaign_id,
    candidate.series_name,
    candidate.cond_id,
    candidate.cand_rate - baseline.base_rate AS acceptance_delta,
    100.0 * (candidate.cand_tps - baseline.base_tps)
        / NULLIF(baseline.base_tps, 0) AS decode_delta_percent
FROM candidate
JOIN baseline
  ON baseline.campaign_id = candidate.campaign_id
 AND baseline.cond_id = candidate.cond_id;

-- Baseline drift guard: flag a campaign whose baseline moved more than 1.5
-- acceptance points away from the recorded reference for the same identity.
CREATE VIEW baseline_drift AS
SELECT
    c.campaign_id,
    c.exact_target,
    AVG(d.acceptance_rate) AS observed_rate,
    r.reference_rate,
    ABS(AVG(d.acceptance_rate) - r.reference_rate) > 0.015 AS drift_exceeded
FROM campaign AS c
JOIN series AS s ON s.campaign_id = c.campaign_id AND s.is_baseline
JOIN run_decomposition AS d ON d.series_id = s.series_id
JOIN condition AS cond ON cond.cond_id = d.cond_id AND cond.tier = 'A'
JOIN baseline_reference AS r
  ON r.exact_target = c.exact_target
 AND r.model_lock_sha = c.model_lock_sha
GROUP BY c.campaign_id, c.exact_target, r.reference_rate;
