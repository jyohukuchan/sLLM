# Generic reasoning release evidence v0.1

Schema ID: `ullm.generic_reasoning_release_evidence.v1`

This document defines the hash-only release evidence consumed by
`tools/validate-generic-reasoning-release.py`. It is a candidate-release
artifact; it does not activate a served model and must not contain user
conversation text or credentials.

The validator reads at most 16 MiB, accepts at most 4,096 cases, and bounds
each case at 1,000,000 SSE chunks. Producers should keep evidence well below
these limits by recording hashes and aggregate metadata rather than bodies.

## Root object

The root object has exactly these fields:

```text
schema_version: "ullm.generic_reasoning_release_evidence.v1"
status: "incomplete" | "complete"
production_activation_performed: false
source_commit: lowercase 40-character Git commit
active_promotion_source_commit: lowercase 40-character Git commit
source_commit_aligned: boolean
git_worktree_clean: boolean
git_worktree_status_sha256: lowercase SHA-256 of bounded Git status text
identity: object
cases: nonempty array
lifecycle: object
```

The validator recomputes `source_commit_aligned` by comparing the two commit
values and rejects a declaration that differs from that comparison.
The producer computes `git_worktree_status_sha256` from
`git status --porcelain=v1 --untracked-files=all -- . ':(exclude).rocprofv3'`.
The profiling directory is pre-existing workspace state and is outside this
release's worktree scope. A production candidate must set
`git_worktree_clean=true`; a dirty declaration remains structurally valid but
is not production-gate eligible.

`tools/prepare-generic-reasoning-release-evidence.py` assembles this root from
a pre-sanitized measured-case array and hashes the manifest, worker, and the
tokenizer files named by the manifest. Before hashing, it re-runs the existing
served-model contract validator, so an evidence artifact cannot bind to a
manifest that the gateway would reject. It rejects forbidden cleartext fields
and runs the independent release validator before publishing. The optional
`--lifecycle` input contains sanitized `request_released` records correlated by
case ID; a complete artifact must contain one matching event for every case.
`--status complete` additionally requires the recomputed production gate to be
eligible.

`tools/run-generic-reasoning-release-campaign.py` is the production collector
for the measured-case input. It requires an immutable HTTP probe image, a v2
served-model manifest, and the five fixture IDs. Before any request it checks
that gfx1201 has a resident `ullm-aq4-worker` and no `llama-server` or other
positive-VRAM process. Each streamed request is correlated with one matching
`request_released` event through a temporary Unix datagram observer. The
collector publishes only `cases.json`, sanitized `lifecycle.json`, bounded
resource samples, and a summary; prompt and response text remain in memory for
the quality check and are never written.

The validator report also contains `timing_percentiles` grouped by mode and
timing field. It recomputes p50, p95, and p99 with linear interpolation over
the raw case values and includes the contributing sample count; producer-side
percentile declarations are not trusted.

The report also contains `quality_summary` grouped by mode with raw total,
correct count, and recomputed accuracy.

It contains `resource_percentiles` grouped by mode and resource field, with
p50, p95, p99, maximum, and contributing sample count recomputed from raw
RSS/VRAM/temperature/power observations.

`lifecycle` has the following shape:

```text
schema_version: "ullm.generic_reasoning_lifecycle_evidence.v1"
events: array
```

Each event contains only bounded accounting and timing metadata: `case_id`,
stream/outcome, prompt and completion token counts, reset completion,
reasoning/forced-end counts (or `null` for disabled requests), and the three
nonnegative lifecycle durations. The validator matches every event to its
case, recomputes the accounting comparison, and rejects unknown or duplicate
case IDs. Request IDs, prompt text, response text, and credentials are not
stored.

`identity` has exactly `manifest_sha256`, `worker_binary_sha256`,
`tokenizer_sha256`, and `openwebui_image`. The first three values are lowercase
SHA-256 strings. `openwebui_image` is a content-addressed image reference in
the form `name@sha256:<64 lowercase hex characters>`.

## Case object

Each case has exactly these fields:

```text
id: bounded nonempty string
mode: disabled | budget-32 | budget-128 | budget-256 | unbounded
prompt_fixture_id: bounded nonempty string
prompt_sha256: lowercase SHA-256
stream: boolean
http_status: 200
sse_chunk_count: nonnegative integer
finish_reason: stop | length
raw: object
timing: object
resource: object
quality: object
```

`raw` records `prompt_tokens`, `completion_tokens`, `reasoning_tokens`,
`forced_end_tokens`, `answer_tokens`, `budget_overshoot`,
`empty_answer`, and `usage_completion_tokens`. The validator recomputes:

```text
completion_tokens = reasoning_tokens + forced_end_tokens + answer_tokens
usage_completion_tokens = completion_tokens
budget_overshoot = 0
empty_answer = false
answer_tokens >= 1
```

The disabled mode must have zero reasoning and forced-end tokens. Bounded
modes must not exceed their named reasoning budget. The unbounded mode has no
configured reasoning-token ceiling in this schema.

`timing` contains bounded nonnegative values or `null` for prefill rate,
first reasoning token, first answer token, reasoning decode rate, answer
decode rate, total decode rate, and latency. `resource` contains bounded
nonnegative RSS delta, VRAM delta, GPU temperature, and power values.
`quality` contains a boolean `correct` and a score in the inclusive range
`0..1`.

The following keys are forbidden anywhere in the object, including nested
objects: `prompt`, `response`, `request_body`, `response_body`,
`authorization`, `api_key`, `token`, and `conversation`. Fixture IDs and hashes
are used instead.

## Gate semantics

Structural validity means that every object and value follows this contract.
Production-gate eligibility additionally requires all five modes, aligned
source identity, `status=complete`, `quality.correct=true` for every case, and
the common timing values (`prefill_tokens_per_second`,
`first_answer_token_ms`, `answer_decode_tokens_per_second`,
`decode_tokens_per_second`, and `latency_ms`) present for every case. A
structurally valid but incomplete artifact is expected during a measurement run
and must produce `gate_eligible=false`. `--require-complete` exits with status
`2` for that condition and with status `1` for structural or security
violations.

After the six release artifacts have been staged in one directory,
`tools/prepare-generic-reasoning-release-bundle.py` records their relative
hashes and the hashes of the previous active manifest, systemd unit, and
environment file. It invokes `tools/validate-generic-reasoning-release-bundle.py`
before publishing. The producer rejects symlinked or out-of-directory artifact
paths and `--status complete` requires the recomputed bundle gate to be
eligible.
