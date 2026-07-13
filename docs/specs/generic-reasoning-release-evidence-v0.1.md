# Generic reasoning release evidence v0.1

Schema ID: `ullm.generic_reasoning_release_evidence.v1`

This document defines the hash-only release evidence consumed by
`tools/validate-generic-reasoning-release.py`. It is a candidate-release
artifact; it does not activate a served model and must not contain user
conversation text or credentials.

## Root object

The root object has exactly these fields:

```text
schema_version: "ullm.generic_reasoning_release_evidence.v1"
status: "incomplete" | "complete"
production_activation_performed: false
source_commit: lowercase 40-character Git commit
active_promotion_source_commit: lowercase 40-character Git commit
source_commit_aligned: boolean
identity: object
cases: nonempty array
```

The validator recomputes `source_commit_aligned` by comparing the two commit
values and rejects a declaration that differs from that comparison.

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
