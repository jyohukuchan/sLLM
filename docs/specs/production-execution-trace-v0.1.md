# Production execution trace v0.1

## Status

Accepted as the P0 evidence contract for staged implementation. This
specification defines the evidence sidecar for a generic production inference
executor. It does not change a worker, gateway, or manifest wire format by
itself.

## Purpose

`ullm.production_execution_trace.v1` records the execution plan that actually
ran. It binds the model graph, state schema, executor, operator selections,
batch widths, memory plan, commit outcome, and product identity into one
strict JSON document.

The trace answers questions that a benchmark result alone cannot answer:

- whether a measured prefill used a real token/request batch or a token loop;
- which operator implementation and fallback chain actually ran;
- whether state was prepared, committed, discarded, or reset correctly; and
- whether a result came from a component, a full model, or a production server
  request.

This trace complements `inference-benchmark-result-v0.1`; it does not replace
benchmark measurements, correctness evidence, or the served-model manifest.

## Trust boundary

The producer is untrusted evidence. In particular,
`producer.verified=true` only reports the producer's own assertion and MUST
NOT make a trace valid, promotable, or comparable.

An independent validator MUST parse the trace strictly, obtain referenced
identity files through trusted paths, recompute the required digests, and
reconstruct field and counter consistency. A promotion gate accepts only an
independent-validation result of `valid`. The validator MUST be independent of
the producer process and MUST NOT accept a producer-provided hash, fallback
counter, workspace estimate, or `verified` value without recomputing it from
the execution records and referenced artifacts.

The trace is an evidence artifact, not an authorization artifact. The
served-model manifest remains the immutable launch and product contract
defined by `docs/specs/served-model-manifest-v0.1.md`.

## File encoding and bounds

A trace MUST be a non-symlink regular file, MUST NOT be world-writable, and
MUST be at most 4 MiB. It is one strict UTF-8 JSON value with exactly one
top-level object. JSON Lines is not used for this artifact.

The parser MUST reject invalid UTF-8, duplicate keys, trailing data, unknown
keys, missing keys, wrong JSON types, JSON nesting deeper than 24 levels,
more than 32,768 JSON nodes, strings longer than 16,384 UTF-8 bytes, C0
control characters in text, non-finite numbers, negative counters, and
integers outside the JSON safe-integer range `0..=9007199254740991`.

Every object in this specification has an exact field set. Optional values are
represented only by an explicitly permitted `null`; omitted fields are
invalid. Arrays have these upper bounds:

| Field | Maximum entries |
| --- | ---: |
| `phases` | 4,096 |
| `operator_resolutions` | 16,384 |
| `fallback.events` | 4,096 |
| `aggregation.source_trace_sha256s` | 4,096 |
| `shape_bucket.dimensions` | 16 |
| `selection_reason.matched_constraints` | 8 |
| validator failure-code arrays | 128 |

Identifiers use printable ASCII without whitespace unless a field explicitly
allows a human-readable label. SHA-256 values are exactly 64 lowercase
hexadecimal characters. RFC 3339 timestamps are UTC and use an explicit `Z`
suffix.

## Exact top-level shape

Every trace has exactly these fields:

```json
{
  "schema_version": "ullm.production_execution_trace.v1",
  "trace_id": "string",
  "status": "ok|unsupported|oom|failed|skipped",
  "scope": "component|full_model|production_server",
  "created_at": "RFC 3339 UTC timestamp",
  "producer": {},
  "identity": {},
  "graph": {},
  "executor": {},
  "request_summary": {},
  "phases": [],
  "operator_resolutions": [],
  "fallback": {},
  "memory": {},
  "state_commit": {},
  "aggregation": {},
  "server": null,
  "verification": {},
  "failure": null
}
```

`server` is an object only for `scope="production_server"`; it is `null` for
the other scopes. `failure` is `null` exactly when `status="ok"`; all other
statuses require a failure object except a deliberately skipped trace, which
uses a `failure` object with `class="skipped"`.

## Complete JSON example

This complete example is a successful single-request production-server trace.
It is illustrative; the identities are placeholders and are not promotion
evidence.

```json
{
  "schema_version": "ullm.production_execution_trace.v1",
  "trace_id": "prod-2026-07-12T010203Z-0001",
  "status": "ok",
  "scope": "production_server",
  "created_at": "2026-07-12T01:02:03Z",
  "producer": {
    "id": "generic-model-executor",
    "version": "0.1.0",
    "binary_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "verified": true
  },
  "identity": {
    "model": {
      "id": "ullm-qwen3.5-9b-aq4",
      "revision": "aq4-generic-executor-v0.1",
      "format_id": "AQ4_0",
      "implementation_id": "qwen35_aq4_rdna4_v1"
    },
    "served_model_manifest_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
    "worker": {
      "protocol": "ullm.worker.v1",
      "binary_sha256": "1111111111111111111111111111111111111111111111111111111111111111"
    },
    "product": {
      "id": "ullm-qwen3.5-9b-aq4",
      "revision": "aq4-generic-executor-v0.1",
      "identity_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
      "promotion_receipt_sha256": "4444444444444444444444444444444444444444444444444444444444444444"
    },
    "artifact": {
      "manifest_sha256": null,
      "content_sha256": null
    },
    "package": {
      "manifest_sha256": "5555555555555555555555555555555555555555555555555555555555555555"
    }
  },
  "graph": {
    "model_graph": {
      "schema_id": "ullm.model_graph.v0.1",
      "schema_version": "0.1",
      "sha256": "6666666666666666666666666666666666666666666666666666666666666666",
      "source": "adapter_derived"
    },
    "state_schema": {
      "schema_id": "ullm.state_schema.v0.1",
      "schema_version": "0.1",
      "sha256": "7777777777777777777777777777777777777777777777777777777777777777",
      "source": "adapter_derived"
    },
    "compatibility_key_sha256": "8888888888888888888888888888888888888888888888888888888888888888"
  },
  "executor": {
    "id": "generic_model_executor",
    "version": "0.1.0",
    "mode": "graph_lowered",
    "backend": "hip",
    "device": {
      "runtime_device_index": 1,
      "name": "Radeon AI PRO R9700",
      "architecture": "RDNA4"
    }
  },
  "request_summary": {
    "fixture_id": "release-chat-ja-short-v1",
    "request_count": 1,
    "prompt_token_count": 128,
    "cached_prefix_token_count": 0,
    "generated_token_count": 16,
    "context_tokens_at_decode_start": 128,
    "prompt_or_token_content_recorded": false
  },
  "phases": [
    {
      "phase_id": "cold-prefill-0",
      "kind": "cold_prefill",
      "executor_id": "generic_model_executor",
      "executor_version": "0.1.0",
      "prefill_mode": "cold",
      "chunk_width_tokens": 32,
      "actual_token_batch_width": 32,
      "actual_request_batch_width": 1,
      "request_count": 1,
      "input_token_count": 128,
      "output_token_count": 0,
      "cached_prefix_token_count": 0,
      "context_tokens_before": 0,
      "context_tokens_after": 128,
      "wall_time_ms": 42.5
    },
    {
      "phase_id": "decode-0",
      "kind": "decode",
      "executor_id": "generic_model_executor",
      "executor_version": "0.1.0",
      "prefill_mode": null,
      "chunk_width_tokens": 1,
      "actual_token_batch_width": 1,
      "actual_request_batch_width": 1,
      "request_count": 1,
      "input_token_count": 16,
      "output_token_count": 16,
      "cached_prefix_token_count": 0,
      "context_tokens_before": 128,
      "context_tokens_after": 144,
      "wall_time_ms": 320.0
    }
  ],
  "operator_resolutions": [
    {
      "phase_kind": "cold_prefill",
      "operator_instance_id": "layer-0-self-attn-qkv",
      "op_kind": "fused_linear_group",
      "implementation_id": "aq4_matvec_batch_rdna4_v1",
      "implementation_version": "1",
      "resolution_status": "selected",
      "backend": "hip",
      "device": "Radeon AI PRO R9700",
      "formats": {
        "weight": "AQ4_0",
        "activation": "F32",
        "state": null,
        "layout": "row_major_grouped"
      },
      "shape_bucket": {
        "id": "m32-k3584-n4096",
        "dimensions": [
          { "name": "m", "value": 32 },
          { "name": "k", "value": 3584 },
          { "name": "n", "value": 4096 }
        ]
      },
      "selection_reason": {
        "kind": "highest_specificity_priority",
        "candidate_count": 3,
        "score": 6,
        "priority": 20,
        "matched_constraints": ["format", "gpu_arch", "gpu_name"]
      },
      "architecture_constraint": {
        "model_arch": "Qwen3.5",
        "gpu_arch": "RDNA4",
        "gpu_name": "Radeon AI PRO R9700"
      },
      "workspace": {
        "planned_bytes": 8388608,
        "observed_peak_bytes": 7340032
      },
      "invocation_count": 4
    }
  ],
  "fallback": {
    "fallback_count": 0,
    "unexpected_fallback_count": 0,
    "unsupported_count": 0,
    "fail_closed_count": 0,
    "events": []
  },
  "memory": {
    "vram_capacity_bytes": 34359738368,
    "resident_bytes": 9985798963,
    "persistent_state_bytes": 268435456,
    "planned_temporary_bytes": 67108864,
    "planned_total_bytes": 10392062147,
    "planned_headroom_bytes": 23967676221,
    "observed_peak_bytes": 10468982784,
    "observed_headroom_bytes": 23890755584,
    "observer": {
      "kind": "runtime_allocator",
      "sample_count": 6,
      "complete": true
    },
    "oom": null
  },
  "state_commit": {
    "prepared_batch_count": 5,
    "committed_batch_count": 5,
    "discarded_batch_count": 0,
    "stale_nonce_count": 0,
    "cancelled_batch_count": 0,
    "error_batch_count": 0,
    "reset": {
      "required": true,
      "attempted": true,
      "complete": true,
      "failed": false
    }
  },
  "aggregation": {
    "is_aggregated": false,
    "source_trace_sha256s": [],
    "component_trace_count": 0,
    "full_model_trace_count": 0,
    "coverage": "production_server"
  },
  "server": {
    "transport": "jsonl_sidecar",
    "protocol": "ullm.worker.v1",
    "observation": "per_request",
    "request_trace_count": 1,
    "request_count": 1,
    "ready_observed": true,
    "release_observed": true,
    "gateway": "openai_gateway",
    "openwebui_observed": false
  },
  "verification": {
    "producer_verified": true,
    "independent_validation": {
      "status": "not_run",
      "validator_id": null,
      "validator_version": null,
      "report_sha256": null,
      "failure_codes": []
    }
  },
  "failure": null
}
```

## Field rules

### Common values

- `schema_version` is exactly `ullm.production_execution_trace.v1`.
- `trace_id` is nonempty, unique within its producer run, and contains no
  prompt, generated text, token ID, request ID, account ID, or user data.
- `status` is one of `ok`, `unsupported`, `oom`, `failed`, or `skipped`.
- `scope` is one of `component`, `full_model`, or `production_server`.
- `created_at` is the time that the producer finalized the trace, not a
  synthetic benchmark timestamp.
- Every byte count and counter is a safe nonnegative integer. Every duration
  is a finite nonnegative JSON number in milliseconds.

### `producer`

`producer` has exactly `id`, `version`, `binary_sha256`, and `verified`.
`binary_sha256` identifies the executable that emitted the trace. `verified`
is a boolean self-assertion only. It has no effect on validation or promotion.

### `identity`

`identity` has exactly `model`, `served_model_manifest_sha256`, `worker`,
`product`, `artifact`, and `package`.

- `model` has exactly `id`, `revision`, `format_id`, and `implementation_id`.
- `worker` has exactly `protocol` and `binary_sha256`.
- `product` has exactly `id`, `revision`, `identity_sha256`, and
  `promotion_receipt_sha256`.
- `artifact` has exactly `manifest_sha256` and `content_sha256`; both are
  `null` only if this execution has no separate quantization artifact.
- `package` has exactly `manifest_sha256`.

For `full_model` and `production_server`, every identity digest is required
except the two artifact digests when `artifact` is absent. The worker digest
must equal `producer.binary_sha256`. For `production_server`,
`served_model_manifest_sha256` and the worker protocol must match the active
served-model manifest. Component traces may use `null` for identities that do
not exist in a synthetic or isolated fixture, but that makes them ineligible
for production promotion and production aggregation.

### `graph`

`graph` has exactly `model_graph`, `state_schema`, and
`compatibility_key_sha256`.

Both `model_graph` and `state_schema` have exactly `schema_id`,
`schema_version`, `sha256`, and `source`. `source` is `serialized` or
`adapter_derived`. Their digests are over the canonical, validated schema
representation, not a file path or producer label. The compatibility-key
digest binds the graph digest, state-schema digest, logical format/layout, and
backend compatibility inputs used to form each `ExecutionBatch`.

### `executor` and `request_summary`

`executor` has exactly `id`, `version`, `mode`, `backend`, and `device`.
`device` has exactly `runtime_device_index`, `name`, and `architecture`.
Executor IDs and versions are stable implementation identities; a benchmark
label is not an executor identity.

`request_summary` has exactly `fixture_id`, `request_count`,
`prompt_token_count`, `cached_prefix_token_count`, `generated_token_count`,
`context_tokens_at_decode_start`, and `prompt_or_token_content_recorded`.
`fixture_id` is a public fixture identifier or `null`; it is never prompt text.
The final boolean MUST be `false`.

### `phases`

Every phase has exactly these fields:

```text
phase_id, kind, executor_id, executor_version, prefill_mode,
chunk_width_tokens, actual_token_batch_width, actual_request_batch_width,
request_count, input_token_count, output_token_count,
cached_prefix_token_count, context_tokens_before, context_tokens_after,
wall_time_ms
```

`kind` is `cold_prefill`, `cached_prefix_prefill`, or `decode`.
`prefill_mode` is `cold` for `cold_prefill`, `cached_prefix` for
`cached_prefix_prefill`, and `null` for `decode`. A phase records its requested
chunk width and the widths actually executed. The actual widths are not
inferred from workload concurrency or progress-event cadence.

For prefill, `chunk_width_tokens` is positive and
`actual_token_batch_width` is the maximum token width that the executor used
for that phase. For decode, the chunk width and token width may be one, but a
request-batched decode must record its actual request width. A zero-width
phase is invalid. Phase counts must reconcile with `request_summary` and with
the commit counters.

### `operator_resolutions`

Every operator-resolution entry has exactly these fields:

```text
phase_kind, operator_instance_id, op_kind, implementation_id,
implementation_version, resolution_status, backend, device, formats,
shape_bucket, selection_reason, architecture_constraint, workspace,
invocation_count
```

- `phase_kind` is one of the phase kinds in `phases`.
- `op_kind` is a stable semantic operator kind from the model graph, not a
  model-name branch.
- `implementation_id` and `implementation_version` identify the concrete
  backend implementation. They correspond to stable backend-registry IDs such
  as the operation/phase/format/GPU descriptors in
  `crates/ullm-engine/src/backend_dispatch.rs`.
- `resolution_status` is `selected`, `fallback`, `unsupported`, or
  `fail_closed`.
- `formats` has exactly `weight`, `activation`, `state`, and `layout`; each is
  a nonempty stable format/layout ID or an explicitly allowed `null` when the
  operator has no state.
- `shape_bucket` has exactly `id` and `dimensions`. Each dimensions entry has
  exactly `name` and `value`.
- `selection_reason` has exactly `kind`, `candidate_count`, `score`,
  `priority`, and `matched_constraints`. Its `kind` is one of
  `exact_match`, `highest_specificity_priority`, `generic_fallback`,
  `workspace_limited_fallback`, `unsupported`, or `fail_closed`.
- `architecture_constraint` is `null` or an object with exactly `model_arch`,
  `gpu_arch`, and `gpu_name`. It records an optional selection constraint; it
  does not redefine graph semantics.
- `workspace` has exactly `planned_bytes` and `observed_peak_bytes`. The
  observed value may be `null` only when the runtime has no operator-level
  observer; production promotion requires a non-null aggregate peak in
  `memory`.

One entry may aggregate repeated invocations only when every listed field is
identical. `invocation_count` is then the exact total. Entries may not hide a
different implementation, fallback, shape bucket, or workspace plan behind a
shared operator name.

### `fallback`, `memory`, and `state_commit`

`fallback` has exactly `fallback_count`, `unexpected_fallback_count`,
`unsupported_count`, `fail_closed_count`, and `events`. Every fallback event
has exactly `phase_kind`, `op_kind`, `from_implementation_id`,
`to_implementation_id`, `reason_code`, and `classification`.
`classification` is `expected`, `unexpected`, `unsupported`, or
`fail_closed`. An event exists for every non-`selected` operator resolution;
the counts must equal the classified events. A fallback is never omitted just
because the final output was correct.

`memory` has exactly `vram_capacity_bytes`, `resident_bytes`,
`persistent_state_bytes`, `planned_temporary_bytes`, `planned_total_bytes`,
`planned_headroom_bytes`, `observed_peak_bytes`, `observed_headroom_bytes`,
`observer`, and `oom`. `observer` has exactly `kind`, `sample_count`, and
`complete`. `oom` is `null` or an object with exactly `stage`, `reason_code`,
`planned_bytes`, and `observed_peak_bytes`. `planned_total_bytes` must be the
sum of resident, persistent-state, and planned-temporary bytes;
`planned_headroom_bytes` must be capacity minus planned total. A non-OOM
production trace requires a complete observer and a non-null observed peak and
headroom. OOM remains visible even if a later smaller run succeeds.

`state_commit` has exactly `prepared_batch_count`, `committed_batch_count`,
`discarded_batch_count`, `stale_nonce_count`, `cancelled_batch_count`,
`error_batch_count`, and `reset`. `reset` has exactly `required`, `attempted`,
`complete`, and `failed`. Every prepared batch is exactly one of committed or
discarded. A stale nonce, cancellation, or execution error must discard the
affected prepared batch before it becomes visible. A request that requires a
reset is promotable only when `reset.complete=true` and `reset.failed=false`.
These rules preserve the prepare/publish/commit/reset ordering in
`crates/ullm-engine/src/worker_driver.rs`.

### `aggregation`, `server`, `verification`, and `failure`

`aggregation` has exactly `is_aggregated`, `source_trace_sha256s`,
`component_trace_count`, `full_model_trace_count`, and `coverage`.
`coverage` is `component`, `full_model`, or `production_server` and must equal
the top-level scope. Sources are content digests of complete source traces;
they do not replace the target trace's own identity validation.

`server` is `null` except for `production_server`. A server object has exactly
`transport`, `protocol`, `observation`, `request_trace_count`, `request_count`,
`ready_observed`, `release_observed`, `gateway`, and `openwebui_observed`.
`observation` is `per_request` or `run_summary`. A per-request trace has
`request_trace_count=1` and `request_count=1`; a run summary has equal,
nonzero trace and request counts unless it explicitly aggregates one trace per
request through `aggregation.source_trace_sha256s`. `ready_observed` and
`release_observed` must be true for a successful production trace.

`verification` has exactly `producer_verified` and `independent_validation`.
It must copy `producer.verified` exactly. `independent_validation` has exactly
`status`, `validator_id`, `validator_version`, `report_sha256`, and
`failure_codes`; its status is `not_run`, `valid`, or `invalid`. `valid`
requires all validator identity fields and a report digest. `not_run` requires
those three values to be `null` and an empty failure-code array.

`failure` is `null` for `ok`; otherwise it has exactly `class`, `stage`,
`reason_code`, and `message`. `class` is `unsupported`, `oom`, `execution`,
`validation`, or `skipped`. Messages must be bounded diagnostic text and MUST
NOT contain prompt text, token IDs, generated text, or user identifiers.

## Scope semantics

The scope is an assertion about the executed boundary, not a quality label.

| Scope | Required coverage | Promotion eligibility |
| --- | --- | --- |
| `component` | One operator or declared subgraph only | Never eligible |
| `full_model` | Every graph node through final logits/sampling for the declared request | Never eligible by itself |
| `production_server` | Full-model execution observed through the resident worker/server request boundary | The only eligible scope |

A `component` trace MUST use `aggregation.coverage="component"` and cannot
claim server observation. A `full_model` trace MUST prove all graph nodes,
including final head/sampling, but it may still be an offline runner. A
`production_server` trace MUST identify the active served-model manifest,
worker binary, server protocol, and ready/release observation.

Production promotion requires all of the following in addition to separate
correctness and performance gates:

1. `scope="production_server"` and `status="ok"`;
2. complete non-null production identities and validated graph/state digests;
3. `verification.independent_validation.status="valid"`;
4. no `unexpected` fallback, unsupported operator, fail-closed event, OOM, or
   incomplete reset; and
5. reconciled phase, operator, memory, and commit counters.

## Operator resolution

The trace records the resolved implementation after graph lowering. It does
not record merely the preferred implementation or a registry catalog entry.
The selection reason must make it possible to distinguish exact format/device
selection, a priority choice, a generic fallback, a workspace-limited fallback,
unsupported capability, and a fail-closed rejection.

The backend registry may use operation, phase, format, optional model
architecture, GPU architecture, GPU name, and priority as matching inputs.
The trace captures the evaluated result, including the optional architecture
constraint, so a production result can be audited without assuming that a
specific GPU override was selected.

An operator trace is grouped only after selection. For example, two calls to
the same semantic linear operator with different `M` shape buckets require two
entries, even if they share an implementation ID.

## Fallback and errors

Fallback is an observable result, not a silent recovery mechanism.

- `selected` means the initially chosen supported implementation ran.
- `fallback` means a different supported implementation ran; its event must
  identify the prior implementation and stable reason code.
- `unsupported` means no supported implementation exists for the requested
  operation/shape/backend. It must not be converted into a generic result.
- `fail_closed` means execution stopped rather than choosing an unapproved or
  unvalidated implementation.

An `unexpected` fallback is a promotion blocker even when the trace status is
`ok`. An `unsupported` or `fail_closed` operator requires a non-`ok` trace
status unless the entire phase was intentionally skipped before execution.
Validator reconstruction must compare every operator resolution with the
fallback events and counts; a producer cannot hide a fallback by reporting
only a successful final output.

## Identity binding

For a production trace, the independent validator MUST verify these bindings:

1. `identity.served_model_manifest_sha256` equals the canonical active
   `ullm.served_model.v1` manifest digest.
2. The manifest worker binary digest equals both
   `identity.worker.binary_sha256` and `producer.binary_sha256`.
3. The manifest model ID, revision, format ID, implementation ID, package
   manifest digest, artifact identity when present, and product/promotion
   identities equal their trace counterparts.
4. The graph and state schema were derived from, or validated against, those
   package/artifact identities and the recorded compatibility key.
5. The executor ID/version and backend/device identity match the resolved
   runtime selection, not an environment label supplied by a benchmark tool.

Any mismatch is an invalid trace, even if its numerical output is correct.
Component traces may omit unavailable identities with `null`, but they cannot
be transformed into a promotion trace by adding external labels later.

## Aggregation

Aggregation preserves scope rather than upgrading it.

- Component traces can be summarized as component evidence only. Their
  operator timings or batch widths cannot prove full-model execution.
- A full-model trace may list component source traces, but it must independently
  execute and record every graph node. It cannot inherit full-model status
  merely by listing components.
- A production-server trace must independently observe the full model through
  the server boundary. It may reference full-model/component source traces for
  diagnosis, but those traces cannot supply missing server evidence.
- A server run summary must represent either one independently valid trace per
  request or an explicitly validated aggregate over request traces. It must
  preserve actual batch widths, fallback counts, OOM outcomes, and commit/reset
  totals; it must not average away an unexpected fallback or a failed request.

The validator rejects a scope/coverage mismatch, a source trace with a higher
claimed scope than the target can prove, duplicate source digests, or an
aggregate whose counters do not equal the sums of its sources.

## Validation

Independent validation is required before a production trace is used in a
promotion receipt or compared as production performance evidence. The
validator MUST:

1. enforce the strict JSON, UTF-8, size, depth, node, field-set, string,
   array, finite-number, and safe-integer rules;
2. recompute and cross-check manifest, binary, product, artifact, package,
   graph, state-schema, compatibility-key, source-trace, and validator-report
   digests from canonical inputs;
3. verify scope-specific required identities and the server ready/release
   boundary;
4. reconstruct requested and actual phase token/request counts, context
   transitions, and batch widths from executor records;
5. recompute operator-selection compatibility, selected implementation IDs,
   workspace estimates, fallback classifications, and invocation totals;
6. verify memory arithmetic, capacity/headroom arithmetic, OOM visibility, and
   observer completeness;
7. verify prepared/committed/discarded accounting, stale nonce/cancel/error
   handling, and reset completion; and
8. reject forbidden prompt text, token IDs, generated text, request IDs, or
   user identifiers.

The producer's `verified` field is not an input to any of these checks. A
validator may write an external signed or content-addressed report and then
set `independent_validation` through a separate, auditable finalization step.

## Privacy

Traces MUST NOT contain prompt text, prompt token IDs, generated token IDs,
generated text, request IDs, account IDs, client addresses, authorization
data, or raw HTTP headers. They contain only aggregate counts and, optionally,
a public fixture ID. Logs cited by a trace must apply the same rule or be kept
outside promotion evidence.

## Compatibility

The current `ullm.worker.v1` protocol is unchanged. During the first rollout,
the trace is written as a sidecar artifact after the worker/server execution
boundary and is associated by a trusted local run record, not by extending
current JSONL events. Existing `ready` and `released` events therefore remain
valid v1 messages.

A later protocol revision may add a trace digest to `ready` and/or `released`
after explicit capability negotiation and strict parser updates. Until then,
an absent wire digest is expected and must not be interpreted as a validated
trace. Sidecar attachment does not grant promotion eligibility; the scope and
independent-validation rules in this specification still apply.

## References

- `docs/plans/generic-production-inference-optimization-plan-v0.1.md`
- `docs/decisions/0004-model-graph-and-state-schema.md`
- `docs/decisions/0005-backend-operation-registry.md`
- `docs/specs/inference-benchmark-result-v0.1.md`
- `docs/specs/prefill-validation-v0.1.md`
- `docs/specs/served-model-manifest-v0.1.md`
- `crates/ullm-engine/src/backend_dispatch.rs`
- `crates/ullm-engine/src/worker_driver.rs`
