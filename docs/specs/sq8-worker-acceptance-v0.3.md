# SQ8 Standalone Worker Acceptance Evidence v0.3

Status: ratified production contract

Date: 2026-07-24

## 1. Version boundary

This contract incorporates the complete workload, R9700 identity, HIP guards,
clock, deadlines, cancellation matrix, KFD double-collect snapshots, resource
sampling, frozen statistics, publication, and failure behavior from
`sq8-worker-acceptance-v0.2.md`.

It changes only the candidate/manifest identity and worker protocol boundary:

- raw schema: `ullm.sq8.worker_acceptance.raw.v3`;
- validation schema: `ullm.sq8.worker_acceptance.validation.v3`;
- manifest schema: `ullm.served_model.v2`; and
- command/event schema: `ullm.worker.v2`.

The current producer emits only raw.v3. The validator dispatches by the first
record to explicit raw.v1, raw.v2, or raw.v3 contracts. Every later record must
retain that discriminator. Raw.v1 and raw.v2 remain worker-v1 schemas with
their historical KFD and result shapes; no v3 field is accepted in them.

## 2. Candidate identity

The producer takes one required `--served-model-manifest` path and launches
the validated manifest's exact worker as:

```text
<worker.binary> --served-model-manifest <manifest-path>
```

The manifest must independently pass the common served-model validator and
must bind:

- `ullm.served_model.v2`;
- public ID `ullm-qwen3-14b-sq8`;
- upstream `Qwen/Qwen3-14B-FP8`;
- revision `9a283b4a5efbc09ce247e0ae5b02b744739e525a`;
- format `SQ8_0` / implementation `qwen3_sq8_rdna4_v1`;
- vocabulary `151936`, EOS `[151645,151643]`, completion maximum `512`,
  and sampling capability `top_k=20`, temperature/top-p enabled;
- `ullm.worker.v2`, `gfx1201`, and `rdna4_w8a8_block_ck`;
- the ten frozen HIP guards;
- canonical artifact manifest/content and package manifest identities from
  v0.2; and
- the exact Qwen3 reasoning object from
  `served-model-manifest-v0.2.md`.

The manifest-declared worker file is rehashed before launch. Artifact,
package, tokenizer, promotion receipt, and worker identities are validated
through the common manifest loader rather than accepted from separate legacy
`--artifact` or `--package` arguments.

The independent validator receives expected Git commit, worker binary SHA-256,
and manifest SHA-256 separately. Raw.v3 validation fails if any is absent or
differs. Supplying a manifest hash while validating raw.v1/raw.v2 is an error.

## 3. Frozen request schedule

The v0.2 schedule and counts are unchanged:

- 2 cancellation warmups and 10 measured cancellations, each followed by
  normal recovery;
- 10 resource warmups;
- 100 resource-measured requests, cancelling every fifth block at offset four;
- 34 total cancellations;
- 134 releases;
- 169 commands including cancels and shutdown;
- baseline plus 100 five-sample resource points; and
- the same cancellation, slope, delta, request, progress, and shutdown gates.

Every v3 generate command adds the required exact summary and raw field:

```json
{
  "reasoning": {
    "enabled": false,
    "budget_tokens": null,
    "dialect_id": "qwen3-thinking-v1",
    "end_token_ids": [151668],
    "forced_end_token_ids": [151668],
    "reserved_answer_tokens": 1
  }
}
```

The rest of each generate request remains the v0.2 greedy measurement request.
Using disabled reasoning deliberately preserves the historical latency/resource
workload while proving that the candidate runs the explicit v2 protocol. The
enabled state-machine cases are mandatory CPU conformance prerequisites and
fresh OpenWebUI campaign cases, not silently mixed into this frozen resource
series.

Cancel and shutdown keep their historical exact field sets with the v2
discriminator.

## 4. Raw header

The raw.v3 header contains every exact raw.v2 top-level field plus one
`served_model` object. It has exactly:

```text
manifest_path
manifest_sha256
schema_version
model_id
model_revision
format_id
worker_protocol
worker_arguments
reasoning
```

`worker_arguments` is exactly
`["--served-model-manifest", "<validated-manifest-path>"]`.
`reasoning` is the complete manifest reasoning object, not a caller summary.
The manifest path and worker executable are regular non-symlink files and are
rehashed by the validator against independent expected values.

Every other header, environment, device, schedule, and threshold field is
byte-semantic raw.v2. KFD snapshots use the raw.v2 double-collect structure.

## 5. Commands and worker events

Raw command strings and stdout strings remain authoritative and are hashed,
strictly reparsed, and JSON-type-sensitively compared with their summaries.

Every raw.v3 worker event has `schema_version=ullm.worker.v2`. A v1 event in a
v3 stream, a v2 event in a legacy stream, a reasoning omission, or any mixed
raw discriminator is rejected.

Every v3 `released` event contains both `reasoning_tokens` and
`forced_end_tokens`. Because the frozen acceptance requests disable reasoning,
both must be integer zero for normal and cancelled releases. Their omission,
nonzero value, boolean substitution, or sum above completion usage is invalid.
Cancelled releases forbid timings. Non-cancelled v3 releases require the exact
timing object; raw.v1 forbids it and raw.v2 retains its already-published
historical optional-timing branch.

## 6. Independent result

The raw.v3 validator reconstructs every v0.2 gate and emits the v0.2 result
sections plus:

```json
{
  "schema_version": "ullm.sq8.worker_acceptance.validation.v3",
  "served_model": {
    "manifest_sha256": "<independently supplied digest>",
    "schema_version": "ullm.served_model.v2",
    "worker_protocol": "ullm.worker.v2",
    "reasoning_dialect_id": "qwen3-thinking-v1"
  }
}
```

Raw.v1 results retain `validation.v1` and omit KFD snapshot counters introduced
by raw.v2. Raw.v2 results retain `validation.v2`, its exact KFD counters, and no
served-model object. A validator implementation must not normalize one legacy
branch into another.

## 7. Failure and production boundary

All v0.2 fail-closed and fresh-run rules remain. A producer failure leaves no
successful raw publication. A validation failure produces no admissible
acceptance result.

The physical R9700 v3 run is a required fresh candidate-bound GPU gate and is
not performed by CPU unit tests. This specification and its tooling do not
authorize service control, `active.json` mutation, a production campaign, or
final activation.
