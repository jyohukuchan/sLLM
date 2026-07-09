# SQ FP8 Artifact v0.1

## Purpose

This specification defines the first FP8 SQ8_0 artifact produced by
`tools/build-sq-fp8-w8a16-artifact.py`.

The public format ID is `SQ8_0`. The previous `sq-fp8-w8a16-r9700-v0` name is retained as an
implementation lineage / legacy alias, not as the public format name.

This schema is retained for legacy BF16/F16 requantization and connection diagnostics. The builder
must reject selected `F8_E4M3` source tensors because v0.1 cannot preserve their source 2D block
scales. Existing FP8 checkpoints must use the source-correct canonical
[`sq-fp8-artifact-v0.2`](sq-fp8-artifact-v0.2.md) importer.

## Layout

```text
ARTIFACT/
  sq_manifest.json
  fp8/
    TENSOR_NAME.fp8_e4m3
  scales/
    TENSOR_NAME.scale_f32
```

`sq_manifest.json` uses `schema_version = "sq-fp8-artifact-v0.1"`.

## Candidate Semantics

```json
{
  "candidate": {
    "id": "SQ8_0",
    "format_id": "SQ8_0",
    "implementation_id": "sq-fp8-w8a16-r9700-v0",
    "weight_payload_dtype": "fp8_e4m3",
    "activation_dtype": "bf16_or_f32",
    "scale_granularity": "row",
    "default_scale_granularity": "row",
    "scale_dtype": "f32"
  }
}
```

The initial SQ8_0 runtime target is language-model 2D weight tensors:

- embedding
- lm_head
- self-attention q/k/v/o projection
- linear-attention in/out projection
- MLP gate/up/down projection

Visual tower, MTP tensors, RDNA2 fallback, and tensor parallel layout are deferred.

## FP8 Tensor Entries

Each `fp8_tensors[]` entry must include:

- `name`
- `family`
- `source_dtype`
- `shape`
- `elements`
- `source_file`
- `payload_dtype`
- `payload_file`
- `payload_bytes`
- `scale_granularity`
- `scale_dtype`
- `scale_file`
- `scale_elements`
- `scale_bytes`

When payload files are generated, entries should also include:

- `payload_sha256`
- `scale_sha256`

For row scale, `scale_elements == shape[0]`. For tensor scale, `scale_elements == 1`.

For row-block scale:

- `scale_granularity == "row_block"`
- `scale_block_cols` is required and must be greater than zero
- `scale_elements == shape[0] * ceil(shape[1] / scale_block_cols)`
- scale values are stored row-major by `(row, column_block)`

For mixed per-tensor scale policies:

- candidate-level `scale_granularity == "mixed"`
- candidate-level `scale_layout == "per_tensor"`
- each `fp8_tensors[]` entry is authoritative for `scale_granularity`
- entries with `scale_granularity == "row_block"` carry their own `scale_block_cols`
- `scale_override_id` may be present when a policy override selected that tensor's scale layout

## Passthrough Entries

Each `passthrough_tensors[]` entry records tensors intentionally left outside the FP8 payload:

- `name`
- `dtype`
- `shape`
- `elements`
- `source_file`
- `reason`

Common reasons:

- `not_fp8_target_family`
- `not_2d_weight`
- `excluded_by_regex`
- `not_selected`

Passthrough entries are metadata in v0.1. The runtime consumer may still load them from the source
package or source model until the final SQ package format is defined.

## Storage Fields

`storage` must include:

- `fp8_tensor_count`
- `passthrough_tensor_count`
- `fp8_payload_bytes`
- `fp8_scale_bytes`
- `passthrough_source_bytes_estimate`
- `compact_resident_bytes_estimate`
- `materialized_working_set_bytes_estimate`

`compact_resident_bytes_estimate` includes FP8 payload bytes, FP8 scale bytes, and estimated
passthrough source bytes. This is intentionally conservative for the first candidate.

`materialized_working_set_bytes_estimate` is the largest selected FP8 tensor materialized as F32.
Later runtime paths may reduce this with native FP8 matmul or smaller materialization windows.

## Policy Input

The artifact builder accepts:

```text
tools/build-sq-fp8-w8a16-artifact.py --policy-json POLICY_JSON
```

For `schema_version = "sq-fp8-policy-v0.1"`, the policy supplies default values for:

- `candidate_id`
- selected FP8 tensor `include_regex`
- scale granularity
- row-block scale width
- optional scale overrides under `scale.overrides[]`

Explicit CLI values still override policy defaults where applicable. Generated manifests include a
`policy` block with policy ID, source policy path, selected FP8 rule, fallback policy, and prompt
bundle result.

`scale.overrides[]` entries use `include_regex` to match selected tensor names. The override may set
`granularity`, `block_cols`, and an optional `id`; later matching overrides win. This is used by T2
model-loop guards to test mixed layouts such as `k_proj` row-block16 with `up_proj` row-block32 in
one SQ artifact.

## Runtime Path Status

The v0.1 artifact is the payload and metadata boundary for T2. Runtime execution support is staged:

1. Build and validate `sq_manifest.json`. Done for the first smoke path and current policy artifact.
2. Generate FP8 payload artifacts and verify file sizes/checksums. Done for the first small smoke path and for the current `kup6_gate5_down5` policy artifact.
3. Add a runtime loader that can resolve FP8 payload plus scale files for selected tensors. Partially done for row and row-block scale.
4. Connect the loader to short prompt guard. Partially done for a one-tensor `q_proj` overlay.

The first runtime loader entrypoint is:

```text
ullm-engine sq-fp8-materialize-smoke ARTIFACT_DIR [DEVICE_INDEX] [TENSOR_SELECTOR] [ROW_COUNT] [START_ROW]
```

It validates `sq_manifest.json`, selects an FP8 tensor by index, exact name, or unique substring,
decodes selected rows from FP8 E4M3 plus F32 scales into F32, copies the materialized rows into a
runtime buffer, then reads the buffer back to verify the runtime transfer.

This is not yet the short prompt guard. It proves the runtime can consume the FP8 artifact boundary
without expanding the full model at once.

The first package-path overlay guard is:

```text
ullm-engine sq-fp8-token-ids-logits-smoke PACKAGE_DIR ARTIFACT_DIR [DEVICE_INDEX] [CHUNK_BYTES] [LAYERS_CSV|all] [TOKEN_IDS_CSV] [TOP_K] [LM_HEAD_CHUNK_ROWS] [ROTARY_DIM] [ROPE_BASE] [POSITION_OFFSET]
```

It loads exact-name SQ FP8 tensors from the artifact and falls back to the existing AQ4 package
tensors for names not present in the artifact. The first R9700 guard covered only
`model.language_model.layers.3.self_attn.q_proj.weight`. The second guard covered all layer 3
self-attention and MLP projection tensors. A later layers `3,7` guard found top1 ranking drift while
keeping the AQ4 top1 inside SQ top8. A family split guard then identified `q/v/down` as the first
risky row-scale FP8 families and `k/o/gate/up` as the first strict-top1-safe subset in that small
guard. A later safe-subset scaling guard found that `k/o/gate/up` is not globally safe under strict
top1: it passes layers `3,7`, but layers `3,7,11,15` fail one of three short prompts. These are
runtime boundary and short-quality guards, not full SQ candidate quality results.

The next runtime boundary extension is `row_block` scale. In the first layers `3,7` guard,
row-block32 recovers `q`, row-block64 recovers `down`, but `v` still fails strict top1. A mixed
candidate that keeps `v` as fallback and stores `q/k/o/gate/up/down` as row-block32 FP8 passes
`3 / 3` short prompts for layers `3,7`.

The same mixed candidate was expanded further in
`benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-layer-scaling-guard-v0.1.md`.
It passes strict top1 for layers `3,7,11,15` across `3 / 3` short prompts and passes layers
`3,7,11,15,19` for the len4 case. It fails strict top1 for layers `3,7,11,15,19,23` and for all
self-attention probe layers `3,7,11,15,19,23,27,31`. Layer `23` alone can be recovered with `q/v`
fallback, but `q/v` fallback across all six tested layers still leaves cumulative drift. Therefore
the row-block32 mixed candidate is a partial T2 quality policy, not a final SQ artifact policy.

T2 promotion rule v0.1 is strict top1 match for every evaluated short-guard case. Top-k overlap,
AQ4 top1 rank inside SQ top-k, and SQ top1 minus AQ4 top1 logit gap are diagnostic-only fields
until a text-level guard is implemented and explicitly accepted. The current mixed candidate
acceptance result is saved in
`benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-acceptance-v0.1.md` and is not accepted for
T2 promotion.

A later six-layer boundary guard found that only `k/up` row-block32 remains strict-top1-safe across
the tested `3 / 3` short prompts for layers `3,7,11,15,19,23`. `o/gate/down` fail strict top1
individually with row-block32 and still fail with row-block16. The `k/up` subset is a regression
guard and not a final SQ policy because it leaves `q/v/o/gate/down` as fallback.

A later per-layer combination guard found that `o/gate/down` are strict-top1-safe over layers
`3,7,11,15,19`, and that `k/up` over all six layers plus any one or two of those five-layer
families still passes len4. Adding all three five-layer families fails len4. The next partial
prompt-bundle candidate is `kup6_gate5_down5`, but it is not a promoted SQ policy until the full
prompt bundle passes.

The `kup6_gate5_down5` prompt bundle later passed len4, case_a, and case_b strict top1. It stores
`k/up` for layers `3,7,11,15,19,23` and `gate/down` for layers `3,7,11,15,19` as FP8 row-block32,
while keeping `q/v/o` and layer `23` `gate/down` as fallback. This is the current six-layer
strict-top1 regression subset. It is still not a promoted full SQ policy because case_a top8 overlap
is only `2 / 8` and broader coverage is missing. The reproducible policy record is saved as
`benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json`.

The same policy has been materialized into a real payload artifact at
`/tmp/ullm-sq-fp8-kup6-gate5-down5-policy-v0.1-artifact`. The manifest contains a policy block,
`22` FP8 tensors, `753` passthrough tensors, and row-block32 F32 scales. R9700
`sq-fp8-materialize-smoke` verified `model.language_model.layers.3.self_attn.k_proj.weight` with
`roundtrip_max_abs_diff=0` and `verified=true`. The result record is saved as
`benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-artifact-v0.1.md`. This verifies the
policy-to-artifact-to-runtime boundary, not throughput or final SQ acceptance. Because the payload
artifact is under `/tmp`, it must be regenerated from the policy JSON when absent.

For repeatable guard runs, use:

```text
tools/run-sq-fp8-overlay-logits-guard.py
```

The script generates case-specific SQ FP8 artifacts, runs AQ4 and SQ logits smokes, and writes a
top-k comparison JSON.
