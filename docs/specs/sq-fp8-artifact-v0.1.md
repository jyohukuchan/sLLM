# SQ FP8 Artifact v0.1

## Purpose

This specification defines the first FP8 SQ candidate artifact produced by
`tools/build-sq-fp8-w8a16-artifact.py`.

The first candidate is `sq-fp8-w8a16-r9700-v0`. It is a runtime-evaluation artifact, not the final
SQ file format.

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
    "id": "sq-fp8-w8a16-r9700-v0",
    "weight_payload_dtype": "fp8_e4m3",
    "activation_dtype": "bf16_or_f32",
    "scale_granularity": "row",
    "scale_dtype": "f32"
  }
}
```

The initial target is language-model 2D weight tensors:

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

## Runtime Path Status

The v0.1 artifact is the payload and metadata boundary for T2. Runtime execution support is staged:

1. Build and validate `sq_manifest.json`.
2. Generate a small FP8 payload artifact and verify file sizes/checksums.
3. Add a runtime loader that can resolve FP8 payload plus scale files for selected tensors.
4. Connect the loader to short prompt guard.
