# SQ8_0 Canonical FP8 Artifact v0.2

## Purpose

This specification defines a source-correct canonical artifact for checkpoints that already store
FP8 E4M3 weights and block scales. It replaces neither the source checkpoint nor the legacy
`sq-fp8-artifact-v0.1` connection-diagnostic format.

The canonical artifact preserves model semantics. GPU-specific transpose, padding, swizzle, scale
expansion, and other prepack layouts are derived caches and must not be stored as the canonical
payload.

## Compatibility

- schema version: `sq-fp8-artifact-v0.2`
- artifact kind: `canonical`
- public format ID: `SQ8_0`
- import mode: `fp8_checkpoint`
- payload encoding: `raw_safetensors_payload`

Readers must dispatch by schema version. They must not reinterpret a v0.1 artifact as v0.2 or
promote it to source-correct status. A source checkpoint is required to build v0.2 because a v0.1
artifact does not contain the original block scales.

## Directory Layout

```text
ARTIFACT/
  sq_manifest.json
  weights/
    TENSOR_NAME_SHA256.f8_e4m3
  scales/
    TENSOR_NAME_SHA256.bf16
```

The file ID is the lowercase SHA-256 of the UTF-8 weight tensor name. This avoids collisions from
sanitized names.

## Manifest

```json
{
  "schema_version": "sq-fp8-artifact-v0.2",
  "artifact_kind": "canonical",
  "format_id": "SQ8_0",
  "source": {
    "model_name": "Qwen3-14B-FP8",
    "config_file": "config.json",
    "config_sha256": "...",
    "index_file": "model.safetensors.index.json",
    "index_sha256": "...",
    "quantization": {
      "quant_method": "fp8",
      "format": "e4m3",
      "activation_scheme": "dynamic",
      "weight_block_shape": [128, 128]
    }
  },
  "import": {
    "mode": "fp8_checkpoint",
    "encoding": "raw_safetensors_payload"
  },
  "coverage": {
    "scope": "full_model",
    "source_tensor_count": 723,
    "source_fp8_weight_count": 280,
    "source_scale_count": 280,
    "paired_tensor_count": 280,
    "selected_pair_count": 280,
    "unpaired_tensor_count": 0,
    "passthrough_tensor_count": 163
  },
  "storage": {
    "weight_payload_bytes": 13212057600,
    "scale_payload_bytes": 1612800,
    "total_payload_bytes": 13213670400
  },
  "quantized_tensors": [],
  "passthrough_tensors": [],
  "integrity": {
    "content_sha256": "..."
  }
}
```

`coverage.scope` is `full_model` when all source FP8 weight/scale pairs are present and
`selected_tensors` for a bounded component artifact. Pairing and orphan validation always cover
the complete source inventory even for a selected artifact.

v0.2 fixes the source block shape to `[128, 128]`. A checkpoint with another positive block shape
requires a later schema version rather than a reader-specific reinterpretation.

## Quantized Tensor Pair

Each `quantized_tensors[]` entry contains one weight and its source scale:

```json
{
  "name": "model.layers.0.self_attn.q_proj.weight",
  "family": "attn_q",
  "shape": [5120, 5120],
  "elements": 26214400,
  "weight": {
    "dtype": "F8_E4M3",
    "encoding": "raw_safetensors_payload",
    "file": "weights/ID.f8_e4m3",
    "bytes": 26214400,
    "sha256": "...",
    "source_file": "model-00001-of-00004.safetensors"
  },
  "scale": {
    "name": "model.layers.0.self_attn.q_proj.weight_scale_inv",
    "dtype": "BF16",
    "encoding": "raw_safetensors_payload",
    "file": "scales/ID.bf16",
    "shape": [40, 40],
    "elements": 1600,
    "bytes": 3200,
    "sha256": "...",
    "source_file": "model-00001-of-00004.safetensors",
    "layout": "block_2d",
    "block_shape": [128, 128],
    "order": "row_major",
    "semantic": "dequant_multiplier"
  }
}
```

The weight logical shape is `[out_features, in_features]`. The scale logical shape is
`[out_block, in_block]` and must equal:

```text
[ceil(out_features / block_rows), ceil(in_features / block_cols)]
```

## Reconstruction

For a weight `Wq`, scale `S`, and block shape `[BN, BK]`:

```text
W_f32[n, k] =
  decode_e4m3fn(Wq[n, k]) *
  bf16_to_f32(S[floor(n / BN), floor(k / BK)])
```

`weight_scale_inv` is the checkpoint tensor name. Its stored value is the dequantization
multiplier and is multiplied during reconstruction. Neither weight nor scale is transposed during
canonical import. Partial edge blocks use the same scale lookup and are clipped by the logical
weight shape.

## Byte Preservation

The importer copies safetensors data regions directly from the offsets in the file header.

- FP8 payload bytes must be identical to the source region.
- BF16 scale bytes must be identical to the source region.
- F32 conversion and requantization are forbidden in `fp8_checkpoint` import mode.
- Copying and hashing use bounded chunks and process one tensor pair at a time.
- Non-finite FP8 encodings and non-positive or non-finite scale values are rejected.

BF16-to-SQ8 quantization is a different operation and must use a separately identified builder
mode and implementation contract.

## Integrity

Each payload carries its byte length and lowercase SHA-256. The manifest `content_sha256` is the
SHA-256 of the manifest object with `integrity` removed, serialized as UTF-8 JSON with sorted object
keys, no whitespace separators, and unescaped Unicode:

```python
json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

The manifest file itself may be hashed separately by benchmark result metadata.

## Atomic Output

The importer writes to a temporary directory under the output parent, verifies all payloads and the
manifest, writes the manifest last, and then renames the directory. A first promotion uses Linux
`renameat2(RENAME_NOREPLACE)` so an output that appears during the build is never replaced.
Overwrite is permitted only when the existing output is itself a verified v0.2 canonical artifact.
The importer records that directory's device and inode identity before and after validation, uses
`renameat2(RENAME_EXCHANGE)` to atomically swap the new and previous directories, and verifies both
sides after the exchange. An identity conflict is rolled back without deleting the conflicting
entry. Owned temporary and previous-artifact directories are moved into a private cleanup
quarantine and re-identified before removal. Source and output paths must not be equal or contain
one another. A failed build must not expose a partially complete artifact or replace or delete an
arbitrary entry.

## Required Validation

The builder and reader reject:

- duplicate tensor names;
- safetensors data regions that overlap, contain gaps, or leave trailing bytes;
- JSON booleans in numeric shape, offset, byte-count, element-count, coverage, or storage fields;
- an index whose tensor keys do not exactly match shard contents, map a tensor to the wrong shard,
  omit a shard, or reference a path outside the model directory;
- missing or orphan scale tensors;
- non-2D weight or scale tensors;
- weight dtype other than `F8_E4M3`;
- scale dtype other than the source contract's `BF16`;
- scale shape or block shape mismatch;
- unsafe artifact-relative paths;
- byte length or SHA-256 mismatch;
- non-finite FP8 values;
- non-positive or non-finite scales;
- inconsistent coverage and storage totals.

## Execution Boundary

v0.2 defines storage and reconstruction, not an optimized execution profile. The source-correct
W8A16 reference and the dynamic W8A8 RDNA4 path are separate execution profiles. A legacy v0.1
row/row-block matvec must not consume v0.2 scale data without an explicit block-2D API.
