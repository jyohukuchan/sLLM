# SQ8_0 Canonical Artifact v0.2

Date: 2026-07-10

## Result

Qwen3-14B-FP8 was imported without F32 conversion or requantization. The importer copied raw
`F8_E4M3` weight regions and their BF16 128x128 `weight_scale_inv` regions directly from the
safetensors shards while computing SHA-256 in bounded chunks.

Full artifact coverage:

| field | value |
| --- | ---: |
| source tensors | 723 |
| FP8 weight/scale pairs | 280/280 |
| selected pairs | 280 |
| unpaired tensors | 0 |
| passthrough tensors | 163 |
| FP8 weight bytes | 13,212,057,600 |
| BF16 scale bytes | 1,612,800 |

- artifact schema: `sq-fp8-artifact-v0.2`
- manifest SHA-256: `23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2`
- canonical content SHA-256: `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`
- artifact path: `/tmp/ullm-qwen3-14b-fp8-sq8-canonical-full-v0.2`

The artifact directory is temporary and must be regenerated when absent.

## Rebuild

```bash
python3 tools/build-sq8-canonical-artifact.py \
  --source-model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8 \
  --output-artifact /tmp/ullm-qwen3-14b-fp8-sq8-canonical-full-v0.2 \
  --copy-chunk-bytes 67108864 \
  --overwrite
```

## Verification

- synthetic byte-exact, edge-block, deterministic, malformed pair, tamper, and atomic-output tests;
- real checkpoint config/index identity;
- real inventory `280` complete pairs and `163` passthrough tensors;
- layer0 q projection raw weight/scale SHA-256 and five-block reconstruction golden;
- Rust v0.2 typed reader with canonical content hash, payload checksum, coverage/storage/path checks,
  and row/block reconstruction;
- Python-generated layer0 artifact read and reconstructed by the Rust reader.

This proves P1 canonical storage and reconstruction. It does not claim a GPU execution or
performance result; block-2D runtime execution begins in P2.
