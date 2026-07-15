# Qwen3.5 AQ4 QKV/Z SQ8 overlay tools

## Scope

- Added a create-new wrapper around `tools/build-sq-fp8-w8a16-artifact.py`.
- The wrapper derives the 24 linear-attention layers from the pinned Qwen3.5-9B config and admits exactly the 48 `in_proj_qkv` / `in_proj_z` weights.
- The payload stays in the external product directory. No FP8 payload is added to Git.
- Added a layer-0 CPU oracle comparing direct BF16 matvecs with SQ8 overlay outputs and existing production AQ4 captures.

## Identity contract

- Binding schema: `ullm.qwen35_aq4_sq8_qkv_z_overlay.v1`.
- Overlay format: `SQ8_0`, with `row_block` scales, 256 columns per block, and f32 scale payloads.
- Binding includes the source config/index SHA-256, AQ4 package manifest SHA-256, SQ manifest SHA-256, content SHA-256, tensor-set SHA-256, and all 48 exact tensor names.
- The content and tensor-set digest domains match the Rust runtime admission implementation.
- A combined `ullm.aq4_resident_promotion.v1` receipt inherits the existing pinned AQ4 evidence identity and adds `overlay.content_sha256`.
- Artifact, binding, promotion receipt, summary, and oracle report are all create-new outputs.

## Memory policy

- The legacy builder encodes source weights in bounded row chunks; the production command uses `row_chunk=256` and one BLAS/OpenMP thread.
- The oracle uses `row_chunk=128`, one PyTorch thread, and retains only one output family at a time. It records `ru_maxrss_kib` in its report.

## Validation

- `pytest -q tests/test_qwen35_aq4_sq8_overlay_tools.py`: 5 passed.
- `python3 -m py_compile tools/build-qwen35-aq4-sq8-overlay.py tools/run-qwen35-aq4-sq8-overlay-cpu-oracle.py`: passed.
- `git diff --check` for the two tools and their test: passed.

## External generation status

The external artifact and CPU oracle were not published in this commit. Their intended create-new paths are:

- artifact: `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/artifacts/sq8-linear-qkv-z-rowblock256-v0.1`
- promotion receipt: `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/promotion-sq8-linear-qkv-z-overlay-v0.1.json`

An accidentally started generation was interrupted before manifest/binding publication, and its partial directory was removed. Both intended create-new paths were rechecked as absent.
