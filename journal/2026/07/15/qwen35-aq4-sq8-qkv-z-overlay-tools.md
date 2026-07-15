# Qwen3.5 AQ4 QKV/Z SQ8 overlay tools

## Scope

- Added a create-new wrapper around `tools/build-sq-fp8-w8a16-artifact.py`.
- The wrapper derives the 24 linear-attention layers from the pinned Qwen3.5-9B config and admits exactly the 48 `in_proj_qkv` / `in_proj_z` weights.
- The payload stays in the external product directory. No FP8 payload is added to Git.
- Added a layer-0 CPU oracle comparing direct BF16 matvecs with SQ8 overlay outputs and existing production AQ4 captures.

## Identity contract

- Binding schema: `ullm.qwen35_aq4_sq8_qkv_z_overlay.v2`.
- Overlay format: `SQ8_0`, with `row_block` scales, 256 columns per block, and f32 scale payloads.
- Binding includes the source config/index SHA-256, the two actually read safetensors shard SHA-256/size identities, all 48 logical BF16 tensor SHA-256/dtype/shape identities, their payload/scale mappings, AQ4 package manifest SHA-256, SQ manifest SHA-256, content SHA-256, and tensor-set SHA-256.
- The content and tensor-set digest domains match the Rust runtime admission implementation.
- A combined `ullm.aq4_resident_promotion.v1` receipt inherits the existing pinned AQ4 evidence identity and binds content, binding, tensor-set, and immutable artifact inventory identities.
- Publication builds and validates a sibling temporary artifact, seals directories to `0555` and files to `0444`, and uses Linux atomic rename exchange for a same-content hardening replacement.

## Memory policy

- The legacy builder encodes source weights in bounded row chunks; the production command uses `row_chunk=256` and one BLAS/OpenMP thread.
- The oracle uses `row_chunk=128`, one PyTorch thread, and retains only one output family at a time. It records `ru_maxrss_kib` in its report.

## Validation

- `pytest -q tests/test_qwen35_aq4_sq8_overlay_tools.py`: 10 passed.
- `python3 -m py_compile tools/build-qwen35-aq4-sq8-overlay.py tools/run-qwen35-aq4-sq8-overlay-cpu-oracle.py`: passed.
- `git diff --check` for the two tools and their test: passed.

## External generation result

- The same-content hardened external artifact was atomically exchanged at `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/artifacts/sq8-linear-qkv-z-rowblock256-v0.1`.
- It contains 48 SQ8 tensors in 98 regular files totaling 1,227,177,520 bytes and 3 directories. All files are `0444`, all directories are `0555`, every file has `nlink=1`, ownership is `uid=1000/gid=1000`, and no symlink is present. No FP8 payload is stored in Git.
- Binding SHA-256: `3153546fd419a66cfda521cd4e3558165769e2d2f3786fda7e4092a3f337caea`.
- Content SHA-256: `b7d5ef6c3e4ebceae9c9dee8e3094c5a29a8d444bdb2ca1170a559c9539a13fc`.
- Tensor-set SHA-256: `6fbf047fe19b27a6c9075f06a76fa4bf376ba08ff9d39c84da43461fdf606846`.
- The combined promotion receipt was atomically replaced at `/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/promotion-sq8-linear-qkv-z-overlay-v0.1.json`, SHA-256 `d100c8fcb73bae98baaa2fbb73f738d3f535059844769d44c89939f1e505a633`.

## CPU oracle result

- Status: `valid`; peak RSS: 880,336 KiB. Every reported numerical metric is unchanged from the pre-hardening oracle.
- QKV SQ8 versus direct BF16: relative L2 `0.00857736`, cosine `0.99996332`; existing AQ4 relative L2 `0.02451886`.
- Z SQ8 versus direct BF16: relative L2 `0.01130386`, cosine `0.99993643`; existing AQ4 relative L2 `0.03092805`.
- The existing BF16 capture agrees with the new direct calculation at relative L2 below `1.0e-7` for both tensors.
- Git evidence: `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3/sq8-linear-qkv-z-overlay-v0.1/`.

## Served-model materialization boundary

- The dedicated profile binds the overlay receipt and content identity, and its worker admission contract is covered by Rust tests.
- `tools/generate-served-model.py` correctly refuses to materialize the profile without overlay-specific GPU promotion evidence. The inherited AQ4 receipt predates the required GPU-exclusivity preflight; the unchanged default AQ4 profile is refused for the same reason.
- No GPU promotion run, service mutation, or worker-binary publication was performed in this work. A materialized served-model manifest must wait for a separately authorized overlay GPU promotion run.
