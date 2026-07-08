# Runtime causal attention F32 flash2 tiled online softmax v1

Date: 2026-07-08

Scope:

- Device: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`.
- Workload: cold prefill causal attention with real batch mode.
- Dtype: F32 Q/K/V/output.
- New executor: `causal_attn_batch_f32_flash2`.
- Baseline executor: `causal_attn_batch_f32`.
- Kernel requirement envs:
  - baseline: `ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_KERNEL=1`
  - flash2: `ULLM_REQUIRE_HIP_CAUSAL_ATTN_BATCH_F32_FLASH2_KERNEL=1`

## Implementation shape

- `ullm_runtime_causal_attn_f32_flash2` and `ullm_runtime_causal_attn_batch_f32_flash2` were added as separate C ABI entry points.
- Rust FFI exposes `causal_attn_f32_flash2` and `causal_attn_batch_f32_flash2`.
- `runtime-causal-attn-batch-smoke` now accepts `EXECUTOR=causal_attn_batch_f32|default|flash2|causal_attn_batch_f32_flash2`.
- The HIPRTC kernels map one block to one `(batch, timestep, q head)` row.
- Source tokens are processed in 64-token tiles.
- Scores are stored in shared memory per tile, then online softmax updates the running max, denominator, and weighted value without materializing the full attention matrix.

This is FlashAttention2-style rather than a direct FlashAttention2 port. It does not yet use WMMA/MFMA and does not yet group multiple query rows per block.

## Results

| batch | seq | head_dim | value_dim | executor | repeats | mean ms | input tok/s | pair/s mean | sampled diff |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 64 | 64 | causal_attn_batch_f32 | 5 | 0.316805 | 404033.517105 | 13131089.305927 | 0.000000002 |
| 2 | 64 | 64 | 64 | causal_attn_batch_f32_flash2 | 5 | 0.266783 | 479791.050997 | 15593209.157412 | 0.000000011 |
| 4 | 128 | 64 | 64 | causal_attn_batch_f32 | 3 | 2.087880 | 245224.860500 | 15817003.502277 | 0.000000002 |
| 4 | 128 | 64 | 64 | causal_attn_batch_f32_flash2 | 3 | 1.694279 | 302193.499849 | 19491480.740280 | 0.000000013 |
| 1 | 256 | 64 | 64 | causal_attn_batch_f32 | 1 | 2.095043 | 122193.196035 | 15701825.690451 | 0.000000002 |
| 1 | 256 | 64 | 64 | causal_attn_batch_f32_flash2 | 1 | 1.695679 | 150971.970520 | 19399898.211867 | 0.000000050 |
| 1 | 128 | 256 | 256 | causal_attn_batch_f32 | 1 | 0.606176 | 211159.795175 | 13619806.788787 | 0 |
| 1 | 128 | 256 | 256 | causal_attn_batch_f32_flash2 | 1 | 0.510066 | 250947.916544 | 16186140.617097 | 0 |

## Ratios

| batch | seq | head_dim | value_dim | flash2 / baseline input tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 64 | 64 | 1.188x |
| 4 | 128 | 64 | 64 | 1.232x |
| 1 | 256 | 64 | 64 | 1.236x |
| 1 | 128 | 256 | 256 | 1.188x |

## Interpretation

- The first cold-prefill F32 causal flash2 kernel improves the existing causal batch path across the initial safe grid.
- The current gain is modest but consistent because the old kernel was already online-softmax for `value_dim <= 256`; the new kernel mainly improves tile-local score handling rather than changing the launch granularity.
- Larger improvements likely require grouping multiple query rows per block, reusing K/V tiles across rows, and then moving QK/V accumulation toward RDNA4 MFMA/WMMA-friendly layouts.
- The existing `runtime-causal-attn-batch-smoke` 8 MiB allocation guard limits initial cold-prefill grid size. That is intentional for OOM safety, but a dedicated benchmark harness will be needed for larger prefill contexts.

## Verification

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-runtime-sys causal_attn_f32_flash2 -- --test-threads=1`
- `cargo test -p ullm-runtime-sys causal_attn_batch_f32_flash2 -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 baseline and flash2 smokes with HIP-kernel-required envs
- Backward-compatibility smoke without the optional executor argument
