# Runtime cached prefix FP8 flash2 tiled online softmax v1

Date: 2026-07-08

Scope:

- Device: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`.
- Workload: cached prefix prefill with `q_heads=16`, `kv_heads=4`, `head_dim=256`, `value_dim=256`.
- KV cache dtype: `fp8_e4m3`, per-tensor K/V scales.
- New executor: `cached_prefix_flash2`.
- Baseline executor: `cached_prefix_chunked`.
- Kernel requirement envs:
  - baseline: `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_KERNEL=1`
  - flash2: `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL=1`

## Implementation shape

- `ullm_runtime_cached_prefix_attn_fp8_e4m3_flash2` was added as a separate C ABI entry point.
- Rust FFI exposes `cached_prefix_attn_fp8_e4m3_flash2`.
- `runtime-cached-prefix-attn-smoke` now accepts `cached_prefix_flash2`.
- The HIPRTC kernel maps one block to one `(new token, q head)` pair.
- K/V source tokens are processed in 64-token tiles.
- Scores are stored in shared memory per tile, then online softmax updates the running max, denominator, and weighted value without materializing the full attention matrix.

This is FlashAttention2-style rather than a direct FlashAttention2 port. It does not yet use WMMA/MFMA for QK/V matmul, and currently targets the FP8 cached-prefix path with `value_dim <= 256`.

## Results: M=16

Command shape:

```bash
tools/run-runtime-cached-prefix-sweep.py \
  --binary target/release/ullm-engine \
  --device-index 2 \
  --cached-prefix-tokens 4096,16384,65536 \
  --new-tokens 16 \
  --executors cached_prefix_chunked,cached_prefix_flash2 \
  --measured-repeats 3 \
  --long-measured-repeats 3 \
  --output-jsonl /tmp/ullm-flash2-sweep.jsonl \
  --summary-md /tmp/ullm-flash2-sweep.md
```

| L | executor | repeats | mean ms | new tok/s | pair/s mean | sampled diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | cached_prefix_chunked | 3 | 4.259759 | 3756.080743 | 15416833.407959 | 0 |
| 4096 | cached_prefix_flash2 | 3 | 3.698491 | 4326.088667 | 17756430.933589 | 0.000001974 |
| 16384 | cached_prefix_chunked | 3 | 15.575064 | 1027.283076 | 16839737.826230 | 0 |
| 16384 | cached_prefix_flash2 | 3 | 13.066830 | 1224.474490 | 20072198.077116 | 0.000011012 |
| 65536 | cached_prefix_chunked | 3 | 115.500919 | 138.527036 | 9079685.305131 | 0 |
| 65536 | cached_prefix_flash2 | 3 | 76.947130 | 207.934982 | 13628994.408797 | 0.000031359 |

## Results: M=128

Command shape:

```bash
tools/run-runtime-cached-prefix-sweep.py \
  --binary target/release/ullm-engine \
  --device-index 2 \
  --cached-prefix-tokens 4096,16384,65536 \
  --new-tokens 128 \
  --executors cached_prefix_chunked,cached_prefix_flash2 \
  --measured-repeats 1 \
  --long-measured-repeats 1 \
  --output-jsonl /tmp/ullm-flash2-sweep-m128.jsonl \
  --summary-md /tmp/ullm-flash2-sweep-m128.md
```

| L | executor | repeats | mean ms | new tok/s | pair/s mean | sampled diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | cached_prefix_chunked | 1 | 31.585184 | 4052.532985 | 16860563.484449 | 0 |
| 4096 | cached_prefix_flash2 | 1 | 27.188505 | 4707.871948 | 19587101.240028 | 0.000002004 |
| 16384 | cached_prefix_chunked | 1 | 124.997071 | 1024.023995 | 16843658.680610 | 0 |
| 16384 | cached_prefix_flash2 | 1 | 100.886273 | 1268.755364 | 20869122.601050 | 0.000011086 |
| 65536 | cached_prefix_chunked | 1 | 1263.489173 | 101.306764 | 6645774.399525 | 0 |
| 65536 | cached_prefix_flash2 | 1 | 994.903424 | 128.655704 | 8439878.482115 | 0.000031389 |

## Ratios

| L | M | flash2 / baseline tok/s |
| ---: | ---: | ---: |
| 4096 | 16 | 1.152x |
| 16384 | 16 | 1.192x |
| 65536 | 16 | 1.501x |
| 4096 | 128 | 1.162x |
| 16384 | 128 | 1.239x |
| 65536 | 128 | 1.270x |

## Interpretation

- The first FlashAttention2-style FP8 cached-prefix kernel improves the old FP8 cached-prefix path across the tested representative grid.
- The largest gain appears at long prefix and small chunk (`L=65536,M=16`), where the old path had high variance and poor long-prefix behavior.
- The sampled diff is non-zero because the tile online-softmax update changes floating-point accumulation order. The observed range is small enough for the current sampled guard.
- This still leaves major optimization headroom:
  - The kernel still computes one query head per block and does not use WMMA/MFMA.
  - K/V FP8 values are still decoded as scalar values.
  - There is no batch or tensor-parallel scheduling yet.
  - Cold prefill causal attention has not been converted to this tiled path yet.

## Verification

- `cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_flash2 -- --test-threads=1`
- `cargo check -p ullm-engine`
- `cargo fmt --all --check`
- `cargo build -p ullm-engine --release`
- R9700 smokes with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_FP8_E4M3_FLASH2_KERNEL=1`
- R9700 sweep comparison for `M=16` and `M=128`
