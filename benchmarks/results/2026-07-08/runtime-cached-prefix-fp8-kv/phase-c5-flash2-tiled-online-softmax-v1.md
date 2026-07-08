# Runtime cached prefix FP8 flash2 tiled online softmax v1

Date: 2026-07-08

Scope:

- Device: R9700/RDNA4, runtime device index `2`, backend `hip`, name `AMD Radeon Graphics`.
- Workload: cached prefix prefill with `q_heads=16`, `kv_heads=4`, `head_dim=256`, `value_dim=256`.
- Primary KV cache dtype: `fp8_e4m3`, per-tensor K/V scales.
- Isolation comparison: `f32` KV cache with the same cached-prefix flash2 executor.
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

This is FlashAttention2-style rather than a direct FlashAttention2 port. It does not yet use WMMA/MFMA for QK/V matmul. The first implementation now covers cached-prefix `fp8_e4m3` and `f32` KV cache paths with `value_dim <= 256`.

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

## Results: M=512

Command shape:

```bash
tools/run-runtime-cached-prefix-sweep.py \
  --binary target/release/ullm-engine \
  --device-index 2 \
  --cached-prefix-tokens 4096,16384,65536 \
  --new-tokens 512 \
  --executors cached_prefix_chunked,cached_prefix_flash2 \
  --measured-repeats 1 \
  --long-measured-repeats 1 \
  --output-jsonl /tmp/ullm-flash2-sweep-m512.jsonl \
  --summary-md /tmp/ullm-flash2-sweep-m512.md
```

| L | executor | repeats | mean ms | new tok/s | pair/s mean | sampled diff |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | cached_prefix_chunked | 1 | 128.438328 | 3986.348997 | 17350584.009471 | 0 |
| 4096 | cached_prefix_flash2 | 1 | 103.406210 | 4951.346732 | 21550736.653050 | 0.000002097 |
| 16384 | cached_prefix_chunked | 1 | 506.355296 | 1011.147714 | 16826003.534088 | 0 |
| 16384 | cached_prefix_flash2 | 1 | 410.238352 | 1248.054936 | 20768258.156419 | 0.000011355 |
| 65536 | cached_prefix_chunked | 1 | 4864.578827 | 105.250633 | 6924702.260560 | 0 |
| 65536 | cached_prefix_flash2 | 1 | 3575.130608 | 143.211551 | 9422245.980223 | 0.000031497 |

## FP8 ratios

| L | M | flash2 / baseline tok/s |
| ---: | ---: | ---: |
| 4096 | 16 | 1.152x |
| 16384 | 16 | 1.192x |
| 65536 | 16 | 1.501x |
| 4096 | 128 | 1.162x |
| 16384 | 128 | 1.239x |
| 65536 | 128 | 1.270x |
| 4096 | 512 | 1.242x |
| 16384 | 512 | 1.234x |
| 65536 | 512 | 1.361x |

## F32 isolation sweep

Command shape:

```bash
tools/run-runtime-cached-prefix-sweep.py \
  --binary target/release/ullm-engine \
  --device-index 2 \
  --kv-cache-dtype f32 \
  --executors cached_prefix_chunked,cached_prefix_flash2 \
  --cached-prefix-tokens 4096,16384,65536 \
  --new-tokens 16,512 \
  --measured-repeats 3 \
  --long-measured-repeats 1 \
  --max-estimated-attention-work 9000000 \
  --output-jsonl /tmp/ullm-f32-flash2-sweep.jsonl \
  --summary-md /tmp/ullm-f32-flash2-sweep.md
```

`L=65536,M=512` was intentionally skipped in this F32 isolation pass by the attention-work cap. The goal here was to isolate the tiled online-softmax executor effect without spending time on the slowest F32 cached-prefix workload.

| L | M | executor | repeats | mean ms | new tok/s | pair/s mean | sampled diff |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | cached_prefix_chunked | 3 | 4.186972 | 3821.377662 | 15684844.615221 | 0 |
| 4096 | 16 | cached_prefix_flash2 | 3 | 3.479678 | 4598.126168 | 18873008.855704 | 0 |
| 4096 | 512 | cached_prefix_chunked | 1 | 129.971796 | 3939.316188 | 17145873.709401 | 0 |
| 4096 | 512 | cached_prefix_flash2 | 1 | 100.957924 | 5071.419654 | 22073354.044007 | 0 |
| 16384 | 16 | cached_prefix_chunked | 1 | 19.162237 | 834.975582 | 13687337.235209 | 0 |
| 16384 | 16 | cached_prefix_flash2 | 1 | 14.499501 | 1103.486251 | 18088898.369675 | 0 |
| 16384 | 512 | cached_prefix_chunked | 1 | 672.380142 | 761.474006 | 12671308.189825 | 0 |
| 16384 | 512 | cached_prefix_flash2 | 1 | 450.947258 | 1135.387766 | 18893420.125863 | 0 |
| 65536 | 16 | cached_prefix_chunked | 1 | 78.707749 | 203.283669 | 13324126.446559 | 0 |
| 65536 | 16 | cached_prefix_flash2 | 1 | 57.776118 | 276.931032 | 18151306.046557 | 0.000020176 |

| L | M | F32 flash2 / F32 baseline tok/s |
| ---: | ---: | ---: |
| 4096 | 16 | 1.203x |
| 4096 | 512 | 1.287x |
| 16384 | 16 | 1.322x |
| 16384 | 512 | 1.491x |
| 65536 | 16 | 1.362x |

## Interpretation

- The first FlashAttention2-style FP8 cached-prefix kernel improves the old FP8 cached-prefix path across the tested representative grid, including `M=512`.
- The F32 isolation sweep also improves consistently, which means the gain is not only from FP8 KV bandwidth reduction; the tiled online-softmax executor structure itself is helping.
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
- R9700 F32 isolation sweep with `ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL=1`
