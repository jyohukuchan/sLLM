# Phase C13: RDNA4 FP8 rocWMMA Cached-Prefix Value Group 64 v1

## Summary

- Reworked the `cached_prefix_rocwmma_fp8` kernel so one block handles a 64-column value group.
- This reduces QK/online-softmax recomputation for `value_dim=256` from 16 value tiles to 4 value groups while keeping enough block-level parallelism.
- A full-value dynamic-shared accumulator variant was tried first, but it lost too much block parallelism and was not kept.

## Verification

```text
cargo fmt --all --check
cargo check -p ullm-engine
cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_rocwmma -- --test-threads=1
cargo build -p ullm-engine --release
git diff --check -- crates/ullm-engine/src/main.rs crates/ullm-runtime-sys/src/lib.rs runtime/src/ullm_runtime.cpp runtime/src/ullm_runtime_api_attention.inc runtime/src/ullm_runtime_hiprtc_sources.inc tools/run-runtime-cached-prefix-sweep.py
```

Note: the full repository `git diff --check` is currently blocked by an unrelated dirty `README.md` trailing whitespace change outside this work.

## R9700 Smoke Results

Device: R9700, runtime device index `2`.

| executor | L | M | q_heads | kv_heads | head_dim | value_dim | repeats | wall_ms_mean | input tok/s | sampled max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cached_prefix_rocwmma_fp8 | 31 | 3 | 32 | 2 | 32 | 32 | 3 | 0.055748 | 53813.911494 | 0.000000004 |
| cached_prefix_rocwmma_fp8 | 256 | 4 | 16 | 1 | 256 | 256 | 3 | 1.492044 | 2680.886086 | 0.000000067 |
| cached_prefix_rocwmma_fp8 | 4096 | 16 | 16 | 1 | 256 | 256 | 1 | 15.438269 | 1036.385621 | 0.000000719 |

## Comparison

For `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256`:

- Phase C12 value-tile-16 rocWMMA: `17.222257 ms`.
- Phase C13 value-group-64 rocWMMA: `15.438269 ms`.
- Existing scalar `cached_prefix_flash2 fp8_e4m3`: `3.952818 ms`.

The value-group-64 path is a correctness-preserving improvement over the previous rocWMMA 256-dimensional path, but it is still slower than scalar flash2. The remaining issue is that QK/softmax is still recomputed four times for `value_dim=256`.

