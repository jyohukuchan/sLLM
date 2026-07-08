# Phase C15: RDNA4 FP8 rocWMMA Value Group Heuristic v1

## Summary

- Replaced the fixed `cached_prefix_rocwmma_fp8` value group width with a runtime-selected value group width.
- `ULLM_ROCWMMA_CACHED_PREFIX_VALUE_GROUP_WIDTH={16,32,64,128,256}` can override the selection.
- Without the env override, the launcher currently uses:
  - `16` when `new_tokens < 64`
  - `64` otherwise
- The kernel receives the chosen width as an argument, so the HIPRTC module no longer needs to be recompiled for each value group width.

## Verification

```text
cargo fmt --all --check
cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_rocwmma -- --test-threads=1
cargo build -p ullm-engine --release
git diff --check -- runtime/src/ullm_runtime.cpp runtime/src/ullm_runtime_hiprtc_sources.inc
```

Note: the full repository `git diff --check` is still blocked by an unrelated dirty `README.md` trailing whitespace change outside this work.

## R9700 Value Group Width Sweep

Device: R9700, runtime device index `2`.

Shape: `L=4096,M=16,q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`, `repeats=3`.

| value group width | wall_ms_mean | input tok/s | attention pair/s | sampled max abs diff |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 16.797953 | 952.497010 | 3909523.975927 | 0.000000719 |
| 32 | 19.943304 | 802.274274 | 3292934.756566 | 0.000000719 |
| 64 | 21.011732 | 761.479361 | 3125492.036631 | 0.000000719 |
| 128 | 32.050161 | 499.217461 | 2049038.068795 | 0.000000719 |
| 256 | 41.351419 | 386.927472 | 1588143.807109 | 0.000000719 |

For decode-like `M=16`, more value-group parallelism wins over reducing QK/softmax recomputation.

## R9700 Executor Comparison

Device: R9700, runtime device index `2`.

Shape: `L=4096,q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`, `repeats=2`.

| M | executor | Q dtype | value group width | wall_ms_mean | input tok/s | attention pair/s | sampled max abs diff |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | cached_prefix_rocwmma_fp8 | FP8 | heuristic 16 | 18.469110 | 866.311347 | 3555774.924840 | 0.000000719 |
| 128 | cached_prefix_flash2 | F32 | n/a | 26.629428 | 4806.712333 | 19998326.663269 | 0.000000585 |
| 128 | cached_prefix_flash2_fp8q | FP8 | n/a | 28.595555 | 4476.220168 | 18623314.008069 | 0.000002202 |
| 128 | cached_prefix_rocwmma_fp8 | FP8 | heuristic 64 | 15.409175 | 8306.739059 | 34560187.856904 | 0.000000723 |
| 512 | cached_prefix_flash2 | F32 | n/a | 103.509898 | 4946.386890 | 21529148.939598 | 0.000000522 |
| 512 | cached_prefix_flash2_fp8q | FP8 | n/a | 113.356638 | 4516.718289 | 19659016.351561 | 0.000002287 |
| 512 | cached_prefix_rocwmma_fp8 | FP8 | heuristic 64 | 70.911643 | 7220.252956 | 31426150.991876 | 0.000000730 |

## Interpretation

- `cached_prefix_rocwmma_fp8` is still not a good decode-like path for `M=16`; scalar `cached_prefix_flash2_fp8q` remains faster there.
- For larger prefill chunks, rocWMMA now beats scalar flash2:
  - `M=128`: rocWMMA is about `1.73x` faster than F32-Q scalar flash2 and `1.86x` faster than FP8-Q scalar flash2.
  - `M=512`: rocWMMA is about `1.46x` faster than F32-Q scalar flash2 and `1.60x` faster than FP8-Q scalar flash2.
- This supports keeping the RDNA4 FP8 WMMA path for chunked prefill/SQ-format evaluation, while using scalar flash2/fp8q as the short-chunk baseline.

## Next Action

- Add a cleaner benchmark/sweep dimension for rocWMMA value group width if more tuning is needed.
- Continue toward a true FlashAttention2-like multi-query-token tile structure, especially for larger `M`, because this is where rocWMMA has started to pay off.
