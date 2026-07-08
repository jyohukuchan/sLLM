# Phase C17: RDNA4 FP8 rocWMMA Long-Prefix Grid v1

## Summary

- Ran the saved long-prefix cached-prefix grid with `cached_prefix_rocwmma_fp8`.
- Compared `rocWMMA value group = auto,16,64` on `L={4096,16384,65536}`, `M={16,128,512}`.
- Re-ran scalar `cached_prefix_flash2` and `cached_prefix_flash2_fp8q` on the same `q_heads=16,kv_heads=1,head_dim=256,value_dim=256` shape so the comparison is aligned with rocWMMA.

## Verification

```text
python3 tools/run-runtime-cached-prefix-sweep.py --executors cached_prefix_rocwmma_fp8 --cached-prefix-tokens 4096,16384,65536 --new-tokens 16,128,512 --kv-cache-dtype fp8_e4m3 --q-heads 16 --kv-heads 1 --head-dim 256 --value-dim 256 --measured-repeats 2 --long-measured-repeats 1 --long-prefix-threshold 16384 --long-new-token-threshold 512 --rocwmma-value-group-widths auto,16,64 --output-jsonl /tmp/ullm-rocwmma-vgw-longgrid.jsonl --summary-md /tmp/ullm-rocwmma-vgw-longgrid.md --timeout-seconds 180 --keep-going
python3 tools/run-runtime-cached-prefix-sweep.py --executors cached_prefix_flash2,cached_prefix_flash2_fp8q --cached-prefix-tokens 4096,16384,65536 --new-tokens 16,128,512 --kv-cache-dtype fp8_e4m3 --q-heads 16 --kv-heads 1 --head-dim 256 --value-dim 256 --measured-repeats 2 --long-measured-repeats 1 --long-prefix-threshold 16384 --long-new-token-threshold 512 --output-jsonl /tmp/ullm-flash2-kv1-longgrid.jsonl --summary-md /tmp/ullm-flash2-kv1-longgrid.md --timeout-seconds 180 --keep-going
```

## R9700 rocWMMA Value Group Grid

Device: R9700, runtime device index `2`.

Shape: `q_heads=16,kv_heads=1,head_dim=256,value_dim=256`, `kv_cache_dtype=fp8_e4m3`.

| L | M | auto ms | vgw16 ms | vgw64 ms | best | sampled max abs diff |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 4096 | 16 | 18.163768 | 18.344952 | 20.857098 | auto | 0.000000719 |
| 4096 | 128 | 15.355270 | 47.369292 | 15.497008 | auto | 0.000000723 |
| 4096 | 512 | 73.213893 | 146.624590 | 73.472518 | auto | 0.000000730 |
| 16384 | 16 | 58.739723 | 56.719604 | 83.820396 | 16 | 0.000002488 |
| 16384 | 128 | 59.295966 | 255.846758 | 59.160101 | 64 | 0.000002498 |
| 16384 | 512 | 452.380198 | 992.538208 | 444.854566 | 64 | 0.000002533 |
| 65536 | 16 | 285.538104 | 286.489041 | 332.435952 | auto | 0.000005206 |
| 65536 | 128 | 292.829989 | 699.096054 | 298.131400 | auto | 0.000005225 |
| 65536 | 512 | 944.787375 | 2171.721713 | 950.124268 | auto | 0.000005290 |

The runtime heuristic is fastest or within about `2%` of the best explicit width on this grid.

## Scalar Comparison

Same device and shape.

| L | M | scalar flash2 ms | scalar flash2_fp8q ms | rocWMMA auto ms | roc / flash2 | roc / flash2_fp8q |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 3.951172 | 4.287609 | 18.163768 | 0.218x | 0.236x |
| 4096 | 128 | 26.797079 | 29.051563 | 15.355270 | 1.745x | 1.892x |
| 4096 | 512 | 102.447155 | 111.877889 | 73.213893 | 1.399x | 1.528x |
| 16384 | 16 | 13.817192 | 15.207052 | 58.739723 | 0.235x | 0.259x |
| 16384 | 128 | 109.587689 | 134.301277 | 59.295966 | 1.848x | 2.265x |
| 16384 | 512 | 459.709158 | 580.372267 | 452.380198 | 1.016x | 1.283x |
| 65536 | 16 | 54.797003 | 59.837912 | 285.538104 | 0.192x | 0.210x |
| 65536 | 128 | 429.821067 | 468.840212 | 292.829989 | 1.468x | 1.601x |
| 65536 | 512 | 1678.206541 | 1839.400876 | 944.787375 | 1.776x | 1.947x |

`roc / flash2` is the scalar wall time divided by rocWMMA wall time, so values above `1.0x` mean rocWMMA is faster.

## Interpretation

- `M=16` remains a scalar flash2 path. rocWMMA loses by roughly `4-5x` there.
- `M=128` is consistently rocWMMA-favorable across all tested prefix lengths.
- `M=512` is also rocWMMA-favorable at `L=4096` and `L=65536`; at `L=16384` it is only slightly faster than F32-Q scalar flash2 but still faster than FP8-Q scalar flash2.
- For SQ candidate evaluation, `cached_prefix_rocwmma_fp8` is now a useful R9700/RDNA4 baseline for chunked prefill, while scalar `cached_prefix_flash2_fp8q` should remain the short-chunk/decode-like baseline.

## Next Action

- Route future SQ cached-prefix measurements by chunk size: scalar flash2/fp8q for short chunks, rocWMMA for `M>=128`.
- The next kernel change should target multi-query-token tiling to reduce scalar flash2's advantage at `M=16` without losing rocWMMA's larger-chunk advantage.
