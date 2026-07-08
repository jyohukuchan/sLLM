# Phase C12: RDNA4 FP8 rocWMMA Cached-Prefix 16n Dimensions v1

## Summary

- Extended `cached_prefix_rocwmma_fp8` from fixed `head_dim=16,value_dim=16` to dimensions that are multiples of 16.
- QK now accumulates over `head_dim` in 16-wide rocWMMA chunks.
- `value_dim` is split into 16-column output tiles, with one HIP block per `(new token, KV head, Q-head group, value tile)`.
- The implementation still recomputes QK and online softmax for each value tile, so this is a correctness/shape expansion step, not the final efficient real-head-dim kernel.

## Verification

```text
cargo fmt --all --check
git diff --check
cargo check -p ullm-engine
cargo test -p ullm-runtime-sys cached_prefix_attn_fp8_e4m3_rocwmma -- --test-threads=1
cargo build -p ullm-engine --release
```

The targeted runtime-sys tests now use `head_dim=32,value_dim=32` and passed on CPU fallback plus RDNA4 HIP.

## R9700 Smoke Results

Device: R9700, runtime device index `2`.

| executor | L | M | q_heads | kv_heads | head_dim | value_dim | repeats | wall_ms_mean | input tok/s | sampled max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cached_prefix_rocwmma_fp8 | 31 | 3 | 32 | 2 | 32 | 32 | 3 | 0.061021 | 49163.402763 | 0.000000004 |
| cached_prefix_rocwmma_fp8 | 4096 | 16 | 16 | 1 | 256 | 256 | 1 | 17.222257 | 929.030382 | 0.000000719 |
| cached_prefix_flash2 | 4096 | 16 | 16 | 1 | 256 | 256 | 1 | 3.952818 | 4047.745178 | 0.000000611 |

## Notes

- The 256-dimensional rocWMMA path is currently slower than scalar `cached_prefix_flash2` because QK and softmax are recomputed for each 16-column value tile.
- The next kernel shape should keep value-tile parallelism while avoiding QK/softmax recomputation, or use a wider block layout where multiple lanes cooperate on the full `value_dim` accumulation.
- The sweep tool now rejects invalid `cached_prefix_rocwmma_fp8` shapes before launching:
  - `q_heads / kv_heads` must be a multiple of 16.
  - `head_dim` and `value_dim` must be multiples of 16.

