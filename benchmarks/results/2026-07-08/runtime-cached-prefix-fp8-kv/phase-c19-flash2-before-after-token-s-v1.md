# Phase C19: FlashAttention2-style Before/After Token/s Comparison v1

## Scope

- Device: R9700/RDNA4, runtime device index `2`.
- Workload: cached-prefix prefill, `kv_cache_dtype=fp8_e4m3`.
- Shape: `q_heads=16,kv_heads=4,head_dim=256,value_dim=256`.
- Before: `cached_prefix_chunked`.
- After: `cached_prefix_flash2`.
- Source: `phase-c5-flash2-tiled-online-softmax-v1.md`.

This table intentionally does not mix in `cached_prefix_rocwmma_fp8` or `cached_prefix_rdna4_fp8_auto`, because those later comparisons used `kv_heads=1` to satisfy the current rocWMMA grouping constraint. They should be compared in a separate aligned-shape table.

## Token/s Comparison

`tok/s` here is `new_prefill_tokens / wall_ms_mean * 1000`, reported by `runtime-cached-prefix-attn-smoke` as `new tok/s`.

| L cached prefix | M new tokens | before executor | before tok/s | after executor | after tok/s | speedup |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 4096 | 16 | `cached_prefix_chunked` | 3756.080743 | `cached_prefix_flash2` | 4326.088667 | 1.152x |
| 4096 | 128 | `cached_prefix_chunked` | 4052.532985 | `cached_prefix_flash2` | 4707.871948 | 1.162x |
| 4096 | 512 | `cached_prefix_chunked` | 3986.348997 | `cached_prefix_flash2` | 4951.346732 | 1.242x |
| 16384 | 16 | `cached_prefix_chunked` | 1027.283076 | `cached_prefix_flash2` | 1224.474490 | 1.192x |
| 16384 | 128 | `cached_prefix_chunked` | 1024.023995 | `cached_prefix_flash2` | 1268.755364 | 1.239x |
| 16384 | 512 | `cached_prefix_chunked` | 1011.147714 | `cached_prefix_flash2` | 1248.054936 | 1.234x |
| 65536 | 16 | `cached_prefix_chunked` | 138.527036 | `cached_prefix_flash2` | 207.934982 | 1.501x |
| 65536 | 128 | `cached_prefix_chunked` | 101.306764 | `cached_prefix_flash2` | 128.655704 | 1.270x |
| 65536 | 512 | `cached_prefix_chunked` | 105.250633 | `cached_prefix_flash2` | 143.211551 | 1.361x |

## Interpretation

- The FlashAttention2-style scalar cached-prefix executor improved every measured FP8 KV case in this aligned grid.
- The improvement range was `1.152x-1.501x`.
- The largest token/s ratio appeared at `L=65536,M=16`, but that long-prefix/small-chunk region was also the most variance-sensitive in earlier runs.
- This is not yet the full RDNA4 FP8 rocWMMA path. It is the before/after comparison for introducing the tiled online-softmax FlashAttention2-style executor.
