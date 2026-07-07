# Runtime decode attention head-parallel v1

Date: 2026-07-07

Change:

- `ullm_decode_attn_f32_kernel` now uses a head-parallel online-softmax
  path when `head_dim <= 256` and `value_dim <= 256`.
- The normal Qwen3.5-like decode shape is `q_heads=16`, `kv_heads=4`,
  `head_dim=256`, `value_dim=256`, so the optimized path is used.
- The old element-parallel path remains available and can be forced with
  `ULLM_DISABLE_DECODE_ATTN_HEAD_PARALLEL=1` for diagnostics.

Command shape for optimized path:

```bash
ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L 1 REPEATS 16 4 256 256 decode_loop
```

Command shape for old diagnostic path:

```bash
ULLM_REQUIRE_HIP_DECODE_ATTN_KERNEL=1 \
ULLM_DISABLE_DECODE_ATTN_HEAD_PARALLEL=1 \
  target/release/ullm-engine runtime-cached-prefix-attn-smoke \
  2 L 1 REPEATS 16 4 256 256 decode_loop
```

## M=1 Decode-Like Boundary

| L | repeats | old decode-loop ms | head-parallel decode-loop ms | speedup | decode-loop tok/s | decode-loop pair/s | chunked online ms | decode-loop vs chunked |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 3 | 103.281946 | 3.488255 | 29.608x | 286.676289 | 1174512.757812 | 4.039271 | 1.158x faster |
| 16384 | 1 | 522.266906 | 16.385404 | 31.874x | 61.029926 | 999975.343910 | 18.882177 | 1.152x faster |
| 65536 | 1 | 2035.323208 | 66.730666 | 30.501x | 14.985614 | 982112.182126 | 76.723657 | 1.150x faster |

## M=16 Decode Loop Check

The same decode loop is still a poor executor for multi-token chunks because
it launches one decode kernel per new token. The cached-prefix chunked kernel
remains the right path for `M >= 16`.

| L | M | decode-loop ms | decode-loop tok/s | chunked online ms | chunked online tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 16 | 49.781116 | 321.407017 | 4.193457 | 3815.467763 |
| 16384 | 16 | 305.251376 | 52.415816 | 19.256383 | 830.893320 |
| 65536 | 16 | 1199.161926 | 13.342652 | 78.523667 | 203.760224 |

## Interpretation

- The decode-like `M=1` boundary improved by roughly `30x` over the old
  element-parallel `decode_attn_f32` kernel.
- For `M=1`, `decode_attn_f32_loop` is now about `1.15x` faster than the
  chunked cached-prefix kernel on the measured `L=4096/16384/65536` cases.
- For `M=16`, decode-loop remains much slower because it serializes the
  chunk as one kernel launch per token. Use `cached_prefix_chunked` for
  `M >= 16`.
- This gives a practical executor split for Phase C4:
  `decode_attn_f32_loop` for `M=1`, `cached_prefix_chunked` for larger chunks.

## Verification

- `cargo fmt --all --check`
- `cargo test -p ullm-runtime-sys decode_attn -- --test-threads=1`
- `cargo build -p ullm-engine --release`
- R9700 smokes with required HIP kernel flags shown above
