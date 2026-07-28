# Gemma4 512-wide batched full-attention reader v0.1

## First validation increment

The isolated 8Q/1KV/512 F32 reader compiled and ran on R9700/gfx1201 after a
clean `cargo clean -p ullm-runtime-sys` rebuild.  It launches one CTA per full
attention Q head and reads each K/V source once into LDS for the query chunk.
The ordinary 256-wide paged-decode, chunk, and WMMA bodies are unchanged.

`raw/validation.json` is the first full-model result (resident load 9.724 s):

- Causal future-token probe: `0 / 0` max abs/rel.
- Full-model real-activation differential: layer output max abs
  `2.288818359375e-5`; final norm `4.00543212890625e-5`; logits
  `2.2411346435546875e-5`; resident and host top-1 both `9079 / 22.510112762451172`.
- Both known cached/re-prefill continuations agree exactly.
- The K/V source snapshot confirms source layer 14 is complete before full
  consumers 19, 24, 29, and 34 execute.

This is not a promotion result.  The layer-major path remains gated pending
tile-width sensitivity and end-to-end timing.

## First throughput evidence

One-repeat clean-release timing with the same fixed token-2 prompt found:

| context | query tile | prefill tok/s | decode tok/s | full reader launches | total reader launches |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 16 | 47.371 | 17.253 | 56 | 3,640 |
| 128 | 128 | 53.748 | 16.393 | 7 | 3,591 |
| 512 | 128 | 41.686 | 14.190 | 28 | 14,364 |

The M=16/M=128 comparison has the same final top-1 (`184`) for the 128-token
fixed prompt.  M=128 is 13.5% faster and is the candidate width.  Its reader
count is exactly `7 * ceil(N / 128)`; the remaining calls are the untouched
28 sliding layers (`28 * N`).

N=2048 is running detached under the R9700 lock at the time of this increment.
The recovery monitor will restart `ullm-openai` only after that lock releases.
