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
