# Gemma4 paged-decode 512-wide cliff: source audit

## Finding

The 512-wide Gemma4 full-attention shape takes the scalar fallback in
`ullm_paged_decode_attn_f32_kernel`; it is not an occupancy, spill, LDS, or
vector-load cliff in the 256-wide head-parallel kernel.  The launcher selects
the head-parallel grid only when both widths are at most 256.  Gemma4's
8Q/1KV/512/512 shape therefore launches 16 blocks (8 * 512 output elements /
256 threads), instead of one 256-thread block per Q head.

In that fallback, every output element independently runs a loop over the
entire `head_dim` for both the max pass and the weighted-value pass.  Thus, for
each Q head and source KV entry, the 512-wide route computes 512 identical
512-term Q dot K products.  Its dot-product work is `512 * 512 = 262,144`
FMAs/head/source, plus 512 value FMAs.  A 256-thread head-parallel 512-wide
kernel would compute 512 dot FMAs (two per thread) plus 512 value FMAs: 1,024
FMAs/head/source.  The source-visible amplification is therefore
`262,656 / 1,024 = 256.5x` (or exactly 512x for the dot-product portion).

For the 256-wide sliding shape, the existing head-parallel path performs 256
dot plus 256 value FMAs, 512 total/head/source.  The necessary width-only work
ratio from 256 to 512 is 2x.  The fallback instead costs
`262,656 / 512 = 513x` of the 256-wide arithmetic, so the arithmetic excess
over the expected 2x is 256.5x.  This is sufficient to explain 100% of the
observed 21x unexplained portion; it overpredicts the measured 43x per-layer
ratio because the session-DT timing region also includes projections, KV writes,
norms, launches, synchronization, and the full/sliding layer configurations
differ outside the reader.

## Evidence

At commit `4252b219`, the N=512 attention-region timing was 62.063165 s for
the 7 full layers (8.866 s/layer) and 5.713319 s for 28 sliding layers
(0.204 s/layer), a 43.46x per-layer ratio.  The original records are in the
adjacent `gemma4-prefill-attention-split-v0.1` directory.

The relevant source statements are:

- `runtime/src/parts/paged_decode_attention_kernels.inc`: launcher guard
  `head_dim <= 256 && value_dim <= 256` and scalar grid selection otherwise.
- `runtime/src/kernels/attention/attention_sources.inc`: matching in-kernel
  guard, followed by the fallback's `for (dim = 0; dim < head_dim; ++dim)` in
  each output thread in both passes.

The canonical 256-wide body has no width-512 execution here, so it cannot
itself have an occupancy/spill/LDS/vectorisation transition.  The fallback
does not use a width-dependent vector load either: it is scalar indexed loads.
No claim about live VGPR/AGPR or launch residency is made in this audit; those
metrics are not needed to establish the control-flow and arithmetic cause and
were not inferred from code-object metadata.

## Fix boundary

Add a separate exact-512, 256-thread/head-parallel path: each thread owns
dimensions `tid` and `tid + 256`, reduces their two Q*K products with the
existing block reducer, and accumulates two V outputs.  Keep the complete
existing `<=256` branch byte-for-byte unchanged, preserving the Qwen3.5
16Q/4KV/256/256 route.  Re-record the runtime translation-unit guard in the
same kernel-change commit.
