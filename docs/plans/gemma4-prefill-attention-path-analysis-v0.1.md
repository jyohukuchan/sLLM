# Gemma4 prefill-attention path analysis v0.1

## Decision

The working hypothesis is **verified** for the current Gemma4 executor: its
public `prefill(&[N])` is a token loop, and every token traverses all 35
decoder layers.  Each layer invokes the M=1 `paged_decode_attn_f32` reader
once.  Consequently one N-token prefill makes exactly **35 N** paged-decode
attention launches: 4,480 at N=128, 17,920 at N=512, and 71,680 at N=2048.

This is not inferred from imports.  `prefill` delegates to `execute_step`
([`gemma4_text_executor.rs`](../../crates/ullm-engine/src/gemma4_text_executor.rs));
`execute_step` loops over `input_token_ids`; `forward_token` loops from layer
0 through layer 34; and the resident attention path calls
`attention_device_resident`, which calls `paged_decode_attn_f32`.  The
non-resident/device-KV bridge has the same one-token reader call in
`device_attention`.  The two call sites are alternative execution routes, not
two readers per layer.

The emitted logical accounting independently corroborates the count: the
resident N=128 and N=512 records report 4,480 and 17,920 attention calls,
respectively (`benchmarks/results/2026-07-27/gemma4-realistic-remeasurement-v0.1/raw/`).

## Production Qwen3.5 comparison

The production AQ4 worker is a different prefill design.  Its frozen binary
hashes to the manifest's required
`5a274733710d9b80a24d34a31ec6a99ac0b2d1e8fcce45904e906926a0e2e903`; its
strings contain the production IDs
`hip.paged-kv-write-chunk-f32.m2-m128`,
`hip.paged-causal-gqa-chunk-sigmoid-gate-f32.m2-m128`, and
`hip.paged-causal-gqa-chunk-wmma-sigmoid-gate-f32.gfx1201.q16-kv4-d256-page256.m2-m128`.
The active manifest requires both the chunk and WMMA feature guards.

The corresponding source route is explicit:

- The AQ4 session default is M=128 and its native prefill accepts M=2..128.
- `Qwen35Aq4ModelRuntime::dispatch_prefill_chunk_for_phase` keeps an `[M,H]`
  ping-pong activation buffer and invokes each decoder layer once per chunk.
- Each self-attention layer's `run_device_sequence_for_phase` batch-projects
  Q/K/V, runs batched Q/K norm + RoPE, launches one paged K/V chunk writer,
  then launches one paged causal GQA chunk reader.  On the exact Qwen geometry
  (16Q/4KV, 256/256, page 256, gated) registry priority selects the WMMA reader.

Qwen3.5 has eight self-attention layers among 32 decoder layers.  Therefore,
for an aligned N-token M=128 prefill it makes
`8 * ceil(N / 128)` causal-GQA reader launches (and the same number of chunk
writes): 8, 32, and 128 at N=128, 512, and 2048.  This count is specifically
for self-attention readers; the 24 linear-attention layers use their own
whole-sequence kernels.  It is not comparable to Gemma's 35 tokenwise
attention launches without retaining this model-architecture difference.

## Can Gemma4 use the existing chunk path?

### Sliding layers (28 layers, 8Q/1KV, 256/256, window 512)

The scalar paged chunk kernel itself is shape-compatible:

- It only requires `q_heads / kv_heads` to be integral; 8/1 is valid.
- Its kernel guard permits `head_dim <= 256` and `value_dim <= 256`; Gemma's
  local 256/256 satisfies that.
- A null gate is supported, so Gemma's lack of a Q gate is not a kernel-level
  blocker.
- It reads a page table, so a page table is not inherently a blocker either.

It is **not usable as-is by this executor**, for three independent reasons.

1. Gemma owns M=1 activation buffers and uses `matvec_bf16_f32`, `bf16_row_f32`,
   and CPU F32 transforms between primitives.  There is no `[M,H]` input,
   Q/K/V, output, or layer workspace, and no batched BF16 projection API in
   its executor.  Replacing only the reader cannot make M>1 queries exist.
2. The current local cache has `block_size = 1` and is a 512-entry ring.  Its
   `read_table` is rebuilt after each append to map logical oldest-to-newest
   sources to modulo-512 physical entries.  The chunk writer/reader instead
   uses `cache_start + row` and requires all causal source lengths in the
   chunk to be addressable.  It works directly only before the ring rolls
   over; at/after 512 it needs a Gemma-specific sliding/ring-aware chunk
   protocol (including chunk-boundary handling), not merely the existing call.
3. Gemma has K/V sharing after layer 14: later local layers read source-layer
   13 and later full layers read source-layer 14.  A layer-major chunk executor
   can preserve this, but only after it has materialized the entire source
   layer's chunk K/V before its shared consumers.  The existing token-major
   executor cannot do so without becoming a real chunk executor.

The Qwen WMMA specialization cannot be reused: it is intentionally exact to
16Q/4KV, 256/256, 256-token pages, and sigmoid gating.  Gemma would need a
new specialization if WMMA is desired; changing that production kernel is out
of scope and unsafe.

### Full layers (7 layers, 8Q/1KV, 512/512)

Neither existing paged scalar chunk reader nor the F32 Flash2 HIP readers can
run value/head dimension 512: both reject values over the 256-thread block
width.  A 512-wide reader (or a split/two-pass design) is necessary for a
batched full-layer path.  Leaving them tokenwise is semantically possible,
but it cannot be assumed to be inexpensive in prefill.

The optimistic attention-work-only Amdahl bound makes that concrete.  Count
QK plus AV work in proportion to `layers * q_heads * head_dim * attended_keys`.
For N <= 512, local attention contributes
`28 * 8 * 256` units per attended token and full attention contributes
`7 * 8 * 512`, so the local fraction is exactly 2/3.  Even instantaneous local
attention would therefore cap an attention-only workload at **3.000x**, not
5x.  At N=2048, the local window is capped while full attention keeps growing:

| N | local attention-work fraction | full fraction | ideal ceiling if only local attention vanished |
| ---: | ---: | ---: | ---: |
| 128 | 0.666667 | 0.333333 | 3.000x |
| 512 | 0.666667 | 0.333333 | 3.000x |
| 2048 | 0.466615 | 0.533385 | 1.875x |

These are deliberately optimistic ceilings: they exclude projections, MLP,
PLE, norms, transfers, CPU work, and all remaining launch overhead.  The
current Gemma profile does not time the resident attention primitive
separately, so it cannot support a stronger empirical speedup claim.

## Flash2 comparison

Flash2 is not a better drop-in path.

- `causal_attn_f32_flash2` consumes temporary contiguous `[T,H,D]` Q/K/V for
  an entire sequence.  It has no persistent paged-cache or sliding-ring
  contract, so it cannot retain Gemma decode state without an additional
  cache/materialization path.
- `cached_prefix_attn_f32_flash2` handles a contiguous cached prefix plus a
  contiguous M-token suffix.  It supports generic GQA such as 8Q/1KV and
  256-wide values, but it indexes K/V linearly and also has the 256 value-dim
  limit.  Gemma's full layers fail the latter condition, and its rolled sliding
  ring fails the former unless K/V are copied/gathered to contiguous storage or
  a new ring-aware kernel is written.
- The only grouped Flash2 fast path is exact 5:1 GQA with 128-wide heads on
  gfx1201, so it does not accelerate Gemma's 8:1 / 256 geometry.

Thus Flash2 can be a useful reference implementation for an initial contiguous
local-prefill segment, but it does not solve the required whole-model chunk
executor, the long-prompt ring state, or the 512-wide full layers.  The paged
chunk design is the closer semantic target once a Gemma-specific layer-major
M=128 activation path exists.

## Scope decision

No implementation is justified in this session.  The missing item is not one
kernel import: it is a Gemma-specific batched activation graph plus sliding
cache protocol, then a 512-wide/full-layer strategy.  Wiring a reader alone
would either be unreachable (all inputs remain M=1) or change the cache
semantics after the 512-token boundary.  Therefore no GPU work, no runtime
source modification, and no production-kernel change was performed.

The mandatory Qwen production probe remains byte-identical in the existing
rollback record: SHA-256
`30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`, top-1
220 at logit `8.529029846191406`.  This analysis touched neither the worker
binary nor the runtime source guard.

## Follow-up implementation plan

1. Add Gemma-only `[M,H]` resident activation/workspace ownership and batch
   versions of every required projection, RMSNorm, PLE, residual, and dense
   MLP operation; retain the existing M=1 decode route byte-for-byte.
2. Add a Gemma-specific F32 paged local-attention chunk reader/writer contract
   for 8Q/1KV, 256/256, block-1 ring pages, including an exact 512-boundary
   and wrap protocol.  Do not modify Qwen's scalar or WMMA kernels.
3. Preserve K/V sharing by committing source layer 13/14 K/V for the full
   chunk before dispatching consumer layers.
4. Add either a 512-wide paged chunk reader or a verified split/full-layer
   fallback; measure its actual prefill share before choosing the design.
5. Validate multi-step hidden states and logits against captured host
   activations, then queue 128/512/2048 timings behind
   `/run/ullm/r9700.lock` after a clean rebuild and production probe.
