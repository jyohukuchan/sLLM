# Gemma4 prefill optimisation summary v1.0

## Verdict

The Gemma4 E2B BF16 resident path is substantially better. The previous
statement that no further prefill change was justified was incomplete: the
sliding direct-ring reader was left disabled, so its already-implemented
split-KV path had never run. Session EG activated and measured that combination
on the R9700, then promoted it with an explicit `=0` rollback.

This document consolidates the completed effort. It distinguishes measured
throughput and trace attribution from derived Amdahl bounds; those bounds are
not additive and are never performance forecasts.

## Measurement contract and result

All uLLM before/after throughput rows below are matched clean release builds,
median of five runs, serialised with `flock /run/ullm/r9700.lock` on the
R9700 only (amd-smi GPU 2, HIP ordinal 1, `gfx1201`). `ullm-openai` was
stopped for the GPU window and restored afterward. The pre-residency baseline
was built separately, rather than inferred by toggling a live binary. The
benchmark uses the same fixed workload at each listed context.

The llama.cpp comparison is the matched-settings record for the same
`gemma-4-E2B-BF16.gguf` on the same GPU: F32 K/V and Flash Attention off.
It is a comparison reference, not evidence that uLLM and llama.cpp share
implementation or sampling noise. Ratios are llama.cpp tok/s divided by the
final uLLM tok/s.

| context | pre-residency prefill | final prefill | gain | llama.cpp prefill | llama/uLLM | pre-residency decode | final decode | gain | llama.cpp decode | llama/uLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12.047 | 59.02 | 4.90x | 3,858.17 | 65.4x | 6.596 | 22.62 | 3.43x | 73.459 | 3.25x |
| 512 | 5.613 | 58.99 | 10.51x | 7,532.68 | 127.7x | 2.948 | 18.32 | 6.21x | 73.278 | 4.00x |
| 2048 | 1.827 | 54.04 | 29.58x | 6,305.79 | 116.7x | 0.933 | 15.53 | 16.64x | 73.369 | 4.72x |

The important shape result is that prefill and decode stopped collapsing with
context. The original prefill scaling was approximately N^1.551 to N^1.810;
the promoted path is nearly flat over these points. The gap to llama.cpp is
still real, especially for prefill, and must not be described as closed.

## Correction: sliding split-KV was not previously measured

The earlier `0.916x` direct-ring result predates sliding split-KV. Its reader
ran at 8 CTAs / 64 CUs (0.125 CTA/CU), the root cause later fixed for the full
reader. The `ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED` dispatch flag remained
off, which made the sliding split default unreachable; it was therefore wrong
to describe it as a tried-and-failed final candidate or to require a new
sliding architecture before measuring it.

With the ring route enabled, sliding split factor 16 is the current
end-to-end N=2048 winner (56.069 tok/s in the one-cold-run factor sweep; 32
minimizes reader-only round-trip time but loses to merge/other costs). A kernel
trace of factor 16 reports 128 CTAs per sliding partial dispatch, **2.0
CTA/CU**, and 103.504 derived source-accounted QK+AV GFLOPS, replacing the old
0.125 CTA/CU / 10.977 GFLOPS state. These are dispatch timing and derived work,
not PMC/HBM counters.

## What was actually wrong, in discovery order

1. **Resident weights and KV did not make the activation graph resident.**
   The early decode wall accounting found 81.062% of the wall outside any
   kernel. Only 2.488% of that gap elapsed after the next launch API returned,
   which ruled out bare launch latency as the explanation. Gemma4 uploaded
   BF16 weights and retained KV on the GPU, but copied each activation through
   pageable host memory: result D2H, stream synchronization, host F32 decode
   and transform, then the next H2D. The detailed twelve-token profile later
   apportioned 46.98% to D2H submission and 22.10% to synchronization; these
   are overlapping stages of that serial host graph, not independent GPU time.

2. **There originally was no batched Gemma4 prefill attention path.** Before
   the layer-major work, a public prefill was a token loop through all 35
   decoder layers and therefore made 4,480 / 17,920 / 71,680 M=1 reader
   launches at N=128 / 512 / 2048. Production Qwen3.5, by contrast, uses
   M=128 chunks and makes 8 / 32 / 128 self-attention reader launches at the
   same contexts (eight self-attention layers times `ceil(N/128)`). This is a
   design comparison, not an assertion that the models have the same layer
   mix.

3. **The full 512-wide layers had a catastrophic scalar fallback.** Its
   output-thread mapping recomputed the complete 512-term Q dot K for each
   output element: 262,656 FMAs per head/KV-token rather than the 1,024 FMAs
   needed to form QK once and apply the resulting weight to V. The promoted
   full reader batches query rows and uses a split-KV partial/merge route;
   it removes this specific recomputation but does not make the whole graph
   device-resident.

4. **The surviving reader problem was a grid-supply problem, not an HBM or
   scalar-FLOP roof.** Both M=1 sliding and the underfilled merge dispatch use
   a 2,048-thread grid and a 256-thread block: 8 CTAs for 64 CUs, or 0.125
   CTA/CU. The source-accounted QK+AV rates were only 10.835--17.010 GFLOPS,
   0.023--0.036% of the R9700 47.8-TFLOPS FP32 vector peak. This is the
   root cause that explains why large-looking logical-read and FMA reductions
   barely moved end-to-end throughput.

`rocprof` PMC mode could not validate physical HBM traffic: PMC collection
opens the GPU before the resident target, and the target intentionally rejects
the resulting R9700 process. Consequently no claim here relies on physical
HBM bytes, per-CU residency, LDS-bank conflicts, or counter-derived
occupancy. The byte and FLOP figures are source-accounted work; kernel traces
establish launch geometry and duration.

## Changes that were accepted

- The resident execution path removed the original repeated weight/KV
  residency problem while retaining the exact F32 host-side numerical route
  where required.
- A Gemma-only layer-major M=128 prefill path was introduced for the full
  512-wide attention layers, with explicit shared-KV source ordering.
- Full attention uses F32 split-KV partial/merge readers; the token-major
  path is still available as a rollback.
- Sliding 256-wide attention now defaults to the direct-ring batched split-KV
  reader (factor 16); `ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED=0` restores
  the former M=1 route. This activation was validated separately for future
  keys, `j-512`, a 2,052-token cached continuation, logits, multi-step cache
  equivalence, and source-13/source-14 shared-KV consumers.
- Gemma-only BF16 batched matmul moved from M=8 to M=16 while retaining the
  F32 input, accumulation, and reduction order. Its clean five-median gains
  were 1.040x / 1.075x / 1.024x at N=128 / 512 / 2048. The representative
  12,288 x 1,536 x 128 MLP dispatch achieved 1.024 TFLOPS and was well
  populated (up to 1,536 CTA/CU); it is not another underfilled reader.

## What did not work, and why

| attempt | measured result | why it was rejected |
| --- | --- | --- |
| Isolated direct-weight RMSNorm port | 12.297 decode tok/s versus 15.733 primitive baseline; 13.078 versus 18.544 prefill tok/s | It returned the norm output to the host while its producer and consumer remained host-mediated. Extra H2D, launch, and D2H outweighed the saved row read. |
| Sliding reader batching, snapshot gather | Numerically validated, but did not meet the promoted throughput baseline | Reducing logical reads did not supply enough useful work to the GPU or retain the whole activation graph. |
| Sliding reader batching, direct-ring before split-KV | About 100x fewer logical K/V reads, **0.916x** realised N=512/N=2048 throughput | This result is historical only: it used the 8-CTA reader before split-KV supplied enough CTAs. It must not be used to reject the activated split route. |
| Reducing the 512-wide fallback's FMAs | About 256x less redundant FMA work, only about +20% | The reader was neither compute- nor bandwidth-roofed; 8 CTAs left 56 CUs idle. |
| Split merge-grid tuning | Free-merge Amdahl ceiling was 1.013x in its measured configuration | Even though its 8-CTA merge grid was underfilled, it was too small a fraction of whole prefill. It was correctly rejected. |
| More small BF16 matmul tiling | M=8 to M=16 realised only 1.075x at N=512 and 1.024x at N=2048 | The remaining obvious tile adjustment is below the project bar; matrix-instruction approaches round F32 operands and have no accepted numerical route. |

The unifying lesson is important: do not promote a local primitive, a lower
logical byte count, or a lower FMA count merely because its isolated bound is
large. First inspect the actual grid. A kernel with 0.125 CTA/CU can be far
from both arithmetic and bandwidth limits while still dominating elapsed time.

## Numerical and production discipline

Every accepted Gemma change used the following discipline.

- Causal perturbations were tested on both sides of the boundary: future keys
  must not affect past queries, and the excluded `j-512` key must not affect a
  sliding-window query.
- Ring rollover was tested at a wrapping length, including the N=2048 route,
  rather than only before the first 512-token overwrite.
- Split-KV tile widths were checked at multiple split factors. Softmax
  split/merge error is non-monotonic, so a single passing tile is not a proof;
  the Qwen3-14B SQ8_0 precedent had errors 1.309 and 2.376 after 40 layers.
- Real activation differentials included layer outputs, final norm, and
  logits, not just greedy text. The promoted M=16 record, for example, had
  maximum layer/final-norm/logit absolute differences of
  2.288818359375e-5 / 3.0517578125e-5 / 1.621246337890625e-5 and identical
  resident/host top-1 `9079 / 22.510112762451172`.
- Cached continuation and full reprefill routes were compared, and shared-KV
  consumers were checked to use source layers 13 and 14 as intended.
- The production Qwen3.5 AQ4_0 fixture was byte-identical after every step:
  SHA-256 `30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`,
  top-1 `220 / 8.529029846191406`. Its frozen production worker remained
  `5a274733710d9b80a24d34a31ec6a99ac0b2d1e8fcce45904e906926a0e2e903`.

The runtime translation-unit guard at the final promoted state is
`475204184566b0798883a931c1f1528b86dec79b6b1aeb8310a1637d2153f699`.

## What remains, honestly

The llama.cpp gap in the first table remains the primary performance debt.
The final N=512/N=2048 current-build trace found these non-additive bounds:

| future candidate | current measured share (512 / 2048) | free-component ceiling | conclusion |
| --- | ---: | ---: | --- |
| Further M=16 BF16 matmul redesign | 25.62% / 21.51% end-to-end | 1.344x / 1.274x | Actual adjacent M=8-to-M=16 result is <=1.075x / <=1.024x; no quick accepted variant remains. |
| Further sliding attention architecture | 32.66% / 39.13% reader envelope (pre-activation trace) | 1.485x / 1.643x | Reassess only after the activated direct-ring split baseline; the cited 0.916x result was pre-split and is not a rejection of this path. |
| Reader transport only | 9.33% / 7.16% | 1.103x / 1.077x | Below the bar across target contexts and inseparable from device-graph work. |
| Full split partial | 1.16% / 3.09% GPU | 1.012x / 1.032x | Too small. |
| Split merge | 0.063% / 0.047% GPU | 1.0006x / 1.0005x | Too small. |

The following controls are intentionally retained:

- `ULLM_GEMMA4_PREFILL_SLIDING_RING_BATCHED=0` rolls back the promoted sliding
  batched route to the former M=1 reader. Omission is the promoted default.
- `ULLM_GEMMA4_PREFILL_LAYER_MAJOR=0` rolls back to the former token-major
  route.
- `ULLM_GEMMA4_FULL_ATTN_SPLIT_KV=0` and
  `ULLM_GEMMA4_SLIDING_ATTN_SPLIT_KV=0` roll back the corresponding split-KV
  readers.

## Known defects, recorded but not fixed

1. The MoE prewarm RMS path hardcodes epsilon `1e-5` while the descriptor
   specifies `1e-6`. This is a real correctness/configuration defect, not a
   Gemma prefill performance result.
2. A claimed 262,144-token MoE capability does **not** reproduce from a fresh
   source build: the final full-attention V cache allocation OOMs. Do not quote
   262,144 tokens as a current capability.

## Record corrections and limits

- Historical reader reports that described the old M=1 full reader are not
  descriptions of the current default; current full prefill uses the batched
  split reader. Historical launch counts remain valid only when labelled as
  pre-change evidence.
- The 1.013x merge rejection is a valid historical upper bound for its then
  measured configuration. The final trace's merge share is smaller
  (0.063% / 0.047%), so it strengthens rather than reverses that decision.
- The final candidate trace is a one-cold-run attribution instrument. Its
  wall time is deliberately not substituted for the clean five-median
  throughput table above.
