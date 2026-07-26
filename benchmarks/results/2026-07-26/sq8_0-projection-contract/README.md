# SQ8_0 handwritten projection cumulative-contract diagnosis

Date: 2026-07-26 JST

## Decision

The private gfx1201 handwritten WMMA projection remains **NO-GO**. The
ordinary SQ8_0 dispatch remains CK; no active manifest, campaign,
authorization, release, or default was changed. Candidate timing was not run,
because the unchanged numerical gate still fails.

For the evaluated wave32 handwritten route, the answer to “can it be faster
while preserving the CK contract?” is **no at present**: it does not preserve
the contract. A general impossibility result for every hand-written
implementation is **unconfirmed**. Reproducing CK would require an exact
within-K128 WMMA fragment/load/issue mapping that this investigation did not
decode, and its performance would need a fresh gate before timing.

## Evidence validity

| attempt | service window | status | use in conclusion |
| --- | --- | --- | --- |
| attempt-1 | 09:03:17–09:03:36 | Valid, but only an isolated layer-0 reconstruction | No: it was exact and therefore could not locate the full-model divergence. |
| attempt-2 | 09:11:53–09:13:08 | Invalid for numerical evidence | No: ullm-openai.service started at 09:12:48 while diagnostic artifacts were still written at 09:13:06–09:13:08. It is retained for audit only. |
| attempt-3 | 09:19:29–09:20:46 | Valid, fully isolated actual-serving trace | Yes: all numerical conclusions below use this attempt only. |

attempt-3/diagnostic/report.json is the machine-readable primary record. It
captures the exact 512-token raw-p0512 fixture, M8-chunked prefill, and the
first ordinary M=1 feedback decode. The terminal diagnostic reads layer
workspaces but neither emits a head token nor commits a session mutation.

## First divergence and the component-gate gap

Both profiles produced prefill token 66; their first decode input token and
position were also 66 and 512. The actual layer trace is nevertheless not
equal:

| boundary | CK vs handwritten result |
| --- | --- |
| Layers 0–2, every captured stage | bitwise equal |
| Layer 3, down_projected | 2 / 5,120 bits differ; first index 1,954; max abs 6.1035156e-5 |
| Layer 3, final layer output | the same 2 / 5,120 differences |
| Direct replay of that down projection | the same 2 differences; SHA-256 differs (b4c633… vs 0b82d2…) |

The direct replay and the activation replay both match the actual layer-3
trace, so this is the real projection boundary rather than a reconstructed
input mistake. The former component gate verified four synthetic,
single-projection cases at the BF16 boundary. It did **not** exercise the
actual layer-3 artifact activation, its 136 K128 blocks, the full serving
stage sequence, or feedback quantization. Token equality therefore remained
an insufficient gate, exactly as the frozen full-model result indicated.

## K128 / K16 contract result

The replay first zeroes all but a requested subset of original F32 activation
values, then invokes the existing block-local quantizer and both unchanged
projection routes. Results are observed after CK's real BF16 workspace
boundary. All per-prefix and per-block vectors are retained under
attempt-3/diagnostic/.

- The 136 cumulative K128 prefixes are non-monotonic: prefixes 1–5 are exact,
  prefix 6 first differs, prefix 8 is exact again, and the full 136-block
  result differs. This is an observation; the reason for cancellation after
  the BF16 boundary is **unconfirmed**.
- Isolating a single K128 block finds the first mismatch at block 1
  (K=128–255): 1 / 5,120 differs at output 1,986, max abs 9.536743e-7.
  Fifteen isolated K128 blocks differ in total. Therefore the discrepancy is
  already present before association among different K128 blocks.
  K128-to-K128 scale accumulation is not its sole cause.
- Within isolated block 1, cumulative K16 prefixes 1 through 7 (K=16…112)
  are bitwise exact. Adding prefix 8 (K=128, the sub-tile at offsets 112–127)
  first gives the same 1-element 9.536743e-7 discrepancy.
- The one-hot lane probe for source-K lanes 0–15 and the first output tile is
  16 / 16 bitwise exact. It rules out a gross transpose/lane error for that
  probe only; it does not establish the opaque fragment mapping for the
  eighth K16 issue or the complete output tile.

Thus the measured difference is inside one K128 block, at or when its eighth
K16 WMMA contribution is incorporated. The exact cause is **not confirmed**:
it may be the final K16 operand/fragment mapping, the WMMA reduction/issue
association, or both. It is not valid to attribute it uniquely to a K128
scale application order.

## Static CK comparison

Read-only static evidence is in STATIC-ANALYSIS.md. At the source level, CK
clears a per-scale accumulator, runs its XDL/WMMA operations for that
ScaleBlockK=128, then adds raw × (activation scale × weight scale) to the
FP32 C accumulator. The handwritten body expresses the same high-level
K128-before-scale intent. CK's selected down form is a 256-thread 16x128x256
block, while the private body is a 32-thread, N=16-tile wave. The inspected
CK gfx1201 object has interleaved WMMA and FP32-FMAC register sequences; the
private body serially issues eight WMMA operations then materializes an opaque
rocWMMA fragment through LDS.

This proves a material schedule/fragment-contract mismatch, but it does not
decode the unique CK lane/register permutation needed for a safe replacement.
No contract-aligned handwritten implementation was made, so post-alignment
performance is **unmeasured**.

## Service isolation and device record

Three stop/isolate/restore cycles occurred. attempt-1 and attempt-3 were
valid; attempt-2 is explicitly excluded above. The final valid window used
only AMD SMI GPU 2, R9700 gfx1201, BDF 0000:47:00.0; HIP_VISIBLE_DEVICES=1
was used for the diagnostic. The V620 was not used.

For attempt-3, the service was active/running with NRestarts=0 before the
window, inactive/dead at 09:19:30, and restored with a single start at
09:20:45; it was active/running with NRestarts=0 at 09:20:46.
llama-qwen35-udq4.service remained inactive/disabled and gdm3 inactive. The
no-process sentinel was observed before GPU execution.

Telemetry snapshots record edge/hotspot/memory temperatures of 38/38/36 C
before stop, 46/47/46 C after the diagnostic, gfx clocks of 2,833 MHz and
49 MHz respectively, socket power of 16 W and 14 W, and throttle states
UNTHROTTLED at both endpoints. The immediately post-stop snapshot was
THROTTLED (38/38/36 C, 822 MHz, 23 W). The physical cause of throttle state
changes is **unconfirmed**. These were numerical diagnostics, not timing
measurements.
