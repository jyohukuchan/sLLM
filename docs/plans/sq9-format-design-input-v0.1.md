# SQ9_0 Format Design Input v0.1

> Status: design input.  This document fixes the v0.1 payload, conversion, special-value, and
> evaluation contracts, but does not authorize an artifact, runtime-kernel, campaign, release, or
> activation implementation.

## 前回の要点

- `SQ8_0` is the existing FP8 E4M3 format.  Its historical implementation plan explicitly makes
  RDNA4 the first target and leaves native RDNA2/V620 FP8 optimization out of scope.
- The local GPU capability reference identifies Radeon Pro V620 as RDNA2/gfx1030 with no WMMA or
  matrix instruction, whereas R9700/RDNA4 has FP8 conversion builtins and wave32 WMMA.
- The current AQ4 decode path already establishes two relevant implementation techniques:
  16-byte-aligned `uint4` wide loads and wave-local shuffle reductions.  Those are useful
  structural precedents, not evidence that an `SQ9_0` kernel is fast.

## 今回の変更点

- The initial exact format ID is fixed as `SQ9_0`: signed IEEE-style E5M3 stored in nine bits.
- `SQ9_0` uses a 128-element byte-plane plus bit-plane payload layout.  It retains exactly nine
  bits per stored element while making both planes and every 128-element tile 16-byte aligned.
- The normative no-scale variant, finite-value encoder policy, RNE rounding policy, and IEEE
  special-value decoder behavior are fixed below.
- A CPU test now exhaustively verifies all 512 sign-inclusive patterns against independent binary16
  semantics.  It passed on 2026-07-26.

## 次の行動

1. Implement a standalone `SQ9_0` pack/unpack reference and a deterministic RNE encoder only after
   the evaluation gates in this document are accepted.
2. Benchmark the no-scale direct decode path first on V620/gfx1030; do not infer its throughput from
   the bit-conversion proof.
3. Compare it against the same-shape `SQ8_0` fallback and `AQ4_0` path using the bandwidth and
   quality gates defined below before considering a runtime integration plan.

## Goal

Define `SQ9_0` as a weight-only signed E5M3 format for GPUs that lack a useful FP8 execution path.
The stored nine-bit E5M3 code must become an IEEE binary16 bit pattern through shifts and masks only:

```text
fp16_bits = (sign << 15) | (exponent << 10) | (mantissa << 7)
          = sq9_0_code << 7
```

The equality on the second line holds when `sq9_0_code` is the validated nine-bit field
`sign:exponent[4:0]:mantissa[2:0]`.  There is no exponent rebiasing, denormal normalization,
codebook lookup, or scale multiplication in the normative `SQ9_0` dequantization path.

The primary hardware hypothesis is Radeon Pro V620/gfx1030: it has no FP8 matrix instruction or
FP8 conversion builtin path in the project capability reference, but can run ordinary integer,
FP16, and FP32 instructions.  `SQ9_0` is not defined as an automatic replacement for `SQ8_0` on
R9700/RDNA4 or a later GPU with native FP8 execution.

## Success Criteria

- The public format ID, bit fields, payload layout, row padding, byte order, and metadata needed by
  an independent reader are unambiguous.
- Every valid stored code maps to the specified binary16 bit sequence with only a shift after code
  assembly; the all-512-pattern CPU proof remains passing.
- The normative format has no block, row, tensor, or per-value reconstruction scale.
- `exp=0` and `exp=31` have IEEE-compatible behavior that retains the shift-only mapping.
- The implementation phase has a falsifiable throughput gate, a KV-cache-inclusive bandwidth
  efficiency metric, and an activation-weighted quality gate rather than a claim based on storage
  density alone.
- Any later artifact records logical/stored shapes, padding, resident bytes, KV-cache bytes,
  dequant execution mode, measured decode TPS, and the calculated bandwidth efficiency.

## Non-Goals

- This document does not implement a quantizer, artifact schema, loader, HIP kernel, GEMV/GEMM,
  benchmark campaign, release, or activation.
- It does not modify `AQ4_0`, `SQ8_0`, existing candidate/release material, `/opt/ullm`, the active
  manifest, or a service lifecycle.
- It does not claim that direct E5M3 rounding has acceptable model quality.  No full-tensor or
  model-level `SQ9_0` quality result exists at this stage.
- It does not make `SQ9_0` a native FP9 arithmetic format.  The target arithmetic remains binary16
  or FP32 after bit construction.
- It does not fold calibrated row compensation into the normative format.  Existing row-scale
  compensation is an effective-weight modification and must remain separately measured.

## Confirmed Inputs And Evidence Boundary

| confirmed item | evidence | implication for `SQ9_0` |
| --- | --- | --- |
| V620 is gfx1030/RDNA2 and has no WMMA/matrix instruction. | `docs/reference/gpu-architecture-capabilities-rocm7.2.1.md`, target-GPU and WMMA sections; local `rocminfo` also lists Radeon Pro V620 as `gfx1030`. | The direct path cannot rely on FP8 WMMA or an FP8 conversion builtin. |
| R9700/gfx1201 has FP8 E4M3 conversion builtins and wave32 WMMA. | Same capability reference, FP8 and WMMA sections. | `SQ8_0` remains the default comparison format on RDNA4; extra `SQ9_0` bytes need measured compensation. |
| `SQ8_0` retains tensor/row/row-block scale layouts and scopes native RDNA2 FP8 optimization out. | `docs/plans/sq8-implementation-plan-v0.1.md`, Naming/Scope sections. | A no-scale path is a material semantic and kernel difference, not merely one extra bit. |
| AQ4 keeps a 16-byte `uint4` wide-load fallback on every architecture, while its width-8 wave32 shuffle production path is limited to gfx1201/RPB=32. | `runtime/src/ullm_runtime_hiprtc_sources.inc`, `aq4_matvec_wide_load_reference_kernel_source()` and `aq4_matvec_kernel_source()`. | `uint4` alignment is a packing precedent; the gfx1201 shuffle speedup must not be extrapolated to V620. |
| The AQ quantizer plan requires bounded chunks, deterministic repeatability tests, activation-weighted follow-up evaluation, and explicit scale/codebook metadata. | `docs/plans/aq-full-quantizer-design-v0.1.md` and `docs/research/quantization-method-survey-2026-07-01.md`. | `SQ9_0` needs the same reproducibility and quality evidence, despite avoiding LUT use in decode. |
| Existing row-scale compensation is a calibrated approximation, not faithful quantization. | `docs/plans/quantizer-row-compensation-plan-v0.1.md`. | A scale/compensation experiment must not be silently called the normative no-scale format. |

Unconfirmed items are deliberately not promoted to facts: V620 `SQ9_0` instruction count, VGPR
occupancy, achieved memory bandwidth, direct E5M3 model quality, source saturation rate, and any
end-to-end TPS advantage are all unmeasured.

## Working Hypotheses

1. On a GPU without a usable FP8 execution path, assembling an E5M3 field and shifting it into a
   binary16 representation can be cheaper than E4M3 rebiasing/denormal handling plus a scale path.
2. The gain can only matter if it outweighs the larger payload: the raw quantized payload is 12.5%
   larger than an unscaled eight-bit payload.
3. E5M3's FP16-matching exponent range may remove the need for a reconstruction scale on typical
   weights, but its three mantissa bits may still make scale-free quantization unacceptable for
   some tensors.  This is a quality hypothesis, not an established result.
4. A 128-element plane tile, rather than a 32-element 36-byte record, is needed to preserve exact
   nine-bit density and make `uint4` wide loads naturally aligned.

## Exact Value Definition

### Fields And Numerical Semantics

`SQ9_0` stores a nine-bit unsigned code `q` in this logical order:

```text
bit:  8       7 ........ 3       2 .... 0
      sign    exponent[4:0]      mantissa[2:0]
```

The exponent bias is 15, exactly matching IEEE binary16.

| exponent | mantissa | `SQ9_0` meaning | binary16 result after `q << 7` |
| --- | --- | --- | --- |
| 0 | 0 | signed zero | signed zero |
| 0 | 1..7 | subnormal: `(-1)^sign * mantissa * 2^-17` | exact binary16 subnormal; binary16 fraction bits 6..0 are zero |
| 1..30 | 0..7 | normal: `(-1)^sign * (1 + mantissa/8) * 2^(exponent-15)` | exact binary16 normal with the same exponent |
| 31 | 0 | signed infinity | signed binary16 infinity |
| 31 | 1..7 | NaN with the three-bit payload in binary16 fraction bits 9..7 | binary16 NaN with that exact payload bit placement |

The finite positive range is from the least positive subnormal `2^-17` to the maximum finite
value `(1 + 7/8) * 2^15 = 61440`.  Binary16 has additional smaller subnormal values, but no
special handling is required: the E5M3 fraction simply lands at binary16 fraction bit 7 or above.

### Decoder Contract

After assembling `q` from the two payload planes, the decoder must perform:

```c
uint16_t fp16_bits = (uint16_t)(q << 7);
half value = bit_cast<half>(fp16_bits);
```

The range check belongs at a serialization/debug boundary; it must not be a per-element decoding
operation.  A reader must reject a malformed code source rather than mask a wider input silently.
The normal hot path receives `q` already assembled from one low byte and one validated high bit.

Converting the resulting binary16 value to the accumulator type is still necessary for arithmetic.
The claim is intentionally narrow: stored E5M3 bits become binary16 bits without a LUT, exponent
arithmetic, denormal branch, or reconstruction scale multiplication.

### Special-Value Decision

`SQ9_0` reserves `exp=31` for IEEE-style infinity and NaN.  It does **not** reinterpret that
exponent as finite range, because doing so would make the shift-only binary16 mapping false.

The encoder policy is different from the decoder policy:

- A static weight quantizer rejects NaN and infinity source values with tensor name and coordinate.
- A finite source value whose rounded magnitude would exceed 61440 is clamped to signed 61440 and
  increments a per-tensor saturation counter; it must never encode `exp=31` as a finite value.
- A decoder nevertheless accepts all 512 bit patterns and preserves infinity/NaN semantics exactly.
  This makes artifact inspection and corruption diagnostics unambiguous without adding a decoder
  special case.
- Signed zero and all 14 signed subnormal patterns are preserved.  There is no flush-to-zero rule.

## CPU Conversion Proof

The conversion premise is tested in
[`tests/test_sq9_e5m3_bit_conversion.py`](../../tests/test_sq9_e5m3_bit_conversion.py).

```text
python3 -m unittest tests/test_sq9_e5m3_bit_conversion.py -v
```

Result on 2026-07-26: 3 tests passed.  The first test enumerated all 512 codes and compared the
shift result to an independent mathematical E5M3 finite-value decoder followed by the host's IEEE
binary16 conversion.  Infinity/NaN payloads are checked as bit patterns instead of being routed
through a host float, and the second test explicitly checks signed zero, the smallest E5M3
subnormal, the smallest normal, both infinities, all 14 NaNs, and the finite maximum.  The third
test proves that out-of-range input is rejected before the shift.

This proves the bit-conversion premise only.  It does not test a GPU compiler's half bit-cast,
packing traffic, numerical model quality, or throughput.

## Payload And Packing Decision

### Adopted `SQ9_0` Layout

The normative matrix payload is row-major with two contiguous planes in one payload object.
For a logical matrix `[rows, cols]`, define:

```text
stored_cols = ceil(cols / 128) * 128
low_plane_bytes  = rows * stored_cols
high_plane_bytes = rows * (stored_cols / 8)

payload[0 .. low_plane_bytes)                         = low-byte plane
payload[low_plane_bytes .. low_plane_bytes+high_plane_bytes) = high-bit plane
```

The payload base is 16-byte aligned.  `stored_cols` is a multiple of 128, so both plane bases,
each row start, and each 128-element tile start are also 16-byte aligned.  The required metadata
is `rows`, `cols`, `stored_cols`, `tile_elements=128`, `plane_layout="lo8_then_hi1"`,
`bit_order="lsb_first"`, and both plane byte counts.

For element `[row, col]` where `col < stored_cols`:

```text
low  = low_plane[row * stored_cols + col]
high = (high_plane[row * (stored_cols / 8) + (col >> 3)] >> (col & 7)) & 1
q    = low | (high << 8)
```

The final padded columns have code zero (`+0`) and are excluded from the logical dot-product
length.  Padding is a storage alignment rule, not an extra quantized value.  The exact physical
payload is `rows * stored_cols * 9 / 8` bytes; any row-tail overhead must be reported as
`stored_elements - logical_elements` and included in resident-byte measurements.

Although 32 logical values are exactly 288 bits = 36 bytes, `SQ9_0` groups four such units for
the physical access tile: 128 values are 128 low bytes plus 16 high bytes = 144 bytes.  This is
still exactly nine bits/value and is divisible by 16.

### `uint4` Decode Access

One 128-element tile uses eight aligned 16-byte `uint4` loads from the low plane and one aligned
16-byte `uint4` load from the high plane: nine 128-bit loads for 144 bytes.  The implementation
must stream the eight low `uint4` values rather than retain an array of all of them in VGPRs.  A
single retained high-plane `uint4` supplies two sign bytes for each sequential low `uint4`.

The initial kernel benchmark must compare two high-plane distribution choices:

1. cooperative `uint4` load followed by wave-lane exchange or small LDS staging; and
2. coalesced per-lane two-byte high-plane reads.

They are bit-identical.  The chosen artifact layout does not assume which one wins before a V620
profile records global-load transactions, VALU instructions, VGPR count, occupancy, and elapsed
time.

### Packing Alternatives Considered

| layout | exact density | `uint4` alignment and traffic | extraction work | decision |
| --- | --- | --- | --- | --- |
| continuous 9-bit stream | 9 bpp | 32-value starts advance by 36 bytes, so start alignment cycles through 0/4/8/12 bytes | each code can cross byte/word boundaries; needs funnel/bit-field extraction | reject |
| 16 values in 18 bytes | 9 bpp | 18-byte starts repeatedly break 16-byte alignment | same cross-boundary issue at a smaller unit | reject |
| contiguous 32-value `lo8[32] + hi1[4]` records | 9 bpp | extraction is simple, but a 36-byte record has the same 0/4/8/12 alignment cycle | one byte plus one bit, but unaligned/cross-record wide loads | reject |
| interleaved 128-value `lo8[128] + hi1[16]` records | 9 bpp | each 144-byte tile is aligned | simple; high bytes are physically adjacent | not selected; a valid future version only if measured superior |
| adopted two full planes with 128-element tiles | 9 bpp plus explicit tail padding | all plane/row/tile offsets are aligned; eight low-plane plus one high-plane `uint4` loads | one low-byte load and one bit extraction per value | adopt |

The plane-versus-interleaved-128 choice has not been benchmarked.  The adopted plane layout is
selected for its unambiguous address algebra, whole-plane streaming, and guaranteed `uint4`
alignment—not an unmeasured claim that it has lower latency.  A wire-incompatible performance
change requires a new exact format ID.

## Scale Decision

### Decision: No Reconstruction Scale In `SQ9_0`

`SQ9_0` has `scale.kind = "none"`.  It directly rounds each finite source weight to E5M3 and
decodes it to the matching binary16 value.  This eliminates scale payload fetches and a scale
multiplication from the normative dequantization path in addition to eliminating a decode LUT.

The rationale is architectural, not a completed quality claim: E5M3 and binary16 share their
five-bit exponent/bias, so a scale is not required to bridge their exponent ranges.  A scale may
still reduce reconstruction error for a tensor distribution, which is why the evaluation plan
keeps it as a separate ablation.

| candidate | metadata cost | extra arithmetic | relationship to the exact format |
| --- | --- | --- | --- |
| no scale | none beyond tensor headers/padding | none | normative `SQ9_0` |
| tensor scale | one scalar per tensor | normally one result/row multiplication | diagnostic only; not `SQ9_0` |
| row scale | `32 / cols` bpp for FP32 scales | normally one result/row multiplication | diagnostic only; calibrated row compensation remains separately labeled |
| K-block scale of `g` values | `32 / g` bpp for FP32 scales; 0.25 bpp at `g=128` | scale each block partial before reduction or equivalent | diagnostic only; defeats the simplest inner loop |

The scale-free candidate must be evaluated against tensor, row, and K-block (`g=128` minimum)
ablation rows.  A quality failure does not authorize silently adding a scale to `SQ9_0`; it calls
for a new exact format definition whose name, payload bpp, and dequantization equation state that
extra operation.

## Quantization And Rounding Decision

The first encoder uses deterministic round-to-nearest, ties-to-even (RNE) over the finite E5M3
set after finite-range clamping.  RNE is selected because the existing AQ conversion design
requires deterministic repeated output, manifest provenance, and re-read verification.  The AQ
documents do not establish an `SQ9_0` quality result, so this is a reproducibility decision rather
than an assertion that RNE is universally optimal.

| policy | decision | reason |
| --- | --- | --- |
| RNE | adopt | deterministic artifacts, stable metric comparisons, no per-weight random-state metadata |
| stochastic rounding | defer to a controlled ablation | may require seed/version provenance and makes byte-exact repeatability harder; no local evidence yet shows a model-quality need |
| sequential error feedback/compensation | do not use in the initial encoder | creates order/chunk dependence and does not preserve independent per-weight encoding; evaluate only with a separately specified objective |
| calibrated row correction | retain as a separate post-dequant experiment if needed | existing evidence classifies it as an effective-model modification, not faithful quantization |

The encoder must record source non-finite count, saturation count, signed-zero count, normal versus
subnormal code counts, and the RNE implementation/version.  Its verification must pack, reread,
decode, and compare selected source chunks without materializing a full model.

## Decode Cost And Bandwidth Trade-off

### Per-Value Work

For each value, `AQ4_0` requires packed-nibble extraction plus a codebook lookup and group/tensor
scale handling.  `SQ8_0` stores one byte but its E4M3-to-binary16 fallback needs an exponent-format
conversion and may use scale metadata.  `SQ9_0` needs a low-byte read, a high-bit extraction, an
OR, and the binary16 bit shift.  It removes the table/scale work but adds bit-plane assembly and
reads more bytes.

The precise instruction count is unconfirmed until HIP compiles the target kernels.  In particular,
the half bit-cast and arithmetic conversion must be inspected in generated ISA; this document does
not equate a source-level shift with zero dequantization instructions.

For a 128-value tile, the comparison is structurally:

| format | raw weight bytes / 128 values | representative decode-side metadata/action |
| --- | ---: | --- |
| `AQ4_0` | 64 bytes of nibbles | group scale-index reads, scale-table read, codebook lookup, scale multiply |
| `SQ8_0` | 128 bytes before scale metadata | E4M3 conversion/rebias fallback and configured scale layout |
| `SQ9_0` | 144 bytes | eight low-plane + one high-plane `uint4` loads, bit extraction, shift/bit-cast |

`AQ4_0` values in the table exclude its scale bytes because the existing AQ policy mixes group sizes;
the measured package-size comparison below includes its actual metadata estimate.

### KV-Cache-Inclusive Theoretical Limit

The existing AQ4 roofline note is a weight-read lower bound that intentionally ignored the small
KV term for its then-tested contexts.  To satisfy the present comparison requirement, `SQ9_0`
extends that indicator with an explicit KV-cache term rather than mislabeling the historical value
as KV-inclusive.  Use this KV-inclusive lower-bound indicator for a matched workload:

```text
D = resident_weight_bytes_read_per_generated_token + kv_cache_bytes_read_per_generated_token
TPS_theoretical = theoretical_memory_bandwidth_bytes_per_second / D
decode_bandwidth_efficiency = measured_decode_TPS / TPS_theoretical
```

`D` is an explicitly declared lower-bound traffic model, not a claim that caches, activations,
output writes, metadata, or kernel launches cost zero.  For the verified V620 geometry of 8
self-attention layers, 4 KV heads, K/V dimensions of 256, and f32 K/V storage, one cache position
is `8 * 4 * (256 + 256) * 4 = 65,536` bytes.  At decode context `C`, use:

```text
kv_read_bytes(C)  = 65,536 * C
kv_write_bytes    = 65,536
D(C)              = resident_weight_bytes_read_per_generated_token
                    + kv_read_bytes(C) + kv_write_bytes
```

The actual reader may fetch more traffic.  A result row must state `C`, the K/V dtype/layout, and
whether a profiler-derived actual read count is also available; it must use the same policy for all
formats.

The following calculation is a storage-only planning estimate using facts already recorded for the
Qwen3.5-9B AQ plan:

- total source tensor bytes: `19,306,216,416`;
- pass-through payload bytes: `5,049,777,120`;
- therefore targeted BF16 matrix elements: `(19,306,216,416 - 5,049,777,120) / 2 = 7,128,219,648`;
- `AQ4_0` p4p46 estimated output bytes: `9,121,922,016`;
- the current SQ input workload records `50,331,648` bytes (48 MiB) of KV allocation, which equals
  `65,536 * 768`; the conservative `C=768` per-token model below adds one 65,536-byte KV write;
- R9700 theoretical bandwidth reference point: `640 GB/s = 640,000,000,000 bytes/s`.

`SQ8_0` below is deliberately an optimistic no-scale lower bound, not an observed `SQ8_0` artifact.
It favors `SQ8_0`; actual scale metadata can only increase its traffic.  Neither `SQ8_0` nor
`SQ9_0` padding/header bytes are included, so an implementation must replace these with measured
resident payload bytes.

| format/storage row | package bytes used in `D` | `D(768)` with 48 MiB KV read + 64 KiB write | theoretical TPS at 640 GB/s | status |
| --- | ---: | ---: | ---: | --- |
| `AQ4_0` p4p46 estimate | 9,121,922,016 | 9,172,319,200 | 69.78 | existing storage estimate, not a matched `SQ9_0` benchmark |
| `SQ8_0` no-scale lower bound | 12,177,996,768 | 12,228,393,952 | 52.34 | favorable theoretical baseline only |
| `SQ9_0` no-scale estimate | 13,069,024,224 | 13,119,421,408 | 48.78 | format-size estimate only |

Within only the targeted quantized matrices, `SQ9_0` is exactly `9/8 = 1.125` times the `SQ8_0`
payload.  Across the stated package, pass-through tensors reduce the total-byte increase to 7.32%;
at `C=768`, the `SQ9_0` theoretical limit is 93.21% of the favorable no-scale `SQ8_0` limit.  Thus:

```text
required effective-bandwidth/overhead improvement over no-scale SQ8_0
  > (D_sq9_0 / D_sq8_0) - 1
  = 7.29% for this whole-package + 48 MiB-KV estimate

required improvement when all relevant weights are eight-bit versus nine-bit
  > 12.5%
```

The added 891,027,456 package bytes cost at least 1.392 ms/token at a hypothetical full 640 GB/s
stream and about 1.762 ms/token at 79% of that bandwidth.  Those figures isolate only the added
weight traffic; they are not a TPS prediction.  `SQ9_0` wins only when eliminating fallback
conversion/scale overhead and improving realized traffic efficiency saves more than that cost.

No V620 peak-bandwidth fact is asserted here: the local capability reference leaves its bandwidth
field unspecified.  An older AQ4 roofline note used a 512 GB/s V620 assumption, but that number is
not revalidated by this design.  The V620 acceptance benchmark must first measure an appropriate
streaming bandwidth baseline on the actual device, then use that measured/reference bandwidth in
the same equation.

## Position Relative To Existing Formats

| exact format | stored value payload | nominal payload bpp | reconstruction metadata | direct decode cost | hardware position | quality position |
| --- | --- | ---: | --- | --- | --- | --- |
| `AQ4_0` | 4-bit index | 4 bpp; existing policies include group-scale overhead | codebook plus group/tensor scale policy | nibble unpack, LUT, scale handling | primary compact path; current kernels use wide loads and wave reductions | calibrated existing policy; quality remains gated |
| `SQ8_0` | FP8 E4M3 | 8 bpp before scale metadata | tensor/row/row-block scales | native FP8 path where available; fallback conversion otherwise | preferred on R9700/RDNA4, whose FP8 builtins/WMMA are documented | existing format with its own policy and regression gates |
| `SQ9_0` | signed E5M3 in two planes | exactly 9 bpp before row-tail padding | none | low-byte + high-bit assembly, shift/bit-cast; no LUT/rebias/scale | primary experiment for V620/RDNA2 and other no-FP8-path GPUs | unmeasured; direct RNE must pass the planned quality gates |

`SQ9_0` has a plausible performance advantage over `SQ8_0` only on an architecture where E4M3
cannot use native FP8 conversion/arithmetic and the fallback is materially conversion/scale-bound.
V620/RDNA2/gfx1030 is the named primary case.  It is not expected to beat `SQ8_0` by default on
R9700/RDNA4 or a later native-FP8 GPU: `SQ8_0` has lower weight traffic and documented native FP8
support there.  On either class, the explicit bandwidth-efficiency and matched-quality gates—not
the format label—decide the winner.

## Kernel Implementation Implications

### What Can Be Reused

- The resident-payload loader lifecycle, sidecar/manifest validation patterns, backend dispatch,
  direct matvec launch plumbing, shape guards, and model-loop telemetry conventions can be adapted
  from the existing AQ4/SQ8 work.
- The AQ4 M=1 organization supplies relevant discipline: 16-byte alignment proofs, streaming
  `uint4` loads, row-tail guards, and a comparison of LDS versus wave-local reduction.  The current
  width-8 shuffle result is gfx1201/RPB=32-specific, so V620 must measure its own reduction choice.
- The benchmark result must keep direct execution distinct from any FP16/F32 materialized fallback,
  following the existing `SQ8_0` reporting rule.

### What Requires New Code

- `SQ9_0` payload validation, plane offsets, padding logic, CPU reference pack/unpack, and RNE
  encoder are new; no AQ4 nibble payload or `SQ8_0` byte payload reader is wire-compatible.
- The dequant hot loop is new: it eliminates codebook/scale streams but combines the low byte and
  bit-plane byte before the binary16 bit-cast.
- The kernel must not retain eight low-plane vectors plus the sign vector at once.  The initial
  target is one current `uint4`, four sign dwords, a small number of activation values, and the
  accumulator(s), followed by ISA/VGPR verification.
- The direct decode kernel must derive its wave width from the compilation/runtime target rather
  than assume CDNA wave64.  RDNA2 and RDNA4 tuning must be separate configurations.
- Prefill/GEMM is deferred.  V620 has no WMMA, so its first `SQ9_0` implementation is a direct
  decode/GEMV path; it must not pretend to have a native FP9 matrix instruction.

### Required Kernel Measurements

For every direct-kernel candidate, record:

- aligned versus deliberately guarded tail paths and their exact shape coverage;
- global bytes/read transactions, VALU instruction count, VGPR count, LDS use, occupancy, and
  wave width from the profiler/compiler output;
- per-projection and model-loop decode time with the same source tensors, context, and KV policy;
- `D`, `TPS_theoretical`, and `decode_bandwidth_efficiency` from the preceding section; and
- CPU/HIP numerical agreement plus a no-materialized-fallback assertion.

## Design Questions And Evaluation Plan

| question | required experiment | pass decision |
| --- | --- | --- |
| Does no-scale direct E5M3 preserve useful quality? | bounded chunk reconstruction, activation-weighted relative MSE, saturation/subnormal counts, then golden-prefix/logit and prompt-suite checks | no-scale `SQ9_0` must meet the predeclared quality floor; otherwise it is not promoted |
| Does a scale recover enough quality to be worth it? | tensor, row, and K=128 scale ablations with identical RNE source and exact metadata bytes | report as a separate experimental exact format; never relabel it as `SQ9_0` |
| Is the plane layout efficient on V620? | isolated 128-tile decode microbenchmark for both high-plane distribution choices, plus unaligned-36-byte control | chosen layout must retain aligned transactions and beat or match the control without higher VGPR pressure |
| Does `SQ9_0` beat `SQ8_0` fallback? | same source/model subset, output quality, context, KV policy, and direct-only dispatch | measured `SQ9_0` TPS and bandwidth efficiency must exceed matched `SQ8_0`; otherwise retain `SQ8_0` |
| Is the full model path useful versus `AQ4_0`? | identical full-package workload and memory accounting | require quality pass and a documented memory/performance trade-off; `SQ9_0` is not expected to win on resident bytes |

## Decision Tree

```text
source weight finite?
  no  -> reject quantization with tensor/coordinate evidence
  yes -> clamp finite magnitude to 61440; deterministic RNE to E5M3

artifact declares exact SQ9_0 plane metadata and zero-valued tail padding?
  no  -> reject reader input
  yes -> assemble low byte + high bit; fp16_bits = q << 7

target has a validated native FP8 E4M3 path?
  yes -> SQ8_0 is the default comparison; require SQ9_0 to overcome its byte penalty empirically
  no  -> run the SQ9_0 direct FP16/FP32 decode microbenchmark

no-scale SQ9_0 passes quality and direct-kernel gates?
  no  -> stop; evaluate explicitly named scale-bearing experiments only if separately authorized
  yes -> run matched V620 model-loop and full-package comparisons

all quality, bandwidth-efficiency, resident-memory, and fallback-disclosure gates pass?
  no  -> keep SQ9_0 as an unpromoted design candidate
  yes -> write a separate implementation plan; no activation follows from this document
```

## Risks

| risk | impact | handling |
| --- | --- | --- |
| three mantissa bits cause unacceptable direct-rounding error | no-scale quality fails despite exponent range | activation-weighted and model-level gates precede implementation promotion |
| high-bit plane extraction erases the expected conversion gain | `SQ9_0` loses to `SQ8_0` fallback | profile ISA/VGPR/transactions and compare both sign-distribution methods |
| 12.5% quantized-payload increase is bandwidth-dominant | lower theoretical decode ceiling | enforce the calculated crossover threshold and report KV-inclusive efficiency |
| row-tail padding grows storage on unusual shapes | actual bpp exceeds 9 | record logical/stored elements and reject an unjustified tail policy |
| a scale is silently reintroduced during quality tuning | the shift-only claim becomes misleading | reserve no scale for `SQ9_0`; require a new exact ID for every scale-bearing variant |
| NaN/inf is treated as finite range | corrupts IEEE mapping or hides source defects | reserve exp=31 and reject non-finite source weights |
| RDNA2 assumptions are copied from RDNA4 | wrong wave/ISA or unsupported FP8/WMMA path | compile and profile per target; keep V620 direct path independent of WMMA |
| historical `SQ8_0` results are compared as if same workload | misleading performance conclusion | require matched source, context, KV policy, direct execution mode, and quality gate |

## Next Actions

1. Add a CPU reference pack/unpack test for the adopted two-plane layout, including 1/31/32/33/127/
   128/129-column tails, 16-byte offsets, and zero padding.  Do not change a runtime reader yet.
2. Implement a bounded-memory RNE encoder prototype that records finite/saturation/subnormal
   telemetry and re-reads chunks for verification.
3. Run no-scale, tensor-scale, row-scale, and K=128-scale quality ablations on a fixed tensor set
   using the AQ project's activation-weighted and prompt/logit evidence conventions.
4. Build an isolated V620/gfx1030 M=1 direct decoder with both high-plane distribution choices;
   compare it to a matched `SQ8_0` fallback and collect ISA/VGPR/profiler evidence.
5. If the microbenchmark and quality gates pass, write a separate `SQ9_0` implementation plan for
   loader/manifest integration, direct projection coverage, model-loop telemetry, and regression
   tests.  Any artifact/campaign/release/activation decision remains a later, separately approved
   action.

## 2026-07-26 Offline Evaluation Result: discard SQ9_0

This section is the disposition of the design input above. It supersedes the earlier SQ9_0
evaluation and next-action direction; it does not rewrite the historical format definition.

### Scope and evidence boundary

All work in this section was CPU-only or offline compilation. No HIP runtime API, GPU kernel launch,
R9700, V620, service, release, candidate, or activation state was used or changed.

The source-correct reference is the local Qwen/Qwen3-14B-FP8 checkpoint. Each source value was
reconstructed as OCP F8_E4M3FN payload times its BF16 [128,128] weight_scale_inv multiplier.
The fixed, depth-balanced sample contains the Q/K/V/O and gate/up/down projections from layers 0,
20, and 39: 21 real tensors and 990,904,320 reconstructed weights. This is a substantial real
model sample, not a claim of a full-280-tensor measurement. Error is incremental relative to the
already-FP8 source reconstruction; error relative to the unavailable pre-FP8 training checkpoint
is unconfirmed.

Raw error evidence is in
../../benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/quantization-error/.
Raw gfx1030 code object, disassembly, compiler metadata, checksums, and count output are in
../../benchmarks/results/2026-07-26/sq9_0-vs-q8_0-offline/isa/.

### Static gfx1030 ISA result

The probes have a fixed K=128, are fully unrolled at -O3, and compile with
--offload-arch=gfx1030. They do not execute. The Q8_0 label below is the conventional
int8-plus-block-scale baseline, not a uLLM public format ID. The proposed uLLM name is given
later.

| profile | emitted work per 128 weights | VALU / total instructions | VGPR / SGPR | LDS / private / spill |
| --- | --- | ---: | ---: | --- |
| Q8_0-style W8A8, g=32 | 32 v_dot4c_i32_i8, 4 int32-to-f32 conversions, 4 scale multiplications, 4 scale FMAs | 67 / 96 | 41 / 54 | 0 / 0 / 0 |
| SQ9_0 W8A8, g=32 activation scale | 128 int8-to-f32 conversions, 132 FP32 mixed FMAs, 514 emitted bitfield-or-shift instructions | 813 / 985 | 28 / 18 | 0 / 0 / 0 |
| Q8_0-style W8A16, g=32 | 128 int8-to-f32 conversions, 128 weight-scale multiplications, 128 FP32 mixed FMAs | 399 / 458 | 57 / 18 | 0 / 0 / 0 |
| SQ9_0 W8A16 | 128 FP32 mixed FMAs, 514 emitted bitfield-or-shift instructions | 648 / 704 | 58 / 18 | 0 / 0 / 0 |

The raw count output divides the named instruction classes by 128. Therefore, the decisive W8A8
result is 0.25 dot4 instruction per element for the Q8_0-style path versus SQ9_0's 1.0
int8-to-float conversion plus 1.03125 FMA/mixed-FMA instructions per element, before its
4.015625 bitfield-or-shift instruction class per element. The bitfield-or-shift class includes
some address arithmetic; the raw opcode histogram is retained so this is not misrepresented as
plane assembly alone. It nevertheless shows the direction unambiguously: the supposed shift-only
conversion does not compile into a shift-only inner loop.

For W8A16, the Q8_0-style path does pay one int8-to-f32 conversion, one scale multiplication, and
one mixed FMA per element. It is still not tied with SQ9_0: 399 emitted VALU instructions versus
648, with essentially the same VGPR use and zero spills in both probes. Thus preserving FP16
activations removes the direct dot advantage but does not reverse the static ALU result.

The packed-FP16 companion probes provide the prefill arithmetic-rate check. Both produce 64
v_pk_fma_f16 instructions for 128 weights, or 0.5 packed-F16 FMA instruction per element. The
Q8_0-style packed W8A16 probe additionally emits 128 int8 conversions; the SQ9_0 packed probe
emits 432 bitfield-or-shift instructions. In the W8A8 form relevant to an int8 activation path,
v_dot4c_i32_i8 performs four MACs per instruction while a packed FP16 FMA performs two. This is
an ISA structural advantage for the int8-dot path, not a measured GPU throughput claim.

The compiler metadata reports no LDS, private memory, VGPR spill, or SGPR spill for any of the six
probes. The absence of a spill does not rescue SQ9_0 because its W8A8 and W8A16 instruction mixes
remain materially larger. Measured V620 timing, achieved bandwidth, occupancy, and end-to-end TPS
remain unconfirmed because GPU execution is prohibited for this task.

### Real-weight reconstruction error

| format or ablation | persistent payload bpp | relative L2 | relative MSE | mean absolute error | maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| SQ9_0, E5M3 no scale | 9.000000 | 0.0265181390 | 0.000703211698 | 0.000407516100 | 0.0625000000 |
| Q8_0-style, signed int8 + FP16 scale per 32 | 8.500000 | 0.00562448658 | 0.0000316348493 | 0.000108919572 | 0.00988006592 |
| signed int8 + FP16 scale per 128 ablation | 8.125000 | 0.00696264838 | 0.0000484784724 | 0.000136232033 | 0.00988006592 |

The required Q8_0-style g=32 comparison is 4.7148 times lower in relative L2 and 22.2290 times
lower in error SSE than SQ9_0. No source value saturated either format in this sample. SQ9_0
rounded 130,179 nonzero values to zero and used 2,139,523 subnormal codes; these are observations,
not a model-level acceptance result.

Block distribution does affect int8 quality as expected, but not enough to change the conclusion.
The Q8_0-style g=32 blocks with max-absolute-value/RMS in [1,2), [2,4), and [4,8) cover
10.744%, 88.932%, and 0.325% of sampled values respectively. Their Q8_0-style relative MSE is
0.0000187986, 0.0000325908, and 0.000113299; SQ9_0 is respectively 37.78x, 21.52x, and 7.55x
higher in those bins. No sampled block fell in the two bins at or above 8. This is evidence for
the block-scale trade-off on actual weights, not an inference from a synthetic distribution.

### Byte allocation and the existing SQ8_0 source contract

The actual source SQ8_0 contract uses one BF16 scale per [128,128] block. Its physical scale
overhead is 16 / (128 * 128) = 0.0009765625 bpp, so its source-correct resident payload is
8.0009765625 bpp before container metadata. The Q8_0-style g=32 payload is 8.5 bpp, and SQ9_0 is
9 bpp.

Consequently, Q8_0-style g=32 saves 0.5 bpp, or 5.56% of persistent weight bytes, versus SQ9_0.
SQ9_0 spends one full bit per weight on its E5 exponent range, whereas a FP16 scale amortized over
32 weights spends only 0.5 bpp. The measured 22.229x SSE improvement shows that this exchange is
efficient for this source sample. The g=128 ablation saves another 0.375 bpp but worsens error,
which supports g=32 as the initial int8 design point.

SQ8_0 remains smaller than either alternative and preserves the source payload plus its source
scale without an extra requantization error. Q8_0-style g=32 is 6.237% larger than source-correct
SQ8_0; SQ9_0 is 12.486% larger. Therefore this decision does not replace SQ8_0 on RDNA4, where
the existing source-preserving FP8 path remains the correct default. It only rejects SQ9_0 as the
no-native-FP8 fallback proposed for gfx1030-class hardware.

### W8A8 activation contract and quality boundary

An int8 weight format gains its decisive dot instruction only when activations are also quantized.
The initial design must use symmetric dynamic int8 activations per token and contiguous K=32
block, with RNE code q_a = clamp(round(a / s_a), -127, 127) and an FP16 scale
s_a = max(abs(a)) / 127. For each output row and K block, the kernel computes an int32 dot and
applies s_w times s_a once to that block partial. It must not apply one scale after reducing across
multiple K blocks, because the weight scale changes per block.

This task has no retained raw activation corpus suitable for measuring activation quantization
error, so W8A8 activation relative L2, saturation rate, and output/logit impact are unconfirmed.
They must be measured with held-out prompt activations before any implementation adoption. This
does not weaken the SQ9_0 disposition: the static W8A16 comparison already favors the int8
block-scale path, and W8A8 adds the verified dot4 advantage.

### Decision and replacement direction

Decision: discard SQ9_0 as a runtime, artifact, or campaign candidate. Retain this document only
as a rejected design record and conversion proof. Do not implement its packer, reader, GPU kernel,
candidate, release, or activation path.

The next candidate should be named SQ8_1. This is a proposal only, not a format-registry change:
it follows the existing SQ8 category and exact-version naming convention while remaining
wire-incompatible with SQ8_0.

| field | proposed SQ8_1 direction |
| --- | --- |
| values | row-major signed int8, one byte per logical weight |
| weight scale | one FP16 positive dequantization multiplier per contiguous K=32 weights; shape [rows, ceil(cols / 32)] |
| quantization | symmetric RNE q_w = clamp(round(w / s_w), -127, 127), s_w = max(abs(w)) / 127, with zero tails only for physical K padding |
| persistent density | 32 int8 values plus one FP16 scale = 34 bytes = 8.5 bpp before row-tail/container metadata |
| W8A8 | dynamic per-token, K=32 signed-int8 activation plus FP16 scale; v_dot4c_i32_i8 into int32 partials and one scale product per K block |
| W8A16 fallback | int8-to-float conversion plus weight-scale multiplication per value; required where activation quantization is not accepted |
| SQ8_0 relationship | separate artifact and dispatch; preserve SQ8_0 raw F8_E4M3 plus BF16 [128,128] scale source contract and retain it on native-FP8 paths |
| AQ4_0 relationship | unchanged compact format; SQ8_1 is a higher-fidelity / int8-dot fallback candidate, not an AQ4_0 replacement |

### Replacement next actions

1. Write a separate SQ8_1 design input and format-ID review; do not change the existing SQ8_0 or
   AQ4_0 registry, artifacts, dispatch, campaigns, candidate files, releases, or service state.
2. Add a bounded-memory CPU reference with pack/unpack, 1/31/32/33-column tails, deterministic
   RNE, FP16-scale underflow/overflow accounting, and source-correct F8_E4M3-to-SQ8_1 error checks.
3. Define an activation capture and held-out evaluation plan for the K=32 dynamic W8A8 scale,
   including activation saturation, activation relative L2, linear-output error, and logit/prompt
   gates. Until then W8A8 quality is unconfirmed.
4. Only under a separately authorized GPU window, compare SQ8_1 W8A8, SQ8_1 W8A16, and the retained
   SQ8_0 fallback on matched V620 inputs. Record transactions, occupancy, clocks, timing, and
   numerical differentials; do not infer those measurements from this offline ISA evidence.

## V620 (gfx1030) 実機測定結果（2026-07-26、サーマルガード付き再実行）

この節は設計本文を変更せず、同日に行った再実行の実測を追記するものである。対象は
AMD Radeon Pro V620、`gcnArchName=gfx1030`、PCI BDF `0000:03:00.0`（DRM `card0`）だけである。
HIP の可視 ordinal と DRM card 番号を混同しないため、ベンチマークは
`hipDeviceGetPCIBusId` を `/sys/class/drm/card*/device` と照合し、その一致カードの
`hwmon/hwmon5/temp2_input`（`temp2_label=junction`）を読んだ。R9700 では測定を実行していない。

前回の card1（`0000:43:00.0`）で junction 100 C / 148 W に達した結果は履歴として保存したが、
ここでの集計からは除外した。今回は junction `>= 85 C` を中断閾値とし、各 warmup と各 timed
launch の前後に温度を採取し、各測定点を `<= 42 C` から始めた。使用した有効データの最高値は
M=128 の 51 C で、85 C ガードおよび cooldown timeout は発動しなかった。M=512 は旧呼び出しで
`--shape` が漏れて全 shape へ進み始めたため、58–59 C / 42 C 復帰待ち 43 C で SIGKILL して終了した。
partial M=512 ログは残すが、以下の性能判断には使用しない。以後の実装は full suite に
`--shape` と `--m-values` の両方を明示要求して fail closed とした。

測定形状は Qwen3-14B FP8 `self_attn.q_proj`、5,120 x 5,120 である。実装から再現した
`SQ8_0` は F8 E4M3 payload 26,214,400 B と 128x128 row-major BF16 scale 3,200 B の artifact
であり、V620 fallback の resident scale は F32 なので合計 26,220,800 B である。`SQ9_0` は
low plane 26,214,400 B と high/sign plane 3,276,800 B、計 29,491,200 B であった。従って実際の
fallback resident layout に対する `SQ9_0` の増分は 12.4725408836% である。これは scale を含めた
実数であり、単純な 12.5% 仮定ではない。両 `SQ9_0` high-plane 実装、`SQ8_0`、FP16 reference は
タイミング前に CPU 参照との GPU 正当性確認を通過した。

M=1 は 32 warmup、31 timed trial の独立 3 run の run-median 中央値で比較した。各 run の
開始は 41–42 C、終了は 42–43 C、ピークは 43 C だった。帯域値はベンチマーク内の modeled weight
stream であり、512 GB/s reference に対する比率はプロファイラ実測の物理トランザクション率ではない。

| M=1 path | median ms | modeled GB/s | 512 GB/s 比 | `SQ8_0` 比 throughput |
| --- | ---: | ---: | ---: | ---: |
| `SQ8_0` E4M3 + F32 block scale fallback | 0.639007 | 41.034 | 8.014% | baseline |
| `SQ9_0` lane high byte | 0.612567 | 48.144 | 9.403% | +4.316% |
| `SQ9_0` cooperative LDS high plane | 0.602446 | 48.952 | 9.561% | +6.069% |
| FP16 raw reference | 0.589446 | 88.946 | 17.372% | +8.408% |

したがって、M=1 では両 `SQ9_0` path とも raw elapsed time は `SQ8_0` より短いが、最速でも
+6.069% に留まり、全 package + KV の採算条件 +7.29% を満たさない。decode replacement としては
このデータで `SQ9_0` を採用しない。

バッチ条件では lane `SQ9_0` に条件付きの優位性が観測された。M=8 は二つの 15-trial run median の
平均で `SQ8_0` 1.241054 ms 対 0.989531 ms（+25.418%、peak 45/46 C）、M=32 は一つの 9-trial run
で 5.861702 ms 対 4.631769 ms（+26.554%、peak 48 C）、M=128 は限定した 3-trial run で
24.630720 ms 対 19.803831 ms（+24.374%、peak 51 C）だった。M=2–7、M=512 の統計、他 projection、
固定 clock、quality、KV/context を含む model loop は未測定である。このため、M>=8 の結果は
batched microbenchmark に限った条件付きの観測であり、decode 結論や format promotion には用いない。

`SQ8_0` fallback dequant には大きな ALU/control 成分の証拠がある。guarded dequant-only では
`SQ8_0` が 0.245603 ms、同 kernel の non-load-only raw control が 0.110122 ms であった。gfx1030
ISA の `dequant_sum_kernel` は `SQ8_0` の static `v_*` 377 本に対し lane `SQ9_0` は 250 本である。
lane `SQ9_0` は unroll 16 value ごとに 16 本の `v_lshlrev_b16`（仕様の `q << 7`）と 16 本の
`v_cvt_f32_f16` を持つ。この静的・分離測定は `SQ8_0` fallback の変換/scale 負担を支持するが、
full M=1 GEMV が純粋に ALU 律速であることを単独では証明しない。

生データ、温度履歴、ISA resources/disassembly の再現手順、完全な制約は
`benchmarks/results/2026-07-26/sq9-v620-viability/summary.md` とその `static/isa-analysis.md` に保存した。
この追記は candidate、campaign、release、service、activation の承認ではない。

