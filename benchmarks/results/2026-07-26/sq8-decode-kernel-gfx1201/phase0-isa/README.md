# SQ8_0 generic matvec — Phase 0 ISA audit

This directory is the no-GPU Phase 0 audit for the exact HIPRTC source in
`runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc`.

| input | value |
|---|---|
| source SHA-256 | `1feb3582f128ec67a23ddb0cc9d88ec075ea8f5bd35a8fbd8be91b048fdb554c` |
| compiler | ROCm HIP 7.2.53211 / AMD clang 22.0.0git (full version in `compiler-version.txt`) |
| HIPRTC-equivalent flags | `--offload-arch=<arch> --std=c++17 -O3` |
| audited targets | `gfx1201`, `gfx1030` |
| GPU work in this phase | none |

The source was extracted by `tools/extract-sq8_0-hiprtc-source.py` and compiled
as a device-only translation unit. The `.hsaco`, `.disasm`, `.notes.txt`, and
`.ll` files are retained for both targets. `*.summary.json` is a mechanically
generated static instruction/resource summary; `SHA256SUMS` records every input
and retained artifact.

## Phase 0 conclusion (before Phase 1)

The proposed *per-element 64-bit software divide* diagnosis is **not supported
by the final gfx1201 ISA**. LLVM IR still contains `udiv i64` inside the
source-level loop (for example `gfx1201.ll:58` and `:60`), but the backend
strength-reduces the invariant-divisor expression before emitting the hot loop.
The actual element backedge for the single kernel is
`0x36e0 -> 0x3640`; its 28 static instructions contain 11 VALU instructions,
one `global_load_u8`, two `global_load_b32`, one
`v_cvt_f32_fp8_e32`, one `v_mul_f32_e32`, and one `v_fmac_f32_e32`.
It contains **zero** `v_rcp_iflag_f32`, `v_mul_hi_u32`, or
`v_mad_co_u64_u32` instructions. The corresponding inner loops of batch,
pair, and triple are also 11 VALU per scalar element and contain none of those
divide-sequence instructions.

The reciprocal/multiply-high sequences do exist in the gfx1201 code object,
but in scale/setup paths outside that element backedge: static whole-function
counts are three `v_rcp_iflag_f32` / three `v_mul_hi_u32` for single and batch,
and two / three for pair and triple. They are setup cost, not a dynamic
software divide once per `col` value. Thus a hand-written recurrence is still
semantically valid, but Phase 0 does not predict a material speedup from it:
the compiler already produced the intended recurrence-like hot loop.

`ullm_sq_fp8_e4m3fn_to_f32` is a single native
`v_cvt_f32_fp8_e32` in each gfx1201 element loop. The payload is already a
`global_load_u8` and is passed directly to that conversion; no payload
mask/shift exists in the hot loop. Consequently the packed-word byte-selector
proposal is inapplicable to this path.

See [instruction-accounting.md](instruction-accounting.md) for exact counts and
the gfx1030 comparison, and [roofline-and-dispatch.md](roofline-and-dispatch.md)
for the serving-path and roofline conclusion.
