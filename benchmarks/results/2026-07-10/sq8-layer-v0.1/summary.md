# SQ8 Qwen3-14B layer 0 M=8

## Contract

- Source: canonical artifact content SHA-256 `2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147`.
- Shape: one Qwen3-14B decoder layer, `M=8`, position offset `0`.
- Optimized profile: dynamic OCP FP8 activation, canonical FP8 weight, CK ABScale, BF16 output converted to F32.
- Reference latency profile: source-correct W8A16 block-2D HIP kernel.
- QKV shares one activation quantization. Gate/up shares one activation quantization. The layer has four quantizations and seven projections.
- Five `ULLM_REQUIRE_HIP_*` guards make non-projection host staging an error. All projection paths are typed. No fallback or timed host staging is allowed.

## Result

- Gate: passed.
- Optimized p50: `0.777319 ms`; p95: `0.793614 ms`.
- Reference p50: `16.530962 ms`; optimized speedup: `21.266652x`.
- Final output: relative L2 `0.003996148`, cosine `0.999992019301`, max absolute error `0.030761719`, non-finite values `0`.
- All 17 intermediate/final tensor checks passed. The largest relative L2 was `0.011561786` at down projection; its cosine was `0.999933386518`.
- All four GPU activation quantizations matched CPU re-quantization byte-for-byte, including F32 scale bits.
- Output health: 40,960 finite values, range `[-2.2922363, 5.060791]`, SHA-256 `c80dfbcd6961fef4012e45904a72d9229ef25f7ef57e0cb69053522441e150a6`.
- Independent CPU oracle elapsed time: `4113.013 ms`.

## Dispatch

- q/k/v/o: `MemV1DefaultTile16x128x128`.
- gate/up at M=8: `MemV1KPaddingTile16x128x256`.
- down at M=8: `MemV1DefaultTile16x128x256`.
- The remaining production ID, gate/up at M=128, passed its real fixture with `MemV1DefaultTile16x256x128`, relative L2 `0.001658324`, and cosine `0.999998624980`.

`layer0-m8.json` is the typed layer result. `dispatch-runtime-validation.json` records real-fixture validation for all four production implementation IDs. `environment.json` records the command, guards, hashes, and device.
