# Instruction accounting

`static_instruction_totals` in the JSON files counts each emitted instruction once. It is intentionally not labelled "per element": setup and reduction paths have different dynamic trip counts. The per-element figures below are the innermost payload-loop backedge bodies selected by the exact gfx1201 ISA.

| gfx1201 kernel | backedge | static instructions / element | VALU / element | payload loads / element | FP8 conversion | divide-sequence VALU in loop |
|---|---:|---:|---:|---:|---:|---:|
| single | `0x36e0 -> 0x3640` | 28 | 11 | `1 × global_load_u8`, `2 × global_load_b32` | `1 × v_cvt_f32_fp8_e32` | 0 |
| batch | `0x40f8 -> 0x4054` | 29 | 11 | `1 × global_load_u8`, `2 × global_load_b32` | `1 × v_cvt_f32_fp8_e32` | 0 |
| pair | `0x4b24 -> 0x4a88` | 27 | 11 | `1 × global_load_u8`, `2 × global_load_b32` | `1 × v_cvt_f32_fp8_e32` | 0 |
| triple | `0x547c -> 0x53e0` | 27 | 11 | `1 × global_load_u8`, `2 × global_load_b32` | `1 × v_cvt_f32_fp8_e32` | 0 |

The 11 VALU include address/control work as well as one native FP8 conversion, one floating multiply, and one FMA. This is a scalar-byte streaming loop, but it is not an element-by-element software integer-divide loop.

| gfx1201 whole function | VGPR | SGPR | LDS bytes | static VALU | `v_rcp_iflag_f32` | `v_mul_hi_u32` | `v_mad_co_u64_u32` | static barrier signal/wait sites |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 19 | 47 | 1,024 | 105 | 3 | 3 | 4 | 2 / 2 |
| batch | 19 | 52 | 1,024 | 107 | 3 | 3 | 4 | 2 / 2 |
| pair | 18 | 43 | 1,024 | 95 | 2 | 3 | 4 | 2 / 2 |
| triple | 18 | 50 | 1,024 | 95 | 2 | 3 | 4 | 2 / 2 |

The legacy source writes `partial[256]`, then executes one synchronization before the tree and eight tree stages. Therefore it executes **nine workgroup barriers** dynamically. The two static barrier sites in the code object are expected: one site is in the dynamically iterated tree.

## gfx1030 comparison

The gfx1030 specialized single and batch paths are materially different from the gfx1201 generic path:

| target / kernel family | LDS | payload instruction present | conversion implementation | reduction handoff |
|---|---:|---|---|---|
| gfx1201 all four generic kernels | 1,024 B | `global_load_u8` | one native `v_cvt_f32_fp8_e32` per scalar byte | 256-way LDS tree, 9 dynamic barriers |
| gfx1030 single / batch | 32 B | `global_load_dwordx4` (16 B) | manual E4M3 reconstruction (`v_bfe_u32`, `v_and_b32`, `v_cvt_f32_ubyte0`, shifts) | eight wave partials, 1 barrier |
| gfx1030 pair / triple | 1,024 B | no wide payload load in the optimized sense | manual E4M3 reconstruction | legacy LDS tree, 9 dynamic barriers |

The gfx1030 static totals are 302/303 VALU for specialized single/batch and 140/140 for legacy pair/triple. These cannot be divided by an arbitrary scalar element count: the former has a 16-byte load, a nested byte loop, scalar tail fallback, and wave reduction in the same function. The source and ISA do, however, establish the useful qualitative difference: the single/batch specialization removes the full CTA reduction and uses a wide payload load; pair/triple do neither.

The real Qwen3-14B artifact metadata records `scale_kind=2` and `scale_block=128`. For its 256-thread launch, `col` advances by 256, so a source recurrence would advance the scale-column by two blocks per iteration. The emitted baseline gfx1201 loop already advances the prepared scale address without a reciprocal or multiply-high instruction in that loop body.
