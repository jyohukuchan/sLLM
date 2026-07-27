# Offline ISA verification (ROCm 7.2.1)

The standalone source was built with `hipcc -O3 --save-temps` for both target
architectures and its generated HSACO was disassembled with LLVM objdump.

| target | required observed instruction | count | BF16/FP8 resource metadata |
|---|---|---:|---|
| gfx1201 | `v_wmma_f32_16x16x16_fp8_fp8` | 1 | BF16: VGPR 24/SGPR 20; FP8: VGPR 20/SGPR 20; LDS 0, private 0, no spills, wave32. AGPR is not reported for this RDNA4 code object. |
| gfx942 | `v_mfma_f32_16x16x32_fp8_fp8` | 1 | BF16/FP8: VGPR 16/SGPR 20/AGPR 0; LDS 0, private 0, no spills, wave64. |

Also observed: gfx1201 `v_wmma_f32_16x16x16_bf16`; gfx942
`v_mfma_f32_16x16x16_bf16`. This is a static resource check, not a substitute
for target-GPU runtime occupancy; the runner records the limitation explicitly.

After the CP STREAM repair, both fresh audits additionally extract the exact
`stream_read_kernel` symbol. Each has two global/flat load instructions and
zero global/flat atomic instructions. The gfx942 rerun retained one
`v_mfma_f32_16x16x32_fp8_fp8`; this check is run before rental timing as well
as in the target's result directory.
