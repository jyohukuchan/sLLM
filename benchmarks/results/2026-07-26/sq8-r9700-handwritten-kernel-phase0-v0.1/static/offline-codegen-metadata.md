# Offline code-generation and source audit

## Method and evidence boundary

The exact HIPRTC source strings used by the handwritten runtime were compiled in an isolated `/tmp` workspace with `--offload-arch=gfx1201 --std=c++17 -O3`; ELF metadata was read from the resulting HSACO. `quantize_activation_block128` and `bf16_to_f32` were compiled from the same source file in an isolated HIP device compile. The selected CK projection bodies are precompiled archive objects rather than HIPRTC strings, so their matching `gfx1201` code objects were extracted read-only from `/opt/rocm-7.2.1/lib/libdevice_gemm_operations.a` and inspected with the same ELF/ISA tools.

No production build tree or `/opt/ullm` file was written. Source/archive checksums at audit time:

| input | SHA-256 |
|---|---|
| `runtime/src/sq8_ck_gfx1201.hip.cpp` | `508546381f8e6357d502a56ca5c85e1a5d72c8b81779a65ca24338d687cc435c` |
| `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` | `4643860b68ef498baefc4dbf29529919a851b10123c2cfb349aba0a7091b3b8a` |
| `runtime/src/ullm_runtime_hiprtc_sources.inc` | `ad050da7137df13ba7f088099085f1539f9f1ea818e525306fc91e9a4f032b57` |
| CK archive | `b813d872442a9ff7485106a80218dc3a842d6c8102c6cc324f4871ea1fb7bd61` |

## Static resources

The values below are compiler/code-object metadata, not achieved hardware occupancy. All listed probes report zero spills/private segment.

| target | compilation route | LDS | VGPR | SGPR | wave | observation |
|---|---|---:|---:|---:|---:|---|
| `quantize_activation_block128` | isolated HIP | 516 B | 13 | 18 | 32 | source has `absolute_values[128]` plus `block_scale` |
| `bf16_to_f32` | isolated HIP | 0 B | 4 | 7 | 32 | scalar conversion helper |
| `ullm_segmented_rmsnorm_f32_kernel` | HIPRTC | 1024 B | 11 | 20 | 32 | selected decode and prefill helper |
| `ullm_paged_decode_attn_f32_kernel` | HIPRTC | 1024 B | 25 | 52 | 32 | selected decode hot path |
| `ullm_cached_prefix_attn_f32_flash2_kernel` | HIPRTC | 1292 B | 21 | 46 | 32 | selected prefill hot path |
| generic `ullm_sq_fp8_matvec_f32_kernel` | HIPRTC | 1024 B | 19 | 47 | 32 | not observed in either trace |
| generic `ullm_sq_fp8_matvec_batch_f32_kernel` | HIPRTC | 1024 B | 19 | 52 | 32 | not observed in either trace |
| generic `ullm_sq_fp8_matvec_pair_f32_kernel` | HIPRTC | 1024 B | 18 | 43 | 32 | not observed in either trace |
| generic `ullm_sq_fp8_matvec_triple_f32_kernel` | HIPRTC | 1024 B | 18 | 50 | 32 | not observed in either trace |

For the matching CK archive objects, exact offline metadata is:

| selected CK form | LDS | VGPR | SGPR | nominal LDS-only ceiling with 256-thread CTA |
|---|---:|---:|---:|---|
| `Default` 128x128, VmemReadVec 8 | 18,432 B | 100 | 47 | 3 workgroups/CU = 24 wave32 = 75% of 32-wave reference |
| `KPadding` 128x256, VmemReadVec 16 | 36,864 B | 242 | 48 | 1 workgroup/CU = 8 wave32 = 25% |
| `Default` 128x256, VmemReadVec 16 | 36,864 B | 175 | 46 | 1 workgroup/CU = 8 wave32 = 25% |
| `Default` 256x128, VmemReadVec 8 | 34,816 B | 154 | 49 | 1 workgroup/CU = 8 wave32 = 25% |

The LDS calculation uses the documented 64 KiB/CU and 32-wave reference. It establishes a resource ceiling only; register allocation, block scheduling, cache behavior, and achieved residency remain unmeasured. The two 36,864-B CK forms account for `30.8020%` of selected decode kernel time, so their one-workgroup LDS ceiling is a concrete high-risk projection-replacement design constraint. The 34,816-B form accounts for `5.6574%` of selected prefill kernel time.

ROCprof reports runtime resource fields as well (for example, the selected 36,864-B CK forms report VGPR 248/224 and SGPR 128; Flash2 reports LDS 1536 B, VGPR 24, SGPR 128). They are retained in the raw CSV. This document does not assume why those profiler fields differ from the code-object metadata.

## Reduction inventory

| selected or available body | reduction observed in source | status |
|---|---|---|
| `quantize_activation_block128` | 128-element LDS max tree, barriers for strides 64 through 1 | selected; low-risk shuffle prototype candidate |
| `ullm_segmented_rmsnorm_f32_kernel` | `float partial[256]` plus full CTA sum tree | selected; low-risk shuffle prototype candidate |
| `ullm_cached_prefix_attn_f32_flash2_kernel` | `float reduce[256]` and repeated full CTA trees for score/max/sum | selected; prefill-dominant, numerically higher-risk candidate |
| `ullm_paged_decode_attn_f32_kernel` | default helper uses wave shuffle and a small LDS cross-wave handoff | selected; full LDS tree exists only under `ULLM_PAGED_DECODE_USE_SHARED_REDUCE` |
| four generic reference matvec kernels | `float partial[256]` plus full CTA tree | not selected in either trace |

`ULLM_PAGED_DECODE_USE_SHARED_REDUCE` is injected when `ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE` is set. The environment value for the profile run was not captured, so activation of that fallback is **未確認**; it must not be reported as an observed full-tree decode path.

## Load-width audit

The selected CK objects contain wide buffer loads. Counts of `buffer_load_b128` mnemonics in the matching objects are 12 (Default 128x128), 29 (KPadding 128x256), 29 (Default 128x256), and 24 (Default 256x128). This proves 128-bit vector memory instructions occur in the selected code objects; it does not classify every operand access as a weight access.

By contrast, the generic reference source loads `payload[row_offset + col]` as one byte and its generated ISA contains `global_load_u8`. It is the explicit scalar E4M3FN decode / F32 FMA / LDS-tree implementation described in the investigation, but it is not a currently observed hot path. `quantize_activation_block128` scans activation data rather than projection weights, while paged/Flash2 attention scans F32 KV cache; neither is evidence of a narrow `SQ8_0` weight scan.

Conclusion: applying the prior `AQ4_0` wide-load change to the generic reference kernel is conditionally useful only if future profiling selects that fallback. It is not a measured serving-path lever on this R9700 workload.
