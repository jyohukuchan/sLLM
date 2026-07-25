# Scoped kernel attribution

The raw `rocprofv3` CSV files are intentionally retained beside this summary. The listed time is dispatch-duration sum; asynchronous overlap is not inferred from a sum.

## Steady-state decode

The selected range contains exactly 16 M=1 decode steps after a 1024-token seed and four excluded warm-up decode steps. Its cache window is `1028 -> 1044`, with `21,824` dispatches and `962.034418 ms` summed kernel duration.

| kernel family | calls | summed time | share |
|---|---:|---:|---:|
| `ullm_paged_decode_attn_f32_kernel` (handwritten HIPRTC) | 640 | 490.607067 ms | 50.9968% |
| CK `KPadding`, 128x256 tile | 1,280 | 200.239820 ms | 20.8142% |
| CK `Default`, 128x256 tile | 640 | 96.085611 ms | 9.9878% |
| CK `Default`, 128x128 tile | 2,560 | 89.743369 ms | 9.3285% |
| *CK subtotal (not additional)* | 4,480 | 386.068800 ms | 40.1305% |
| `ullm_matvec_bf16_f32_kernel` (LM head) | 16 | 39.206603 ms | 4.0754% |
| `ullm_segmented_rmsnorm_f32_kernel` (handwritten HIPRTC) | 2,576 | 18.325536 ms | 1.9049% |
| `bf16_to_f32` (handwritten helper) | 4,480 | 7.057437 ms | 0.7336% |
| `quantize_activation_block128` (handwritten helper) | 2,560 | 6.755150 ms | 0.7022% |
| all remaining kernels | 7,072 | 14.013825 ms | 1.4567% |

The final aggregate row is a residual: its call count is not used as a correctness invariant. The exact individual rows, including copies, RoPE, add, KV write, SiLU-mul, and BF16 row, are in `decode/rocprof-attempt2/sq8-decode_kernel_trace.csv`.

The 4,480 CK dispatches equal `16 * 40 layers * 7 projections`; the 2,560 activation quantizations equal `16 * 40 layers * 4`. No `ullm_sq_fp8_matvec_{f32,batch,pair,triple}_kernel` symbol appears in this selected trace. In particular, the generic `matvec_pair` route is not a steady-state decode hot path here.

## M=128 prefill

The selected range contains a 1024-token prompt processed in eight M=128 chunks. It has `134,219` dispatches and `2.918473188 s` summed kernel duration.

| kernel family | calls | summed time | share |
|---|---:|---:|---:|
| `ullm_cached_prefix_attn_f32_flash2_kernel` (handwritten HIPRTC) | 320 | 2.184211467 s | 74.8409% |
| `__amd_rocclr_copyBuffer` | 83,273 | 174.365646 ms | 5.9746% |
| CK `Default`, 128x128 tile | 1,600 | 165.126083 ms | 5.6580% |
| CK `Default`, 256x128 tile | 640 | 165.109642 ms | 5.6574% |
| *CK subtotal (not additional)* | 2,240 | 330.235725 ms | 11.3154% |
| `ullm_paged_kv_write_f32_kernel` | 40,960 | 100.449454 ms | 3.4418% |
| `quantize_activation_block128` (handwritten helper) | 1,280 | 45.594550 ms | 1.5623% |
| `bf16_to_f32` (handwritten helper) | 2,240 | 26.814157 ms | 0.9188% |
| `ullm_segmented_rmsnorm_f32_kernel` (handwritten HIPRTC) | 1,281 | 15.859070 ms | 0.5434% |
| all remaining kernels | 2,625 | 40.943119 ms | 1.4029% |

As in decode, the two residual call counts are deliberately not reported because they combine unrelated launch families. No generic `ullm_sq_fp8_matvec_*` symbol appears in the prefill trace.

## CK / handwritten boundary

`Sq8LayerExecutionProfile::Rdna4W8a8BlockCk` calls `sq8_ck_projection_f32` from `run_projection` and reaches `ullm_sq8_ck_gfx1201_projection`; its body selects `DeviceGemmMultipleD_ABScale` / `DeviceGemmXdlUniversal` in `runtime/src/sq8_ck_gfx1201.hip.cpp`. These are the projection rows labelled CK above.

The same file contains the handwritten `quantize_activation_block128` and `bf16_to_f32` helpers. The actual attention, RMSNorm, RoPE, KV-write, and LM-head helpers come from HIPRTC source strings. The four generic kernels in `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc` remain a reference/fallback implementation in this workload, not an observed serving hot path.
