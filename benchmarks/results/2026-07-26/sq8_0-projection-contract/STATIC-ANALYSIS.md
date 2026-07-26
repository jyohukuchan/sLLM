# Read-only CK / rocWMMA contract evidence

This note records static evidence used to interpret the valid attempt-3
measurement. It does not modify /opt/ullm, the CK body, or any served artifact.

## Inputs

| input | SHA-256 |
| --- | --- |
| /opt/rocm-7.2.1/include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp | 548579485eb36f4f500ee784c438d2fe8bb98b4205215ef92dc5e362ca57e9e8 |
| /opt/rocm-7.2.1/lib/libdevice_gemm_operations.a | b813d872442a9ff7485106a80218dc3a842d6c8102c6cc324f4871ea1fb7bd61 |
| runtime/src/sq8_handwritten_gfx1201.hip.cpp | 0c15cb1f9dc98286ddc84386879db172ff4b9166598cd46af68f81d6d987cc35 |

The inspected archive member was
device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp.o.
The ROCm LLVM tools used were under /opt/rocm-7.2.1/llvm/bin/; temporary
extraction/disassembly files were kept outside this repository.

## What is confirmed

- crates/ullm-runtime-sys/build.rs supplies CK_USE_OCP_FP8=1 to the CK
  compilation. The selected object metadata names ck::f8_ocp_t. rocWMMA
  defines float8_t as hip_fp8_e4m3. Thus a FNUZ-versus-OCP FP8 encoding
  mismatch is not supported by this evidence.
- CK source lines 647–650 clear c_thread_buf_per_scale for each scale-K block.
  Lines 652–678 issue the XDL operation into that raw buffer, and lines
  680–690 add the raw partial multiplied by its scale to the FP32 C buffer.
  This confirms the high-level K128 raw-then-scale contract.
- The handwritten body keeps a FragC raw_block for each K128 block, issues
  eight 16-wide mma_sync calls (lines 120–148), then obtains its selected
  result with store_matrix_sync and multiplies it by the K128 scale (lines
  150–156). It performs a BF16 round trip only at output (line 161).
- The CK default code object contains gfx1201 v_wmma_f32_16x16x16_fp8_fp8
  sequences followed by FP32 v_fmac_f32 scale accumulation. Its metadata
  permits 256-thread blocks; the selected down configuration is
  DeviceGemmXdlUniversal block size 256, tile 16x128x256 in
  runtime/src/sq8_ck_gfx1201.hip.cpp. The handwritten body launches 32
  threads for an N=16 tile.

## Boundary of the conclusion

The code-object observation shows a different register/issue schedule, and
the dynamic K16 result locates the first observable mismatch when the eighth
16-wide contribution of an isolated K128 block is included. It does **not**
decode an exact mapping between CK XDL registers and the rocWMMA fragment
stored by the handwritten body. Consequently neither a unique lane mapping
nor a unique reduction-association explanation is asserted here.

An exact compatibility implementation would need that mapping plus a new
component and full-model gate. It must not be inferred from a synthetically
passing one-hot probe.
