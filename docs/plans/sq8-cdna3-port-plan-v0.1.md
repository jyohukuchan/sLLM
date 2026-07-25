# Independent SQ8_0 CDNA3 (gfx942) Port Plan v0.1

- Status: Phase 0 complete for source/static evidence; all execution, numerical, and timing claims remain blocked on a physical gfx942 device
- Date: 2026-07-26
- Scope: Qwen3-14B-FP8's independent `SQ8_0` execution path on CDNA3 (`gfx942`: MI300X / MI300A / MI325X)
- Boundary: this is not the Qwen3.5 `AQ4_0` legacy 48-QKV/Z tensor overlay lineage. No `AQ4_0` implementation, campaign, candidate, release, or activation is in scope.

## Goal

Add a separately selected CDNA3 implementation for independent `SQ8_0` while preserving the existing gfx1201/RDNA4 external ABI, profile contract, and dispatch behavior byte-for-byte for that architecture.

The performance candidate is a native CDNA3 FP8 MFMA projection path, not a mechanical conversion of RDNA4 WMMA. Its input contract must remain the canonical `SQ8_0` artifact: raw OCP E4M3FN weights and `[128,128]` block scales. Any FNUZ representation is an internal, derived cache/prepack and must never replace or mutate the artifact.

The work follows the already established safety sequence:

1. isolated prototype;
2. offline HIPRTC/HIP compilation plus VGPR/SGPR/LDS/ISA audit;
3. physical-gfx942 differential and timing validation; and
4. only then, an internal production symbol-body replacement behind the unchanged external ABI/dispatch.

Phase 0 has completed only steps that do not require a gfx942 device. This host has gfx1030 and gfx1201 GPUs only; it must not be used to infer gfx942 numerical correctness or performance.

## Success Criteria

### Correctness and format

- A CDNA3 projection computes the same `SQ8_0` logical operation as the canonical OCP E4M3FN payload plus its `[128,128]` scale grid, including block boundaries, tails, zero handling, and the model's actual M shapes.
- A byte-level OCP-to-FNUZ prepack oracle is frozen before MFMA integration. It must cover all 256 payload bytes, every scale conversion, real-artifact byte histograms, non-finite rejection, and the OCP negative-zero (`0x80`) normalization rule.
- Each native FP8 MFMA kernel has a CPU/reference-oracle differential and then a real-gfx942 differential for every Qwen3-14B projection shape and the decode/prefill M set.
- A `dequant-to-FP16` (or BF16 where the chosen CDNA3 primitive requires it) path exists as a correctness/bring-up control. It is not accepted as a substitute for proving the native path.

### Architecture isolation

- Existing public C signatures in `runtime/include/ullm_runtime.h`, the Rust calls that use them, the gfx1201 worker protocol profile, and the current gfx1201 execution selection remain unchanged.
- `rocm-ck-gfx1201`, `runtime/src/sq8_ck_gfx1201.hip.cpp`, and their gfx1201 code path are not retargeted. CDNA3 gets a separate build feature, source/body, cache key, and internal profile selected only after exact `gfx942` detection.
- The architecture selector uses the actual HIP `gcnArchName` (or an equally exact verified source), not a synthesized major/minor string. An unrecognized gfx9 target fails closed to the existing non-optimized/reference behavior rather than accidentally loading gfx940 code for gfx942.
- Every new CDNA3 kernel is compiled for wave64 and audited for its own lane map, shuffles, LDS layout, and resource metadata. No wave32 fragment, reduction, or WMMA helper is reused by assumption.

### Static and physical gates

- Before a gfx942 device is available, a candidate must compile in an isolated directory, emit the intended `v_mfma_*` ISA, report no spills, record VGPR/SGPR/AGPR/LDS and compiler occupancy, and keep a source-to-ISA evidence manifest.
- On the first physical device, the gate is: exact device/SKU/firmware/ROCm/partition manifest; `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`; deterministic kernel and end-to-end differentials; sanitizer/error checks where available; thermal-steady timing; and profiler counters for HBM/L2/XCD behavior.
- A production-body change is eligible only if native FP8 MFMA meets the numerical gate and wins or ties the predeclared performance decision against the dequant control on the same real device and partition. Otherwise the CDNA3 optimized profile remains unavailable; it must not silently route to an unvalidated fast path.

## Non-Goals

- Do not change `/etc/ullm/served-models/active.json`, `ullm-openai.service`, any `SQ8_0`/`AQ4_0` campaign, authorization, candidate, release, or anything under `/opt/ullm`.
- Do not alter the existing release/build trees, the existing gfx1201 external ABI/dispatch, `llama-qwen35-udq4.service`, or `gdm3`.
- Do not claim real-gfx942 numerical correctness, achieved occupancy, HBM bandwidth, or tokens/s from an offline code object.
- Do not treat the independent `SQ8_0` path as the old Qwen3.5 `AQ4_0` overlay path, and do not import its P3 shuffle/WMMA prototypes into this port without a separately scoped proof that the independent `SQ8_0` route actually selects them.
- Do not overwrite the canonical OCP payload with FNUZ data. A derived FNUZ prepack is disposable/cacheable implementation state only.

## Phase 0 Findings

### Environment and evidence boundary

The local toolchain is ROCm 7.2.1 / HIP `7.2.53211-e1a6bc5663`. `/opt/rocm/llvm/bin/llc -march=amdgcn -mcpu=help` lists `gfx940`, `gfx942`, and `gfx950`, and `/opt/rocm/amdgcn/bitcode/oclc_isa_version_942.bc` is present. `rocminfo` lists local gfx1030 and gfx1201 devices, but no gfx942 device. All Phase 0 compile products were written under unique `/tmp/ullm-sq8-cdna3-*` directories, never into a project build/release tree.

The ROCm 7.2.1 headers establish that `__gfx942__` selects FNUZ FP8, while gfx1200/gfx1201 select OCP FP8 (`hip/amd_detail/amd_hip_fp8.h`). `rocwmma/internal/config.hpp` selects wave64 mode for gfx942. `rocwmma/internal/mfma_impl.hpp` provides gfx942 `float8_fnuz_t` MFMA specializations for `16x16x32` and `32x32x16`; it does not make the independent `SQ8_0` OCP payload directly acceptable as an MFMA fragment.

### Architecture-dependent source inventory

The inventory below is a source reading of the independent `SQ8_0` route. “No literal WMMA/MFMA” means that the handwritten source does not name such an intrinsic; it does not prove what a CK template would emit after a full supported instantiation.

| Area | File and symbol(s) | Observed dependency | CDNA3 consequence |
|---|---|---|---|
| Worker admission | `crates/ullm-engine/src/sq8_worker_backend.rs`: `require_sq8_worker_build_feature`, worker load path | Requires `rocm-ck-gfx1201`; validates one visible HIP device, R9700 identity, device 0. | Add a distinct CDNA3 admission/profile path; preserve current gfx1201 branch unchanged. |
| Serving admission | `crates/ullm-engine/src/sq8_serving_runtime.rs`: `validate_qwen3_14b_sq8_r9700_device_info` call sites | Stack/model/head construction and component comparison revalidate exact R9700 assumptions. | Factor a private architecture capability check, retaining the R9700 predicate for the existing profile. |
| Worker protocol | `crates/ullm-engine/src/sq8_worker_protocol.rs` | Emits `device="gfx1201"` and `execution_profile="rdna4_w8a8_block_ck"`. | Introduce a new CDNA3 profile value only; do not reinterpret the existing values. |
| Worker runtime | `crates/ullm-engine/src/sq8_worker_runtime.rs` | CPU thread/channel orchestration only; no GPU intrinsic, wave, or LDS code was found. | No kernel port, but profile/admission wiring will need a CDNA3 case. |
| Model head / embedding | `sq8_model_head_runtime.rs`: R9700 gate and `Rdna4W8a8BlockCk` binding; `sq8_embedding_runtime.rs`: corresponding gates | Exact `AMD Radeon Graphics`, gfx1201, compute 12.0, and R9700-memory-range assumptions. | Add a separate CDNA3 capability contract and tests; do not relax the R9700 contract globally. |
| Layer dispatch | `crates/ullm-engine/src/sq8_layer_runtime.rs`: `Sq8LayerExecutionProfile`, `run_projection` | Only `ReferenceW8a16Block2d` and `Rdna4W8a8BlockCk`; optimized route performs four activation quantizations and seven projections/layer. Weight blocks are 128x128. | Add a third, CDNA3-specific internal profile with its own report/validation counts. |
| Stack dispatch | `crates/ullm-engine/src/sq8_stack_runtime.rs` | Optimized checks require 280 CK projections and 160 activation quantizations; optimized methods hard-code the RDNA4 profile. | Add CDNA3-specific reporting without changing RDNA4 expected counts or fallback semantics. |
| Cargo build gate | `crates/ullm-runtime-sys/build.rs` | Feature `rocm-ck-gfx1201` rejects `GPU_ARCH != gfx1201` and builds only `runtime/src/sq8_ck_gfx1201.hip.cpp` with OCP-FP8 CK flags. | A separate CDNA3 feature/body is mandatory. Passing `GPU_ARCH=gfx942` to this feature is deliberately invalid. |
| Existing optimized projection | `runtime/src/sq8_ck_gfx1201.hip.cpp`: `DeviceOp`, measured `DeviceGemmXdlUniversal` strings, `validate_device`, `ullm_sq8_ck_gfx1201_projection` | CK `DeviceGemmMultipleD_ABScale` uses `ck::f8_t`, tile contracts `1,128,128`, four measured gfx1201 instances, 256-thread blocks, and rejects anything except gfx1201/compute 12.0. The source contains no literal WMMA intrinsic. | It is an RDNA4-only implementation, not a retargetable source. Design a native CDNA3 MFMA body and its own tile/fragment layout. |
| Existing activation quantizer | same file: `quantize_activation_block128` | 128-thread CTA, `float absolute_values[128]` plus `float block_scale` in LDS (516 B), CTA tree strides 64 to 1. On RDNA4 it is four wave32; on CDNA3 it would be two wave64. | Revalidate/rewrite independently; the reduction is CTA-correct in shape but not a proof of wave64 performance or data-layout correctness. |
| Existing CK ABI body | `runtime/src/ullm_runtime_api_sq8_ck.inc`: `sq8_ck_projection_implementation`, `ullm_runtime_sq8_ck_quantize_activation_f32`, `ullm_runtime_sq8_ck_projection_f32` | Public/stable call shape enforces measured M set and N/K multiples of 128 and calls the gfx1201 helper under `ULLM_RUNTIME_ROCM_CK_GFX1201`. | Keep signatures and Rust callers stable; select the CDNA3 body/cache only inside a private exact-architecture implementation selector behind these calls. |
| Generic reference matvec | `runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc`: `ullm_sq_fp8_matvec_f32_kernel`, `_batch_`, `_pair_`, `_triple_` | All four are scalar OCP-FP8-to-F32 decode plus F32 dot/FMA. `__builtin_amdgcn_cvt_f32_fp8` is guarded to gfx1200/gfx1201; gfx942 takes the manual OCP E4M3FN decoder. Each uses `__shared__ float partial[256]` (1024 B) and a full-CTA tree. No WMMA, MFMA, or literal shuffle appears here. | This path can be an offline/reference control but is not the native FP8 matrix route. A 256-thread launch becomes four wave64 instead of eight wave32. |
| Generic HIPRTC launch/cache | `runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc`: four `HipSqFp8MatvecKernelCache` variants and four launcher functions | All variants use `block_size = 256` and `hip_arch_candidates(device_id)`. | Cache identity must include an exact architecture/implementation variant; 256 threads is a deliberate CDNA3 tuning variable, not an inherited optimum. |
| HIPRTC architecture selection | `runtime/src/ullm_runtime_parts/part_00.inc`: `hip_arch_candidates`, `compile_kernel` | Compiler accepts `--offload-arch=<arch>`, but the current major=9 branch constructs `gfx9<minor>0`; this could form `gfx940` rather than verified `gfx942`. | Replace synthesized gfx9 selection with exact arch-name handling and unit tests before enabling CDNA3. |
| Generic block-2d ABI | `runtime/src/ullm_runtime_api_sq8_0.inc`: block2d matvec path | Uses the 2-D `[ceil(N/128), ceil(K/128)]` scale layout; unlike some generic paths, block2d has no host-staging fallback if HIP fails. | Preserve the scale contract and add explicit CDNA3 error/fallback behavior before route selection. |
| Actual SQ8_0 paged decode | `runtime/src/ullm_runtime_hiprtc_sources.inc`: paged decode kernel using `lane=tid%warpSize`, `wave=tid/warpSize`, `__shfl_down(..., warpSize)`; `part_01.inc` launch | Uses dynamic `warpSize`, not literal 32, but is launched as a 256-thread CTA and shares LDS reductions with the fast path. Independent `SQ8_0` uses F32 Flash2 cached-prefix attention, not the unrelated FP8 rocWMMA cached-prefix path. | Compiles plausibly for wave64, but must receive real-device differential/timing; CTA changes from eight wave32 to four wave64. |
| Other selected SQ8_0 helpers | `ullm_runtime_hiprtc_sources.inc`: segmented RMSNorm, model-head BF16 matvec, top-1, F32 Flash2; launch sites in `part_00.inc`/`part_01.inc` | Several use 256-element LDS trees (`partial`, `reduce`, and top-1 float/u32 arrays); embedding and KV write are scalar/no-LDS. | Audit all selected kernels as wave64/CTA/LDS work, even though they have no literal wave32 intrinsic. |

The P3 `AQ4_0` wave-shuffle reduction group is not selected by this independent `SQ8_0` execution route and is therefore excluded from the above port inventory. If a future CDNA3 implementation elects to reuse it, that reuse starts as a new wave64 design/review rather than an inherited fact.

### Offline gfx942 build classification

The production HIPRTC source string was extracted and compiled through HIPRTC with the exact relevant options: `--offload-arch=gfx942 --std=c++17 -O3`. It produced a 23,440-byte gfx942 HSACO successfully. ELF metadata and generated ISA gave the following results.

| Existing target / probe | Compile result | VGPR / SGPR / AGPR / LDS | Wave / compiler annotation | Meaning |
|---|---|---:|---|---|
| `ullm_sq_fp8_matvec_f32_kernel` | success | 20 / 54 / 0 / 1024 B | 64 / `Occupancy: 8` | Scalar software OCP decode, F32 FMA, LDS tree; no native FP8 matrix ISA. |
| `ullm_sq_fp8_matvec_batch_f32_kernel` | success | 20 / 59 / 0 / 1024 B | 64 / `Occupancy: 8` | Same classification. |
| `ullm_sq_fp8_matvec_pair_f32_kernel` | success | 19 / 52 / 0 / 1024 B | 64 / `Occupancy: 8` | Same classification. |
| `ullm_sq_fp8_matvec_triple_f32_kernel` | success | 19 / 56 / 0 / 1024 B | 64 / `Occupancy: 8` | Same classification. |
| `sq8_ck_gfx1201.hip.cpp` copied to an isolated device compile with its current flags | compile success only | quantizer: 12 / 22 / 0 / 516 B; BF16 conversion: 5 / 12 / 0 / 0 B | 64 / `Occupancy: 8` | Not a valid CDNA3 projection result: runtime/build gates reject gfx942 and the standalone TU did not instantiate the CK GEMM body. |
| gfx1201 WMMA intrinsic probe | gfx942 compile failure | — | requires `gfx12-insts,wavefrontsize32` | Direct RDNA4 WMMA port is invalid. |
| `__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8` FNUZ probe | success | 6 / 14 / 0 / 0 B | 64 / `Occupancy: 8` | ISA contains `v_mfma_f32_16x16x32_fp8_fp8`. |
| `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` FNUZ probe | success | 18 / 14 / 0 / 0 B | 64 / `Occupancy: 8` | ISA contains `v_mfma_f32_32x32x16_fp8_fp8`. |
| rocWMMA `float8_fnuz_t` 16x16x32 probe | success | 7 / 14 / 0 / 0 B | 64 / `Occupancy: 8` | Confirms the header route lowers to native MFMA. |
| rocWMMA OCP `float8_t` equivalent probe | compile failure | — | — | OCP FP8 is not an admissible direct gfx942 rocWMMA MFMA input. |

All four existing HIPRTC kernels have zero VGPR/SGPR spills and zero private segment. `block_size = 256` therefore represents four wave64s/CTA statically. The compiler's `Occupancy: 8` annotation is a resource-derived code-generation result, not measured achieved occupancy, active CTAs/CU, or latency hiding. The latter requires `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` and profiler evidence on the actual target device.

The successful existing HIPRTC build is classified as **compile-pass, semantic-runtime-unverified, and performance-inappropriate for the native FP8 goal**. Its disassembly has scalar `v_fmac` and LDS traffic, but no `v_mfma`, `v_wmma`, or native FP8 conversion. The copied CK compile is **compile-pass only / semantic-runtime-invalid** for the same reason: it is intentionally guarded to gfx1201 and is not evidence that its GEMM template runs on CDNA3.

### Native FP8 MFMA and `SQ8_0` format compatibility

The canonical `SQ8_0` artifact specification is OCP E4M3FN raw bytes plus BF16 `[128,128]` block multipliers. The existing Rust decoder in `crates/ullm-engine/src/sq.rs` implements that OCP semantics. By contrast, the verified gfx942 MFMA forms accept FNUZ FP8 fragments. Directly reinterpreting the artifact byte stream is not correct:

- OCP `0x7f`/`0xff` are NaNs whereas FNUZ interprets those bit patterns as finite values; canonical artifacts must reject non-finite payload values rather than rely on reinterpretation.
- OCP `0x80` is negative zero whereas FNUZ `0x80` is NaN.
- Enumerating all bytes showed that for the 254 finite OCP codes, after normalizing `0x80` to FNUZ `0x00`, `OCP(raw) = 2 * FNUZ(mapped_raw)` holds exactly. The scale multiplier must therefore become `2 * scale` for each converted weight or activation; multiplying two converted operands makes the post-MFMA scale product four times the original pair's product.

This yields a conditional native route, not direct compatibility:

| Candidate | Advantages | Required proof / cost | Decision status |
|---|---|---|---|
| A. FNUZ-prepacked native FP8 MFMA | Uses proven gfx942 `v_mfma_f32_16x16x32_fp8_fp8` or `32x32x16` with FP32 accumulation; avoids elementwise FP16 dequant before matrix work; likely performance leader if packing and memory traffic are controlled. | Derived immutable weight cache with OCP-to-FNUZ mapping; FNUZ activation quantizer; `0x80`/non-finite/scale-range artifact gate; fragment/lane map; correct per-128-K-block scale application; real-device differential. | **Performance primary, conditional on the format gate.** |
| B. Dequant-to-FP16/BF16 then matrix work | Retains the canonical OCP decoder and is the simpler numerical/control path; useful for staging and oracle comparison. | Extra conversion/bandwidth and no native FP8 MFMA benefit; CDNA3 matrix primitive/tile still needs its own validation. | **Bring-up correctness control, not the performance primary.** |

`[128,128]` scale blocks align arithmetically with both candidate MFMA K dimensions: four `K=32` iterations or eight `K=16` iterations per K block, and 16/32 divide the 128-row/column boundaries. That is only a block-boundary observation. The current row-major weight `[N,K]` can be linearly reinterpreted as a `B[K,N]` column-major storage view, but the MFMA fragment loader/lane mapping is still unverified. Since a weight scale changes at every 128-K boundary, the kernel must scale each partial accumulator before adding across scale blocks; applying one scale after an entire K reduction would be wrong.

## Working Hypotheses

1. **H1 — CDNA3 must use MFMA, not RDNA4 WMMA.** The static WMMA failure explicitly requires gfx12 wave32 features; a verified gfx942 FNUZ probe emits the target MFMA instructions on wave64.
2. **H2 — FNUZ prepack is feasible but not yet validated on real artifacts.** The finite-code mapping and scale relation are verified offline. Presence of `0x80`, scale-overflow/underflow behavior, actual payload tails, and end-to-end numerical error remain unconfirmed until the prepack oracle runs against the actual artifact and a gfx942 executes it.
3. **H3 — A separate MFMA kernel is safer than attempting to retarget CK XDL instances.** The current CK instances were measured for gfx1201 and encode tile/WaveMap/CTA assumptions. An isolated CDNA3 implementation makes ISA, lane layout, and resource changes auditable and leaves RDNA4 untouched.
4. **H4 — Wave64 affects all fixed-CTA work even where source has no literal `32`.** The 256-thread scalar/reference and attention helpers change from eight wave32 to four wave64; the 128-thread quantizer changes from four to two. CTA LDS trees may remain functionally valid but have no demonstrated CDNA3 scheduling/performance behavior.
5. **H5 — A 256-thread CTA is a starting control, not a CDNA3 tuning decision.** Candidate MFMA tiles must evaluate 64/128/256-thread CTA shapes, register pressure, LDS staging, and compiler-reported occupancy offline; the winner cannot be chosen without hardware timing.
6. **H6 — CDNA3's bandwidth model must be partition-specific.** MI300X, MI300A, and MI325X share gfx942 but differ in HBM and XCD configuration. AMD's partition documentation identifies six XCDs for MI300A and eight for MI300X/MI325X; MI300X's published peak is 5.3 TB/s and MI325X's is 6 TB/s. Those values are SKU maxima, not a substitute for an actual XCD/NPS partition query.
7. **H7 — LDS/L2 topology is a design constraint, not sufficient performance evidence.** Local ROCm profiler metadata lists 16 L2 banks and 32 LDS banks/CU for gfx942; rocWMMA declares a 64 KiB LDS maximum. Bank conflicts, L2 reuse, HBM service, and XCD locality are unconfirmed until profiled on a real partition.

### CDNA3 decode bandwidth accounting contract

Define a fixed logical streaming accounting model before timing:

\[
B_{stream}(L) = B_{weights,payload+scales} + B_{KV,read}(L) + B_{KV,write} + B_{activation/output/page/workspace}
\]

For the current independent `SQ8_0` Qwen3-14B F32 KV cache (`40` layers, `8` KV heads, K/V dimension `128`), its explicit logical terms are:

\[
B_{KV,read}(L) = 40 \cdot 8 \cdot (128+128) \cdot 4 \cdot L = 327{,}680L\ \mathrm{B/token}
\]

\[
B_{KV,write} = 327{,}680\ \mathrm{B/token}.
\]

Thus the logical KV read at context length 4096 is 1,342,177,280 B/token (1.25 GiB/token), before weights and other traffic. `B_{weights,payload+scales}` is measured from the executed 280-projection schedule and exact scale layout, not from total VRAM allocation. The manifest records whether it assumes streaming/no L2 reuse; it must not quietly substitute allocated bytes for traffic.

For an actual SKU and active partition `p`, define the streaming HBM roof and normalized decode metric as:

\[
T_{HBM,stream}^{roof}(L,p) = \frac{BW_{HBM,peak}(\mathrm{SKU},p)}{B_{stream}(L)}
\]

\[
\eta_{decode,stream}(L,p) = \frac{T_{measured}(L,p)\,B_{stream}(L)}{BW_{HBM,peak}(\mathrm{SKU},p)}.
\]

`\eta_{decode,stream}` is a fixed workload-normalized comparison metric, not a claim that every counted logical byte physically crossed HBM. On hardware report it alongside:

\[
BW_{HBM,measured} = \frac{B_{TCC\ counters}}{elapsed},\qquad
\eta_{HBM} = \frac{BW_{HBM,measured}}{BW_{HBM,empirical\ peak}(\mathrm{SKU},p)}.
\]

This separation prevents L2 reuse from being mislabeled as HBM efficiency. The real-device manifest records SKU, HBM generation/capacity, XCD mode (SPX/DPX/QPX/CPX as available), NPS mode, visible CU/XCD count, memory-clock state, and per-XCD/TCC counters. No single `gfx942` denominator is valid for all MI300 variants or partitions.

## Phase Breakdown

### Phase 0 — source, toolchain, and static reconnaissance (complete)

Completed without modifying production code or build/release trees:

1. Read the independent `SQ8_0` Rust, C++/HIP, HIPRTC, artifact, and ROCm header sources summarized above.
2. Confirmed the offline gfx942 toolchain and absence of a physical gfx942 device.
3. Compiled the existing HIPRTC scalar/reference source to gfx942; captured HSACO metadata, wavefront size, compiler occupancy annotation, LDS, spills, and ISA classification.
4. Compiled native FNUZ FP8 MFMA probes and confirmed `v_mfma_f32_16x16x32_fp8_fp8` and `v_mfma_f32_32x32x16_fp8_fp8`.
5. Demonstrated that direct gfx1201 WMMA and direct OCP rocWMMA FP8 forms are not gfx942 paths.

Exit status: enough evidence to implement an isolated CDNA3 prototype; not enough evidence to route a model, validate a number, report runtime occupancy, or make a performance claim.

### Phase 1 — architecture and format controls (offline-only)

1. Add exact `gcnArchName` detection and unit tests for `gfx1201`, `gfx942`, malformed/unknown gfx9, and CPU/reference fallback. Do not use `major==9` string synthesis.
2. Define a private `Cdna3W8a8FnuZMfma` profile/capability object and a separate `rocm-mfma-gfx942` build feature/source. Keep `rocm-ck-gfx1201` and its source intact.
3. Implement a CPU byte oracle for OCP E4M3FN-to-FNUZ mapping and scale transform; add exhaustive 256-byte tests, artifact scan, and scale-range tests. Reject non-finite canonical payload values and normalize only the established OCP negative-zero case.
4. Decide cache ownership/lifetime/invalidation for FNUZ weights so the canonical `SQ8_0` buffers and public ABI remain unchanged. Record cache memory and prepack latency separately from decode timing.

Offline exit gate: exact arch routing tests, byte oracle, artifact scan, and prepack manifest all pass; otherwise only Candidate B may proceed as a control and Candidate A remains disabled.

### Phase 2 — isolated CDNA3 MFMA and control prototypes (offline-only)

1. Create new, non-production HIP/HIPRTC source bodies for FNUZ activation quantization, OCP-to-FNUZ prepack consumption, and projection. No existing gfx1201 body is edited.
2. Implement both `16x16x32` and `32x32x16` variants with explicit fragment/lane documentation. Align tiles to 128-scale boundaries and apply every K-block scale before cross-block accumulation.
3. Implement a separate dequant-to-FP16/BF16 control with the same scale/oracle contract.
4. Compile every model M/N/K shape and tail path to gfx942; inspect ISA to prove MFMA only in Candidate A, then archive VGPR/SGPR/AGPR/LDS/spill/compiler-occupancy metadata.
5. Add architecture-selector and source-string unit tests that ensure gfx1201 still chooses its original body and does not compile/load a CDNA3 source.

Offline exit gate: all code objects target gfx942/wave64, native candidates contain the intended MFMA ISA, no unexplained spills/resource regressions exist, and all host/CPU oracle tests pass. This still does not prove fragment correctness.

### Phase 3 — physical gfx942 differential and residency gate (hardware-required)

Required hardware: an MI300X, MI300A, or MI325X that reports exact `gfx942`; no other local GPU substitutes.

1. Capture the immutable device/partition manifest before every run: `gcnArchName`, HIP/driver/firmware, SKU, XCD mode, NPS mode, visible CUs, HBM clocks/capacity, and process/device isolation.
2. Run exhaustive/sampled kernel differentials against the CPU oracle and the dequant control for all projection dimensions, 128-block boundaries, tail rows, activation values, `0x80`, and representative real weights.
3. Run end-to-end logits/hidden-state differential through prefill and decode at fixed prompts/context lengths, then enforce the predeclared numerical tolerances.
4. Query HIP runtime occupancy/residency for each code object and record achieved waves, LDS/register limits, errors, and clocks. Compare wave64 CTA candidates rather than assuming the static `Occupancy: 8` annotation is delivered.
5. Check XCD/NPS behavior in SPX first, then every mode actually available on the target. Do not generalize MI300X results to MI300A/MI325X or a different partition.

Hardware exit gate: zero unexplained numerical mismatch, stable repeated execution, and a captured partition-specific residency manifest. Failure returns the path to Phase 2; it does not authorize a body swap.

### Phase 4 — physical performance and memory-hierarchy decision (hardware-required)

1. Establish a per-SKU/partition empirical HBM peak and record the benchmark method.
2. Measure prefill/decode at fixed M and context lengths, including weight-prepack warm/cold state, F32 KV cache, cache-hit behavior, HBM/L2 counters, TCC bytes, XCD balance, clocks, temperatures, and power.
3. Publish `B_stream`, `T_HBM,stream^roof`, `eta_decode,stream`, measured HBM bandwidth, and `eta_HBM` under the accounting contract above.
4. Compare Candidate A and Candidate B under the same model, prompt, partition, clock, batch, and correctness condition. The native route wins only if its benefit remains after prepack cost and memory footprint are stated.

Hardware exit gate: Candidate A is numerically valid and is the chosen performance implementation under the predeclared rule, or Candidate B/reference remains the only supported CDNA3 route.

### Phase 5 — guarded production integration (hardware-required after Phases 3–4)

1. Wire the already-validated internal CDNA3 body/profile behind exact gfx942 selection while preserving every existing gfx1201 symbol body/ABI/dispatch result.
2. Re-run gfx1201 regression tests and CDNA3 differentials/timing on the same manifested device/partition.
3. Stage the candidate using existing safety procedures. A production activation, including any byte change to `/etc/ullm/served-models/active.json`, remains a separate human-approved action and is outside this plan.

## Decision Tree

```text
Start: independent SQ8_0 Qwen3-14B-FP8 only
  |
  +-- Does exact HIP arch identity equal gfx942?
  |     |-- no --> preserve current gfx1201 path or reference/fail-closed behavior
  |     `-- yes
  |
  +-- Does the OCP-to-FNUZ prepack oracle pass all bytes, artifact scan, and scale range?
  |     |-- no --> do not feed raw OCP bytes to MFMA;
  |     |          retain dequant-to-FP16/BF16 only as the correctness control
  |     `-- yes
  |
  +-- Does isolated native source emit wave64 gfx942 MFMA with audited resources?
  |     |-- no --> revise tile/layout offline; no production selection
  |     `-- yes
  |
  +-- Is a physical gfx942 device available?
  |     |-- no --> stop at static evidence; no correctness/performance claim
  |     `-- yes
  |
  +-- Do kernel and end-to-end differentials pass on the recorded partition?
  |     |-- no --> disable native profile; return to offline design
  |     `-- yes
  |
  `-- Does native MFMA meet the partition-specific timing/bandwidth decision vs control?
        |-- no --> leave it unselected; keep validated control/reference only
        `-- yes --> guarded internal CDNA3 body integration; gfx1201 path remains unchanged
```

## Risks

| Risk | Why static work cannot close it | Mitigation and gate |
|---|---|---|
| Code written without a gfx942 device is numerically wrong | Compiler/ISA checks do not execute fragments, barriers, loads, or scale arithmetic. | CPU oracle plus real-gfx942 kernel and end-to-end differential before any selection. |
| An implicit wave32 assumption is missed | Source can use fixed 128/256 CTAs or reduction structure without spelling `32`; wave64 changes waves/CTA and scheduling. | Per-kernel wave/CTA/LDS inventory, wave64 rewrite review, HIP occupancy query, and timing/differential on device. |
| MFMA lane/fragment layout is wrong | `v_mfma` in disassembly proves instruction selection, not that rows/columns/scales map to the intended logical tensor. | Fragment-layout specification, boundary-focused oracle vectors, all projection-shape differentials, and ISA review. |
| OCP/FNUZ format mismatch corrupts values | Raw-byte reinterpretation maps OCP NaNs and negative zero differently; scale factor is non-obvious. | Exhaustive byte oracle, artifact scan, private prepack, scale range test, and no mutation of canonical payload. |
| XCD partitioning changes behavior | Offline toolchain does not reveal active XCD/NPS topology, HBM locality, CU visibility, or counter distribution. | Record partition manifest; test every available mode separately; report per-XCD/TCC counters. |
| HBM/L2 accounting is misleading | Logical weight/KV bytes and physical HBM reads differ under L2/cache reuse. | Freeze `B_stream`, report counter-derived HBM bytes separately, and never compare unlike SKU/partition denominators. |
| RDNA4 regression | Relaxing global gates or changing a shared source/cache could retarget existing gfx1201 behavior. | Separate feature/body/cache/profile; exact-arch unit tests; gfx1201 source/ABI/dispatch regression before and after CDNA3 work. |
| Prepack cost exceeds native-math benefit | Weight conversion/cache allocation can dominate small batches or damage residency. | Measure cold/warm prepack time and memory separately; include both in Phase 4 decision. |

## Next Actions

1. Implement Phase 1's exact-arch selector and OCP-to-FNUZ CPU oracle in an isolated branch of the runtime code, without changing the gfx1201 source/body.
2. Scan the actual independent `SQ8_0` artifact for `0x80`, non-finite OCP codes, scale range, and `[128,128]` tail behavior; save a reproducible manifest.
3. Build Phase 2's standalone FNUZ-prepack/MFMA and dequant-control prototypes, then repeat the existing isolated gfx942 metadata/ISA audit for every selected tile.
4. Obtain access to a physical gfx942 system before beginning Phase 3. Its exact SKU and XCD/NPS partition are hard prerequisites, not optional benchmarking detail.
5. Keep final activation out of scope. Any future change to `/etc/ullm/served-models/active.json` requires explicit human approval after the Phase 3–5 gates.

### Sources consulted in Phase 0

- Local source and headers: `crates/ullm-engine/src/sq8_*.rs`, `runtime/src/sq8_ck_gfx1201.hip.cpp`, `runtime/src/kernels/sq8_0/*`, `runtime/src/ullm_runtime_*`, `/opt/rocm/include/hip/amd_detail/amd_hip_fp8.h`, `/opt/rocm/include/rocwmma/internal/{config,mfma_impl,constants}.hpp`, and local ROCm profiler gfx942 metadata.
- AMD architecture/partition references: [AMD SMI GPU partition documentation](https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/partition.html), [AMD Instinct MI300 product specifications](https://www.amd.com/en/products/accelerators/instinct/mi300.html), and [HIP compiler architecture documentation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/compilers.html).
