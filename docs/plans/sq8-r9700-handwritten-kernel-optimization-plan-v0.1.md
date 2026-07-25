# `SQ8_0` R9700 Handwritten Kernel Optimization Plan v0.1

- Status: Phase 0 complete — R9700 (`gfx1201`) hot paths, scoped profiles, logical-bandwidth baseline, and offline resource audit captured; no production body was changed.
- Date: 2026-07-26
- Scope: Qwen3-14B-FP8 independent `SQ8_0` execution on R9700 (`gfx1201`, PCI `0000:47:00.0`) only.
- Boundary: preserve the external ABI and dispatch boundary exactly. This plan changes neither an activation file nor any campaign, candidate, release, unit file, `/opt/ullm` content, or existing build/release tree.

## Goal

Build an evidence-led handwritten optimization path for the actual `SQ8_0` serving hot kernels on R9700, starting with low-risk reductions that were proven useful for `AQ4_0` and retaining a high-risk handwritten projection replacement option behind the unchanged ABI.

The goal is not to assume that the generic `SQ8_0` matvec source is serving. It is to improve the kernels selected by the measured M=1 decode and M=128 prefill workloads, using the fixed Phase 0 baseline and the following safe sequence for every candidate:

1. isolated prototype under a non-production symbol;
2. offline HIPRTC/HIP code-object metadata and ISA audit;
3. R9700 differential plus scoped timing/thermal measurement; and
4. only in a separately approved later task, a body replacement behind the unchanged external ABI and dispatch.

## Success Criteria

### Evidence and selection

- Every priority is justified by a selected-region R9700 trace, not by source reachability alone. The reference workload remains: decode M=1, 16 steps, cache window `1028 -> 1044`, and prefill 1024 tokens in M=128 chunks.
- The next candidate repeats the Phase 0 ROCTx scope: model load, seed prefill, warm-up, finish, and reset remain outside the decode profile. Both raw trace and aggregation are saved with the candidate.
- The unprofiled decode baseline is retained as `15.294955751 tok/s` (`65.381032563 ms/token`, five repeats). A profiler-instrumented timing is never substituted for this baseline.

### Correctness, resources, and operating safety

- Each prototype differentials against the current selected body for all actual projection shapes and M values it changes; reduction candidates also cover zeros, extrema, all-equal blocks, and real artifact activations. Numerical acceptance tolerance is frozen before timing rather than adjusted after observing a result.
- HIPRTC/HIP metadata records VGPR, SGPR, LDS, wave size, private/spill state, and ISA evidence for every changed kernel. A new spill or a resource regression needs an explicit measured justification.
- Each R9700 run records exact `gfx1201`/BDF identity, temperatures, clocks, and power. `llama-qwen35-udq4.service` must be verified `inactive`/`disabled` before every measurement and remain so. If `ullm-openai.service` was active, it is stopped only for the isolated window and restored to the same active state.

### Performance decision

- The fixed Phase 0 `SQ8_0` logical-stream baseline at midpoint context `C=1036` is `B=15,109,299,200 B/token`, roof `42.358020152 tok/s` at 640 GB/s, measured logical rate `231.096063 GB/s`, and `eta_logical=36.1088%`. This is a KV-inclusive lower-bound traffic metric, not physical HBM-counter efficiency.
- A candidate states both its kernel-time-domain Amdahl ceiling and its measured end-to-end decode change. No projected end-to-end speedup is called achieved until the unprofiled timing and differential gates pass.
- A production-body change is eligible only if it is correct, reproducible under the same thermal/clock conditions, and improves the relevant workload without changing the ABI or dispatch. It remains outside this plan's Phase 0 deliverable.

## Non-Goals

- Do not replace a production symbol, alter an external ABI or dispatch decision, or activate any served model. In particular, `/etc/ullm/served-models/active.json` is outside scope and requires separate human approval.
- Do not treat the generic `ullm_sq_fp8_matvec_{f32,batch,pair,triple}_kernel` family as a current hot path: none appears in either Phase 0 selected trace. A wide-load/shuffle rewrite there is conditional future work, not the first implementation.
- Do not rewrite the currently selected paged-decode reduction merely because a full LDS fallback exists in source. The normal source uses wave shuffle; whether the fallback environment variable was set during Phase 0 is **未確認**.
- Do not infer physical HBM traffic, achieved occupancy, or prefill unprofiled tok/s from static metadata or a profiler trace. Those values are **未確認** until separately measured.
- Do not retarget the gfx1201 CK body to CDNA3 or reuse wave32 code by assumption. CDNA3 work has a separate native MFMA design and device gate.

## Working Hypotheses

1. **The actual projection hot path is CK, not the generic reference matvec.** The selected decode trace contains 4,480 CK launches (`16 * 40 layers * 7 projections`) for 40.1305% of kernel time; prefill contains 2,240 CK launches for 11.3154%. `Sq8LayerExecutionProfile::Rdna4W8a8BlockCk` reaches `DeviceGemmMultipleD_ABScale` / `DeviceGemmXdlUniversal` through `ullm_sq8_ck_gfx1201_projection`.
2. **The low-risk `AQ4_0` reduction lesson applies to selected handwritten helper bodies.** `quantize_activation_block128` retains a 128-element barrier tree (0.7022% decode, 1.5623% prefill), and `ullm_segmented_rmsnorm_f32_kernel` retains `float partial[256]` plus a barrier tree (1.9049% decode, 0.5434% prefill).
3. **Prefill has a larger handwritten reduction opportunity, but a higher correctness risk.** `ullm_cached_prefix_attn_f32_flash2_kernel` is 74.8409% of selected prefill kernel time and repeatedly reduces a 256-float LDS array for QK, max, and sum. Softmax/reduction ordering makes it a separate, medium-risk effort rather than a copy-paste of the helper changes.
4. **A current selected projection-wide-load deficit is not established.** The matching CK objects contain `buffer_load_b128` instructions (12/29/29/24 occurrences by selected form). The generic reference objects use `global_load_u8`, but that family was absent from both traces. The wide-load portion of the prior `AQ4_0` method is therefore conditional rather than a Phase 1 serving-path change.
5. **CK resource geometry makes a handwritten projection route worth preserving as a high-risk option.** Two selected decode CK forms use 36,864 B LDS with 242/175 VGPR metadata; given 64 KiB LDS/CU and a 256-thread CTA, their LDS-only ceiling is one workgroup/eight wave32 (25% of a 32-wave reference). A replacement must prove that lower LDS/VGPR pressure is worth giving up CK's existing vectorized loads.
6. **Architecture-neutral semantics and architecture-private execution reduce CDNA3 carry cost.** Canonical payload bytes, BF16 `[128,128]` weight scales, shape validation, differential vectors, and timing accounting can be common. Wave size, fragment/lane map, OCP/FNUZ conversion or prepack, matrix instructions, vector-load geometry, and LDS layout must be private to the target backend. The future CDNA3 continuation is **案 A: handwritten MFMA**, not a CK retarget.

## Phase breakdown

### Phase 0 — selection, audit, and baseline (complete)

Phase 0 used only R9700 and saved raw results in `benchmarks/results/2026-07-26/sq8-r9700-handwritten-kernel-phase0-v0.1/`.

| route / observation | decode kernel share | prefill kernel share | decision |
|---|---:|---:|---|
| CK projections | 40.1305% | 11.3154% | actual projection boundary; high-risk handwritten replacement option |
| paged decode attention | 50.9968% | n/a | already shuffle-based by default source; do not assume its LDS fallback is active |
| Flash2 cached-prefix attention | n/a | 74.8409% | medium-risk prefill reduction/algorithm priority |
| segmented RMSNorm | 1.9049% | 0.5434% | low-risk wave-shuffle candidate |
| activation quantizer | 0.7022% | 1.5623% | low-risk wave-shuffle candidate |
| BF16-to-F32 helper | 0.7336% | 0.9188% | low/medium-risk vectorization candidate, not a weight-load fix |
| generic reference matvec family | 0% observed | 0% observed | defer unless a future scoped trace selects it |

The fixed logical decode metric includes 280 F8 payloads, their actual BF16 scale grids, the BF16 LM head, and F32 KV reads/writes. At `C=1036`, it measures `eta_logical=36.1088%`; physical HBM efficiency is not available from this capture.

### Phase 1 — low-risk handwritten helper prototypes

Each item is a new isolated entry point or source string first. It does not edit `ullm_sq8_ck_gfx1201_projection`, the public declarations, or Rust dispatch.

| priority | candidate | expected effect estimate | required measurement and decision |
|---|---|---|---|
| P1 | `quantize_activation_block128`: 128-thread max tree to wave32 shuffle plus minimal cross-wave handoff | at most 0.7022% of Phase 0 decode kernel time and 1.5623% of prefill kernel time if only this body changes | Compile for gfx1201; verify no spill and reduced/sensible LDS; differential quantized bytes and scales for zeros, extrema, ties, and real activations; repeat both scoped traces. Keep only if correct and its own launch time decreases without regressing occupancy. |
| P2 | `ullm_segmented_rmsnorm_f32_kernel`: `partial[256]` tree to a wave-shuffle reduction | at most 1.9049% decode and 0.5434% prefill kernel time | Offline source/ISA audit, then vector/output differential and scoped traces. Prefer a 256-thread wave32 decomposition with only the necessary inter-wave LDS handoff; reject any unexplained numerical drift or spill. |
| P3 | `bf16_to_f32`: evaluate aligned vector loads/stores and conversion packing | at most 0.7336% decode and 0.9188% prefill kernel time | First inspect emitted vector instructions and alignment for each actual output shape; then differential bit patterns and timing. It is a post-projection conversion, so it must not be represented as an `SQ8_0` weight-load improvement. |

The upper bounds are shares of summed selected kernel duration, not end-to-end throughput promises. The 5.9746% prefill copy row is recorded but is not a handwritten `SQ8_0` kernel target in this phase; it needs its own ownership and dataflow proof before being optimized.

### Phase 2 — prefill Flash2 reduction prototype (medium risk)

Prototype `ullm_cached_prefix_attn_f32_flash2_kernel` under a distinct symbol while holding its F32 KV layout, tile traversal, and public launch contract fixed. Replace one reduction class at a time: QK score reduction, tile maximum, then tile sum. Do not fuse unrelated work merely to alter timing.

- Expected effect estimate: the sole body accounts for 74.8409% of selected prefill kernel time, which is an Amdahl ceiling rather than a predicted speedup.
- Required static gate: record LDS/VGPR/SGPR/spill metadata and inspect wave32 shuffle/LDS instructions. The Phase 0 body is 1,292 B LDS, 21 VGPR, 46 SGPR in offline metadata; a prototype must explain any increase.
- Required correctness gate: compare attention output over variable cached-prefix lengths, short/tail tiles, adversarial score ranges, all-masked/zero cases if supported by the contract, and a real 1024-token prompt. Validate downstream hidden state/logits before timing.
- Required performance gate: repeat the M=128 selected region with the same prompt, record the prefill thermal/clock/power window, and separately report profile-kernel time and unprofiled prefill throughput. The latter is presently 未確認.

If this fails the numerical or resource gate, retain the current Flash2 body and move to the high-risk projection experiment rather than silently weakening the attention implementation.

### Phase 3 — handwritten projection research body (high risk)

This phase implements the user's chosen handwritten route, but only behind an isolated internal symbol. It is not permission to replace the CK body.

1. Freeze an architecture-neutral projection contract: canonical OCP E4M3FN payload semantics, BF16 `[128,128]` scale layout, M/N/K and scale-boundary behavior, output type, workspace lifetime, artifact vectors, and differential harness.
2. Create a gfx1201-private projection body with its own tile/workgroup/load schedule. Compare it only to the exact CK form selected for each M/N/K shape. Preserve the current public `ullm_sq8_ck_gfx1201_projection` signature and Rust call/dispatch selection.
3. Audit each shape's generated gfx1201 ISA for intended vector loads, matrix instructions if used, barriers, LDS, VGPR, SGPR, and spills. The test matrix includes M=1 decode and M=128 prefill plus every actual projection shape/tail selected by the model.
4. Differential before timing, then run the same ROCTx-scoped profile and unprofiled decode timing. Report CK-form replacement time, all-projection subtotal, end-to-end TPS, and `eta_logical` under the unchanged `C=1036` accounting.

The maximum directly addressable subtotal is 40.1305% of selected decode kernel time and 11.3154% of selected prefill kernel time. The primary resource hypothesis is the two 36,864-B/one-workgroup CK forms, but CK already has wide loads and no spill in the audited object; a handwritten body that merely reduces source complexity without winning measured time is rejected.

### Phase 4 — architecture split and CDNA3 handoff (design + separately gated implementation)

The handwritten projection's semantic layer is shared only where architecture-independent:

```text
canonical SQ8_0 payload + [128,128] BF16 scales + shape/differential contract
       |                         |
       |                         +-- common validation, test vectors, timing accounting
       |
       +-- gfx1201 private body: wave32 / R9700 load and LDS schedule
       +-- gfx942 private body: 案 A handwritten MFMA / wave64 / FNUZ-prepack boundary
```

The gfx942 branch inherits the hand-written MFMA objective from the CDNA3 plan, not the RDNA4 CK internals. Its OCP-to-FNUZ conversion/prepack, MFMA fragment layout, wave64 reduction, and scale application remain architecture-private and require the CDNA3 plan's independent device, numerical, and performance gates. Neither branch changes the canonical artifact or external ABI.

### Phase 5 — guarded body replacement (not authorized by Phase 0)

Only after one isolated candidate meets all correctness, resource, timing, and regression gates may a separate task propose replacing an internal production body. That task must re-run source/ABI checks, R9700 scoped profiles, unprofiled timing, and service lifecycle checks. Activation remains a distinct human-approved operation.

## Decision Tree

```text
Start: scoped R9700 trace for independent SQ8_0
  |
  +-- Is the candidate selected in decode or prefill?
  |     |-- no --> keep it as reference/fallback; do not optimize on source appearance alone
  |     `-- yes
  |
  +-- Is it a full LDS tree in the selected body?
  |     |-- quantizer/RMSNorm --> isolated wave-shuffle prototype (Phase 1)
  |     |-- Flash2 --> staged attention prototype with full numerical gate (Phase 2)
  |     `-- paged decode --> first capture the reduction environment; fallback activation is 未確認
  |
  +-- Is it the CK projection subtotal?
  |     |-- no --> profile/resource-prioritize the selected helper
  |     `-- yes --> handwritten gfx1201 body behind unchanged ABI (Phase 3)
  |
  +-- Does offline metadata show intended ISA, no unexplained spill, and acceptable LDS/VGPR?
  |     |-- no --> revise isolated body; no integration
  |     `-- yes
  |
  +-- Do differential and repeated R9700 measurements pass and win?
  |     |-- no --> retain current selected body and record the negative result
  |     `-- yes --> separate guarded body-replacement proposal
  |
  `-- Is target architecture gfx942?
        |-- no --> do not reuse wave64/MFMA assumptions
        `-- yes --> follow CDNA3 案 A handwritten MFMA gates with private backend code
```

## Risks

| risk | mitigation / stop condition |
|---|---|
| Source inventory optimizes a non-serving reference path | Require a fresh scoped trace before every priority decision; generic reference matvec stays deferred while absent. |
| Reduction reassociation changes quantization/normalization/softmax results | Differential before timing; freeze tolerances and test adversarial plus artifact-derived inputs. |
| Wide-load change raises VGPR and lowers occupancy, as seen in prior multi-stream work | Inspect metadata before hardware timing; compare LDS/VGPR/spills and retain shuffle-only variants where wide loads regress. |
| Handwritten projection loses CK vectorization or matrix throughput | Treat exact CK form as control; require per-shape wins, not a source-level claim. |
| Static resource math is mistaken for achieved occupancy | Label it as a ceiling; confirm runtime behavior and timing on R9700. |
| Profiler overhead or unscoped warm-up contaminates a conclusion | Keep ROCTx selected regions and report profiler versus unprofiled timing separately. |
| Service restart/thermal state contaminates a run | Record service/`llama-qwen35-udq4.service` states and telemetry; restore `ullm-openai.service` immediately after each window. |
| CDNA3 leakage into R9700 path | Keep the common layer semantic-only and architecture-private bodies separate; preserve the current gfx1201 ABI and selector. |

## Next Actions

1. Implement P1 as an isolated `quantize_activation_block128` shuffle prototype, then capture its metadata, byte/scale differential, decode/prefill selected traces, and R9700 telemetry.
2. If P1 is correct and beneficial, perform P2 RMSNorm with the same evidence bundle; do not batch the two changes before obtaining attribution.
3. Run the staged Flash2 experiment only after the low-risk helpers establish the prototype harness and numerical reporting format.
4. In parallel only at the design level, freeze the architecture-neutral projection contract and gfx1201/gfx942-private backend split needed for the Phase 3 handwritten body and CDNA3 案 A MFMA handoff.
5. Reconsider generic matvec wide-load/shuffle work only if a future scoped trace shows one of its exact symbols; otherwise preserve it as a reference path.
