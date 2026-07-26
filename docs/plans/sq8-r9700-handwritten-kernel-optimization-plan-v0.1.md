# `SQ8_0` R9700 Handwritten Kernel Optimization Plan v0.1

- Status: Phase 0 and attention-path evidence complete; Flash2 staged-wave32 prototype evaluated NO-GO on full-model numerical gate; explicit paged split API and a single-window full-model M=1 split opt-in were evaluated; no production body or default dispatch was changed.
- Date: 2026-07-26
- Scope: Qwen3-14B-FP8 independent `SQ8_0` execution on R9700 (`gfx1201`, PCI `0000:47:00.0`) only.
- Boundary: preserve the external ABI and dispatch boundary exactly. This plan changes neither an activation file nor any campaign, candidate, release, unit file, `/opt/ullm` content, or existing build/release tree.

## Goal

Build an evidence-led handwritten optimization path for the actual `SQ8_0` serving hot kernels on R9700, starting with low-risk reductions that were proven useful for `AQ4_0` and retaining a high-risk handwritten projection replacement option behind the unchanged ABI.

The goal is not to assume that the generic `SQ8_0` matvec source is serving. It is to improve the kernels selected by the measured M=1 decode and M=128 prefill workloads, using the fixed Phase 0 baseline and the following safe sequence for every candidate:

1. isolated prototype under a non-production symbol;
2. offline HIPRTC/HIP code-object metadata and ISA audit;
3. R9700 differential plus scoped timing/thermal measurement; and
4. only in a separately scoped later task, a body replacement behind the unchanged external ABI and dispatch.

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

- Do not replace a production symbol, alter an external ABI or dispatch decision, or activate any served model. In particular, `/etc/ullm/served-models/active.json` is outside this optimization scope; a later promotion uses the lightweight promotion policy and its rollback transaction.
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

Only after one isolated candidate meets the applicable correctness, resource, timing, and regression evidence may a separate task propose replacing an internal production body. That task must re-run source/ABI checks, R9700 scoped profiles, unprofiled timing, and service lifecycle checks. Activation remains a distinct lightweight promotion operation with automatic rollback on a failed live check.

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

## Attention-focused Phase 1 — execution evidence and quantified priority update (complete)

This addendum records the direct runtime proof that Phase 0 lacked. It is appended rather than retroactively changing the Phase 0 record.

### Scope, qualification, and direct conclusion

The active ullm-openai.service process before the window was an AQ4 worker, not an SQ8_0 worker. Changing the active model is forbidden, so a literal observation of a live SQ8_0 service is unavailable. Instead, a fresh isolated process loaded the production SQ8_0 artifact/package, used the required HIP guards and R9700 only, emitted GPU_DUMP_CODE_OBJECT objects in its own CWD, and was traced in the same process. This is production-artifact, serving-equivalent SQ8_0 evidence; it must not be described as a live active-service SQ8_0 observation.

The paged M=1 decode attention that ran is the wave-shuffle body:

| condition | code-object SHA-256 | ISA of ullm_paged_decode_attn_f32_kernel | measured result |
|---|---|---|---|
| default; both disable variables absent | 26fa813c4b2d35e90361ff50c6648d1d9d5412da041658c189f2fcbc095b6bb1 | 10 ds_bpermute, two LDS rendezvous points | 640 selected launches, grid 10240 / block 256 |
| explicit ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE=1 control | b856136847c042ab6713f8dd4e30d14799a59ff580829993b2acb0156fb1e9fa | no ds_bpermute, nine LDS rendezvous points | same name/geometry, distinct JIT object |

The default object is captured at benchmarks/results/2026-07-26/sq8-r9700-attention-phase1-v0.1/decode/code-objects/default-runtime-dump/_code_object0010.o. Its selected-symbol disassembly and metadata are retained under static/isa/. The trace selects the same symbol. Thus, the answer does not rely on source reachability or an offline-only object.

### Exact selector, launch path, and fallback remediation

- runtime/src/ullm_runtime_hiprtc_sources.inc constructs the paged source preamble by testing only whether ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE is present. It ignores the value, so =0 is still fallback. The arch argument is explicitly unused here. The independent ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX presence test selects two-pass softmax, not the reduction implementation.
- runtime/src/ullm_runtime_parts/part_01.inc launches q_heads workgroups when head_dim and value_dim are at most 256. The measured SQ8_0 shape has 40 query heads, 8 KV heads, and 128/128 head/value dimensions; the trace's 10240-thread grid and 256-thread workgroup are exactly 40 workgroups and confirm this head-parallel branch.
- crates/ullm-engine/src/decoder.rs calls the direct paged interface. No split partial or split merge kernel appears in the trace. The split interface exists as an explicit alternative but has no automatic selection in this execution.
- ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL makes an HIP failure fail closed rather than entering the staging path. It does not select shuffle versus LDS. It was set in the isolated environment; the direct JIT capture and trace show that HIP succeeded.

No production remediation is required because the measured default already uses shuffle. If a future environment contains the disable variable, unset it entirely; do not set it to 0. The isolated negative control provides the recovery magnitude at the exact workload: default unprofiled decode is 15.367844 tok/s versus 14.674604 tok/s for fallback, a 4.724077% throughput recovery. The measured current incremental gain remains 0%.

### Time, resource, and bottleneck decomposition

| body | selected time | share | static resource metadata | runtime trace allocation |
|---|---:|---:|---|---|
| paged decode default | 492.371584 ms / 640 launches | 51.05% decode | LDS 1024 B, VGPR 25, SGPR 52, wave32, no private/spill | LDS 1024 B, VGPR 32, SGPR 128 |
| paged decode forced fallback | 536.261934 ms / 640 launches | 53.20% decode | same aggregate metadata, but different JIT ISA | LDS 1024 B, VGPR 32, SGPR 128 |
| Flash2 prefill | 2196.476598 ms / 320 launches | 75.63% prefill | LDS 1292 B, VGPR 21, SGPR 46, wave32, no private/spill | LDS 1536 B, VGPR 24, SGPR 128 |

For every paged score at C=1036, default executes two source-level CTA rendezvous points (2072 per workgroup over the cache), while forced fallback executes nine (9324). Its 8.914070% profiler-domain penalty and 4.724077% unprofiled throughput penalty are direct measurements. Flash2 has 10 rendezvous points for each score, 10 for tile max, 10 for tile sum, and one final point: 661 for each full 64-token tile. Flash2 has no shuffle instruction in its selected code object.

R9700 exposes 64 CU and 32 waves/CU. Decode supplies only 40 workgroups x 8 wave32 = 320 waves per layer dispatch, or 15.625% of the 2048-wave machine ceiling. Flash2 supplies 5120 workgroups/40960 waves per layer launch. The 64 KiB group-memory pool means the listed 1.0/1.292 KiB static LDS uses do not themselves reduce a four-workgroup/32-wave ceiling; achieved register-limited occupancy remains 未確認.

The selected default paged body has static global_load_b32 instructions but no global b128; Flash2 likewise has global_load_b32 and its ds_load_b128 is LDS, not global. Adjacent lanes access adjacent head/value elements, so this mnemonic form is not proof of narrow physical memory transactions. A P3 uint4 approach is applicable only as a proved lane/tile redesign, not as a blind one-line replacement.

The semantic F32 KV scan rate is 55.157770 GB/s at decode midpoint C=1036 (8.618402% of the 640 GB/s reference) and 391.459814 GB/s for causal prefill 1..1024 (61.165596%). This counts logical K/V vectors and does not claim physical HBM traffic. FETCH_SIZE and VALUInsts were zero in every PMC sample, although Wavefronts correctly reports 320 and 40960 per dispatch. The counter failure cause, physical HBM efficiency, and a final memory-bound/compute-bound classification are therefore **未確認**.

### Priority update and measurable candidate gates

The helper-first order is updated because quantizer plus RMSNorm have only a 2.6067% decode share, whereas attention has direct structural evidence.

| new priority | candidate | confirmed numerical opportunity | performance claim permitted now | required decision experiment |
|---|---|---|---|---|
| P1 | preserve wave-shuffle admission for paged decode | 4.724077% recovered decode tok/s when removing the deliberately forced fallback | current incremental 0%; misconfiguration recovery measured | fresh-process environment admission test; do not modify production environment in this task |
| P2 | Flash2 staged wave32 QK/max/sum reduction | 75.63% selected share and 661 full-tile rendezvous points | performance range **未確認**; reducing source barriers is not a speed estimate | distinct symbol, staged differential, metadata/ISA, selected trace, and unprofiled prefill |
| P3 | paged source-tile split | source_tile=256 gives five splits, 200 partial workgroups/1600 waves and 104000 B workspace at C=1036 | 15.625% -> 78.125% is a work-supply envelope only; speed range **未確認** | direct-versus-split differential and timing for tiles 128/256/512 |
| P4 | Flash2 lane/tile load re-layout or uint4 | no global-wide-load proof; no usable physical byte counter | speed range **未確認** | prove alignment/coalescing, VGPR/LDS impact, output differential, and physical counters before claiming a load win |

P3-proven techniques map cleanly: wave-shuffle is a P2 primitive; static VGPR/LDS/ISA audit gates every candidate; uint4 is conditional P4 work. P2 and P3 are tile/algorithm changes and require their own softmax/order and workspace differentials. No numerical gain has been invented for an unprototyped candidate.

### CDNA3 handoff contract

The common layer is canonical SQ8_0 payload/scale meaning, projection-to-attention input contract, F32 paged-KV and causal/online-softmax semantics, shape validation, adversarial/real-artifact vectors, differential harness, and timing evidence schema. It contains no lane map.

The gfx1201 body remains a wave32 R9700 implementation with its own shuffle and LDS layout. The CDNA3 continuation is a separate wave64/MFMA body with separate fragment mapping, LDS layout, conversion/prepack boundary, code object, and ISA audit. Both compare against the same canonical vectors, but neither reuses the other's wave-level implementation. This preserves the intended CDNA3 案 A handoff without changing the external ABI or dispatch boundary.

## Attention optimization execution addendum — 2026-07-26 (NO-GO for Flash2 body)

Raw evidence for this addendum is retained in
`benchmarks/results/2026-07-26/sq8_0-attention-optimization-u-v0.1/`.
It is an isolated canonical-artifact process on R9700 only, not an observation
of a live SQ8 service.

### PMC diagnosis

The installed ROCm 7.2.1 counter definitions explicitly contain gfx1201
definitions for `FETCH_SIZE`, `VALUInsts`, `SQ_INSTS_VALU`, and
`GL2C_EA_RDREQ_{32B,64B,128B}`.  A purpose-built R9700-only load+FMA kernel
showed all raw instruction and GL2C request counters as zero, while
`SQ_WAVES=32768` per dispatch was nonzero.  The derived probe remained
`FETCH_SIZE=0`, `VALUInsts=0`, `Wavefronts=32768`.

The actual selected Flash2 collection has the same property on all 160
observed launches: `FETCH_SIZE=0`, `VALUInsts=0`, and
`Wavefronts=40960` per launch.  Thus the prior zero values are not a typo or
an unsupported derived-metric name; primitive counter collection is failing
selectively.  The exact root cause below the profiler (permission,
driver/firmware, or ROCm counter-programming behavior) is **未確認**.  A
root-only retry was not opened after the service start-limit budget had been
used.  Physical HBM efficiency and a final memory-bound/compute-bound verdict
therefore remain **未確認**; logical KV rates, ISA, static resources, and
workgroup-supply geometry remain the admissible evidence.

### Flash2 staged wave32 result

An isolated HIPRTC source creates separate legacy, QK-only, QK+max, and full
QK+max+sum staged symbols.  The normal runtime symbol and its default selector
remain unchanged.  Offline full-staged metadata is wave32, LDS 1296 B, VGPR
27, SGPR 48, zero private/spill; the legacy reference is LDS 1296 B, VGPR 21,
SGPR 46, zero private/spill.

The separate-symbol attention differential had no non-finite values.  The
full-staged maximum absolute differences against legacy were `1.1920929e-7`
(short), `1.0430813e-7` (63→68 tail), `2.9802322e-8` (synthetic 896→1024
M=128), and `2.6464462e-5` (adversarial score range).  The synthetic standalone
kernel timing was 13.317192 ms legacy versus 12.876236 ms staged per launch;
this 1.03425x result is not serving throughput.

The unprofiled baseline on the canonical `raw-p0512` vLLM-source fixture ran
four M=128 units in 1.167487403 s (438.548629 input tok/s).  However, the
staged full-model output failed the frozen SQ8 vector gate:

| capture | max abs | relative L2 | cosine | verdict |
|---|---:|---:|---:|---|
| final hidden | 0.7760314941 | 0.0145683599 | 0.9999164687 | fail |
| logits | 0.2401080132 | 0.0084836396 | 0.9999792756 | fail |

The frozen gate was `max_abs <= 2e-5`, `relative_l2 <= 1e-5`, and
`cosine >= 0.999999`.  A temporary service lifecycle overlap additionally
invalidated the staged serving timing, so it is not interpreted.  The quality
failure alone is enough: **do not replace the production Flash2 body**.

The follow-up generalized the prototype handoff from a fixed eight waves to
`blockDim.x`; actual Flash2 records use a 256-thread/eight-wave workgroup, so
this is behaviorally identical for the measured geometry and cannot reverse
the NO-GO.  The normal source remains the selected body.  This failure also
confirms the CDNA3 handoff rule: retain canonical semantics/vectors/harness as
common assets, but keep R9700 wave32 and CDNA3 wave64/MFMA implementations
separate.

### Paged decode explicit source-tile result

The direct legacy selector was not touched.  An explicit existing split API
comparison at M=1/C=1036 reported finite outputs and the following attention
API-plus-stream timings:

| source tile | split count | mean ms | max abs vs direct | partial-WG wave supply |
|---:|---:|---:|---:|---:|
| direct | 1 | 0.643241770 | reference | 320 / 15.625% |
| 128 | 9 | 0.228016370 | 1.34110e-7 | 2880 / 140.625% |
| 256 | 5 | 0.227932360 | 1.26660e-7 | 1600 / 78.125% |
| 512 | 3 | 0.383530140 | 1.34110e-7 | 960 / 46.875% |

Tile 256 is marginally fastest among the two near-tied best results (2.822x
lower isolated attention-call time than direct); tile 512 loses.  This supports
the supply-limit hypothesis, but is not by itself a full-model end-to-end
claim and is not permission to change direct dispatch.

A follow-up leaves the normal direct route as the default and exposes the
existing split API only behind the test-only
`ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE` selector.  The selector accepts
only 128, 256, or 512; its absence retains the exact direct legacy path.  In a
clean R9700-only `raw-p0512` run, the seven post-prefill M=1 generated steps
(cache lengths 513--519) had synchronized whole-model times below.  Model
load, the final M=128 prefill step, reset, and profiler overhead are excluded.

| source tile | mean M=1 ms | median ms | speedup vs direct | partial-WG wave supply at C=513 |
|---:|---:|---:|---:|---:|
| direct | 53.519086 | 49.925305 | 1.0000x | 320 / 15.625% |
| 128 | 43.282296 | 39.768787 | 1.2365x | 1600 / 78.125% |
| 256 | 46.706832 | 43.168066 | 1.1459x | 960 / 46.875% |
| 512 | 55.525563 | 51.986695 | 0.9639x | 640 / 31.25% |

All four cases emitted the same eight greedy token IDs
`[66, 198, 197, 197, 280, 197, 197, 280]`.  Tile 128 is therefore the best
observed opt-in at this depth, and the ranking is consistent with recovering
more of the direct path's 15.625% partial-workgroup supply.  This is an
inference from one seven-step window, not a multi-window production performance
claim.  The raw full-model decode-vector differential was not captured and
remains **未確認**; the finite API-level F32 differentials above remain the
numerical evidence for the split body.  No default-dispatch change follows from
this result.

### Deferred work and operating record

`uint4`/lane re-layout was not started because raw physical PMC values remain
unusable.  `llama-qwen35-udq4.service` was verified inactive/disabled and
`gdm3.service` inactive before each measurement window.  For the completed
full-model window, R9700 was edge/hotspot/memory `37/37/34 C`, gfx `2434 MHz`,
socket `16 W` before stop; immediately after the case sequence it was
`45/51/48 C`, gfx `3307 MHz`, memory `1258 MHz`, socket `103 W`, with AMD SMI
reporting `THROTTLED`.  The cause of that status and the in-kernel peak remain
**未確認**.  After restore it was `44/44/42 C`, gfx `1193 MHz`, socket `13 W`,
and `UNTHROTTLED`.

The primary service stop began at 05:05:32+09:00 and the scripted restore was
active at 05:07:48+09:00.  An accidental tool-lifecycle misread caused one
brief manual start/compensating stop in between; it is retained in raw service
logs and is why staged serving timing is discarded.  A later path-error retry
and the first decode e2e CLI-contract rejection each restored immediately
without launching a GPU kernel.  The completed decode window ran
05:28:41--05:31:25.  These are five total `systemctl` stop/start pairs (the
primary logical window plus the documented manual compensation, path-error
retry, aborted decode attempt, and completed decode window).  Final service
state was active/running, `NRestarts=0`; no systemd unit content or active-model
bytes were modified.

## Paged decode source-tile full-model gate — 2026-07-26 (NO-GO)

The earlier M=1 timing advantage for source-tile 128 was not sufficient to
change the default dispatch: a full-model decode feedback gate was frozen and
run with direct as the reference.  The criterion is recorded in
benchmarks/results/2026-07-26/sq8_0-paged-decode-tile-gate/gate-criteria.json
(SHA-256 645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978).
Every actual real-prompt decode capture must have exact greedy tokens, finite
values, max abs <= 2e-5, relative L2 <= 1e-5, and cosine >= 0.999999.
Missing captures, early EOS, or geometry/hash mismatch fail the route.  The
thresholds were frozen before the measurement and were not relaxed.

Prompt lengths 127/128 and 511/512 yielded decode cache lengths 128--131 and
512--515 respectively, covering source-tile boundaries and non-multiple tails.
For each request, three M=1 decode feedback steps captured final hidden state
and logits as F32LE.  All tokens were exact and all values finite, but both
candidates failed the vector gate:

| candidate | passed pairs | failed pairs | worst max abs | worst relative L2 | minimum cosine | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| tile 128 | 4 / 24 | 20 / 24 | 2.317678451538086 | 0.08369554694605848 | 0.9965189313620728 | FAIL |
| tile 256 | 12 / 24 | 12 / 24 | 1.9435234069824219 | 0.03318822738718883 | 0.9996737107487421 | FAIL |

Tile 128 diverged after the first 128-boundary feedback in some shapes.  Tile
256 passed the 128-group but diverged in the 512-group.  The source-level root
cause remains **未確認**; the geometry/tail correlation is not a causal proof.
Consequently direct remains the default dispatch, and the existing explicit
source-tile opt-in remains investigation-only.  No external ABI, legacy direct
dispatch, active-model manifest, or service deployment was changed.  Tile
64/96 was not explored because the required numerical gate had already failed
and a further service window was not justified.

This gate used one stop/isolated/restore window: 06:34:54--06:39:36 JST.
Preflight recorded R9700 gfx1201 as the sole execution GPU; V620 was not
selected.  llama-qwen35-udq4.service was inactive/disabled and gdm3 inactive.
The service restore was active/running with NRestarts=0.

During 240 one-second samples, AMD SMI reported THROTTLED 119 times and
UNTHROTTLED 121 times.  Raw GPU-metrics v1.3 throttle status contains hotspot
thermal status in the sampled data and one dependent sample also carries PPT0;
no TDC bit was observed.  AMD SMI per-reason fields are unsupported, sampled
temperatures/powers stayed below the reported limits, and the two raw fields
were read separately rather than atomically.  The sustained physical cause is
therefore **未確認**.  Treat timing from this window as conditional pending an
atomic telemetry/all-clear rerun.  This does not invalidate the numerical
NO-GO: frequency/power throttling is a timing concern, not an expected cause
of deterministic source-tile-correlated vector divergence.  No permanent
power-cap/profile change was made.

Raw captures, comparator results, service events, telemetry, and the decoded
throttle caveats are retained under
benchmarks/results/2026-07-26/sq8_0-paged-decode-tile-gate/.

## Paged decode source-tile containment fix — 2026-07-26

The source-level cause of the preceding tile NO-GO is now localized. An
unfixed direct-versus-tile128 capture after decode g0001 at cache length 129
read the logical written KV prefix for all 40 layers: every K and V F32 value
was bit-identical (132,096 elements per component per layer, worst max abs and
relative L2 both zero). Therefore a g0001 KV write-position/range divergence
does not explain the first large g0002 result.

The R9700 API sweep instead has a precise onset when `split_count > 1`:
tile128 is exact at C=128 and nonzero at C=129 (2 splits), while it is also
nonzero at exact-multiple, no-source-tail C=256; tile256 is exact at C=256 and
nonzero at C=257 (2 splits), and is nonzero at exact-multiple C=512. Thus a
tail/page interaction is not required for the bug. The split body computes
per-tile online-softmax max/denominator/numerator states and rescales them in
a merge, whereas direct carries one online state through all sources. The two
have different F32 association once there is more than one tile. The
standalone difference is only about 1e-8--1e-7, but the SQ8 path has 160
activation quantizations across its 40 layers; the real-prompt gate shows
that this violates the direct numerical contract. The exact first
quantizer-boundary crossing was not instrumented and remains **未確認**.

The fix is deliberately a tile-experiment-only containment: one source tile
still uses the split body, while every multi-tile invocation reuses the
existing direct paged-decode kernel. Ordinary environment-absent direct
dispatch, legacy dispatch, and runtime/kernel ABI are unchanged. A test-only
logical KV-prefix capture/evaluator was added so later candidates can repeat
this state check. A genuinely exact multi-tile merge is still required before
the performance implementation can be reconsidered.

The frozen gate criteria are byte-identical to the NO-GO criteria (SHA-256
`645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978`). On
R9700, tile128 and tile256 each pass all 24 full-model hidden/logit vector
pairs with exact generated tokens, finite values, and `max_abs=0.0`. This is
a containment pass, not evidence that multi-tile split is safe; direct remains
the default and there is no default promotion.

At raw-p0512 cache lengths 513--519, both candidates take the direct fallback.
The new synchronized M=1 means are direct 54.277224 ms, tile128 55.596485 ms
(0.9763x), and tile256 54.017278 ms (1.0048x). The earlier tile128 1.2365x
is not retained, as expected when the unsafe multi-tile work is removed.
Telemetry reported throttle states during this one-window timing series, so
the near-1x timing values are conditional; they do not weaken the deterministic
numerical diagnosis. The window stopped once at 07:09:42+09:00 and restored
on the initial start at 07:19:21+09:00, with `NRestarts=0`,
llama-qwen35-udq4 inactive/disabled, and gdm3 inactive. Evidence is retained
under benchmarks/results/2026-07-26/sq8_0-paged-decode-tile-fix/.

## SQ8_0 private handwritten WMMA projection feasibility — 2026-07-26 (numerical NO-GO)

### Scope and unchanged contracts

The projection study used a new **private** gfx1201 WMMA symbol and an explicit
investigation profile. It did not modify runtime/src/sq8_ck_gfx1201.hip.cpp,
the public runtime header, legacy dispatch, ordinary CK selection, the
active-model manifest, campaigns, authorizations, releases, or /opt/ullm.
The ordinary serving path therefore remains the existing CK path.

The intended target was decode M=1. The real model mapping is fixed:

| family | M / N / K | selected CK form | tail observation |
| --- | --- | --- | --- |
| q/o | 1 / 5,120 / 5,120 | Default 16x128x128 | M tail only |
| k/v | 1 / 1,024 / 5,120 | Default 16x128x128 | M tail only |
| gate/up | 1 / 17,408 / 5,120 | KPadding 16x128x256 | M tail only |
| down | 1 / 5,120 / 17,408 | Default 16x128x256 | M tail only |

Every N and K is a multiple of 128. Thus no N/K tail was observed; M=1 is a
tail against CK's MPerBlock=16.

The private body uses gfx1201 v_wmma_f32_16x16x16_fp8_fp8 through rocWMMA,
raw OCP E4M3FN payload, and the same K128 scale-block meaning as CK. The
artifact's weight scale is BF16 on the [128,128] grid; runtime activation scale
is the existing canonical quantizer's F32 output for [M,128] blocks. The
prototype reuses that quantizer rather than changing its semantics.

### Static resource result

Static code-object evidence is retained in
benchmarks/results/2026-07-26/sq8_0-handwritten-projection/static/. The body
is wave32, one 32-thread workgroup per N=16 tile, emits eight FP8 WMMA
instructions, uses 1,280 B LDS, 47 VGPR/thread, 24 SGPR/wave, zero private
bytes, and no VGPR/SGPR spill.

This materially reduces the static resource footprint versus the selected CK
forms: 36,864 B / VGPR 242 (KPadding 128x256), 36,864 B / VGPR 175 (Default
128x256), 34,816 B / VGPR 154 (Default 256x128), and 18,432 B / VGPR 100
(Default 128x128). Under the prior 64-KiB-LDS reference, the first three large
forms have one 8-wave32 CTA (25% of a 32-wave reference), while 128x128 has
three CTAs / 24 waves (75%). At 1,280 B per one-wave prototype block, 32
blocks require only 40,960 B LDS, so LDS alone would not block a 32-wave
reference. This is a static LDS conclusion, **not** a measured occupancy claim;
VGPR/SGPR/hardware workgroup limits remain relevant.

The attempt-2 runtime resource record has one known host-query defect:
threads_per_block=1024 came from maxThreadsPerBlock, not the actual 32-thread
launch. The private source was corrected afterwards, but not remeasured because
another service window was not justified. Its active_blocks_per_cu=51 is HIP's
own per-multiprocessor term; actual CU occupancy is therefore **unconfirmed**.

### CK event baseline and its metric boundary

HIP event timing measured the exact selected CK helper plus its BF16-to-F32
workspace boundary. The rates below are logical route traffic divided by that
time, with 640 GB/s as a nominal reference:

| family (calls/layer) | us / launch | logical GB/s | logical/reference |
| --- | ---: | ---: | ---: |
| q/o (2) | 26.2118 | 1,001.72 | 1.5652 |
| k/v (2) | 26.8975 | 195.39 | 0.3053 |
| gate/up (2) | 158.3728 | 563.61 | 0.8806 |
| down (1) | 148.9054 | 599.03 | 0.9360 |
| seven projections / layer | 571.8696 total | 578.36 | 0.9037 |

The 40-layer projection subtotal is 22,874.7830 us for 13,229,802,240 logical
route bytes. These values are the reproducible CK comparison control, but they
are **not physical achieved HBM bandwidth**: available PMC byte counters were
unusable, and the q/o logical rate exceeding the nominal reference proves that
the metric includes logical traffic rather than a physical bus reading.
Physical HBM efficiency and a memory-versus-compute roofline classification
remain **unconfirmed**.

### Frozen numerical gates and result

Before any candidate timing, the following non-relaxed policy was frozen:

1. all four real M=1 shapes must be finite and F32-bitwise identical to CK
   after the CK BF16 workspace boundary; and
2. with the prototype as the actual full-model M=1 projection path, at least
   two feedback-decode captures must have exact generated IDs, top-1 logits,
   final hidden state, and full logits relative to CK.

The isolated component gate passed all four shapes. That limited raw fixture
used a finite OCP payload cycle and BF16-origin activation scales; source has
since strengthened it to cover all finite payload codes and varied F32
activation scales, but that stronger fixture was not rerun in order to avoid a
third service action. The component result must therefore not be treated as
more than its recorded boundary proof.

The decisive full-model gate failed all three recorded feedback steps despite
equal greedy IDs [66, 198, 197, 197]. Hidden mismatches were 5,120/5,120 with
max abs 0.387939, 0.797844, and 1.287994. Logit mismatches were
151,936/151,936, 151,935/151,936, and 151,936/151,936 with max abs 0.189508,
0.183819, and 0.250601. All values were finite, but top-1 logits were not
bitwise equal. Therefore the prototype is a **numerical NO-GO** and its event
timing was intentionally not run. It has no measured performance comparison
against CK and no eligibility for default replacement.

The exact source-level cause is **未確認**. A difference in WMMA fragment/lane
behavior or K128 scale-block accumulation association relative to CK is a
testable hypothesis only; it is not established from this result. The resource
reduction demonstrates a possible occupancy route, not a proven speed headroom.

### Service and thermal record

There were two stop/isolate/restore attempts. The first
(08:30:12--08:30:48 JST) aborted before GPU work because AMD SMI's no-process
sentinel was parsed as a process, then restored the service. The second
(08:31:52--08:33:27 JST) did the R9700-only work and restored
ullm-openai.service active/running with NRestarts=0. Preflight and final
records show llama-qwen35-udq4.service inactive/disabled and gdm3 inactive.
V620 was not selected.

The second window's 93 AMD SMI samples recorded edge 36--46 C, hotspot 37--60
C, memory 34--48 C, gfx 0--3421 MHz, memory 96--1258 MHz, socket power 7--204
W, and 22 THROTTLED / 71 UNTHROTTLED states. The physical throttle cause is
**未確認**; timing values are conditional on that limitation. No permanent GPU
setting, service unit, activation, authorization, or remote state changed.

### Next actions

1. Capture the actual-artifact input/output around the first divergent
   projection/layer and compare each K128 partial against CK, including
   non-BF16 runtime activation scales.
2. Make the private WMMA reduction/fragment and scale-block association match
   CK's observed contract before considering any new timing window.
3. Rerun the strengthened component fixture and the same multi-step full-model
   gate first. Only a pass may justify one separately approved R9700 timing
   window; default CK remains unchanged otherwise.
4. If numerically exact, record an unambiguous HIP occupancy interpretation
   alongside timing before attributing any gain to LDS headroom.


## SQ8_0 handwritten WMMA projection cumulative-contract diagnosis — 2026-07-26 (NO-GO retained)

### Actual-serving localization

The prior component gate was insufficient to characterize SQ8_0's feedback
contract: it covered four synthetic one-projection cases at the BF16 boundary,
not the actual artifact's per-K128 activation sequence or a complete M=1
serving step. A private terminal-only tracing API was therefore added. It runs
the same 512-token raw-p0512 fixture and M8-chunked prefill as the frozen gate,
then executes the ordinary first M=1 feedback decode through all 40 layers
while reading each layer workspace before head/token commit.

The valid isolated run is
benchmarks/results/2026-07-26/sq8_0-projection-contract/attempt-3/. Both
routes entered decode with token 66 at position 512. Layers 0--2 were bitwise
equal at every captured stage. The first difference is layer 3
down_projected: 2 / 5,120 values differ, first index 1,954, max abs
6.1035156e-5. The layer output has exactly the same two differences. A direct
replay of the actual down projection (M=1, N=5,120, K=17,408) matches the
layer trace for both routes and has the same 2 / 5,120 difference. This
establishes the projection call as the first observed divergence.

### K128 evidence and contract boundary

For the actual layer-3 activation, each replay reuses the existing
block-local quantizer and observes CK's real BF16-workspace-to-F32 boundary.

- Cumulative K128 prefixes are non-monotonic: prefixes 1--5 are exact, prefix
  6 first differs, prefix 8 is exact again, and the full prefix differs.
  The reason for that cancellation/non-monotonicity is **unconfirmed**.
- Isolated K128 blocks locate a mismatch already in block 1 (K=128--255):
  1 / 5,120 at output 1,986, max abs 9.536743e-7. Fifteen isolated blocks
  differ. Hence association among separate K128 blocks is not the sole cause.
- In isolated block 1, K16 prefixes 1--7 are exact; only adding the eighth
  K16 contribution (offsets 112--127) produces the mismatch.
- A one-hot lane probe for K lanes 0--15 of the first output tile passes 16/16.
  This excludes a gross transpose/lane fault for that restricted probe only.

CK source confirms the same high-level scale policy: it zeroes a raw
ScaleBlockK=128 accumulator, executes its XDL operations, then adds
raw × (activation scale × weight scale) to FP32 C. The private body also
holds eight K16 WMMA operations in a K128 raw accumulator before applying the
scale. The selected down CK form is a 256-thread 16x128x256 block; the private
body is a 32-thread N=16 wave, and the inspected gfx1201 CK object has
interleaved WMMA and FP32-FMAC register sequences.

The confirmed result is therefore an **inside-K128 contract discrepancy**.
The exact unique cause remains **unconfirmed**: the eighth K16 operand/
fragment mapping, the WMMA reduction/issue association, or both remain
possible. There is no evidence that an inter-K128 scale-add order alone
explains the failure, and no exact CK register/lane mapping was decoded.

### Decision and performance

No contract-aligned handwritten implementation was made. Accordingly the
unchanged component and multi-step full-model gates could not be rerun as a
pass, candidate event timing was not run, and no default change is eligible.

For the evaluated wave32 handwritten route, CK's contract cannot currently be
kept while claiming a speedup: it is numerically ineligible. Whether a
different handwritten implementation can reproduce CK's exact fragment/
schedule contract and still beat CK is **unconfirmed**. Such a claim requires
an exact mapping implementation, the unchanged component gate, the unchanged
full-model gate, and only then a fresh timing window.

### Evidence hygiene and service record

attempt-1 is a valid but inconclusive isolated layer-0 reconstruction.
attempt-2 is retained but excluded: ullm-openai.service restarted at 09:12:48
while diagnostic artifacts were still written at 09:13:06--09:13:08.
attempt-3 is the sole numerical authority.

There were three stop/isolate/restore cycles. The final valid window was
09:19:29--09:20:46 JST. It used AMD SMI GPU 2 only (R9700 gfx1201,
0000:47:00.0) with HIP_VISIBLE_DEVICES=1; V620 was not selected. After the
no-process sentinel, the diagnostic completed at 09:20:45 and the service was
then restored by a single start. Final state was active/running with
NRestarts=0. llama-qwen35-udq4.service remained inactive/disabled and gdm3
inactive. Endpoint telemetry was 38/38/36 C to 46/47/46 C
(edge/hotspot/memory), 2,833 MHz to 49 MHz gfx, and 16 W to 14 W socket
power; THROTTLED appeared in the post-stop snapshot. The physical throttle
cause is **unconfirmed**. No systemd unit, power setting, active manifest,
campaign, authorization, release, /opt/ullm content, or remote state changed.

Machine-readable evidence and the read-only CK analysis are retained in
benchmarks/results/2026-07-26/sq8_0-projection-contract/.

## llama.cpp decode split-KV comparative evidence — 2026-07-26

### Direct answer and measured geometry

The external llama.cpp Q8_0 F16-KV baseline does use flash-decoding-style KV
parallelism for this exact single-token Qwen3-14B workload. This was not
inferred from a source name: an isolated R9700-only rocprofv3 capture of
`llama-bench -d 1028 -n 16 -ctk f16 -ctv f16 -fa on` recorded, for every one
of the 16 generated tokens, 40 vector-FATTN main dispatches and 40 distinct
combine dispatches. The raw capture, selection script, and aggregate CSV/JSON
are retained in
`benchmarks/results/2026-07-26/llamacpp-attention-analysis/`.

| per layer attention dispatch | raw global grid / workgroup | workgroups | wave32s | 64 CU × 32-wave supply proxy |
|---|---|---:|---:|---:|
| llama.cpp vector FATTN main | `(32, 40, 40)` / `(32, 4, 1)` | 400 | 1,600 | 78.125% |
| llama.cpp FATTN combine | `(128, 40, 1)` / `(128, 1, 1)` | 40 | 160 | 7.8125% |
| uLLM phase1 direct paged attention | `(10240, 1, 1)` / `(256, 1, 1)` | 40 | 320 | 15.625% |

The profiler records global work items, so the first llama.cpp row is
`1 × (40 / 4) × 40 = 400` workgroups, not 32×40×40 workgroups. Its P value is
therefore `Grid_Y / Workgroup_Y = 10`. With a 1280-token internally padded KV
length and a 128-token vector tile, each partial covers one contiguous tile
in this capture. Across the 40 layers, llama.cpp emits 17,600 attention
workgroups per decoded token (16,000 main plus 1,600 combine), versus uLLM
direct's 1,600. The relevant main-dispatch supply grows tenfold; the total
attention workgroup count grows elevenfold because llama.cpp also pays the
merge dispatch.

The 78.125% and 15.625% figures are deliberately only queued-wave supply
proxies. They are not achieved occupancy, concurrent residency, or physical
HBM bandwidth. Those runtime quantities remain **unconfirmed** for this
comparison.

### Structure and format boundaries

The selected llama.cpp body is `flash_attn_ext_vec<128,1,...>`, not its WMMA
body, despite the gfx1201 build having ROCWMMA FATTN enabled. It maps Q head
to KV head with `head / gqa_ratio`; the Qwen shape is 40 Q heads / 8 KV heads
(GQA ratio five). `blockIdx.y` chooses the KV partial and advances at
`gridDim.y * 128` tokens. Each partial produces an online-softmax max,
denominator, and unnormalised weighted-V state; the combine body takes the
global max, reweights every partial by `exp(partial_max - global_max)`, sums,
and divides.

This is the same category of reassociation as uLLM's explicit source-tile
partial/merge API. llama.cpp's cache is a continuous per-layer K/V tensor,
whereas uLLM resolves logical source positions through its paged
`block_table`. This layout difference may affect address overhead and cache
behavior, but it does not by itself create the observed 40-to-400 main
workgroup expansion. The expansion is P=10 KV split.

The external profile used F16 K/V. llama.cpp casts F32 K/V to F16 immediately
before FATTN as well, so its published F32-KV row is not a pure F32 attention
body. uLLM phase1 used F32 paged K/V and SQ8_0 weights. Thus, the measured
0.763730 ms/token llama.cpp attention sum and the uLLM 30.773224 ms/token
attention sum are strong mechanism evidence, but they are not a format-free
throughput substitution. In the respective selected traces attention accounts
for 2.7628% and 51.05% of summed kernel duration.

### Numerical contract consequence

llama.cpp's source does not preserve direct online-softmax association when
`parallel_blocks > 1`; the captured default is P=10. It exposes no public
knob to vary P, and llama-bench has no output/tensor comparator, so the actual
end-to-end numerical difference when P is changed is **unconfirmed**. It is
valid to conclude that llama.cpp operates normally with the split arithmetic;
it is not valid to claim that a particular output difference or tolerance has
been measured.

For SQ8_0, the corresponding uLLM multi-tile route already failed the frozen
bitwise full-model gate: a partial online-softmax merge changes finite-precision
association, and the sequential activation quantizer amplifies the difference
in feedback decode. The existing containment deliberately uses direct fallback
for multi-tile requests. Consequently, copying llama.cpp's P>1 body or its
merge policy cannot satisfy the current bitwise contract.

The current uLLM source remains useful for structural explanation but is not
whole-file-identical to the phase1 runtime snapshot: phase1 records HIPRTC /
launcher hashes `ad050…032b57` / `dcc883…723927`; current files hash
`1600d2…798f3` / `daee4e…a0bbe`. The phase1 raw trace remains the authority for
its measured geometry and timing.

### Resulting implementation policy

1. Keep direct paged decode as the only eligible route under the existing
   bitwise gate. Do not implement llama.cpp-style P>1 split/merge as a default
   or represent it as a semantics-preserving optimization.
2. A direct-order-preserving candidate may still improve page-address work,
   load/coalescing, GQA reuse, or launch overhead. Such work can preserve the
   current contract, but it will not reproduce the central 40-to-400
   workgroup-supply effect by itself.
3. If split is revisited, it must be an explicit non-bitwise candidate under
   the frozen v0.2 artifact-FP32-relative gate, JSON SHA-256
   `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`.
   It must demonstrate candidate error no worse than matched direct control on
   the frozen all-layer/hidden/logit/top-k and feedback-decode criteria before
   performance is considered. Whether it can pass is **unconfirmed**.
4. An exact-state merge that reproduces the direct recurrence would change this
   conclusion, but no such mechanism was identified in llama.cpp and none is
   established here. It remains research, not an implementation task.

## SQ8_0 decode-attention redesign — 2026-07-26

### 旧 split の矛盾を解消した

`1.236512x` の旧 full-model 値は C=513--519 / tile 128、すなわち 5 tile と
200 partial workgroup の測定だった。C=1,036 / 9 tile の raw trace は
`92160 / 256 = 360` partial workgroup を実際に起動しており、tile を一つの
workgroup 内で逐次処理したわけではない。旧 C≈516 の attention を同じ
unprofiled probe の 2.82103x で短縮する Amdahl 再計算は 1.227x となり、観測
1.236512x に近い。短い context の attention 以外の残差が主因である。

generic split は source row ごとの 2 CTA barrier も保持していた。separate
merge dispatch の存在は trace で確認したが、profiler range 時間を throughput
に使わず、merge が支配したとは結論していない。物理 HBM byte、L2 局所性、
achieved occupancy、launch overhead の寄与は gfx1201 PMC が有効でないため
未確認のままである。

### 実装と個別効果

実験経路は `b65e63c3` までの 3 commit で追加した。tile 20 generic split、
5 Q head が同じ KV head を共有する GQA grouped split、そして double-buffer
で source-row barrier を 2 から 1 へ減らす pipeline 版である。いずれも
環境変数 opt-in であり、既定 direct path は変更していない。

R9700/gfx1201 の isolated full-model M=1 decode（prompt 1,024、16 decode
step × 5 repeat、load/prefill/warmup 除外）では次を得た。

| step | full-model tok/s | 直前との比 | 解釈 |
|---|---:|---:|---|
| direct control | 15.228021012 | — | 有効既存 baseline 15.294955751 と -0.438% |
| generic tile 128 | 22.412990396 | 1.471826x | KV split |
| generic tile 20 | 23.872854841 | 1.065135x | より細かい KV split |
| GQA grouped tile 20 | **27.378731052** | **1.146856x** | 最速。semantic K+V 42,434,560 B → 8,486,912 B |
| grouped+pipelined tile 20 | 27.253516733 | 0.995427x | attention-only では速いが full model では採用しない |

最速 grouped は既存 baseline の 1.790050x、llama.cpp 30.468075023 tok/s の
89.8604% である。attention-only diagnostic は generic tile128→20 が 1.283039x、
GQA grouped がさらに 2.138791x、pipeline barrier 削減がさらに 1.080422x
だったが、これを full-model throughput としては扱わない。max abs
split-vs-direct は最大 1.11759e-7、non-finite は全変種で 0 だった。

### 数値、昇格、次段

lightweight promotion policy に従い top-1/logit exact を gate にしていない。
しかし current generic served-model manifest は `required_environment` に
boolean 名しか表せず、必須の `ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE=20`
を表現できない。そのため redesign を実際の service candidate として起動できず、
10 prompt の文章品質比較、generic promotion、rollback は未実施である。これは
品質 failure ではなく、品質が未確認であるため昇格しないという記録である。

F16/BF16 KV は semantic K+V byte を半減する余地がある。既存 direct attention
30.773224 ms/token が完全に半減するという帯域上限だけなら 20.002232 tok/s
（baseline 比 1.307767x）となるが、physical traffic と conversion cost は
未測定であり、redesign の実効予測には使わない。

証跡は `benchmarks/results/2026-07-26/sq8-decode-attention-redesign/` に保存した。
2 回の計測窓のうち第 1 窓は汚染として無効化し、第 2 窓だけを有効とした。最後に
verified start-limit から `reset-failed` と 1 回の start で
`ullm-openai.service` を active/running に復旧し、`llama-qwen35-udq4.service`
は inactive/disabled のまま維持した。


## SQ8_0 decode attention root-cause evidence — 2026-07-26

### Direct launch is substantially under-supplied

The production `ullm_paged_decode_attn_f32_kernel` direct fast path was
measured on isolated R9700 (`gfx1201`) at the actual Qwen3.5 shape: 40 Q
heads, 8 KV heads, head/value dimension 128, C=1,036. The launcher and raw
rocprof dispatch agree on global `(10240, 1, 1)` and workgroup `(256, 1, 1)`:
40 workgroups/layer and 320 wave32s/layer. Against 64 CUs x 32 waves/CU this
is a 15.625% queued-wave supply proxy, not achieved occupancy.

The direct body gives one workgroup a whole Q-head sequence: it serially
resolves the page table and processes all 1,036 source positions. Each score
reduction has two CTA barriers in the normal warp reduction, yielding 2,072
CTA barriers/workgroup/layer. K and V lanes are contiguous and coalesced inside
a page; page changes occur at the block boundary rather than as a per-lane
scatter. The same KV head is semantically reread for each of five GQA Q heads.

The existing llama.cpp capture is the relevant structural contrast: its vector
FATTN main dispatch has 400 workgroups/layer (1,600 waves; 78.125% supply
proxy), P=10 KV partials, followed by 40 combine workgroups/layer. This is
strong evidence for insufficient uLLM workgroup supply and serial scan cost,
but not a format-free comparison because llama.cpp uses F16 continuous KV
whereas this uLLM route uses F32 paged KV.

The C=1,036 unique KV footprint is 8,486,912 B. At a 640 GB/s reference roof
it takes 13.2608 us; the observed 769.3306 us direct dispatch is 58.0154x
larger. Counting the five GQA semantic re-reads gives 42,434,560 B and a
66.304 us roof, still 11.6031x below observed. Neither byte number is a
physical HBM observation. The root rocprof PMCs (`GL2C_EA_RDREQ_*`,
`SQ_INSTS_VALU`, and `SQ_WAVES`) are unusable here: GL2/VALU were zero and
SQ_WAVES contradicted the known geometry even under root. Thus achieved
occupancy, physical bytes, cache hit rate, and physical bandwidth remain
**unconfirmed**.

The old 640-call trace gives 30.773224 ms attention/token across 40 layers.
With the valid 15.294955751 tok/s baseline, this is 47.0675% of wall time. It
is 40.2933x the recorded llama.cpp attention total of 0.763730 ms/token; full
model throughput is a separate roughly 1.992x comparison (30.4680750229 vs
15.294955751 tok/s).

### Split diagnosis and safe implementation status

An isolated minimal paged-KV probe established that a single 128-token tile is
bit-identical to direct. With the same non-contiguous unique page table,
C=130/P=2 differs by max abs 2.9802e-8 (2,250 / 5,120 F32 bits), and
C=1,036/P=9 differs by 1.08033e-7 (4,934 / 5,120 bits), with no non-finite
values. Inspection and these cases rule out the checked initialization,
tail/empty-tile, causal/page-boundary, and obvious merge-scale fault classes;
they do not prove no latent bug at every shape. The best supported explanation
is finite-FP partial-softmax reassociation amplified by SQ8_0 sequential
activation quantization. Existing full-model hard top-1 regressions keep
multi-tile split in its direct-fallback containment.

An opt-in direct-order experiment,
`ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX=1`, makes one lane per
V-owning wave update duplicated scalar softmax state and broadcasts it locally.
It preserves token order and produced byte-identical C=1,036 output, but is
disabled by default. Its only collected host-call-plus-synchronize probe timing
did not improve (0.678809 vs 0.666713 ms) and is not a model benchmark.

An attempted full-model direct/candidate comparison is retained but excluded:
`ullm-openai.service` restarted at 20:19:35 JST during the measurements. Its
14.685730 / 14.959300 tok/s values are contaminated and must not be compared.
No valid post-change full-model tok/s, promotion, or default change exists.
The next action is one coordinated, fixed-HEAD isolated R9700 window that
first records valid output quality and then the full-model throughput; no new
split default is justified before that gate.

## `SQ8_0` attention redesign shipping follow-up — 2026-07-26

### `AQ4_0` applicability is a no-go, not a deferred promotion

The served production model is `AQ4_0` Qwen3.5-9B, not this Qwen3-14B
`SQ8_0` product.  Its source config has 32 layers arranged as eight repetitions
of `linear_attention` ×3 then `full_attention` ×1: 24 linear and 8 full
attention layers.  Full attention is 16 Q heads / 4 KV heads (GQA 4:1) with
head/value dimension 256.

The current C=1339, 32-step P3 ROCprof trace in
`benchmarks/results/2026-07-26/attention-redesign-shipping/phase1/current-p3-compatible-c1339-20260726T160603Z/`
attributes module-launched GPU dispatches to decode markers via correlated
`hipModuleLaunchKernel` launch time.  It finds no marker-contained
`ullm_paged_decode_attn_f32_kernel`; the split partial/merge core is 37.378910
ms of 411.411732 ms inclusive kernel time, or **9.08552%**.  The 16
partial/merge dispatches per step match the eight full-attention layers.  This
is current physical trace evidence and kernel-time composition, explicitly not
profiler-range throughput.  The earlier 8.97854% P3-compatible trace remains
as historical corroboration in the same evidence root.

BH's grouped tile-20 body is restricted to gfx1201, 5 Q heads per KV head, and
128-dimensional K/V.  `AQ4_0` therefore takes its generic fallback even if a
selector could be supplied; its deployed c4 source additionally accepts only
the separate AQ4 split tiles 128 or 256.  The linear layers are outside this
optimization.  A 4:1/256 variant would be a new kernel implementation and
needs a new full-model validation, so it is deliberately not called an
application of BH's redesign.  The conditional Amdahl ceiling obtained by
pretending the 1.790050x SQ8 result accelerated all current 9.08552% is only
1.0417747x (+4.18%), not an AQ4 performance forecast.  No `AQ4_0` promotion
was justified at this initial-audit point.

### `AQ4_0` 4:1×256 specialization and promotion

The literal `SQ8_0` body remains inapplicable, but its GQA-cooperative idea
was implemented as a separate, shape-closed `AQ4_0` body in source commit
`c8074928e22b27801df78d65508fdd619d13a748` (local branch
`bq-aq4-grouped-c807`).  Four consumer wave32s calculate the four query heads
of one KV head while four loader wave32s stage the one 256-wide K row and V
row in LDS.  The existing split workspace and merge body remain unchanged;
every other geometry takes the existing path.  This avoids incorrectly
describing the 5:1×128 tile-20 body as reusable for the 4:1×256 model.

The candidate full-model control at C=1339, six warmup steps and two alternating
32-step measured repeats was 74.110977 tok/s direct versus 74.509830 tok/s
grouped: **1.005382×** (+0.398854 tok/s).  These are profile-driver timed
decode intervals, not ROCprof range times.  All four 32-token greedy sequences
were identical; that is a narrow diagnostic, not the quality criterion.

Candidate manifest `69a5e1eb2e7713a1d017332539a587b9a13cf925cbfb28d7c89719ba6709ec2e`
then passed the same-model lightweight promotion suite.  All ten candidate
requests completed without automated blocking findings and their generated
text matched the P3 control in this deterministic suite.  `tools/promote-served-model.py`
performed one successful service restart and atomically promoted it.  The
active model remains `AQ4_0` Qwen3.5-9B; `SQ8_0` was not promoted.  Full raw
evidence is under
`benchmarks/results/2026-07-26/attention-redesign-shipping/aq4_0-grouped-final-c8074928-window-20260727T015800Z/`
and `aq4_0-grouped-promotion-c8074928-20260727T020500Z/`.

### Service-candidate execution contract

Commit `bfc76a72` adds an optional, fail-closed v2 manifest contract:

```json
{
  "worker": {
    "execution": {
      "paged_decode_attention": {
        "kernel": "gqa_grouped_split",
        "split_tile": 20
      }
    }
  }
}
```

Unknown keys and unsupported values fail validation.  `gqa_grouped_split`
remains limited to `SQ8_0`, gfx1201, `rdna4_w8a8_block_ck`, and tiles
`20|128|256|512`.  The separate `aq4_gqa_grouped_split` is limited to
`AQ4_0`, gfx1201, `rdna4_aq4_resident`, the split-HIP guard, and tile `128`.
Selectors cannot enter through `required_environment`.  Manifest-mode gateway
startup clears inherited experimental selectors and injects only the admitted
setting; the worker independently validates the exact state, and pipeline is
not representable.  The original P3 manifest validates unchanged with
`execution: null` and SHA-256
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`.
Promotion/rollback retain typed fields because they atomically swap raw bytes;
the round-trip test covers this.

### Isolated text evidence and result

The direct and grouped tile-20 `SQ8_0` manifests pin the same Qwen3-14B
product, tokenizer, worker, source commit, guard set, and fixed ten-prompt
suite.  They differ only in the worker execution contract (apart from
human-readable display labels).  In a single isolated window both completed
all ten real requests; `comparison-20260726T160603Z/` records no automated
request/empty/repetition/garble/length finding.  The zero exact-match rate is
only an observation, never a threshold.

Human reading nevertheless holds quality approval: the grouped Python case
does not provide the requested code, its JavaScript explanation says
`Boolean(NaN)` is true, and its Japanese multiturn answer stops incomplete.
Some direct controls are also truncated by the fixed response budgets, so this
does not prove every difference is caused by attention, but it is not evidence
to call the candidate text-quality-approved.  This stays a service-candidate
record only.  It must not use the promotion tool: promoting `SQ8_0` would
replace the active `AQ4_0` product model.

## `SQ8_0` prefill cached-prefix Flash2 GQA staging — 2026-07-26/27

The prefill investigation begins with actual selected kernel traces rather
than assuming that the decode remedy transfers.  On the R9700 F32-KV, M=128
path, `ullm_cached_prefix_attn_f32_flash2_kernel` accounts for 59.873173% of
uLLM summed selected kernel duration at prompt 512, 86.318790% at 2048, and
93.069535% at 4095.  It has 160/640/1280 calls respectively.  Each actual
dispatch is grid 1,310,720, block 256: 5,120 workgroups and 40,960 wave32s,
or 2,000% of the 64 CU x 32-wave queued-supply proxy.  This is not resident
occupancy, but it rules out treating prefill as the decode-like 15.625%
under-supply case.

The matching llama.cpp Q8_0 F32-KV trace emits 40 `flash_attn_ext_f16<...>`
calls at each size, with grid `(128,4,40)`, block `(32,4,1)`: 160 workgroups,
640 wave32s, and a 31.25% queued-supply proxy.  Its attention composition is
3.767997% / 11.275091% / 19.279610%, while `mul_mat_q` is
85.364380% / 75.717628% / 65.722592%.  The symbol name does not override the
recorded F32-KV configuration.  The distinct trace selections and summed
kernel durations are composition evidence only, never tok/s.  Physical HBM
bytes, cache-hit behavior, achieved occupancy, and a memory-bound conclusion
remain **未確認**.

The retained prefill fast path is therefore GQA reuse without a decode-style
split-C reduction: on gfx1201 only, with F32 KV, 5 Q heads per KV head, and
128-dimensional K/V, one CTA owns `(new token, KV head)`.  It stages each
20-token K segment, then each V segment, once in LDS and serially processes
the five Q heads.  Each head retains generic Flash2's 256-thread score/max/sum
trees, token order, and 64-token online-softmax boundary.  K and V reuse one
staging allocation rather than co-residing.  Other shapes/GPU paths retain
generic Flash2; the pre-existing staged-wave32 body remains a separate
explicit opt-in experiment.  `ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED=1` is the
A/B fallback.  This is semantic K/V reuse; it is not a claim that physical HBM
traffic became exactly one fifth.

The same candidate executable, with only that fallback setting changed for
control, gives the following five-repeat full-model prefill rates:

| prompt | generic tok/s | serial GQA tok/s | ratio |
| ---: | ---: | ---: | ---: |
| 128 | 865.157 | 883.021 | 1.020648x |
| 512 | 520.351 | 561.905 | 1.079858x |
| 1024 | 338.308 | 358.745 | 1.060409x |
| 2048 | 189.737 | 196.585 | 1.036094x |
| 4095 | 100.586 | 105.040 | 1.044275x |

The 128/4095 ratio improves only 8.601136x -> 8.406534x.  Thus the result is
a retained local improvement, not a claim that the long-prefix curve is flat:
the same-condition llama.cpp Q8_0/F32-KV reference remains 1,165.756,
1,195.722, 1,145.351, 1,058.379, and 1,008.683 tok/s, leaving a 9.603x gap at
4095.  It was selected on full-model data, not an attention-only probe.

The generic/candidate full-model oracle is F32-byte exact for hidden state and
logits at all five prompt sizes (`max_abs=0`, `relative_l2=0`, no non-finite,
same top-1/generated token).  This is recorded as review evidence under the
lightweight policy, not as a new scalar numerical gate.  The BK cursor-rewind
tail implementation is untouched; the 4095 oracle records expected cache
lengths and 32 prefill advances.  The earlier wave32/exact-tile64 direction
was rejected after a real arithmetic-path difference in the full-model oracle,
not by an arbitrary threshold, and its partial timing is not used here.

The BH grouped tile-20 decode selector was rerun against the current BR
worktree build and reached 27.411786 tok/s versus the 27.378731 reference
(1.001207x), so this prefill source change did not regress the measured decode
condition.  Evidence, source provenance, service/thermal records, and raw
traces are under
`benchmarks/results/2026-07-26/prefill-attention-redesign/`.  No manifest or
service configuration was changed and no `SQ8_0` promotion was attempted.

## `SQ8_0` prefill resident-width expansion — 2026-07-27

The M=128 limit was split into two different contracts.  The serving
`resident_stack_width()` is not an attention tile size: it is the common M
used to allocate the resident stack's layer workspace and hidden buffer, the
serving prompt-chunk hidden buffer, and CK activation/projection workspaces.
The fixed width is therefore a real allocation/shape contract, but its use of
M=128 is not inherently required by Flash2.

`Sq8ServingPrefillMode::fixed_chunk_tokens(M)` now lets the serving scheduler
select a power-of-two width from 2 through the 4096-token context limit.  The
default loader remains M=128.  The `sq8_ck_serving` CLI accepts the same
selection as `m<N>-chunk<N>`.  Short tails retain BK's real-token cursor
rewind: the final M-wide execution overlaps already processed real tokens and
commits only the outstanding suffix.  It never creates a fake token, padding,
or an attention mask.

This is intentionally two-stage admission.  The scheduler accepts and tests
M=256/512/1024/2048/4096, while model loading still requires the measured
lower runtime contract.  At this point
`Qwen3Sq8LayerConfig::validate`, the stack, Rust CK wrapper, and
`ullm_runtime_api_sq8_ck.inc` only admit `{1,2,4,8,16,32,128}`.  A requested
wide M therefore fails before model allocation with an explicit diagnostic;
it is not misreported as a completed GPU run.

For N=4095, the scheduler has the following consequences across 40 layers:

| M | fixed execution units/layer | planned Flash2 calls | tail execution / logical commit |
| ---: | ---: | ---: | --- |
| 128 | 32 | 1,280 | `3967..4094` / 127 |
| 256 | 16 | 640 | `3839..4094` / 255 |
| 512 | 8 | 320 | `3583..4094` / 511 |
| 1024 | 4 | 160 | `3071..4094` / 1,023 |
| 2048 | 2 | 80 | `2047..4094` / 2,047 |
| 4096 | no legal fixed replay at N=4095 | 163,800 M=1 calls | no synthetic row permitted |

These rows are scheduler-unit-test results, not a trace of dispatches.  The
full model cannot reach a wider Flash2 dispatch until the lower contract is
extended.  M=2048 is the largest useful no-padding candidate at N=4095;
M=4096 is capacity-valid only for an exact 4096-token prompt.

The allocation calculation at
`benchmarks/results/2026-07-27/prefill-chunk-width/memory-accounting.md`
shows 539,648 B per resident token.  Even M=4096 gives a requested SQ8_0
total of 18.519 GiB and leaves 6.424 GiB analytically after adding the
observed 7.426 GiB AQ4_0 Qwen3.5-9B allocation to the 31.859 GiB R9700.
Allocator/module overhead and a true co-resident load remain unmeasured, so
this is capacity evidence rather than a production co-residency claim.

No BX-owned Flash2 source change is required by this width extension.  The
F32 cached-prefix launcher already passes `new_tokens` dynamically and uses
no persistent M-sized attention scratch; its relevant Qwen3-14B restriction
is `value_dim <= 256`, not M=128.  The generic CTA uses 1,296 B LDS and the
selected grouped-GQA CTA uses 12,624 B LDS, both independent of M.  The next
owner should instead make the following lower-runtime change after the direct
CK shape probe and a full-model M=256 smoke both succeed:

1. extend the layer and stack measured-M validation/list to the selected
   widths (start with 256, then 512/1024/2048);
2. extend the Rust CK and C++ API M whitelists in lockstep, preserving their
   existing generic `m,n,k` CK argument construction;
3. load a fresh resident model at each M and run hidden/logit diagnostics plus
   actual generated text under the lightweight policy; and
4. trace the completed full-model runs and time the five prescribed prompt
   lengths using the existing non-profiled, five-repeat accounting.

The direct CK prerequisite has completed: in one short R9700 locked window,
the existing helper accepted M=256/512/1024/2048/4096 for all four
Qwen3-14B projection shapes, including activation quantization and output
conversion.  `wide-m-ck-shape-probe.jsonl` has all 24 zero-buffer shape rows.
This does not remove the need for the M=256 full-model smoke: the layer/stack
and public API gate still intentionally reject the unlisted widths, and the
probe neither uses real weights nor evaluates numerical fidelity.

The width scheduler tests, allocation accounting, direct CK probe source, and
explicit unmeasured status are retained under
`benchmarks/results/2026-07-27/prefill-chunk-width/`.  No kernel-only speedup
is used as an acceptance decision.
