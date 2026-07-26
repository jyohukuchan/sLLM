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
