# SQ8_0 R9700 attention Phase 1 evidence

Date: 2026-07-26
Scope: independent SQ8_0 execution on R9700 only (AMD SMI GPU 2, PCI 0000:47:00.0, gfx1201). No production symbol, external ABI, dispatch boundary, active-model file, unit-file content, campaign, candidate, release, or /opt/ullm content was changed.

## Direct answer

The measured SQ8_0 paged-decode attention body is the wave-shuffle implementation, not the full shared-memory reduction fallback.

The proof is tied to one fresh, isolated process per condition:

| condition | selected runtime code object | selected ISA evidence | selected trace evidence |
|---|---|---|---|
| default environment, both reduction-disabling variables absent | _code_object0010.o, SHA-256 26fa813c4b2d35e90361ff50c6648d1d9d5412da041658c189f2fcbc095b6bb1 | ullm_paged_decode_attn_f32_kernel has 10 ds_bpermute instructions, 2 LDS rendezvous points, 2 ds_load_b32 and 2 ds_store_b32 | 640 launches of ullm_paged_decode_attn_f32_kernel; grid 10240, block 256 |
| explicit ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE=1 negative control | _code_object0010.o, SHA-256 b856136847c042ab6713f8dd4e30d14799a59ff580829993b2acb0156fb1e9fa | no ds_bpermute; 9 LDS rendezvous points, 9 LDS stores, and 9 LDS loads | same selected kernel name and geometry, but a distinct JIT object |

GPU_DUMP_CODE_OBJECT=1 wrote the objects in the CWD of exactly the traced fresh process. Symbol lookup identifies the selected kernel in the dumped object; the default trace selects that kernel, and the disassembly in static/isa/paged-default-runtime.s is therefore direct runtime evidence rather than source-only inference.

The active ullm-openai.service before this window was an AQ4 worker, not an SQ8_0 worker. Altering the active model was forbidden, so a literal live-SQ8_0 service observation is unavailable. The isolated driver instead loaded the production SQ8_0 artifact and package recorded below, used the same R9700 HIP runtime and required-kernel guards, and captured the runtime object in that process. This is the precise scope of the conclusion.

## Selection conditions and dispatch

The source checksum is ad050da7137df13ba7f088099085f1539f9f1ea818e525306fc91e9a4f032b57.

- paged_decode_attn_kernel_source_for_arch checks only whether ULLM_DISABLE_PAGED_DECODE_WARP_REDUCE exists in the environment. Its value is not parsed, so setting it to 0 still enables the full-LDS path. The remediation for a mistaken setting is to remove the variable entirely.
- The architecture parameter is explicitly unused in this selector. Device capability and CK feature selection do not choose the paged reduction path.
- ULLM_DISABLE_PAGED_DECODE_ONLINE_SOFTMAX is a separate presence test for the two-pass softmax source preamble. It was absent and is not the reduction selector.
- The actual trace has grid 10240 and block 256, hence 40 workgroups. The launcher uses q_heads workgroups only in the head-parallel fast branch; the measured SQ8_0 geometry is q_heads=40, kv_heads=8, head_dim=value_dim=128, so its fast-branch predicates hold.
- The isolated process had ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1. That guard only makes an HIP launch failure fail closed instead of entering the staging fallback; it does not select shared reduction. The direct JIT object and successful HIP trace rule out staging for this capture.
- The decoder calls the legacy paged interface. The separate split API is compiled and available as an explicit alternative, but no split partial or merge kernel appears in the selected trace.

## Measurement protocol and identity

The selected region excludes model load, seed prefill, four decode warm-up steps, finish, and reset. Decode is 16 M=1 steps over cache length 1028 -> 1044. Prefill is a 1024-token prompt advanced in eight M=128 chunks.

| item | observed value |
|---|---|
| R9700 identity | AMD SMI GPU 2, 0000:47:00.0, gfx1201, 64 CU |
| isolation | HIP_VISIBLE_DEVICES=1; driver reports runtime device 0 as gfx1201 |
| ROCm | 7.2.1 |
| driver SHA-256 | 075a780837f9f124aa32ed152fd6316edbfc83286df691bf92c661d33d198444 |
| artifact content SHA-256 | 2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147 |
| package manifest SHA-256 | c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb |
| service precondition | llama-qwen35-udq4.service=inactive/dead/disabled; gdm.service=inactive/dead/static |

All eight steps in the single stop -> isolate -> restore window returned exit status 0. See service/window-events.log and service/after-restore.txt.

## Measured timing

Profiler dispatch durations and unprofiled timing answer different questions and are kept separate.

| measurement | default | forced shared-LDS control | interpretation |
|---|---:|---:|---|
| paged attention, selected trace | 492.371584 ms / 640 launches | 536.261934 ms / 640 launches | fallback is 8.914070% slower; default uses 43.890350 ms less over 16 steps |
| paged attention per decoded token, selected trace | 30.773224 ms | 33.516371 ms | 2.743147 ms/token reduction in the attention body |
| unprofiled decode, 5 repeats | 1.041134988 s / 16, 15.367844 tok/s | 1.090319008 s / 16, 14.674604 tok/s | default gives 4.724077% more tok/s; it reduces time by 4.510975% relative to the forced fallback |
| Flash2 attention, selected prefill trace | 2196.476598 ms / 320 launches | not applicable | 75.63% of summed selected kernel duration |
| unprofiled prefill, 3 repeats | 3.037386960 s / 1024, 337.131888 tok/s | not applicable | recorded here because it was previously unconfirmed |

The forced fallback is a negative control, not a proposed production configuration. Because the default path is already active, removing the fallback cannot improve the current system: its current incremental gain is 0%. If an environment mistake made the variable present, removing it has a measured recovery of 4.724077% unprofiled decode throughput at this workload.

## Attention time decomposition

### Reduction and synchronization

The source-level rendezvous counts are per reduction invocation, not a claim of a hardware-cycle breakdown.

| body | reduction structure | source-level CTA rendezvous |
|---|---|---:|
| paged default | wave32 shuffle inside each wave, then one 8-wave LDS handoff | 2 per score; 2 * C = 2072 at C=1036 |
| paged forced fallback | write 256 partials, then 8 full-LDS halving stages | 9 per score; 9 * C = 9324 at C=1036 |
| Flash2 | F32 64-token tile; full LDS QK score, max, and sum trees | 10 per score, 10 for tile max, 10 for tile sum, 1 final = 661 per full 64-token tile |

The emitted default ISA contains two static s_barrier_signal/s_barrier_wait pairs and the fallback contains nine pairs. Flash2 contains ten static pairs. Loops make their dynamic count depend on context or tile count; the source-level counts above make that dependence explicit.

### Tile, workgroup, and resource evidence

| body | trace geometry | runtime trace allocation | code-object metadata | implication |
|---|---|---|---|---|
| paged default | 40 workgroups per layer, 8 wave32/workgroup | LDS 1024 B, VGPR 32, SGPR 128 | LDS 1024 B, VGPR 25, SGPR 52, no spills, private 0 | 320 waves total; R9700 exposes 64 CU and max 32 waves/CU, so the dispatch supplies only 15.625% of that 2048-wave machine ceiling |
| paged forced fallback | same geometry | LDS 1024 B, VGPR 32, SGPR 128 | LDS 1024 B, VGPR 25, SGPR 52, no spills, private 0 | resource totals alone do not expose the changed reduction; ISA does |
| Flash2 | 5120 workgroups per layer launch, 8 wave32/workgroup | LDS 1536 B, VGPR 24, SGPR 128 | LDS 1292 B, VGPR 21, SGPR 46, no spills, private 0 | 80 workgroups/CU are queued per launch, so it has sufficient grid parallelism |

The static and runtime fields are different reporting layers and are deliberately not conflated. R9700 reports a 64 KiB group-memory pool and a 32-wave/CU maximum. The stated LDS use does not limit a four-workgroup/32-wave theoretical ceiling, but achieved residency and register-limited occupancy were not measured and remain 未確認.

A split decode with source_tile=256 at C=1036 would use five splits: 200 partial workgroups, 1600 wave32s, and a 104000-byte workspace per attention invocation before the 40-workgroup merge. That is a 78.125% wave-slot envelope, not a predicted speedup; its merge, workspace traffic, and numerical behavior have not been measured.

### Load form and logical bandwidth

Static instruction counts in the selected bodies are:

| body | global load form | LDS load/store form | wide global-load conclusion |
|---|---|---|---|
| paged default | 11 global_load_b32 | 2 ds_load_b32, 2 ds_store_b32 | no global b128 instruction |
| Flash2 | 7 global_load_b32 | 16 ds_load_b32, 1 ds_load_b128, 9 ds_store_b32, 1 ds_store_b64 | its LDS b128 is not an HBM/global wide load |

The scalar global mnemonic is per lane. The source maps adjacent lanes to adjacent head/value elements, so it is not proof of a narrow physical memory transaction. A direct uint4 substitution is therefore not justified. A lane-remapping/tile redesign must prove a transaction or register-pressure improvement and preserve the reduction contract.

For a semantic KV scan metric, each query head reads one F32 K and one F32 V vector of 128 elements for every attended timestep. This deliberately counts GQA KV reuse across query-head workgroups as logical reads and does not include physical cache reuse.

| scope | logical KV bytes | selected attention duration | logical rate | ratio to 640 GB/s |
|---|---:|---:|---:|---:|
| decode midpoint C=1036 | 1,697,382,400 B/token; 27,158,118,400 B for 16 steps | 492.371584 ms | 55.157770 GB/s | 8.618402% |
| causal prefill 1..1024 | 859,832,320,000 B | 2196.476598 ms | 391.459814 GB/s | 61.165596% |

Both bodies have a logical QK-plus-weighted-V arithmetic intensity of 0.5 FLOP/B before softmax overhead. Decode consequently shows both severe grid underfill and long reduction dependence. Flash2 is much closer to the nominal logical bandwidth roof but still contains its full-LDS trees. These facts identify structural targets; they do not classify physical HBM versus ALU saturation.

The PMC capture confirms geometry through nonzero Wavefronts: 320 per paged dispatch and 40960 per Flash2 dispatch. However, all 1080 paged and all 360 Flash2 samples recorded FETCH_SIZE=0 and VALUInsts=0. The cause of those zero counters is 未確認. They are unusable as physical-byte or compute-throughput evidence, so physical HBM efficiency and a final memory-bound/compute-bound classification are 未確認.

## Updated optimization order

| priority | candidate | confirmed quantitative basis | achievable performance claim | required next measurement |
|---|---|---|---|---|
| P1 | retain and test the wave-shuffle reduction admission condition | current default object is shuffle; forced full-LDS loses 4.724077% decode tok/s | current gain 0%; a mistaken present variable has a measured 4.724077% recovery | fresh-process environment admission test that rejects or records either disable variable; no product environment change in this task |
| P2 | Flash2 staged wave32 reduction prototype | 75.63% selected prefill share; 661 source-level rendezvous/full tile; no static spill | 未確認. Removing source barriers is a structural target only, not a time estimate | distinct symbol; replace QK, max, and sum one class at a time; differential short/tail/real prompts; metadata/ISA and unprofiled prefill comparison |
| P3 | decode context split/tile experiment | legacy path supplies only 40 workgroups/layer; source_tile=256 has a 200-workgroup, five-split alternative | 未確認. 15.625% -> 78.125% is a work-supply envelope, not a speed prediction | isolated differential against direct path for source tiles 128/256/512, then attention time plus end-to-end decode timing |
| P4 | Flash2 load/lane re-layout, optionally uint4 within a new tile | selected ISA has global_load_b32 but lane-contiguous source access; no physical FETCH_SIZE | 未確認; no standalone uint4 benefit is established | prove alignment/coalescing with usable physical counters, static VGPR/LDS comparison, and full softmax differential |
| P5 | resource-only occupancy tuning | current static bodies have modest LDS, no spills, and enough prefill workgroups | 未確認; no evidence supports an isolated resource tweak first | only evaluate as part of P2/P3; reject a candidate that increases spills or reduces the measured winner |

This supersedes the old helper-first ordering. The quantizer and RMSNorm shuffle ideas remain valid isolated work, but their combined decode share is only 2.6067%; they no longer justify consuming a full R9700 service window before the attention candidates.

## CDNA3 handoff design

The shared contract must remain semantic, not a copied lane implementation:

~~~text
canonical SQ8_0 payload/scale semantics
  + attention input/output, paged-KV, causal-mask, and online-softmax contract
  + common shape validation, adversarial vectors, real-artifact vectors, and differential harness
  + common timing and code-object evidence schema
       |
       +-- gfx1201/R9700 private body: wave32 shuffle, RDNA4 lane and LDS layout
       `-- CDNA3 private body: wave64, MFMA fragments, CDNA3 lane and LDS layout
~~~

The canonical SQ8_0 payload and BF16 scale interpretation is shared at the projection-to-attention boundary. The attention implementation itself must separately preserve F32 KV indexing, GQA mapping, online-softmax order, and gated-output behavior. Neither the gfx1201 shuffle lane map nor its LDS layout may be reused by the CDNA3 MFMA body. Each body needs its own code object, metadata, ISA audit, and differential result against the common vectors.

## Service and thermal record

At 04:21:33 JST, ullm-openai.service was stopped after confirming llama-qwen35-udq4.service inactive/dead/disabled. GPU 2 had no remaining process before the isolated profiler began. The window ran from 04:21:33 to 04:28:33 JST. The first service start succeeded at 04:28:34 JST; reset-failed was not needed. Final state is ullm-openai.service=active/running/enabled, llama-qwen35-udq4.service=inactive/dead/disabled, and gdm.service=inactive/dead/static.

Before the window, GPU 2 telemetry was edge/hotspot/memory 37/38/34 C, GFX 66 MHz, and socket power 12 W. Across the recorded window samples, edge was 36..72 C, hotspot 37..92 C, memory 34..76 C, GFX 2..3435 MHz, memory clock 96..1258 MHz, and socket power 8..398 W. AMD SMI reported both THROTTLED and UNTHROTTLED samples; the status string is recorded without assigning a thermal cause.

## Result map

- decode/default-trace/ and decode/shared-fallback-trace/: raw selected-region rocprof CSV.
- decode/pmc/ and prefill/pmc/: raw counter CSV; zero-counter limitation documented above.
- prefill/trace/: raw Flash2 selected-region rocprof CSV.
- decode/code-objects/ and prefill/code-objects/: GPU_DUMP_CODE_OBJECT runtime objects.
- static/isa/: selected-symbol disassemblies and metadata extracted from the runtime objects.
- logs/: raw child stdout/stderr and exit statuses; intentionally ignored by Git but retained in the result directory.
- service/, telemetry/, and preflight/: lifecycle, GPU identity, environment, and thermal records.
