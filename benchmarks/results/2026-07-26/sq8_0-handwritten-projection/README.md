# SQ8_0 gfx1201 private handwritten projection feasibility record

## Scope and verdict

This is an investigation-only, M=1/decode prototype. It adds a private gfx1201
WMMA symbol and an explicit test profile; it does not alter the public ABI,
legacy dispatch, normal CK selection, active-model manifest, campaign, or
release state.

The prototype is a **numerical NO-GO**. Its four isolated projection checks
were bitwise exact after the CK BF16 workspace boundary, but the frozen
full-model feedback-decode gate failed from the first captured step. Per the
predeclared policy, candidate event timing was not run. This record establishes
resource headroom, not a speedup claim.

The precise source-level cause of the full-model mismatch is **unconfirmed**.
Different FP8 WMMA fragment/lane and K128 scale-block accumulation association
from CK are hypotheses to test, not a diagnosis.

## Evidence map

| Evidence | Location | Result |
| --- | --- | --- |
| Static code-object audit | static/handwritten-isa-summary.json | gfx1201, wave32, 8 WMMA instructions, LDS 1,280 B, VGPR 47, SGPR 24, no spills |
| CK M=1 control timing | attempt-2/component/ck-baseline.json | event-timed selected CK helper plus BF16-to-F32 boundary |
| Component differential | attempt-2/component/gate.json | 4/4 actual projection shapes finite and F32-bitwise exact at that boundary |
| Full-model feedback gate | attempt-2/full-model-multistep/gate.json | FAIL: all captured hidden/logit vectors differ despite 4/4 equal greedy IDs |
| No candidate timing receipt | attempt-2/component/handwritten-measurement-not-run.txt | timing correctly withheld after the full-model failure |
| Service/GPU evidence | attempt-2/{preflight,service,telemetry}/ | R9700-only isolation, telemetry, and restore record |

The raw attempt-2 component fixture had narrower payload/activation-scale
coverage than the subsequently strengthened source fixture: it used a small
finite OCP E4M3FN payload cycle and BF16-origin activation scales. It must not
be generalized beyond its recorded boundary check. The authoritative decision
is the actual-artifact full-model gate above.

## CK comparison baseline

All model M=1 cases are M tails against CK's MPerBlock=16; N and K are exact
multiples of 128. The selected forms are q/o and k/v: Default 16x128x128;
gate/up: KPadding 16x128x256; down: Default 16x128x256.

| Family (calls/layer) | CK form | event time / launch | logical route rate | logical / 640 GB/s reference |
| --- | --- | ---: | ---: | ---: |
| q/o (2) | Default 16x128x128 | 26.2118 us | 1,001.72 GB/s | 1.5652 |
| k/v (2) | Default 16x128x128 | 26.8975 us | 195.39 GB/s | 0.3053 |
| gate/up (2) | KPadding 16x128x256 | 158.3728 us | 563.61 GB/s | 0.8806 |
| down (1) | Default 16x128x256 | 148.9054 us | 599.03 GB/s | 0.9360 |
| all seven | mixed | 571.8696 us / layer | 578.36 GB/s | 0.9037 |

Logical route rate is recorded logical payload/scale/output traffic divided by
HIP-event time. It is **not physical HBM achieved bandwidth**: the available
PMC byte counters were unusable, and q/o exceeding the nominal reference
demonstrates why the ratio is a logical-reference metric only. Physical HBM
roofline efficiency remains **unconfirmed**.

For static occupancy comparison, CK's 36,864-B 128x256 forms admit one
8-wave32 workgroup per 64-KiB-LDS CU (25% of the prior 32-wave reference);
the 34,816-B 256x128 form has the same one-workgroup/25% LDS-only ceiling;
the 18,432-B 128x128 form admits three workgroups / 24 waves (75%). The
prototype has one wave32 workgroup and 1,280 B LDS, so 32 such workgroups
would consume 40,960 B: LDS alone does not prevent reaching the 32-wave
reference. This is a static resource bound, not measured runtime occupancy.

The pre-correction attempt-2 query reported threads_per_block=1024 from the
device capability rather than the 32-thread launch; source now returns the
actual launch width, but it was not remeasured to avoid another service window.
Its active_blocks_per_cu=51 is HIP's per-multiprocessor term and must not be
relabeled as confirmed CU occupancy.

## Frozen numerical policy and observed result

Before timing, the candidate had to satisfy both:

1. every actual M=1 projection shape is finite and F32-bitwise identical to CK
   after CK's BF16 workspace boundary; and
2. as the real full-model M=1 projection route, at least two feedback decode
   captures have exact generated IDs, top-1 logits, final hidden state, and
   logits relative to CK.

The component condition passed 4/4. The full-model condition failed at all
three recorded feedback steps. Hidden mismatches were 5,120/5,120 at every
step (max abs 0.387939, 0.797844, 1.287994); logit mismatches were
151,936/151,936, 151,935/151,936, and 151,936/151,936 (max abs 0.189508,
0.183819, 0.250601). All values were finite and generated IDs remained
[66, 198, 197, 197], which is insufficient under this frozen policy.

## Isolated service record

Two stop/isolate/restore attempts were made. Attempt 1 (08:30:12--08:30:48
JST) aborted before any GPU launch because AMD SMI's no-process sentinel was
parsed as a live process; the service was restored. Attempt 2
(08:31:52--08:33:27 JST) performed the R9700-only work and restored
ullm-openai.service to active/running with NRestarts=0.
llama-qwen35-udq4.service was recorded inactive/disabled and gdm3 inactive.
No V620 execution, unit-file change, power-cap/profile change, activation,
authorization, or remote action occurred.

Attempt-2 telemetry contains 93 one-second AMD SMI samples: edge 36--46 C,
hotspot 37--60 C, memory 34--48 C, gfx 0--3421 MHz, memory 96--1258 MHz, and
socket power 7--204 W. It reports 22 THROTTLED and 71 UNTHROTTLED states. The
physical reason for the throttle state is **unconfirmed** by these samples;
timing must be treated as conditional on that fact.

## Next decision point

Do not time or promote this body. First capture actual-artifact projection
inputs and CK outputs at the first divergent layer, then make the candidate
match CK's K128 scale-block and fragment/reduction order exactly. Repeat the
strengthened component gate and the same multi-step full-model gate before a
new, separately approved service window is considered.
