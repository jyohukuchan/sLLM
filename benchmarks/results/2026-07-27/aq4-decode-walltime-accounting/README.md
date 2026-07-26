# AQ4_0 decode wall-clock accounting

## Scope and provenance

This record answers a narrow question: how much of the Qwen3.5-9B `AQ4_0`
decode wall clock is GPU kernel execution, and what the remaining time is.
It never treats a rocprof marker duration as throughput.

- Active served-manifest SHA-256 at audit start:
  `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`.
- Active worker: `/opt/ullm/aq4-p3-deployment-v0.1/releases/aq4-p3-c4c9a9b3/ullm-aq4-worker`,
  SHA-256 `ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`.
- Diagnostic profile binary source: `c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`; binary SHA-256
  `1699e15de7cc54c3b8d87a5fdd3a80cda598d9cb79d9123bdfd7ad6185e0338c`.
- Product package manifest SHA-256:
  `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad`.
- The supplied direct, unprofiled timing is 32 decode steps at C=1339:
  73.4568 tok/s = **13.613452656 ms/token**.  It is a direct kernel-path
  driver timing, not an HTTP gateway timing.

The raw historical P3 rocprof capture is retained as an input because it is
the only pre-existing trace with kernel start/end timestamps.  Its source
predates the active worker, so every number from it is labelled *historical*
below.  The current paired capture described next is the validation that keeps
the historical numbers as a cross-check rather than treating them as the
current production partition.

## Current paired capture: final accounting

The validation capture completed in
`current-p3-c4c9a9b3-service-window-20260727T011830+0900/capture/`.  It used
the exact active manifest, worker and profile-binary hashes before and after
the capture (the three values above), `gfx1201` only, and C=1339 for 32
measured steps.  All 32 greedy tokens were again `4445`.  The profile trace
has one queue and one stream, no overlapping GPU dispatch intervals, and the
same 292 module launches / 294 all-GPU dispatches per token as the historical
trace.

The two rows below answer different but necessary questions.  The first is a
physically matching rocprof run: its CPU `Instant` samples and its kernel
timestamps are from the same 32 steps.  The second uses an immediately
adjacent **unprofiled** direct run and the current trace's kernel sum; that is
the closest production-mode normalization, but it is still a cross-run
comparison rather than a literal partition of one execution.

| Basis | Wall ms/token | Module-kernel ms/token | Kernel-external ms/token | Kernel share |
|---|---:|---:|---:|---:|
| Current paired rocprof run | 14.419178 | 12.831467 | 1.587711 | **88.9889%** |
| Current same-window unprofiled direct run, normalized by current kernel sum | 13.458181 | 12.831467 | 0.626714 | **95.3432%** |
| Supplied 13.613453 direct wall, normalized by current kernel sum | 13.613453 | 12.831467 | 0.781986 | 94.2558% |

The supplied original figures answer the narrow literal arithmetic question as
`412,275,120 ns / 32 / 13.613452656 ms = **94.6387%**` (12.883598 ms kernel,
0.729855 ms residual).  That number mixes the older trace with a later direct
run.  The current capture removes the source/build ambiguity and shows the
same conclusion: **AQ4_0 decode is kernel dominated, not
kernel-external dominated.**  The 88.99% and 95.34% values are observed
instrumented and unprofiled-normalized modes, respectively; they are not a
confidence interval and must not be collapsed into one fabricated percentage.

The unprofiled direct run was 74.3042 tok/s (13.458181 ms/token), versus the
promotion record's 73.4568 tok/s (13.613453 ms/token).  It is a fresh
same-binary run, not a claim that serving throughput changed.

### Current profiled-run closure and GPU idle time

The 32 current rocprof steps close exactly, with all terms in ms/token:

| Term | ms/token | Evidence / interpretation |
|---|---:|---|
| module-launched GPU kernels | 12.831467 | 9,344 module dispatches / 32 |
| GPU gaps between every dispatch | **1.487833** | Direct sum of 47.610653 ms of timestamp holes; GPU idle in this profiled timeline |
| D2H copy GPU execution | 0.006270 | Two `__amd_rocclr_copyBuffer` dispatches/token |
| marker leading + trailing no-GPU time | 0.081632 | 0.034309 + 0.047323; host/runtime boundary around the marked step |
| outer `Instant` scope outside marker | 0.011976 | Difference after the four timestamp-derived terms |
| **paired profiled wall** | **14.419178** | Exact sum |

The gap total is not a host-launch accounting shortcut.  Of the 1.487833
ms/token, 1.317145 ms/token (88.53%) precedes a dispatch whose correlated
`hipModuleLaunchKernel` call had **already returned** before the prior GPU
dispatch finished.  Only 0.163308 ms/token has the next API start after the
prior GPU end, and 0.007380 ms/token has the next API still in progress.
Thus the dominant holes are already queued work waiting on GPU/runtime or
profiling behavior; the trace does not identify a causal split between those
two possibilities.

### What is actually outside the kernels

- `hipModuleLaunchKernel` is called exactly 292 times/token.  A standalone
  `gfx1201` module-launch probe measured **1.553198 microseconds/call mean**
  (1.568368 microseconds median) without rocprof.  Multiplied by 292, its
  base API enqueue cost is 0.453534 ms/token.  This is a useful scale, not an
  additive attribution: enqueue normally overlaps preceding GPU work.
- The same probe under rocprof measures 9.238779 microseconds/call mean; the
  current full trace reports 37.444526 microseconds/call for the API.  Both
  are profiler-instrumented durations and must not be summed into wall time.
- There are exactly two `hipMemcpyDtoHAsync` calls and one
  `hipStreamSynchronize` per token.  The D2H API calls average 982.533
  microseconds in the trace because they overlap the final LM-head execution;
  their actual GPU copy execution is only 6.270 microseconds/token.  The
  one stream synchronization averages 28.921 microseconds in the trace and
  occurs after those copies.
- The measured direct driver contains neither HTTP handling, request
  scheduling, nor tokenizer work in a decode step.  Its remaining CPU work is
  greedy top-1 partial-pair decoding plus CPU-only publish/advance bookkeeping.
  That CPU portion has no separate unprofiled timer, so it remains
  **unattributed** rather than guessed.

HIP Graph work is therefore not the first optimization: even an unrealistically
perfect removal of all 292 measured base enqueue costs would be only a 3.37%
latency scale against the fresh direct wall, and it has not been shown to be on
the critical path.  Similarly, gaps immediately before the 97 normalization
dispatches are 0.463949 ms/token, a 3.45%-of-direct-wall **profiled-timeline
ceiling** for perfect adjacent fusion, not a predicted speedup.

## Trace token count: 32, independently established

The raw outer marker ranges are contiguous `step_index=0..31`, with
`cache_start=1339..1370` and exclusive end 1371.  The kernel cardinalities
independently give the same count:

The checked topology is 32 layers arranged as eight `linear × 3 → full`
cycles (24 linear and 8 full layers), with 16 Q heads, 4 KV heads and
head_dim 256.  The head geometry does not multiply the number of launches in
this M=1 path because the relevant Q/K/V work is fused per layer; it explains
why there are eight full-attention Q/K/V triple and split/merge pairs rather
than a launch per head.

| Per decode token | Per 32-token trace | Why |
|---|---:|---|
| AQ4_0 projection: 129 | 4,128 | 2 `matvec_add` + 1 `silu_mul` for each of 32 layers; 24 linear-attention fused qkv/z/a/b; 8 full-attention q/k/v triples; one final LM head |
| full attention: 16 | 512 | split partial + merge for 8 full-attention layers |
| linear attention: 48 | 1,536 | qkv-prepare + recurrent for 24 linear-attention layers |
| normalization: 97 | 3,104 | 32 input RMSNorm + 32 post RMSNorm + 24 linear segmented RMSNorm/SiLU + 8 q/k norm/RoPE + final RMSNorm |
| other: 2 | 64 | BF16 token-row input plus top-1 reduction |
| **module kernels: 292** | **9,344** | sum |

Thus the 9,280 figure in the request is an approximation that omits the
64 final/head-support dispatches; it is not the trace's actual total.

## Historical trace cross-check

`historical-p3-round2-walltime-accounting.json` is generated directly from
the raw kernel, HIP API, and marker CSVs.  Kernel intervals are attributed by
the start of their correlated asynchronous HIP API call, not merely by a GPU
timestamp falling near a marker.

| Accounting basis | Wall ms/token | Module-kernel ms/token | Numeric residual | Module-kernel share |
|---|---:|---:|---:|---:|
| Historical rocprof run, paired `Instant` wall measurement | 14.503771 | 12.883598 | 1.620173 | 88.8293% |
| Supplied active direct wall, divided by historical trace kernel sum | 13.613453 | 12.883598 | 0.729855 | 94.6387% |

The second row is the literal division requested by the supplied 412,275,120
ns total: `412,275,120 / 32 / 13.613452656 ms = 94.6387%`.  **It is not yet a
physically valid time partition**, because it combines an unprofiled current
wall run with a separately profiled historical run.  In particular, the
historical trace directly contains 1.514498 ms/token of GPU inter-dispatch
gaps, already larger than that cross-run 0.729855 ms residual.  That is
evidence of profiling/run-condition perturbation, not evidence for a
negative host cost.

The result that was already safe to state before the current capture was:

> The available trace bounds AQ4_0 decode as kernel-dominated, not
> kernel-external-dominated.

The current capture above has now confirmed that conclusion with the exact
active P3 diagnostic binary and a direct module-launch measurement.

## Direct GPU-gap evidence from the raw trace

For each marked step, the analyser unions `KERNEL_DISPATCH` intervals on the
sole queue/stream and sums the holes between them.  No dispatch intervals
overlap in this capture.

| Historical trace quantity | Total for 32 steps | Per token |
|---|---:|---:|
| module kernels inclusive | 412.275120 ms | 12.883598 ms |
| all GPU dispatches inclusive (including two D2H copies) | 412.474361 ms | 12.889824 ms |
| gaps between all GPU dispatches | 48.463948 ms | **1.514498 ms** |
| first-to-last GPU activity span | 460.938309 ms | 14.404322 ms |
| marker leading/trailing non-GPU time | 2.765875 ms | 0.086434 ms |
| D2H copy GPU execution | 0.199241 ms | 0.006226 ms |

For the **paired historical profiled wall** (14.503771 ms/token), these pieces
close exactly: 12.883598 ms module kernels + 1.514498 ms GPU gaps + 0.006226
ms D2H GPU execution + 0.086434 ms marker lead/trail with no GPU dispatch +
0.013015 ms outside the outer marker in the driver's `Instant` scope.  The
last two buckets identify a small host/runtime boundary, but they are not a
breakdown of every HIP API duration because those API calls overlap device
execution.

The gaps are a direct trace fact, but their magnitude must not be projected
unchanged to the unprofiled service: the trace enables HIP API callbacks and
shows a mean 37.151 microseconds per `hipModuleLaunchKernel` API call.  API
durations overlap GPU work and are not additive wall-time buckets.  The
paired current capture includes a no-op `hipModuleLaunchKernel` microbenchmark
both with and without rocprof precisely to quantify this observer effect.

The gap-to-next-dispatch timing relation is also directly measurable in this
trace.  Of the 48.463948 ms total gaps, 41.954501 ms (9,292 gaps, 1.311078
ms/token) had their next `hipModuleLaunchKernel` API **already complete**
before the preceding GPU dispatch ended.  Only 6.281944 ms (49 gaps, 0.196311
ms/token) began before the next API call; 32 of those next calls were the
final D2H copies and 17 were module launches.  A further 0.227503 ms (35 gaps,
0.007109 ms/token) had the next API in progress at the prior GPU end.  This
rules out assigning the full 1.514498 ms/token to host launch enqueue time.
It leaves the dominant already-queued holes as GPU/runtime scheduling or
profiling behavior; the trace alone cannot distinguish those causes.

For a fusion scale only, the gaps immediately before the 97 normalization
dispatches total 0.477341 ms/token.  That is the directly adjacent
profiled-timeline hole that a perfect no-extra-cost normalization fusion could
remove; its 0.738159 ms/token of normalization arithmetic would still exist.
It is therefore a 3.51%-of-supplied-wall trace-era ceiling, **not** a claimed
production speedup.

There is one concrete HIP Graph eligibility positive: across all 32 historical
steps (C=1339..1370), every one of the 15 named GPU dispatches, including the
split full-attention kernels, had exactly one observed grid/workgroup shape.
The launch count was also constant at 294 all-GPU dispatches/token.  That
makes a fixed-node graph worth testing for this narrow range.  It does **not**
prove graph capture is valid: position/cache scalar updates, module-kernel
capture support, and the final GPU-resident-top1 D2H/synchronization boundary still
need a separate experiment.

## Identified kernel-external work in this driver

This is evidence about the direct profile driver only.  It intentionally has
no HTTP gateway, request scheduler, or prompt tokenization inside a measured
decode step.

- Exactly two asynchronous D2H copies/token occur after GPU-resident top-1:
  partial values and partial indices.  Their GPU time is only 6.226
  microseconds/token in the historical trace.  The 248,320-token vocabulary
  uses 256-value top-1 blocks, hence 970 f32 values + 970 u32 indices:
  3,880 bytes per copy and **7,760 bytes/token** in total.
- Exactly one `hipStreamSynchronize`/token follows those copies.  The profile
  source states that there is no per-layer decode synchronization; source
  locations are `qwen35_aq4_head_runtime.rs` (GPU-resident top-1 readback and
  synchronization) and `ullm-aq4-decode-step-profile.rs` (measurement scope).
- The long historical `hipMemcpyDtoHAsync` API durations are mostly waiting
  while the final LM-head kernel runs.  They overlap GPU time and must not be
  added to the residual.
- The remaining host work is the greedy top-1 partial-pair decode and
  CPU-only publish/advance bookkeeping.  Its exact unprofiled duration has
  no separate timer even after the paired capture, so it remains
  **unattributed** rather than guessed.

## Roofline from AQ4_0 physical weight payload

`current-projection-roofline.json` parses the active product
package rather than assuming nominal 4 bpp.  AQ4_0 uses a 4-bit index plus an
8-bit scale-table index per group: g16 tensors are 4.5 bpp and g8 tensors are
5 bpp.  The normal decode path has 249 quantized tensors (MTP excluded),
7,935,623,168 elements, and **4,565,106,688 bytes/token** of payload-only
weight reads (4.60214 effective bpp).

At the requested R9700 640 GB/s bandwidth, the payload-only lower bound is
7.132979 ms/token.  Current AQ4_0 projection is 10.382780 ms/token, i.e.
439.681 GB/s effective payload bandwidth or 68.700% of that lower-bound
roofline.  This does *not* count codebook, activation, output, or cache
traffic, so it is an optimistic upper bound on available improvement.

| Projection kernel | Current ms/token | Payload efficiency of 640 GB/s lower bound |
|---|---:|---:|
| `ullm_aq4_matvec_add_f32_kernel` | 3.697842 | 52.46% |
| `ullm_aq4_matvec_silu_mul_f32_kernel` | 3.402800 | 83.20% |
| `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` | 1.551461 | 68.79% |
| `ullm_aq4_matvec_f32_kernel` (LM head) | 1.186687 | 83.70% |
| `ullm_aq4_matvec_triple_f32_kernel` | 0.543989 | 55.42% |

The payload-only floor is a 1.4556x maximum **projection-kernel** speedup;
holding all non-projection work fixed, it is only a 1.3183x end-to-end ceiling
against this direct wall.  A conditional 10% / 20% projection-time reduction
would save 1.038278 / 2.076556 ms/token (7.71% / 15.43% latency; 1.0836x /
1.1824x).  `matvec_add` remains the first target because it combines the
largest absolute time with the weakest large-family payload efficiency;
`matvec_triple` is a separate low-efficiency shape but has much less absolute
time.

No AQ4_0 projection source was changed in this audit: their implementation is
owned by the protected runtime source files.  The handoff specification is
`projection-optimization-handoff.md`.

## Current-capture status

One owned service window ran from 01:18:30 to 01:19:03 JST.  The gateway was
the verified lock owner before the stop; the capture took the flock itself and
the service stayed inactive through both direct runs.  The first restoration
attempt encountered a pre-existing `start-limit-hit`; a recorded
`systemctl reset-failed` followed by one start restored the service to
`ActiveState=active`, `NRestarts=0` (MainPID 4004158).  No service start was
attempted while another lock holder existed.

The R9700 watcher captured 24 samples: max actual gfx clock 3,242 MHz, memory
clock 1,258 MHz, fabric clock 2,016 MHz, socket power 319 W, and hotspot 76 C
(edge 53 C, memory 52 C).  `THROTTLED` appeared in sampled status, but the
actual clocks remained high and the status alone is not interpreted as a
clock-loss event.  The capture scripts verify `gfx1201`, record the hashes and
service/lock states, never change `active.json`, and do not start
`llama-qwen35-udq4.service`.
