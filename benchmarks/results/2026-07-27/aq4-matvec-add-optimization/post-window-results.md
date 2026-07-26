# AQ4_0 `matvec_add` group-specialized candidate — R9700 window

## Scope and comparison contract

This is the locked R9700/gfx1201 comparison from
`r9700-window-20260727T061520+0900`.  The staged worker was rebuilt from a
source tree whose grouped-decode source blobs match the active `c8074928`
release, plus only the two `matvec_add` candidate files.  The retained
shuffle body was selected with `ULLM_AQ4_MATVEC_ADD_USE_SHUFFLE_REFERENCE=1`;
the candidate was selected normally.  Thus the A/B is in-process and uses
the same staged worker, model, 4:1×256 grouped decode, context C=1339, and
R9700.

Before timing, both bodies passed the GPU-vs-CPU production-shape test (g8
4096×4096 and g16 4096×12288), generated the same greedy runtime sequence,
and each trace established 292 module launches/token and 64
`ullm_aq4_matvec_add_f32_kernel` launches/token.  The raw token IDs are not
stored in this report; `runtime/greedy-validation.json` records the match.

## Phase 0 conclusion

The 52.46% historical weight-payload efficiency was not mainly residual I/O.
At 64 add launches/token, the maximum residual read plus f32 output write is
2,097,152 B/token, or only 0.1689% of the 1,241,513,984 B/token physical
weight payload.  Input vectors and row scales are likewise small upper-bound
traffic relative to the repeatedly streamed weights.  Therefore the large
gap to `silu_mul` is not explained by the add epilogue's residual round trip.

Static gfx1201 ISA instead finds a source-structure difference.  The prior
one-stream add walks generic g8/g16 group and byte slots: g8 visits four
groups × eight byte slots while only four slots/group are valid, and g16
retains dynamic start/count/packed-word selection.  `silu_mul` already has
explicit g8/g16 traversal and shares each input pair over gate+up streams;
it also has three times as many WGs per launch (1536 versus add's 512).
Both bodies are already wave32 `ds_bpermute` reductions with no barrier,
LDS allocation, or spills.  There was no LDS-tree reduction to transplant.

The specializer preserves low-nibble-first AQ4_0 decode, g8/g16 scale table
addressing, codebook/zero-point treatment, group scaling, row scaling, f32
output, and `residual + value` order, but replaces the generic traversal with
fixed g8 and g16 uint4 word/byte walks.  It is the directly transferable part
of the SiLU-mul structure; two-stream input reuse is not applicable to add.

Static whole-function code falls from 1,434 to 820 instructions (SALU
922→395; VALU 399→321); static vector-memory instruction count is 108→99.
The tradeoff is higher static VGPR (30→49).  rocprof's dispatch rows likewise
report WG=256, grid=131072 (512 WGs), zero LDS/scratch, and VGPR 32→56.
`OccupancyPercent`, `SQ_INSTS_VALU`, `SQ_WAIT_ANY`, and `SQ_WAVE_CYCLES`
were all returned as zero on gfx1201, while `SQ_WAVES=4096`; achieved hardware
occupancy and dynamic VALU per element are therefore **unconfirmed**, not
zero.  The grid provides eight WGs/CU across 64 CUs, and both bodies use eight
wave32 waves/WG, but this is launch geometry rather than a measured residency.

## R9700 results

All throughput below is an unprofiled full-model `ullm-aq4-decode-step-profile`
mean over two 32-step runs, never a profiler range duration.

| Decode path | shuffle reference tok/s | group-specialized tok/s | ratio |
|---|---:|---:|---:|
| direct | 73.895446 | 77.679674 | 1.051211× |
| production grouped | 74.591159 | 78.284628 | 1.049516× |

The grouped candidate is also 1.050661× the recorded production grouped
control (74.509830 tok/s) and 1.056254× the recorded direct control
(74.110977 tok/s).  Per-run ranges are retained in `decode/summary.json`.

The matched two-token trace is diagnostic only, but confirms that the target
kernel itself moved in the intended direction: add inclusive time was
7.406139 ms→6.284984 ms for 128 launches (1.178386×), and the declared
weight-payload diagnostic rose 335.266→395.073 GB/s (52.385%→61.730% of the
640 GB/s payload roofline).  This is not reported as application throughput.

Cold AQ4_0 prefill at p=2048, M=128 was 974.984645 tok/s for the retained
body and 977.087601 tok/s for the candidate (1.002157×); it remains inside
the expected roughly 970–1,020 tok/s production range.

## Resource and service record

The window used the reachable edge thermal gate `edge <= 45 C` for every GPU
condition.  795 telemetry samples span edge 45–56 C, hotspot 45–80 C, and
GFX clock 0–3363 MHz (the zero/low samples include cooldown/idle).  The
`THROTTLED` label occurred in 202 samples without a concrete violation field;
it is not treated as evidence of clock loss.  The actual sampled peak was
3363 MHz.

This task attempted three owned R9700 windows: two were safely aborted for
runner infrastructure before performance acceptance, and this third window
completed.  One no-lock preflight refusal occurred while another task owned
the device.  The completed window released its flock and restored
`ullm-openai.service` once; the service was active with `NRestarts=1` at that
point.  `llama-qwen35-udq4.service` remained inactive/disabled.

## Candidate release staging

The exact staged worker has SHA-256
`1bc6f12548ea9c100830bd8b14bb15775457fad6fd8c324a814`.  It has been copied
without replacement of any existing release to
`/opt/ullm/aq4-matvec-add-group-specialized-v0.1/releases/aq4-matvec-add-7ecdd4ae-1bc6f125/ullm-aq4-worker`,
owned `root:root`, mode `0555`.  Its root-owned, read-only candidate manifest
and promotion receipt are under the same `/opt/ullm/` release root; manifest
validation passed with manifest SHA-256
`d3d9c4544bb5175d1a3dce6c0eeba22c8641e5bd9c2c8220b58f487eed6038c2`.

The immutable build source commit is
`7ecdd4ae858a94ffa6a5a1c5b6949bacc10c23ba`; integrated source is committed
on main as `cd7c17682cc55ae11f8290cf3bfcc90a08810b0d`.

## Promotion status

`tools/promote-served-model.py --yes` completed a normal lightweight
transaction.  Its immutable `outcome.json` records `status: "activated"`,
candidate manifest SHA-256
`d3d9c4544bb5175d1a3dce6c0eeba22c8641e5bd9c2c8220b58f487eed6038c2`, one
service restart, and no automatic rollback.  The active and candidate sides
each completed all ten real prompt-suite cases.  `comparison.json` reports
`passed: true`, no blocking findings, zero attention findings, and 10/10
output exact matches; top-1 telemetry is unavailable from this gateway but is
non-blocking under the policy.  The content-bearing comparison evidence is
kept in the promotion directory rather than repeated here.

After that transaction completed at 06:41:33 JST, the active manifest changed
at 06:42:24 JST to the pre-existing BZ manifest SHA-256
`3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7`, and the
service was stopped with TERM.  The promotion outcome itself is still
`activated`; this later state change is not an automatic rollback from the
promotion tool.  Its cause is **unconfirmed**.  Per the instruction not to
overwrite an unexpected active manifest, this task did not re-activate the
candidate.  The final service state is active and healthy on that existing
grouped worker; the candidate `/opt/ullm/` release remains immutable and
release-ready but is not the currently served worker.
