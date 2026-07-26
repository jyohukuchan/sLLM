# AQ4_0 projection optimization handoff

## Status

This is a measurement-derived handoff, not an implementation.  The current
audit does not modify AQ4_0 projection sources because their active HIPRTC
implementation is in files owned by another session:

- `runtime/src/ullm_runtime_parts/part_01.inc`
- `runtime/src/ullm_runtime_hiprtc_sources.inc`

`runtime/src/ullm_runtime_parts/part_00.inc` only selects/compiles the source
through `*_kernel_source_for_arch`; it is not the kernel-body authority.  Do
not edit part_00 as a substitute for a kernel optimization.

## Preconditions before changing a kernel

1. Use the paired current capture described in `README.md` as the baseline,
   including the kernel count invariant of 292 module launches/token at C=1339.
2. Preserve AQ4_0 semantics exactly: low-nibble-first indices, g16/g8 scale
   table addressing, group size, zero-point/codebook handling, f32 output and
   residual order.  Validate greedy tokens and runtime checks before comparing
   performance.
3. Benchmark only R9700/gfx1201 under `/run/ullm/r9700.lock`; never use a
   gfx1030 V620 as a fallback.
4. Report both an unprofiled 32-step C=1339 result and a trace-derived
   kernel-family breakdown.  Do not use a rocprof marker duration as a timing
   denominator.

## Priority order under the current P3 trace

| Priority | Kernel | Current time/token | Payload BW | Why |
|---:|---|---:|---:|---|
| 1 | `ullm_aq4_matvec_add_f32_kernel` | 3.697842 ms | 335.74 GB/s (52.46% of 640 GB/s payload bound) | 28.82% of module kernel time; 64 launches/token; largest time and weakest large-family payload efficiency |
| 2 | `ullm_aq4_matvec_triple_f32_kernel` | 0.543989 ms | 354.67 GB/s (55.42%) | Only 8 launches/token but the lowest payload efficiency; full-attention Q/K/V shape is distinct |
| 3 | `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` | 1.551461 ms | 440.24 GB/s (68.79%) | 24 launches/token; substantial linear-attention contribution |
| 4 | `ullm_aq4_matvec_silu_mul_f32_kernel` | 3.402800 ms | 532.48 GB/s (83.20%) | 26.52% of kernel time, but materially nearer its optimistic payload bound |
| 5 | `ullm_aq4_matvec_f32_kernel` LM head | 1.186687 ms | 535.69 GB/s (83.70%) | One launch/token and near the same optimistic payload bound |

The physical-weight payload total is 4.565107 GB/token.  At 640 GB/s its
optimistic lower bound is 7.132979 ms/token, against 10.382780 ms/token of
current projection time.  Therefore even a hypothetical attainment of that
payload-only floor saves at most 3.249801 ms/token of projection work before
accounting for all other traffic.

Applied conditionally to the fresh 13.458181 ms/token direct wall, the
following are *upper-bound scenarios*, not forecasts:

| Projection assumption | Conditional wall | Conditional speedup |
|---|---:|---:|
| 80% of 640 GB/s payload bandwidth | 11.991626 ms/token | 1.1223x |
| 90% of 640 GB/s payload bandwidth | 11.000934 ms/token | 1.2234x |
| 640 GB/s payload-only floor | 10.208381 ms/token | 1.3183x |

These figures assume the current trace's projection duration transfers unchanged,
all non-projection work stays fixed, and no extra traffic is introduced.  The
last condition is deliberately unrealistic; it is a ceiling.

On the same assumptions, a smaller 10% / 20% projection-time reduction would
save 1.038278 / 2.076556 ms/token, respectively.  Against the fresh direct
wall those are 7.71% / 15.43% latency reductions, or 1.0836x / 1.1824x
throughput.  These are the more useful decision-scale estimates for a P3
kernel that is already optimized, not promises of attainable bandwidth.

## Experiment design

For each candidate change, collect a minimal three-way evidence set:

1. A kernel-focused probe for the specific shape and AQ4_0 tensor family.
2. The full 32-step C=1339 direct profile driver, with identical active guards
   and package hash, to catch end-to-end regressions.
3. One rocprof trace analysed by
   `tools/analyze-aq4-decode-walltime-accounting.py`, reporting unchanged
   dispatch cardinality or an intentional, explained change.

Useful candidate mechanisms are wider/coalesced scale-index and index loads,
more effective lane-to-output mapping for `matvec_add`, and eliminating
unnecessary intermediate traffic only when numerical order remains valid.
They are hypotheses, not established fixes.  The trace does not provide
hardware counters, so it cannot by itself prove whether the observed gap to a
payload-only roofline is cache behavior, instruction throughput, occupancy,
or non-payload memory traffic.

## Launch-reduction boundary

The current trace has 1.487833 ms/token between GPU dispatches, but it was
captured with HIP API tracing and is not an unprofiled launch-overhead budget.
`rmsnorm`/segmented norm work totals 97 launches/token and is an appropriate
separate fusion investigation, but it must be measured against the paired
no-op launch probe before claiming a gain.  HIP Graph capture is plausible
only after verifying fixed node shapes/addresses across C=1339..1370 and
handling the mandatory final GPU-resident-top1 D2H/synchronization boundary.  Neither
fusion nor HIP Graph work is implemented by this handoff.

The current trace supplies a positive first check: all 15 named dispatch
types have one observed workgroup/grid shape throughout C=1339..1370, and the
all-GPU launch count is a stable 294/token.  It does not prove graph safety;
the changing token position/cache scalar parameters and HIPRTC module-capture
compatibility still require a small dedicated capture/replay experiment.

A useful trace-only scale is more specific: gaps immediately before the 97
normalization dispatches total 0.463949 ms/token.  If every such dispatch were
absorbed by a neighboring kernel with unchanged arithmetic and no new stall,
that is the maximum directly adjacent gap it could remove from this *profiled*
timeline.  It excludes the 0.744165 ms/token normalization computation, which
fusion cannot simply delete.  Treat 0.463949 ms/token (3.45% of the fresh
direct wall) as a profiling-era ceiling, not an expected production gain.

The unprofiled module-launch probe establishes a narrower HIP Graph scale:
the 1.553198 microseconds/call base `hipModuleLaunchKernel` enqueue cost is
0.453534 ms/token across 292 launches.  It is not a promised graph saving,
because the trace shows most enqueues already overlap previous GPU execution
and graph replay itself has a launch cost.  A capture/replay experiment must
show a critical-path reduction before graph work is prioritized over
`matvec_add`.
