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

1. Re-run the paired current capture described in `README.md`, including the
   kernel count invariant of 292 module launches/token at C=1339.
2. Preserve AQ4_0 semantics exactly: low-nibble-first indices, g16/g8 scale
   table addressing, group size, zero-point/codebook handling, f32 output and
   residual order.  Validate greedy tokens and runtime checks before comparing
   performance.
3. Benchmark only R9700/gfx1201 under `/run/ullm/r9700.lock`; never use a
   gfx1030 V620 as a fallback.
4. Report both an unprofiled 32-step C=1339 result and a trace-derived
   kernel-family breakdown.  Do not use a rocprof marker duration as a timing
   denominator.

## Priority order under the historical P3 trace

| Priority | Kernel | Historical time/token | Payload BW | Why |
|---:|---|---:|---:|---|
| 1 | `ullm_aq4_matvec_add_f32_kernel` | 3.779696 ms | 328.47 GB/s (51.32% of 640 GB/s payload bound) | 29.34% of module kernel time; 64 launches/token; largest time and weakest payload lower-bound efficiency |
| 2 | `ullm_aq4_matvec_triple_f32_kernel` | 0.534722 ms | 360.82 GB/s (56.38%) | Only 8 launches/token but low payload lower-bound efficiency; full-attention Q/K/V shape is distinct |
| 3 | `ullm_aq4_matvec_qkv_z_gate_beta_f32_kernel` | 1.543310 ms | 442.57 GB/s (69.15%) | 24 launches/token; substantial linear-attention contribution |
| 4 | `ullm_aq4_matvec_silu_mul_f32_kernel` | 3.403228 ms | 532.42 GB/s (83.19%) | 26.42% of kernel time, but materially nearer its optimistic payload bound |
| 5 | `ullm_aq4_matvec_f32_kernel` LM head | 1.187213 ms | 535.46 GB/s (83.66%) | One launch/token and near the same optimistic payload bound |

The physical-weight payload total is 4.565107 GB/token.  At 640 GB/s its
optimistic lower bound is 7.132979 ms/token, against 10.448168 ms/token of
historical projection time.  Therefore even a hypothetical attainment of that
payload-only floor saves at most 3.315189 ms/token of projection work before
accounting for all other traffic.

Applied conditionally to the supplied 13.613453 ms/token direct wall, the
following are *upper-bound scenarios*, not forecasts:

| Projection assumption | Conditional wall | Conditional speedup |
|---|---:|---:|
| 80% of 640 GB/s payload bandwidth | 12.081509 ms/token | 1.1268x |
| 90% of 640 GB/s payload bandwidth | 11.090817 ms/token | 1.2275x |
| 640 GB/s payload-only floor | 10.298264 ms/token | 1.3219x |

These figures assume the historical projection duration transfers unchanged,
all non-projection work stays fixed, and no extra traffic is introduced.  The
last condition is deliberately unrealistic; it is a ceiling.

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

The historical trace has 1.514498 ms/token between GPU dispatches, but it was
captured with HIP API tracing and is not an unprofiled launch-overhead budget.
`rmsnorm`/segmented norm work totals 97 launches/token and is an appropriate
separate fusion investigation, but it must be measured against the paired
no-op launch probe before claiming a gain.  HIP Graph capture is plausible
only after verifying fixed node shapes/addresses across C=1339..1370 and
handling the mandatory final direct-top1 D2H/synchronization boundary.  Neither
fusion nor HIP Graph work is implemented by this handoff.

The historical trace supplies a positive first check: all 15 named dispatch
types have one observed workgroup/grid shape throughout C=1339..1370, and the
all-GPU launch count is a stable 294/token.  It does not prove graph safety;
the changing token position/cache scalar parameters and HIPRTC module-capture
compatibility still require a small dedicated capture/replay experiment.

A useful trace-only scale is more specific: gaps immediately before the 97
normalization dispatches total 0.477341 ms/token.  If every such dispatch were
absorbed by a neighboring kernel with unchanged arithmetic and no new stall,
that is the maximum directly adjacent gap it could remove from this *profiled*
timeline.  It excludes the 0.738159 ms/token normalization computation, which
fusion cannot simply delete.  Treat 0.477341 ms/token (3.51% of the supplied
direct wall) as a profiling-era ceiling, not an expected production gain.
