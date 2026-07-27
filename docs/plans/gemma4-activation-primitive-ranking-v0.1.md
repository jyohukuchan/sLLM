# Gemma4 activation-device primitive ranking v0.1

## Decision

The first activation-graph port must consume the output of the resident BF16
matvec on device.  In the three-repeat, four-token R9700 decode measurement,
matvec result readback accounts for the largest share of the round-trip cost:
**75.693 ms D2H submission + 20.127 ms stream synchronization per four
tokens**.  It is therefore the first graph boundary to remove; moving a later
elementwise operation alone would leave its producing matvec's pageable D2H
readback in place.

## Measurement

The resident harness now records the same timing fields both in aggregate and
per primitive.  This adds no dispatch, arithmetic, allocation, or
synchronization: it only assigns timings already taken at each primitive
boundary to its source operation.

R9700-only command (with `HIP_VISIBLE_DEVICES=1`) used the production resident
path after stopping `ullm-openai.service`; the lock was released before the
service was restarted.  Raw evidence is
`benchmarks/results/2026-07-27/gemma4-activation-device-v0.1/raw/per-primitive-baseline.json`.

| result-producing primitive | calls / 4 decode tokens | mean D2H submit ms | mean sync ms | mean combined ms |
| --- | ---: | ---: | ---: | ---: |
| BF16 matvec | 1,108 | 75.693 | 20.127 | **95.819** |
| BF16 row | 1,056 | 35.072 | 24.898 | 59.970 |
| paged attention | 140 | 21.974 | 3.466 | 25.440 |
| paged K/V write | 60 | 0.000 | 0.000 | 0.000 |

The unmodified numerical path generated `[236761, 108, 818, 5279]`.  Decode
was 15.733 tok/s and prefill was 18.544 tok/s, within the established
unprofiled baseline range.  This is an instrumentation baseline, not a port.

## Consequence for the first port

The next change must introduce persistent device activation buffers and a
device-input/device-output resident matvec edge before wiring an elementwise
primitive.  The immediately useful consumers are Gemma's direct-weight
RMSNorm, proportional RoPE, GELU, and residual/multiply operations.  Their
host functions cannot remove the dominant matvec D2H/sync by themselves;
their input must first remain device-resident.
