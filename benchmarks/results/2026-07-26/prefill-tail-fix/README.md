# SQ8_0 prefill M=1 tail fix

Date: 2026-07-26 JST
Scope: Qwen3-14B `SQ8_0`, R9700 (`gfx1201`, AMD SMI GPU 2), single sequence.

## Outcome

The fixed `M=128` prefill scheduler no longer decomposes a tail into one
`M=1` advance per token.  It rewinds the write cursor over a suffix of
**real prompt tokens**, replays one fixed-width chunk, and commits only the
new logical suffix.  This avoids both a ragged resident stack and padding.

At 4095 prompt tokens, the old path made 158 prefill advances
(`31 × M=128 + 127 × M=1`).  The fixed path made 32 advances
(`31 × M=128 + one overlapping M=128`), while accounting for 4095 logical
prompt tokens.  The remeasurement improved 4095 prefill from 71.576 to
**99.532 tok/s** (1.391×).

| prompt tokens | AY old uLLM SQ8_0 tok/s | tail-fix uLLM SQ8_0 tok/s | improvement | uLLM calls old → new | AY llama.cpp Q8_0 / F32 KV tok/s | llama / new uLLM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 851.659 | 865.278 | 1.016× | 1 → 1 | 1,165.756 | 1.347× |
| 512 | 513.676 | 519.872 | 1.012× | 4 → 4 | 1,195.722 | 2.300× |
| 1024 | 335.996 | 337.896 | 1.006× | 8 → 8 | 1,145.351 | 3.390× |
| 2048 | 188.425 | 189.685 | 1.007× | 16 → 16 | 1,058.379 | 5.580× |
| 4095 | 71.576 | **99.532** | **1.391×** | **158 → 32** | 1,008.683 | **10.134×** |

The five tail-fix rows are new five-repeat R9700 measurements.  The llama.cpp
F32-KV column is the unchanged AY comparison baseline, not a newly rerun
llama.cpp job; its model, binary, R9700 selection, `-ub 128`, F32 KV, and
five-repeat timing contract are documented in
[`../r9700-prefill-comparison/conditions.md`](../r9700-prefill-comparison/conditions.md).
This window remeasured the changed uLLM path under those same uLLM conditions;
rerunning an unchanged second engine would have consumed another shared-GPU
window without testing the scheduler change.

The timer is the driver’s `std::time::Instant` around the synchronized
prefill advances.  Model load, same-length warm-up, request start, reset, and
all profiler scopes are excluded.  No profiler range duration is reported as
throughput.

## What changed

Before this change, `plan_next_prefill_unit()` selected a fixed chunk only
when `remaining >= chunk_tokens`, then used `unwrap_or(1)`.  In `M=128` mode,
the `remaining < 128` case therefore entered the decode-style `M=1` loop.

The scheduler now represents an execution range separately from its logical
commit range.  For a fixed-width tail after at least one full chunk:

```text
logical cursor:      3968
logical tail:         127 tokens (3968..4094)
rewind cursor to:    3967
execute real tokens: 3967..4094  (M=128)
commit logical tail: 3968..4094  (127 tokens)
```

The single replayed token at 3967 is real, is overwritten before it can be
read as part of the new cache, and is not counted twice by the request
scheduler.  There is no padded token, synthetic token ID, or attention-mask
exception in this design.  `PagedDecodeState::written_len` is rewound before
execution and restores the logical cache boundary after the chunk finishes.

For a prompt shorter than the first fixed chunk there is no real prefix to
replay, so the existing audited `M=1` seed behavior remains.  This request is
about tails after fixed-width work, not an unsafe short-prompt padding path.

## Resident-stack constraint

`Sq8ServingPrefillMode::resident_stack_width()` is deliberately fixed.  Load
allocates the resident hidden stack and `prompt_chunk_hidden` at that selected
width; every layer workspace and the measured cached-prefix CK/Flash2 path
are built for the same `config.sequence_len`.  The load contract rejects a
different `prefill_chunk_tokens`, so arbitrary ragged `M` cannot simply be
passed into the resident stack.

The removed constraint was narrower: requiring the *prefix position* to be a
multiple of the resident width.  The cached-prefix operation already takes
an explicit prefix position and cache write range, so that alignment was a
scheduler/reporting policy rather than an allocation or kernel-shape
requirement.  The resident M=128 workspace remains fixed; only its starting
cache cursor may be unaligned for the overlapping tail.

## Numerical validation

See [`numerical-validation.md`](numerical-validation.md) and the raw oracle
comparison at [`numerical/comparison.json`](numerical/comparison.json).

- 128, 512, 1024, and 2048 are byte-identical in final hidden state and
  logits between the old binary and tail-fix binary; generated token and
  top-1 also match.
- 129, 1000, and 4095 are compared directly with the pre-change M=1 tail
  path.  Their first generated token and top-1 token match in all three
  deterministic cases.  The full tensor bytes do not match because the old
  final token ran through the M=1 paged-decode attention path while the new
  final token runs through cached-prefix M=128 attention; their operation and
  reduction order differ.  The exact relative-L2 / max-absolute deltas are
  recorded rather than hidden.
- A CPU unit test rewinds a paged cache, overwrites the suffix with deliberately
  different values, and checks causal reference attention against the new
  logical contents.  This specifically rejects stale entries beyond
  `written_len`; it is the cache-state proof for the no-padding implementation.

No independent CPU full-model oracle was run in this window.  The requested
alternative comparison against the old M=1 route was run on the same GPU,
artifact, prompt IDs, and model head capture.

## What remains slow

The tail was real but not the whole problem.  At 4095, 32 fixed M=128 chunks
still take 41.142 s per prompt, or 99.532 tok/s.  The path is now only 1.391×
faster than the M=1-tail baseline but remains 10.134× behind the AY llama.cpp
Q8_0/F32-KV row.  The unchanged 2048 gap is already 5.580×.

That remaining growth belongs to the cached-prefix prefill attention work and
its long causal prefixes, not to the scheduler’s former M=1 fallback.  This
task intentionally does not edit attention kernels: they share
`runtime/src/ullm_runtime_parts/part_01.inc` with the concurrent attention
work.  The handoff to that work is that prefill attention remains the next
dominant target after this call-count fix; the tail fix makes its residual
cost visible without 127 decode-style advances masking it.

## Evidence map

- [`conditions.md`](conditions.md): exact timing, device, thermal, and
  comparison conditions.
- [`accounting.md`](accounting.md): logical versus actual execution units.
- [`numerical-validation.md`](numerical-validation.md): old-M=1 comparison,
  exact-multiple proof, and cache-rewind proof.
- [`source-isolation.md`](source-isolation.md): clean-base candidate build,
  hashes, and isolation from concurrent kernel edits.
- [`service-window.md`](service-window.md): one inherited-inactive GPU window,
  service recovery, and active-manifest audit.
- [`raw/`](raw): per-condition commands, five-repeat driver logs, thermal gate
  samples, and one-second AMD SMI telemetry.
