# Validation status

## Scheduler, build, and admission tests

The committed scheduler tests cover M=128/256/512/1024/2048 at N=4095,
including the real-token cursor rewind, logical suffix commit, planned
1,280/640/320/160/80 layer-expanded attention calls, and the rule that
N=4095/M=4096 must not fabricate a 4096th row.  The fixed-width selector
also rejects invalid or non-power-of-two values.

Focused tests passed with `rocm-ck-gfx1201`, followed by the complete
`sq8_serving_runtime::tests` module: 44 passed, 0 failed.  The worker parser
and binary wiring were additionally checked: 10/10 worker-backend tests,
5/5 worker CLI tests, and `cargo check -p ullm-engine --bin ullm-sq8-worker
--features rocm-ck-gfx1201` passed.  The worker defaults to M=128 and accepts
`ULLM_SQ8_PREFILL_CHUNK_TOKENS=<M>` only after the same selector validation.

The direct gfx1201 CK helper probe accepted all four SQ8_0 projection shapes
at M=256/512/1024/2048/4096.  It is a zero-buffer shape-admission result,
not a substitute for the full-model evidence below.

## Isolated full-model overlay

The product source still rejects M>128 at the BP/BX-owned lower validation
boundary.  To test whether that boundary hid a kernel limit, an isolated
source overlay lifted only the validation lists, layer-oracle ceiling,
model-head row validation, and F32/typed paged-KV API bound.  It left
`runtime/src/ullm_runtime_parts/part_01.inc` and
`runtime/src/ullm_runtime_hiprtc_sources.inc` unchanged.

The performance/trace run
`run-20260727T024801+0900` completed the full five-width sweep and real
traces.  It then failed closed at the first numerical smoke because
`ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL=1` was missing.  No timing value
is derived from a trace or affected by this later guard failure.  Commit
`88607fe0` corrected the guard; `43cd16dd` made a one-window
validation-only continuation.  `run-20260727T044042+0900` completed it with
`window-finished status=0`, released the lock, and restored the service.

## Numerical fidelity

Every candidate prompt for which the selected resident M actually ran was
F32-byte-exact for final hidden state and logits (`max_abs=0`, non-finite
count 0).  The one-token greedy IDs matched in every comparison.

| M | non-byte-exact prompts | max hidden `max_abs` | max logits `max_abs` | cause |
| ---: | --- | ---: | ---: | --- |
| 256 | 128 | 1.435986 | 1.100868 | N<M, audited M=1 fallback |
| 512 | 128 | 1.435986 | 1.100868 | N<M, audited M=1 fallback |
| 1024 | 128, 512 | 1.435986 | 1.100868 | N<M, audited M=1 fallback |
| 2048 | 128, 512, 1024 | 1.601738 | 1.100868 | N<M, audited M=1 fallback |

These recorded scalar differences are descriptive only.  They are not an
accept/reject threshold; text evidence below follows the lightweight
promotion policy.

## Generated text

The fixed 10-case suite ran M=128 plus all four candidates because the
fallback cases were not byte-exact.  Each candidate had 9/10 exact decoded
texts and token IDs.  The only different case was `ja_long_summary`, a short
semantic-preserving wording variation; all policy obvious-collapse diagnostics
were empty.

The additional real-token N=4000 prompt exercises the actual M=256/512/1024/
2048 schedules and a genuine final generation header.  All five widths
produced exactly the same 83 token IDs and 467-character completion.  It does
not use padding, a mask, or a fabricated row.  Retained direct outputs are in
`run-20260727T044042+0900/generation*/summary.{json,md}`.

## Attention trace and decode

At N=4095, real traces observe 1,280/640/320/160/80 cached-prefix Flash2
dispatches for M=128/256/512/1024/2048.  The corresponding attention shares
are 93.005%/92.505%/92.018%/91.016%/90.134%.  Thus M=2048 genuinely removes
15 of every 16 M=128 dispatches, but attention remains the bottleneck.

Fresh M=128 decode at prompt 1024 is **27.552769 tok/s**, above the
27.378731 reference and BR's 27.411786 rerun.  The wide-M generation jobs
also completed their post-prefill M=1 decode paths.

## Service and thermal outcome

Both windows used R9700 only and an edge <=45 C gate.  The successful
continuation's event log records `service-restore-start-return=0`,
`service-restore-active`, and `window-finished status=0`; postflight shows
`ActiveState=active`, `NRestarts=0`.  `llama-qwen35-udq4.service` remained
inactive and disabled.
