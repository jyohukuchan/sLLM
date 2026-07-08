# SQ FP8 Mixed Candidate Acceptance v0.1

Date: 2026-07-08

## Purpose

This records the first machine-readable T2 acceptance evaluation for the mixed row-block SQ FP8
candidate.

This does not rerun inference. It evaluates existing `sq-fp8-overlay-logits-guard-result-v0.1`
JSON files with `tools/evaluate-sq-fp8-overlay-acceptance.py`.

## 前回の要点

- The current partial-quality SQ candidate is `v` fallback plus `q/k/o/gate/up/down` row-block32
  FP8.
- It passes layers `3,7,11,15` across `3 / 3` short prompt cases and layers `3,7,11,15,19` for
  len4.
- It fails strict top1 for layers `3,7,11,15,19,23` and all self-attention probe layers.
- Layer `23` alone can be recovered with `q/v` fallback, but the 6-layer bundle still fails even
  with broader `q/v` fallback.

## 今回の変更点

- Added `tools/evaluate-sq-fp8-overlay-acceptance.py`.
- Added `tests/test_sq_fp8_overlay_acceptance.py`.
- Defined T2 promotion rule v0.1 as `strict_top1`.
- Defined a diagnostic-only top-k rule:
  - `topk_common >= 5`
  - `baseline_top1_rank_in_sq_topk <= 2`
  - `abs(sq_top1_minus_baseline_top1_logit) <= 0.15`
- Evaluated the mixed candidate guard bundle and related layer-23 fallback probes.
- Raw result: `benchmarks/results/2026-07-08/sq-fp8-mixed-candidate-acceptance-v0.1.json`

The diagnostic rule does not override strict top1 failure. It only separates "near ranking drift"
from more severe failures.

## Summary

| field | value |
| --- | ---: |
| case count | `10` |
| strict top1 pass count | `5` |
| strict top1 passed | `false` |
| diagnostic top-k pass count | `5` |
| diagnostic top-k passed | `false` |
| accepted for T2 promotion | `false` |

## Case Results

| case | layers | FP8 tensors | top1 match | strict pass | diagnostic pass | AQ4 top1 rank in SQ top8 | top8 common | gap | failure summary |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 4layer_case_a | `3,7,11,15` | `24` | true | true | false | `1` | `4 / 8` | `0.000000` | diagnostic: low top-k overlap |
| 4layer_len4 | `3,7,11,15` | `24` | true | true | true | `1` | `5 / 8` | `0.000000` | n/a |
| 4layer_case_b | `3,7,11,15` | `24` | true | true | true | `1` | `5 / 8` | `0.000000` | n/a |
| 5layer_len4 | `3,7,11,15,19` | `30` | true | true | true | `1` | `6 / 8` | `0.000000` | n/a |
| 6layer_except_v_len4 | `3,7,11,15,19,23` | `36` | false | false | false | `5` | `6 / 8` | `0.250585` | top1 mismatch, rank/gap fail |
| all_self_attn_except_v_len4 | `3,7,11,15,19,23,27,31` | `48` | false | false | false | not in top8 | `4 / 8` | n/a | top1 mismatch, top-k/rank/gap fail |
| layer23_except_v_len4 | `23` | `6` | false | false | true | `2` | `5 / 8` | `0.014240` | top1 mismatch only |
| layer23_except_qv_len4 | `23` | `5` | true | true | true | `1` | `6 / 8` | `0.000000` | n/a |
| 6layer_except_v_q23_len4 | `3,7,11,15,19,23` | `35` | false | false | false | `5` | `6 / 8` | `0.214042` | top1 mismatch, rank/gap fail |
| 6layer_except_qv_len4 | `3,7,11,15,19,23` | `30` | false | false | false | `4` | `5 / 8` | `0.124241` | top1 mismatch, rank fail |

## Interpretation

The mixed row-block32 policy is not accepted for T2 promotion under the v0.1 rule.

Important details:

- `strict_top1` keeps the promotion rule conservative and reproducible.
- `4layer_case_a` passes strict top1 but fails the diagnostic top-k rule because top8 overlap is
  only `4 / 8`; it should remain a regression case.
- `layer23_except_v` is near under the diagnostic rule, but still fails promotion because top1 moves.
- `q/v` fallback fixes layer `23` alone but not the 6-layer bundle, so full-target promotion needs
  cumulative-drift work rather than only a single layer-23 fix.

## 次の行動

1. Keep `strict_top1` as the T2 promotion rule until a text-level guard is implemented and accepted.
2. Treat top-k/rank/gap as diagnostic fields only.
3. Continue T2 by testing per-layer/family fallback or stronger scale/layout for the 6-layer drift.
4. Do not move the mixed candidate to T5 throughput comparison as a promoted SQ policy yet.
