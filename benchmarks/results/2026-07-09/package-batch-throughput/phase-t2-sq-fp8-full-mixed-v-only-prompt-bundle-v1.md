# T2 SQ FP8 full mixed v-only prompt bundle v1

## 前回の要点

- layer3 `q/k/v` SQ FP8 triple候補は、full mixed prompt bundleで `case_a` のtop1をAQ4 `4105` からSQ `5582` に入れ替えた。
- layer3 `q/k` pair候補は同じprompt bundleでstrict top1 `3 / 3` を維持した。
- 次の切り分けでは、`v_proj` 単体がtop1 swapを起こすか確認する必要があった。

## 今回の変更点

- layer3 `v_proj` 単体候補 `sq-fp8-w8a16-r9700-v0-v-layer3-v32` を追加した。
- policyとartifact summaryを保存し、artifactは `/tmp/ullm-sq-fp8-v-layer3-v32-policy-v0.1-artifact` に生成した。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` を付け、single direct SQ FP8 kernelが実際に踏まれることをtelemetryで確認した。
- AQ4 baselineとSQ v-only候補を同じbatch=3、prefill/decode real batch pathで比較した。

## R9700 result

| row | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | true | true | 67.076031 | 80.852685 | 35.096093 | 4443123712 | `24218,4105,329` |
| SQ `v-layer3-v32` | true | true | 60.580213 | 80.634172 | 31.871183 | 4445179904 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `single` |
| `sq_fp8_single_matvec_count` | 23 |
| `sq_fp8_batch_matvec_count` | 0 |
| `sq_fp8_pair_matvec_count` | 0 |
| `sq_fp8_triple_matvec_count` | 0 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `len4` | 24218 | 24218 | pass | 8 / 8 | 1 | top1 stable |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | SQ top1 margin over rank2 is `0.016001701` |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | top1 stable |

## 判断

- layer3 `v_proj` 単体候補は、full mixed prompt bundleでstrict top1を維持した。
- `case_a` のmarginは `q/k` pair単体時の約 `0.000080586` より広く、`v` 単体ではtop1 swapの直接原因には見えない。
- `q/k` pairと `v` 単体はそれぞれ通るが、`q/k/v` tripleでは落ちるため、現failureは単一tensorの致命的driftではなく累積・相互作用driftとして扱う。
- 次は `q/k/v` を保ったまま、`v_proj` のscale粒度をrow-block16などへ下げるか、`q_proj`/`v_proj` の片方だけをより細かくして `case_a` のtop1 marginを戻せるかを見る。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-v-layer3-v32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-v-layer3-v32-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `aq4/raw.json`
- `sq-v-layer3/raw.json`

## 次の行動

1. `v` 単体はfull mixed prompt-bundle pass境界として保存する。
2. `q/k/v` の同時適用を維持したまま、`v16` または `q16/v16` などのscale粒度を試す。
3. `case_a` のtop1 marginを、少なくとも `q/k` pair単体より明確に広げられる候補を優先する。
