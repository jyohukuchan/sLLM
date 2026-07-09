# T2 SQ FP8 qkv layer15 pair prompt bundle v1

## 前回の要点

- layer15 `q16`、`k16`、`v16` は単独追加ならすべてstrict top1 `3 / 3` を維持した。
- layer15 Q/K/V同時追加では`case_a`だけ `4105 -> 5582` に反転したため、pair splitで累積driftの組み合わせを切り分ける段階だった。

## 今回の変更点

- layer3+7+11 `q16/k16/v16` をbaseにし、layer15の `q+k`、`q+v`、`k+v` をrow-block16 FP8として追加した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1`、`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1`、`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` を設定した。
- 実行telemetryではlayer15 pair部分は `single+triple` 境界で処理され、`sq_fp8_pair_matvec_count=0` だった。

## R9700 result

| row | FP8 tensors | single count | pair count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 q16-k16` | 11 | 46 | 0 | 69 | 59.257498 | 75.399241 | 33.018293 | 4554485760 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 q16-v16` | 11 | 46 | 0 | 69 | 59.700485 | 75.017792 | 33.389594 | 5064097792 | `24218,5582,329` |
| SQ `layers3/7/11 + layer15 k16-v16` | 11 | 46 | 0 | 69 | 55.666617 | 75.741372 | 30.225450 | 5173133312 | `24218,4105,329` |

## Quality comparison

### layer15 q16-k16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278602124 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.003759384 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.196052551 |

### layer15 q16-v16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278126717 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.000548840 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.189418793 |

### layer15 k16-v16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.283506393 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.004536629 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.190866470 |

## 判断

- layer15 `q16+k16` と `k16+v16` はstrict top1 `3 / 3` を維持した。
- layer15 `q16+v16` は`case_a`で `4105 -> 5582` に反転し、full layer15 QKV failureと同じ向きの失敗になった。
- したがってlayer15のdriftは単独projectionではなく、主にQ/Vの組み合わせで発生していると考える。
- telemetry上はpair kernelではなく `single+triple` 境界で処理されているため、このrunは品質切り分けであり、pair kernel速度評価ではない。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q16-k16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-k16-v16-policy-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer3+7+11 `q16/k16/v16` をcurrent passing boundaryとして維持する。
2. layer15は `q+v` interactionをscale再調整の対象にし、`q8/v16`、`q16/v8`、または `q8/v8` を試す。
3. `q+k` と `k+v` は診断passとして保存するが、full SQ policyにはしない。
