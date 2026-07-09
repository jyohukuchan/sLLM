# T2 SQ FP8 full mixed qkv q16 v16 prompt bundle v1

## 前回の要点

- `q32/k16/v16` は `q32/k16/v32` より `case_a` のfailure marginを縮めたが、strict top1はまだ `2 / 3` だった。
- `q/k` pair単体は通るがmarginが薄く、`q/k/v` 同時適用ではQ側のdriftも減らす必要がある可能性があった。

## 今回の変更点

- `q16/k16/v16` policy `sq-fp8-w8a16-r9700-v0-qkv-layer3-q16-k16-v16` を追加した。
- `q_proj`、`k_proj`、`v_proj` をすべてrow-block16にした。
- artifactは `/tmp/ullm-sq-fp8-qkv-layer3-q16-k16-v16-policy-v0.1-artifact` に生成した。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` を付け、triple direct SQ FP8 kernelが踏まれることをtelemetryで確認した。

## R9700 result

| row | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | true | true | 66.905744 | 81.096744 | 35.521656 | 4952731648 | `24218,4105,329` |
| SQ `q16/k16/v16` | true | true | 62.157419 | 79.459594 | 33.945112 | 4346621952 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 23 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | top1 stable |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | SQ top1 margin over rank2 is `0.002023697` |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | top1 stable |

## 判断

- `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- SQ FP8 direct triple境界も `sq_fp8_triple_matvec_count=23` で確認できた。
- `case_a` は `q/k` pair単体のmargin約 `0.000080586` より広い `0.002023697` まで戻った。
- 現在のlayer3 QKV triple候補では、`q16/k16/v16` を最小のpass boundaryとして扱う。
- ただし1 layer / 3 tensorだけの診断候補であり、full SQ policyではない。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q16-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q16-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. `q16/k16/v16` を現在のlayer3 QKV triple pass boundaryとして保存する。
2. 次は同じscale粒度をlayer7以降へ広げる前に、B=1/4/8 short guardまたは追加promptで再確認する。
3. 速度比較では、この候補を通常triple dispatchのままAQ4 baselineと並べる。
