# T2 SQ FP8 qkv layer19 q8 full QKV prompt bundle v1

## 前回の要点

- layer15 `q8/k16/v16` はprompt bundleとB=1/4/8 short batch guardでstrict top1を維持した。
- ただし `case_a` marginが薄いため、layer19へ広げても崩れないか確認する必要があった。

## 今回の変更点

- layer19にもQ/K/V同時追加を広げ、layer19 `q_proj` だけrow-block8、`k_proj`/`v_proj` はrow-block16にした。
- layer3+7+11は `q16/k16/v16`、layer15は `q8/k16/v16` のまま維持した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` を使い、triple direct kernelを必須化した。

## R9700 result

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11/15/19 q8/k16/v16` | 15 | 61.661523 | 73.332847 | 33.547192 | 5238153216 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 115 |
| `sq_passthrough_tensor_count` | 760 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.279256821 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.000214577 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.201775551 |

## 判断

- layer19 `q8/k16/v16` 追加後もfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- `case_a` はpassしたがmarginはまだ薄く、`0.000214577` だった。
- telemetryでは `sq_projection_boundary=triple`、`sq_fp8_triple_matvec_count=115` で、QKV triple direct kernelを踏んでいる。
- これはlayer19までの診断候補であり、まだfull SQ policyとしてpromoteしない。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-19-q8-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-19-q8-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. B=1/4/8 short batch guardでbatch pathでも崩れないか確認する。
2. passした場合はlayer23 QKV extensionか、広いprompt/text guardへ進む。
3. full SQ policyへpromoteする前にtext-level guardを追加する。
