# T2 SQ FP8 qkv layer15 q8 full QKV prompt bundle v1

## 前回の要点

- layer15 `q16/k16/v16` は`case_a`で `4105 -> 5582` に反転した。
- layer15 `q8/v16` と `q8/v8` はstrict top1を回復し、回復に効いているのは主にlayer15 `q_proj` のrow-block8化だと判断した。

## 今回の変更点

- layer15のQ/K/V同時追加に戻し、layer15 `q_proj` だけrow-block8、`k_proj` と `v_proj` はrow-block16にした。
- layer3+7+11は従来どおり `q16/k16/v16` のまま維持した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` を使い、triple direct kernelを必須化した。

## R9700 result

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11/15 q8/k16/v16` | 12 | 62.888070 | 74.903309 | 33.650360 | 4564938752 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 92 |
| `sq_passthrough_tensor_count` | 763 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.280117035 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.000087261 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.191808701 |

## 判断

- layer15 `q8/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- `case_a` はpassしたがmarginは `0.000087261` と非常に薄い。
- したがって、これはlayer15 QKV同時追加を回復できる診断候補であり、まだfull SQ policyとしてpromoteしない。
- telemetryでは `sq_projection_boundary=triple`、`sq_fp8_triple_matvec_count=92` で、QKV triple direct kernelを踏んでいる。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-q8-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-q8-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer15 `q8/k16/v16` をcurrent diagnostic recovery candidateとして保存する。
2. 次はB=1/4/8 short guardまたは広いprompt/text guardで、薄い`case_a` marginがすぐ崩れないか確認する。
3. その後にlayer19以降へ同じQKV boundaryを広げるか、text-level guard実装へ進むかを判断する。
