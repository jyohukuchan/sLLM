# T2 SQ FP8 qkv layers3/7/11/15 q16/k16/v16 prompt bundle v1

## 前回の要点

- layer3+7+11 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- 次はlayer15 QKVを同じrow-block16で足し、累積driftを見る段階だった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-q16-k16-v16` policyを追加した。
- layer3、layer7、layer11、layer15の `q_proj`、`k_proj`、`v_proj` をすべてrow-block16 FP8にした。
- artifactは `/tmp/ullm-sq-fp8-qkv-layers3-7-11-15-q16-k16-v16-policy-v0.1-artifact` に生成した。
- AQ4 baselineは同一入力・同一実行条件の既存runから再利用した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` でtriple direct kernelを必須化した。

## R9700 result

| row | FP8 tensors | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11/15 q16/k16/v16` | 12 | 62.929075 | 74.876312 | 34.095032 | 4556521472 | `24218,5582,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 92 |
| `sq_passthrough_tensor_count` | 763 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.279635429 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.001344681 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.190737725 |

## 判断

- layer3+7+11+15 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `2 / 3` だった。
- 失敗は `case_a` のみで、AQ4 top1 `4105` はSQ top8内rank 2に残り、SQ top1 `5582` との差は `0.001344681` と小さい。
- ただしstrict top1 policyでは失敗なので、このcandidateはpromoteしない。
- 次はlayer15のQ/K/Vを単体に分解し、単独projectionのhard failureか累積driftかを切り分ける。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-q16-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-15-q16-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer3+7+11 `q16/k16/v16` を現在のQKV triple passing boundaryとして維持する。
2. layer15 `q16/k16/v16` はsplit結果を見て、単独projectionでpromote可能か判断する。
3. full layer15 QKV再挑戦は、pair split、より細かいscale、またはtext-level guard追加後に回す。
