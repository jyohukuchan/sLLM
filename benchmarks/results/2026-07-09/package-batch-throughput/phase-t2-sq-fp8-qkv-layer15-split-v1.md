# T2 SQ FP8 qkv layer15 split prompt bundle v1

## 前回の要点

- layer3+7+11+15 `q16/k16/v16` はfull mixed prompt bundleで`case_a`だけstrict top1を落とした。
- 失敗がlayer15の単独projection由来か、Q/K/V同時追加の累積driftかを切り分ける必要があった。

## 今回の変更点

- layer3+7+11 `q16/k16/v16` をbaseにして、layer15の `q_proj`、`k_proj`、`v_proj` を1つずつrow-block16 FP8として追加した。
- 各artifactは `/tmp/ullm-sq-fp8-qkv-layers3-7-11-plus-layer15-{q16,k16,v16}-policy-v0.1-artifact` に生成した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` を使い、tripleとsingle direct kernelを必須化した。

## R9700 result

| row | FP8 tensors | single count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 q16` | 10 | 23 | 69 | 59.071645 | 75.280931 | 33.180809 | 4550275072 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 k16` | 10 | 23 | 69 | 56.575977 | 75.925142 | 31.492802 | 4531376128 | `24218,4105,329` |
| SQ `layers3/7/11 + layer15 v16` | 10 | 23 | 69 | 59.481707 | 75.949459 | 32.897182 | 4531376128 | `24218,4105,329` |

## Quality comparison

### layer15 q16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.276909828 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.004627228 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.194718361 |

### layer15 k16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.281908036 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.009153843 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.195569038 |

### layer15 v16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.282120228 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.005000114 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.189785004 |

## 判断

- layer15 `q16`、`k16`、`v16` はそれぞれ単独追加ならstrict top1 `3 / 3` を維持した。
- layer15 Q/K/Vを同時に追加した場合だけ`case_a`が `4105 -> 5582` に反転するため、現時点では単一projectionのhard failureではなく累積driftとして扱う。
- full layer15 QKV candidateはstrict top1 policyではpromoteしない。
- layer15単独projectionは診断候補として保存するが、full SQ policyではない。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-q16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-k16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-11-plus-layer15-v16-policy-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer3+7+11 `q16/k16/v16` をcurrent passing boundaryとして維持する。
2. layer15はsingle split結果を保存し、pair splitまたはscale再調整でfull layer15 QKVを再探索する。
3. SQ format候補の本命化は、より広いprompt/text guardを用意してから判断する。
