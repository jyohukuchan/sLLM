# T2 SQ FP8 token-id model-loop selected-layer k/o layer11 up scale v1

## 前回の要点

- 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` は `3 / 3` strict top1 passだった。
- current passing branchは、layer3 `k16/o32/up32/down64` + layers 7/11/15/19/23 `k16/o32` だった。
- 次のT2対象は、layer11 `up_proj` row-block32を追加してlayer3以外のMLP入力側をcurrent branchへ足せるかを見ることだった。

## 今回の変更点

- current 14 tensor branchにlayer11 `up_proj` row-block32を追加した15 tensor policyを作成した。
- `up32` がlen4でstrict top1に失敗したため、layer11 `up_proj` だけrow-block16へ狭めた `up16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `comparison.json` に保存した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-plus-layer11-up32` | 15 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 28.741285 | 30.527447 | 28.962318 |
| `selected-layer-ko-layer3-down64-plus-layer11-up16` | 15 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.863404 | 32.437353 | 32.807199 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer11 `up_proj` row-block32を追加すると、len4のtop1が `110784` から `102446` に変わる。
- layer11 `up_proj` row-block16でも同じlen4 failureは回復しない。
- case_a/case_bはAQ4 top1を維持し、len4でもAQ4 top1はSQ top8内の2位に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing branchは14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` のままとする。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-up-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-up32-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-up-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-up32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-up32-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-up16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-up16-policy-artifact-v0.1.json`

## 次の行動

1. 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` をcurrent passing branchとして保持する。
2. layer11 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer11 `down_proj` row-block64を追加して、layer11 MLP output projectionをcurrent branchへ足せるかを見る。
4. layer11 `up_proj` coverageは、より強いscale/layoutまたはtext-level guardの扱いが決まるまでfallbackに残す。
