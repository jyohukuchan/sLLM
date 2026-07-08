# T2 SQ FP8 token-id model-loop selected-layer k/o layer11 down64 v1

## 前回の要点

- 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` は `3 / 3` strict top1 passだった。
- layer11 `up_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- current passing branchは、layer3 `k16/o32/up32/down64` + layers 7/11/15/19/23 `k16/o32` だった。

## 今回の変更点

- current 14 tensor branchにlayer11 `down_proj` row-block64を追加した15 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-plus-layer11-down64` | 15 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.647323 | 28.333764 | 28.606031 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer11 `down_proj` row-block64を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layers 7/15/19/23 `k16/o32` まで広げられる。
- layer11 `up_proj` row-block32/16は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-down64-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-down64-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-down64-v1/sq-selected-layer-ko-layer3-down64-plus-layer11-down64/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-policy-artifact-v0.1.json`

## 次の行動

1. 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` をcurrent passing branchとして保持する。
2. layer11 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer15 `down_proj` row-block64を追加して、同じMLP output projection branchを広げられるかを見る。
4. layer7 `up/gate/down` と layer11 `up_proj` は既存failure guardがあるためfallbackに残す。
