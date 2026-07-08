# T2 SQ FP8 token-id model-loop selected-layer k/o layer3 down64 v1

## 前回の要点

- 13 tensor版 `selected-layer-ko-plus-layer3-o32` は `3 / 3` strict top1 passだった。
- layer3 `gate_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- current passing branchは、layer3 `k16/o32/up32` + layers 7/11/15/19/23 `k16/o32` だった。

## 今回の変更点

- current 13 tensor branchにlayer3 `down_proj` row-block64を追加した14 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32-down64` | 14 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 33.091248 | 32.876952 | 33.063138 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer3 `down_proj` row-block64を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer7/11/15/19/23 `k16/o32` まで広げられる。
- layer3 `gate_proj` row-block32/16は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-down64-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-down64-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-down64-v1/sq-selected-layer-ko-plus-layer3-o32-down64/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-policy-artifact-v0.1.json`

## 次の行動

1. 14 tensor版 `selected-layer-ko-plus-layer3-o32-down64` をcurrent passing branchとして保持する。
2. layer3 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer11 `up_proj` row-block32を追加して、layer3以外のMLP入力側をcurrent branchへ足せるかを見る。
4. layer7 `up/gate/down` は既存failure guardがあるため、layer11以降のMLP familyを先に見る。
