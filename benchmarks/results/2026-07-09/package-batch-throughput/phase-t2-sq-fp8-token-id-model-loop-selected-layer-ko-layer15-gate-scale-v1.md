# T2 SQ FP8 token-id model-loop selected-layer k/o layer15 gate scale v1

## 前回の要点

- 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer15 `down_proj` row-block64はlen4でstrict top1を壊した。
- 次はlayer15 `gate_proj` row-block32を追加して、layer15側のMLP branchを確認する段階だった。

## 今回の変更点

- current 17 tensor branchにlayer15 `gate_proj` row-block32を追加した18 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer15-gate32` | 18 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.562896 | 28.068268 | 28.497393 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer15 `gate_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer15 `k16/o32/gate32` + layer23 `k16/o32/gate32/down64` + layers 7/19 `k16/o32` まで広げられる。
- layer15 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-gate-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-gate-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer15-gate-scale-v1/sq-selected-layer-ko-layer15-gate32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-gate32-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-gate32-plus-layer23-down64-gate32-policy-artifact-v0.1.json`

## 次の行動

1. 18 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer15 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer15 `up_proj` row-block32を追加して、layer15の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15 `down`、layer19 MLP family、layer23 `up_proj` は既存failure guardがあるためfallbackに残す。
