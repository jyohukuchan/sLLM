# T2 SQ FP8 token-id model-loop selected-layer k/o layer19 gate scale v1

## 前回の要点

- 19 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer19 `down_proj` row-block64はlen4でstrict top1を壊したためfailure guardだった。
- 次はlayer19 `gate_proj` row-block32を追加して、layer19側のMLP branchを確認する段階だった。

## 今回の変更点

- current 19 tensor branchにlayer19 `gate_proj` row-block32を追加した20 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer19-gate32` | 20 | 3 / 3 | `110784,237950,182949` | 1 | 1 | 1 | 28.118832 | 28.167761 | 28.125204 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer19 `gate_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer15 `k16/o32/up32/gate32` + layer19 `k16/o32/gate32` + layer23 `k16/o32/gate32/down64` + layer7 `k16/o32` まで広げられる。
- layer19 `down_proj` row-block64は引き続きfailure guardとして残す。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-gate-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-gate-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-gate-scale-v1/sq-selected-layer-ko-layer19-gate32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-down64-gate32-policy-artifact-v0.1.json`

## 次の行動

1. 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer19 `up_proj` row-block32を追加して、layer19の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15 `down`、layer19 `up/down`、layer23 `up_proj` は既存failure guardまたは未選択branchとしてfallbackに残す。
