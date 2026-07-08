# T2 SQ FP8 token-id model-loop selected-layer k/o layer19 down64 v1

## 前回の要点

- 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` はcurrent passing branchとして維持している。
- layer15 `down_proj` row-block64は `len4` でstrict top1を壊したためfailure guardになった。
- 次は同じcurrent branch上でlayer19 `down_proj` row-block64を追加し、MLP output projection branchを別レイヤーで広げられるかを見る段階だった。

## 今回の変更点

- current 15 tensor branchにlayer19 `down_proj` row-block64を追加した16 tensor policyを作成した。
- R9700のsix-layer token-id model-loop prompt bundleで評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer3-down64-layer11-down64-plus-layer19-down64` | 16 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.060194 | 32.529504 | 32.989994 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer19 `down_proj` row-block64を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- ただしT2 promotion ruleはstrict top1なので、この16 tensor branchはpromoteしない。
- 現在のpassing boundaryは、15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-down64-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-down64-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-down64-v1/sq-selected-layer-ko-layer3-down64-layer11-down64-plus-layer19-down64/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer19-down64-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer19-down64-policy-artifact-v0.1.json`

## 次の行動

1. 15 tensor版 `selected-layer-ko-layer3-down64-plus-layer11-down64` をcurrent passing branchとして保持する。
2. layer19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer23 `down_proj` row-block64を追加して、同じMLP output projection branchを別レイヤーで広げられるかを見る。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15 `up/gate/down_proj`、layer19 `up/gate/down_proj` は既存failure guardがあるためfallbackに残す。
