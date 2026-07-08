# T2 SQ FP8 token-id model-loop selected-layer k/o layer19 up scale v1

## 前回の要点

- 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer19 `down_proj` row-block64はlen4でstrict top1を壊したためfailure guardだった。
- 次はlayer19 `up_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認する段階だった。

## 今回の変更点

- current 20 tensor branchにlayer19 `up_proj` row-block32を追加した21 tensor policyを作成した。
- row-block32がlen4でstrict top1を壊したため、layer19 `up_proj` row-block16 recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで2候補を評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer19-up32` | 21 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.172968 | 29.662839 | 32.668729 |
| `selected-layer-ko-layer19-up16` | 21 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.200339 | 32.691546 | 33.133078 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer19 `up_proj` row-block32は、len4でAQ4 top1 `110784` からSQ top1 `102446` へ変わった。
- layer19 `up_proj` row-block16でも同じくlen4が `102446` になり、row-block16では回復しなかった。
- どちらもAQ4 top1はSQ top8内の2位に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing boundaryは20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-up-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-up-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-up-scale-v1/sq-selected-layer-ko-layer19-up32/raw.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer19-up-scale-v1/sq-selected-layer-ko-layer19-up16/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-up32-gate32-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-up16-gate32-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-up32-gate32-plus-layer23-down64-gate32-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer15-up32-gate32-plus-layer19-up16-gate32-plus-layer23-down64-gate32-policy-artifact-v0.1.json`

## 次の行動

1. 20 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-up32-gate32-plus-layer19-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer19 `up_proj` row-block32/16とlayer19 `down_proj` row-block64をfailure guardとして残す。
3. 現在のselected-layer MLP probe setでは、追加でpromoteできる候補がほぼ尽きたため、次はT1 full-package real request-batch throughput runnerへ戻るか、T2をselected layer外へ広げる。
