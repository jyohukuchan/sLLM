# T2 SQ FP8 token-id model-loop selected-layer k/o layer11 gate scale v1

## 前回の要点

- 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` はcurrent passing branchだった。
- layer11 `up_proj` row-block32/16はどちらもlen4でstrict top1を壊した。
- 次はlayer11 `gate_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認する段階だった。

## 今回の変更点

- current 17 tensor branchにlayer11 `gate_proj` row-block32を追加した18 tensor policyを作成した。
- `gate32` がlen4でstrict top1に失敗したため、layer11 `gate_proj` だけrow-block16へ狭めた `gate16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、AQ4 baseline top1と比較した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer11-gate32` | 18 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 33.163512 | 32.595261 | 33.088271 |
| `selected-layer-ko-layer11-gate16` | 18 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.889847 | 32.551791 | 32.845355 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer11 `gate_proj` row-block32を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- layer11 `gate_proj` row-block16でも同じlen4 failureは回復しない。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、layer11 `gate_proj` はpromoteしない。
- current passing branchは17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-gate-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-gate-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-gate-scale-v1/sq-selected-layer-ko-layer11-gate32/raw.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer11-gate-scale-v1/sq-selected-layer-ko-layer11-gate16/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-gate32-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-gate32-plus-layer23-down64-gate32-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-gate16-plus-layer23-down64-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-gate16-plus-layer23-down64-gate32-policy-artifact-v0.1.json`

## 次の行動

1. 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer11 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer15 `gate_proj` row-block32を追加して、layer15側のMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up/gate`、layer15/19 MLP family、layer23 `up_proj` は既存failure guardがあるためfallbackに残す。
