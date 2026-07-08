# T2 SQ FP8 token-id model-loop selected-layer k/o layer23 up scale v1

## 前回の要点

- 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` は `3 / 3` strict top1 passだった。
- current passing branchは、layer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer23 `k16/o32/down64` + layers 7/15/19 `k16/o32` だった。
- 次はlayer23 `up_proj` row-block32を追加し、失敗時はrow-block16 recoveryを確認する段階だった。

## 今回の変更点

- current 16 tensor branchにlayer23 `up_proj` row-block32を追加した17 tensor policyを作成した。
- `up32` がlen4でstrict top1に失敗したため、layer23 `up_proj` だけrow-block16へ狭めた `up16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-layer23-up32` | 17 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 31.149837 | 32.513167 | 31.321143 |
| `selected-layer-ko-layer23-up16` | 17 | 2 / 3 | `102446,237950,182949` | 2 | 1 | 1 | 32.062485 | 27.859754 | 31.443782 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer23 `up_proj` row-block32を追加すると、`len4` のtop1がAQ4 `110784` からSQ `102446` に変わる。
- layer23 `up_proj` row-block16でも同じlen4 failureは回復しない。
- `case_a` と `case_b` はAQ4 top1を維持し、`len4` でもAQ4 top1はSQ top8内の2位に残る。
- T2 promotion ruleはstrict top1なので、layer23 `up_proj` はpromoteしない。
- current passing branchは16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` のままとする。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-up-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-up-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-up-scale-v1/sq-selected-layer-ko-layer23-up32/raw.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer23-up-scale-v1/sq-selected-layer-ko-layer23-up16/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer23-down64-up32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-down64-plus-layer11-down64-plus-layer23-down64-up16-policy-v0.1.json`

## 次の行動

1. 16 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` をcurrent passing branchとして保持する。
2. layer23 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer23 `gate_proj` row-block32を追加して、layer23の残りMLP branchを確認する。
4. layer7 `up/gate/down`、layer11 `up_proj`、layer15/19 MLP familyは既存failure guardがあるためfallbackに残す。
