# T2 SQ FP8 token-id model-loop selected-layer k/o layer3 gate scale v1

## 前回の要点

- layer3 `o_proj` row-block32を追加した13 tensor branchは、3 promptすべてでAQ4 top1を維持した。
- current passing branchは、layer3 `k16/o32/up32` + layers 7/11/15/19/23 `k16/o32` だった。
- 次のT2対象は、layer3 `gate_proj` を追加してlayer3 MLP coverageを `up` から `up+gate` へ広げられるかを見ることだった。

## 今回の変更点

- current 13 tensor branchにlayer3 `gate_proj` row-block32を追加した14 tensor policyを作成した。
- `gate32` がlen4でstrict top1に失敗したため、layer3 `gate_proj` だけrow-block16へ狭めた `gate16` recovery policyも作成した。
- R9700のsix-layer token-id model-loop prompt bundleで両方を評価し、結果を `comparison.json` に保存した。

## R9700 result

| variant | FP8 tensors | pass | final top1 | len4 AQ4 rank in SQ top8 | case_a AQ4 rank in SQ top8 | case_b AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32-gate32` | 14 | 2 / 3 | `102446,237950,182949` | 3 | 1 | 1 | 28.610392 | 28.109230 | 28.544011 |
| `selected-layer-ko-plus-layer3-o32-gate16` | 14 | 2 / 3 | `102446,237950,182949` | 3 | 1 | 1 | 33.121674 | 32.765115 | 33.074727 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer3 `gate_proj` row-block32を追加すると、len4のtop1が `110784` から `102446` に変わる。
- layer3 `gate_proj` row-block16でも同じlen4 failureは回復しない。
- case_a/case_bはAQ4 top1を維持し、len4でもAQ4 top1はSQ top8内の3位に残るが、T2 promotion ruleはstrict top1なのでpromoteしない。
- current passing branchは13 tensor版 `selected-layer-ko-plus-layer3-o32` のままとする。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-gate-scale-v1/comparison.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-gate32-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-token-id-model-loop-selected-layer-ko-layer3-gate-scale-v1/results.jsonl`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-gate32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-gate32-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-gate16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-gate16-policy-artifact-v0.1.json`

## 次の行動

1. 13 tensor版 `selected-layer-ko-plus-layer3-o32` をcurrent passing branchとして保持する。
2. layer3 `gate_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer3 `down_proj` row-block64を追加して、layer3 MLP output projectionをcurrent branchへ足せるかを見る。
4. gate coverageは、より強いscale/layoutまたはtext-level guardの扱いが決まるまでfallbackに残す。
