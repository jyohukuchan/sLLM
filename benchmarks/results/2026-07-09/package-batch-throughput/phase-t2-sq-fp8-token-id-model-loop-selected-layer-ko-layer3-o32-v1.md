# T2 SQ FP8 token-id model-loop selected-layer k/o layer3 o32 v1

## 前回の要点

- layer7 `o32` branchはlayer23まで `k_proj` row-block16と `o_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- ただしlayer3は `k_proj` と `up_proj` のみで、selected-layer `k/o` branchとしてはlayer3 `o_proj` が穴として残っていた。

## 今回の変更点

- current 12 tensor branchにlayer3 `o_proj` row-block32を追加した13 tensor policyを作成した。
- R9700で同じsix-layer token-id model-loop prompt bundleを実行し、AQ4 baseline top1と比較した。

## R9700 result

| row | coverage | FP8 tensors | passthrough tensors | strict top1 pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `selected-layer-ko-plus-layer3-o32` | layer3 k16/o32/up32 + layers 7/11/15/19/23 k16/o32; layer7 up/down/gate fallback | 13 | 762 | 3 / 3 | `110784,237950,182949` | 1 | 32.766710 | 30.752791 | 32.489193 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer3 `o_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/o32/up32` + layer7/11/15/19/23 `k16/o32` まで広げられる。
- この結果はselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-selected-layer-ko-plus-layer3-o32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-selected-layer-ko-plus-layer3-o32-policy-artifact-v0.1.json`

## 次の行動

1. 13 tensor版をcurrent passing branchとして保持する。
2. 次はlayer3 `gate_proj` row-block32を追加して、layer3 MLP coverageを `up` から `up+gate` へ広げられるかを見る。
3. layer7 `up/down/gate` と `o+gate` combined failureは引き続きfailure guardとして残す。
4. full-package real batch throughputは引き続きT1aとして別に進める。
