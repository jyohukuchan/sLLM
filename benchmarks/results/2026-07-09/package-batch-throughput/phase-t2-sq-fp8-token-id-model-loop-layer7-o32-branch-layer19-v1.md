# T2 SQ FP8 token-id model-loop layer7 o32 branch layer19 v1

## 前回の要点

- layer7 `o32` branchにlayer15 `k_proj` row-block16と `o_proj` row-block32を足した8 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次のT2対象は、同じ `o32` branchでlayer19 `k_proj` row-block16、必要ならlayer19 `o_proj` row-block32を追加して、strict top1の境界を見ることだった。

## 今回の変更点

- layer19 `k_proj` row-block16を追加した9 tensor policyを作成した。
- さらにlayer19 `o_proj` row-block32も追加した10 tensor policyを作成した。
- R9700で同じsix-layer token-id model-loop prompt bundleを実行し、AQ4 baseline top1と比較した。

## R9700 result

| row | coverage | FP8 tensors | passthrough tensors | strict top1 pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16` | layer3 k16/up32 + layer7 k16/o32 + layer11 k16/o32 + layer15 k16/o32 + layer19 k16; layer7 up/down/gate fallback | 9 | 766 | 3 / 3 | `110784,237950,182949` | 1 | 33.194584 | 32.799897 | 33.142565 |
| `layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16-o32` | layer3 k16/up32 + layer7 k16/o32 + layer11 k16/o32 + layer15 k16/o32 + layer19 k16/o32; layer7 up/down/gate fallback | 10 | 765 | 3 / 3 | `110784,237950,182949` | 1 | 33.076310 | 32.841953 | 33.045552 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer19 `k_proj` row-block16を足した9 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer19 `o_proj` row-block32も追加した10 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- 現在のpassing boundaryは、layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` + layer15 `k16/o32` + layer19 `k16/o32` まで広げられる。
- ただしこれはselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16/raw.json`
- `sq-layer7-o32-layer11-o32-layer15-o32-plus-layer19-k16-o32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-o32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-o32-policy-artifact-v0.1.json`

## 次の行動

1. 10 tensor版をpassing branchとして保持し、9 tensor版はrollback guardとして残す。
2. 次は同じ `o32` branchでlayer23の `k_proj` row-block16、必要なら `o_proj` row-block32を追加して、どこでstrict top1が崩れるかを見る。
3. layer7 `gate32` branchや `o32+gate32` 回復は、layer方向の広がりを一度見た後に戻る。
4. full-package real batch throughputは引き続きT1aとして別に進める。
