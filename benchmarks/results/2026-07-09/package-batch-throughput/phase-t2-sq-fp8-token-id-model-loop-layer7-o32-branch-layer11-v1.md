# T2 SQ FP8 token-id model-loop layer7 o32 branch layer11 v1

## 前回の要点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定したうえで、layer7 `o_proj` row-block32と`gate_proj` row-block32は単独passした。
- layer7 `o32+gate32` は単独pass同士の組み合わせだが `case_a` でstrict top1を落とした。
- `case_a` のtop1 marginはlayer7 `o32` branchの方が`gate32` branchより広かったため、次のcoverage拡張は`o32` branchから進める判断だった。

## 今回の変更点

- layer7 `o32` branchにlayer11 `k_proj` row-block16を追加した5 tensor policyを作成した。
- さらにlayer11 `o_proj` row-block32も追加した6 tensor policyを作成した。
- R9700で同じsix-layer token-id model-loop prompt bundleを実行し、AQ4 baseline top1と比較した。

## R9700 result

| row | coverage | FP8 tensors | passthrough tensors | strict top1 pass | final top1 | case_a AQ4 rank in SQ top8 | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `layer7-o32-plus-layer11-k16` | layer3 k16/up32 + layer7 k16/o32 + layer11 k16; layer7 up/down/gate fallback | 5 | 770 | 3 / 3 | `110784,237950,182949` | 1 | 33.188099 | 32.853465 | 33.144065 |
| `layer7-o32-plus-layer11-k16-o32` | layer3 k16/up32 + layer7 k16/o32 + layer11 k16/o32; layer7 up/down/gate fallback | 6 | 769 | 3 / 3 | `110784,237950,182949` | 1 | 31.492675 | 32.453352 | 31.614742 |

AQ4 baseline top1: `110784,237950,182949`

## 判断

- layer7 `o32` branchにlayer11 `k_proj` row-block16を足した5 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- layer11 `o_proj` row-block32も追加した6 tensor policyも、3 promptすべてでAQ4 top1を維持した。
- したがって現在のpassing boundaryは、layer3 `k16/up32` + layer7 `k16/o32` + layer11 `k16/o32` まで広げられる。
- ただしこれはselected-layer model-loop guardであり、full LM throughputや最終SQ性能とは扱わない。

## Artifacts

- `results.jsonl`
- `comparison.json`
- `sq-layer7-o32-plus-layer11-k16/raw.json`
- `sq-layer7-o32-plus-layer11-k16-o32/raw.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-layer7-o32-plus-layer11-k16-o32-policy-artifact-v0.1.json`

## 次の行動

1. 6 tensor版をpassing branchとして保持し、5 tensor版はrollback guardとして残す。
2. 次は同じ`o32` branchでlayer15の`k_proj` row-block16、必要なら`o_proj` row-block32を追加して、どこでstrict top1が崩れるかを見る。
3. layer7 `gate32` branchや`o32+gate32`回復は、layer方向の広がりを一度見た後に戻る。
4. full-package real batch throughputはT1aとして別に継続する。
