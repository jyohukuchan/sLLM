# T2 SQ layer7 o32 branch layer11

## 前回の要点

- layer7 `o_proj` row-block32と `gate_proj` row-block32は個別にstrict top1を維持した。
- layer7 `o32+gate32` はrow-block16化しても `case_a` のdriftを回復しなかった。
- `case_a` のmarginは `o32` branchの方が広かったため、次のcoverage拡張は `o32` branchから進めた。

## 今回の変更点

- `sq-fp8-layer7-o32-plus-layer11-k16-policy-v0.1.json` を作り、layer11 `k_proj` row-block16を追加した。
- `sq-fp8-layer7-o32-plus-layer11-k16-o32-policy-v0.1.json` を作り、layer11 `k_proj` row-block16と `o_proj` row-block32を追加した。
- R9700でsix-layer token-id model-loop prompt bundleを実行し、どちらもAQ4 top1 `110784,237950,182949` と一致した。

| variant | FP8 tensors | pass | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-plus-layer11-k16` | 5 | 3 / 3 | 33.188099 | 32.853465 | 33.144065 |
| `layer7-o32-plus-layer11-k16-o32` | 6 | 3 / 3 | 31.492675 | 32.453352 | 31.614742 |

## 次の行動

1. 6 tensor版をcurrent passing branchとして保持する。
2. 5 tensor版はrollback guardとして残す。
3. 次は同じbranchでlayer15 `k_proj` row-block16を追加し、通ればlayer15 `o_proj` row-block32も試す。
