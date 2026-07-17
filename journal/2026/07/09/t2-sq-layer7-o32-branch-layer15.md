# T2 SQ layer7 o32 branch layer15

## 前回の要点

- layer7 `o32` branchにlayer11 `k_proj` row-block16と `o_proj` row-block32を追加した6 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次の対象は同じbranchでlayer15 `k_proj` row-block16、必要ならlayer15 `o_proj` row-block32を追加することだった。

## 今回の変更点

- `sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-policy-v0.1.json` を作り、layer15 `k_proj` row-block16を追加した。
- `sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-policy-v0.1.json` を作り、layer15 `k_proj` row-block16と `o_proj` row-block32を追加した。
- R9700でsix-layer token-id model-loop prompt bundleを実行し、どちらもAQ4 top1 `110784,237950,182949` と一致した。

| variant | FP8 tensors | pass | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-plus-layer15-k16` | 7 | 3 / 3 | 28.249777 | 28.294170 | 28.255560 |
| `layer7-o32-layer11-o32-plus-layer15-k16-o32` | 8 | 3 / 3 | 32.938634 | 29.980802 | 32.520152 |

## 次の行動

1. 8 tensor版をcurrent passing branchとして保持する。
2. 7 tensor版はrollback guardとして残す。
3. 次は同じbranchでlayer19 `k_proj` row-block16を追加し、通ればlayer19 `o_proj` row-block32も試す。
