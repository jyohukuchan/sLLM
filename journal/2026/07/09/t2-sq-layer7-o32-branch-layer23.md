# T2 SQ layer7 o32 branch layer23

## 前回の要点

- layer7 `o32` branchにlayer19 `k_proj` row-block16と `o_proj` row-block32を追加した10 tensor policyは、3 promptすべてでAQ4 top1を維持した。
- 次の対象は同じbranchでlayer23 `k_proj` row-block16、必要ならlayer23 `o_proj` row-block32を追加することだった。

## 今回の変更点

- `sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-o32-plus-layer23-k16-policy-v0.1.json` を作り、layer23 `k_proj` row-block16を追加した。
- `sq-fp8-layer7-o32-plus-layer11-k16-o32-plus-layer15-k16-o32-plus-layer19-k16-o32-plus-layer23-k16-o32-policy-v0.1.json` を作り、layer23 `k_proj` row-block16と `o_proj` row-block32を追加した。
- R9700でsix-layer token-id model-loop prompt bundleを実行し、どちらもAQ4 top1 `110784,237950,182949` と一致した。

| variant | FP8 tensors | pass | prefill tok/s | decode tok/s | end-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16` | 11 | 3 / 3 | 32.943496 | 32.707575 | 32.912531 |
| `layer7-o32-layer11-o32-layer15-o32-layer19-o32-plus-layer23-k16-o32` | 12 | 3 / 3 | 33.056640 | 32.555004 | 32.990334 |

## 次の行動

1. 12 tensor版をcurrent passing branchとして保持する。
2. 11 tensor版はrollback guardとして残す。
3. 次はlayer3 `o_proj` row-block32を追加して、selected-layer `k/o` branchの穴を埋められるかを見る。
