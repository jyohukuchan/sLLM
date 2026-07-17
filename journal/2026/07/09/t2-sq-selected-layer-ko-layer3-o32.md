# T2 SQ selected-layer k/o layer3 o32

## 前回の要点

- layer7 `o32` branchはlayer23まで `k_proj` row-block16と `o_proj` row-block32を追加しても、3 promptすべてでAQ4 top1を維持した。
- layer3は `k_proj` と `up_proj` のみで、selected-layer `k/o` branchとしてはlayer3 `o_proj` が未確認だった。

## 今回の変更点

- current 12 tensor branchにlayer3 `o_proj` row-block32を追加した13 tensor policyを作った。
- R9700 six-layer token-id model-loop prompt bundleで検証した。
- `selected-layer-ko-plus-layer3-o32` は `3 / 3` strict top1 passだった。

## 結果

| item | value |
| --- | --- |
| final top1 | `110784,237950,182949` |
| AQ4 baseline top1 | `110784,237950,182949` |
| prefill tok/s | `32.766710` |
| decode tok/s | `30.752791` |
| end-to-end tok/s | `32.489193` |
| current boundary | layer3 `k16/o32/up32` + layers 7/11/15/19/23 `k16/o32` |

## 次の行動

1. 13 tensor版をcurrent passing branchとして保持する。
2. 次はlayer3 `gate_proj` row-block32を追加して、layer3 MLP coverageを `up` から `up+gate` へ広げられるかを見る。
3. layer7 `up/down/gate` と `o+gate` combined failureはfailure guardとして維持する。
