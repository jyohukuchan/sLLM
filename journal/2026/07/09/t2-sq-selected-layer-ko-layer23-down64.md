# T2 SQ selected-layer k/o layer23 down64

## 前回の要点

- `selected-layer-ko-layer3-down64-plus-layer11-down64` はcurrent passing branch。
- layer15/19 `down_proj` row-block64はいずれも `len4` でstrict top1を壊したためfailure guardになった。
- 次はlayer23 `down_proj` row-block64を同じcurrent branch上で試す段階だった。

## 今回の変更点

- layer23 `down_proj` row-block64を追加した16 tensor policy artifactを評価した。
- R9700 six-layer token-id model-loop prompt bundleで `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` を実行した。
- 結果、final top1は `110784,237950,182949` で、AQ4 baseline `110784,237950,182949` に対して `3 / 3` strict top1 passだった。
- current passing branchはlayer3 `k16/o32/up32/down64` + layer11 `k16/o32/down64` + layer23 `k16/o32/down64` + layers 7/15/19 `k16/o32` に広がった。

## 次の行動

1. current passing branchは `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` とする。
2. layer15/19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer23 `up_proj` row-block32を追加し、必要ならrow-block16 recoveryを試す。
