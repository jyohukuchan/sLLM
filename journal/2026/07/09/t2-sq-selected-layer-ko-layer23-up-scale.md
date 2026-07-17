# T2 SQ selected-layer k/o layer23 up scale

## 前回の要点

- `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` はcurrent passing branch。
- layer23 `down_proj` row-block64は `3 / 3` strict top1 passだった。
- 次はlayer23 `up_proj` row-block32を追加し、必要ならrow-block16 recoveryを試す段階だった。

## 今回の変更点

- layer23 `up_proj` row-block32を追加した17 tensor policy artifactを評価した。
- `up32` がlen4で失敗したため、layer23 `up_proj` だけrow-block16に狭めた `up16` も評価した。
- 結果、up32/up16ともfinal top1は `102446,237950,182949` で、AQ4 baseline `110784,237950,182949` に対して `2 / 3` strict top1 passだった。
- `len4` のAQ4 top1はSQ top8内の2位に残るが、strict top1 promotion ruleではpromoteしない。

## 次の行動

1. current passing branchは `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-down64` のまま保持する。
2. layer23 `up_proj` row-block32/16はfailure guardとして残す。
3. 次はlayer23 `gate_proj` row-block32を同じcurrent branch上でprobeする。
