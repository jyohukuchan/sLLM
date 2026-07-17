# T2 SQ selected-layer k/o layer19 down64

## 前回の要点

- `selected-layer-ko-layer3-down64-plus-layer11-down64` はcurrent passing branch。
- layer15 `down_proj` row-block64は `len4` でstrict top1を壊したためfailure guardになった。
- 次はlayer19 `down_proj` row-block64を同じcurrent branch上で試す段階だった。

## 今回の変更点

- layer19 `down_proj` row-block64を追加した16 tensor policy artifactを評価した。
- R9700 six-layer token-id model-loop prompt bundleで `selected-layer-ko-layer3-down64-layer11-down64-plus-layer19-down64` を実行した。
- 結果、final top1は `102446,237950,182949` で、AQ4 baseline `110784,237950,182949` に対して `2 / 3` strict top1 passだった。
- `len4` のAQ4 top1はSQ top8内の2位に残るが、strict top1 promotion ruleではpromoteしない。

## 次の行動

1. current passing branchは `selected-layer-ko-layer3-down64-plus-layer11-down64` のまま保持する。
2. layer19 `down_proj` row-block64はfailure guardとして残す。
3. 次はlayer23 `down_proj` row-block64を同じcurrent branch上でprobeする。
