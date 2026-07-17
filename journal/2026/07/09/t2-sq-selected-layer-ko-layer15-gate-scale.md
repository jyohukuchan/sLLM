# T2 SQ selected-layer k/o layer15 gate scale

## 前回の要点

- current passing branchは `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` だった。
- layer11 `gate_proj` row-block32/16はlen4でstrict top1を壊したためfailure guardになった。
- 次はlayer15 `gate_proj` row-block32を試す段階だった。

## 今回の変更点

- layer15 `gate_proj` row-block32をcurrent passing branchへ追加した18 tensor policyを作成した。
- R9700 prompt bundleでは `3 / 3` strict top1 passで、final top1はAQ4 baselineと同じ `110784,237950,182949` だった。
- layer15 gate32をcurrent passing branchへ昇格し、layer15 `down_proj` row-block64はfailure guardのまま残した。
- report、comparison、policy/artifact JSON、state freeze、SQ計画を更新対象にした。

## 次の行動

1. 18 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer15-gate32-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer15 `down_proj` row-block64をfailure guardとして残す。
3. 次はlayer15 `up_proj` row-block32をcurrent passing branch上でprobeする。
