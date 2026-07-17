# T2 SQ selected-layer k/o layer11 gate scale

## 前回の要点

- current passing branchは `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` だった。
- layer11 `up_proj` row-block32/16はlen4でstrict top1を壊していた。
- 次はlayer11 `gate_proj` row-block32と、失敗時のrow-block16 recoveryを見る段階だった。

## 今回の変更点

- layer11 `gate_proj` row-block32 policyとrow-block16 recovery policyを作成済みのR9700 prompt bundle結果で整理した。
- `gate32` と `gate16` はどちらも `2 / 3` strict top1 passで、len4がAQ4 `110784` からSQ `102446` へ変わった。
- AQ4 len4 top1はSQ top8内の2位に残るが、strict top1 promotion ruleでは不合格なので、layer11 gateはfailure guardとして固定した。
- report、comparison、policy/artifact JSON、state freeze、SQ計画を更新対象にした。

## 次の行動

1. 17 tensor版 `selected-layer-ko-layer3-down64-layer11-down64-plus-layer23-gate32-down64` をcurrent passing branchとして保持する。
2. layer11 `gate_proj` row-block32/16をfailure guardとして残す。
3. 次はlayer15 `gate_proj` row-block32をcurrent passing branch上でprobeする。
