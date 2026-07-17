# T2 SQ layer7 o/gate scale

## 前回の要点

- layer7 add-family probeでは、layer7 `o_proj` row-block32と `gate_proj` row-block32は個別に `3 / 3` passした。
- ただし `o32+gate32` の同時追加は `case_a` で `193706` へ入れ替わった。
- 次の確認は、`o/gate` の同時追加がrow-block幅の強化で回復するかを見ることだった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` と `down_proj` はfallbackのまま、`o/gate` の組み合わせを `o16+gate32`、`o32+gate16`、`o16+gate16` で評価した。
- 3候補すべて `case_a` が `237950` から `193706` へ入れ替わり、`2 / 3` passだった。
- 失敗時もAQ4 top1はSQ top8 rank `2` に残るため、ranking driftとして扱う。

## 次の行動

1. `o+gate` 同時追加は現行W8A16/F32 row-block scaleではfailure guardとして保持する。
2. 次は `o32` branchまたは `gate32` branchのどちらかを選び、coverageを広げる。
3. `o+gate` 同時追加は、別scale layout、別dtype、またはtext-level acceptance guardの導入後に再評価する。
