# T2 SQ layer7 add family

## 前回の要点

- layer7 `up_proj` scale probeでは、layer3 k16/up32 + layer7 k16のpassing subsetを維持するにはlayer7 `up_proj` fallbackが必要だった。
- 直近のT2対象は、このpassing boundaryにlayer7の追加familyを1つずつ戻し、`case_a` driftが再発する境界を探すことだった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` はfallbackのまま、layer7 `o_proj` row-block32、`gate_proj` row-block32、`down_proj` row-block64を個別に追加した。
- `layer7-plus-o32` はAQ4 baseline top1 `[110784,237950,182949]` と一致して `3 / 3` passだった。
- `layer7-plus-gate32` も `3 / 3` passだった。
- `layer7-plus-down64` は `case_a` が `237950` から `111791` へ入れ替わり、`2 / 3` passだった。
- 単独pass同士の `layer7-plus-o32-gate32` も `case_a` が `193706` へ入れ替わり、`2 / 3` passだった。

## 次の行動

1. `layer7-plus-o32` と `layer7-plus-gate32` はpassing probesとして保持する。
2. `layer7-plus-down64` と `layer7-plus-o32-gate32` はfailure guardsとして残す。
3. 次は `o32+gate32` の組み合わせをより強いscale/layoutで回復できるか試すか、`o32` または `gate32` の片側branchでcoverageを広げる。
