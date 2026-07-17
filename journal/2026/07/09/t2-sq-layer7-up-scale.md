# T2 SQ layer7 up scale

## 前回の要点

- `kup1-layer3-k16-up32` はR9700 token-id model-loop prompt bundleで `3 / 3` strict top1を通った。
- layer7 isolationでは、`layer7-k16-up32` と `layer3-kup-plus-layer7-k16` は通ったが、`layer3-kup-plus-layer7-up32` は `case_a` で崩れた。
- 直近の境界はlayer7 `up_proj` とlayer3 k/up mixed-scale probeのinteractionだった。

## 今回の変更点

- layer3 `k_proj` row-block16 + layer3 `up_proj` row-block32 + layer7 `k_proj` row-block16を固定した。
- layer7 `up_proj` について、fallback、row-block16、row-block64の3条件をpolicy artifact化した。
- R9700 six-layer token-id model-loop prompt bundleで測定した。
- `layer7-up-fallback` はAQ4 baseline top1 `[110784,237950,182949]` と一致して `3 / 3` passだった。
- `layer7-up16` と `layer7-up64` はどちらも `case_a` が `237950` から `193706` へ入れ替わり、`2 / 3` passだった。
- 失敗時もAQ4 top1はSQ top8 rank `2` に残るため、壊滅的崩壊ではなくranking driftとして扱う。

## 次の行動

1. layer7 `up_proj` はfallback維持で現在のT2 passing boundaryへ反映する。
2. `layer7-up16` と `layer7-up64` はfailure guardとして残す。
3. 次はlayer3 k16/up32 + layer7 k16 + layer7 up fallbackを基準に、追加family/layerを1つずつ戻して `case_a` driftが再発する境界を探す。
