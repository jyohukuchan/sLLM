# FlashAttention2-style before/after token/s table

## 前回の要点

- `cached_prefix_flash2` はFlashAttention2-styleのtiled online softmax executorとして導入した。
- 後段の `cached_prefix_rocwmma_fp8` / `cached_prefix_rdna4_fp8_auto` は、比較shapeが `kv_heads=1` に変わっている。

## 今回の変更点

- `cached_prefix_chunked` 導入前相当と `cached_prefix_flash2` 導入後を、同一shapeのtoken/sで比較する表を作成した。
- 保存先は `uLLM-project/benchmarks/results/2026-07-08/runtime-cached-prefix-fp8-kv/phase-c19-flash2-before-after-token-s-v1.md`。
- 数値は `phase-c5-flash2-tiled-online-softmax-v1.md` の同一sweep結果を使った。

## 次の行動

1. rocWMMA/autoまで含めた現行best比較を作る場合は、`kv_heads=1` の同一shapeで `cached_prefix_chunked` も再測してから別表にする。
2. SQ候補評価では、表の `cached_prefix_flash2` を「FlashAttention2-style scalar導入後」、`cached_prefix_rdna4_fp8_auto` を「現行RDNA4 FP8 best routing」として分けて扱う。
