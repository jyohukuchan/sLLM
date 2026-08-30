# Phase 53 follow-up: E5M2 block16 scale selector experiment

> 状態: 完了・両候補棄却（2026-08-27）

## 目的

exact `gfx1030`／Qwen3.5-4B BF16だけで、E5M2 block16のscale selectorを比較し、standard MXFP8 block32および過去の
block16 recipeより品質が改善するか確認する。公開format、descriptor、default mapping、E4M3、gfx1201、gfx942は変更しない。

## 固定比較

- baseline block16 v1: KLD p99 `0.04331390780013198`。
- baseline block16 v2 `StandardMxFloorPowerV1`: KLD p99 `0.03659844555378746`。
- reference-only MXFP8 E5 block32: KLD p99 `0.03218873133110086`。
- candidate A: `LocalMinMse`。各16値で`e16`、`e16-1`、`e16+1`をRNE＋SATし、再構成SSE最小のE8M0 scaleを選ぶ。
- candidate B: `Parent32GuardedMinMse`。candidate Aへ同じ32値parentのstandard MX exponent `e32`を追加する。

## 受入条件

- host／GPUのscale byte、E5M2 value byte、direct attention numerical oracleが一致し、fallback 0、cleanup 0である。
- FP16、candidate block16、MXFP8を同時常駐させず完全直列に一回計測する。
- KLD p99、top-1、perplexity、task、long-contextを既存dataset／model lockで比較する。
- MXFP8 KLD p99を下回った候補だけを「relative accuracy improvement」とする。default採用はこの診断の対象外である。
- 診断用selectorは公開descriptorへ混在させず、採用判断まではproduction既定sourceを`StandardMxFloorPowerV1`へ戻す。

## Closeout

exact gfx1030で両候補のhost／GPU byte exact、direct attention、fallback 0、cleanup 0をPASSした。完全直列の一回品質測定は
両候補ともKLD p99 `0.04063529273873547`、top-1 `0.8`、long-context loss `0.16666666666666663`で同一だった。
旧block16 v1の`0.04331390780013198`より改善したが、production block16 v2の`0.03659844555378746`とMXFP8 block32の
`0.03218873133110086`には届かなかったため両方棄却した。

診断summaryは`external:phase53/gfx1030/e5-scale-selector-diagnostic-summary-v1.json`、SHA-256
`17622a66dee6d7312028a8699683f9e76a7919764bf6a1a856a994f48b2aebf6`である。raw runner reportは再利用したv2 descriptor文字列を
持つためaggregateへ入力せず、summaryがselectorと別binary digestを結合する。production sourceは`StandardMxFloorPowerV1`へ復元し、
default mappingを変更しない。MI300Xはdeferredのまま、本実験では使用していない。

[全体計画](../../../../main-plan.md) /
[Phase 53保存済み計画](../../../../archive/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md) /
[Phase 53履歴](../../../../../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)
