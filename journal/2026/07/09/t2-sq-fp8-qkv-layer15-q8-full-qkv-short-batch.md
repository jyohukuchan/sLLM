# T2 SQ FP8 QKV layer15 q8 full QKV short batch guard

## 前回の要点

- layer15 `q8/k16/v16` はprompt bundleでstrict top1 `3 / 3` を維持した。
- `case_a` marginが薄いため、短いbatch guardで確認する必要があった。

## 今回の変更点

- B=1/4/8、`len:2xB`、generated token 1でSQ候補を測定した。
- AQ4 baselineは既存の同一workload短batch guardから再利用した。

## 結果

- strict top1: `3 / 3`
- SQ prefill tok/s: B1 `20.867536`, B4 `50.204700`, B8 `61.288319`
- SQ decode tok/s: B1 `74.871646`, B4 `75.248488`, B8 `75.568954`
- SQ triple count: B1 `12`, B4 `48`, B8 `96`

## 判断

- short batch guardはpassした。
- ただしprompt bundleの薄いmarginは残るため、full SQ policyにはpromoteしない。

## 次の行動

- 広いprompt/text guardまたはlayer19 QKV extensionで次の境界を確認する。
