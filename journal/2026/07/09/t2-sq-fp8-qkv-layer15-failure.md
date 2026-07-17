# T2 SQ FP8 QKV layer15 failure

## 前回の要点

- layer3+7+11 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- 次はlayer15 QKVを追加して累積driftを見る段階だった。

## 今回の変更点

- layer3+7+11+15 `q16/k16/v16` policyとartifactを作成した。
- R9700 full mixed prompt bundleでAQ4 baselineとSQを比較した。

## 結果

- SQ prefill tok/s: `62.929075`
- SQ decode tok/s: `74.876312`
- SQ end-to-end tok/s: `34.095032`
- strict top1: `2 / 3`
- final top1: AQ4 `24218,4105,329` / SQ `24218,5582,329`
- `case_a` はAQ4 top1 `4105` がSQ top8 rank 2に残り、SQ top1 `5582` との差は `0.001344681`。

## 次の行動

- full layer15 QKVはpromoteしない。
- layer15 Q/K/V splitで単独projectionのhard failureか累積driftかを切り分ける。
