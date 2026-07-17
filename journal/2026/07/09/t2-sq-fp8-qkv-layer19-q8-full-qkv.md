# T2 SQ FP8 QKV layer19 q8 full QKV

## 前回の要点

- layer15 `q8/k16/v16` はprompt bundleとshort batch guardでpassした。
- 次は同じQKV境界をlayer19へ広げる段階だった。

## 今回の変更点

- layer19にも `q8/k16/v16` を追加した。
- prompt bundleとB=1/4/8 short batch guardをR9700で測定した。

## 結果

- prompt bundle strict top1: `3 / 3`
- prompt SQ prefill/decode/e2e tok/s: `61.661523` / `73.332847` / `33.547192`
- prompt case_a margin: `0.000214577`
- short batch strict top1: `3 / 3`
- short batch SQ decode tok/s: B1 `71.386978`, B4 `74.061735`, B8 `73.850648`
- short batch SQ triple count: B1 `15`, B4 `60`, B8 `120`

## 判断

- layer19までのQKV triple境界は診断guard上は維持できた。
- ただしcase_a marginは薄いので、まだfull SQ policyにはしない。

## 次の行動

- layer23 QKV extensionか、広いprompt/text guardへ進む。
