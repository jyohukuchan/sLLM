# T2 kup6_gate5_down5 prompt bundle

## 前回の要点

- `k/up` row-block32は6層prompt bundleでstrict top1一致だったが、coverageが低かった。
- `k/up` 全6層に `o/gate/down` のうち2 familyまでをlayers `3,7,11,15,19` で足すとlen4 strict top1を維持した。
- len4上の最有力候補は `kup6_gate5_down5` だった。

## 今回の変更点

- `kup6_gate5_down5` をcase_a/case_bへ広げた。
- len4、case_a、case_bの `3 / 3` でstrict top1一致だった。
- case_aのtop8 overlapは `2 / 8` と低く、full SQ policyではなく6層regression subsetとして扱うことにした。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-six-layer-kup6-gate5-down5-prompt-bundle-v0.1.md` に保存した。
- 選択FP8/fallback方針を `benchmarks/results/2026-07-08/sq-fp8-kup6-gate5-down5-policy-v0.1.json` と `.md` に保存した。

## 次の行動

1. `kup6_gate5_down5` を現在の6層strict-top1 regression subsetとして使う。
2. `kup6_ogatedown5` はnear-miss failure guardとして残す。
3. T1 real batch runnerを進め、SQ候補評価で使えるthroughput行を作る。
