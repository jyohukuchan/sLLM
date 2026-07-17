# T2 six-layer per-layer combination boundary

## 前回の要点

- `k/up` row-block32は6層3 promptでstrict top1一致だった。
- `o/gate/down` は6層単独ではstrict top1不一致だった。
- `o/gate/down` のrow-block16でも6層strict top1は回復しなかった。

## 今回の変更点

- layers `3,7,11,15,19` では、`o/gate/down` row-block32がそれぞれstrict top1一致だった。
- `k/up` 全6層に `o5`、`gate5`、`down5` のどれか1 familyを足した3ケースはすべてlen4でstrict top1一致だった。
- `k/up` 全6層に `o5/gate5`、`o5/down5`、`gate5/down5` の2 familyを足した3ケースもすべてlen4でstrict top1一致だった。
- `k/up` 全6層に `o5/gate5/down5` の3 familyをすべて足すとlen4でstrict top1不一致だった。
- len4上の次のprompt-bundle候補は、top8 overlapが最も高い `kup6_gate5_down5` とした。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-six-layer-per-layer-combination-boundary-v0.1.md` に保存した。

## 次の行動

1. `kup6_gate5_down5` をcase_a/case_bへ広げる。
2. `kup6_ogatedown5` はnear-miss failure guardとして残す。
3. prompt bundleが通っても、full SQ policyにはまだcoverage不足なので昇格しない。
