# T2 six-layer family boundary

## 前回の要点

- T2 promotion rule v0.1は `strict_top1` にした。
- `v` fallback + `q/k/o/gate/up/down` row-block32 FP8は、4-5層では有望だったが6層で崩れた。
- layer `23` 単体は `q/v` fallbackで回復したが、6層bundleは回復しなかった。

## 今回の変更点

- layers `3,7,11,15,19,23` のfamily splitをrow-block32で実行した。
- `k` と `up` は単独でstrict top1一致だった。
- `o`、`gate`、`down` は単独でstrict top1不一致だった。
- `o/gate/down` をrow-block16にしてもstrict top1には戻らなかった。
- `k/up` row-block32を同時にFP8化した6層部分候補は、len4、case_a、case_bの3 promptでstrict top1一致だった。
- case_aのtop8 overlapは `2 / 8` と低いため、`k/up` は回帰guardとしては使えるが、promoted SQ policyではない。
- 結果を `benchmarks/results/2026-07-08/sq-fp8-six-layer-family-boundary-v0.1.md` に保存した。

## 次の行動

1. `k/up` row-block32は6層strict-top1 regression subsetとして保持する。
2. `q/v/o/gate/down` はfallbackまたは別format/scale/layoutが必要な対象として扱う。
3. 診断gapだけで順序を付けるなら、`o/down` を `gate` より先に見る。
4. full SQ policyとしてT5 throughputへ進むには、coverageがまだ不足している。
