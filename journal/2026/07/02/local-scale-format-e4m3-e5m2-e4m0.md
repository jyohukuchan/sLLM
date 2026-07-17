# Local-scale format comparison: E4M3, E5M2, E4M0

## 目的

familyごとのraw-value幅が違うため、`local-scale` をE4M3/E5M2で使い分ける価値があるか確認した。
また、E4M3の代わりに符号無しE4M0を使うとどれくらい悪化するかを測った。

## 条件

- source rows: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
- target: 36 sampled tensor rows
- aq: 4bit codebook-index, `block-size=16`, 16-entry `codebook`
- `global-scale`: FP16
- iterations: 8
- E4M0定義: 16個のpower-of-two値 `2^(code-7)`、4bit scale

## 結果

| local-scale | effective bpp | mean relative MSE | vs E4M3 | low/high scale clamp |
|---|---:|---:|---:|---:|
| E4M3 | 4.50 | 0.005071093 | 1.0000x | 0 / 0 |
| E5M2 | 4.50 | 0.005784078 | 1.1406x | 0 / 0 |
| unsigned E4M0 | 4.25 | 0.008741902 | 1.7239x | 0 / 0 |

family別でも、今回の36 rowsでは全familyでE4M3がE5M2より良かった。

## 解釈

- 今回の範囲ではE4M3のrange不足は出ていない。scale indexの下限/上限張り付きは0だった。
- E5M2はrangeが広いが、mantissaがE4M3より粗いため、rangeが不要なtensorでは悪化する。
- E4M0はlocal-scale overheadを `8/16=0.5 bpp` から `4/16=0.25 bpp` へ下げられるが、MSEは約72%悪化した。
- E5M2を採用するなら、E4M3でscale clampが出るtensor/familyだけに限定するのが妥当。

## Artifact

- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-local-scale-e4m3-e5m2-e4m0-g16-it8-ud36.json`
