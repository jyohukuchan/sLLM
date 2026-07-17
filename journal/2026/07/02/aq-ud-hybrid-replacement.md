# AQ and Unsloth Dynamic hybrid replacement check

## 目的

UD `Q4_K_XL` のうち `IQ4_XS`、`Q4_K`、`Q5_K` を今回の aq g16 交互最適化で置き換え、`Q6_K` と `Q8_0` は残す案を確認した。

## 結果

- 対象: `2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl` の36 sampled tensor rows。
- 置換実測: `IQ4_XS`、`Q4_K`、`Q5_K` の27 rowsに `aq4_e4m3_g16_ts_flloyd16`、8 iterationsを適用。
- 置換結果JSON: `benchmarks/results/2026-07-02/aq/2026-07-02-aq-hybrid-ud-replace-iq4-q4-q5-g16-it8.json`

| policy | replaced rows | parameter-weighted bpp | arithmetic relative MSE | element-weighted relative MSE estimate |
|---|---:|---:|---:|---:|
| original UD mixed | 0 | 5.255106 | 0.002857467 | 0.003330449 |
| replace Q4 only | 14 | 5.255106 | 0.002757221 | 0.003227169 |
| replace IQ4 only | 2 | 5.274714 | 0.002810871 | 0.003264665 |
| replace IQ4+Q4 | 16 | 5.274714 | 0.002710625 | 0.003161386 |
| replace IQ4+Q4+Q5 | 27 | 5.039420 | 0.003849839 | 0.004040111 |

## 解釈

- `IQ4_XS` と `Q4_K` の置換は有効。
- `Q5_K` の置換は悪化が大きい。UDの `Q5_K` rowsは平均 relative MSE `0.001341953` だが、aq g16 交互最適化では `0.005070291` まで悪化した。
- 現時点の有力案は `IQ4_XS` と `Q4_K` だけをaq g16へ置換し、`Q5_K`、`Q6_K`、`Q8_0` は残す案。
