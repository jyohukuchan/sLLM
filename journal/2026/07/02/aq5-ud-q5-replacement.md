# AQ5 replacement for UD Q5 rows

## 目的

UD `Q4_K_XL` の `Q5_K` rowsを、現在のcodebook決定アルゴリズムを5bit/32-entry codebookへ拡張した aq5 g16 で置換できるか確認した。

## 条件

- source: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
- Q5置換: `aq5_e4m3_g16_ts_flloyd32`
  - `codebook-index`: 5bit
  - `codebook`: 32 entries, FP16-rounded
  - `block-size`: 16
  - `local-scale`: E4M3
  - `global-scale`: FP16
  - iterations: 8
- IQ4/Q4置換は前回の aq4 g16 8-iteration結果を再利用。
- Q6/Q8はUDのまま維持。

## 結果

| policy | parameter-weighted bpp | arithmetic relative MSE | element-weighted relative MSE estimate |
|---|---:|---:|---:|
| original UD mixed | 5.255106 | 0.002857467 | 0.003330449 |
| Q4+Q5 replaced, IQ4 kept | 5.255106 | 0.002745454 | 0.003218382 |
| IQ4+Q4+Q5 replaced | 5.274714 | 0.002698858 | 0.003152598 |

Q5 rows only:

| metric | value |
|---|---:|
| UD Q5 mean relative MSE | 0.001341953 |
| aq5 g16 baseline mean relative MSE | 0.001349332 |
| aq5 g16 alternating mean relative MSE | 0.001303442 |

## 解釈

- aq5 g16 交互最適化はUD `Q5_K` rowsを少し上回った。
- `Q4+Q5`だけを置換すれば、parameter-weighted bppを元UDと同じ `5.255106` に保ったままMSEを下げられる。
- `IQ4`も置換するとMSEはさらに下がるが、parameter-weighted bppは `5.274714` へ少し上がる。
- この比較はsampled tensor上の推定であり、次は同じpolicyでactivation-weighted metricとmodel-level lossを見る必要がある。

## Artifact

- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-hybrid-ud-replace-iq4-q4-aq4-q5-aq5-g16-it8.json`
