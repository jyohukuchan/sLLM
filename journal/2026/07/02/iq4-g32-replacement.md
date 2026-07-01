# IQ4 replacement with aq4 g32

## 目的

UD `IQ4_XS` を aq4 g32 に置き換え、IQ4部分のbppを `4.25` のまま保てるか確認した。

## 条件

- source: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
- IQ4置換: `aq4_e4m3_g32_ts_flloyd16`, 8 iterations
- Q4置換: aq4 g16, 8 iterations
- Q5置換: aq5 g16, 8 iterations
- Q6/Q8: UDのまま維持

## 結果

| policy | parameter-weighted bpp | arithmetic relative MSE | element-weighted relative MSE estimate |
|---|---:|---:|---:|
| original UD mixed | 5.255106 | 0.002857467 | 0.003330449 |
| Q4+Q5 replaced, IQ4 kept | 5.255106 | 0.002745454 | 0.003218382 |
| IQ4 g32 + Q4+Q5 replaced | 5.255106 | 0.002770702 | 0.003254025 |
| IQ4 g16 + Q4+Q5 replaced | 5.274714 | 0.002698858 | 0.003152598 |

IQ4 rows only:

| metric | value |
|---|---:|
| UD IQ4 mean relative MSE | 0.005891522 |
| aq4 g32 baseline mean relative MSE | 0.006662778 |
| aq4 g32 alternating mean relative MSE | 0.006345974 |

## 解釈

- IQ4をg32にすると、bppはUD `IQ4_XS` と同じ `4.25` に保てる。
- ただしIQ4 rows単体ではUDより悪化する。
- hybrid全体ではQ4/Q5置換の改善が勝つため、元UDよりは良い。
- 同じbppなら、IQ4は置換せずUD維持の方がMSEは良い。
- bpp増加を許すならIQ4 g16が最も良い。

## Artifact

- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-hybrid-ud-replace-iq4-g32-q4-aq4-q5-aq5-it8.json`
