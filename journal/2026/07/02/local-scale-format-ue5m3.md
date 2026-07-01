# Local-scale format comparison: unsigned E5M3

## 目的

符号無しE5M3を `local-scale` として使った場合、E4M3/E5M2/E4M0と比べてどうなるか確認した。

## 条件

- source rows: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
- target: 36 sampled tensor rows
- aq: 4bit codebook-index, `block-size=16`, 16-entry `codebook`
- `global-scale`: FP16
- iterations: 8
- UE5M3定義: `tools/run-aq-tensor-sample.py` の `decode_ue5m3()`

## 結果

| local-scale | effective bpp | mean relative MSE | vs E4M3 | low/high scale clamp |
|---|---:|---:|---:|---:|
| E4M3 | 4.50 | 0.005071093 | 1.0000x | 0 / 0 |
| unsigned E5M3 | 4.50 | 0.005092034 | 1.0041x | 0 / 0 |
| E5M2 | 4.50 | 0.005784078 | 1.1406x | 0 / 0 |
| unsigned E4M0 | 4.25 | 0.008741902 | 1.7239x | 0 / 0 |

UE5M3は36 rows中8 rowsでE4M3より良かった。特に `linear_attn_qkv` の一部と `attn_k` で改善が見えたが、平均ではE4M3より少し悪い。

## 解釈

- UE5M3はE5M2よりかなり良い。rangeを広げつつmantissaを3bit保つので、E5M2ほど粗くならない。
- 今回のsampleではE4M3にscale range不足がないため、平均ではE4M3が最良。
- UE5M3はfamily/tensorごとの選択候補としては残す価値がある。特にE4M3でclampが出るtensorや、実測でUE5M3が勝つfamilyに限定して使うのが妥当。

## Artifact

- `benchmarks/results/2026-07-02/aq/2026-07-02-aq-local-scale-ue5m3-g16-it8-ud36.json`
