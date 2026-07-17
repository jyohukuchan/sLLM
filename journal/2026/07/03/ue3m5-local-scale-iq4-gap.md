# UE3M5 local-scale and IQ4 gap check

## Context

IQ4/Q4 置換で `aq4_e4m3_g16_ts_flloyd16` がどこまで効くかを見ていたが、LUT 遅延の検証から 8bit local-scale code を FP16 payload へ LUT 変換する cost は大きくなさそうだった。そこで、計算しやすい E4M3 固定から、精度寄りの UE3M5/E3M4 を優先候補として試した。

比較対象:

- source rows: `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`
- model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- AQ: 4bit codebook-index, `block-size=16`, free-Lloyd16 codebook, FP16 global-scale, 8 iterations
- formats: E3M4, UE3M5, E4M3

## g16 Results

All 36 UD rows:

| local-scale | arithmetic mean relative MSE | element-weighted relative MSE | high clamp count |
| --- | ---: | ---: | ---: |
| E3M4 | 0.005537808 | 0.005133294 | 15 |
| UE3M5 | 0.005311130 | 0.004997983 | 14 |
| E4M3 | 0.005068075 | 0.005058458 | 0 |

Arithmetic mean では E4M3 が最良だが、これは Q8_0 の `linear_attn.out_proj` rows で UE3M5/E3M4 が range 上限に張り付いて大きく悪化したため。element-weighted では UE3M5 が E4M3 を少し上回った。

IQ4_XS/Q4_K rows:

| GGML type | UD mean | E3M4 g16 | UE3M5 g16 | E4M3 g16 |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS | 0.005891522 | 0.004758629 | 0.004703785 | 0.005031143 |
| Q4_K | 0.005324426 | 0.004785084 | 0.004729071 | 0.005061026 |

IQ4_XS/Q4_K では UE3M5 が明確に E4M3 より良い。E4M3 と比べると IQ4_XS で約 `6.5%`、Q4_K で約 `6.6%` の改善。

## IQ4 same-bpp g32

IQ4_XS rowsだけを `block-size=32` で比較した。

| local-scale | bpp | mean relative MSE | vs UD IQ4 |
| --- | ---: | ---: | ---: |
| UD IQ4_XS | 4.25 | 0.005891522 | baseline |
| E3M4 g32 | 4.25 | 0.006083797 | 1.0326x |
| UE3M5 g32 | 4.25 | 0.005972981 | 1.0138x |
| E4M3 g32 | 4.25 | 0.006322118 | 1.0731x |

UE3M5 は同じ 4.25 bpp の g32 でも E4M3 よりかなり改善し、UD IQ4 との差を約 `1.4%` まで縮めた。ただし、まだ UD IQ4 には届いていない。

## Hybrid Policy Impact

| policy | parameter-weighted bpp | arithmetic relative MSE | element-weighted relative MSE |
| --- | ---: | ---: | ---: |
| original UD mixed | 5.255106 | 0.002857467 | 0.003330449 |
| replace Q4 only with UE3M5 g16 | 5.255106 | 0.002625940 | 0.003066975 |
| replace IQ4 g32 + Q4 g16 with UE3M5 | 5.255106 | 0.002630466 | 0.003073364 |
| replace IQ4+Q4 with E4M3 g16 | 5.274714 | 0.002707235 | 0.003157062 |
| replace IQ4+Q4 with E3M4 g16 | 5.274714 | 0.002584785 | 0.003003941 |
| replace IQ4+Q4 with UE3M5 g16 | 5.274714 | 0.002559955 | 0.002973819 |

同じ parameter-weighted bpp を保つなら、IQ4はUDのまま残して Q4_K だけ UE3M5 g16 へ置換する方が、IQ4をUE3M5 g32へ置換するよりわずかに良い。少し bpp 増加を許すなら、IQ4+Q4 を UE3M5 g16 へ置換する方がさらに良い。

## Interpretation

IQ4に勝てなかった理由のうち、E4M3 local-scale の精度不足はかなり大きかった。UE3M5 は必要 range が `~16` 以内に収まる tensor では、mantissa が5bitあるため E4M3 より有利になる。

一方で、同じ 4.25 bpp の g32 UE3M5 は UD IQ4 にまだ少し負ける。したがって残差は local-scale だけでなく、`block-size=32` の自由度不足、IQ4側の super-block/layout、または局所分布への適応度にあると考える。

Q8_0 の `linear_attn.out_proj` rows では UE3M5/E3M4 が上限に張り付き、E4M3の広い range が必要だった。したがって UE3M5 を全 tensor に一律適用するのは危険で、family/tensorごとの local-scale format 選択が必要。

## Next

1. `UE3M5 g16` を IQ4/Q4 型の主要候補へ昇格する。
2. `linear_attn_out` など range が広い tensor は E4M3 または UE4M4 系を fallback にする。
3. 次の切り分けは `block-size=32` の残差を見るため、IQ4_XS rowsで g32 の codebook粒度、super-block補正、outlier補正を試す。

## Artifacts

- helper: `tools/run-aq-local-scale-format-comparison.py`
- g16 result: `benchmarks/results/2026-07-03/aq/2026-07-03-aq-local-scale-e3m4-ue3m5-e4m3-g16-it8-ud36.json`
- g32 IQ4 result: `benchmarks/results/2026-07-03/aq/2026-07-03-aq-local-scale-e3m4-ue3m5-e4m3-g32-it8-iq4.json`
- hybrid summary: `benchmarks/results/2026-07-03/aq/2026-07-03-aq-hybrid-local-scale-ue3m5-policy-summary.json`
