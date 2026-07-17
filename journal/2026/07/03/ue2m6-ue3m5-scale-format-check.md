# UE2M6 and UE3M5 scale format check

## Context

前回の UE4M4/UE5M4 比較に続いて、同じ `attn_k` tensor・同じ g16 codebook・同じ model で UE2M6 と UE3M5 を試し、UE4M4 と UE5M3 と比較した。

対象:

- model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- tensor: `model.language_model.layers.3.self_attn.k_proj.weight`
- family: `attn_k`
- codebook: g16 codebook を各 candidate 名に複製し、local-scale 形式だけを変更

## Scale Tables

| format | scale count | min | max |
| --- | ---: | ---: | ---: |
| UE2M6 | 191 | 0.015625 | 3.96875 |
| UE3M5 | 223 | 0.0078125 | 15.75 |
| UE4M4 | 239 | 0.0009765625 | 248.0 |
| UE5M3 | 247 | 0.00000762939453125 | 61440.0 |

## Results: scale-window=4

| format | relative MSE | vs UE4M4 | quant local-scale value range |
| --- | ---: | ---: | ---: |
| UE2M6 | 0.005659325674 | +9.8785% | 0.203125-3.96875 |
| UE3M5 | 0.005083352279 | -1.3043% | 0.203125-12.5 |
| UE4M4 | 0.005150528234 | baseline | 0.203125-12.5 |
| UE5M3 | 0.005399506075 | +4.8340% | 0.203125-13.0 |

## Results: scale-window=all

| format | relative MSE | vs UE4M4 | quant local-scale value range |
| --- | ---: | ---: | ---: |
| UE2M6 | 0.005358542485 | +6.3010% | 0.203125-3.96875 |
| UE3M5 | 0.004932830485 | -2.1441% | 0.203125-12.5 |
| UE4M4 | 0.005040912632 | baseline | 0.203125-12.5 |
| UE5M3 | 0.005373172413 | +6.5913% | 0.203125-13.0 |

## Interpretation

この tensor では UE3M5 が最良だった。UE3M5 は UE4M4 より exponent range が狭いが、今回必要な local-scale 値域 `0.203125-12.5` をカバーしており、mantissa が 5 bit ある分だけ BF16-error が下がったと考える。

UE2M6 は mantissa が 6 bit だが、最大値が `3.96875` なので今回の tensor では range が足りない。実際に quantized local-scale の最大 index が scale table の上限に到達している。

UE5M3 は range が広すぎる一方で mantissa が 3 bit なので、今回のように必要 range が UE3M5/UE4M4 内へ収まる tensor では不利だった。

## Artifacts

- codebook artifact: `benchmarks/results/2026-07-03/aq/2026-07-03-aq-scale-format-rust-codebooks-qwen35-9b-g16-ue2m6-ue5m3.json`
- window4 summary: `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-ue2m6-ue5m3-scale-format-window4-attn-k.json`
- exhaustive summary: `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-ue2m6-ue5m3-scale-format-exhaustive-attn-k.json`
- logs: `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-scale-format-{window4,exhaustive}-ue{2m6,3m5,4m4,5m3}-attn_k.log`

## Next

今回の `attn_k` tensor だけなら UE3M5 が有望。次に見るなら、family 全体で UE3M5 が必要 local-scale range を満たすか、UE2M6 の上限張り付きがどの family で起きるかを集計する。
