# UE4M4 vs UE5M4 scale check

## Context

前回の `attn_k` dry-run では `aq4_ue5m4_g16_ts_flloyd16` が BF16-error の relative MSE `0.005150528234` だった。君の確認依頼に合わせて、UE4M4 がこの値と同じになるかを同じ tensor・同じ codebook・同じ `scale-window=4` 条件で確認した。

対象:

- model: `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- tensor: `model.language_model.layers.3.self_attn.k_proj.weight`
- family: `attn_k`
- codebook: `aq4_ue5m4_g16_ts_flloyd16` の g16 codebook を UE4M4/UE5M4 candidate 名に複製
- local-scale search: `scale-window=4`

## Result

| candidate | scale count | quant scale index range | quant local-scale value range | relative MSE |
| --- | ---: | ---: | ---: | ---: |
| `aq4_ue4m4_g16_ts_flloyd16` | 239 | 73-168 | 0.203125-12.5 | 0.005150528234 |
| `aq4_ue5m4_g16_ts_flloyd16` | 495 | 201-296 | 0.203125-12.5 | 0.005150528234 |

差分:

- `UE4M4 - UE5M4` relative MSE delta: `0.0`
- ratio: `1.0`

UE4M4 の scale table は UE5M4 の部分集合で、今回の quantized local-scale は `0.203125` から `12.5` に収まった。したがって今回の tensor では UE5M4 の追加 exponent 範囲は使われず、UE4M4 と UE5M4 は完全に同じ BF16-error になった。

## Artifacts

- codebook artifact: `benchmarks/results/2026-07-03/aq/2026-07-03-aq-scale-dominance-rust-codebooks-qwen35-9b-g16-ue4m4.json`
- summary: `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-ue4m4-vs-ue5m4-attn-k.json`
- logs:
  - `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-ue4m4-vs-ue5m4-ue4m4-attn_k.log`
  - `benchmarks/results/2026-07-03/aq/2026-07-03-ullm-quant-ue4m4-vs-ue5m4-ue5m4-attn_k.log`

## Next

この結果だけなら、`attn_k` のこの tensor では UE4M4 で十分だと見ていい。ただし tensor family 全体で同じとは限らないので、次に見るなら `attn_k` 全層または family 複数で UE4M4/UE5M4 の差分と scale clamp の有無を集計する。
