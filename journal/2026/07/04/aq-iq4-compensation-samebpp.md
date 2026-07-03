# AQ IQ4 same-bpp compensation experiment

## 前回の要点

- Qwen3.5-9BのUD IQ4_XS部分を4bit + g32のAQ方式で置換すると、UE3M5 + E3M4 monotonic floorでも平均BF16-errorはUDより悪かった。
- 前回基準値は、UD IQ4_XS平均relative MSE `0.005891522356767763`、AQ baseline UE3M5 g32平均relative MSE `0.005972981358`だった。
- したがって、同じ `4.25 bpp` 枠でIQ4_XSを超えるには、local-scaleやcodebook-indexだけではなく、追加metadataの使い方を試す必要があった。

## 今回の変更点

- `tools/run-aq-iq4-compensation-experiment.py` を追加した。
- 対象は `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl` の `IQ4_XS` 2 tensor。
- モデルは `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`。
- 条件は `block-size=32`、`max-elements-per-tensor=262144`、`iterations=8`、`scale-window=4`、`local-scale=UE3M5`、`global-scale=FP16`、`E3M4 monotonic floor`、`torch-threads=64`。
- 結果は `benchmarks/results/2026-07-04/aq/2026-07-04-aq-iq4-compensation-qwen35-9b-iq4-samebpp.json` に保存した。

## 試した方式

| 方式 | bpp | 内容 |
|---|---:|---|
| `baseline_ue3m5_g32` | 4.250000 | 4bit codebook-index + 8bit local-scale / 32 raw-values |
| `sb_index_bias_l7_s8` | 4.250000 | 7bit local-scale index + superblockごとの8bit base補正 |
| `sb_scale8_l7_s8` | 4.250000 | 7bit local-scale + 8bit superblock-scale / 8 blocks |
| `sb_scale4_l7_s4` | 4.250000 | 7bit local-scale + 4bit superblock-scale / 4 blocks |
| `outlier1_s16_l7_i7` | 4.250000 | 7bit local-scale + 16 blocksごとに1つのoutlier補正 |
| `outlier1_s8_l6_i8` | 4.250000 | 6bit local-scale + 8 blocksごとに1つのoutlier補正 |
| `dual_codebook_l7_sel1` | 4.250000 | 7bit local-scale + blockごとの1bit codebook-selector |
| `clipped_p95_l8` | 4.250000 | absmaxではなくp95をlocal-scale目標にする診断 |
| `local7_no_extra` | 4.218750 | 7bit local-scaleだけの診断 |
| `local6_no_extra` | 4.187500 | 6bit local-scaleだけの診断 |

## 結果

| 方式 | 平均relative MSE | UD比 | UDに勝った行数 |
|---|---:|---:|---:|
| UD IQ4_XS reference | 0.005891522357 | 1.000000 | - |
| `dual_codebook_l7_sel1` | 0.005562838514 | 0.944211 | 2/2 |
| `outlier1_s16_l7_i7` | 0.005848631029 | 0.992720 | 2/2 |
| `outlier1_s8_l6_i8` | 0.005918562321 | 1.004590 | 0/2 |
| `sb_scale4_l7_s4` | 0.005940293842 | 1.008278 | 0/2 |
| `sb_scale8_l7_s8` | 0.005952852801 | 1.010410 | 0/2 |
| `baseline_ue3m5_g32` | 0.005972981358 | 1.013826 | 0/2 |
| `sb_index_bias_l7_s8` | 0.005972981358 | 1.013826 | 0/2 |
| `local7_no_extra` | 0.005982546657 | 1.015450 | 0/2 |
| `local6_no_extra` | 0.006171810687 | 1.047575 | 0/2 |
| `clipped_p95_l8` | 0.016078342910 | 2.729064 | 0/2 |

## 判断

- 同じ `4.25 bpp` 枠でUD IQ4_XSを超える候補は `dual_codebook_l7_sel1` と `outlier1_s16_l7_i7`。
- `dual_codebook_l7_sel1` は2 tensor平均でUDより約 `5.58%` relative MSEが低く、現時点で最も有望。
- `outlier1_s16_l7_i7` はUDより約 `0.73%` 低いだけなので、単独の主方式にするには弱いが、実装コストやmetadata構造次第では補助候補になる。
- `superblock補正` は今回の単純なscale補正ではUDに届かなかった。index base圧縮はbaselineと同じ誤差になったため、精度改善というよりmetadata圧縮の診断だった。
- p95 clippingは大きく悪化した。IQ4置換では外れ値を捨てる方向ではなく、codebook分割または残差補正で扱う方向がいいと考える。
- 注意点として、これは `mlp_gate` と `mlp_up` の2 tensor、各262144 raw-valuesサンプルに対するBF16-error比較であり、全tensor・activation-weighted・実推論品質の結論ではない。

## 次の行動

- `dual_codebook_l7_sel1` をより多いIQ4_XS tensorに広げ、familyごとの安定性を確認する。
- codebook-selectorの保存コストを、codebook-scope単位のcodebook追加コスト込みで見積もる。
- `outlier1_s16_l7_i7` はselector方式と組み合わせる価値があるかを、同bpp制約の中で別途調べる。
