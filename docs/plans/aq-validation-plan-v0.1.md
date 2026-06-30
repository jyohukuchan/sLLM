# aq validation plan v0.1

## Purpose

この文書は、`aq` の初期検証を始めるための実験計画である。

現時点では `aq` の仕様を固定しない。固定するのは、保存値を 4bit index として扱うこと、16 値 LUT/codebook を使うこと、scale と codebook の候補を実験軸として記録すること、採用判断に使う評価手順である。

## Current Assumptions

- weight 本体は、scale などの補助情報を除き、4bit index として保存する。
- 各 index は 16 種類の LUT/codebook 値へ対応する。
- LUT/codebook の値は、候補ごとに FP8、FP16、BF16 相当へ割り当てられる。
- 実効値は概ね `dequant(x) = tensor_scale * family_scale * group_scale * codebook[index]` として扱う。ただし `tensor_scale` と `family_scale` は候補ごとに無効化できる。
- scale は arbitrary 8bit LUT にしない。実行時に毎回 arbitrary LUT lookup が必要になる形は、初期候補から外す。
- まずは BF16 weight との差が小さいことを目標にする。activation-aware weighting や perplexity は次段階で加える。
- 初期検証は Python/PyTorch で tensor-level simulation を行い、上位候補だけ HIP C++ kernel へ進める。

## External Reference Points

- OCP MX 系は、共有 scale を持つ block 形式で、MXFP4 では 32 要素 block と E8M0 scale が代表的な参照点になる。
- NVIDIA NVFP4 系は、16 要素 micro-block、E4M3 scale、tensor-level scale を参照点にする。
- uLLM `aq` はこれらと互換である必要はない。目的は、R9700/V620/MI300X/CPU で扱いやすく、同一 bpp 帯で高精度な保存形式を見つけることである。

References:

- OCP MX specification landing page: <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>
- NVIDIA NVFP4 inference blog: <https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/>

## Experiment Axes

### 1. Scale format

Scale は 8bit 保存を前提に、次の候補から始める。

| ID | Format | Runtime interpretation | Strength | Risk |
| --- | --- | --- | --- | --- |
| `scale_e8m0` | E8M0 | power-of-two scale, exponent adjust / shift-like dequant | 最も単純で速い。MX 系との比較軸になる。 | scale が粗く、group 内の値分布によっては誤差が大きい。 |
| `scale_e5m2` | FP8 E5M2 | FP8 scale, non-power-of-two | E4M3 より range が広い。 | mantissa が少なく、scale 精度は E4M3 より粗い。 |
| `scale_ue5m3` | unsigned E5M3 | unsigned 8bit scale expanded to FP16/BF16 | sign bit を scale 精度へ回せる。FP16/BF16 dequant へ寄せやすい。 | 標準 hardware format ではないため、bit decode 実装と互換性を検証する必要がある。 |
| `scale_e4m3` | FP8 E4M3 | FP8 scale, non-power-of-two | scale 精度が高い。NVFP4 系との比較軸になる。 | range が狭いため、tensor/family scale が必要になる可能性が高い。 |

初期方針:

- `scale_e8m0` は tensor/family scale なしでも試す。
- `scale_e4m3` と `scale_ue5m3` は tensor scale ありを基本候補にする。
- scale の decode は、実行 kernel では bit 操作または hardware FP8 conversion で処理できる形に限定する。

### 2. Group size

Group size は、1 個の scale を共有する weight 数である。

| Group size | Scale overhead | Effective bpp without tensor/family overhead | Use |
| ---: | ---: | ---: | --- |
| 8 | 1.000 bpp | 5.000 bpp | 精度上限の確認用。初期の主要候補にはしない。 |
| 16 | 0.500 bpp | 4.500 bpp | NVFP4 参照点。局所分布に強い。 |
| 32 | 0.250 bpp | 4.250 bpp | MXFP4 参照点。最初の主候補。 |
| 64 | 0.125 bpp | 4.125 bpp | overhead 削減候補。精度低下の確認用。 |
| 128 | 0.0625 bpp | 4.0625 bpp | かなり粗い。大きい tensor の一部だけで探索する。 |

初期方針:

- Round 1 は group size `16` と `32` を中心にする。
- `64` は scale overhead を減らしたい候補として追加する。
- `8` と `128` は上下限を見るための sampling-only 候補にする。

### 3. Tensor and family scale

`tensor_scale` と `family_scale` は、block scale の range と精度を補うための second-level scale である。

| Candidate | Meaning | Initial priority |
| --- | --- | --- |
| `none` | group scale と codebook だけで表す | E8M0 baseline として必要 |
| `tensor_bf16` | tensor ごとに BF16/FP16 scale を 1 個持つ | E4M3/E5M2/UE5M3 で優先 |
| `family_bf16` | attention、MLP、embed、lm_head、MoE expert など family ごとに scale を持つ | tensor scale の次に検証 |
| `tensor_plus_family` | tensor と family の両方を持つ | 精度は上がる可能性があるが、採用は後回し |

初期 family 分類:

- `attn_q`
- `attn_k`
- `attn_v`
- `attn_o`
- `mlp_gate`
- `mlp_up`
- `mlp_down`
- `embed`
- `lm_head`
- `moe_router`
- `moe_expert`

Qwen3-14B ではまず dense transformer 部分を対象にし、MoE family は Qwen3-30B-A3B 以降で追加する。

### 4. Codebook/LUT constraints

16 値 codebook は自由に割り当てられるが、完全自由だけで始めると探索空間が広すぎる。初期候補は制約を切り替える。

| Codebook mode | Description | Why |
| --- | --- | --- |
| `free16` | 16 値を完全自由に学習する | 上限精度を見る |
| `zero_free15` | 0 を必ず含め、残り 15 値を自由にする | exact zero を確保する |
| `symmetric7` | 0、正 7 値、負 7 値、予備 1 値 | signed weight に自然で、探索が安定しやすい |
| `positive_unsigned` | 非負値だけを持つ | scale や特殊 tensor 用。weight 本体では基本候補にしない |

初期方針:

- Round 1 は `zero_free15` と `symmetric7` を中心にする。
- `free16` は精度上限確認として sampling-only で回す。
- codebook は tensor ごと、または family ごとに持つ。全 model 共有 codebook は精度が足りない可能性が高いため、baseline 扱いにする。

## Initial Candidate Matrix

Round 1 では候補を増やしすぎない。

| Candidate ID | Scale | Group | Tensor scale | Family scale | Codebook | Purpose |
| --- | --- | ---: | --- | --- | --- | --- |
| `aq4_e8m0_g32_zf15` | E8M0 | 32 | no | no | zero_free15 / per-family | MX-like baseline |
| `aq4_e8m0_g16_zf15` | E8M0 | 16 | no | no | zero_free15 / per-family | group size effect |
| `aq4_e4m3_g16_ts_zf15` | E4M3 | 16 | BF16 | no | zero_free15 / per-family | NV-like local scale |
| `aq4_e4m3_g32_ts_zf15` | E4M3 | 32 | BF16 | no | zero_free15 / per-family | scale precision vs overhead |
| `aq4_e5m2_g32_ts_zf15` | E5M2 | 32 | BF16 | no | zero_free15 / per-family | wider FP8 scale range |
| `aq4_ue5m3_g32_ts_zf15` | unsigned E5M3 | 32 | BF16 | no | zero_free15 / per-family | FP16/BF16-friendly custom scale |
| `aq4_e8m0_g64_sym7` | E8M0 | 64 | no | no | symmetric7 / per-family | low-overhead baseline |

Round 2 では、Round 1 上位に対して次を追加する。

- tensor scale と family scale の有無。
- codebook を per-tensor にするか per-family にするか。
- group layout を contiguous K 方向、N 方向、tile 方向で変える。
- activation-aware weighting を入れる。

## Quantization Objective

最初の目的関数は BF16 weight との weighted MSE とする。

For each group:

```text
minimize sum_j weight_j * (x_j - tensor_scale * family_scale * group_scale * codebook[index_j])^2
```

Round 1 では `weight_j = 1` とする。Round 2 以降で、calibration activation、Hessian diagonal 近似、layer output sensitivity から `weight_j` を入れる。

### Group-level search

1. `tensor_scale` と `family_scale` を仮固定する。
2. 対象 group の `x_j` を正規化する。
3. 候補 scale code を列挙する。
   - 全 256 code を列挙してもよい。
   - 速度が足りない場合は `amax / max_abs(codebook)` 近傍の scale code だけを試す。
4. 各 scale code について、各 `x_j` に最も近い codebook entry を割り当てる。
5. group error が最小の `(scale_code, indices)` を採用する。

### Codebook optimization

初期は次の coordinate descent で進める。

1. tensor/family ごとに sample を取る。
2. 正規化値に対して k-means または quantile 初期化で 16 値 codebook を作る。
3. group-level search で scale と indices を更新する。
4. 固定された scale/indices に対して codebook 値を least squares で更新する。
5. 3-4 を数回繰り返す。
6. codebook 値を保存 target 型に丸めて、再度 error を測る。

注意:

- codebook に exact zero を含める候補では、zero entry は固定する。
- `symmetric7` では正側 7 値だけを更新し、負側は反転で作る。
- outlier を codebook で吸いすぎると小さい値の精度が落ちるため、outlier handling は別候補として扱う。

## Evaluation Metrics

Tensor-level:

- MSE
- weighted MSE
- relative MSE
- max absolute error
- cosine similarity
- saturation rate
- zero preservation rate
- effective bpp

Layer-level:

- linear output MSE against BF16
- output cosine similarity
- top-k activation error if activation samples are available

Model-level:

- perplexity on a small fixed calibration/eval set
- short generation smoke output stability
- later: MMLU-style small subset, coding subset, needle subset

Runtime-level:

- load time
- dequant throughput
- GEMM replay throughput
- VRAM footprint
- packed payload bandwidth
- candidate-specific decode/prefill speed once runtime exists

## First Implementation Steps

1. Add an `aq` experiment result schema.
2. Add a Python tensor sampler that streams safetensors tensors without loading the whole model at once.
3. Implement scale encode/decode for:
   - E8M0
   - E5M2
   - unsigned E5M3
   - E4M3
4. Implement codebook initialization:
   - `zero_free15`
   - `symmetric7`
5. Implement group-level exhaustive scale search for sampled tensors.
6. Run Round 1 on a small tensor subset:
   - `attn_q`
   - `attn_o`
   - `mlp_gate`
   - `mlp_up`
   - `mlp_down`
7. Record result JSONL under `benchmarks/results/YYYY-MM-DD/aq/`.
8. Promote the top 2-3 candidates to full tensor simulation.
9. Add layer replay after tensor-level candidates are narrowed.
10. Only after this, decide whether a HIP dequant prototype is worth writing for the candidate.

The first helper, `tools/run-aq-tensor-sample.py`, is a tensor-sample smoke tool. It initializes codebooks per tensor sample and records `codebook.granularity=per_tensor_sample`. Family-level codebook aggregation is a follow-up step, not part of the first smoke path.

## Initial Stop Conditions

Stop a candidate early if:

- effective bpp exceeds the comparison band without a clear accuracy win.
- saturation rate is high in multiple tensor families.
- codebook optimization is unstable across random samples.
- scale decode would require arbitrary LUT lookup in the hot path.
- group layout is incompatible with practical HIP/CPU vectorized loads.

## Open Questions

- Should `aq` codebook be per-family by default, or per-tensor despite metadata overhead?
- Should exact zero be mandatory for all weight tensors?
- Should group layout follow tensor logical K dimension, hardware GEMM tile layout, or converter-friendly contiguous storage first?
- Should tensor scale be BF16, FP16, or FP32 in the stored candidate?
- Should outliers be represented by a separate side table, or forced into the 16-value codebook?
- How much activation-aware weighting is needed before perplexity starts matching tensor-level MSE rankings?
