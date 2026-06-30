# aq validation results 2026-07-01

## Scope

This note records the first tensor-level aq validation pass for Qwen/Qwen3.5-9B.

Reference model:

- `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- dtype: BF16 safetensors

External quantized baselines:

- ModelOpt NVFP4: `/home/homelab1/datapool/ai_models/safetensors/hf/AxionML-Qwen3.5-9B-NVFP4/model.safetensors`
- Unsloth Dynamic GGUF: `/home/homelab1/datapool/ai_models/gguf/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-UD-Q4_K_XL.gguf`

All aq runs record CPU thread settings. The current WRX80 default is `--torch-threads 64` and `--torch-interop-threads 1`.

## Measurement Tools

- `tools/compare-quantized-weight-error.py`
  - Compares ModelOpt-style NVFP4 safetensors against BF16 reference tensors.
  - Decodes packed E2M1 values with E4M3 group scales and FP32 tensor scale.
  - Samples aligned 16-value groups without materializing full dequantized tensors.

- `tools/compare-gguf-weight-error.py`
  - Compares GGUF tensors against BF16 reference tensors.
  - Imports llama.cpp `gguf-py` from `reference-src/llama.cpp/gguf-py`.
  - Dequantizes one GGUF tensor at a time to avoid retaining multiple large arrays.
  - Excludes `embed` and `lm_head` by default to avoid multi-GB dequantization.

- `tools/run-aq-tensor-sample.py`
  - Samples BF16 reference tensors and simulates aq candidates.
  - Supports family-balanced tensor selection with `--max-tensors-per-family`.
  - Supports sample-local and family-level LUT construction.
  - Supports optional `--activation-stats` for activation-weighted metrics.

- `tools/collect-activation-stats.py`
  - Collects per-module input second moments for activation-weighted aq evaluation.
  - Stores reductions only, not raw activations.

- `tools/run-aq-weighted-sample.py`
  - Thin entry point over `tools/run-aq-tensor-sample.py`.
  - Intended for runs that pass `--activation-stats`.

## External Baselines

### ModelOpt NVFP4

Result file:

- `benchmarks/results/2026-07-01/aq/2026-07-01-nvfp4-error-qwen35-9b.jsonl`

24 tensor-group samples across MLP, linear attention, and full attention:

| metric | value |
| --- | ---: |
| relative MSE mean | 0.008996 |
| relative MSE min | 0.008853 |
| relative MSE max | 0.009099 |
| cosine similarity mean | 0.995502 |
| mean abs error mean | 0.000956 |

NVFP4 is a useful first target for aq because it is a simple 4.5 bpp reference point: 4-bit E2M1 values, one E4M3 scale per 16 values, and one FP32 tensor scale.

### Unsloth Dynamic Q4_K_XL GGUF

Result file used for all-family summary:

- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-reordered.jsonl`

36 tensor samples across MLP, Qwen3.5 linear attention, and full-attention projection tensors. The comparison applies llama.cpp's Qwen3.5 V-head reorder to the HF reference for linear-attention tensors.

| metric | value |
| --- | ---: |
| relative MSE mean | 0.002857 |
| relative MSE min | 0.000030 |
| relative MSE max | 0.005902 |
| cosine similarity mean | 0.998570 |
| mean abs error mean | 0.000512 |

Breakdown by GGML type:

| type | mean bpp | tensors | relative MSE mean |
| --- | ---: | ---: | ---: |
| IQ4_XS | 4.25 | 2 | 0.005901 |
| Q4_K | 4.50 | 14 | 0.005324 |
| Q5_K | 5.50 | 11 | 0.001342 |
| Q6_K | 6.5625 | 5 | 0.000331 |
| Q8_0 | 8.50 | 4 | 0.000030 |

The earlier run without the Qwen3.5 V-head reorder is stored at:

- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b.jsonl`

Treat its `linear_attn_*` rows as invalid comparison rows. The issue was a reference-layout mismatch, not an Unsloth quality issue.

## aq Candidate Results

Balanced round2 result:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-round2-qwen35-9b-balanced.jsonl`

30 tensors, 3 per family, 10 families, 262144 sampled elements per tensor:

| candidate | effective bpp | relative MSE mean |
| --- | ---: | ---: |
| `aq4_e4m3_g16_ts_flloyd16` | 4.50 | 0.005255 |
| `aq4_e4m3_g16_ts_zlloyd15` | 4.50 | 0.005643 |
| `aq4_ue5m3_g16_ts_zlloyd15` | 4.50 | 0.005669 |
| `aq4_e4m3_g16_ts_free16` | 4.50 | 0.005757 |
| `aq4_e5m2_g16_ts_zlloyd15` | 4.50 | 0.006356 |
| `aq4_e4m3_g16_ts_zf15` | 4.50 | 0.006974 |
| `aq4_e4m3_g32_ts_zf15` | 4.25 | 0.008219 |
| `aq4_e8m0_g16_zlloyd15` | 4.50 | 0.009716 |
| `aq4_e8m0_g16_zf15` | 4.50 | 0.011559 |
| `aq4_e8m0_g32_zf15` | 4.25 | 0.012201 |

The current best aq candidate is:

```text
aq4_e4m3_g16_ts_flloyd16
```

Meaning:

- 4-bit value index
- 16 entries in a free codebook
- codebook initialized by quantiles and refined by Lloyd updates
- E4M3 group scale
- group size 16
- BF16 tensor scale

Widening scale search from `--scale-window 4` to `--scale-window 16` only improved relative MSE from `0.005255` to `0.005235`, so the main gains are from codebook and group layout rather than wider local scale search.

### Group Size Sweep

Result file:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-round3-qwen35-9b-group-sizes.jsonl`

All rows use `E4M3 + BF16 tensor scale + free Lloyd16 codebook`.

| group size | effective bpp | relative MSE mean |
| ---: | ---: | ---: |
| 64 | 4.125 | 0.008292 |
| 32 | 4.25 | 0.006873 |
| 16 | 4.50 | 0.005244 |
| 8 | 5.00 | 0.003573 |

At the same nominal 4.5 bpp, the current aq g16 free-Lloyd candidate slightly beats the sampled UD `Q4_K` rows (`0.00524` vs `0.00534`) and clearly beats the sampled NVFP4 rows (`0.00524` vs `0.00900`). Caveat: aq currently uses sample-local codebooks, so this is not yet a final storage-format result.

### Family-Level LUT Check

Result file:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-lut-qwen35-9b-balanced.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-lut-qwen35-9b-wide.jsonl`

The current top codebook mode was tested with one shared LUT per family instead of one sample-local LUT per tensor.

| candidate | effective bpp | sample-local relative MSE | family-LUT relative MSE |
| --- | ---: | ---: | ---: |
| `aq4_e4m3_g32_ts_flloyd16` | 4.25 | 0.006873 | 0.006873 |
| `aq4_e4m3_g16_ts_flloyd16` | 4.50 | 0.005244 | 0.005241 |
| `aq4_e4m3_g8_ts_flloyd16` | 5.00 | 0.003573 | 0.003573 |

With 3 tensors per family, the free-Lloyd codebook is stable enough that family-level sharing did not meaningfully hurt tensor reconstruction. This needs a larger tensor set before becoming a format decision, but it reduces the concern that sample-local codebooks are hiding a large penalty.

The wider 8-tensor/family check remained close:

| candidate | effective bpp | family-LUT relative MSE, 8 tensors/family |
| --- | ---: | ---: |
| `aq4_e4m3_g32_ts_flloyd16` | 4.25 | 0.006922 |
| `aq4_e4m3_g16_ts_flloyd16` | 4.50 | 0.005268 |
| `aq4_e4m3_g8_ts_flloyd16` | 5.00 | 0.003588 |

This makes per-family LUTs a plausible first storage target. The remaining risk is layer-depth and activation sensitivity, not obvious tensor-distribution instability.

### Activation-Weighted Smoke

Result files:

- `benchmarks/results/2026-07-01/aq/activation-smoke-qwen35-9b/`
- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-smoke-qwen35-9b.jsonl`

One CPU smoke run collected activation statistics for:

```text
language_model.layers.0.mlp.down_proj
```

The weighted evaluator then ran `aq4_e4m3_g16_ts_flloyd16` on:

```text
model.language_model.layers.0.mlp.down_proj.weight
```

| metric | value |
| --- | ---: |
| samples | 1 |
| tokens | 15 |
| sampled elements | 16384 |
| unweighted relative MSE | 0.005158237 |
| weighted relative MSE | 0.004603734 |

This is only a tool smoke, not a quality conclusion. It verifies that
Transformers module names can be mapped to checkpoint tensor names and that the
weighted metric path works with real activation reductions.

### R9700 Activation-Weighted Comparison

Activation stats:

- `benchmarks/results/2026-07-01/aq/activation-r9700-smoke-qwen35-9b-s512/`
- environment: `build/envs/vllm-rocm-nightly`
- device selector: `ROCR_VISIBLE_DEVICES=1`
- device reported by PyTorch: `cuda:0`
- samples: 4 default prompts
- tokens seen: 1403
- modules with stats: 152

Result files:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-r9700-stats-qwen35-9b-balanced.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-search-r9700-stats-qwen35-9b-balanced.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-nvfp4-weighted-r9700-stats-qwen35-9b-family4.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-weighted-r9700-stats-qwen35-9b-family4.jsonl`

The comparison uses 4 tensors per family for:

```text
mlp_down, mlp_gate, mlp_up, linear_attn_out, attn_q, attn_k, attn_v, attn_o
```

| candidate / format | mean bpp | mean relative MSE | mean weighted relative MSE |
| --- | ---: | ---: | ---: |
| aq g16, unweighted scale search | 4.5000 | 0.005269024 | 0.008698592 |
| aq g16, weighted scale search | 4.5000 | 0.005972846 | 0.004922713 |
| aq g8, unweighted scale search | 5.0000 | 0.003647685 | 0.007701098 |
| aq g8, weighted scale search | 5.0000 | 0.004234023 | 0.003684397 |
| ModelOpt NVFP4 | 4.5000 | 0.008967095 | 0.010255294 |
| Unsloth Dynamic Q4_K_XL mixed | 5.4668 | 0.003607886 | 0.002460200 |

Weighted scale search is the first clear activation-aware improvement. For
g16, it worsened ordinary tensor MSE from `0.005269024` to `0.005972846`, but
improved weighted relative MSE from `0.008698592` to `0.004922713`. This is a
better trade-off for aq if activation-weighted error tracks model behavior.

The main outlier was `linear_attn_out`. With unweighted scale search, aq g16 had
`linear_attn_out` weighted relative MSE `0.027143953`; weighted scale search
reduced it to `0.011702844`. NVFP4 was `0.017121338` for the same family.

Unsloth Dynamic is still ahead on this weighted sample, but it is not an equal
bpp comparison: this 32-row subset averages `5.4668` bpp and stores
`linear_attn_out` as `Q8_0`, giving that family weighted relative MSE
`0.000249632`. This strongly suggests aq needs family-specific bit/scale policy
experiments, not only one uniform g16/g8 setting.

### R9700 Calib32 Stability Check

The 4-prompt calibration was expanded to a small 32-prompt calibration file:

- `benchmarks/calibration/qwen35-aq-smoke-prompts-v0.1.txt`
- stats output:
  - `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512/`
- samples: 32 prompts
- tokens seen: 14061
- modules with stats: 152

Result files:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-r9700-calib32-qwen35-9b-family4.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-search-r9700-calib32-qwen35-9b-family4.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-nvfp4-weighted-r9700-calib32-qwen35-9b-family4.jsonl`
- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-weighted-r9700-calib32-qwen35-9b-family4.jsonl`

| candidate / format | mean bpp | mean relative MSE | mean weighted relative MSE | `linear_attn_out` weighted relative MSE |
| --- | ---: | ---: | ---: | ---: |
| aq g16, unweighted scale search | 4.5000 | 0.005269024 | 0.007682577 | 0.018924633 |
| aq g16, weighted scale search | 4.5000 | 0.005900905 | 0.004622421 | 0.009085352 |
| aq g8, unweighted scale search | 5.0000 | 0.003647685 | 0.006697035 | 0.019346728 |
| aq g8, weighted scale search | 5.0000 | 0.004163366 | 0.003439578 | 0.007488695 |
| ModelOpt NVFP4 | 4.5000 | 0.008967095 | 0.009864150 | 0.013873237 |
| Unsloth Dynamic Q4_K_XL mixed | 5.4668 | 0.003607886 | 0.002471176 | 0.000153408 |

The direction remained stable after expanding calibration:

- weighted scale search improves aq weighted error substantially,
- aq g16 with weighted scale search beats NVFP4 at the same 4.5 bpp on this metric,
- aq g8 with weighted scale search closes part of the gap to Unsloth Dynamic,
- Unsloth Dynamic remains ahead because it uses mixed precision and protects
  sensitive families such as `linear_attn_out`.

### Weighted Codebook And Family Policy

Result file:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-codebook-r9700-calib32-qwen35-9b-family4.jsonl`
- policy summary:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-policy-r9700-calib32-qwen35-9b.json`

The next variant used both activation-weighted scale search and
activation-weighted Lloyd refinement for the family-level codebook.

| candidate | mean bpp | mean relative MSE | mean weighted relative MSE | `linear_attn_out` weighted relative MSE |
| --- | ---: | ---: | ---: | ---: |
| aq g16, weighted scale only | 4.5000 | 0.005900905 | 0.004622421 | 0.009085352 |
| aq g16, weighted scale + codebook | 4.5000 | 0.006031252 | 0.004038034 | 0.005163994 |
| aq g8, weighted scale only | 5.0000 | 0.004163366 | 0.003439578 | 0.007488695 |
| aq g8, weighted scale + codebook | 5.0000 | 0.004204903 | 0.002821072 | 0.003941077 |

Weighted Lloyd worsened ordinary tensor MSE slightly, but improved the
activation-weighted metric again. This is consistent with the goal of aq:
minimize the error that matters to layer outputs, not only raw tensor MSE.

Using the same rows, a simple family-policy simulation was computed by choosing
g16 or g8 per family. The following combined weighted relative MSE values use
sample-level weighted SSE/denominator reconstruction and parameter-weighted bpp.

| policy | parameter-weighted bpp | combined weighted relative MSE |
| --- | ---: | ---: |
| aq all g16, weighted scale + codebook | 4.500000 | 0.003798456 |
| aq g8 for `attn_k,attn_o,attn_v,linear_attn_out` | 4.592593 | 0.003053866 |
| aq g8 for `attn_k,attn_o,attn_q,attn_v,linear_attn_out` | 4.666667 | 0.002900270 |
| aq g8 except `mlp_down` | 4.888889 | 0.002673004 |
| aq all g8, weighted scale + codebook | 5.000000 | 0.002582475 |
| ModelOpt NVFP4 | 4.500001 | 0.008990352 |
| Unsloth Dynamic Q4_K_XL mixed | 5.206019 | 0.002364278 |

This does not prove model quality, but it changes the next aq direction:

- weighted codebook fitting should stay in the search loop,
- a uniform bpp policy is probably leaving accuracy on the table,
- family-specific g16/g8 allocation may approach UD-like weighted error with
  lower bpp than the sampled UD mix,
- model-level logit/perplexity checks are now needed before further tensor-only
  optimization.

### Module-Level Logit Smoke

Tool:

- `tools/run-aq-module-logit-smoke.py`

Result file:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-linear-attn-out-r9700-calib32-qwen35-9b.jsonl`
- 8-prompt follow-up:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-linear-attn-out-r9700-calib32-qwen35-9b-prompts8.jsonl`

Scope:

- model: Qwen3.5-9B CausalLM
- device: R9700 through `build/envs/vllm-rocm-nightly`
- module quantized: `model.layers.0.linear_attn.out_proj`
- prompt count: 1
- sequence length cap: 64
- comparison: final-token logits against BF16 reference

| variant | logit relative MSE | mean abs error | max abs error | KL(ref, candidate) | top1 match | top10 overlap |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| g16 unweighted scale/codebook | 0.002045509 | 0.094623752 | 0.625000000 | 0.001206253 | true | 10 |
| g16 weighted scale + codebook | 0.000198949 | 0.028513012 | 0.203125000 | 0.000491175 | true | 10 |
| g8 weighted scale + codebook | 0.000101244 | 0.020070247 | 0.125000000 | 0.001576327 | true | 10 |

This is not a full-model quality result, but it confirms that the
activation-weighted variants also reduce logit error for the most suspicious
single module from the tensor analysis. The KL result is not strictly monotonic
with logit MSE in this one-prompt smoke, so the next check should use more
prompts and eventually full-model replacement.

The 8-prompt follow-up preserved the direction:

| variant | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| g16 unweighted scale/codebook | 0.002274514 | 0.084321837 | 0.005510745 | 8 / 8 | 9.75 |
| g16 weighted scale + codebook | 0.000214926 | 0.027823837 | 0.000705097 | 8 / 8 | 9.875 |
| g8 weighted scale + codebook | 0.000253724 | 0.030042848 | 0.000899909 | 8 / 8 | 10.0 |

On this small logit smoke, g16 weighted was slightly better than g8 weighted
despite g8 being better in tensor weighted MSE. Candidate ranking therefore
needs model-level checks, not only tensor metrics.

Two additional modules were checked with the same 8 prompts:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-extra-modules-r9700-calib32-qwen35-9b-prompts8.jsonl`

| module | variant | mean logit relative MSE | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | --- | ---: | ---: | ---: | ---: |
| `model.layers.0.mlp.up_proj` | g16 unweighted | 0.000218958 | 0.001802187 | 8 / 8 | 9.875 |
| `model.layers.0.mlp.up_proj` | g16 weighted | 0.000168724 | 0.001214926 | 8 / 8 | 9.75 |
| `model.layers.0.mlp.up_proj` | g8 weighted | 0.000162605 | 0.000821250 | 8 / 8 | 9.875 |
| `model.layers.3.self_attn.v_proj` | g16 unweighted | 0.000262743 | 0.001237670 | 8 / 8 | 9.875 |
| `model.layers.3.self_attn.v_proj` | g16 weighted | 0.000293307 | 0.001418085 | 8 / 8 | 9.75 |
| `model.layers.3.self_attn.v_proj` | g8 weighted | 0.000222151 | 0.001376848 | 8 / 8 | 9.75 |

This reinforces the need for family-specific policy and model-level checks:
weighted codebook/scale is helpful for some modules, but not uniformly better
for every family and metric.

A cumulative smoke then quantized three modules together:

- `model.layers.0.linear_attn.out_proj`
- `model.layers.0.mlp.up_proj`
- `model.layers.3.self_attn.v_proj`
- result:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-cumulative3-r9700-calib32-qwen35-9b-prompts8.jsonl`

| variant | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| g16 unweighted | 0.002544046 | 0.090560542 | 0.005718965 | 8 / 8 | 9.625 |
| g16 weighted | 0.000297915 | 0.032316454 | 0.001522995 | 8 / 8 | 9.875 |
| g8 weighted | 0.000249932 | 0.029874566 | 0.001281331 | 8 / 8 | 9.75 |

For this cumulative three-module smoke, g8 weighted was best by logit relative
MSE and KL. The result is still far from a full-model replacement, but it
supports carrying weighted g16/g8 policies into the next stage.

`tools/run-aq-module-logit-smoke.py` was then extended with mixed family policy
support:

- `--policy NAME=family1,family2` uses `g8_weighted` for the listed families
  and `g16_weighted` for the remaining selected modules.
- cumulative runs now keep original selected weights on CPU and refuse runs
  above `--max-original-weight-mib` to avoid accidental GPU/host memory spikes.
- policy rows include per-module family and selected variant metadata.

Two mixed-policy smokes were run with the same 8 prompts:

- layer0 policy result:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-layer0-policy-r9700-calib32-qwen35-9b-prompts8.jsonl`
- policy5 result:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-policy5-r9700-calib32-qwen35-9b-prompts8.jsonl`

`tools/select-aq-logit-smoke-modules.py` was added to select reproducible
module sets from activation stats. It reads `activation_second_moments` keys,
uses the existing aq family classifier, and can emit text, JSON, or shell
`--module` arguments.

Layer0 scope:

- `model.layers.0.linear_attn.out_proj`
- `model.layers.0.mlp.gate_proj`
- `model.layers.0.mlp.up_proj`
- `model.layers.0.mlp.down_proj`

| variant / policy | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all g16 weighted | 0.000299323 | 0.032173163 | 0.001534273 | 8 / 8 | 9.625 |
| all g8 weighted | 0.000198038 | 0.026597451 | 0.000739757 | 8 / 8 | 10.0 |
| p4p6: g8 for `linear_attn_out`; MLP g16 | 0.000302250 | 0.032795076 | 0.001551366 | 8 / 8 | 10.0 |
| p4p9: g8 for `linear_attn_out,mlp_gate,mlp_up`; `mlp_down` g16 | 0.000192349 | 0.026307318 | 0.001143397 | 8 / 8 | 10.0 |

Policy5 scope:

- `model.layers.0.linear_attn.out_proj`
- `model.layers.3.self_attn.k_proj`
- `model.layers.3.self_attn.v_proj`
- `model.layers.3.self_attn.o_proj`
- `model.layers.0.mlp.up_proj`

| variant / policy | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all g16 weighted | 0.000286738 | 0.031830961 | 0.001103859 | 8 / 8 | 9.75 |
| all g8 weighted | 0.000284312 | 0.031472139 | 0.001183611 | 8 / 8 | 9.75 |
| p4p6: g8 for `attn_k,attn_o,attn_v,linear_attn_out`; `mlp_up` g16 | 0.000225818 | 0.028638312 | 0.001248148 | 8 / 8 | 9.875 |
| p4p9: same as all g8 for this scope | 0.000284312 | 0.031472139 | 0.001183611 | 8 / 8 | 9.75 |

The policy5 result is a useful early signal for mixed precision: keeping
`mlp_up` at g16 while moving the attention-sensitive families to g8 reduced
logit relative MSE against both all-g16 and all-g8 in this small smoke. KL did
not improve, so the result should be treated as a candidate-ordering signal,
not a quality conclusion.

A broader policy10 smoke then selected 10 modules across layers 0, 3, and 7:

- selection:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-policy10.json`
- result:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-policy10-r9700-calib32-qwen35-9b-prompts8.jsonl`

Policy10 scope:

- `model.layers.0.linear_attn.out_proj`
- `model.layers.0.mlp.up_proj`
- `model.layers.3.mlp.up_proj`
- `model.layers.3.self_attn.k_proj`
- `model.layers.3.self_attn.o_proj`
- `model.layers.3.self_attn.v_proj`
- `model.layers.7.mlp.up_proj`
- `model.layers.7.self_attn.k_proj`
- `model.layers.7.self_attn.o_proj`
- `model.layers.7.self_attn.v_proj`

| variant / policy | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all g16 weighted | 0.000398490 | 0.038127334 | 0.001178040 | 8 / 8 | 9.875 |
| all g8 weighted | 0.000426076 | 0.037648361 | 0.001987679 | 8 / 8 | 9.75 |
| p4p6: attention-sensitive families g8; `mlp_up` g16 | 0.000369140 | 0.036939442 | 0.001034530 | 8 / 8 | 9.875 |
| p4p9: same as all g8 for this scope | 0.000426076 | 0.037648361 | 0.001987679 | 8 / 8 | 9.75 |

For policy10, p4p6 improved both logit relative MSE and KL over all-g16 and
all-g8. This supports treating `mlp_up` as a family that may not benefit from
spending g8 budget as early as attention-sensitive families.

## Interpretation

The current evidence supports continuing measurement and quantizer optimization together, not doing a long isolated quantizer-theory phase before measuring. The best gains so far came from trying concrete variants and measuring them quickly.

However, a dedicated quantization-tool optimization track is necessary before full-model conversion:

- Full quantization must be CPU-multithreaded and chunked.
- The current Python tool is acceptable for tensor sampling, but not for final full-model quantization.
- Family-level or tensor-level LUT aggregation must be tested because sample-local codebooks are too optimistic for final format decisions.
- Zero-preserving versus free16 codebooks must be evaluated with model-level quality, not only MSE.
- GGUF linear-attention comparison must apply the Qwen3.5 V-head reorder used by llama.cpp conversion.

## Next Actions

1. Add activation-stat collection for selected Qwen3.5-9B linear modules.
2. Expand calibration with longer contexts or an external text set after the current 32-prompt smoke.
3. Use the family-policy summary to choose candidates for model-level checks.
4. Expand the module-level logit smoke to more prompts/modules and then full-model replacement.
5. Extend `ullm-quant` from skeleton to safetensors metadata planning and then chunked CPU quantization.
