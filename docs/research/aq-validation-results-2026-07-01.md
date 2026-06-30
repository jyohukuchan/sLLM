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

- `tools/verify-aq-one-tensor.py`
  - Chunked Python reference for one full tensor with an exported family
    codebook.
  - Uses bounded group chunks for scale-window search instead of expanding the
    full tensor into one large distance matrix.
  - Used to cross-check Rust `ullm-quant` dry-run metrics.

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

`ullm-quant` was then connected to the p4p6 policy at the planning level:

- CLI options added:
  - `--aq-policy all-g16|all-g8|p4p6|p4p9|p4p46_inproj|p4p65_inproj|custom`
  - `--aq-high-family FAMILY` for custom policies
  - `--aq-low-format` / `--aq-high-format`
- plan schema version: `ullm-quant-plan-v0.3`
- p4p6 plan:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p6.json`

| plan item | count | input bytes | estimated output bytes |
| --- | ---: | ---: | ---: |
| quantize low / `aq4_e4m3_g16_ts_flloyd16` | 204 | 12,998,148,096 | 3,655,729,152 |
| quantize high / `aq4_e4m3_g8_ts_flloyd16` | 51 | 1,258,291,200 | 393,216,000 |
| passthrough | 520 | 5,049,777,120 | 5,049,777,120 |
| total | 775 | 19,306,216,416 | 9,098,722,272 |

This does not quantize payloads yet, but it makes the candidate p4p6 policy
explicit in the full-model conversion plan. The estimated output size excludes
container metadata and shared codebook overhead, so it is a payload estimate,
not a final `.ullm` file size.

The same planner was run for the main policy presets:

- summary:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-policy-size-summary-qwen35-9b.json`

| policy | estimated output bytes | output/input ratio | delta vs all-g16 |
| --- | ---: | ---: | ---: |
| all-g16 | 9,059,400,672 | 0.469248 | 0 |
| p4p6 | 9,098,722,272 | 0.471285 | +39,321,600 |
| p4p9 | 9,325,214,688 | 0.483016 | +265,814,016 |
| all-g8 | 9,504,914,400 | 0.492324 | +445,513,728 |

Combined with the policy10 logit smoke, p4p6 is currently the most attractive
candidate because its estimated size is close to all-g16 while its logit
relative MSE and KL were better in the 10-module smoke.

After the in-projection activation-stat fix, `ullm-quant` was extended with
named policy presets for the strongest wider-smoke follow-up candidates:

- `p4p46_inproj`: high format for
  `attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_z`
- `p4p65_inproj`: high format for
  `attn_k,attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_qkv`
- plan outputs:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p46-inproj.json`
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-plan-qwen35-9b-p4p65-inproj.json`
- updated size summary:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-policy-size-summary-qwen35-9b-inproj.json`

| policy | high tensors | low tensors | estimated output bytes | output/input ratio | delta vs all-g16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all-g16 | 0 | 255 | 9,059,400,672 | 0.469248 | 0 |
| p4p6 | 51 | 204 | 9,098,722,272 | 0.471285 | +39,321,600 |
| p4p46_inproj | 114 | 141 | 9,121,922,016 | 0.472486 | +62,521,344 |
| p4p65_inproj | 123 | 132 | 9,149,447,136 | 0.473912 | +90,046,464 |
| p4p9 | 126 | 129 | 9,325,214,688 | 0.483016 | +265,814,016 |
| all-g8 | 255 | 0 | 9,504,914,400 | 0.492324 | +445,513,728 |

p4p46_inproj costs only `23,199,744` estimated bytes more than p4p6, while
p4p65_inproj costs `50,724,864` more than p4p6. This makes both candidates
small enough for the next full-policy prototype comparison.

### Rust `ullm-quant` Payload Dry-Run

`ullm-quant` can now load the exported family codebook JSON, stream a real
safetensors tensor payload in bounded chunks, choose direct E4M3 group scales,
assign each value to the nearest 4-bit codebook entry, and accumulate
reconstruction metrics without writing a converted output file.

Dry-run result files:

- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer0-mlp-up-g16.txt`
- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer3-attn-k-g8.txt`

| tensor | candidate | elements | groups | relative MSE | max abs error |
| --- | --- | ---: | ---: | ---: | ---: |
| `model.language_model.layers.0.mlp.up_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | 50,331,648 | 3,145,728 | 0.006231116836 | 0.006380409 |
| `model.language_model.layers.3.self_attn.k_proj.weight` | `aq4_e4m3_g8_ts_flloyd16` | 4,194,304 | 524,288 | 0.004610619768 | 0.012256019 |

The first dry-run used direct nearest group scales and did not apply the
candidate tensor scale.

`ullm-quant` was then extended to estimate tensor scale for `_ts_` candidates
and scan nearby scale values per group with `--scale-window 4`.

Scale-window result files:

- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`
- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-dry-run-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`

| tensor | candidate | tensor scale | relative MSE | max abs error | groups changed by window |
| --- | --- | ---: | ---: | ---: | ---: |
| `model.language_model.layers.0.mlp.up_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | 0.014789051376 | 0.005283509762 | 0.005970601 | 1,612,071 |
| `model.language_model.layers.3.self_attn.k_proj.weight` | `aq4_e4m3_g8_ts_flloyd16` | 0.018260609359 | 0.003677692937 | 0.012858063 | 259,635 |

Python reference check files:

- `benchmarks/results/2026-07-01/aq/2026-07-01-python-verify-qwen35-9b-layer0-mlp-up-g16-scale-window4.json`
- `benchmarks/results/2026-07-01/aq/2026-07-01-python-verify-qwen35-9b-layer3-attn-k-g8-scale-window4.json`

The Python reference matched Rust for `attn_k` down to the index-count vector.
For `mlp_up`, relative MSE matched within roughly `1.3e-10`; a few index counts
and improved-group counts differ by 1-3 due to tensor-scale rounding and exact
tie behavior. This is acceptable for validating the Rust chunk path.

### Prototype `.ullm.d` Tensor Output

`ullm-quant` can now write a temporary directory-form prototype for one
inspected tensor. This is not the final `.ullm` container, but it is enough to
test packed index bytes, scale indices, codebook storage, manifest metadata, and
re-read/dequant verification.

Prototype output:

- directory:
  - `benchmarks/results/2026-07-01/aq/prototype-qwen35-9b-layer3-attn-k-g8-scale-window4.ullm.d/`
- run log:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`

Files:

| file | bytes | note |
| --- | ---: | --- |
| `manifest.json` | 1,844 | one tensor manifest |
| `codebooks/attn_k__aq4_e4m3_g8_ts_flloyd16.f32` | 64 | 16 little-endian F32 entries |
| `tensors/model_language_model_layers_3_self_attn_k_proj_weight.idx4` | 2,097,152 | two 4-bit indices per byte |
| `tensors/model_language_model_layers_3_self_attn_k_proj_weight.scale_u8` | 524,288 | one scale-table index per group |

Verification:

| item | value |
| --- | ---: |
| tensor | `model.language_model.layers.3.self_attn.k_proj.weight` |
| elements | 4,194,304 |
| groups | 524,288 |
| relative MSE | 0.003677692937 |
| max abs error | 0.012858063 |
| re-read/dequant relative MSE | 0.003677692937 |
| elapsed wall time | 1.71 s |
| maximum RSS | 8,232 KiB |

The re-read/dequant path reads `manifest.json`, binary codebook, packed idx4
file, and scale-index file, reconstructs values from the original safetensors
payload, and fails if relative MSE differs from the in-flight manifest metric by
more than `1e-9`.

For a larger write-only benchmark, `ullm-quant` was run with `--skip-inspect`
and `--prototype-skip-verify` so that the timing covers tensor-scale estimation
plus prototype quantize/write, without duplicate inspection or re-read verify.
The binary output was written under `/tmp` and only the run log was retained in
the repository.

Run log:

- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-benchmark-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`

| item | value |
| --- | ---: |
| tensor | `model.language_model.layers.0.mlp.up_proj.weight` |
| elements | 50,331,648 |
| groups | 3,145,728 |
| relative MSE | 0.005283509762 |
| idx4 bytes | 25,165,824 |
| scale bytes | 3,145,728 |
| elapsed wall time | 8.76 s |
| maximum RSS | 21,560 KiB |
| elements/s | 5,745,622 |

This is still scalar Rust prototype code and reads the source tensor twice
because tensor scale is estimated before quantization. It is useful as a
correctness and memory baseline, not as the intended final CPU throughput.

The chunk hot loop was then moved behind a C++20 BF16 kernel for:

- best-scale search,
- nearest-codebook assignment,
- idx4 packing,
- scale-index output,
- metric accumulation.

The Rust side still owns metadata, safetensors reads, tensor-scale estimation,
manifest writing, and verification. The C++ kernel currently requires a
16-entry codebook and supports BF16/F16 input. Rust calls it through
`ullm_aq_quantize_chunk_v1`, whose request struct includes a `struct_size`
field and an explicit dtype id; unsupported dtype ids currently return an
unsupported status.

Run logs:

- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-write-benchmark-cxx-qwen35-9b-layer0-mlp-up-g16-scale-window4.txt`
- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-quant-prototype-cxx-verify-qwen35-9b-layer3-attn-k-g8-scale-window4.txt`

| run | tensor | relative MSE | elapsed | max RSS | elements/s |
| --- | --- | ---: | ---: | ---: | ---: |
| scalar Rust write-only | `mlp_up` g16 | 0.005283509762 | 8.76 s | 21,560 KiB | 5,745,622 |
| C++ BF16 write-only | `mlp_up` g16 | 0.005283509762 | 7.13 s | 21,516 KiB | 7,059,137 |
| C++ BF16 one-pass with tensor-scale override | `mlp_up` g16 | 0.005283509762 | 6.99 s | 4,180 KiB | 7,200,522 |
| C++ BF16 write + verify | `attn_k` g8 | 0.003677692937 | 0.74 s | 8,220 KiB | n/a |

The first C++ kernel is only a scalar baseline, but it preserved metrics and
improved the large-tensor write path by about `1.23x`. The next optimization
target is not only SIMD: the prototype still reads the tensor twice because
tensor-scale estimation is a pre-pass.

`--tensor-scale-override` was added to isolate one-pass quantize/write speed
when a correct tensor scale is already known. For `mlp_up`, this reduced wall
time only slightly (`7.13 s -> 6.99 s`) but reduced peak RSS substantially
(`21,516 KiB -> 4,180 KiB`) by avoiding the group-target-scale vector used for
exact median tensor-scale estimation.

### Multi-Tensor Prototype Policy Smoke

`tools/run-ullm-prototype-policy-smoke.py` was added as a small driver around
`ullm-quant`. It reads a plan JSON, selects tensors whose family/candidate has
an exported codebook, runs one prototype output directory per tensor, and stores
only summary/log files in the repository. Binary prototype directories were
written under `/tmp`.

Result files:

- summary:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k.json`
- logs:
  - `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k-logs/`

Scope:

- plan: p4p6
- families: `mlp_up`, `attn_k`
- max tensors: 4
- per family: 2
- verification: skipped for speed

| family | tensor | candidate | relative MSE | elapsed | max RSS |
| --- | --- | --- | ---: | ---: | ---: |
| `mlp_up` | `model.language_model.layers.0.mlp.up_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | 0.005283509762 | 0:07.40 | 19,476 KiB |
| `mlp_up` | `model.language_model.layers.1.mlp.up_proj.weight` | `aq4_e4m3_g16_ts_flloyd16` | 0.005288028063 | 0:07.79 | 21,544 KiB |
| `attn_k` | `model.language_model.layers.11.self_attn.k_proj.weight` | `aq4_e4m3_g8_ts_flloyd16` | 0.003723889112 | 0:00.73 | 8,232 KiB |
| `attn_k` | `model.language_model.layers.15.self_attn.k_proj.weight` | `aq4_e4m3_g8_ts_flloyd16` | 0.003702330162 | 0:00.73 | 7,208 KiB |

This is still not a single multi-tensor `.ullm.d` container, but it verifies
that the p4p6 plan, exported codebooks, and C++ chunk kernel can be driven
across multiple real tensors without manual command construction.

`tools/merge-ullm-prototype-dirs.py` was then added to merge per-tensor
prototype directories into one directory with a shared manifest and de-duplicated
codebook files.

Merge summary:

- `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-mlp-up-attn-k.json`

Merged output was written under `/tmp`:

- `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-mlp-up-attn-k-merged.ullm.d`

| item | value |
| --- | ---: |
| tensor count | 4 |
| shared codebook count | 2 |
| manifest bytes | 6,496 |
| total file bytes | 61,872,608 |

This is the first single-directory multi-tensor prototype. It still lacks
passthrough tensors and full-model metadata, but it establishes the merge shape:
one manifest, one `tensors/` directory, and shared `codebooks/`.

`ullm-quant` can also verify an existing prototype directory from its manifest:

- command mode:
  - `--verify-prototype-dir`
  - `--verify-prototype-all`
- verify log:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-mlp-up-attn-k.txt`

The merged 4-tensor prototype verified successfully:

| item | value |
| --- | ---: |
| verified tensors | 4 |
| elapsed wall time | 0.74 s |
| maximum RSS | 29,764 KiB |

Verified relative MSE values matched the prototype write metrics:

- `mlp_up` layer0: `0.005283509762`
- `mlp_up` layer1: `0.005288028063`
- `attn_k` layer11: `0.003723889112`
- `attn_k` layer15: `0.003702330162`

### Full-Family Prototype Policy Smoke

The first p4p6 prototype smoke covered only `mlp_up` and `attn_k`. A wider
export was then generated for all p4p6 quantized families:

- `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted.json`
- log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted.log`

The export contains 24 codebooks: 12 families times 2 candidates. Activation
stats currently exist for `mlp`, dense self-attention, and `linear_attn.out_proj`.
They do not exist for `linear_attn.in_proj_qkv`, `in_proj_a`, `in_proj_b`, or
`in_proj_z`, because the current activation collector recorded only
`linear_attn.out_proj` for those modules. The export therefore uses weighted
codebooks where stats exist and records an explicit
`unweighted_missing_activation_stats` fallback for the linear-attention in-proj
families.

Export resource use:

| item | value |
| --- | ---: |
| codebooks | 24 |
| weighted codebooks | 16 |
| fallback unweighted codebooks | 8 |
| elapsed wall time | 11.31 s |
| maximum RSS | 617,952 KiB |

Using this codebook set, the p4p6 prototype smoke was expanded to one tensor per
quantized family:

- summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families.json`
- driver log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families-driver.log`
- per-tensor logs:
  `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-all-families-logs/`

All 12 selected tensors returned success and passed per-tensor verification.

| family | candidate | relative MSE | elapsed | max RSS |
| --- | --- | ---: | ---: | ---: |
| `linear_attn_a` | `aq4_e4m3_g16_ts_flloyd16` | 0.005253958207 | 0:00.03 | 3,112 KiB |
| `linear_attn_b` | `aq4_e4m3_g16_ts_flloyd16` | 0.005458763018 | 0:00.02 | 3,112 KiB |
| `linear_attn_qkv` | `aq4_e4m3_g16_ts_flloyd16` | 0.005195938521 | 0:05.02 | 20,916 KiB |
| `linear_attn_z` | `aq4_e4m3_g16_ts_flloyd16` | 0.005203894074 | 0:02.85 | 13,108 KiB |
| `linear_attn_out` | `aq4_e4m3_g8_ts_flloyd16` | 0.003765302590 | 0:02.63 | 15,444 KiB |
| `mlp_down` | `aq4_e4m3_g16_ts_flloyd16` | 0.005318504789 | 0:07.75 | 30,248 KiB |
| `mlp_gate` | `aq4_e4m3_g16_ts_flloyd16` | 0.005198360634 | 0:08.05 | 30,536 KiB |
| `mlp_up` | `aq4_e4m3_g16_ts_flloyd16` | 0.005245190541 | 0:07.20 | 31,148 KiB |
| `attn_k` | `aq4_e4m3_g8_ts_flloyd16` | 0.003724312490 | 0:00.66 | 7,204 KiB |
| `attn_o` | `aq4_e4m3_g8_ts_flloyd16` | 0.003642895769 | 0:02.54 | 16,424 KiB |
| `attn_q` | `aq4_e4m3_g16_ts_flloyd16` | 0.005336517833 | 0:05.15 | 21,188 KiB |
| `attn_v` | `aq4_e4m3_g8_ts_flloyd16` | 0.003817673166 | 0:00.73 | 7,244 KiB |

The 12 per-family prototype directories were then merged:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-all-families.json`
- output:
  `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-all-families-merged.ullm.d`

| item | value |
| --- | ---: |
| tensor count | 12 |
| codebook count | 12 |
| total file bytes | 158,503,771 |

Merged verification also passed:

- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-all-families.txt`

| item | value |
| --- | ---: |
| verified tensors | 12 |
| elapsed wall time | 2.16 s |
| maximum RSS | 101,196 KiB |

The same full-family smoke was then widened to two tensors per family:

- summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family2.json`
- driver log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family2-driver.log`
- per-tensor logs:
  `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-family2-logs/`

All 24 selected tensors returned success and passed per-tensor verification.

| family | tensors | relative MSE min | relative MSE max | max RSS |
| --- | ---: | ---: | ---: | ---: |
| `attn_k` | 2 | 0.003696580544 | 0.003724312490 | 8,276 KiB |
| `attn_o` | 2 | 0.003639662156 | 0.003642895769 | 16,424 KiB |
| `attn_q` | 2 | 0.005336517833 | 0.005339968314 | 22,252 KiB |
| `attn_v` | 2 | 0.003816023096 | 0.003817673166 | 8,252 KiB |
| `linear_attn_a` | 2 | 0.005253958207 | 0.005313893821 | 3,112 KiB |
| `linear_attn_b` | 2 | 0.005458763018 | 0.005540913549 | 3,156 KiB |
| `linear_attn_out` | 2 | 0.003732074646 | 0.003765302590 | 16,424 KiB |
| `linear_attn_qkv` | 2 | 0.005195938521 | 0.005203145169 | 21,612 KiB |
| `linear_attn_z` | 2 | 0.005192328385 | 0.005203894074 | 12,760 KiB |
| `mlp_down` | 2 | 0.005318504789 | 0.005344374105 | 29,564 KiB |
| `mlp_gate` | 2 | 0.005196248710 | 0.005198360634 | 30,852 KiB |
| `mlp_up` | 2 | 0.005245190541 | 0.005250488442 | 30,492 KiB |

Family2 driver resource use:

| item | value |
| --- | ---: |
| selected tensors | 24 |
| successful tensors | 24 |
| elapsed wall time | 1:27.16 |
| maximum RSS | 30,852 KiB |

The 24-tensor prototype was also merged and verified:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-family2.json`
- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-family2.txt`

| item | value |
| --- | ---: |
| tensor count | 24 |
| codebook count | 12 |
| total file bytes | 317,004,099 |
| verify elapsed wall time | 4.09 s |
| verify maximum RSS | 101,216 KiB |

The smoke was then widened again to four tensors per family, matching the
sampling width used for the full-family codebook export:

- summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family4.json`
- driver log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-family4-driver.log`
- per-tensor logs:
  `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-family4-logs/`

All 48 selected tensors returned success and passed per-tensor verification.

| family | tensors | relative MSE min | relative MSE max | max RSS |
| --- | ---: | ---: | ---: | ---: |
| `attn_k` | 4 | 0.003689322353 | 0.003779590027 | 8,232 KiB |
| `attn_o` | 4 | 0.003639662156 | 0.003662249925 | 16,464 KiB |
| `attn_q` | 4 | 0.005322125903 | 0.005344111781 | 22,184 KiB |
| `attn_v` | 4 | 0.003816023096 | 0.003853964074 | 8,232 KiB |
| `linear_attn_a` | 4 | 0.005253958207 | 0.005408427461 | 3,156 KiB |
| `linear_attn_b` | 4 | 0.005458763018 | 0.005741676939 | 3,156 KiB |
| `linear_attn_out` | 4 | 0.003732074646 | 0.003765302590 | 16,416 KiB |
| `linear_attn_qkv` | 4 | 0.005195938521 | 0.005247204745 | 22,952 KiB |
| `linear_attn_z` | 4 | 0.005192328385 | 0.005217610417 | 12,784 KiB |
| `mlp_down` | 4 | 0.005318504789 | 0.005344374105 | 31,456 KiB |
| `mlp_gate` | 4 | 0.005196248710 | 0.005224491836 | 32,076 KiB |
| `mlp_up` | 4 | 0.005245190541 | 0.005280311022 | 31,252 KiB |

Family4 driver resource use:

| item | value |
| --- | ---: |
| selected tensors | 48 |
| successful tensors | 48 |
| elapsed wall time | 2:45.31 |
| maximum RSS | 32,076 KiB |

The 48-tensor prototype was also merged and verified:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-family4.json`
- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-family4.txt`

| item | value |
| --- | ---: |
| tensor count | 48 |
| codebook count | 12 |
| total file bytes | 634,004,817 |
| verify elapsed wall time | 8.12 s |
| verify maximum RSS | 101,252 KiB |

Finally, all 255 p4p6 quantized tensors were converted. This includes 7 MTP
linear tensors whose names match the existing `mlp_*` or `self_attn.*_proj`
family rules. This run still excludes passthrough tensors, so it is a
quantized-weight payload prototype, not a full model package. Per-tensor re-read
verification was skipped during conversion to avoid duplicate verification work;
the merged output was verified afterwards.

- summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized.json`
- driver log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-driver.log`
- per-tensor logs:
  `benchmarks/results/2026-07-01/aq/prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-logs/`

All 255 selected tensors returned success.

| family | tensors | relative MSE min | relative MSE max | relative MSE mean | max RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `attn_k` | 9 | 0.003667359312 | 0.003927970222 | 0.003758196034 | 8,272 KiB |
| `attn_o` | 9 | 0.003639662156 | 0.003797789239 | 0.003681134613 | 16,460 KiB |
| `attn_q` | 9 | 0.005314884734 | 0.005368310821 | 0.005336698987 | 16,424 KiB |
| `attn_v` | 9 | 0.003797044088 | 0.003857622082 | 0.003828035419 | 8,276 KiB |
| `linear_attn_a` | 24 | 0.005253958207 | 0.005783048676 | 0.005468881683 | 3,156 KiB |
| `linear_attn_b` | 24 | 0.005458763018 | 0.005775173912 | 0.005661891291 | 3,152 KiB |
| `linear_attn_out` | 24 | 0.003732074646 | 0.003780259396 | 0.003753065595 | 16,464 KiB |
| `linear_attn_qkv` | 24 | 0.005195938521 | 0.005256533937 | 0.005227698373 | 16,468 KiB |
| `linear_attn_z` | 24 | 0.005183823424 | 0.005237578805 | 0.005201258029 | 12,372 KiB |
| `mlp_down` | 33 | 0.005304337195 | 0.005461632439 | 0.005333573096 | 22,608 KiB |
| `mlp_gate` | 33 | 0.005186197377 | 0.005285911704 | 0.005223280056 | 22,616 KiB |
| `mlp_up` | 33 | 0.005245190541 | 0.005290551044 | 0.005267596002 | 22,612 KiB |

Full quantized-only driver resource use:

| item | value |
| --- | ---: |
| selected tensors | 255 |
| successful tensors | 255 |
| per-tensor re-read verification | skipped |
| elapsed wall time | 17:23.16 |
| maximum RSS | 22,616 KiB |
| parts directory size | 3.8 GiB |

The 255-tensor prototype was merged and verified:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-quantized.json`
- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-full-quantized.txt`

| item | value |
| --- | ---: |
| tensor count | 255 |
| codebook count | 12 |
| total file bytes | 4,049,329,404 |
| verify elapsed wall time | 47.48 s |
| verify maximum RSS | 103,892 KiB |
| verification relative-MSE max delta vs summary | about 1e-12 |

### Full-Package Directory Smoke

`tools/merge-ullm-prototype-dirs.py` was extended with an optional
`--include-passthrough` mode. It copies passthrough safetensors payloads in a
streaming way and adds them to the manifest under a top-level
`passthrough_tensors` field. Quantized tensors remain under `tensors`, so the
existing Rust verifier can continue to verify quantized tensors without seeing
passthrough entries as quantized data.

Passthrough entry fields:

- `name`
- `source_file`
- `dtype`
- `shape`
- `family`
- `elements`
- `payload_file`
- `payload_encoding`
- `payload_bytes`
- `payload_sha256`

Full package prototype:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-package.json`
- merge log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-qwen35-9b-p4p6-full-package.log`
- output:
  `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-package.ullm.d`

| item | value |
| --- | ---: |
| quantized tensors | 255 |
| passthrough tensors | 520 |
| codebooks | 12 |
| passthrough payload bytes | 5,049,777,120 |
| total file bytes | 9,099,409,599 |
| directory size | 8.5 GiB |
| merge elapsed wall time | 8.71 s |
| merge maximum RSS | 36,240 KiB |

The existing Rust prototype verifier also accepts the full-package manifest and
verified the 255 quantized tensors while ignoring the top-level passthrough
field:

- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-qwen35-9b-p4p6-full-package.txt`

| item | value |
| --- | ---: |
| verified quantized tensors | 255 |
| verify elapsed wall time | 48.63 s |
| verify maximum RSS | 103,296 KiB |

`ullm-quant` was then extended with explicit passthrough payload verification:

- CLI flag:
  - `--verify-passthrough`
- release verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-merged-verify-passthrough-qwen35-9b-p4p6-full-package.txt`

The verifier streams each passthrough payload, checks its byte length, and
compares SHA-256 against `payload_sha256` without materializing the payloads.

| item | value |
| --- | ---: |
| verified quantized tensors | 255 |
| verified passthrough tensors | 520 |
| verified passthrough payload bytes | 5,049,777,120 |
| elapsed wall time | 55.37 s |
| maximum RSS | 104,596 KiB |

### Rust Merge CLI Smoke

The Python merge behavior was moved into `ullm-quant` so prototype directories
can be merged from the Rust CLI. New merge flags:

- `--merge-policy-summary`
- `--merge-plan-json`
- `--merge-output-dir`
- `--merge-summary-output`
- `--merge-include-passthrough`
- `--merge-copy-buffer-bytes`
- `--merge-overwrite`

The Rust merge keeps the same manifest structure: quantized tensors under
`tensors`, shared codebooks under `codebooks`, and passthrough safetensors
payloads under top-level `passthrough_tensors`. Passthrough copying is streamed
from safetensors payload offsets while computing SHA-256.

Full quantized-only Rust merge:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-quantized.json`
- merge log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-quantized.log`
- output:
  `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-quantized-rust-merged.ullm.d`
- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-verify-qwen35-9b-p4p6-full-quantized.txt`

| item | value |
| --- | ---: |
| quantized tensors | 255 |
| passthrough tensors | 0 |
| codebooks | 12 |
| total file bytes | 4,049,329,123 |
| directory size | 3.8 GiB |
| merge elapsed wall time | 1.55 s |
| merge maximum RSS | 2,076 KiB |
| verify elapsed wall time | 52.55 s |
| verify maximum RSS | 102,216 KiB |

Full package Rust merge:

- merge summary:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-package.json`
- merge log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-qwen35-9b-p4p6-full-package.log`
- output:
  `/tmp/ullm-prototype-policy-smoke-qwen35-9b-p4p6-full-package-rust-merged.ullm.d`
- verify log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-ullm-prototype-policy-smoke-rust-merged-verify-passthrough-qwen35-9b-p4p6-full-package.txt`

| item | value |
| --- | ---: |
| quantized tensors | 255 |
| passthrough tensors | 520 |
| codebooks | 12 |
| passthrough payload bytes | 5,049,777,120 |
| total file bytes | 9,099,409,318 |
| directory size | 8.5 GiB |
| merge elapsed wall time | 8.77 s |
| merge maximum RSS | 12,372 KiB |
| verify elapsed wall time | 54.52 s |
| verify maximum RSS | 103,288 KiB |

The Rust and Python merge manifests are semantically equivalent for tensor and
passthrough contents, but their manifest JSON float formatting differs by 281
bytes in this run. The verifier checked all 255 quantized tensors plus all 520
passthrough payload SHA-256 values successfully.

## Linear Attention In-Projection Stats

The previous activation-stat collection pattern matched
`linear_attn.out_proj` but missed Qwen3.5's gated delta-net input projections:

- `linear_attn.in_proj_qkv`
- `linear_attn.in_proj_a`
- `linear_attn.in_proj_b`
- `linear_attn.in_proj_z`

`tools/collect-activation-stats.py` now includes these names in the default
module regex. Qwen3.5 applies all four modules directly to hidden states, so
the existing input-second-moment weighting path can be used without a new
activation source.

New R9700 activation stats:

- stats dir:
  `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512-inproj/`
- log:
  `benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512-inproj.log`
- calibration prompts: 32
- calibration tokens: 14,061
- matched modules: 248
- safetensors stat keys: 744
- `linear_attn.in_proj_qkv/a/b/z/out_proj`: 24 modules each
- elapsed wall time: 33.02 s
- maximum RSS: 15,902,624 KiB

The weighted family codebook export was rerun with
`--missing-activation-stats error`:

- output:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted-inproj.json`
- log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-codebooks-qwen35-9b-p4p6-families-weighted-inproj.log`
- codebooks: 24
- activation-weighted codebooks: 24
- activation fallback rows: 0
- tensor samples used for export: 48
- elapsed wall time: 12.30 s
- maximum RSS: 618,104 KiB

The new in-projection activation stats changed the linear-attention in-proj
family codebooks materially. Max absolute codebook deltas versus the old
fallback-unweighted export were about `0.018` to `0.060` for
`linear_attn_a/b/qkv/z`; `linear_attn_out` was unchanged because it was already
weighted.

Family4 weighted tensor sample with the new stats:

- output:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-codebook-r9700-calib32-inproj-qwen35-9b-family4.jsonl`
- log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-scale-codebook-r9700-calib32-inproj-qwen35-9b-family4.log`
- rows: 96
- failures: 0
- elapsed wall time: 15.44 s
- maximum RSS: 633,212 KiB

Policy summary from the new family4 weighted rows:

- output:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-family-policy-r9700-calib32-inproj-qwen35-9b-family4.json`

| policy / cap | high-candidate families | parameter-weighted bpp | combined weighted relative MSE |
| --- | --- | ---: | ---: |
| all g16 | none | 4.500000 | 0.003225949 |
| cap 4.55 | `attn_k,attn_v,linear_attn_a,linear_attn_b,linear_attn_out` | 4.545885 | 0.002518783 |
| cap 4.60 | `attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_z` | 4.598865 | 0.002340079 |
| cap 4.65 | `attn_k,attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_qkv` | 4.636708 | 0.002260232 |
| cap 4.70 | `attn_k,attn_o,attn_v,linear_attn_a,linear_attn_b,linear_attn_out,linear_attn_qkv,linear_attn_z` | 4.666982 | 0.002139622 |
| all g8 | all 12 sampled families | 5.000000 | 0.001886067 |

This tensor-level result says the in-proj families are worth considering for
g8 promotion. It is not enough to update the main policy by itself because the
module-level logit results are more mixed.

An in-projection-focused cumulative logit smoke was run on 12 modules:

- selection:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-inproj12.json`
- result:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj12-r9700-calib32-qwen35-9b-prompts8.jsonl`
- log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj12-r9700-calib32-qwen35-9b-prompts8.log`
- scope: layer 0 and layer 1
  `linear_attn.in_proj_a/b/qkv/z`, `linear_attn.out_proj`, and `mlp.up_proj`
- prompts: 8
- sequence length: 64
- total original selected weight bytes: 470,810,624
- elapsed wall time: 5:39.00
- maximum RSS: 16,365,640 KiB

| variant / policy | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all g16 weighted | 0.000347698 | 0.035510909 | 0.001960925 | 8 / 8 | 9.625 |
| all g8 weighted | 0.000402234 | 0.037558737 | 0.001103623 | 8 / 8 | 9.750 |
| p4p6: only `linear_attn_out` high in this scope | 0.000416993 | 0.038574185 | 0.001656361 | 8 / 8 | 9.875 |
| p4p46: `linear_attn_a,b,out,z` high; `qkv,mlp_up` low | 0.000349033 | 0.036177373 | 0.001686816 | 8 / 8 | 9.875 |
| p4p65: `linear_attn_a,b,out,qkv` high; `z,mlp_up` low | 0.000361837 | 0.035491206 | 0.001986985 | 8 / 8 | 9.875 |
| p4p10: all linear-attention modules high; `mlp_up` low | 0.000403207 | 0.036969288 | 0.001373638 | 8 / 8 | 9.875 |

For this in-proj-heavy smoke, all-g16 was best by logit relative MSE, all-g8
was best by KL, and p4p46 was the best mixed policy by logit relative MSE.
Compared with the old p4p6 policy in the same scope, p4p46 reduced mean logit
relative MSE from `0.000416993` to `0.000349033`. This supports keeping
in-proj g8 promotion as an active candidate, but it does not justify replacing
the main p4p6 policy before a wider module-level or perplexity run.

The wider follow-up mixed linear-attention in-proj modules with dense
self-attention modules from non-adjacent layers:

- selection:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-logit-smoke-selection-inproj22-selfattn.json`
- result:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.jsonl`
- log:
  `benchmarks/results/2026-07-01/aq/2026-07-01-aq-module-logit-smoke-inproj22-selfattn-r9700-calib32-qwen35-9b-prompts8.log`
- scope:
  - layer 0 and layer 12:
    `linear_attn.in_proj_a/b/qkv/z`, `linear_attn.out_proj`, `mlp.up_proj`
  - layer 3 and layer 7:
    `self_attn.q/k/v/o_proj`, `mlp.up_proj`
- prompts: 8
- sequence length: 64
- total original selected weight bytes: 907,018,240
- elapsed wall time: 11:11.54
- maximum RSS: 16,367,660 KiB

| variant / policy | mean logit relative MSE | mean abs error | mean KL(ref, candidate) | top1 matches | mean top10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| all g16 weighted | 0.000579478 | 0.046201342 | 0.001758983 | 8 / 8 | 10.000 |
| all g8 weighted | 0.000452192 | 0.040764980 | 0.002217304 | 8 / 8 | 9.750 |
| p4p6 | 0.000392221 | 0.037963312 | 0.001303061 | 8 / 8 | 9.875 |
| p4p46: `attn_o,v` and `linear_attn_a,b,out,z` high | 0.000384154 | 0.037423817 | 0.001293804 | 8 / 8 | 9.875 |
| p4p65: `attn_k,o,v` and `linear_attn_a,b,out,qkv` high | 0.000412484 | 0.039246285 | 0.001134097 | 8 / 8 | 9.875 |
| p4p70: `attn_k,o,v` and `linear_attn_a,b,out,qkv,z` high | 0.000387565 | 0.037508543 | 0.001182557 | 8 / 8 | 10.000 |
| p4p80: `attn_q,k,o,v` and all linear-attention families high | 0.000426451 | 0.039827092 | 0.001241760 | 8 / 8 | 9.875 |

This wider result changes the interpretation. Once dense self-attention modules
are included, mixed policies beat both all-g16 and all-g8 by logit relative MSE
and KL. p4p46 is the best mixed policy by relative MSE, p4p65 is the best by
KL, and p4p70 is close to p4p46 while preserving full top10 overlap. The
current conservative p4p6 policy still performs well, but p4p46 is now a real
candidate rather than only a tensor-MSE artifact.

## Interpretation

The current evidence supports continuing measurement and quantizer optimization together, not doing a long isolated quantizer-theory phase before measuring. The best gains so far came from trying concrete variants and measuring them quickly.

However, a dedicated quantization-tool optimization track is necessary before full-model conversion:

- Full quantization must be CPU-multithreaded and chunked.
- The current Python tool is acceptable for tensor sampling, but not for final full-model quantization.
- Family-level or tensor-level LUT aggregation must be tested because sample-local codebooks are too optimistic for final format decisions.
- Zero-preserving versus free16 codebooks must be evaluated with model-level quality, not only MSE.
- GGUF linear-attention comparison must apply the Qwen3.5 V-head reorder used by llama.cpp conversion.
- Activation weighting now covers Qwen3.5 linear-attention in-projection
  modules, but tensor-level improvements and logit-level improvements do not
  rank the same. The wider self-attention smoke nevertheless supports treating
  p4p46 and p4p65 as named follow-up candidates alongside the conservative
  p4p6 policy.

## Next Actions

1. Add named plan support for p4p46 and p4p65, or run an equivalent custom
   full-policy conversion, then compare against the existing p4p6 package.
2. Replace the current per-tensor temporary conversion driver with a single
   `ullm-quant` full-conversion command now that Rust-side merge exists.
3. Replace or approximate the exact tensor-scale pre-pass before scaling from
   12 tensors to all policy-selected tensors.
4. Run a small perplexity or next-token loss smoke for p4p6, p4p46, and p4p65;
   logit relative MSE alone is not enough to settle the policy.
5. Add SIMD and multithreaded scheduling only after the scalar C++ semantics
   remain stable across wider conversion.
