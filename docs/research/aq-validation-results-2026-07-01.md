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
  - Current codebooks are still sample-local, not final family-level LUTs.

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

Result file used for reliable summary:

- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b-mlp-selfattn.jsonl`

24 tensor samples across MLP and full-attention projection tensors:

| metric | value |
| --- | ---: |
| relative MSE mean | 0.004175 |
| relative MSE min | 0.000318 |
| relative MSE max | 0.005902 |
| cosine similarity mean | 0.997911 |
| mean abs error mean | 0.000602 |

Breakdown by GGML type:

| type | mean bpp | tensors | relative MSE mean |
| --- | ---: | ---: | ---: |
| IQ4_XS | 4.25 | 2 | 0.005901 |
| Q4_K | 4.50 | 16 | 0.005336 |
| Q5_K | 5.50 | 1 | 0.001351 |
| Q6_K | 6.5625 | 5 | 0.000331 |

The broader GGUF run including Qwen3.5 linear attention tensors is stored at:

- `benchmarks/results/2026-07-01/aq/2026-07-01-udq4kxl-error-qwen35-9b.jsonl`

That run found very large errors for `linear_attn_*` name mappings. Treat those rows as unresolved mapping/packing validation, not as Unsloth quality measurements.

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

## Interpretation

The current evidence supports continuing measurement and quantizer optimization together, not doing a long isolated quantizer-theory phase before measuring. The best gains so far came from trying concrete variants and measuring them quickly.

However, a dedicated quantization-tool optimization track is necessary before full-model conversion:

- Full quantization must be CPU-multithreaded and chunked.
- The current Python tool is acceptable for tensor sampling, but not for final full-model quantization.
- Family-level or tensor-level LUT aggregation must be tested because sample-local codebooks are too optimistic for final format decisions.
- Zero-preserving versus free16 codebooks must be evaluated with model-level quality, not only MSE.
- GGUF linear-attention mapping needs separate validation before using those rows as external baseline data.

## Next Actions

1. Add family-level LUT aggregation and compare it against sample-local LUTs.
2. Add a row/block-aware optimizer that can optimize codebook and scale jointly for each candidate.
3. Resolve GGUF Qwen3.5 linear-attention tensor mapping before comparing those tensors.
4. Run a small perplexity or logit-difference check for the top aq candidates after tensor-level narrowing.
5. Start designing the full CPU multithreaded quantizer path in Rust/C++ or a Python driver plus C++ worker, avoiding Python element loops.
