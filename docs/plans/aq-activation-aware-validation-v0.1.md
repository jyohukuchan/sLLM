# aq activation-aware validation plan v0.1

## Purpose

The first aq pass found promising tensor-level candidates, but tensor MSE alone
cannot decide the format. This plan adds activation-aware validation before any
aq layout decision is treated as stable.

The plan is deliberately measurement-first. aq and sq formats are still
experimental, so this phase should add evidence without locking the final
container or quantization scheme.

## Inputs

- Reference model:
  - `/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B`
- Existing baselines:
  - ModelOpt NVFP4 safetensors
  - Unsloth Dynamic Q4_K_XL GGUF
- Current aq candidates:
  - `aq4_e4m3_g16_ts_flloyd16`
  - `aq4_e4m3_g8_ts_flloyd16`

## Activation Statistics

Add a calibration tool:

```text
tools/collect-activation-stats.py
```

Initial implementation status:

- Added.
- Writes `activation_second_moments.safetensors` plus `metadata.json`.
- Uses forward pre-hooks on matching `torch.nn.Linear` modules.
- Stores reductions only: second moment, mean absolute value, and max absolute value.
- Defaults to `AutoModel` instead of `AutoModelForCausalLM` to avoid materializing huge logits.

For each selected linear module, collect:

- module name,
- tensor family,
- input feature count,
- sample count,
- per-input-channel second moment `E[x_j^2]`,
- optional per-input-channel mean absolute value,
- optional per-input-channel max absolute value,
- prompt and sequence-length metadata.

Do not store full activations. Accumulate streaming reductions from forward
hooks and write compact stats to disk.

Initial calibration size:

- 128-512 text samples,
- sequence lengths 512 and 2048 as separate runs if practical,
- deterministic prompt ordering and seed recording.

If GPU execution is available, use R9700 first. If not, CPU execution is
acceptable for a small smoke run, but it should not block the quantizer work.

## Weighted Error Metric

Add a weighted evaluator:

```text
tools/run-aq-weighted-sample.py
```

Initial implementation status:

- Added as a thin entry point over `tools/run-aq-tensor-sample.py`.
- `tools/run-aq-tensor-sample.py` now accepts `--activation-stats`.
- When stats are present, result rows include `weighted_mse` and `weighted_relative_mse`.
- The optimizer metadata switches from `mse` to `activation_weighted_mse`.

For a weight matrix `W` with shape `[out_features, in_features]`, and activation
second moments `h_j`, compute:

```text
weighted_rel_mse = sum_i sum_j h_j * (W_ij - Wq_ij)^2
                 / sum_i sum_j h_j * W_ij^2
```

Record both unweighted and weighted metrics in every row. The weighted result
should be used to rank candidates only after checking that tensor coverage and
activation stats line up with the module naming.

## Candidate Variants

Run these variants first:

| candidate | purpose |
| --- | --- |
| current family-LUT g16 | baseline 4.5 bpp aq row |
| current family-LUT g8 | 5.0 bpp accuracy point |
| activation-weighted Lloyd g16 | test whether codebook centers should prefer high-activation channels |
| clipped-scale g16 | test AWQ/OmniQuant-style outlier handling without changing runtime format |
| zero-preserving g16 | check whether free16 codebooks hurt model-level behavior despite lower tensor MSE |

The first weighted implementation can reuse the Python sampler. The production
path still belongs in `ullm-quant` with C++20 kernels and explicit CPU threading.

## Logit And Perplexity Smoke

After weighted tensor ranking, run a small model-level check:

- select a limited prompt set,
- replace or simulate quantized weights for a small candidate set,
- compare logits against BF16,
- record max/mean logit error and next-token KL divergence if available,
- add perplexity only after the replacement path is reliable.

This can be slower than tensor-level sampling, so keep the first run narrow.

## Threading And Memory Rules

- Quantization/search code must make CPU thread count explicit.
- Avoid Python element loops for full-model quantization.
- Do not materialize the full model or full dequantized tensors.
- Process tensors and activation stats in chunks.
- Keep activation statistics as reductions, not raw activation dumps.
- Record thread settings, model path, calibration set, sequence length, and git commit in every result file.

## Acceptance Criteria For This Phase

This phase is complete when:

1. activation stats exist for representative MLP, linear-attention, and full-attention families,
2. weighted error rows exist for NVFP4, Unsloth Dynamic, and at least two aq candidates,
3. at least one activation-weighted aq variant is compared against the current unweighted family-LUT candidate,
4. a small logit-difference smoke test is either run or explicitly blocked by implementation constraints.

Only after that should aq candidate discussions move from "tensor-level
candidate" to "format candidate".

## Current Verification

The weighted path was first smoke-tested on one Qwen3.5-9B tensor with unit
activation weights:

```text
model.language_model.layers.14.mlp.down_proj.weight
```

With unit activation weights and candidate `aq4_e4m3_g16_ts_flloyd16`,
`weighted_relative_mse` was `0.005159932654350996`, matching the unweighted
relative MSE as expected for all-one weights.

A real CPU activation-stat smoke also succeeded:

- stats output:
  - `benchmarks/results/2026-07-01/aq/activation-smoke-qwen35-9b/`
- weighted result:
  - `benchmarks/results/2026-07-01/aq/2026-07-01-aq-weighted-smoke-qwen35-9b.jsonl`
- module:
  - `language_model.layers.0.mlp.down_proj`
- samples/tokens:
  - 1 prompt, 15 tokens
- candidate:
  - `aq4_e4m3_g16_ts_flloyd16`
- unweighted relative MSE:
  - `0.0051582370266549235`
- weighted relative MSE:
  - `0.004603734239935875`

The default Python environment currently has CPU-only PyTorch. Small CPU
activation smoke runs work, but full activation collection should use an
R9700-capable environment.
