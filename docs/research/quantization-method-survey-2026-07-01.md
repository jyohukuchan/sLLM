# quantization method survey 2026-07-01

## Scope

This note records the first literature-oriented check for the aq validation track.
The goal is not to select a final aq format yet. The goal is to decide which
measurements should come after the initial tensor-MSE search.

Search and page extraction were done with Firecrawl. Firecrawl search returned
empty results for exact paper-title queries in this environment, so known arXiv
abs pages were scraped directly.

## Sources Checked

| method | source | relevant point for uLLM |
| --- | --- | --- |
| GPTQ | https://arxiv.org/abs/2210.17323 | one-shot weight quantization using approximate second-order information |
| SmoothQuant | https://arxiv.org/abs/2211.10438 | moves activation outlier difficulty into weights through an equivalent transform |
| AWQ | https://arxiv.org/abs/2306.00978 | uses activation statistics to identify salient weight channels for weight-only quantization |
| OmniQuant | https://arxiv.org/abs/2308.13137 | learns clipping and equivalent transforms with block-wise error minimization |
| AQLM | https://arxiv.org/abs/2401.06118 | additive multi-codebook quantization for extreme 2-3 bpp compression |
| QuIP# | https://arxiv.org/abs/2402.04396 | Hadamard incoherence plus lattice/vector codebooks for low-bit weight-only PTQ |
| QuaRot | https://arxiv.org/abs/2404.00456 | rotations remove outliers and support 4-bit weights, activations, and KV cache |
| MR-GPTQ / microscaling FP4 | https://arxiv.org/abs/2509.23202 | MXFP4/NVFP4 need format-specific quantization; FP4 is not automatically better than INT4 |
| MXFP4 OAS/MBS | https://arxiv.org/abs/2603.08713 | software scaling schemes can close much of the MXFP4-vs-NVFP4 accuracy gap |
| MXFP4 native training | https://arxiv.org/abs/2605.09825 | deterministic Hadamard rotations can stabilize MXFP4 training paths; inference relevance is indirect |

## Observations

The initial aq measurements are useful but incomplete. They measure direct
weight reconstruction error against BF16 tensors, while several strong PTQ
methods optimize a proxy closer to layer output error.

GPTQ shows that second-order information can matter for low-bit weight-only
quantization. For uLLM, the immediate practical version is not full Hessian
optimization. A diagonal approximation based on input activation second moments
is a lower-cost next step and can be added to the current tensor sampler.

AWQ is especially relevant because aq is currently weight-only. AWQ's key
message for this project is that important channels are better identified from
activation distribution than from weights alone. This means a candidate that
looks good under unweighted tensor MSE can still be poor if it damages high-use
input channels.

SmoothQuant and OmniQuant both support the idea that equivalent transforms and
outlier migration are useful. They are more invasive than the first aq format,
but they suggest two concrete experiments: learnable or searched clipping for
group scales, and optional per-channel scaling during offline quantization.

AQLM, QuIP#, and QuaRot are valuable accuracy references, but they should not be
folded into the first runtime format without measuring cost. AQLM's additive
multi-codebook approach may require multiple table lookups and additions per
value. QuIP#/QuaRot-style rotations or vector codebooks can improve low-bit
accuracy, but they change the runtime path more than the current scalar 4-bit
index plus scale design.

Recent FP4 work is relevant mostly to sq and future aq variants. The FP4 papers
indicate that MXFP4/NVFP4 formats need format-specialized quantization; simply
using a hardware-supported 4-bit float does not guarantee good accuracy. This
matches the local NVFP4 baseline, which was weaker than the current aq candidate
on sampled tensor reconstruction.

## Implications For aq

Do not finalize aq from tensor-MSE results alone. The next quality gate should
measure activation-weighted error and then a small logit or perplexity check.

For a linear layer `y = x W^T`, a diagonal activation proxy can weight each input
column by `h_j = E[x_j^2]`:

```text
weighted_rel_mse = sum_i sum_j h_j * (W_ij - Wq_ij)^2
                 / sum_i sum_j h_j * W_ij^2
```

This is cheaper than GPTQ-style block Hessian optimization but captures the main
AWQ-style concern: high-activation channels should not be treated the same as
rarely used channels.

The first candidate to carry forward remains:

```text
aq4_e4m3_g16_ts_flloyd16
```

The 5.0 bpp comparison point remains:

```text
aq4_e4m3_g8_ts_flloyd16
```

Both should be measured with shared family-level LUTs, because the wider
family-LUT check did not show a meaningful tensor-MSE penalty.

## Next Experiments

1. Collect activation statistics for selected Qwen3.5-9B linear modules.
2. Add activation-weighted error evaluation for aq candidates and external baselines.
3. Try activation-weighted codebook fitting for the current family-LUT candidates.
4. Add scale/clipping variants inspired by AWQ and OmniQuant.
5. Run a small logit-difference or perplexity smoke test before treating any aq row as a format candidate.

The first implementation should avoid storing full activations. Store only
per-module input second moments, counts, and minimal diagnostic statistics.
