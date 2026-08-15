# Weight NVFP4 encoding contract

## Source lock

- canonical implementation/documentation: NVIDIA Transformer Engine `v2.18`
- annotated tag object: `62f366a50b8e5a96fac7f123a554ab4db928b2a9`
- peeled source commit: `27486e03cfc1fa41f6932dcecdc47c71c47eac3e`
- license: BSD-3-Clause
- documentation: `docs/examples/fp8_primer.ipynb` and
  `transformer_engine/common/include/transformer_engine/recipe.h`
- upstream: <https://github.com/NVIDIA/TransformerEngine/tree/v2.18>

This source is a format and independent-reference input. sLLM does not copy CUDA kernels or training recipe code. The source lock fixes the
numeric interpretation; it does not imply NVIDIA hardware support on AMD targets.

## Versioned sLLM contract

`sllm-weight-nvfp4-v1` is weight-only, row-major, K-axis 1D block scaling:

- value: finite E2M1, two consecutive logical values per byte, lower nibble first;
- E2M1 positive code points: `0, 0.5, 1, 1.5, 2, 3, 4, 6`; bit 3 is the sign;
- block: 16 consecutive K-axis values within one matrix row;
- block scale: OCP E4M3FN byte, one per block, row-major `[N, ceil(K/16)]`;
- tensor scale: one little-endian FP32 value for the complete weight tensor;
- reconstruction: `e2m1(value) * e4m3fn(block_scale) * tensor_scale`;
- tensor scale: `global_amax / (448 * 6)`, or canonical `1.0` for an all-zero tensor;
- unrounded block scale: `(block_amax / 6) / tensor_scale`, encoded to OCP E4M3FN with nearest-even and finite saturation;
- a zero block has E4M3 scale zero. A positive decode scale below the E4M3 representable range may also round to zero; its E2M1 values are then canonically zero rather than being divided by zero;
- E2M1 value: source divided by the decoded block scale and tensor scale, then nearest-even with finite saturation;
- odd total element count: the final unused high nibble is zero; every partial K block is scaled only from logical values;
- source NaN/Inf, noncanonical tail bits, missing scale, zero/nonfinite tensor scale, shape/range/hash mismatch are rejected. Zero block scale is canonical as described above.

The derived artifact stores packed values, block scales, and tensor scale as distinct safetensors entries. Runtime residency may concatenate them
into one checked allocation, but descriptor identity retains their offsets, scale types, logical/padded shape, provider, target, and sidecar
fingerprint.

## Explicit exclusions

- Transformer Engine training-only 16x16 weight scaling, stochastic rounding, random Hadamard transforms, and higher-precision layer policy;
- 4over6 adaptive scaling and E4M3 maximum 256 variants introduced after the base recipe;
- MXFP4 E8M0 scales, W4A4, FP4 activation/attention/KV, and NVIDIA 128x4 hardware scale swizzle;
- any claim of native FP4 arithmetic on RDNA2, RDNA4, or CDNA3.

Provider labels describe execution: `packed-dequant` consumes v1 packed residency directly, while `converted-bf16` expands once at model load.
Neither is called `native`.

## Phase 15Q matched Gemma 4 attribution

Phase 15Q fixes `google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` as the BF16 source and compares only its 144 MLP gate/up/down weights. Attention and all other
weights remain BF16, activations remain BF16, and KV remains FP16. The Unsloth mixed W4A4/W8A8/FP8 checkpoint is not executed as the primary
control.

The importer reads `unsloth/gemma-4-12b-it-NVFP4` revision
`b1f649734b34aa5575b03d186abd1b9be3d0d5c4` positionally. Its low-nibble-first E2M1 and E4M3 block-scale bytes are preserved. The
compressed-tensors `weight_global_scale` is reciprocal, so the sLLM multiplicative tensor scale is exactly `1 / weight_global_scale`.
`input_global_scale` is provenance only in this W4A16 comparison and is not applied. The independent decoder confirmed all 16 E2M1 codes,
E4M3 values, ties, zero blocks, and non-aligned synthetic boundaries before the imported payload reached the GPU provider.

The matched variants are:

- `S0`: current per-tensor min-max scaling;
- `U0`: the losslessly imported Unsloth `imatrix_mse` MLP payload;
- `O0`: the same sLLM quantizer with a bounded per-tensor scale search minimizing sampled weight MSE.

Across all 144 tensors, U0 weight MSE was worse than S0 in every tensor, with median U0/S0 MSE ratio `1.3933`. Nevertheless, over 32 fixed
prompts and 96 teacher-forced positions, U0 reduced full-model median KLD from `0.3315` to `0.1619` on `gfx1201` and from `0.3715` to `0.1736`
on `gfx1030`; top-1 agreement rose from `61.46%` to `79.17%` and from `62.50%` to `76.04%`. O0 improved sampled weight MSE in 120/144
tensors but produced only `0.2880`/`0.3433` median KLD and inconsistent worst cases. Therefore weight-only MSE is not an adequate calibration
objective for this model; activation-aware calibration has material value within the same E2M1/block-16 format.

The layer intervention supports a mixed attribution. On `gfx1201`, U0 improved the single-layer median KLD at layers 0, 1, and 47, but the
selected six-layer cumulative U0 case still reached maximum KLD `12.5620`. The benefit is therefore layer- and prompt-dependent rather than a
uniform replacement rule.

The result does not remove the format/configuration ceiling. U0 improved only 66/96 `gfx1201` positions and 61/96 `gfx1030` positions; its
maximum KLD remained `9.1781`/`7.5777`, far above the unchanged `0.05` budget. No candidate is adopted as a default or production promotion.
S0/U0/O0 therefore remain unadopted sLLM PTQ converter candidates; sensitive-tensor mixed precision and a reproducible activation-aware
converter remain follow-ups. This result does not classify vendor PTQ/QAT checkpoints, native low-bit models, the NVFP4 encoding, or a runtime
that faithfully executes the same quantized artifact as `correctness-only`.

## First-class FP4 model input policy

NVFP4 and OCP MXFP4 are first-class model input formats, not debug-only formats. Their contracts remain distinct: current NVFP4 uses E2M1,
block-16 E4M3 scale, and a global FP32 scale, while MXFP4 uses E2M1 with block-32 E8M0 microscaling. W4A16, W4A4, MXFP4/MXFP8, FP8 attention,
and FP8 KV are model recipe properties and must be read from locked artifact metadata rather than inferred from a generic FP4 label.

Quality gates depend on artifact provenance:

- sLLM-created PTQ from BF16 keeps the corresponding BF16 KLD, top-1, and task-quality gates;
- vendor PTQ/QAT uses exact decode, the same quantized checkpoint in a reference runtime, and relevant task/evaluation retention;
- a native low-bit model without a BF16 source uses artifact fidelity, reference-runtime behavior, and task evaluation, not a nonexistent BF16 KLD.

The choice of a quantized artifact is sufficient user intent. The final GGUF interface uses the same load/generate/serve commands for BF16,
FP8, NVFP4, and MXFP4, automatically selecting a verified provider for the exact target. Low-bit precision alone does not trigger a warning or
confirmation. Provider details are optional diagnostics; corrupt metadata, unsupported encodings, and impossible target contracts fail closed.
The present safetensors sidecar and provider flags are transitional development interfaces.
