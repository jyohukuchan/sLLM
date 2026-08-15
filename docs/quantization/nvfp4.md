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
