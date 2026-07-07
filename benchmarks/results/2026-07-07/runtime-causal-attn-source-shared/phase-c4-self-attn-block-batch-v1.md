# Self-attention block batch v1

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- device: R9700, device index `2`
- layer: `3`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+aq4_matvec_batch_f32+add_f32`
- required env:
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1`

## Successful runs

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | attention verification | attention checked values | o proj checked values | verification ms | attention diff | output gate diff | o proj diff | block diff |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3 | 11.099879 | 10.578737 | 12.033792 | 11531.656891 | full | 524288 | 15 | 312.756027 | 0.000011265 | 0.000000477 | 0.000000864 | 0.000000000 |
| 512 | 3 | 54.498359 | 53.478065 | 55.043362 | 9394.778233 | full | 2097152 | 15 | 4794.963222 | 0.000011265 | 0.000000477 | 0.000000864 | 0.000000000 |
| 1024 | 1 | 141.180790 | 141.180790 | 141.180790 | 7253.111418 | sampled | 15 | 15 | 294.265679 | 0.000000130 | 0.000000477 | 0.000000864 | 0.000000000 |
| 2048 | 1 | 433.086886 | 433.086886 | 433.086886 | 4728.843256 | sampled | 15 | 15 | 673.064117 | 0.000000209 | 0.000000477 | 0.000000864 | 0.000000000 |
| 4096 | 1 | 1450.365419 | 1450.365419 | 1450.365419 | 2824.115872 | sampled | 15 | 15 | 1276.855249 | 0.000000209 | 0.000000477 | 0.000000864 | 0.000000000 |
| 8192 | 1 | 5382.886791 | 5382.886791 | 5382.886791 | 1521.859983 | sampled | 15 | 15 | 2104.677288 | 0.000000104 | 0.000000477 | 0.000000864 | 0.000000000 |
| 16384 | 1 | 21844.617459 | 21844.617459 | 21844.617459 | 750.024578 | sampled | 15 | 15 | 4094.617209 | 0.000000320 | 0.000000477 | 0.000000864 | 0.000000000 |

## Attention-only comparison

| prompt tokens | attention-only ms | block ms | block delta ms | attention-only tok/s | block tok/s | block/attention wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.637947 | 11.099879 | 3.461932 | 16758.430420 | 11531.656891 | 1.453x |
| 512 | 43.796889 | 54.498359 | 10.701470 | 11690.327961 | 9394.778233 | 1.244x |
| 1024 | 116.215921 | 141.180790 | 24.964869 | 8811.185173 | 7253.111418 | 1.215x |
| 2048 | 374.883299 | 433.086886 | 58.203587 | 5463.033444 | 4728.843256 | 1.155x |
| 4096 | 1339.278313 | 1450.365419 | 111.087106 | 3058.363568 | 2824.115872 | 1.083x |
| 8192 | 5157.917832 | 5382.886791 | 224.968959 | 1588.237786 | 1521.859983 | 1.044x |
| 16384 | 20944.388749 | 21844.617459 | 900.228710 | 782.262027 | 750.024578 | 1.043x |

## Notes

- `package-self-attn-block-batch-smoke` adds output gate, AQ4 `o_proj` batch projection, and residual add to the existing self-attention attention batch path.
- Attention verification switches to sampled mode for `1024+` because full host reference was already about `18.25s` at 1024 tokens.
- `o_proj` is verified with sampled AQ4 row dot products to avoid materializing a full host projection reference for long prompts.
- Long prompts remain dominated by causal attention. The block delta is visible, but block/attention wall ratio shrinks to about `1.04x` at 8192/16384.
