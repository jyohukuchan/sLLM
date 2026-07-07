# Self-attention layer batch v1

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- device: R9700, device index `2`
- layer: `3`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32+sigmoid_mul_f32+aq4_matvec_batch_f32+add_f32+segmented_rmsnorm_f32+aq4_matvec_batch_f32+silu_mul_f32+aq4_matvec_batch_f32+add_f32`
- required env:
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1`

## Successful Runs

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | attention verification | attention checked values | o proj checked values | MLP proj checked values | verification ms | attention diff | o proj diff | mlp norm diff | mlp gate diff | mlp up diff | mlp activation diff | mlp down diff | layer diff |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3 | 30.783820 | 30.599823 | 31.077481 | 4158.028516 | full | 524288 | 15 | 15/15/15 | 351.262561 | 0.000011265 | 0.000000864 | 0.000005245 | 0.000003099 | 0.000002980 | 0.000000954 | 0.000001550 | 0.000000000 |
| 512 | 3 | 141.768580 | 138.892384 | 143.225358 | 3611.519562 | full | 2097152 | 15 | 15/15/15 | 4821.393215 | 0.000011265 | 0.000000864 | 0.000008106 | 0.000002503 | 0.000004172 | 0.000000954 | 0.000001550 | 0.000000000 |
| 1024 | 1 | 318.234144 | 318.234144 | 318.234144 | 3217.756546 | sampled | 15 | 15 | 15/15/15 | 620.521753 | 0.000000130 | 0.000000864 | 0.000008106 | 0.000002503 | 0.000001371 | 0.000001907 | 0.000001550 | 0.000000000 |
| 2048 | 1 | 777.894286 | 777.894286 | 777.894286 | 2632.748481 | sampled | 15 | 15 | 15/15/15 | 1197.727610 | 0.000000209 | 0.000000864 | 0.000008106 | 0.000002503 | 0.000001371 | 0.000001907 | 0.000001550 | 0.000000000 |
| 4096 | 1 | 2182.970006 | 2182.970006 | 2182.970006 | 1876.342776 | sampled | 15 | 15 | 15/15/15 | 2688.281090 | 0.000000209 | 0.000000864 | 0.000008106 | 0.000001907 | 0.000001550 | 0.000001907 | 0.000001550 | 0.000000000 |
| 8192 | 1 | 6892.180390 | 6892.180390 | 6892.180390 | 1188.593382 | sampled | 15 | 15 | 15/15/15 | 4737.847447 | 0.000000104 | 0.000000864 | 0.000008106 | 0.000001431 | 0.000001796 | 0.000001907 | 0.000001550 | 0.000000000 |
| 16384 | 1 | 24825.171928 | 24825.171928 | 24825.171928 | 659.975288 | sampled | 15 | 15 | 15/15/15 | 9133.427373 | 0.000000320 | 0.000000864 | 0.000008106 | 0.000001431 | 0.000001371 | 0.000001907 | 0.000001788 | 0.000000000 |

## Block Comparison

| prompt tokens | attention-only ms | block ms | layer ms | layer-block delta ms | layer/block wall | layer/attention wall |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 7.637947 | 11.099879 | 30.783820 | 19.683941 | 2.773x | 4.030x |
| 512 | 43.796889 | 54.498359 | 141.768580 | 87.270221 | 2.601x | 3.237x |
| 1024 | 116.215921 | 141.180790 | 318.234144 | 177.053354 | 2.254x | 2.738x |
| 2048 | 374.883299 | 433.086886 | 777.894286 | 344.807400 | 1.796x | 2.075x |
| 4096 | 1339.278313 | 1450.365419 | 2182.970006 | 732.604587 | 1.505x | 1.630x |
| 8192 | 5157.917832 | 5382.886791 | 6892.180390 | 1509.293599 | 1.280x | 1.336x |
| 16384 | 20944.388749 | 21844.617459 | 24825.171928 | 2980.554469 | 1.136x | 1.185x |

## Notes

- `package-self-attn-layer-batch-smoke` extends the self-attention block batch path with post-attention RMSNorm, MLP gate/up/down AQ4 batch projections, SiLU-mul, and final residual add.
- Attention verification is sampled for `1024+`; `o_proj` and MLP projections use sampled AQ4 row dot checks.
- Short prompts are MLP-heavy: layer/block wall is about `2.77x` at 128 tokens and `2.60x` at 512 tokens.
- Long prompts remain attention-heavy, but MLP is still material: layer/block wall is about `1.28x` at 8192 tokens and `1.14x` at 16384 tokens.
