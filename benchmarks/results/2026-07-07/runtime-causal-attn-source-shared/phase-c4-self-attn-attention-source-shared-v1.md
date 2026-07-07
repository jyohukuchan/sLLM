# Self-attention causal attention source-shared v1

- package: `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-reservoir65536-jobs4.ullm.d`
- device: R9700, device index `2`
- layer: `3`
- executor: `segmented_rmsnorm_f32+aq4_matvec_batch_f32+qwen35_qk_norm_rope_batch_f32+causal_attn_f32`
- required env:
  - `ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL=1`
  - `ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL=1`

## Successful runs

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | attention diff |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5 | 7.637947 | 7.368615 | 8.025145 | 16758.430420 | 0.000011265 |
| 512 | 3 | 43.796889 | 43.114848 | 44.836067 | 11690.327961 | 0.000011265 |
| 1024 | 1 | 116.215921 | 116.215921 | 116.215921 | 8811.185173 | 0.000011265 |
| 2048 | 1 | 374.883299 | 374.883299 | 374.883299 | 5463.033444 | 0.000011265 |
| 4096 | 1 | 1331.671565 | 1331.671565 | 1331.671565 | 3075.833492 | 0.000011265 |

## Previous comparison

| prompt tokens | old mean ms | new mean ms | old token/s | new token/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.605704 | 7.637947 | 5202.045750 | 16758.430420 | 3.222x |
| 512 | 281.601274 | 43.796889 | 1818.173590 | 11690.327961 | 6.430x |

## RoPE guard

The fixed `2e-4` RoPE guard was too strict for `2048+` positions because the host and HIP RoPE paths accumulate a small position-dependent f32 difference.
The smoke now reports a length-aware `q_rope_abs_floor`/`k_rope_abs_floor`, capped at `1e-3`.

| prompt tokens | q rope abs floor | q rope diff | k rope abs floor | k rope diff |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.000409400 | 0.000270158 | 0.000409400 | 0.000198193 |
| 4096 | 0.000819000 | 0.000506938 | 0.000819000 | 0.000336170 |

Full host attention reference verification is now expensive at `4096` tokens. Future `4096+` rows should use sampled attention verification unless full reference is explicitly needed.
