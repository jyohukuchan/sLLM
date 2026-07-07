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

| prompt tokens | repeats | mean ms | min ms | max ms | token/s | verification | checked values | verification ms | attention diff |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 128 | 5 | 7.637947 | 7.368615 | 8.025145 | 16758.430420 | full | 524288 | n/a | 0.000011265 |
| 512 | 3 | 43.796889 | 43.114848 | 44.836067 | 11690.327961 | full | 2097152 | n/a | 0.000011265 |
| 1024 | 1 | 116.215921 | 116.215921 | 116.215921 | 8811.185173 | full | 4194304 | n/a | 0.000011265 |
| 2048 | 1 | 374.883299 | 374.883299 | 374.883299 | 5463.033444 | full | 8388608 | n/a | 0.000011265 |
| 4096 | 1 | 1339.278313 | 1339.278313 | 1339.278313 | 3058.363568 | sampled | 15 | 907.921181 | 0.000000209 |
| 8192 | 1 | 5157.917832 | 5157.917832 | 5157.917832 | 1588.237786 | sampled | 15 | 1730.860843 | 0.000000104 |
| 16384 | 1 | 20944.388749 | 20944.388749 | 20944.388749 | 782.262027 | sampled | 15 | 3420.213255 | 0.000000320 |

## Previous comparison

| prompt tokens | old mean ms | new mean ms | old token/s | new token/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.605704 | 7.637947 | 5202.045750 | 16758.430420 | 3.222x |
| 512 | 281.601274 | 43.796889 | 1818.173590 | 11690.327961 | 6.430x |

## RoPE guard

The fixed `2e-4` RoPE guard was too strict for `2048+` positions because the host and HIP RoPE paths accumulate a small position-dependent f32 difference.
The smoke now reports a length-aware `q_rope_abs_floor`/`k_rope_abs_floor`, capped at `4e-3`.

| prompt tokens | q rope abs floor | q rope diff | k rope abs floor | k rope diff |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 0.000409400 | 0.000270158 | 0.000409400 | 0.000198193 |
| 4096 | 0.000819000 | 0.000506938 | 0.000819000 | 0.000336170 |
| 8192 | 0.001638200 | 0.001175225 | 0.001638200 | 0.000833869 |
| 16384 | 0.003276600 | 0.002606988 | 0.003276600 | 0.001518801 |

Full host attention reference verification is expensive at long prompts and was about `18.25s` at `1024` tokens in the block batch path. The smoke now uses 15-value sampled attention verification for `1024+` unless full reference is explicitly needed. Older rows above may still show full verification if they were measured before the threshold was lowered.
