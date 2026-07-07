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

## Previous comparison

| prompt tokens | old mean ms | new mean ms | old token/s | new token/s | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24.605704 | 7.637947 | 5202.045750 | 16758.430420 | 3.222x |
| 512 | 281.601274 | 43.796889 | 1818.173590 | 11690.327961 | 6.430x |

## Guard note

`prompt_tokens=2048` stopped before attention verification because the existing Q RoPE guard exceeded its current tolerance:

```text
self-attn qkv RoPE batch q RoPE mismatch: max_abs_diff=0.00022828579 tolerance=0.0002
```

This is not a causal attention source-shared failure. Long-token RoPE guard tolerance or reference precision should be reviewed before using `2048+` component rows as pass/fail evidence.
