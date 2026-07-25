# SQ8_1 W8A8 full-model quality gate

**Primary 248-projection decision: `no-go`.**

## Scope

- CPU-only Qwen3.5-9B floating-point reference and fake-quant candidates; no GPU or service was used.
- Primary scope quantizes all 248 selected transformer projections. `lm_head` remains unmodified FP32 there; the separate 249-Linear stress scope adds it explicitly.
- W8A8 uses per-token K=32 signed symmetric int8 activations and per-row K=32 signed symmetric int8 weights, with RNE codes and upward-rounded FP16 scales.
- SQ8_1 values are reconstructed in FP32, then passed through the same floating-point `F.linear` operand boundary as the reference. This is a full-model quantization-propagation gate, not a GPU accumulation-order or performance result.

## Full-model logits

| candidate | relative L2 | max abs | mean KL | top-1 agreement | top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0 | 0 | 0 | 4243/4243 (100.000000%) | 100.000000% |
| w8a16 | 0.016971283 | 13.834223 | 0.00039904574 | 4218/4243 (99.410794%) | 99.066698% |
| w8a8 | 0.023506802 | 7.889154 | 0.00066585346 | 4189/4243 (98.727316%) | 98.378506% |
| outlier_bypass_ge4 | 0.015238139 | 8.9919987 | 0.00026408272 | 4208/4243 (99.175112%) | 98.920575% |
| all_linear_w8a16 | 0.017404657 | 13.847681 | 0.00042023974 | 4212/4243 (99.269385%) | 99.038416% |
| all_linear_w8a8 | 0.024505412 | 7.8782945 | 0.00072750735 | 4197/4243 (98.915861%) | 98.286590% |

## Hidden-state propagation

| candidate | final hidden relative L2 | final hidden max abs | worst layer (relative L2) |
| --- | ---: | ---: | --- |
| w8a16 | 0.024402447 | 14.862573 | model.layers.31: 0.027739784 |
| w8a8 | 0.031263589 | 13.696337 | model.layers.30: 0.033571295 |
| outlier_bypass_ge4 | 0.020038844 | 3.4483917 | model.layers.27: 0.021157756 |
| all_linear_w8a16 | 0.024402447 | 14.862573 | model.layers.31: 0.027739784 |
| all_linear_w8a8 | 0.031263589 | 13.696337 | model.layers.30: 0.033571295 |

## Outliers

- Base W8A8 K=32 outlier blocks `[4,8)`: `14.326868%`; `[8,inf)`: `0.000000%`.
- The diagnostic `outlier_bypass_ge4` bypassed `14.331775%` of activation blocks and removed `100.000000%` of the W8A8-to-W8A16 aggregate-logit-L2 gap.
- Outlier-side-route outlook under the frozen rule: `promising`.

## Gate checks

| check | observed | threshold | pass |
| --- | --- | --- | --- |
| coverage | {"domain_counts": {"chat": 4, "code": 4, "general": 4, "multilingual_ja": 4, "reasoning_math": 4}, "samples_seen": 20, "valid_scored_positions": 4243} | 20 records; five domains x4; >=4,000 scored positions | yes |
| weight_storage_validity | {"code_max": 127, "code_min": -127, "nonfinite_scale_count": 0, "true_clipping_count": 0} | zero clipping; finite scales; code range [-127,127] | yes |
| control_logits_relative_l2 | 0.0 | <= 1e-5 | yes |
| control_logits_max_abs | 0.0 | <= 2e-5 | yes |
| control_final_hidden_relative_l2 | 0.0 | <= 1e-5 | yes |
| control_final_hidden_max_abs | 0.0 | <= 2e-5 | yes |
| w8a16_aggregate_logits_relative_l2 | 0.01697128293481438 | <= 0.040 | yes |
| w8a16_worst_prompt_logits_relative_l2 | 0.05622755428422274 | <= 0.060 | yes |
| w8a8_activation_storage_validity | {"nonfinite_scale_count": 0, "true_clipping_count": 0} | zero clipping; finite scales | yes |
| w8a8_aggregate_logits_relative_l2 | 0.02350680162133803 | <= 0.060 | yes |
| w8a8_worst_prompt_logits_relative_l2 | 0.04395519988723561 | <= 0.080 | yes |
| w8a8_logits_max_abs | 7.889153957366943 | <= 1.0 | NO |
| w8a8_mean_token_kl | 0.0006658534580613823 | <= 0.005 | yes |
| w8a8_worst_prompt_mean_kl | 0.0018419428961351514 | <= 0.010 | yes |
| w8a8_incremental_logits_penalty_ratio | 1.3850927894859901 | W8A8 <= 1.60 * W8A16 | yes |
| w8a8_incremental_logits_penalty_absolute | 0.006535518686523651 | W8A8 <= W8A16 + 0.020 | yes |
| w8a8_maximum_layer_relative_l2 | {"layer": "model.layers.30", "relative_l2_error": 0.033571295373033556} | <= 0.080 | yes |
| w8a8_final_hidden_relative_l2 | 0.031263588663064334 | <= 0.060 | yes |
| w8a8_final_hidden_max_abs | 13.69633674621582 | <= 1.0 | NO |
| w8a8_incremental_final_hidden_penalty_ratio | 1.2811661277644932 | W8A8 <= 1.60 * W8A16 | yes |
| w8a8_incremental_final_hidden_penalty_absolute | 0.006861141559958223 | W8A8 <= W8A16 + 0.020 | yes |
| w8a8_top10_overlap | 0.9837850577421635 | >= 0.950 | yes |
| w8a8_reference_top1_retained_in_top10 | 1.0 | 1.0 | yes |
| w8a8_top1_agreement | {"disallowed_mismatch_count": 16, "rate": 0.9872731557860005, "wilson_lower_95": 0.9834324305604255} | rate >= 0.990; Wilson lower >= 0.985; every mismatch allowed near-margin runner-up swap | NO |

The complete frozen contract is `gate-criteria.md`; reproducibility/provenance is in `measurement-manifest.json`.  Per-prompt values, every layer, mismatch margins, and activation outlier counts are retained as structured evidence in this directory. Raw model tensors are intentionally not written.
