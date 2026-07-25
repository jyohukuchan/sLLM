# SQ8_1 W8A8 full-model quality gate

**Primary 248-projection decision: `no-go`.**

## Scope

- CPU-only Qwen3.5-9B floating-point reference and fake-quant candidates; no GPU or service was used.
- Primary scope quantizes all 248 selected transformer projections. `lm_head` remains BF16 there; the separate 249-Linear stress scope adds it explicitly.
- W8A8 uses per-token K=32 signed symmetric int8 activations and per-row K=32 signed symmetric int8 weights, with RNE codes and upward-rounded FP16 scales.
- SQ8_1 values are reconstructed in FP32, then passed through the same floating-point `F.linear` operand boundary as the reference. This is a full-model quantization-propagation gate, not a GPU accumulation-order or performance result.

## Full-model logits

| candidate | relative L2 | max abs | mean KL | top-1 agreement | top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0 | 0 | 0 | 3568/3568 (100.000000%) | 100.000000% |
| w8a16 | 0.017910877 | 13.834223 | 0.00045208583 | 3547/3568 (99.411435%) | 99.033072% |
| w8a8 | 0.024598992 | 7.1152434 | 0.00071874342 | 3522/3568 (98.710762%) | 98.318386% |
| outlier_bypass_ge4 | 0.01664955 | 9.3396282 | 0.00028427927 | 3534/3568 (99.047085%) | 98.878924% |
| all_linear_w8a16 | 0.018333168 | 13.847681 | 0.00047509551 | 3541/3568 (99.243274%) | 99.007848% |
| all_linear_w8a8 | 0.025584687 | 7.1028128 | 0.00078864276 | 3525/3568 (98.794843%) | 98.217489% |

## Hidden-state propagation

| candidate | final hidden relative L2 | final hidden max abs | worst layer (relative L2) |
| --- | ---: | ---: | --- |
| w8a16 | 0.025637208 | 14.862573 | model.layers.31: 0.02911926 |
| w8a8 | 0.032337778 | 13.696337 | model.layers.27: 0.035223129 |
| outlier_bypass_ge4 | 0.021467616 | 3.5575879 | model.layers.27: 0.023209132 |
| all_linear_w8a16 | 0.025637208 | 14.862573 | model.layers.31: 0.02911926 |
| all_linear_w8a8 | 0.032337778 | 13.696337 | model.layers.27: 0.035223129 |

## Outliers

- Base W8A8 K=32 outlier blocks `[4,8)`: `14.734257%`; `[8,inf)`: `0.000000%`.
- The diagnostic `outlier_bypass_ge4` bypassed `14.738308%` of activation blocks and removed `100.000000%` of the W8A8-to-W8A16 aggregate-logit-L2 gap.
- Outlier-side-route outlook under the frozen rule: `promising`.

## Gate checks

| check | observed | threshold | pass |
| --- | --- | --- | --- |
| coverage | {"domain_counts": {"chat": 4, "code": 4, "general": 4, "multilingual_ja": 4, "reasoning_math": 4}, "samples_seen": 20, "valid_scored_positions": 3568} | 20 records; five domains x4; >=4,000 scored positions | NO |
| weight_storage_validity | {"code_max": 127, "code_min": -127, "nonfinite_scale_count": 0, "true_clipping_count": 0} | zero clipping; finite scales; code range [-127,127] | yes |
| control_logits_relative_l2 | 0.0 | <= 1e-5 | yes |
| control_logits_max_abs | 0.0 | <= 2e-5 | yes |
| control_final_hidden_relative_l2 | 0.0 | <= 1e-5 | yes |
| control_final_hidden_max_abs | 0.0 | <= 2e-5 | yes |
| w8a16_aggregate_logits_relative_l2 | 0.017910877325348188 | <= 0.040 | yes |
| w8a16_worst_prompt_logits_relative_l2 | 0.05622755428422274 | <= 0.060 | yes |
| w8a8_activation_storage_validity | {"nonfinite_scale_count": 0, "true_clipping_count": 0} | zero clipping; finite scales | yes |
| w8a8_aggregate_logits_relative_l2 | 0.024598992145342637 | <= 0.060 | yes |
| w8a8_worst_prompt_logits_relative_l2 | 0.04471749066430905 | <= 0.080 | yes |
| w8a8_logits_max_abs | 7.115243434906006 | <= 1.0 | NO |
| w8a8_mean_token_kl | 0.00071874341505659 | <= 0.005 | yes |
| w8a8_worst_prompt_mean_kl | 0.0018419428961351514 | <= 0.010 | yes |
| w8a8_incremental_logits_penalty_ratio | 1.373410788232532 | W8A8 <= 1.60 * W8A16 | yes |
| w8a8_incremental_logits_penalty_absolute | 0.006688114819994449 | W8A8 <= W8A16 + 0.020 | yes |
| w8a8_maximum_layer_relative_l2 | {"layer": "model.layers.27", "relative_l2_error": 0.035223129064534266} | <= 0.080 | yes |
| w8a8_final_hidden_relative_l2 | 0.03233777801570575 | <= 0.060 | yes |
| w8a8_final_hidden_max_abs | 13.69633674621582 | <= 1.0 | NO |
| w8a8_incremental_final_hidden_penalty_ratio | 1.2613611616338796 | W8A8 <= 1.60 * W8A16 | yes |
| w8a8_incremental_final_hidden_penalty_absolute | 0.006700570371054919 | W8A8 <= W8A16 + 0.020 | yes |
| w8a8_top10_overlap | 0.9831838565022422 | >= 0.950 | yes |
| w8a8_reference_top1_retained_in_top10 | 1.0 | 1.0 | yes |
| w8a8_top1_agreement | {"disallowed_mismatch_count": 17, "rate": 0.9871076233183856, "wilson_lower_95": 0.98284727341492} | rate >= 0.990; Wilson lower >= 0.985; every mismatch allowed near-margin runner-up swap | NO |

The complete frozen contract is `gate-criteria.md`; reproducibility/provenance is in `measurement-manifest.json`.  Per-prompt values, every layer, mismatch margins, and activation outlier counts are retained as structured evidence in this directory. Raw model tensors are intentionally not written.
