# Package Golden Prefix Self-Attention Input RMSNorm Summary

## Finding

The package golden-prefix self-attention branch passed raw layer input directly into q/k/v projection preparation. Qwen3.5 self-attention should apply `input_layernorm` before q/k/v projection, while residual add still uses the original layer input.

## Change

- Read `model.language_model.layers.N.input_layernorm.weight` for self-attention layers.
- Convert Qwen3.5 additive RMSNorm weights through `effective_rmsnorm_weight_values`.
- Feed the normalized sequence to `qwen3_self_attn_prepare_sequence_for_paged_decode_f32`.
- Keep the original layer input as `residual_sequence` for decoder-layer residual add.
- Emit self-attention hot vectors including q/k/v projection-side internals.

## Golden-Before Results

| package | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity |
| --- | ---: | ---: | ---: | ---: |
| p4p6 | 0.000511560667 | 0.016369533 | 0.645427704 | 0.998966216 |
| p4p46-inproj | 0.000487887468 | 0.016121546 | 0.486274719 | 0.999005169 |
| p4p65-inproj | 0.000489076887 | 0.015966775 | 0.612869263 | 0.999012379 |

## Actual-Prefix Results

| package | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity |
| --- | ---: | ---: | ---: | ---: |
| p4p6 | 0.002535515858 | 0.037895676 | 0.894840240 | 0.993748165 |
| p4p46-inproj | 0.001895285663 | 0.033220127 | 0.665708542 | 0.995342680 |
| p4p65-inproj | 0.002149916914 | 0.035086083 | 0.828275681 | 0.994710240 |

## Interpretation

The previous layer 3 self-attention `~6.99` max-abs mismatch was an implementation error. After fixing the self-attention input RMSNorm boundary, p4p46-inproj is the best of the tested policies on this prefix fixture. The remaining error is now sub-1.0 max abs in `actual_prefix 0..8`, with the largest residuals still concentrated around hot hidden channels such as `3994` and `3456`.
