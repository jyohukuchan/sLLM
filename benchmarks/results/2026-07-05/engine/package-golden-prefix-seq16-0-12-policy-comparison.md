# Package Golden Prefix Seq16 0..12 Policy Comparison

## Scope

Fixture: `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Backend: CPU device `0`

Layers: `0..12`

Policies:

- p4p6
- p4p46-inproj
- p4p65-inproj

## Prefix Results

| policy | run_mode | max_mse | max_mean_abs_diff | max_abs_diff | min_cosine_similarity | worst_layer | worst_token | worst_hidden | layer10_max_abs | layer10_token | layer10_hidden |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p4p6 | golden_before_each_layer | 0.000373014131 | 0.014173072 | 1.040985107 | 0.998943663 | 10 | 0 | 3456 | 1.040985107 | 0 | 3456 |
| p4p46-inproj | golden_before_each_layer | 0.000361034838 | 0.014148540 | 0.875896454 | 0.998973254 | 10 | 0 | 3456 | 0.875896454 | 0 | 3456 |
| p4p65-inproj | golden_before_each_layer | 0.000361180356 | 0.014173072 | 0.886669159 | 0.998976313 | 10 | 0 | 3456 | 0.886669159 | 0 | 3456 |
| p4p6 | actual_prefix | 0.003687207709 | 0.046535894 | 2.235244751 | 0.993695660 | 10 | 0 | 3456 | 2.235244751 | 0 | 3456 |
| p4p46-inproj | actual_prefix | 0.003190722428 | 0.043881594 | 1.744266510 | 0.994555928 | 10 | 0 | 3456 | 1.744266510 | 0 | 3456 |
| p4p65-inproj | actual_prefix | 0.003322313414 | 0.044468240 | 1.787147522 | 0.994327194 | 10 | 0 | 3456 | 1.787147522 | 0 | 3456 |

## Layer 10 Hidden 3456 Module Split

| policy | package_output_diff | delta_error | attention_row_only | attention_activation | mlp_row_only | mlp_activation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p4p6 | -1.040985107 | -1.040985107 | -0.169056547 | -0.209640273 | -0.613532790 | -0.148687291 |
| p4p46-inproj | -0.875896454 | -0.875896454 | -0.169056547 | -0.021181354 | -0.613532790 | -0.172058034 |
| p4p65-inproj | -0.886669159 | -0.886669159 | -0.169056547 | -0.055811652 | -0.613532790 | -0.148199963 |

## Interpretation

p4p46-inproj remains the best tested policy on this fixture. It has the lowest `actual_prefix` max MSE, mean absolute diff, max absolute diff, and the highest actual-prefix cosine similarity among the three policies.

The layer 10 hidden `3456` outlier is not fixed by the in-projection policy changes. The row-only errors for `linear_attn.out_proj[3456]` and `mlp.down_proj[3456]` are identical across these policies:

- attention row-only error: `-0.169056547`
- MLP row-only error: `-0.613532790`

The policy difference mainly changes activation-path error. That explains why p4p46-inproj improves the outlier compared with p4p6, but the dominant MLP row-dot sensitivity remains. Further improvement likely needs an output-row/down-row treatment rather than another in-projection-only policy swap.
