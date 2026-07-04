| layer | kind | mode | range | hot | tok | output_diff | input_diff | delta_diff | attn | mlp | expected_delta | actual_delta | dominant | shape |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | linear_attention | actual_prefix | 0..8 | 3994 | 0 | -5.48943 | 0 | -5.48943 | 1.68055 | -0.133359 | 7.03662 | 1.54719 | attention | missing_expected_delta |
| 1 | linear_attention | actual_prefix | 0..8 | 3994 | 0 | -9.84638 | -5.48943 | -4.35696 | 0.0239499 | -0.068407 | 4.3125 | -0.0444572 | mlp | missing_expected_delta |
| 2 | linear_attention | actual_prefix | 0..8 | 3994 | 0 | -15.9208 | -9.84638 | -6.07439 | 0.0355665 | -0.234955 | 5.875 | -0.199389 | mlp | missing_expected_delta |
| 3 | self_attention | actual_prefix | 0..8 | 3994 | 0 | -14.527 | -15.9208 | 1.39381 | 0.619089 | -0.225277 | -1 | 0.393812 | attention | mixed_delta |
| 4 | linear_attention | actual_prefix | 0..8 | 3994 | 1 | -18.8456 | -9.13796 | -9.70761 | -0.0531771 | -7.59193 | 2.0625 | -7.64511 | mlp | spurious_actual_delta |
| 5 | linear_attention | actual_prefix | 0..8 | 3994 | 1 | -24.4622 | -18.8456 | -5.61662 | 0.00846917 | -3.75009 | 1.875 | -3.74162 | mlp | opposite_delta |
| 6 | linear_attention | actual_prefix | 0..8 | 3994 | 0 | -44.6183 | -18.2257 | -26.3926 | 0.00787164 | -1.1505 | 25.25 | -1.14263 | mlp | missing_expected_delta |
| 7 | self_attention | actual_prefix | 0..8 | 3994 | 1 | -31.4694 | -27.3829 | -4.08646 | -1.43204 | -3.02942 | -0.375 | -4.46146 | mlp | spurious_actual_delta |
| 4 | linear_attention | golden_before_each_layer | 4..8 | 3994 | 7 | -9.68382 | 0 | -9.68382 | -1.36084 | -6.44798 | 1.875 | -7.80882 | mlp | spurious_actual_delta |
| 5 | linear_attention | golden_before_each_layer | 4..8 | 3994 | 6 | -7.195 | 0 | -7.195 | 0.110342 | -5.61784 | 1.6875 | -5.5075 | mlp | spurious_actual_delta |
| 6 | linear_attention | golden_before_each_layer | 4..8 | 3994 | 0 | -25.1466 | 0 | -25.1466 | -0.0136022 | 0.116955 | 25.25 | 0.103353 | mlp | missing_expected_delta |
| 7 | self_attention | golden_before_each_layer | 4..8 | 3994 | 1 | -5.71349 | 0 | -5.71349 | -4.582 | -1.50648 | -0.375 | -6.08849 | attention | spurious_actual_delta |
