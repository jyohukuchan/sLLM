| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 10 | 3994 | -0.902752 | 0.75 | 0.385962 | 0.413713 | -0.0277519 | 0.00717885 | -0.011264 | -0.00937685 | 0.0230917 | ok | 0.000369486 | 2902:0.061407 | missing_package | - | - |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 10 | 3994 | attention_input_normed | ok | 0.000534513 | -0.000826138 | -0.134861 | 3994:-0.134861 | 2366:0.00438941 |
| 7 | 10 | 3994 | attention_k_normed | ok | -0.00527059 | -0.00209867 | -0.237227 | 57:0.237227 | 520:-0.186171 |
| 7 | 10 | 3994 | attention_k_projected | ok | -0.000563617 | 0.00361735 | 0.195962 | 669:0.195962 | 520:-0.17408 |
| 7 | 10 | 3994 | attention_projection_input | ok | 0.000369486 | 0.000690894 | -0.061407 | 2902:0.061407 | 2902:0.061407 |
| 7 | 10 | 3994 | attention_q_gate | ok | -0.0115674 | -0.0108572 | -0.079937 | 1450:0.197096 | 2366:0.146205 |
| 7 | 10 | 3994 | attention_q_normed | ok | 0.000878982 | 0.00287704 | 0.406051 | 274:-0.406051 | 2174:0.0986599 |
| 7 | 10 | 3994 | attention_q_query | ok | -0.00274709 | -0.00486994 | -0.147144 | 1591:-0.20128 | 2174:0.0764966 |
| 7 | 10 | 3994 | attention_v_projected | ok | -0.000437037 | -0.00307891 | -0.0955148 | 344:0.0955148 | 548:-0.0626825 |

## Skipped

| layer | reason | package_hidden | fullref_hidden | token |
|---:|---|---:|---:|---:|
| 0 | missing_full_reference | - | - | - |
| 1 | missing_full_reference | - | - | - |
| 2 | missing_full_reference | - | - | - |
| 3 | missing_full_reference | - | - | - |
| 4 | missing_full_reference | - | - | - |
| 5 | missing_full_reference | - | - | - |
| 6 | missing_full_reference | - | - | - |
| 8 | missing_full_reference | - | - | - |
| 9 | missing_full_reference | - | - | - |
| 10 | missing_full_reference | - | - | - |
| 11 | missing_full_reference | - | - | - |
