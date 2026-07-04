| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 13 | 3994 | -1.14528 | 4 | 3.1974 | 3.09268 | 0.104715 | 0.0731562 | 0.0927905 | 0.0525991 | -0.0397414 | ok | 0.000238551 | 4:0.141014 | missing_package | - | - |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | 13 | 3994 | attention_input_normed | ok | -0.000307124 | -0.000188548 | 0.0200729 | 3994:0.0200729 | 43:0.00359571 |
| 11 | 13 | 3994 | attention_k_normed | ok | -0.000584126 | -2.30895e-05 | -0.228539 | 448:-0.228539 | 132:0.276159 |
| 11 | 13 | 3994 | attention_k_projected | ok | -0.00469889 | -0.00614733 | -0.201444 | 139:0.221691 | 132:0.178466 |
| 11 | 13 | 3994 | attention_projection_input | ok | 0.000238551 | 0.000176886 | -0.13795 | 4:0.141014 | 4:0.141014 |
| 11 | 13 | 3994 | attention_q_gate | ok | -0.0112015 | -0.0119571 | -0.00272417 | 54:-0.226113 | 4:0.165068 |
| 11 | 13 | 3994 | attention_q_normed | ok | -0.00172074 | -0.000841567 | -0.00861835 | 2304:-0.238042 | 43:-0.130897 |
| 11 | 13 | 3994 | attention_q_query | ok | -0.00277064 | -0.00458345 | -0.127317 | 2304:-0.528173 | 772:0.106471 |
| 11 | 13 | 3994 | attention_v_projected | ok | -0.00171131 | -0.00505616 | -0.219748 | 593:-0.219748 | 878:-0.081767 |

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
| 7 | missing_full_reference | - | - | - |
| 8 | missing_full_reference | - | - | - |
| 9 | missing_full_reference | - | - | - |
| 10 | missing_full_reference | - | - | - |
