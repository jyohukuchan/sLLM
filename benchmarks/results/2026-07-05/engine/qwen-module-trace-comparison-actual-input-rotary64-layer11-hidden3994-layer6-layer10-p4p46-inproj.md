| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 7 | 3994 | -0.645338 | 2.25 | 1.98165 | 2.00199 | -0.0203381 | 0.0683678 | -0.0520625 | -0.0286724 | -0.0100836 | ok | -0.000106721 | 3182:-0.11399 | missing_package | - | - |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | 7 | 3994 | attention_input_normed | ok | -1.81857e-05 | -0.00103263 | -0.129604 | 3994:-0.129604 | 260:0.00186187 |
| 11 | 7 | 3994 | attention_k_normed | ok | -0.00122353 | -0.000266284 | -0.251634 | 448:-0.251634 | 132:0.216279 |
| 11 | 7 | 3994 | attention_k_projected | ok | -0.00687441 | -0.0084525 | -0.171152 | 448:-0.171152 | 132:0.137925 |
| 11 | 7 | 3994 | attention_projection_input | ok | -0.000106721 | -0.000923903 | -0.11399 | 3182:-0.11399 | 3182:-0.11399 |
| 11 | 7 | 3994 | attention_q_gate | ok | -0.0106632 | -0.0116345 | 0.052362 | 2385:0.254726 | 3182:-0.125698 |
| 11 | 7 | 3994 | attention_q_normed | ok | -4.51755e-05 | 0.00010545 | -0.0801163 | 1849:-0.169865 | 43:-0.143092 |
| 11 | 7 | 3994 | attention_q_query | ok | -0.00132322 | -0.00428444 | -0.523489 | 1081:0.523489 | 43:-0.0881879 |
| 11 | 7 | 3994 | attention_v_projected | ok | -0.00027284 | -0.00391558 | -0.163671 | 593:-0.163671 | 878:-0.182372 |

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
