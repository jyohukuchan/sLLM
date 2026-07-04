| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 3994 | -0.627647 | -20.75 | -20.897 | -21.0194 | 0.122353 | 0.111241 | -0.152723 | -0.00284968 | 0.0750907 | ok | 0.000920822 | 595:-0.0797873 | missing_package | - | - |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 3994 | attention_input_normed | ok | 8.58447e-05 | 0.000141802 | 0.0645409 | 3456:-0.0886517 | 2622:-0.00133938 |
| 7 | 0 | 3994 | attention_k_normed | ok | 0.00193069 | 0.000742925 | -0.00963974 | 789:0.116063 | 83:0.065655 |
| 7 | 0 | 3994 | attention_k_projected | ok | -0.00334428 | -0.0177805 | -0.240496 | 134:-0.369038 | 83:0.123664 |
| 7 | 0 | 3994 | attention_projection_input | ok | 0.000920822 | 0.000677879 | -0.0538611 | 595:-0.0797873 | 595:-0.0797873 |
| 7 | 0 | 3994 | attention_q_gate | ok | -0.007296 | -0.00592312 | 0.106205 | 2110:-0.186563 | 595:-0.163948 |
| 7 | 0 | 3994 | attention_q_normed | ok | 0.000391324 | 0.000625341 | 0.306007 | 25:0.486732 | 2622:0.469929 |
| 7 | 0 | 3994 | attention_q_query | ok | -0.00069596 | -0.00278513 | 0.158631 | 311:-0.714926 | 2622:0.300192 |
| 7 | 0 | 3994 | attention_v_projected | ok | 0.000136879 | -0.00788753 | -0.237398 | 314:0.321188 | 83:0.0688748 |

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
