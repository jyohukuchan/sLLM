| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 7 | 3994 | -0.610977 | 2.25 | 2.00122 | 1.9872 | 0.0140228 | 0.0687093 | -0.0597938 | -0.0287338 | -0.0121957 | ok | -9.72332e-05 | 3182:-0.0968925 | missing_package | - | - |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | 7 | 3994 | attention_input_normed | ok | -0.000132822 | -0.0010563 | -0.0930023 | 3994:-0.0930023 | 260:0.00304806 |
| 11 | 7 | 3994 | attention_k_normed | ok | -0.00093577 | 0.000112617 | -0.251618 | 448:-0.251618 | 132:0.202223 |
| 11 | 7 | 3994 | attention_k_projected | ok | -0.00742744 | -0.00925382 | -0.174188 | 930:-0.176427 | 132:0.13946 |
| 11 | 7 | 3994 | attention_projection_input | ok | -9.72332e-05 | -0.000935537 | -0.0968925 | 3182:-0.0968925 | 3182:-0.0968925 |
| 11 | 7 | 3994 | attention_q_gate | ok | -0.0115221 | -0.0126204 | 0.0542173 | 2385:0.252708 | 3182:-0.126531 |
| 11 | 7 | 3994 | attention_q_normed | ok | -0.000245863 | -4.08023e-05 | -0.0803509 | 1849:-0.223094 | 43:-0.143312 |
| 11 | 7 | 3994 | attention_q_query | ok | -0.001483 | -0.0043174 | -0.55842 | 1081:0.55842 | 43:-0.08824 |
| 11 | 7 | 3994 | attention_v_projected | ok | -0.000354947 | -0.0042178 | -0.172163 | 593:-0.172163 | 878:-0.182911 |

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
