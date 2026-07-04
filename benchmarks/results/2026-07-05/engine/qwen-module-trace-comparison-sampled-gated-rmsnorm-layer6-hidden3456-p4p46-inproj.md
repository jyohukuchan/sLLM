| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3456 | 22.6322 | 19.7969 | 42.429 | 19.7969 | 22.6322 | -0.228395 | 21.2335 | -0.442789 | 2.05607 | ok | 0.0333317 | 2656:-58.9123 | ok | 0.00320794 | 8644:-9.36542 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3456 | attention_a_projection | ok | -0.018124 | -0.0214128 | 0.0717573 | 9:-0.176606 | - |
| 6 | 0 | 3456 | attention_b_projection | ok | -0.0259887 | -0.022034 | -0.113469 | 30:-0.113469 | - |
| 6 | 0 | 3456 | attention_beta | ok | -0.00291439 | -0.00258832 | -0.00021708 | 14:-0.00483674 | - |
| 6 | 0 | 3456 | attention_conv | ok | 0.0158833 | 0.0529521 | 0.255256 | 6794:0.231856 | - |
| 6 | 0 | 3456 | attention_gate | ok | -0.028619 | -0.107702 | -0.615042 | 21:0.615042 | - |
| 6 | 0 | 3456 | attention_gate_silu | ok | -0.00124203 | -0.00246279 | -0.076355 | 2340:0.101573 | 2656:-0.076355 |
| 6 | 0 | 3456 | attention_input_normed | ok | 1.5441e-05 | 0.000334552 | 0.0372658 | 3994:0.0372658 | 3109:0.00187039 |
| 6 | 0 | 3456 | attention_pre_gate_normed | ok | 0.0439909 | 0.0950861 | -0.356891 | 2698:-3.09302 | 2703:-3.8611 |
| 6 | 0 | 3456 | attention_projection_input | ok | 0.0333317 | 0.921025 | 58.9123 | 2656:-58.9123 | 2656:-58.9123 |
| 6 | 0 | 3456 | attention_qkv_projection | ok | -0.00328003 | -0.00844965 | -0.228619 | 4082:-0.271646 | - |
| 6 | 0 | 3456 | attention_recurrent | ok | 0.000165778 | 0.0017421 | 0.0618228 | 2703:-0.0727483 | 2703:-0.0727483 |
| 6 | 0 | 3456 | attention_z_projection | ok | -0.00312707 | -0.00437269 | -0.076355 | 3948:0.14323 | 2656:-0.076355 |
| 6 | 0 | 3456 | mlp_activation | ok | 0.00320794 | 0.0844376 | 9.36542 | 8644:-9.36542 | 8644:-9.36542 |
