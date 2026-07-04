| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3456 | 21.9872 | 19.7969 | 41.7841 | 19.7969 | 21.9872 | -0.228395 | 20.6754 | -0.442789 | 1.96916 | ok | 0.0335195 | 2656:-57.3472 | ok | 0.00297999 | 8644:-9.023 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3456 | attention_a_projection | ok | -0.0428721 | -0.0441565 | -0.0606213 | 22:-0.15901 | - |
| 6 | 0 | 3456 | attention_b_projection | ok | -0.00720042 | -0.0145109 | -0.0738244 | 11:-0.119499 | - |
| 6 | 0 | 3456 | attention_beta | ok | 0.00180571 | 0.000788787 | -0.000138342 | 14:0.0062167 | - |
| 6 | 0 | 3456 | attention_conv | ok | 0.0158789 | 0.0521133 | 0.230161 | 6794:0.222785 | - |
| 6 | 0 | 3456 | attention_gate | ok | 0.0226698 | 0.225421 | 1.35119 | 21:-1.35119 | - |
| 6 | 0 | 3456 | attention_gate_silu | ok | -0.00353762 | -0.00533674 | -0.111973 | 2656:-0.111973 | 2656:-0.111973 |
| 6 | 0 | 3456 | attention_input_normed | ok | 1.5441e-05 | 0.000334552 | 0.0372658 | 3994:0.0372658 | 3109:0.00187039 |
| 6 | 0 | 3456 | attention_pre_gate_normed | ok | 0.0450962 | 0.0975204 | -0.347733 | 2698:-3.08502 | 2703:-3.90719 |
| 6 | 0 | 3456 | attention_projection_input | ok | 0.0335195 | 0.89666 | 57.3472 | 2656:-57.3472 | 2656:-57.3472 |
| 6 | 0 | 3456 | attention_qkv_projection | ok | -0.00251259 | -0.00857524 | -0.301666 | 7957:-0.301666 | - |
| 6 | 0 | 3456 | attention_recurrent | ok | 0.000171788 | 0.00182627 | 0.0652994 | 2703:-0.0765739 | 2703:-0.0765739 |
| 6 | 0 | 3456 | attention_z_projection | ok | -0.00499606 | -0.00651498 | -0.111973 | 3948:0.153257 | 2656:-0.111973 |
| 6 | 0 | 3456 | mlp_activation | ok | 0.00297999 | 0.0812911 | 9.023 | 8644:-9.023 | 8644:-9.023 |
