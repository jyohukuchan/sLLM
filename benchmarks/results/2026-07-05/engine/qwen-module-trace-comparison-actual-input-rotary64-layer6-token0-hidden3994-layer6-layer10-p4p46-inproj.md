| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3994 | -0.480637 | 25.25 | 24.6623 | 25.3929 | -0.730637 | 0.0218794 | -0.096949 | -0.376917 | -0.269932 | ok | -0.000348537 | 2656:-0.194641 | ok | -2.28129e-05 | 8644:0.389259 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 0 | 3994 | attention_a_projection | ok | -0.0223796 | -0.0242254 | 0.114887 | 9:-0.191788 | 30:0.114887 |
| 6 | 0 | 3994 | attention_b_projection | ok | -0.0287917 | -0.0248061 | -0.0933161 | 30:-0.0933161 | 30:-0.0933161 |
| 6 | 0 | 3994 | attention_beta | ok | -0.003093 | -0.00281409 | -0.000176728 | 17:0.00255084 | 21:-0.0103186 |
| 6 | 0 | 3994 | attention_conv | ok | -9.42144e-05 | -0.000423501 | -0.0238209 | 6794:-0.0332158 | 7259:0.00733069 |
| 6 | 0 | 3994 | attention_conv_pre_silu | ok | -0.00014428 | -0.000109365 | -0.0161126 | 6799:0.0371132 | 6799:0.0371132 |
| 6 | 0 | 3994 | attention_gate | ok | -0.0310591 | -0.114741 | -0.651808 | 21:0.651808 | 21:0.651808 |
| 6 | 0 | 3994 | attention_gate_silu | ok | -0.00114352 | -0.00273909 | -0.129032 | 2656:-0.129032 | 2656:-0.129032 |
| 6 | 0 | 3994 | attention_input_normed | ok | 0.000393583 | 0.000584122 | 0.0655937 | 3994:0.0655937 | 2803:0.00304759 |
| 6 | 0 | 3994 | attention_pre_gate_normed | ok | -0.00106151 | -0.00271744 | -0.0818415 | 2429:-0.113606 | 3163:0.0802312 |
| 6 | 0 | 3994 | attention_projection_input | ok | -0.000348537 | 0.00288443 | 0.194641 | 2656:-0.194641 | 2656:-0.194641 |
| 6 | 0 | 3994 | attention_qkv_projection | ok | -0.00304107 | -0.00784167 | -0.238207 | 7957:-0.238207 | 7259:0.194293 |
| 6 | 0 | 3994 | attention_recurrent | ok | -1.62024e-06 | -6.9279e-06 | -5.53951e-06 | 2698:-0.000648795 | 3163:0.00011131 |
| 6 | 0 | 3994 | attention_recurrent_k | ok | -0.000113016 | -1.7259e-08 | -0.000874519 | 1147:-0.019334 | 1395:0.00531773 |
| 6 | 0 | 3994 | attention_recurrent_q | ok | -5.83858e-06 | -4.92193e-10 | -0.00155946 | 342:0.0044114 | 719:-0.00109766 |
| 6 | 0 | 3994 | attention_recurrent_v | ok | -4.62247e-05 | -0.000516521 | -0.0227356 | 2698:-0.0332158 | 3163:0.00733069 |
| 6 | 0 | 3994 | attention_z_projection | ok | -0.00289136 | -0.00452033 | -0.129032 | 3948:0.138128 | 2656:-0.129032 |
| 6 | 0 | 3994 | mlp_activation | ok | -2.28129e-05 | -0.00363346 | -0.389259 | 8644:0.389259 | 8644:0.389259 |
| 6 | 0 | 3994 | mlp_gate_projection | ok | -0.00240349 | -0.00246378 | -0.110079 | 8644:-0.110079 | 8644:-0.110079 |
| 6 | 0 | 3994 | mlp_gate_silu | ok | -0.000513938 | -0.00104958 | -0.110089 | 8644:-0.110089 | 8644:-0.110089 |
| 6 | 0 | 3994 | mlp_up_projection | ok | 0.000646928 | 0.000227186 | -0.013957 | 3778:0.102806 | 2232:0.0652182 |

## Skipped

| layer | reason | package_hidden | fullref_hidden | token |
|---:|---|---:|---:|---:|
| 0 | missing_full_reference | - | - | - |
| 1 | missing_full_reference | - | - | - |
| 2 | missing_full_reference | - | - | - |
| 3 | missing_full_reference | - | - | - |
| 4 | missing_full_reference | - | - | - |
| 5 | missing_full_reference | - | - | - |
| 7 | missing_full_reference | - | - | - |
