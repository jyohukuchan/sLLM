| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 12 | 3994 | 0.403383 | -0.3125 | -0.4058 | -0.496683 | 0.0908833 | 0.000485314 | 0.0120055 | -0.0119364 | 0.106516 | ok | 0.000192499 | 3140:0.133111 | ok | -9.55028e-05 | 7685:-0.171022 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 6 | 12 | 3994 | attention_a_projection | ok | -5.02334e-05 | -0.0106778 | 0.0336518 | 30:0.140305 | 30:0.140305 |
| 6 | 12 | 3994 | attention_b_projection | ok | -0.0125348 | -0.0197036 | -0.0993745 | 11:0.0993745 | 24:0.0467062 |
| 6 | 12 | 3994 | attention_beta | ok | 0.000949795 | 0.00020447 | -0.00635576 | 5:-0.0105609 | 19:0.00823963 |
| 6 | 12 | 3994 | attention_conv | ok | -4.41668e-05 | -0.000113414 | -0.0723639 | 1113:0.0991104 | 7236:0.00328179 |
| 6 | 12 | 3994 | attention_conv_pre_silu | ok | -0.000113117 | -0.000164695 | 0.0200176 | 3459:-0.0829346 | 7236:0.00605069 |
| 6 | 12 | 3994 | attention_gate | ok | -0.00132778 | -0.00412411 | -0.0246132 | 21:0.0246132 | 6:0.00599492 |
| 6 | 12 | 3994 | attention_gate_silu | ok | -0.000542797 | -0.00235235 | -0.0928388 | 2688:0.137801 | 3643:0.0651722 |
| 6 | 12 | 3994 | attention_input_normed | ok | 8.41125e-05 | -0.000138057 | -0.0105896 | 3842:0.0283222 | 333:0.00031665 |
| 6 | 12 | 3994 | attention_pre_gate_normed | ok | 0.000269877 | -0.00084761 | -0.207301 | 2698:-0.207301 | 3140:0.0519872 |
| 6 | 12 | 3994 | attention_projection_input | ok | 0.000192499 | 5.01346e-05 | -0.0487975 | 3140:0.133111 | 3140:0.133111 |
| 6 | 12 | 3994 | attention_qkv_projection | ok | -0.00114432 | -0.00167504 | 0.063024 | 6794:0.235783 | 4429:-0.101846 |
| 6 | 12 | 3994 | attention_recurrent | ok | -3.53387e-08 | -7.41766e-06 | -0.000499968 | 2698:-0.000671607 | 3909:-7.72637e-05 |
| 6 | 12 | 3994 | attention_recurrent_k | ok | 0.00015846 | -2.71059e-09 | -0.00395149 | 1877:-0.0265816 | 1851:0.0112439 |
| 6 | 12 | 3994 | attention_recurrent_q | ok | -1.13809e-05 | -1.78775e-09 | -0.0010466 | 84:0.00247565 | 1851:0.000733791 |
| 6 | 12 | 3994 | attention_recurrent_v | ok | -9.98762e-05 | -0.000822511 | -0.0723639 | 2698:-0.0723639 | 3140:0.00328179 |
| 6 | 12 | 3994 | attention_z_projection | ok | -0.00166491 | -0.00457821 | -0.179089 | 3576:0.179089 | 3643:0.0598111 |
| 6 | 12 | 3994 | mlp_activation | ok | -9.55028e-05 | -0.00118966 | -0.171022 | 7685:-0.171022 | 7685:-0.171022 |
| 6 | 12 | 3994 | mlp_gate_projection | ok | 0.00130852 | 0.000752281 | -0.112983 | 7685:-0.125258 | 7685:-0.125258 |
| 6 | 12 | 3994 | mlp_gate_silu | ok | 0.000161744 | -0.000758553 | -0.12274 | 7685:-0.136099 | 7685:-0.136099 |
| 6 | 12 | 3994 | mlp_up_projection | ok | -0.000285105 | -0.00100128 | -0.0609872 | 8472:0.112933 | 5726:-0.0655475 |

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
| 8 | missing_full_reference | - | - | - |
| 9 | missing_full_reference | - | - | - |
| 10 | missing_full_reference | - | - | - |
| 11 | missing_full_reference | - | - | - |
