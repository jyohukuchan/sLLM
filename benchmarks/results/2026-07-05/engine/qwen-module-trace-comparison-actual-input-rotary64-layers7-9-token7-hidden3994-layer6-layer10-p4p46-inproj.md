| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 3994 | -0.15629 | 2.125 | 1.68231 | 1.7136 | -0.0312901 | 0.0128176 | 0.0684276 | -0.0346915 | -0.111107 | token_mismatch | 0.0177234 | - | missing_package | - | - |
| 8 | 7 | 3994 | 0.296179 | 0.625 | 1.07747 | 0.90629 | 0.171179 | -0.0532685 | 0.125137 | -0.0178453 | 0.0700611 | token_mismatch | 0.00530186 | 3947:-0.387851 | token_mismatch | 0.012411 | 10920:-0.732595 |
| 9 | 7 | 3994 | -0.280066 | 2.375 | 1.79876 | 1.82882 | -0.0300655 | -0.049057 | -0.161947 | 0.0272599 | 0.0944397 | token_mismatch | -0.000729198 | 2521:0.102803 | token_mismatch | 0.00999936 | 8739:-0.403225 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 3994 | attention_input_normed | token_mismatch | -0.0676705 | 0.0580158 | -8.18546 | 3994:-8.18546 | - |
| 7 | 7 | 3994 | attention_k_normed | token_mismatch | -0.566755 | -0.0695678 | 8.20911 | - | - |
| 7 | 7 | 3994 | attention_k_projected | token_mismatch | 0.349204 | 1.85538 | 26.8533 | 669:13.2793 | - |
| 7 | 7 | 3994 | attention_projection_input | token_mismatch | 0.0177234 | 0.066478 | 3.44614 | - | - |
| 7 | 7 | 3994 | attention_q_gate | token_mismatch | -0.546451 | -0.526188 | -1.86255 | - | - |
| 7 | 7 | 3994 | attention_q_normed | token_mismatch | 0.151242 | 0.18594 | 1.99351 | 274:1.69977 | - |
| 7 | 7 | 3994 | attention_q_query | token_mismatch | -0.0768809 | -0.228441 | -5.52887 | 2327:-8.29772 | - |
| 7 | 7 | 3994 | attention_v_projected | token_mismatch | 0.0412168 | 0.87202 | 23.5126 | - | - |
| 8 | 7 | 3994 | attention_a_projection | token_mismatch | -0.292483 | -0.207221 | -0.0836582 | 22:1.13638 | 13:1.33224 |
| 8 | 7 | 3994 | attention_b_projection | token_mismatch | -0.202305 | -0.153527 | 0.0445864 | 9:0.460284 | 13:0.676684 |
| 8 | 7 | 3994 | attention_beta | token_mismatch | 0.022851 | 0.00792162 | 0.00319147 | 4:-0.0689108 | 13:0.150558 |
| 8 | 7 | 3994 | attention_conv | token_mismatch | 0.00863476 | 0.0421327 | 3.19297 | 6722:3.95909 | 8043:0.211598 |
| 8 | 7 | 3994 | attention_conv_pre_silu | token_mismatch | 0.0163223 | 0.0349827 | 0.902401 | 1020:-1.39937 | 8043:0.193281 |
| 8 | 7 | 3994 | attention_gate | token_mismatch | 0.0423281 | 0.104149 | 0.507035 | 30:-0.507035 | 30:-0.507035 |
| 8 | 7 | 3994 | attention_gate_silu | token_mismatch | 0.0499737 | 0.0837347 | 0.343801 | 3874:1.47654 | 3947:-0.0951583 |
| 8 | 7 | 3994 | attention_input_normed | token_mismatch | 0.0663373 | 0.0334668 | -4.9205 | 3994:-4.9205 | 3947:-1.16896 |
| 8 | 7 | 3994 | attention_pre_gate_normed | token_mismatch | 0.00378855 | 0.0136473 | 0.137743 | 4071:0.816185 | 4071:0.816185 |
| 8 | 7 | 3994 | attention_projection_input | token_mismatch | 0.00530186 | 0.00262993 | -1.56228 | 3947:-0.387851 | 3947:-0.387851 |
| 8 | 7 | 3994 | attention_qkv_projection | token_mismatch | 0.164039 | 0.223076 | 1.08517 | 3587:3.39767 | 8167:0.868416 |
| 8 | 7 | 3994 | attention_recurrent | token_mismatch | 2.44446e-05 | 0.000224941 | 0.0176641 | 3947:0.0176641 | 3947:0.0176641 |
| 8 | 7 | 3994 | attention_recurrent_k | token_mismatch | -0.00381013 | 1.87263e-07 | 0.00505936 | 845:0.0664775 | 2023:0.238069 |
| 8 | 7 | 3994 | attention_recurrent_q | token_mismatch | 0.000454903 | 5.57386e-09 | -0.0148673 | 1342:-0.0148673 | 2027:-0.00191458 |
| 8 | 7 | 3994 | attention_recurrent_v | token_mismatch | 0.00666792 | 0.0594114 | 3.19297 | 2626:3.95909 | 3947:0.211598 |
| 8 | 7 | 3994 | attention_z_projection | token_mismatch | 0.14382 | 0.156307 | 0.0289888 | 2437:-0.0706902 | 3947:-0.697901 |
| 8 | 7 | 3994 | mlp_activation | token_mismatch | 0.012411 | 0.0181702 | 0.732595 | 10920:-0.732595 | 10920:-0.732595 |
| 9 | 7 | 3994 | attention_a_projection | token_mismatch | -0.743948 | -0.807637 | -1.29125 | 16:1.49844 | 26:1.1883 |
| 9 | 7 | 3994 | attention_b_projection | token_mismatch | -0.0683699 | -0.00787946 | 0.652158 | 8:0.91523 | 19:0.363184 |
| 9 | 7 | 3994 | attention_beta | token_mismatch | 0.0182808 | 0.0195075 | 0.0758364 | 13:-0.190741 | 19:0.0656174 |
| 9 | 7 | 3994 | attention_conv | token_mismatch | 0.00689933 | 0.0126814 | 0.0639768 | 383:0.179232 | 6617:0.164618 |
| 9 | 7 | 3994 | attention_conv_pre_silu | token_mismatch | 0.0147722 | 0.028314 | 0.521286 | 1384:1.51238 | 6617:0.164126 |
| 9 | 7 | 3994 | attention_gate | token_mismatch | -0.00579803 | -0.0910661 | -0.53783 | 18:0.53783 | 19:-0.00332541 |
| 9 | 7 | 3994 | attention_gate_silu | token_mismatch | 0.0813838 | 0.125869 | 0.233288 | 3659:0.445806 | 2521:-0.00687549 |
| 9 | 7 | 3994 | attention_input_normed | token_mismatch | 0.0578791 | 0.0240783 | -5.22913 | 3994:-5.22913 | 2521:-0.894473 |
| 9 | 7 | 3994 | attention_pre_gate_normed | token_mismatch | -0.00922297 | 0.00139655 | -0.614662 | 4089:2.10507 | 2521:-0.614662 |
| 9 | 7 | 3994 | attention_projection_input | token_mismatch | -0.000729198 | -0.00158268 | -0.102803 | 2521:0.102803 | 2521:0.102803 |
| 9 | 7 | 3994 | attention_qkv_projection | token_mismatch | 0.13757 | 0.177547 | -0.537203 | 2305:2.94388 | 6617:0.572477 |
| 9 | 7 | 3994 | attention_recurrent | token_mismatch | -7.98734e-06 | 1.19299e-05 | -0.00175625 | 4089:0.00454055 | 2521:-0.00175625 |
| 9 | 7 | 3994 | attention_recurrent_k | token_mismatch | -0.00132716 | 1.50125e-07 | -0.229576 | 1529:0.111826 | 1241:0.0516094 |
| 9 | 7 | 3994 | attention_recurrent_q | token_mismatch | 0.000405893 | 2.11885e-09 | -0.0156726 | 1986:-0.0186446 | 1241:-0.00903236 |
| 9 | 7 | 3994 | attention_recurrent_v | token_mismatch | 0.00345257 | 0.00649563 | 0.0296435 | 1246:-0.356296 | 2521:0.164618 |
| 9 | 7 | 3994 | attention_z_projection | token_mismatch | 0.0884736 | 0.0512199 | 0.178098 | 262:0.551293 | 2521:0.0753896 |
| 9 | 7 | 3994 | mlp_activation | token_mismatch | 0.00999936 | 0.0126439 | 0.260558 | 8739:-0.403225 | 8739:-0.403225 |

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
| 10 | missing_full_reference | - | - | - |
| 11 | missing_full_reference | - | - | - |
