| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | -1.04099 | 22.75 | 21.709 | 22.75 | -1.04099 | -0.169057 | -0.20964 | -0.613533 | -0.148687 | ok | -1.86622e-06 | 1673:0.549463 | ok | 0.000303071 | 5301:0.229916 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 3456 | attention_a_projection | ok | -0.0289494 | -0.033292 | -0.027617 | 7:-0.243847 | 22:-0.106326 |
| 10 | 0 | 3456 | attention_b_projection | ok | -0.0158412 | -0.0194576 | 0.0683823 | 22:-0.315084 | 22:-0.315084 |
| 10 | 0 | 3456 | attention_beta | ok | -0.000853742 | -0.000813786 | 0.000112236 | 22:-0.00836712 | 22:-0.00836712 |
| 10 | 0 | 3456 | attention_conv | ok | -5.79051e-06 | -0.000157488 | -0.0104375 | 3221:-0.0701092 | 6968:-0.0102951 |
| 10 | 0 | 3456 | attention_conv_pre_silu | ok | -1.32584e-05 | 0.000364353 | 0.0527387 | 6507:-0.0748687 | 6968:-0.010669 |
| 10 | 0 | 3456 | attention_gate | ok | -0.00540078 | -0.00877608 | -0.0063051 | 27:0.117953 | 13:0.00974929 |
| 10 | 0 | 3456 | attention_gate_silu | ok | -0.00192352 | -0.0050733 | -0.0307159 | 3561:0.0815144 | 1759:-0.132616 |
| 10 | 0 | 3456 | attention_input_normed | ok | 3.71349e-05 | 0.000478762 | 0.0561371 | 3456:0.0561371 | 3124:-0.00194448 |
| 10 | 0 | 3456 | attention_pre_gate_normed | ok | 0.000509337 | 0.000288893 | 0.0113459 | 2079:-0.211968 | 1673:0.0530081 |
| 10 | 0 | 3456 | attention_projection_input | ok | -1.86622e-06 | -0.00823942 | -0.549463 | 1673:0.549463 | 1673:0.549463 |
| 10 | 0 | 3456 | attention_qkv_projection | ok | -0.000604009 | -0.00671491 | -0.324604 | 7217:0.324604 | 7220:-0.115597 |
| 10 | 0 | 3456 | attention_recurrent | ok | 2.14881e-07 | 1.51825e-07 | 0.000268942 | 3121:0.000377865 | 1673:0.000371125 |
| 10 | 0 | 3456 | attention_recurrent_k | ok | 0.000112319 | 6.77174e-09 | 0.00712776 | 1173:-0.0423457 | 1597:-0.00307349 |
| 10 | 0 | 3456 | attention_recurrent_q | ok | -2.62993e-07 | -6.77878e-11 | -0.000288472 | 278:0.00554108 | 872:-0.000431686 |
| 10 | 0 | 3456 | attention_recurrent_v | ok | -1.53565e-05 | -0.000215321 | -0.0104375 | 3595:-0.0104375 | 2872:-0.0102951 |
| 10 | 0 | 3456 | attention_z_projection | ok | -0.00405939 | -0.00644201 | -0.0306892 | 774:0.0737844 | 1759:-0.122498 |
| 10 | 0 | 3456 | mlp_activation | ok | 0.000303071 | -0.00220681 | -0.229916 | 5301:0.229916 | 5301:0.229916 |

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
| 11 | missing_full_reference | - | - | - |
