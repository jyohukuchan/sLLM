| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7 | 3994 | 1.52773 | 1.875 | 3.40273 | 1.875 | 1.52773 | 0.0349335 | 1.28195 | -0.00482151 | 0.267753 | ok | 0.0139444 | 1225:1.04039 | ok | 0.00309959 | 6591:-0.475168 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 4 | 7 | 3994 | attention_a_projection | ok | -0.00241787 | 0.00259859 | -0.126391 | 15:0.193229 |
| 4 | 7 | 3994 | attention_b_projection | ok | 0.0135055 | 0.0310979 | 0.364756 | 3:-0.364756 |
| 4 | 7 | 3994 | attention_beta | ok | 0.00139699 | 0.00171442 | -0.00527185 | 29:-0.0162248 |
| 4 | 7 | 3994 | attention_conv | ok | 0.0263345 | 0.0812561 | 2.93148 | 1702:0.129484 |
| 4 | 7 | 3994 | attention_gate | ok | 0.000896066 | -0.00306282 | -0.107192 | 18:0.155131 |
| 4 | 7 | 3994 | attention_input_normed | ok | -1.25744e-07 | -0.000865553 | -0.0876617 | 3994:-0.0876617 |
| 4 | 7 | 3994 | attention_projection_input | ok | 0.0139444 | 0.0287327 | 0.505239 | 1225:1.04039 |
| 4 | 7 | 3994 | attention_qkv_projection | ok | -0.000636438 | -0.00137943 | -0.0989151 | 1702:-0.354998 |
| 4 | 7 | 3994 | attention_recurrent | ok | 0.000100686 | 0.00025177 | 0.00536845 | 662:0.00910269 |
| 4 | 7 | 3994 | attention_z_projection | ok | -0.0030318 | -0.00490634 | -0.245604 | 3253:0.245604 |
| 4 | 7 | 3994 | mlp_activation | ok | 0.00309959 | 0.00586172 | 0.213044 | 6591:-0.475168 |
