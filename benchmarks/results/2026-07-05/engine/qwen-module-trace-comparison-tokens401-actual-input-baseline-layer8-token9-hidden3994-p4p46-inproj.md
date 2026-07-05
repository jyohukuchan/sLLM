| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 9 | 3994 | -0.959307 | 1.25 | 0.755413 | 0.71472 | 0.0406933 | 0.0660108 | 0.141028 | 0.0163918 | -0.210903 | ok | -0.000462346 | 1986:-0.373372 | ok | -0.0011468 | 288:-0.182608 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 | sampled_top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 8 | 9 | 3994 | attention_a_projection | ok | -0.0020284 | -0.00340546 | -0.0342832 | 14:-0.191479 | 14:-0.191479 |
| 8 | 9 | 3994 | attention_b_projection | ok | 0.0210945 | 0.0095585 | 0.0103991 | 6:0.138943 | 6:0.138943 |
| 8 | 9 | 3994 | attention_beta | ok | -0.000931481 | -0.000133305 | 0.00121492 | 2:0.008856 | 6:0.0255914 |
| 8 | 9 | 3994 | attention_conv | ok | -9.68979e-05 | 0.000342997 | 0.0307716 | 8043:0.0307716 | 6082:-0.00712419 |
| 8 | 9 | 3994 | attention_conv_pre_silu | ok | -0.000433784 | -0.000712758 | 0.0493388 | 2135:-0.290638 | 6082:-0.0104151 |
| 8 | 9 | 3994 | attention_gate | ok | -0.00369648 | -0.0144206 | -0.080534 | 30:0.080534 | 12:0.0393345 |
| 8 | 9 | 3994 | attention_gate_silu | ok | -0.000743661 | -0.0024051 | -0.00671339 | 1099:-0.0981793 | 1831:0.125359 |
| 8 | 9 | 3994 | attention_input_normed | ok | 0.000300249 | -0.000202722 | -0.0425911 | 3994:-0.0425911 | 1582:0.00309622 |
| 8 | 9 | 3994 | attention_pre_gate_normed | ok | -0.00223974 | -0.00424553 | -0.293329 | 4071:-0.293329 | 1986:-0.085502 |
| 8 | 9 | 3994 | attention_projection_input | ok | -0.000462346 | -0.00384267 | -0.373372 | 1986:-0.373372 | 1986:-0.373372 |
| 8 | 9 | 3994 | attention_qkv_projection | ok | -0.00273187 | -0.00338462 | 0.122442 | 3053:-0.183343 | 6082:0.0922883 |
| 8 | 9 | 3994 | attention_recurrent | ok | -5.24228e-06 | -2.3101e-05 | -0.00122956 | 4071:-0.00137868 | 1986:-0.00023336 |
| 8 | 9 | 3994 | attention_recurrent_k | ok | -3.21681e-05 | -1.42914e-09 | 0.0018301 | 1434:-0.0250378 | 385:-0.010715 |
| 8 | 9 | 3994 | attention_recurrent_q | ok | -1.27894e-05 | 4.41864e-11 | 0.00089436 | 1910:-0.00158596 | 814:0.000497896 |
| 8 | 9 | 3994 | attention_recurrent_v | ok | -7.93228e-06 | 0.000888993 | 0.0307716 | 2620:0.204643 | 1986:-0.00712419 |
| 8 | 9 | 3994 | attention_z_projection | ok | -0.00289937 | -0.00521671 | -0.0786624 | 77:0.117658 | 1831:0.114279 |
| 8 | 9 | 3994 | mlp_activation | ok | -0.0011468 | -0.00273519 | -0.141102 | 288:-0.182608 | 288:-0.182608 |
| 8 | 9 | 3994 | mlp_gate_projection | ok | -0.00144043 | -0.00245975 | 0.0200715 | 8981:0.0932734 | 1161:-0.0911299 |
| 8 | 9 | 3994 | mlp_gate_silu | ok | -0.000940865 | -0.00206368 | -0.0686113 | 2540:-0.154525 | 1161:-0.0946784 |
| 8 | 9 | 3994 | mlp_up_projection | ok | -0.00485143 | -0.00606085 | -0.107363 | 4318:-0.107363 | 288:-0.067996 |

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
| 9 | missing_full_reference | - | - | - |
| 10 | missing_full_reference | - | - | - |
| 11 | missing_full_reference | - | - | - |
