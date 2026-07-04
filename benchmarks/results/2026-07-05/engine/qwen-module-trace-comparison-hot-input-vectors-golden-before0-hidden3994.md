| layer | token | hidden | pkg_out_diff | expected_delta | pkg_delta | full_delta | delta_error | attn_row_only | attn_activation | mlp_row_only | mlp_activation | attn_hot_status | attn_hot_abs_mean_err | attn_hot_top1 | mlp_hot_status | mlp_hot_abs_mean_err | mlp_hot_top1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 3994 | 14.151 | 6.3186 | 20.4696 | 6.28735 | 14.1822 | 0.0236746 | 2.45946 | -0.0051128 | 11.6909 | ok | 0.021351 | 2404:3.59415 | ok | 0.0741986 | 3894:2.85943 |

## Hot Input Vector Stage Errors

| layer | token | hidden | stage | status | abs_mean_err | rms_err | max_abs_err | top1 |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 3 | 3994 | attention_a_projection | ok | 0.00346623 | 0.0121495 | -0.165411 | 12:-0.195509 |
| 0 | 3 | 3994 | attention_b_projection | ok | -0.00862351 | -0.00257795 | 0.106178 | 4:0.106178 |
| 0 | 3 | 3994 | attention_beta | ok | -0.000931914 | -0.00125853 | 0.000592172 | 25:-0.00376457 |
| 0 | 3 | 3994 | attention_conv | ok | 0.117312 | 0.158237 | -0.148081 | 7846:-0.196072 |
| 0 | 3 | 3994 | attention_gate | ok | -0.00335538 | -0.00766973 | -0.0471119 | 21:0.0471119 |
| 0 | 3 | 3994 | attention_input_normed | ok | -3.85503e-05 | -9.72455e-05 | -0.00583887 | 3279:-0.0132403 |
| 0 | 3 | 3994 | attention_projection_input | ok | 0.021351 | 0.14929 | 4.25088 | 2404:3.59415 |
| 0 | 3 | 3994 | attention_qkv_projection | ok | -0.00736859 | -0.0145725 | 0.150265 | 4264:0.497711 |
| 0 | 3 | 3994 | attention_recurrent | ok | 0.000144913 | -0.00454752 | -0.0991468 | 3615:-0.120138 |
| 0 | 3 | 3994 | attention_z_projection | ok | -0.0109078 | -0.0140739 | 0.170426 | 1767:0.235564 |
| 0 | 3 | 3994 | mlp_activation | ok | 0.0741986 | 0.296766 | 28.3077 | 3894:2.85943 |
