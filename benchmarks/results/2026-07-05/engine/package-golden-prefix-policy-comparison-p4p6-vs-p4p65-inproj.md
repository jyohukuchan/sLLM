| layer | kind | p4p6_max_abs | p4p65_max_abs | delta | p4p6_cos | p4p65_cos | p4p65_hot |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | linear_attention | 14.151 | 14.2385 | 0.0875034 | 0.708968 | 0.704322 | t1 h3994 |
| 1 | linear_attention | 1.87504 | 1.81335 | -0.0616865 | 0.989375 | 0.989455 | t0 h3994 |
| 2 | linear_attention | 6.99691 | 7.19946 | 0.202547 | 0.962869 | 0.959951 | t0 h3994 |
| 3 | self_attention | 6.99446 | 6.99446 | 0 | 0.903654 | 0.903654 | t3 h3994 |
| 4 | linear_attention | 1.52773 | 1.48379 | -0.0439377 | 0.989743 | 0.990084 | t7 h3994 |
| 5 | linear_attention | 3.39466 | 3.38895 | -0.00571585 | 0.940972 | 0.942744 | t0 h3456 |
| 6 | linear_attention | 21.9872 | 22.265 | 0.277809 | 0.960706 | 0.960836 | t0 h3456 |
| 7 | self_attention | 1.81278 | 1.81278 | 0 | 0.982205 | 0.982205 | t6 h310 |

## Layer 6 Hot Vector

| policy | out_diff | attention_activation_path | mlp_activation_path | attention_projection_input_abs_mean_err | attention_projection_input_rms_err | attention_projection_input_max_abs_err | mlp_activation_abs_mean_err |
|---|---:|---:|---:|---:|---:|---:|---:|
| p4p6 | 21.9872 | 20.6754 | 1.96916 | 0.0335195 | 0.89666 | 57.3472 | 0.00297999 |
| p4p65-inproj | 22.265 | 20.8898 | 2.03253 | 0.0337507 | 0.904866 | 57.8818 | 0.00308157 |
