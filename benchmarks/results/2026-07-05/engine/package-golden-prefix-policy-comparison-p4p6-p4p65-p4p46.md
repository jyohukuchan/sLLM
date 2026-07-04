| layer | kind | p4p6_max_abs | p4p65_max_abs | p4p46_max_abs | p4p6_cos | p4p65_cos | p4p46_cos | p4p46_hot |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 0 | linear_attention | 14.151 | 14.2385 | 13.9422 | 0.708968 | 0.704322 | 0.710566 | t1 h3994 |
| 1 | linear_attention | 1.87504 | 1.81335 | 1.89757 | 0.989375 | 0.989455 | 0.989534 | t0 h3994 |
| 2 | linear_attention | 6.99691 | 7.19946 | 7.25549 | 0.962869 | 0.959951 | 0.960071 | t0 h3994 |
| 3 | self_attention | 6.99446 | 6.99446 | 6.98854 | 0.903654 | 0.903654 | 0.903647 | t3 h3994 |
| 4 | linear_attention | 1.52773 | 1.48379 | 1.46616 | 0.989743 | 0.990084 | 0.99012 | t7 h3994 |
| 5 | linear_attention | 3.39466 | 3.38895 | 3.35409 | 0.940972 | 0.942744 | 0.943553 | t0 h3456 |
| 6 | linear_attention | 21.9872 | 22.265 | 22.6322 | 0.960706 | 0.960836 | 0.960039 | t0 h3456 |
| 7 | self_attention | 1.81278 | 1.81278 | 1.78135 | 0.982205 | 0.982205 | 0.982399 | t6 h310 |

## Layer 6 Hot Vector

| policy | out_diff | attention_activation_path | mlp_activation_path | attention_projection_input_abs_mean_err | attention_projection_input_rms_err | attention_projection_input_max_abs_err | recurrent_max_abs_err | qkv_max_abs_err | z_max_abs_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| p4p6 | 21.9872 | 20.6754 | 1.96916 | 0.0335195 | 0.89666 | 57.3472 | 0.0652994 | -0.301666 | -0.111973 |
| p4p65-inproj | 22.265 | 20.8898 | 2.03253 | 0.0337507 | 0.904866 | 57.8818 | 0.0651251 | -0.308559 | -0.129066 |
| p4p46-inproj | 22.6322 | 21.2335 | 2.05607 | 0.0333317 | 0.921025 | 58.9123 | 0.0618228 | -0.228619 | -0.076355 |
