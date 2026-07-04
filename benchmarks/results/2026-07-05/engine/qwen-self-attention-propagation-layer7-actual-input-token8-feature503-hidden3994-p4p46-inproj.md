| hidden | worst_input_token | input_error_source_o_row | worst_total_token | total_error_package_o_row |
| ---: | ---: | ---: | ---: | ---: |
| 3994 | 0 | -0.172235489 | 1 | -0.101672083 |

| stage | mse | mean_abs | max_abs |
| --- | ---: | ---: | ---: |
| source_o_input_replay_vs_layer_hook | 0 | 0 | 0 |
| package_q_projection_vs_source | 0.00774847922 | 0.067142907 | 0.779000759 |
| package_k_projection_vs_source | 0.00542517419 | 0.0576851755 | 0.381331086 |
| package_v_projection_vs_source | 0.00252606837 | 0.0397878324 | 0.314338684 |
| package_o_input_vs_source | 8.74197566e-05 | 0.00545038012 | 0.172676802 |

## Feature Traces

| token | feature | stage | source | package | diff |
| ---: | ---: | --- | ---: | ---: | ---: |
| 8 | 503 | query_projection | -0.00527954102 | 0.0835278928 | 0.0888074338 |
| 8 | 503 | gate_projection | 0.458984375 | 0.50150311 | 0.0425187349 |
| 8 | 503 | key_projection | -0.62109375 | -0.613185644 | 0.00790810585 |
| 8 | 503 | value_projection | -1.7734375 | -1.84476984 | -0.0713323355 |
| 8 | 503 | query_normed | -0.0061340332 | 0.0977159739 | 0.103850007 |
| 8 | 503 | key_normed | -0.78515625 | -0.774710417 | 0.0104458332 |
| 8 | 503 | query_rope | -0.0061340332 | 0.0977159739 | 0.103850007 |
| 8 | 503 | key_rope | -0.78515625 | -0.774710417 | 0.0104458332 |
| 8 | 503 | raw_attention | 1.0234375 | 1.01123846 | -0.0121990442 |
| 8 | 503 | gate_sigmoid | 0.61328125 | 0.62281251 | 0.00953125954 |
| 8 | 503 | o_input | 0.62890625 | 0.629811943 | 0.000905692577 |
