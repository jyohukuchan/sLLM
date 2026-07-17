# AQ4 multi-layer accumulation growth curve

| stage | kind | scope | relative L2 | cosine | max abs | records |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| layer 0 | linear_attention | full hidden | 0.042451 | 0.999107 | 0.069627 | 9 |
| layer 1 | linear_attention | full hidden | 0.075076 | 0.997375 | 0.174330 | 9 |
| layer 2 | linear_attention | full hidden | 0.092594 | 0.995869 | 0.253928 | 9 |
| layer 3 | self_attention | full hidden | 0.106254 | 0.994378 | 0.202241 | 9 |
| layer 4 | linear_attention | full hidden | 0.119419 | 0.992886 | 0.466560 | 9 |
| layer 5 | linear_attention | full hidden | 0.125536 | 0.992172 | 0.557333 | 9 |
| layer 6 | linear_attention | full hidden | 0.077143 | 0.997134 | 1.431293 | 9 |
| layer 7 | self_attention | full hidden | 0.094488 | 0.995626 | 1.429813 | 9 |
| layer 8 | linear_attention | full hidden | 0.094775 | 0.995630 | 1.403173 | 9 |
| layer 9 | linear_attention | full hidden | 0.092623 | 0.995813 | 1.345047 | 9 |
| layer 10 | linear_attention | full hidden | 0.074961 | 0.997391 | 2.475082 | 9 |
| layer 11 | self_attention | full hidden | 0.080827 | 0.996919 | 2.402580 | 9 |
| layer 12 | linear_attention | full hidden | 0.082044 | 0.996822 | 2.437775 | 9 |
| layer 13 | linear_attention | full hidden | 0.080715 | 0.996935 | 2.471966 | 9 |
| layer 14 | linear_attention | full hidden | 0.077336 | 0.997172 | 2.444794 | 9 |
| layer 15 | self_attention | full hidden | 0.086443 | 0.996394 | 2.267082 | 9 |
| layer 16 | linear_attention | full hidden | 0.090945 | 0.995938 | 2.238621 | 9 |
| layer 17 | linear_attention | full hidden | 0.096662 | 0.995381 | 2.109135 | 9 |
| layer 18 | linear_attention | full hidden | 0.086750 | 0.996378 | 2.433884 | 9 |
| layer 19 | self_attention | full hidden | 0.141563 | 0.989972 | 2.836046 | 9 |
| layer 20 | linear_attention | full hidden | 0.148977 | 0.988851 | 3.186935 | 9 |
| layer 21 | linear_attention | full hidden | 0.150603 | 0.988610 | 3.164642 | 9 |
| layer 22 | linear_attention | full hidden | 0.121062 | 0.992646 | 14.237411 | 9 |
| layer 23 | self_attention | full hidden | 0.147672 | 0.989042 | 7.387459 | 9 |
| layer 24 | linear_attention | full hidden | 0.144614 | 0.989491 | 8.158051 | 9 |
| layer 25 | linear_attention | full hidden | 0.131568 | 0.991308 | 7.411285 | 9 |
| layer 26 | linear_attention | full hidden | 0.082310 | 0.996785 | 25.694946 | 9 |
| layer 27 | self_attention | full hidden | 0.158523 | 0.987408 | 6.456627 | 9 |
| layer 28 | linear_attention | full hidden | 0.156894 | 0.987690 | 7.032410 | 9 |
| layer 29 | linear_attention | full hidden | 0.170875 | 0.985320 | 6.312180 | 9 |
| layer 30 | linear_attention | full hidden | 0.151532 | 0.988456 | 5.233986 | 9 |
| layer 31 | self_attention | full hidden | 0.127881 | 0.991806 | 15.799225 | 9 |
| final_norm | final_rmsnorm | full_hidden | 0.501033 | 0.965215 | 24.646279 | 9 |
| lm_head | aq4_lm_head_projection | fixed_logit_rows; rows 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,220,41330 | 0.586050 | 0.969689 | 8.347778 | 9 |

Shape: `nonmonotonic_or_layer_jump`; selected model: `observed_full_decoder_stack`.
Layer-31 observation: `0.127881` vs observed production final `0.615000` (20.8%); verdict: `does_not_explain`.
Linear extrapolation: `0.127881`.
Geometric extrapolation: `0.12788133069570853` (mean ratio `1.0362126277895845`).

Final norm is a full-hidden comparison. LM head is explicitly fixed-row sampled, not a full-vocabulary comparison.
This is a CPU-only diagnostic, not a production-path or GPU-kernel measurement.
