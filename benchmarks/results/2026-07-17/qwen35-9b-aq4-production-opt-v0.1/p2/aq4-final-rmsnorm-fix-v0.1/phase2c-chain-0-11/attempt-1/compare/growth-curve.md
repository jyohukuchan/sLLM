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

Shape: `nonmonotonic_or_layer_jump`; selected model: `linear_conservative`.
Layer-31 extrapolation: `0.215539` vs observed production final `0.615000` (35.0%); verdict: `partially_explains`.
Linear extrapolation: `0.215539`.
Geometric extrapolation: `0.26063783435335297` (mean ratio `1.0602885020036263`).

Final norm is a full-hidden comparison. LM head is explicitly fixed-row sampled, not a full-vocabulary comparison.
This is a CPU-only diagnostic, not a production-path or GPU-kernel measurement.
