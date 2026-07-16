# AQ4 multi-layer accumulation growth curve

| layer | kind | relative L2 | cosine | max abs | records |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | linear_attention | 0.042451 | 0.999107 | 0.069627 | 9 |
| 1 | linear_attention | 0.075076 | 0.997375 | 0.174330 | 9 |
| 2 | linear_attention | 0.092594 | 0.995869 | 0.253928 | 9 |
| 3 | self_attention | 0.106254 | 0.994378 | 0.202241 | 9 |

Shape: `approximately_linear_or_sublinear`; selected model: `linear`.
Layer-31 extrapolation: `0.850029` vs observed production final `0.615000` (138.2%); verdict: `explains`.
Linear extrapolation: `0.850029`.
Geometric extrapolation: `556.1975589876293` (mean ratio `1.357742241876362`).

This is a CPU-only diagnostic extrapolation, not a production-path or GPU-kernel measurement.
