# Qwen actual-input trace for hidden 3994

Fixture:

- `benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16`

Package:

- `/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d`

Input dump:

- `/tmp/qwen35-9b-prefix0-12-layer6-layer10-actual-inputs-p4p46-inproj`

Generated artifacts:

- `package-golden-prefix-cpu-actual-prefix0-12-manifest-row-scale-layer6-layer10-p4p46-inproj-input-dump.jsonl`
- `qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.jsonl`
- `qwen-layer-module-trace-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`
- `qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.json`
- `qwen-module-trace-comparison-actual-input-layers7-9-11-hidden3994-layer6-layer10-p4p46-inproj.md`

## Local package error with actual-prefix inputs

This comparison feeds the same actual-prefix layer input into the package path
and the full-reference Qwen layer. The `delta_error` column is therefore local
package error for that layer/input, not inherited upstream drift.

| layer | token | package output diff | fullref delta | package delta | local delta error | attention row-only | attention activation-path | MLP row-only | MLP activation-path |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 8 | -0.756856918 | 1.719599724 | 1.087742805 | -0.631856918 | 0.006025539 | -1.085749765 | -0.035078970 | 0.414899407 |
| 8 | 12 | 0.675428391 | 2.329922676 | 2.505351067 | 0.175428391 | -0.024038778 | 0.129673032 | -0.024429226 | 0.052237593 |
| 9 | 3 | -0.417654037 | 1.733904037 | 1.816250000 | 0.082345963 | 0.022902974 | -0.145378160 | 0.080553811 | 0.028742901 |
| 11 | 6 | -0.889577866 | 1.375562668 | 1.360984802 | -0.014577866 | 0.065380225 | -0.119163659 | -0.042665088 | 0.066413937 |

## Interpretation

- Layer `11` is not the local source of the remaining hidden `3994` max error.
  Its package-vs-full-reference local delta error is only `-0.014577866`, while
  the package output diff is `-0.889577866`.
- Layer `7` is the strongest local package-error target under actual-prefix
  input. Its local delta error is `-0.631856918`.
- Layer `7` row-only error is tiny (`0.006025539`) but the attention
  activation-path error is large (`-1.085749765`). This points inside the
  self-attention activation path before `o_proj`, not at the final `o_proj`
  row itself.
- Layer `8` and layer `9` have smaller local errors. They still contribute to
  propagation, but they do not explain the remaining max as strongly as layer
  `7`.

## Layer 7 stage split

After adding self-attention stage aliases to the full-reference trace, layer `7`
token `8` shows the following package/full-reference stage differences:

| stage | abs_mean_err | rms_err | max_abs_err | sampled feature 503 diff |
| --- | ---: | ---: | ---: | ---: |
| attention_input_normed | -0.000434046 | -0.000033842 | 0.028762818 | 0.000284731 |
| attention_q_query | -0.002295107 | -0.004452199 | 0.130254745 | 0.091293752 |
| attention_q_gate | -0.013947085 | -0.015171049 | -0.171232224 | 0.043898165 |
| attention_q_normed | 0.001036406 | 0.001181325 | 0.397748947 | 0.106798269 |
| attention_k_projected | -0.000230737 | 0.003530975 | -0.010082245 | - |
| attention_k_normed | -0.006306366 | -0.002296827 | 0.304644108 | - |
| attention_v_projected | -0.001881816 | -0.004012927 | -0.129554272 | - |
| attention_projection_input | -0.001356650 | 0.000840115 | 0.194449067 | 0.499136567 |

The layer input after RMSNorm is effectively aligned. The larger feature `503`
drift appears after the attention composition/gating path: `attention_q_query`
and `attention_q_gate` differ, but not enough by themselves to explain the
`attention_projection_input` feature `503` jump from `0.62890625` to
`1.128042817`.

Next useful target:

- Trace layer `7` self-attention internals for token `8`, hidden `3994`,
  especially the causal attention output feeding `o_proj` input feature `503`.
  The full-reference trace currently exposes the gated `o_proj` input but not
  the raw pre-gate attention vector, so that raw vector should be the next dump.
