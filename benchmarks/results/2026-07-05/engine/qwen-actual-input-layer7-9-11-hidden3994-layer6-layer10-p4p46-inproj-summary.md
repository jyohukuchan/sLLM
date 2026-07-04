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
- `qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.json`
- `qwen-self-attention-propagation-layer7-actual-input-token8-feature503-hidden3994-p4p46-inproj.md`
- `package-golden-prefix-cpu-actual-prefix-layer0-8-causal-attn-diag-layer7-p4p46-inproj.jsonl`

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

## Layer 7 feature 503 replay

`tools/analyze-qwen-self-attention-propagation.py` was extended to replay the
same layer input with source q/k/v and dequantized package q/k/v, then emit
per-feature stage values.

For layer `7`, token `8`, feature `503`:

| stage | source | package replay | diff |
| --- | ---: | ---: | ---: |
| query_projection | -0.005279541 | 0.083527893 | 0.088807434 |
| gate_projection | 0.458984375 | 0.501503110 | 0.042518735 |
| key_projection | -0.621093750 | -0.613185644 | 0.007908106 |
| value_projection | -1.773437500 | -1.844769835 | -0.071332335 |
| query_normed | -0.006134033 | 0.097715974 | 0.103850007 |
| key_normed | -0.785156250 | -0.774710417 | 0.010445833 |
| raw_attention | 1.023437500 | 1.011238456 | -0.012199044 |
| gate_sigmoid | 0.613281250 | 0.622812510 | 0.009531260 |
| o_input | 0.628906250 | 0.629811943 | 0.000905693 |

This replay exactly matches the source layer hook
(`source_o_input_replay_vs_layer_hook.max_abs = 0`). With dequantized package
q/k/v, the feature `503` gated `o_proj` input remains close to source
(`+0.000905693`). That does **not** reproduce the actual package runtime
`attention_projection_input` value from the JSONL (`1.128042817`, diff
`0.499136567`).

This narrows the next target further: the mismatch is not explained by
dequantized q/k/v replay alone. The next diagnostic should compare the package
runtime causal attention output against a pure host reference on the exact
layer `7` actual-prefix q/k/v tensors, or dump the full package prepared q/k/v
and raw attention vectors for that token.

## Layer 7 Rust causal attention diagnostic

`package-golden-prefix-smoke` now emits
`causal_attention_runtime_diagnostic` for self-attention layers. Re-running
layers `0..8` reproduces the prior layer `7` max diff (`0.756856918`), so this
uses the same accumulated actual-prefix condition as the earlier JSONL.

For layer `7`, token `8`, feature `503`:

| value | result |
| --- | ---: |
| prepared attention output | 1.810265899 |
| layer attention output | 1.810265899 |
| host causal attention output | 1.810265899 |
| q gate | 0.502882540 |
| sigmoid(q gate) | 0.623136520 |
| prepared projection input | 1.128042817 |
| layer projection input | 1.128042817 |
| host projection input | 1.128042817 |

The full-vector max diffs are all `0` for:

- prepared attention vs pure host causal attention
- layer paged attention vs pure host causal attention
- layer paged attention vs prepared causal attention
- layer projection input vs host projection input
- layer projection input vs prepared projection input
- layer projection input vs replayed sigmoid gate application

So the Rust runtime causal attention path and gate application reproduce the
package value exactly. The discrepancy is now between the Python dequantized
package replay (`o_input = 0.629811943`) and Rust's prepared q/k/v attention
path (`attention_projection_input = 1.128042817`), not between Rust paged
attention and a host reference.

Next useful target:

- Dump or compare Rust prepared `q_rope` / `k_rope` / `v_projected` against the
  Python replay tensors for layer `7`, token `8`, especially the head containing
  feature `503`. The likely remaining mismatch is tensor layout, RoPE placement,
  or q/k/v projection expansion in the Python replay.
