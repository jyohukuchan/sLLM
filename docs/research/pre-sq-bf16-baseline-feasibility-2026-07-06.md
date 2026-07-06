# Pre-SQ BF16 Baseline Feasibility 2026-07-06

## 前回の要点

- pre-sq TPS基盤では、sq候補と比較するための基準線としてBF16/materialized-AQ baselineが必要だった。
- materialized-AQ baselineはR9700 `512/256` で完走し、decodeは `0.140 tok/s` だった。
- 同じ現行decode経路で長時間runを繰り返す価値は低いため、V620 materialized-AQ baselineは途中で意図的に停止した。

## 今回の変更点

- 既存 `.ullm.d` package一覧と対象package manifestを確認した。
- 対象packageは `255` quantized tensors、`520` passthrough tensorsであり、BF16/passthrough-only full decoder packageではないことを確認した。
- loaderはBF16 passthrough payloadを読めるが、既存関数はf32値へ展開する。decoder matrixはAQ4からf32 runtime bufferへmaterializeされる。

## 次の行動

- pre-sq範囲では真のBF16 baselineをdeferする。
- T4の短いreference guardを優先する。
- sq候補策定では、現行baselineを「materialized-AQ f32 residency lower bound」として扱う。

## Findings

Checked package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d
```

Manifest summary:

| item | value |
| --- | ---: |
| quantized tensors | 255 |
| passthrough tensors | 520 |
| `tensors/` bytes | 4072144896 |
| `passthrough/` bytes | 5049777120 |

Quantized tensor families:

| family | count |
| --- | ---: |
| `attn_k` | 9 |
| `attn_o` | 9 |
| `attn_q` | 9 |
| `attn_v` | 9 |
| `linear_attn_a` | 24 |
| `linear_attn_b` | 24 |
| `linear_attn_out` | 24 |
| `linear_attn_qkv` | 24 |
| `linear_attn_z` | 24 |
| `mlp_down` | 33 |
| `mlp_gate` | 33 |
| `mlp_up` | 33 |

Passthrough tensor families:

| family | count |
| --- | ---: |
| `embed` | 1 |
| `lm_head` | 1 |
| `other` | 518 |

Relevant runtime behavior:

- `read_named_passthrough_f32`, `read_named_passthrough_f32_rows`, and `read_named_passthrough_f32_row_range` read passthrough payloads and return f32 values.
- `resolve_passthrough_dtype` accepts `BF16`, `F16`, and `F32`, but the read path decodes into f32.
- `materialize_selected_aq4_matrix` selects quantized tensor payloads and dequantizes them into f32 runtime buffers.

## Decision

The current artifacts and runtime cannot produce a true BF16 baseline for Qwen3.5-9B full decoder inference. A true BF16 baseline would require:

- a full decoder package where the large decoder matrices are passthrough BF16 rather than AQ4 tensors;
- loader/runtime branches that select those passthrough decoder matrices;
- a clear decision whether the baseline means BF16 storage with f32 compute, or BF16 storage and BF16 compute;
- if BF16 compute is required, runtime kernels or buffer paths that preserve BF16 semantics.

For the current pre-sq stage, implementing that path is longer than the value it adds. The materialized-AQ f32-resident result should be used as the lower-bound baseline, and BF16 should be listed as a deferred comparison rather than a missing blocker.
