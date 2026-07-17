# AQ codebook-scope 100M split

## Context

これまでの実パッケージでは codebook は `family × quant_format` ごとに共有していた。Qwen3.5-9B p4p6 では量子化対象 `7,128,219,648` parameters に対して codebook は `12` 個だけで、平均 `594,018,304` parameters/codebook だった。MLP family では1つの codebook が `1,660,944,384` parameters を担当していた。

君の要望は、細かい調整ではなく機械的な分割で codebook の切り替え頻度を上げ、最大でもそれぞれ約1億 parameters にすること。

## Implementation

Rust `ullm-quant`:

- `TensorPlan` に optional `codebook_scope` を追加。
- `ModelPlan` に optional `codebook_scope_max_elements` を追加。
- CLI に `--aq-codebook-max-elements <N>` を追加。
- plan 生成時に、同じ `(family, quant_format)` の中で tensor 名順に累積し、次の tensor を足すと `N` を超える場合に次の `codebook_scope` へ切り替える。
- 既存 plan/export との互換性のため、`codebook_scope` が無い場合は従来通り `family` を codebook key として扱う。
- direct package と per-tensor prototype/merge の codebook lookup を `codebook_scope` 優先に変更。
- manifest の tensor/codebook entry に optional `codebook_scope` を追加。

Python `tools/export-aq-family-codebooks.py`:

- `--plan-json` を追加。
- plan がある場合は `codebook_scope` と tensor ごとの `quant_format` に従って codebook を export する。
- plan が無い場合は従来通り family 単位の export とする。

## 100M p4p6 Plan Result

Command:

```bash
cargo run -p ullm-quant -- \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --aq-policy p4p6 \
  --aq-codebook-max-elements 100000000 \
  --plan-output benchmarks/results/2026-07-04/aq/2026-07-04-ullm-quant-plan-qwen35-9b-p4p6-codebook100m.json
```

Result:

| item | value |
| --- | ---: |
| quantized parameters | 7,128,219,648 |
| codebook-scopes | 132 |
| max parameters/codebook-scope | 83,886,080 |
| average parameters/codebook-scope | 54,001,664 |

By family:

| family | scopes | parameters | max parameters/scope |
| --- | ---: | ---: | ---: |
| attn_k | 1 | 37,748,736 | 37,748,736 |
| attn_o | 2 | 150,994,944 | 83,886,080 |
| attn_q | 5 | 301,989,888 | 67,108,864 |
| attn_v | 1 | 37,748,736 | 37,748,736 |
| linear_attn_a | 1 | 3,145,728 | 3,145,728 |
| linear_attn_b | 1 | 3,145,728 | 3,145,728 |
| linear_attn_out | 5 | 402,653,184 | 83,886,080 |
| linear_attn_qkv | 12 | 805,306,368 | 67,108,864 |
| linear_attn_z | 5 | 402,653,184 | 83,886,080 |
| mlp_down | 33 | 1,660,944,384 | 50,331,648 |
| mlp_gate | 33 | 1,660,944,384 | 50,331,648 |
| mlp_up | 33 | 1,660,944,384 | 50,331,648 |

## Codebook Export

Command:

```bash
tools/export-aq-family-codebooks.py \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B \
  --plan-json benchmarks/results/2026-07-04/aq/2026-07-04-ullm-quant-plan-qwen35-9b-p4p6-codebook100m.json \
  --activation-stats benchmarks/results/2026-07-01/aq/activation-r9700-calib32-qwen35-9b-s512 \
  --weighted-codebook \
  --missing-activation-stats unweighted \
  --max-elements-per-tensor 262144 \
  --torch-threads 64 \
  --torch-interop-threads 1 \
  --output benchmarks/results/2026-07-04/aq/2026-07-04-aq-codebooks-qwen35-9b-p4p6-codebook100m-weighted.json
```

Result:

| item | value |
| --- | ---: |
| exported codebooks | 132 |
| weighted codebooks | 106 |
| unweighted fallback codebooks | 26 |
| fallback records | 103 |
| elapsed wall time | 0:26.78 |
| max RSS | 1,039,640 KiB |

Fallback は activation stats が無い tensor を含む codebook-scope に出ている。既存の p4p6 family export と同じく、missing stats は `unweighted_missing_activation_stats` として明示した。

## Direct Package Smoke

Command:

```bash
cargo run -p ullm-quant -- \
  --convert-plan-json benchmarks/results/2026-07-04/aq/2026-07-04-ullm-quant-plan-qwen35-9b-p4p6-codebook100m.json \
  --codebook-json benchmarks/results/2026-07-04/aq/2026-07-04-aq-codebooks-qwen35-9b-p4p6-codebook100m-weighted.json \
  --convert-package-output-dir /tmp/ullm-quant-scope100m-mlp-up2.ullm.d \
  --convert-package-summary-output benchmarks/results/2026-07-04/aq/2026-07-04-ullm-quant-direct-package-scope100m-mlp-up2.json \
  --convert-family mlp_up \
  --convert-max-tensors 2 \
  --convert-jobs 2 \
  --convert-verify \
  --convert-overwrite \
  --tensor-scale-estimator reservoir \
  --scale-window 4
```

Result:

| item | value |
| --- | ---: |
| selected tensors | 2 |
| package codebooks | 2 |
| failures | 0 |
| total file bytes | 56,626,952 |
| elapsed wall time | 0:22.04 |
| max RSS | 170,008 KiB |

Selected tensors:

| tensor | codebook-scope | relative MSE | old family-level relative MSE | ratio |
| --- | --- | ---: | ---: | ---: |
| layer0 `mlp.up_proj` | `mlp_up_s000` | 0.005148647296 | 0.005245252663 | 0.981582 |
| layer1 `mlp.up_proj` | `mlp_up_s001` | 0.005168949628 | 0.005250488442 | 0.984470 |

The smoke confirms that direct package conversion uses separate codebooks for separate codebook-scopes.

## Verification

- `python3 -m py_compile tools/export-aq-family-codebooks.py`
- `cargo test -p ullm-quant`
- `cargo build -p ullm-quant --release`

All passed.

## Notes

This change only increases codebook switching frequency. It does not change local-scale block-size, codebook-index bit width, tensor-scale, or payload layout. The `.ullm.d` manifest remains backward-compatible: if `codebook_scope` is absent, readers use `family` as before.
