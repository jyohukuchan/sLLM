# Gemma4 E2B non-quantized architecture trace

## Scope

This is a diagnostic-only BF16 source-weight / F32 activation bring-up for
`google/gemma-4-E2B`. It is not an SQ8_0/AQ4_0 artifact, campaign, FP32
reference corpus, bit-exact gate, serving benchmark, or promotion result.

- Source directory: `/home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B`
- `config.json` SHA-256: `e5faef0dd1a8f2437f6010721146b85433eaa90e679ef011e803c7ffefae73b8`
- `model.safetensors` SHA-256: `76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57`
- HF reference: Transformers 5.12.1 / Torch 2.12.0+cpu, CPU threads=8, float32.
- Candidate: `Gemma4TextExecutor`, raw BF16 safetensors weights and F32
  activations, HIP R9700 identity `AMD Radeon Graphics` / `gfx1201` /
  compute 12.0.
- Candidate invocations set `ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1`; the
  runtime rejects host-staging fallback.

## Trace inventory

| Reference | Candidate | Inputs / generated IDs | Tensor count | Observed largest abs error |
| --- | --- | --- | ---: | ---: |
| `hf-fp32-token2-step1` | `ullm-bf16-f32-token2-step1` | `2` → `184` | 38 | `3.9100647e-5` |
| `hf-fp32-token2-step2-pre-shared-kv-fix` | `ullm-bf16-f32-token2-step2` | `2` → `184, 3910` | 76 | `3.2901764e-5` at decode step 1 |
| `hf-fp32-capital-france-step4` | `ullm-bf16-f32-capital-france-step4` | `2,818,5279,529,7001,563` → `9079,236761,108,818` | 152 | `1.0681152e-4` |
| `hf-fp32-once-upon-step4` | `ullm-bf16-f32-once-upon-step4` | `2,14946,3324,496,990,236764` → `528,496,1902,1298` | 152 | `1.1825562e-4` |

The second HF directory has a provisional name from checking whether the
existing trace tool preserved Gemma4 shared KV state over decode. The capture
completed successfully without a tool change, so it is valid evidence despite
that name.

`capital-france-continuation-08-*` extends the first prompt by four more
tokens. Both engines generated `5279,529,7001,563`; its 152 tensors are in
`capital-france-continuation-08-comparison.json`.

## How to read the comparisons

The JSON files are layer-localization evidence, not numerical acceptance
thresholds. They record all embedding, 35 decoder-layer, final-RMSNorm, and
soft-capped-logit tensors in F32. No first structural mismatch was observed:
the layerwise errors stay at ordinary F32 reduction/operation-order scale and
greedy IDs match for every recorded step.

The matched decoded examples are:

```text
The capital of France is Paris.

The

Once upon a time, in a world where
```

The first continuation repeats its prompt after more than four generated
tokens in both engines; that is the base checkpoint's greedy behavior rather
than a candidate-only corruption.
