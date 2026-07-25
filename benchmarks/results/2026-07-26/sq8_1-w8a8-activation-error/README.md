# SQ8_1 W8A8 activation quantization error — CPU measurement

## Scope

- CPU-only real-Qwen3.5-9B forward pre-hook measurement; no HIP runtime API or GPU was used.
- The existing `tools/collect-activation-stats.py` loader, corpus parser, and Linear-module convention were reused.
- Raw activations were quantized in-process and discarded. This directory contains only aggregate/error evidence.
- `SQ8_1` activation rule: per-token, contiguous K block, symmetric signed int8, RNE codes, `s=max(abs(x))/127`, stored FP16 scale rounded upward to a representable value.

## Activation error

| K | scale | sampled values | relative L2 | max abs error | true clipping rate | edge-code rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 16 | float16_rne | 81788928 | 0.0072425009 | 0.36181641 | 0.031854899 | 0.062909689 |
| 16 | float16_ceil | 81788928 | 0.007251451 | 0.36181641 | 0 | 0.06284984 |
| 16 | bfloat16_rne | 81788928 | 0.0073565088 | 0.375 | 0.03602314 | 0.063048167 |
| 16 | bfloat16_ceil | 81788928 | 0.0074830633 | 0.375 | 0 | 0.037930525 |
| 32 | float16_rne | 81788928 | 0.0099361432 | 0.36181641 | 0.01594585 | 0.031488296 |
| 32 | float16_ceil | 81788928 | 0.0099441518 | 0.36181641 | 0 | 0.031464858 |
| 32 | bfloat16_rne | 81788928 | 0.010012222 | 0.375 | 0.018176861 | 0.031555799 |
| 32 | bfloat16_ceil | 81788928 | 0.010113692 | 0.375 | 0 | 0.018773678 |
| 64 | float16_rne | 81788928 | 0.013580917 | 0.36376953 | 0.0079851151 | 0.015749931 |
| 64 | float16_ceil | 81788928 | 0.013588756 | 0.36376953 | 0 | 0.015740541 |
| 64 | bfloat16_rne | 81788928 | 0.013632509 | 0.375 | 0.0091833335 | 0.015782637 |
| 64 | bfloat16_ceil | 81788928 | 0.013720901 | 0.375 | 0 | 0.0093523417 |
| 128 | float16_rne | 81788928 | 0.018316605 | 0.40087891 | 0.0039842801 | 0.0078723736 |
| 128 | float16_ceil | 81788928 | 0.018325496 | 0.40087891 | 0 | 0.0078690969 |
| 128 | bfloat16_rne | 81788928 | 0.018351799 | 0.40234375 | 0.0045269575 | 0.0078914716 |
| 128 | bfloat16_ceil | 81788928 | 0.018438588 | 0.40234375 | 0 | 0.0047391378 |

`true clipping` counts values outside the post-storage-scale [-127,127] range before clamp. `edge-code` is the fraction represented by ±127 and is intentionally reported separately. The `ceil` scale policy preserves scale positivity and avoids an RNE-down-rounding clip without changing scale bytes.

## Sampled linear-output error

Each selected Linear tensor uses deterministic evenly spaced raw-token and output-row samples. `W8A16` quantizes only weights; `W8A8` uses int32 block dots and applies the upward-rounded FP16 activation and weight scales once per K=32 partial.

| path | sampled outputs | relative L2 | max abs error |
| --- | ---: | ---: | ---: |
| activation_only | 253952 | 0.0064620244 | 0.082850218 |
| w8a16 | 253952 | 0.0042676649 | 0.054093957 |
| w8a8 | 253952 | 0.0077510931 | 0.10254812 |

## Activation-only logit smoke

- Relative L2: `0.01401899`; max abs: `0.4921875`; mean KL: `0.0003239545`.
- Top-1 matches: `16/16`.
- This is activation-only. Full-model `SQ8_1` W8A8 logits remain unmeasured here because no production weight reader/kernel was implemented.

## Reproduction

The exact command line, file hashes, model/corpus provenance, selected-row rule, thread settings, and tool hashes are in `measurement-manifest.json`.
