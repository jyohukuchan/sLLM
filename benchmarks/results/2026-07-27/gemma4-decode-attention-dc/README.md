# Gemma4 decode-attention dispatch, session DC

This directory is an append-only record for the Gemma4 E2B resident-BF16
decode-attention investigation on the R9700 (`gfx1201`).

## Baseline command

```sh
# HIP device ordinal 1 maps to the sole gfx1201 R9700; physical amd-smi index 2.
HIP_VISIBLE_DEVICES=1 \
ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL=1 \
ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL=1 \
ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL=1 \
target/release/ullm-gemma4-resident \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B \
  --output benchmarks/results/2026-07-27/gemma4-decode-attention-dc/baseline-resident.json \
  --mode benchmark --benchmark-repeats 3
```

The driver uses the fixed six-token BL capital-France prompt
`[2,818,5279,529,7001,563]`, resets context for each timed prefill, and times
four decode tokens after an untimed six-token prefill for each decode repeat.
The JSON record contains the timings, generated IDs, context lengths, selected
device identity, and preflight evidence.

Observed baseline result: 17.2248 prefill tok/s (18 tokens / 1.045004219 s)
and 15.5647 decode tok/s (12 tokens / 0.770975161 s).  The one-repeat
kernel-trace companion is `baseline-kernel-trace/homelab1-WRX80-Creator/3317421_kernel_trace.csv`.
It records 770 launches of `ullm_paged_decode_attn_f32_kernel`; it records no
`ullm_paged_decode_attn_split_partial_f32_kernel` or split merge launch.  Thus
the baseline uses the direct generic F32 paged-decode body, not the split body.
