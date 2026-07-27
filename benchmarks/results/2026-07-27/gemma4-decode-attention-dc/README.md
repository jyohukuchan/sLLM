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

## Step 3 repeated measurement

The exact baseline command above was repeated with only `--output` changed to
`after-step2-resident.json`.  It measured 17.7499 prefill tok/s (18 tokens /
1.014087818 s) and 16.3721 decode tok/s (12 tokens / 0.732953555 s).  No
runtime code changed between the runs, so this small difference is normal run
variation, not an optimisation result.

## Qwen3.5 AQ4_0 production regression

The fixed 128-token production probe was run with `HIP_VISIBLE_DEVICES=1`,
`ULLM_HIP_VISIBLE_DEVICES=1`, and all `worker.required_environment` guards
from `/etc/ullm/served-models/active.json`.  The result in
`qwen35-aq4-regression-after-step2.json` is byte-identical (`sha256`
`30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`) to
the pre-session R9700 record
`benchmarks/results/2026-07-27/qwen35-moe-loader-wiring-v0.1/qwen35-9b-baseline-probe-ce-physical-262k.json`.
It has the same top-1 token/logit, `220` / `8.529029846191406`, and same full
top-10 JSON.  The probe binary SHA-256 was
`51934931f27c76e1c82d42986f0eb6981af6fed47eacfea48eeebcad071aebb5`.
