# Measurement conditions

## Fixed timing boundary

The AQ4_0 rows use the same engine-loop accounting boundary as
[`../../2026-07-26/r9700-prefill-comparison/conditions.md`](../../2026-07-26/r9700-prefill-comparison/conditions.md): model load, request construction,
same-length warm-up, and finish/reset are excluded; each prefill row has one
unprofiled warm-up and five timed repeats. Prefill is single-stream M=128.
Decode pre-fills a 3,968-token prefix outside the timed interval, then times
128 greedy steps with the same warm-up/repeat count.

The model is deliberately different from the earlier cross-engine table:
these rows are the served-format AQ4_0 Qwen3.5-9B package, not SQ8_0
Qwen3-14B or llama.cpp. They therefore compare **KV dtype within the same
full model**, rather than claiming a cross-model absolute comparison.

## Native-kernel requirement

F16 and FP8 rows require all three variables below. The measurement binary
refuses to run a non-F32 dtype without them, so host/staging fallback cannot
produce one of the reported numbers.

```text
ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_KERNEL=1
ULLM_REQUIRE_HIP_TYPED_PAGED_DECODE_SPLIT_KERNEL=1
ULLM_REQUIRE_HIP_TYPED_PAGED_KV_WRITE_KERNEL=1
```

The physical device is R9700 / gfx1201 only, exposed as the one HIP device by
`HIP_VISIBLE_DEVICES=1`. The wrapper owns `/run/ullm/r9700.lock` for the whole
window and verifies the inactive/disabled legacy llama service before work.

## Thermal and service protocol

Each of the 26 conditions was gated at edge <=45 C. The condition that follows
a long run may start its cooling trace as high as 73 C, but execution begins
only after a recorded pass in the 37--45 C range. Raw AMD SMI JSON sequences
are under `run-20260727T021656+0900/thermal/`.

The wrapper stopped `ullm-openai.service` once, held the exclusive lock for all
AQ4/SQ8 work, then released the lock and recorded a successful service start
and active observation. Any later service/lock activity belongs to a separate
owner and is not part of this measurement window.
