# `SQ8_0` decode logical-bandwidth efficiency

## Metric definition

This is a KV-inclusive **logical streaming lower-bound** metric. It normalizes a real, unprofiled decode timing run, but it is not a claim that every counted byte crossed HBM or that uncounted traffic is free.

For a decode step with context length `C`, define

```text
B_SQ8_0(C) = B_projection_payload + B_projection_BF16_scales
             + B_LM_head_BF16 + B_KV_read(C) + B_KV_write

B_KV_read(C) = 40 layers * 8 KV heads * (128 K + 128 V) * 4 B * C
             = 327,680 * C B
B_KV_write   = 327,680 B

TPS_roof(C)  = 640,000,000,000 B/s / B_SQ8_0(C)
eta_logical(C) = TPS_measured / TPS_roof(C)
```

The `640 GB/s` decimal peak is the R9700 reference point in `docs/reference/gpu-architecture-capabilities-rocm7.2.1.md`. Physical HBM/TCC byte counters were not captured, so physical-HBM efficiency is **未確認**.

## Measured denominator and result

The selected decode timing has cache length `1028 -> 1044`. To avoid assuming whether each call reads cache before or after its corresponding write, this result declares `C = (1028 + 1044) / 2 = 1036` as the fixed midpoint convention.

| logical term | bytes / generated token | evidence |
|---|---:|---|
| 280 `SQ8_0` projection F8 payloads | 13,212,057,600 | `sq_manifest.json`: 280 quantized tensors, sum of `weight.bytes` |
| 280 BF16 `[128,128]` projection-scale grids | 1,612,800 | manifest: sum of `scale.bytes`; all scales are BF16 block-2D |
| BF16 LM head | 1,555,824,640 | manifest: `777,912,320` BF16 elements; trace has one LM-head launch/step |
| KV read at `C=1036` | 339,476,480 | formula above |
| KV write | 327,680 | formula above |
| **`B_SQ8_0(1036)`** | **15,109,299,200** | declared lower-bound stream |

The five-repeat, unprofiled measured mean is `15.294955751 tok/s` (`65.381032563 ms/token`). Therefore:

```text
TPS_roof(1036) = 42.358020152 tok/s
logical stream rate = 231.096063 GB/s
eta_logical(1036) = 0.361087598 = 36.1088%
```

Projection launch counts prove that the full 280-projection schedule was selected. They do not prove that each logical byte was a distinct DRAM transaction: L2 reuse, activation/output/workspace traffic, page-table accesses, copies, and launch overhead are outside this denominator. Future comparisons must keep this exact accounting policy and report hardware counters separately when available.
