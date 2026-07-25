# `SQ8_0` R9700 handwritten-kernel Phase 0 evidence

Date: 2026-07-26
Scope: Qwen3-14B-FP8 independent `SQ8_0`, R9700 only. No production symbol, external ABI, dispatch boundary, `/opt/ullm` content, unit file, or activation file was changed.

## Device and run identity

| item | observed value |
|---|---|
| AMD SMI GPU | `2` |
| PCI BDF | `0000:47:00.0` |
| HIP architecture | `gfx1201` |
| CU count | `64` |
| ROCm | `7.2.1` |
| artifact manifest SHA-256 | `c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb` |
| valid-profile driver SHA-256 | `075a780837f9f124aa32ed152fd6316edbfc83286df691bf92c661d33d198444` |

`static/r9700-static.json` and `static/run-identity.txt` retain the machine and original-run captures. The latter predates the final ROCTx scope correction; the valid profile driver hash is the one in this table. No V620 (`gfx1030`) device was selected or profiled.

## Profile validity and raw inputs

`rocprofv3 --selected-regions` was armed with ROCTx pause/resume. The decode seed prefill, four decode warm-up steps, model load, finish, and reset are outside the selected range.

| phase | workload in selected range | result | raw evidence |
|---|---|---|---|
| Decode timing | 5 unprofiled repeats, 16 M=1 steps/repeat, cache length `1028 -> 1044` | mean `1.046096521 s`/16 steps = `15.294955751 tok/s` | `decode/timing.jsonl` |
| Decode kernel trace | one 16-step M=1 range with the same cache window | marker range `1.127416757 s`; kernel-time sum `962.034418 ms` | `decode/rocprof-attempt2/` and `decode/profile-driver-attempt2.jsonl` |
| Prefill kernel trace | one 1024-token prompt in M=128 chunks | marker range `3.800431442 s`; kernel-time sum `2.918473188 s` | `prefill/rocprof-attempt1/` |

The prefill capture is a profiler trace, not an unprofiled throughput benchmark; unprofiled prefill tok/s is **未確認**. Kernel share is calculated against the sum of dispatch durations in each corresponding raw kernel CSV, not against wall-clock range duration.

## Thermal, clock, and power record

Raw AMD SMI JSON is retained in `telemetry/`. For the unprofiled decode timing run, the before / sampled-load / after readings were respectively `37 C / 1015 MHz / 16 W`, `73 C / 3298 MHz / 250 W`, and `66 C / 3439 MHz / 123 W` (hotspot / GFX clock / socket-power field). The prefill isolation-window sampler reached a maximum recorded hotspot of `78 C` and GFX clock of `3420 MHz`; its maximum socket-power field was `411 W`. These are sampled telemetry values, not an assertion about the cause of an AMD SMI `THROTTLED` status string.

## Results map

- `kernel-attribution.md`: scoped decode/prefill kernel time attribution and CK-vs-handwritten boundary.
- `efficiency.md`: the KV-inclusive logical streaming denominator and measured decode normalization.
- `static/offline-codegen-metadata.md`: HIPRTC / code-object resource metadata, LDS-tree inventory, and load-width audit.
- `service-lifecycle-final.md`: stop/restore record and final service state.
- `decode/rocprof-attempt2/` and `prefill/rocprof-attempt1/`: unedited rocprofv3 CSV outputs.
