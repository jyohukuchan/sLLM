# SQ8_1 V620 optimization evidence — 2026-07-26

This directory records the staged V620 (`gfx1030`) validation of the optimized
`SQ8_1` matvec kernels. The measured workload is the decode-like Qwen3-14B
`self_attn.q_proj` shape: 5,120 rows × 5,120 columns, M=1.

The scoped answer is **yes**: both optimized `SQ8_1` paths are faster than the
matched existing `SQ8_0` V620 fallback baseline. The three-run median is
0.237362 ms for `SQ8_1` W8A16 and 0.249762 ms for explicit W8A8, versus
0.639007 ms for `SQ8_0`.

`summary.md` is the human-readable result and `summary.json` is its compact
machine-readable form. Raw events and temperature samples are separate:

- `raw/final-m1-r{1,2,3}-card0.jsonl`: numerical gates, metadata, and the
  32-warmup/31-trial timing records for each independent run.
- `thermal/final-m1-r{1,2,3}-card0.jsonl`: every card0-junction sample taken
  before/after warmups and timed launches.
- `raw/v620-runtime-differential-k65.jsonl`: runtime-path tail differential
  (7 × 65, physical stride 80) for W8A16 and W8A8.
- `static-optimized/`: device-only HIPRTC-source compilation, ISA
  disassemblies, resource notes, analyzer JSON, and checksums for all runtime
  whitelist targets. `isa-comparison.md` compares gfx1030 with the committed
  reference artifact.

The benchmark selected the physical device only after obtaining its BDF with
`hipDeviceGetPCIBusId`. It accepted only `0000:03:00.0` / DRM `card0` and
resolved the junction sensor through that same device:
`/sys/class/drm/card0/device/hwmon/hwmon5/temp2_input`. `HIP_VISIBLE_DEVICES=2`
made that physical card HIP ordinal 0 for the isolated process; no R9700 run
was performed.

Rebuild and run without touching a production build tree:

```bash
tools/build-bench-sq8_1-v620-optimization.sh \
  "$PWD/benchmarks/results/2026-07-26/sq8_1-v620-optimization/build/bench-sq8_1-v620-optimization-hip"

HIP_VISIBLE_DEVICES=2 \
  benchmarks/results/2026-07-26/sq8_1-v620-optimization/build/bench-sq8_1-v620-optimization-hip \
  --pci-bus-id 0000:03:00.0 \
  --jsonl-output "$PWD/benchmarks/results/2026-07-26/sq8_1-v620-optimization/raw/new-run.jsonl" \
  --thermal-output "$PWD/benchmarks/results/2026-07-26/sq8_1-v620-optimization/thermal/new-run.jsonl"
```

The historical `SQ8_0` comparison records are intentionally referenced rather
than copied: `../sq9-v620-viability/raw/final-m1-r{1,2,3}-card0-v4.jsonl`.
They use the same shape, card, 32/31 protocol, 42 C cooldown target, and
512 GB/s modeled-efficiency convention. The formats were not co-dispatched
inside one process, so this is a matched-protocol comparison rather than a
single-process A/B trace.
