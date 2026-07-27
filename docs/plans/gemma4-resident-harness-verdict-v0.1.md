# Gemma4 resident harness verdict v0.1

## Verdict

The 81.062% Gemma4 decode wall-clock interval outside GPU kernels is **not**
an artifact of timing the non-resident diagnostic path.

Session DD's raw profiler stderr records this exact command:

```text
rocprofv3 --kernel-trace --runtime-trace \
  target/release/ullm-gemma4-resident \
  --model-dir /home/homelab1/datapool/ai_models/safetensors/gemma-4-E2B \
  --output benchmarks/results/2026-07-27/gemma4-moe-profile-v0.1/raw/gemma-benchmark.json \
  --mode benchmark --benchmark-repeats 3
```

The output JSON independently identifies its producer as
`ullm-gemma4-resident` and its model format as `source BF16 safetensors,
resident text decoder`. The binary invokes `Gemma4TextExecutor::load_resident`;
its measured decode path is therefore the same resident executor exposed to
the worker backend, not `ullm-gemma4-text-trace` or another host-weight
diagnostic.

Consequently the gap is a real property of the resident Gemma4 execution
path. The next step is host-side localisation inside that executor.

## Evidence

- Profile report and raw capture: commit `f5514bc4`,
  `benchmarks/results/2026-07-27/gemma4-moe-profile-v0.1/raw/gemma-benchmark.stderr`.
- Resident driver: `crates/ullm-engine/src/bin/ullm-gemma4-resident.rs`.
- Production worker backend: `crates/ullm-engine/src/gemma4_worker_backend.rs`.
