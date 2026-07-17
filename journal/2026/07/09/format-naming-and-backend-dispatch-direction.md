# Format naming and backend dispatch direction

Date: 2026-07-09

## User Direction

- Rename the existing AQ4 family to `AQ4_0`.
- Rename/adopt the existing FP8 E4M3 SQ work as `SQ8_0`.
- Interpret `4` and `8` as approximate bits per parameter, and `_0` as version 0.
- Treat `SQ8_0` as the officially adopted FP8 quantization format.
- Stop treating FP8 quality itself as an open research question. FP8 E4M3 is already broadly accepted; uLLM needs correct implementation and regression tests.
- Split very large Rust files where feasible, aiming for each file to stay under about 10,000 lines.
- Make C++ operator selection composable by model architecture, GPU architecture, and GPU name, with specific GPU overrides able to refine architecture defaults.

## Local File Size Baseline

Current largest source files:

- `crates/ullm-engine/src/main.rs`: about 48,478 lines.
- `crates/ullm-runtime-sys/src/lib.rs`: about 14,931 lines.
- `runtime/src/ullm_runtime.cpp`: about 14,670 lines.
- `crates/ullm-quant/src/main.rs`: about 6,081 lines.
- `runtime/src/ullm_runtime_hiprtc_sources.inc`: about 6,048 lines.

Primary split targets are therefore `ullm-engine/src/main.rs`, `ullm-runtime-sys/src/lib.rs`, and `runtime/src/ullm_runtime.cpp`.

## External Design Notes

Firecrawl was attempted first for web research but its local service was unavailable (`ECONNREFUSED 127.0.0.1:3002`), so browser search was used as fallback.

vLLM:

- Uses a platform plugin/current-platform mechanism for CUDA/ROCm/XPU/CPU detection.
- Platform interface exposes device capability/name/memory/CU queries and attention backend selection hooks.
- Model architecture support is routed through a model registry.
- Quantization is routed through a quantization config registry.
- FP8 is documented as a supported quantization format, not only as an experiment.

SGLang:

- Uses explicit attention backends with separate `forward_extend` and `forward_decode` requirements.
- Supports separate CLI selection for attention backends, including prefill attention backend in some paths.
- New model support is model-registry based and can also be external-package based.
- FP8 is documented as a supported quantization mode, including `modelopt_fp8` and torchao FP8 modes.
- Hardware/platform logic exists but is scattered enough that SGLang has open work/RFCs toward hardware plugin or abstraction layers.

## uLLM Implications

1. Format naming should become stable product naming:
   - internal legacy strings may remain as aliases temporarily;
   - new docs/results should prefer `AQ4_0` and `SQ8_0`;
   - `sq-fp8-w8a16-r9700-v0` should become an implementation/artifact lineage under `SQ8_0`, not the public format name.

2. Guard semantics should change:
   - strict AQ4-vs-FP8 logit/top1 equality is a drift diagnostic only;
   - correctness should mean the SQ8_0 artifact is decoded and executed according to its format spec;
   - cross-device and exact-path tests should catch implementation regressions.

3. Source splitting should be staged:
   - first extract CLI command dispatch and smoke families from `main.rs`;
   - then split `ullm-runtime-sys/src/lib.rs` by API family while preserving the public wrapper surface;
   - then split `runtime/src/ullm_runtime.cpp` into runtime core, HIP/HIPRTC loader, kernel cache/launcher families, host fallback, and C ABI families.

4. C++/Rust backend dispatch should use a registry/scoring model rather than ad hoc branching:
   - identify `FormatId`, `ModelArch`, `GpuVendor`, `GpuArch`, `GpuName`, `OperationKind`, and `Phase` (`prefill`, `decode`, `materialize`, etc.);
   - register operator implementations with match predicates and priorities;
   - choose the most specific supported implementation, with env/CLI override for debugging.

## Suggested First Implementation Slice

1. Add format identifiers and alias handling for `AQ4_0` and `SQ8_0` in docs/tools/runtime metadata.
2. Reframe SQ FP8 docs from candidate-quality promotion to SQ8_0 implementation validation.
3. Extract `ullm-engine/src/main.rs` command dispatch into a smaller CLI module without behavior change.
4. Introduce a small backend selection data model on the Rust side, initially used only for runtime/cached-prefix attention executor naming.
5. Mirror that selection into C++ only after the Rust boundary is stable.
