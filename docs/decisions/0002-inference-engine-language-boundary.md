# ADR 0002: Inference engine language boundary

## Status

Accepted for v0.1 implementation.

## Context

uLLM needs both a low-level runtime for model execution and a higher-level
control plane for request scheduling, telemetry, benchmarking, and future
prefill/decode disaggregation. The low-level runtime must eventually own HIP
C++ kernels, CPU kernels, and backend-specific memory management. The control
plane needs safer orchestration primitives and should evolve faster than the
kernel layer.

Existing engines split these concerns in different ways. uLLM will use them as
design references, but will not copy implementation code from `reference-src/`.

## Decision

- Rust owns the control plane: CLI, request lifecycle, scheduler, benchmark
  harness, telemetry, package metadata, and prefill/decode worker assignment.
- C++20 owns the runtime layer: model execution objects, backend abstraction,
  memory handles, stream/event handles, and calls into HIP/CPU kernels.
- The Rust/C++ boundary is a stable C ABI under `runtime/include/`.
- `crates/ullm-runtime-sys` provides the Rust FFI binding.
- `crates/ullm-engine` is the first user-facing control-plane binary.
- Python remains a reference/evaluation tool and is not in the production
  inference critical path.

## Consequences

- Runtime ABI changes require explicit versioning.
- The C ABI must avoid Rust-owned or C++-owned complex types crossing the
  boundary directly.
- The first implementation can ship a minimal runtime smoke path before model
  execution is complete.
- Future GPU backends can be added behind the same runtime interface.
