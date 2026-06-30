# Setup fast compile environment

## Done

- Installed Ninja, ccache, sccache, mold, clang, lld, rustup, libssl-dev, and related build packages.
- Installed Rust stable through rustup.
- Added repository-local Cargo config to use clang and mold.
- Added `tools/fast-build-env.sh`.
- Added `tools/check-compile-env.sh`.
- Added `docs/plans/fast-compile-environment-v0.1.md`.
- Set ccache max size to 50G.
- Started sccache server and set intended cache size to 50G through the repo environment script.
- Saved compile environment report to `journal/2026/06/30/compile-env-report.txt`.
- Ran Rust smoke build in `build/fast-compile-smoke/rust`.
- Confirmed Rust uses `sccache`, `clang`, and `mold`.
- Configured llama.cpp HIP with Ninja and ccache in `build/reference/llama.cpp-hip-ninja`.
- Built `ggml-base` with Ninja and ccache.
- Rebuilt `ggml-base` after clean and confirmed 9 direct ccache hits.
- Saved Rust and C++ smoke logs under `journal/2026/06/30/`.
- Saved updated reference environment report with Rust available.

## Notes

- The machine has 128 CPU threads and about 109GiB RAM.
- Default CMake build parallelism is set to 24.
- Default HIP build parallelism is set to 8 to avoid memory spikes.
- `apt update` reported duplicate Vivaldi apt source entries.
