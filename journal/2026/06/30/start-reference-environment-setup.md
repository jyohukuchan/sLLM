# Start reference environment setup

## Done

- Corrected ATOM reference source from `efeslab/Atom` to `ROCm/ATOM`.
- Verified `ROCm/ATOM` has an MIT license in the checked source tree.
- Documented that V620 should only be expected to run llama.cpp among the external reference runtimes.
- Added `docs/plans/reference-environment-setup-v0.1.md`.
- Added `tools/check-reference-env.sh` for hardware and toolchain reporting.
- Checked the local environment: Ubuntu 24.04.4, HIP 7.2, two V620 GPUs, one gfx1201 GPU, Python 3.12, CMake 3.28, g++ 13.3.
- Saved environment report to `journal/2026/06/30/reference-env-report.txt`.
- Configured llama.cpp with HIP enabled.
- Built `llama-cli` and `llama-bench` under `build/reference/llama.cpp-hip`.
- Verified `llama-cli --version` runs.
- Verified `llama-bench --help` detects three ROCm devices.
