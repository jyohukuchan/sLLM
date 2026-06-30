# Reference environment setup v0.1

## Purpose

uLLM の初期開発では、実装対象と比較対象を分ける。手元の V620/R9700 では HIP と llama.cpp を中心に確認し、vLLM、SGLang、ROCm/ATOM は参照、比較、将来の MI300X 検証に備えた環境として扱う。

## Local Hardware Baseline

確認済み環境:

- OS: Ubuntu 24.04.4 LTS
- Kernel: Linux 6.17.0-35-generic
- HIP: 7.2.53211
- ROCm: `/opt/rocm-7.2.1`
- GPU 0: AMD Radeon Pro V620, gfx1030, 32GB VRAM
- GPU 1: AMD Radeon Pro V620, gfx1030, 32GB VRAM
- GPU 2: AMD Radeon Graphics, gfx1201, 34GB VRAM
- Python: 3.12.3
- CMake: 3.28.3
- g++: 13.3.0
- Rust: not found in current PATH at the time of this check

## Reference Sources

Fetch or update reference source trees:

```bash
tools/fetch-reference-sources.sh
```

Expected local paths:

```text
reference-src/llama.cpp
reference-src/vllm
reference-src/sglang
reference-src/atom
reference-src/tensorrt-llm
```

The reference source trees are ignored by Git.

## V620 Compatibility Note

For the first local environment, assume the following:

| Project | V620 status | Role |
| --- | --- | --- |
| llama.cpp | expected to be usable | local build/run smoke tests, GGUF comparison |
| vLLM | not expected to run usefully on V620 | source reference, scheduler/API comparison, later MI300X/NVIDIA validation |
| SGLang | not expected to run usefully on V620 | source reference, serving/disaggregation comparison, later MI300X/NVIDIA validation |
| ROCm/ATOM | not expected to run usefully on V620 | ROCm/HIP reference and later MI300X validation |
| TensorRT-LLM | not expected to run on V620 | source reference and NVIDIA validation later |

This means V620 work should not be blocked on making vLLM, SGLang, ATOM, or TensorRT-LLM execute locally. For V620, the practical starting point is llama.cpp plus uLLM's own HIP C++ kernels.

## Setup Order

### Step 1: Environment report

Run:

```bash
tools/check-reference-env.sh
```

Record the output in `journal/` when hardware, driver, or compiler versions change.

### Step 2: Reference source fetch

Run:

```bash
tools/fetch-reference-sources.sh
```

Then update `docs/research/reference-source-inventory-v0.1.md` if commits changed.

### Step 3: llama.cpp local build

Use llama.cpp as the first external runtime smoke test on V620.

Initial build target:

```bash
cmake -S reference-src/llama.cpp -B build/reference/llama.cpp-hip \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/reference/llama.cpp-hip -j "$(nproc)"
```

If HIP compilation fails on V620/gfx1030, record the failure and fall back to CPU llama.cpp for comparison while uLLM HIP kernels are developed separately.

### Step 4: Python serving references

Do not spend early time forcing vLLM or SGLang to run on V620. For now:

1. Create a Python environment only when their source-level behavior needs to be inspected or tests need to be run.
2. Prefer reading scheduler, API, and benchmark code paths.
3. Defer GPU execution to MI300X or later hardware.

### Step 5: ROCm/ATOM reference

Use `ROCm/ATOM` as a ROCm/HIP reference source. Do not assume it is runnable on V620. Treat it as a later MI300X validation target.

### Step 6: uLLM HIP start

Start uLLM's own HIP C++ environment after Step 3:

1. Create a minimal HIP build skeleton.
2. Compile one dummy kernel for gfx1030 and gfx1201.
3. Add a GEMM microbenchmark scaffold.
4. Record kernel capability and runtime selection logs.

## Done Criteria

- Reference sources are present under `reference-src/`.
- ATOM points to `https://github.com/ROCm/ATOM.git`.
- V620 compatibility limits are documented.
- llama.cpp has a build attempt recorded.
- uLLM has a minimal HIP C++ build path ready to start.

## Current Status

- `tools/check-reference-env.sh` has been run and the report is stored in `journal/2026/06/30/reference-env-report.txt`.
- llama.cpp HIP configure succeeded with `-DGGML_HIP=ON`.
- `llama-cli` and `llama-bench` built successfully under `build/reference/llama.cpp-hip`.
- `llama-bench --help` detected three ROCm devices: two V620 GPUs and one gfx1201 GPU.
- OpenSSL was not found during llama.cpp configure, so HTTPS support in the bundled HTTP library is disabled for this build.
