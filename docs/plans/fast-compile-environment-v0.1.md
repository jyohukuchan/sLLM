# Fast compile environment v0.1

## Purpose

uLLM は C++20、HIP C++、Rust を併用する。開発初期からコンパイル時間を抑えるため、Ninja、ccache、sccache、mold、lld、rustup stable を標準のローカル開発環境にする。

## Installed Tooling

Installed or verified:

- CMake
- Ninja
- g++
- clang++
- ROCm clang++
- hipcc
- ccache
- sccache
- mold
- lld
- rustup stable
- rustc
- cargo
- rustfmt
- clippy
- libssl-dev
- pkg-config

## Defaults

Use:

```bash
source tools/fast-build-env.sh
```

Default values:

- `CMAKE_GENERATOR=Ninja`
- `CMAKE_BUILD_PARALLEL_LEVEL=24`
- `ULLM_HIP_BUILD_JOBS=8`
- `CCACHE_MAXSIZE=50G`
- `SCCACHE_CACHE_SIZE=50G`
- `RUSTC_WRAPPER=sccache`

The normal C++ build parallelism is intentionally lower than the machine thread count. The host has many CPU threads, but HIP/C++ template-heavy builds can consume significant memory. Use higher values only after measuring memory pressure.

## Rust

Rust uses `sccache` through `RUSTC_WRAPPER`. Repository-local Cargo config uses clang with mold:

```toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

Check:

```bash
source tools/fast-build-env.sh
cargo build
sccache --show-stats
```

## C++ and HIP

For CMake projects, prefer Ninja and ccache:

```bash
source tools/fast-build-env.sh
cmake -S . -B build/dev \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build build/dev -j "${CMAKE_BUILD_PARALLEL_LEVEL}"
```

For HIP-heavy builds, prefer:

```bash
cmake --build build/dev -j "${ULLM_HIP_BUILD_JOBS}"
```

## llama.cpp Reference Build

The existing Makefile-style build succeeded. Future rebuilds should use Ninja and ccache:

```bash
source tools/fast-build-env.sh
cmake -S reference-src/llama.cpp -B build/reference/llama.cpp-hip-ninja \
  -G Ninja \
  -DGGML_HIP=ON \
  -DGGML_CCACHE=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_HIP_COMPILER_LAUNCHER=ccache
cmake --build build/reference/llama.cpp-hip-ninja --target ggml-base -j "${ULLM_HIP_BUILD_JOBS}"
```

## Verification

Run:

```bash
tools/check-compile-env.sh
```

Record the output under `journal/` when the compiler environment changes.

Current verification:

- `tools/check-compile-env.sh` succeeded and wrote `journal/2026/06/30/compile-env-report.txt`.
- Rust smoke build succeeded under `build/fast-compile-smoke/rust`.
- Rust build command used `sccache`, `clang`, and `mold`.
- CMake/Ninja/ccache configure succeeded for llama.cpp HIP under `build/reference/llama.cpp-hip-ninja`.
- `ggml-base` built with Ninja and ccache.
- Rebuilding `ggml-base` after clean produced 9/9 direct ccache hits.

## Notes

- `apt update` currently reports duplicate Vivaldi apt source entries. It does not block compiler setup, but it should be cleaned later to reduce package-manager noise.
- `build/` is ignored by Git.
- `reference-src/` is ignored by Git.
