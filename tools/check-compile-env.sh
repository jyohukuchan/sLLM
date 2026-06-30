#!/usr/bin/env bash
set -euo pipefail

show_cmd() {
  local label="$1"
  shift
  printf '\n## %s\n' "${label}"
  if command -v "$1" >/dev/null 2>&1; then
    "$@" || true
  else
    printf 'missing: %s\n' "$1"
  fi
}

printf '# uLLM compile environment report\n'
printf 'date: %s\n' "$(date -Iseconds)"
printf 'host: %s\n' "$(hostname)"
printf 'nproc: %s\n' "$(nproc)"

printf '\n## Memory\n'
free -h

show_cmd 'CMake' cmake --version
show_cmd 'Ninja' ninja --version
show_cmd 'g++' g++ --version
show_cmd 'clang++' clang++ --version
show_cmd 'ROCm clang++' /opt/rocm-7.2.1/lib/llvm/bin/clang++ --version
show_cmd 'HIP' hipcc --version
show_cmd 'mold' mold --version
show_cmd 'lld' ld.lld --version
show_cmd 'ccache' ccache --version
show_cmd 'ccache stats' ccache --show-stats
show_cmd 'sccache' sccache --version
show_cmd 'sccache stats' sccache --show-stats
show_cmd 'rustc' rustc --version
show_cmd 'cargo' cargo --version
show_cmd 'rustfmt' rustfmt --version
show_cmd 'clippy' cargo clippy --version

printf '\n## Recommended env\n'
printf 'CMAKE_GENERATOR=%s\n' "${CMAKE_GENERATOR:-Ninja}"
printf 'CMAKE_BUILD_PARALLEL_LEVEL=%s\n' "${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
printf 'ULLM_HIP_BUILD_JOBS=%s\n' "${ULLM_HIP_BUILD_JOBS:-8}"
printf 'RUSTC_WRAPPER=%s\n' "${RUSTC_WRAPPER:-sccache}"

