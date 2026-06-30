#!/usr/bin/env bash
# Source this file before local development builds:
#   source tools/fast-build-env.sh

export CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-24}"
export ULLM_HIP_BUILD_JOBS="${ULLM_HIP_BUILD_JOBS:-8}"

export CCACHE_DIR="${CCACHE_DIR:-${HOME}/.cache/ccache}"
export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-50G}"

export SCCACHE_DIR="${SCCACHE_DIR:-${HOME}/.cache/sccache}"
export SCCACHE_CACHE_SIZE="${SCCACHE_CACHE_SIZE:-50G}"
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-sccache}"

if command -v sccache >/dev/null 2>&1; then
  sccache --start-server >/dev/null 2>&1 || true
fi

