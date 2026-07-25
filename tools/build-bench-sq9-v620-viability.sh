#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 /absolute/output/path [--save-temps]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output=$1
extra_flags=()
if [[ $# -eq 2 ]]; then
  if [[ $2 != --save-temps ]]; then
    echo "unknown option: $2" >&2
    exit 2
  fi
  extra_flags+=(--save-temps)
fi
mkdir -p -- "$(dirname -- "$output")"
hipcc -O3 -std=c++17 --offload-arch=gfx1030 "${extra_flags[@]}" \
  "$script_dir/bench-sq9-v620-viability-hip.cpp" \
  -o "$output"
