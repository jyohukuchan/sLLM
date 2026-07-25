#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
# Build the thermal-guarded SQ8_1 V620 differential against an isolated runtime archive.

set -euo pipefail

if [[ $# -ne 2 || $1 != /* || $2 != /* ]]; then
  echo "usage: $0 /absolute/output-binary /absolute/libullm_runtime.a" >&2
  exit 2
fi

output=$1
runtime_archive=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
if [[ ! -f $runtime_archive ]]; then
  echo "runtime archive is absent: $runtime_archive" >&2
  exit 1
fi
if [[ -e $output ]]; then
  echo "refusing to overwrite output: $output" >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$output")"
hipcc -O2 -std=c++20 --offload-arch=gfx1030 \
  -I"$repo_root/runtime/include" \
  "$script_dir/sq8_1-v620-differential.cpp" \
  -Xlinker "$runtime_archive" -ldl -lpthread -o "$output"
