#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 1 || $1 != /* ]]; then
  echo "usage: $0 /absolute/output-binary" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
output=$1
if [[ -e $output ]]; then
  echo "refusing to overwrite output: $output" >&2
  exit 1
fi
mkdir -p -- "$(dirname -- "$output")"
hipcc -O3 -std=c++17 --offload-arch=gfx1030 "$script_dir/bench-sq8_1-v620-optimization-hip.cpp" \
  -lhiprtc -o "$output"
