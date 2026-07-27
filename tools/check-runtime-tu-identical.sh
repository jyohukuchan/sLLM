#!/usr/bin/env bash
# Copyright 2026 uLLM contributors
# SPDX-License-Identifier: Apache-2.0
#
# Behaviour-preserving gate for restructuring runtime/src.
#
# runtime/src/ullm_runtime.cpp is a single translation unit assembled entirely
# from #include'd .inc files.  Any pure file-splitting refactor must therefore
# leave the *preprocessed* translation unit byte-identical: -P suppresses line
# markers, and the preprocessor strips comments, so newly added per-file licence
# headers and include guards vanish.
#
#   ./tools/check-runtime-tu-identical.sh --record   # write the baseline
#   ./tools/check-runtime-tu-identical.sh            # compare against it
#
# A mismatch means the refactor changed the code the compiler sees.  Diff
# ${BASELINE_DIR}/tu-baseline.i against the freshly generated .i to find where.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="${ROOT}/runtime/baseline"
BASELINE_I="${BASELINE_DIR}/tu-baseline.i"
BASELINE_SHA="${BASELINE_DIR}/tu-baseline.sha256"
ROCM="${ROCM_PATH:-/opt/rocm}"

preprocess() {
  g++ -E -P -std=c++20 \
    -I"${ROOT}/runtime/include" \
    -I"${ROCM}/include" \
    -D__HIP_PLATFORM_AMD__ \
    "${ROOT}/runtime/src/ullm_runtime.cpp"
}

if [[ "${1:-}" == "--record" ]]; then
  mkdir -p "${BASELINE_DIR}"
  preprocess > "${BASELINE_I}"
  sha256sum "${BASELINE_I}" | awk '{print $1}' > "${BASELINE_SHA}"
  printf 'recorded baseline\n  bytes  %s\n  sha256 %s\n' \
    "$(stat -c%s "${BASELINE_I}")" "$(cat "${BASELINE_SHA}")"
  exit 0
fi

if [[ ! -f "${BASELINE_SHA}" ]]; then
  echo "no baseline at ${BASELINE_SHA}; run with --record first" >&2
  exit 2
fi

CURRENT_I="$(mktemp)"
trap 'rm -f "${CURRENT_I}"' EXIT
preprocess > "${CURRENT_I}"

want="$(cat "${BASELINE_SHA}")"
got="$(sha256sum "${CURRENT_I}" | awk '{print $1}')"

if [[ "${want}" == "${got}" ]]; then
  printf 'PASS  preprocessed TU unchanged (%s)\n' "${got}"
  exit 0
fi

printf 'FAIL  preprocessed TU changed\n  expected %s\n  actual   %s\n' "${want}" "${got}" >&2
if [[ -f "${BASELINE_I}" ]]; then
  printf '  first differing lines:\n' >&2
  diff -u "${BASELINE_I}" "${CURRENT_I}" | head -40 >&2 || true
fi
exit 1
