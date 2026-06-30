#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="${ULLM_REFERENCE_SRC_DIR:-${root_dir}/reference-src}"

mkdir -p "${target_dir}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dest="${target_dir}/${name}"

  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch --depth=1 origin
    git -C "${dest}" checkout --detach FETCH_HEAD
  else
    git clone --depth=1 --filter=blob:none "${url}" "${dest}"
  fi

  printf '%s %s %s\n' "${name}" "$(git -C "${dest}" rev-parse --short=12 HEAD)" "${url}"
}

clone_or_update llama.cpp https://github.com/ggml-org/llama.cpp.git
clone_or_update vllm https://github.com/vllm-project/vllm.git
clone_or_update sglang https://github.com/sgl-project/sglang.git
clone_or_update atom https://github.com/ROCm/ATOM.git
clone_or_update tensorrt-llm https://github.com/NVIDIA/TensorRT-LLM.git
