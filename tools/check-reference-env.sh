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

printf '# uLLM reference environment report\n'
printf 'date: %s\n' "$(date -Iseconds)"
printf 'host: %s\n' "$(hostname)"

printf '\n## OS\n'
uname -a
if command -v lsb_release >/dev/null 2>&1; then
  lsb_release -a || true
elif [[ -f /etc/os-release ]]; then
  cat /etc/os-release
fi

show_cmd 'Python' python3 --version
show_cmd 'CMake' cmake --version
show_cmd 'g++' g++ --version
show_cmd 'HIP' hipcc --version
show_cmd 'ROCm SMI' rocm-smi --showproductname --showuniqueid --showdriverversion --showmeminfo vram
show_cmd 'Rust' rustc --version
show_cmd 'Cargo' cargo --version

printf '\n## Reference sources\n'
for d in reference-src/*; do
  [[ -d "${d}/.git" ]] || continue
  name="$(basename "${d}")"
  commit="$(git -C "${d}" rev-parse --short=12 HEAD)"
  url="$(git -C "${d}" remote get-url origin)"
  printf '%s %s %s\n' "${name}" "${commit}" "${url}"
done
