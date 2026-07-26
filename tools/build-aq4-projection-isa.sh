#!/usr/bin/env bash
# Compile the exact production gfx1201 AQ4_0 add or SiLU-mul HIPRTC body for ISA audit.
set -euo pipefail

if [[ $# -ne 2 || $1 != /* || ( $2 != add && $2 != add-group-specialized && $2 != silu-mul ) ]]; then
  echo "usage: $0 /absolute/output-directory {add|add-group-specialized|silu-mul}" >&2
  exit 2
fi

output_dir=$1
kind=$2
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
rocm_path=${ROCM_PATH:-/opt/rocm-7.2.1}
hipcc_bin=${HIPCC_BIN:-/usr/bin/hipcc}
bundler="$rocm_path/lib/llvm/bin/clang-offload-bundler"
objdump="$rocm_path/lib/llvm/bin/llvm-objdump"
readelf="$rocm_path/lib/llvm/bin/llvm-readelf"
extractor="$script_dir/extract-aq4-projection-hiprtc-source.py"
analyzer="$script_dir/analyze-aq4-projection-isa.py"
runtime_source="$repo_root/runtime/src/ullm_runtime_hiprtc_sources.inc"

for command in "$hipcc_bin" "$bundler" "$objdump" "$readelf" python3; do
  if [[ ! -x $command ]] && ! command -v "$command" >/dev/null 2>&1; then
    echo "required command is unavailable: $command" >&2
    exit 1
  fi
done

mkdir -p -- "$output_dir"
source_file="$output_dir/aq4-${kind}-production-gfx1201.hip.cpp"
bundle="$output_dir/aq4-${kind}-production-gfx1201.bundle"
code_object="$output_dir/aq4-${kind}-production-gfx1201.hsaco"
disassembly="$output_dir/aq4-${kind}-production-gfx1201.disasm"
notes="$output_dir/aq4-${kind}-production-gfx1201.notes.txt"
summary="$output_dir/aq4-${kind}-production-gfx1201.summary.json"
kernel="ullm_aq4_matvec_add_f32_kernel"
if [[ $kind == silu-mul ]]; then
  kernel="ullm_aq4_matvec_silu_mul_f32_kernel"
fi

python3 "$extractor" --runtime-source "$runtime_source" --kernel "$kind" --output "$source_file"
"$hipcc_bin" -O3 -std=c++17 --offload-arch=gfx1201 --offload-device-only "$source_file" -o "$bundle"
"$bundler" --unbundle --type=o --input="$bundle" \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1201 --output="$code_object"
"$objdump" --disassemble --mcpu=gfx1201 "$code_object" >"$disassembly"
"$readelf" --notes "$code_object" | sed '${/^$/d;}' >"$notes"
python3 "$analyzer" --kernel "$kernel" --notes "$notes" --disassembly "$disassembly" --output "$summary"
"$hipcc_bin" --version >"$output_dir/compiler-version.txt"
"$objdump" --version >>"$output_dir/compiler-version.txt"
{
  sha256sum "$runtime_source" "$extractor" "$analyzer" "$script_dir/build-aq4-projection-isa.sh"
  find "$output_dir" -maxdepth 1 -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 -r sha256sum
} >"$output_dir/SHA256SUMS"
