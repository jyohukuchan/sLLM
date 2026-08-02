#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_RUST_VERSION='1.97.1'
readonly EXPECTED_MSRV_VERSION='1.85.0'
readonly EXPECTED_ROCM_VERSION='7.14.0'
readonly EXPECTED_LLVM_MAJOR='23'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR

fail() {
    printf 'check-environment: ERROR: %s\n' "$*" >&2
    exit 1
}

note() {
    printf 'check-environment: %s\n' "$*"
}

require_command() {
    local command_name="$1"
    command -v "$command_name" >/dev/null 2>&1 || fail "required command not found: $command_name"
}

path_is_within_rocm() {
    local candidate="$1"
    local resolved
    resolved="$(realpath -e -- "$candidate")" || return 1
    [[ "$resolved" == "$ROCM_PATH"/* ]]
}

cleanup() {
    if [[ -z "${TEMP_DIR:-}" || ! -d "$TEMP_DIR" ]]; then
        return
    fi
    if [[ "$TEMP_DIR" != "$TEMP_PARENT"/ullm-environment.* ]]; then
        printf 'check-environment: refusing to clean unexpected path: %s\n' "$TEMP_DIR" >&2
        return
    fi
    rm -rf -- "$TEMP_DIR"
}

check_host() {
    local rust_version
    local msrv_version
    local rustfmt_version
    local clippy_version
    local cmake_version
    local cmake_major
    local cmake_minor
    local ninja_version
    local cxx_command="${CXX:-c++}"
    local compiler_path
    local compiler_version
    local compiler_first_line
    local clang_format_version
    local shellcheck_version
    local component

    require_command rustc
    rust_version="$(rustc --version)"
    [[ "$rust_version" == "rustc $EXPECTED_RUST_VERSION "* ]] || \
        fail "expected development Rust $EXPECTED_RUST_VERSION, found: $rust_version"

    require_command rustup
    msrv_version="$(rustup run "$EXPECTED_MSRV_VERSION" rustc --version)" || \
        fail "MSRV toolchain $EXPECTED_MSRV_VERSION is not installed"
    [[ "$msrv_version" == "rustc $EXPECTED_MSRV_VERSION "* ]] || \
        fail "expected MSRV Rust $EXPECTED_MSRV_VERSION, found: $msrv_version"

    require_command cargo
    require_command rustfmt
    rustfmt_version="$(rustfmt --version)"
    [[ -n "$rustfmt_version" ]] || fail 'rustfmt returned an empty version'
    clippy_version="$(cargo clippy --version)" || fail 'clippy is not installed for the development toolchain'
    [[ -n "$clippy_version" ]] || fail 'clippy returned an empty version'

    require_command cmake
    cmake_version="$(cmake --version | sed -n '1s/^cmake version //p')"
    [[ "$cmake_version" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]] || \
        fail "could not parse CMake version: ${cmake_version:-<empty>}"
    IFS='.' read -r cmake_major cmake_minor _ <<<"$cmake_version"
    if ((cmake_major < 3 || (cmake_major == 3 && cmake_minor < 21))); then
        fail "CMake >=3.21 is required, found $cmake_version"
    fi

    require_command ninja
    ninja_version="$(ninja --version)"
    [[ -n "$ninja_version" ]] || fail 'Ninja returned an empty version'

    require_command "$cxx_command"
    if ! printf '%s\n' \
        '#include <optional>' \
        'int main() { std::optional<int> value = 42; return *value == 42 ? 0 : 1; }' | \
        "$cxx_command" -std=c++17 -Werror -x c++ - -o "$TEMP_DIR/cxx17-smoke"; then
        fail "C++17 compilation failed with $cxx_command"
    fi
    "$TEMP_DIR/cxx17-smoke" || fail 'C++17 probe returned a failure'

    require_command clang-format
    clang_format_version="$(clang-format --version)"
    [[ -n "$clang_format_version" ]] || fail 'clang-format returned an empty version'

    require_command shellcheck
    shellcheck_version="$(shellcheck --version | sed -n 's/^version: //p')"
    [[ -n "$shellcheck_version" ]] || fail 'could not determine ShellCheck version'

    [[ -n "${ROCM_PATH:-}" && -n "${HIP_PATH:-}" ]] || \
        fail 'ROCM_PATH and HIP_PATH must both be set by activate-rocm.sh'
    [[ "$ROCM_PATH" == "$HIP_PATH" ]] || fail 'ROCM_PATH and HIP_PATH select different roots'
    [[ "$(realpath -e -- "$ROCM_PATH")" == "$ROCM_PATH" ]] || \
        fail "ROCM_PATH is not canonical: $ROCM_PATH"
    [[ "$(<"$ROCM_PATH/.info/version")" == "$EXPECTED_ROCM_VERSION" ]] || \
        fail "selected ROCm root is not $EXPECTED_ROCM_VERSION"

    compiler_path="$ROCM_PATH/bin/amdclang++"
    path_is_within_rocm "$compiler_path" || fail 'amdclang++ resolves outside ROCM_PATH'
    [[ "$(realpath -e -- "$(command -v amdclang++)")" == \
        "$(realpath -e -- "$compiler_path")" ]] || \
        fail 'PATH does not resolve amdclang++ from ROCM_PATH'
    compiler_version="$("$compiler_path" --version)"
    compiler_first_line="${compiler_version%%$'\n'*}"
    [[ "$compiler_first_line" =~ ^(AMD[[:space:]]clang|clang)[[:space:]]version[[:space:]]${EXPECTED_LLVM_MAJOR}\. ]] || \
        fail "expected amdclang++ LLVM major $EXPECTED_LLVM_MAJOR, found: $compiler_first_line"

    for component in \
        "$ROCM_PATH/include/hip/hip_runtime.h" \
        "$ROCM_PATH/lib/cmake/hip" \
        "$ROCM_PATH/amdgcn/bitcode" \
        "$ROCM_PATH/lib/libamdhip64.so"; do
        [[ -e "$component" ]] || fail "required ROCm component is missing: $component"
        path_is_within_rocm "$component" || \
            fail "ROCm component resolves outside the selected root: $component"
    done

    note "Rust=$EXPECTED_RUST_VERSION MSRV=$EXPECTED_MSRV_VERSION"
    note "rustfmt=$rustfmt_version clippy=$clippy_version"
    note "CMake=$cmake_version Ninja=$ninja_version CXX=$cxx_command"
    note "clang-format=$clang_format_version ShellCheck=$shellcheck_version"
    note "ROCm=$EXPECTED_ROCM_VERSION LLVM=$EXPECTED_LLVM_MAJOR root=$ROCM_PATH"
    note 'host checks passed'
}

parse_architectures() {
    local architecture_csv="$1"
    local architecture
    local -A seen=()

    [[ -n "$architecture_csv" ]] || fail 'ULLM_HIP_ARCHITECTURES must not be empty'
    if [[ "$architecture_csv" == ,* || "$architecture_csv" == *, || \
        "$architecture_csv" == *,,* ]]; then
        fail "invalid comma-separated ULLM_HIP_ARCHITECTURES: $architecture_csv"
    fi

    IFS=',' read -r -a HIP_ARCHITECTURES <<<"$architecture_csv"
    for architecture in "${HIP_ARCHITECTURES[@]}"; do
        if [[ ! "$architecture" =~ ^gfx[0-9a-f]+(:[[:alnum:]_+-]+)*$ ]]; then
            fail "invalid exact HIP target: $architecture"
        fi
        if [[ -v "seen[$architecture]" ]]; then
            fail "duplicate HIP target: $architecture"
        fi
        seen["$architecture"]=1
    done
}

check_gpu() {
    local architecture_csv="${ULLM_HIP_ARCHITECTURES:-gfx1030,gfx1201}"
    local architecture
    local compiler="$ROCM_PATH/bin/amdclang++"
    local smoke_binary="$TEMP_DIR/hip-smoke"
    local ldd_output
    local runtime_library
    local runtime_dependency_output
    local hsa_runtime_library
    local -a compile_arguments=(
        -std=c++17
        -x hip
        "--rocm-path=$ROCM_PATH"
        "--hip-path=$HIP_PATH"
    )

    parse_architectures "$architecture_csv"
    for architecture in "${HIP_ARCHITECTURES[@]}"; do
        compile_arguments+=("--offload-arch=$architecture")
    done
    compile_arguments+=(
        "-DULLM_HIP_ARCHITECTURES=\"$architecture_csv\""
        "$SCRIPT_DIR/hip-smoke.cpp"
        "-L$ROCM_PATH/lib"
        "-Wl,-rpath,$ROCM_PATH/lib"
        -o "$smoke_binary"
    )

    note "compiling HIP fat binary for targets=$architecture_csv"
    "$compiler" "${compile_arguments[@]}" || fail 'HIP fat-binary compilation failed'
    [[ -x "$smoke_binary" ]] || fail 'HIP compiler did not produce an executable'

    ldd_output="$(LD_LIBRARY_PATH="$ROCM_PATH/lib" ldd "$smoke_binary")" || \
        fail 'ldd failed for the HIP smoke binary'
    runtime_library="$(awk '$1 ~ /^libamdhip64\.so/ && $2 == "=>" { print $3; exit }' \
        <<<"$ldd_output")"
    [[ -n "$runtime_library" && -e "$runtime_library" ]] || \
        fail 'HIP smoke binary did not resolve libamdhip64'
    path_is_within_rocm "$runtime_library" || \
        fail "HIP runtime resolved outside ROCM_PATH: $runtime_library"
    note "resolved HIP runtime=$runtime_library"

    runtime_dependency_output="$(LD_LIBRARY_PATH="$ROCM_PATH/lib" ldd "$runtime_library")" || \
        fail 'ldd failed for the HIP runtime library'
    hsa_runtime_library="$(awk '$1 ~ /^libhsa-runtime64\.so/ && $2 == "=>" { print $3; exit }' \
        <<<"$runtime_dependency_output")"
    [[ -n "$hsa_runtime_library" && -e "$hsa_runtime_library" ]] || \
        fail 'HIP runtime did not resolve libhsa-runtime64'
    path_is_within_rocm "$hsa_runtime_library" || \
        fail "HSA runtime resolved outside ROCM_PATH: $hsa_runtime_library"
    note "resolved HSA runtime=$hsa_runtime_library"

    note 'running HIP smoke on every visible device'
    "$smoke_binary" || fail 'HIP smoke failed'
    note 'gpu checks passed'
}

if (($# != 1)); then
    fail 'usage: scripts/dev/check-environment.sh {host|gpu}'
fi
readonly MODE="$1"
if [[ "$MODE" != 'host' && "$MODE" != 'gpu' ]]; then
    fail "unknown mode: $MODE (expected host or gpu)"
fi

require_command realpath
TEMP_PARENT="$(realpath -m -- "${TMPDIR:-/tmp}")"
readonly TEMP_PARENT
TEMP_DIR="$(mktemp -d "$TEMP_PARENT/ullm-environment.XXXXXXXX")"
readonly TEMP_DIR
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# SCRIPT_DIR is resolved and validated at runtime.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/activate-rocm.sh"
check_host
if [[ "$MODE" == 'gpu' ]]; then
    check_gpu
fi
