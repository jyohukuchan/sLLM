#!/usr/bin/env bash
# shellcheck shell=bash

# This file is intended to be sourced so the selected ROCm tree is also used by
# later build and test commands in the caller's shell.

_sllm_activate_rocm_main() {
    local expected_rocm_version="7.14.0"
    local expected_llvm_major="23"
    local default_rocm_root="/opt/rocm/core-7.14"
    local selected_root
    local canonical_root
    local version
    local compiler_output
    local compiler_first_line
    local compiler_path
    local configured_path
    local configured_ld_library_path

    if ! command -v realpath >/dev/null 2>&1; then
        printf 'activate-rocm: realpath is required\n' >&2
        return 1
    fi

    if [[ -v SLLM_ROCM_PATH ]]; then
        if [[ -z "$SLLM_ROCM_PATH" ]]; then
            printf 'activate-rocm: SLLM_ROCM_PATH is set but empty\n' >&2
            return 1
        fi
        selected_root="$SLLM_ROCM_PATH"
    elif [[ -v ROCM_PATH ]]; then
        if [[ -z "$ROCM_PATH" ]]; then
            printf 'activate-rocm: ROCM_PATH is set but empty\n' >&2
            return 1
        fi
        selected_root="$ROCM_PATH"
    else
        selected_root="$default_rocm_root"
    fi

    if [[ ! -d "$selected_root" ]]; then
        printf 'activate-rocm: selected ROCm root is not a directory: %s\n' "$selected_root" >&2
        return 1
    fi
    if ! canonical_root="$(realpath -e -- "$selected_root")"; then
        printf 'activate-rocm: failed to canonicalize ROCm root: %s\n' "$selected_root" >&2
        return 1
    fi

    if [[ ! -f "$canonical_root/.info/version" ]]; then
        printf 'activate-rocm: ROCm version file is missing under %s\n' "$canonical_root" >&2
        return 1
    fi
    IFS= read -r version <"$canonical_root/.info/version" || true
    if [[ "$version" != "$expected_rocm_version" ]]; then
        printf 'activate-rocm: expected ROCm %s at %s, found %s\n' \
            "$expected_rocm_version" "$canonical_root" "${version:-<empty>}" >&2
        return 1
    fi

    compiler_path="$canonical_root/bin/amdclang++"
    if [[ ! -x "$compiler_path" ]]; then
        printf 'activate-rocm: compiler is missing or not executable: %s\n' "$compiler_path" >&2
        return 1
    fi
    if ! compiler_output="$("$compiler_path" --version 2>&1)"; then
        printf 'activate-rocm: failed to execute %s --version\n' "$compiler_path" >&2
        return 1
    fi
    compiler_first_line="${compiler_output%%$'\n'*}"
    if [[ ! "$compiler_first_line" =~ ^(AMD[[:space:]]clang|clang)[[:space:]]version[[:space:]]${expected_llvm_major}\. ]]; then
        printf 'activate-rocm: expected amdclang++ LLVM major %s, found: %s\n' \
            "$expected_llvm_major" "$compiler_first_line" >&2
        return 1
    fi

    local required_path
    for required_path in \
        "$canonical_root/include/hip" \
        "$canonical_root/lib/cmake/hip" \
        "$canonical_root/lib/llvm/amdgcn/bitcode" \
        "$canonical_root/lib/libamdhip64.so"; do
        if [[ ! -e "$required_path" ]]; then
            printf 'activate-rocm: required ROCm component is missing: %s\n' "$required_path" >&2
            return 1
        fi
        if [[ "$(realpath -e -- "$required_path")" != "$canonical_root"/* ]]; then
            printf 'activate-rocm: ROCm component resolves outside the selected root: %s\n' \
                "$required_path" >&2
            return 1
        fi
    done

    _sllm_rocm_deduplicate_path configured_path "$canonical_root/bin" \
        "$canonical_root/llvm/bin" -- "${PATH:-}"
    _sllm_rocm_deduplicate_path configured_ld_library_path "$canonical_root/lib" -- \
        "${LD_LIBRARY_PATH:-}"

    export ROCM_PATH="$canonical_root"
    export HIP_PATH="$canonical_root"
    export PATH="$configured_path"
    export LD_LIBRARY_PATH="$configured_ld_library_path"

    if ! command -v amdclang++ >/dev/null 2>&1; then
        printf 'activate-rocm: amdclang++ is not discoverable after activation\n' >&2
        return 1
    fi
    if [[ "$(realpath -e -- "$(command -v amdclang++)")" != \
        "$(realpath -e -- "$compiler_path")" ]]; then
        printf 'activate-rocm: PATH resolved amdclang++ outside the selected ROCm root\n' >&2
        return 1
    fi

    printf 'ROCm %s activated from %s (LLVM %s)\n' \
        "$expected_rocm_version" "$canonical_root" "$expected_llvm_major"
}

_sllm_rocm_deduplicate_path() {
    local output_name="$1"
    shift
    local -a entries=()
    local existing
    local item
    local key
    local result=""
    local separator=""
    local reached_existing=0
    local -A seen=()

    while (($# > 0)); do
        if [[ "$1" == "--" ]]; then
            reached_existing=1
            shift
            continue
        fi
        if ((reached_existing == 0)); then
            entries+=("$1")
        else
            existing="$1"
            IFS=':' read -r -a _sllm_existing_entries <<<"$existing"
            entries+=("${_sllm_existing_entries[@]}")
            unset _sllm_existing_entries
        fi
        shift
    done

    for item in "${entries[@]}"; do
        if [[ -z "$item" ]]; then
            key='<empty-path-entry>'
        elif [[ -e "$item" ]]; then
            key="$(realpath -e -- "$item")"
        else
            key="$item"
        fi
        if [[ -v "seen[$key]" ]]; then
            continue
        fi
        seen["$key"]=1
        result+="${separator}${item}"
        separator=':'
    done

    printf -v "$output_name" '%s' "$result"
}

_sllm_activate_rocm_main
_sllm_rocm_status=$?
unset -f _sllm_activate_rocm_main _sllm_rocm_deduplicate_path

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    if ((_sllm_rocm_status == 0)); then
        unset _sllm_rocm_status
        return 0
    fi
    unset _sllm_rocm_status
    return 1
fi
exit "$_sllm_rocm_status"
