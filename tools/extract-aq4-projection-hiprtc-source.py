#!/usr/bin/env python3
"""Materialize the exact production AQ4_0 projection HIPRTC source for ISA audit.

The runtime keeps these kernels as C++ raw strings and applies small, deliberate
source transformations for the gfx1201 production path.  A static HIP compiler
cannot consume those builders directly, so this tool reproduces only those
checked transformations and writes a standalone HIP translation unit.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ADD_LDS_SETUP = """    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float partial[256];
    float sum = 0.0f;"""
ADD_SHUFFLE_SETUP = """    // RPB=8 maps each row's contiguous 32 threads to one gfx1201 wave32.
    float sum = 0.0f;"""
ADD_LDS_TREE = """    partial[tid] = sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            partial[partial_offset + lane] += partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    // Preserve the production residual-add epilogue: it is not another AQ4 stream.
    if (lane == 0 && row < rows) {
        float value = partial[partial_offset];"""
ADD_SHUFFLE_REDUCTION = """    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, threads_per_row);
    }
    // Preserve the production residual-add epilogue: it is not another AQ4 stream.
    if (lane == 0 && row < rows) {
        float value = sum;"""

SILU_LDS_SETUP = """    const unsigned int partial_offset = row_in_block * threads_per_row;
    __shared__ float gate_partial[256];
    __shared__ float up_partial[256];
    float gate_sum = 0.0f;
    float up_sum = 0.0f;"""
SILU_SHUFFLE_SETUP = """    // RPB=8 maps each row's contiguous 32 threads to one gfx1201 wave32.
    // Gate and up keep independent accumulators, but neither needs cross-wave/LDS reduction.
    float gate_sum = 0.0f;
    float up_sum = 0.0f;"""
SILU_LDS_TREE = """    gate_partial[tid] = gate_sum;
    up_partial[tid] = up_sum;
    __syncthreads();
    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        if (lane < offset) {
            gate_partial[partial_offset + lane] += gate_partial[partial_offset + lane + offset];
            up_partial[partial_offset + lane] += up_partial[partial_offset + lane + offset];
        }
        __syncthreads();
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_partial[partial_offset];
        float up_value = up_partial[partial_offset];"""
SILU_SHUFFLE_REDUCTION = """    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset, threads_per_row);
        up_sum += __shfl_down(up_sum, offset, threads_per_row);
    }
    if (lane == 0 && row < rows) {
        float gate_value = gate_sum;
        float up_value = up_sum;"""


# The production dispatch in part_00.inc rejects every add shape other than g8/g16 with
# 32-element chunks.  This candidate removes only the uniform, runtime group-size traversal
# from that already-narrow contract.  Keep this here (rather than hand-editing a generated HIP
# file) so the Phase 0 static comparison is reproducible against the exact source in the tree.
ADD_SPECIALIZED_TRAVERSAL = """    if (row < rows) {
        const unsigned long long row_offset = row * cols;
        const unsigned long long row_byte_offset = row_offset >> 1;
        const unsigned long long chunks_per_row = cols / 32ull;
        if (group_size == 8ull) {
            const unsigned long long groups_per_row = cols >> 3;
            for (unsigned long long chunk = lane; chunk < chunks_per_row;
                 chunk += threads_per_row) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(
                    indices + row_byte_offset + chunk * 16ull);
                const unsigned long long col_base = chunk * 32ull;
                const unsigned long long group_base = row * groups_per_row + chunk * 4ull;
#pragma unroll
                for (unsigned int group_in_chunk = 0u; group_in_chunk < 4u; ++group_in_chunk) {
                    const unsigned int scale_index =
                        static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
                    if (scale_index >= scale_count) {
                        continue;
                    }
                    const unsigned int word = group_in_chunk == 0u ? packed.x :
                        (group_in_chunk == 1u ? packed.y :
                            (group_in_chunk == 2u ? packed.z : packed.w));
                    float raw_sum = 0.0f;
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(group_in_chunk * 8u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
                    sum += raw_sum * scale_values[scale_index] * tensor_scale;
                }
            }
        } else if (group_size == 16ull) {
            const unsigned long long groups_per_row = cols >> 4;
            for (unsigned long long chunk = lane; chunk < chunks_per_row;
                 chunk += threads_per_row) {
                const uint4 packed = *reinterpret_cast<const uint4 *>(
                    indices + row_byte_offset + chunk * 16ull);
                const unsigned long long col_base = chunk * 32ull;
                const unsigned long long group_base = row * groups_per_row + chunk * 2ull;
#pragma unroll
                for (unsigned int group_in_chunk = 0u; group_in_chunk < 2u; ++group_in_chunk) {
                    const unsigned int scale_index =
                        static_cast<unsigned int>(scale_indices[group_base + group_in_chunk]);
                    if (scale_index >= scale_count) {
                        continue;
                    }
                    const unsigned int first_word =
                        group_in_chunk == 0u ? packed.x : packed.z;
                    const unsigned int second_word =
                        group_in_chunk == 0u ? packed.y : packed.w;
                    float raw_sum = 0.0f;
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (first_word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(group_in_chunk * 16u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
#pragma unroll
                    for (unsigned int byte_offset = 0u; byte_offset < 4u; ++byte_offset) {
                        const unsigned int packed_byte =
                            (second_word >> (8u * byte_offset)) & 0xffu;
                        const unsigned long long col = col_base +
                            static_cast<unsigned long long>(
                                group_in_chunk * 16u + 8u + byte_offset * 2u);
                        raw_sum += codebook[packed_byte & 0x0fu] * input[col];
                        raw_sum += codebook[(packed_byte >> 4) & 0x0fu] * input[col + 1ull];
                    }
                    sum += raw_sum * scale_values[scale_index] * tensor_scale;
                }
            }
        }
    }
"""


def raw_string(source: str, function_name: str) -> str:
    marker = f"{function_name}() {{"
    function_start = source.find(marker)
    if function_start < 0:
        raise ValueError(f"missing source builder: {function_name}")
    raw_start_marker = 'return R"(\n'
    raw_start = source.find(raw_start_marker, function_start)
    if raw_start < 0:
        raise ValueError(f"missing raw source start: {function_name}")
    raw_start += len(raw_start_marker)
    raw_end = source.find('\n)";\n    }', raw_start)
    if raw_end < 0:
        raise ValueError(f"missing raw source end: {function_name}")
    return source[raw_start:raw_end]


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one match, found {count}")
    return source.replace(old, new, 1)


def production_source(runtime_source: str, kernel: str) -> str:
    if kernel == "add":
        result = raw_string(runtime_source, "aq4_matvec_add_wide_load_reference_kernel_source")
        result = replace_once(result, ADD_LDS_SETUP, ADD_SHUFFLE_SETUP, "add setup")
        return replace_once(result, ADD_LDS_TREE, ADD_SHUFFLE_REDUCTION, "add reduction")
    if kernel == "silu-mul":
        result = raw_string(runtime_source, "aq4_matvec_silu_mul_scalar_reference_kernel_source")
        result = replace_once(result, SILU_LDS_SETUP, SILU_SHUFFLE_SETUP, "silu setup")
        return replace_once(result, SILU_LDS_TREE, SILU_SHUFFLE_REDUCTION, "silu reduction")
    raise ValueError(f"unsupported kernel: {kernel}")


def specialized_add_source(runtime_source: str) -> str:
    result = production_source(runtime_source, "add")
    start_marker = """    if (row < rows) {
        const unsigned long long row_offset = row * cols;
"""
    end_marker = "    for (unsigned int offset = threads_per_row >> 1; offset > 0; offset >>= 1) {"
    start = result.find(start_marker)
    if start < 0:
        raise ValueError("add specialization: missing traversal start")
    end = result.find(end_marker, start)
    if end < 0 or not result[start:end].endswith("    }\n"):
        raise ValueError("add specialization: missing traversal end")
    return result[:start] + ADD_SPECIALIZED_TRAVERSAL + result[end:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", required=True, type=Path)
    parser.add_argument(
        "--kernel",
        required=True,
        choices=("add", "add-group-specialized", "silu-mul"),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runtime_source = args.runtime_source.read_text(encoding="utf-8")
    body = (
        specialized_add_source(runtime_source)
        if args.kernel == "add-group-specialized"
        else production_source(runtime_source, args.kernel)
    )
    result = (
        "// Generated by tools/extract-aq4-projection-hiprtc-source.py.\n"
        "// Exact production gfx1201/RPB=8 builder output, for static ISA inspection only.\n"
        "#include <hip/hip_runtime.h>\n"
        "#define ULLM_AQ4_ROWS_PER_BLOCK 8\n\n"
        + body
        + "\n"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
