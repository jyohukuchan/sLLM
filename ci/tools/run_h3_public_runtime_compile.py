#!/usr/bin/env python3
"""Build one isolated, compile-only H3 public HIP runtime row.

This runner intentionally never invokes a produced ELF.  Device code is
extracted from the probe object's fatbin, while host bundle evidence is read
back from the linked host ELF's own ``.hip_fatbin`` and host ABI symbols are
checked in that linked ELF.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import platform
import re
import resource
import selectors
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("gfx1030", "gfx1201")
E_FLAGS = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}
BUNDLE_IDS = {target: f"hipv4-amdgcn-amd-amdhsa--{target}" for target in TARGETS}
HOST_BUNDLE_ID = "host-x86_64-unknown-linux-gnu-"
PINNED_IMAGE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
PINNED_CONFIG = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
ZERO_SHA256 = "0" * 64
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
EXPECTED_FEATURES = {"xnack": "unsupported", "sramecc": "unsupported", "generic_processor_version": 0}
EXPECTED_SOURCE_PATHS = (
    "include/sllm/hip.h",
    "native/hip/src/hip_compile_probe.hip.cpp",
    "native/hip/src/public_runtime.hip.cpp",
    "native/hip/src/public_runtime_internal.hpp",
)
PUBLIC_RUNTIME_API_SOURCE_PATHS = (
    "native/hip/src/argmax_api.cpp",
    "native/hip/src/attention_preprocess_api.cpp",
    "native/hip/src/causal_attention_api.cpp",
    "native/hip/src/elementwise_api.cpp",
    "native/hip/src/embedding_api.cpp",
    "native/hip/src/kv_state_api.cpp",
    "native/hip/src/linear_attention_api.cpp",
    "native/hip/src/matmul_api.cpp",
    "native/hip/src/moe_expert_api.cpp",
    "native/hip/src/moe_route_api.cpp",
    "native/hip/src/rmsnorm_api.cpp",
    "native/hip/src/residual_rmsnorm_api.cpp",
    "native/hip/src/rotary_api.cpp",
    "native/hip/src/token_selector_api.cpp",
    "native/hip/src/windowed_attention_api.cpp",
)
PUBLIC_RUNTIME_KERNEL_SOURCE_PATHS = (
    "native/hip/src/argmax_kernel.hip.cpp",
    "native/hip/src/attention_preprocess_kernel.hip.cpp",
    "native/hip/src/causal_attention_kernel.hip.cpp",
    "native/hip/src/elementwise_kernel.hip.cpp",
    "native/hip/src/embedding_kernel.hip.cpp",
    "native/hip/src/gdn_projection_bundle_kernel.hip.cpp",
    "native/hip/src/gemma_attention_kernel.hip.cpp",
    "native/hip/src/kv_state_kernel.hip.cpp",
    "native/hip/src/linear_attention_kernel.hip.cpp",
    "native/hip/src/matmul_kernel.hip.cpp",
    "native/hip/src/mlp_gate_up_silu_bundle_kernel.hip.cpp",
    "native/hip/src/moe_expert_kernel.hip.cpp",
    "native/hip/src/moe_route_kernel.hip.cpp",
    "native/hip/src/rmsnorm_kernel.hip.cpp",
    "native/hip/src/rotary_kernel.hip.cpp",
    "native/hip/src/token_selector_kernel.hip.cpp",
)
PUBLIC_RUNTIME_DIRECT_INCLUDE_PATHS = (
    "native/hip/src/argmax_api.hpp",
    "native/hip/src/argmax_kernel_internal.hpp",
    "native/hip/src/argmax_runtime.inc",
    "native/hip/src/attention_preprocess_api.hpp",
    "native/hip/src/attention_preprocess_kernel_internal.hpp",
    "native/hip/src/attention_preprocess_runtime.inc",
    "native/hip/src/causal_attention_api.hpp",
    "native/hip/src/causal_attention_kernel_internal.hpp",
    "native/hip/src/causal_attention_runtime.inc",
    "native/hip/src/elementwise_api.hpp",
    "native/hip/src/elementwise_kernel_internal.hpp",
    "native/hip/src/embedding_api.hpp",
    "native/hip/src/embedding_kernel_internal.hpp",
    "native/hip/src/embedding_runtime.inc",
    "native/hip/src/evidence_abi.h",
    "native/hip/src/gdn_projection_bundle_kernel_internal.hpp",
    "native/hip/src/gdn_projection_bundle_runtime.inc",
    "native/hip/src/gemma_attention_kernel_internal.hpp",
    "native/hip/src/kv_state_api.hpp",
    "native/hip/src/kv_state_kernel_internal.hpp",
    "native/hip/src/linear_attention_api.hpp",
    "native/hip/src/linear_attention_kernel_internal.hpp",
    "native/hip/src/linear_attention_runtime.inc",
    "native/hip/src/matmul_api.hpp",
    "native/hip/src/matmul_kernel_internal.hpp",
    "native/hip/src/matmul_runtime.inc",
    "native/hip/src/mlp_gate_up_silu_bundle_kernel_internal.hpp",
    "native/hip/src/mlp_gate_up_silu_bundle_runtime.inc",
    "native/hip/src/moe_expert_api.hpp",
    "native/hip/src/moe_expert_kernel_internal.hpp",
    "native/hip/src/moe_expert_runtime.inc",
    "native/hip/src/moe_route_api.hpp",
    "native/hip/src/moe_route_kernel_internal.hpp",
    "native/hip/src/moe_route_runtime.inc",
    "native/hip/src/residual_rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_kernel_internal.hpp",
    "native/hip/src/rotary_api.hpp",
    "native/hip/src/rotary_kernel_internal.hpp",
    "native/hip/src/rotary_runtime.inc",
    "native/hip/src/token_selector_api.hpp",
    "native/hip/src/token_selector_kernel_internal.hpp",
    "native/hip/src/token_selector_runtime.inc",
    "native/hip/src/windowed_attention_api.hpp",
    "native/hip/src/windowed_attention_runtime.inc",
)
EXPECTED_DIRECT_COMPILE_SOURCE_PATHS = tuple(sorted(set(
    EXPECTED_SOURCE_PATHS
    + PUBLIC_RUNTIME_API_SOURCE_PATHS
    + PUBLIC_RUNTIME_KERNEL_SOURCE_PATHS
    + PUBLIC_RUNTIME_DIRECT_INCLUDE_PATHS
)))
EXPECTED_OUTPUT = {"root_prefix": "sllm-h3-public-runtime-", "directory_pattern": "h3-public-gfx{target}", "build_directory_pattern": "build", "probe_object_pattern": "hip-compile-probe-{target}.o", "public_runtime_object_pattern": "public-runtime-{target}.o", "rmsnorm_kernel_object_pattern": "rmsnorm-kernel-{target}.o", "rmsnorm_api_object_pattern": "rmsnorm-api-{target}.o", "host_elf_pattern": "public-runtime-{target}.elf", "probe_fatbin_pattern": "probe-{target}.fatbin", "device_object_pattern": "device-code-object-{target}.elf"}
_AT_EMPTY_PATH = 0x1000
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_INPUT_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
_TMPFILE_FLAGS = (
    os.O_RDWR
    | getattr(os, "O_TMPFILE", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
PUBLIC_SYMBOLS = tuple(sorted({
    "sllm_argmax_execute", "sllm_argmax_plan_release", "sllm_argmax_prepare",
    "sllm_attention_preprocess_execute", "sllm_attention_preprocess_plan_release", "sllm_attention_preprocess_prepare",
    "sllm_backend_probe", "sllm_buffer_copy_d2d", "sllm_buffer_copy_d2h",
    "sllm_buffer_copy_h2d", "sllm_buffer_create", "sllm_buffer_release",
    "sllm_buffer_size", "sllm_causal_attention_execute", "sllm_completion_finalize_after",
    "sllm_completion_query", "sllm_completion_read", "sllm_completion_release",
    "sllm_completion_timing", "sllm_completion_wait", "sllm_context_create",
    "sllm_context_probe", "sllm_context_release", "sllm_device_count",
    "sllm_device_query", "sllm_elementwise_execute", "sllm_elementwise_plan_release",
    "sllm_elementwise_prepare", "sllm_embedding_execute", "sllm_embedding_plan_release",
    "sllm_embedding_prepare", "sllm_event_create", "sllm_event_release",
    "sllm_gdn_projection_bundle_execute", "sllm_gdn_projection_bundle_plan_release", "sllm_gdn_projection_bundle_prepare",
    "sllm_get_abi_version", "sllm_kv_state_append", "sllm_kv_state_append_cancel",
    "sllm_kv_state_create", "sllm_kv_state_create_v2", "sllm_kv_state_export",
    "sllm_kv_state_fork", "sllm_kv_state_fork_query", "sllm_kv_state_image_plane_size",
    "sllm_kv_state_image_query", "sllm_kv_state_import", "sllm_kv_state_import_finalize",
    "sllm_kv_state_query", "sllm_kv_state_release", "sllm_kv_state_rewind_last",
    "sllm_kv_state_snapshot", "sllm_kv_view_query", "sllm_kv_view_release",
    "sllm_linear_attention_cancel", "sllm_linear_attention_execute", "sllm_linear_attention_state_create",
    "sllm_linear_attention_state_export", "sllm_linear_attention_state_fork", "sllm_linear_attention_state_image_plane_size",
    "sllm_linear_attention_state_image_query", "sllm_linear_attention_state_import", "sllm_linear_attention_state_import_finalize",
    "sllm_linear_attention_state_query", "sllm_linear_attention_state_release", "sllm_linear_attention_state_rewind_last",
    "sllm_matmul_execute", "sllm_matmul_plan_release", "sllm_matmul_prepare",
    "sllm_mlp_gate_up_silu_bundle_execute", "sllm_mlp_gate_up_silu_bundle_plan_release", "sllm_mlp_gate_up_silu_bundle_prepare",
    "sllm_moe_expert_execute", "sllm_moe_expert_plan_release", "sllm_moe_expert_prepare",
    "sllm_moe_route_execute", "sllm_moe_route_plan_release", "sllm_moe_route_prepare",
    "sllm_query_version", "sllm_queue_create", "sllm_queue_fence",
    "sllm_queue_release", "sllm_queue_set_completion_mode", "sllm_residual_rmsnorm_execute",
    "sllm_residual_rmsnorm_plan_release", "sllm_residual_rmsnorm_prepare", "sllm_rmsnorm_execute",
    "sllm_rmsnorm_plan_release", "sllm_rmsnorm_prepare", "sllm_rotary_execute",
    "sllm_rotary_plan_release", "sllm_rotary_prepare", "sllm_token_selector_execute",
    "sllm_token_selector_plan_release", "sllm_token_selector_prepare", "sllm_windowed_attention_execute",
    "sllm_windowed_attention_plan_release", "sllm_windowed_attention_prepare",
}))


def declared_public_symbols(repo: Path) -> tuple[str, ...]:
    """Extract the canonical public C ABI declarations from the umbrella header."""

    header = repo / "include/sllm/hip.h"
    try:
        source = header.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RuntimeContractError(f"public ABI header cannot be read: {header}") from exc
    start_marker = 'extern "C" {'
    end_marker = '} /* extern "C" */'
    start = source.find(start_marker)
    end = source.find(end_marker, start + len(start_marker)) if start >= 0 else -1
    if start < 0 or end < 0 or source.find(start_marker, start + len(start_marker)) >= 0:
        raise RuntimeContractError("public ABI header must contain exactly one closed extern C block")
    names = re.findall(r"\b(sllm_[A-Za-z0-9_]+)\s*\(", source[start + len(start_marker) : end])
    if not names or len(names) != len(set(names)):
        raise RuntimeContractError("public ABI header declarations are missing or duplicated")
    declared = tuple(sorted(names))
    if declared != PUBLIC_SYMBOLS:
        missing = sorted(set(PUBLIC_SYMBOLS) - set(declared))
        extra = sorted(set(declared) - set(PUBLIC_SYMBOLS))
        raise RuntimeContractError(f"public ABI header symbol set drift: missing={missing}, extra={extra}")
    return declared


def validate_public_symbol_contract(repo: Path) -> None:
    declared_public_symbols(repo)


KERNEL_SYMBOLS = (
    "sllm_argmax_bf16_f32_v1",
    "sllm_attention_preprocess_headwise_norm_rope_v1",
    "sllm_attention_preprocess_headwise_norm_rope_wave32_v1",
    "sllm_elementwise_add_bf16_fp32_v1",
    "sllm_elementwise_broadcast_add_bf16_fp32_v1",
    "sllm_elementwise_copy_bf16_v1",
    "sllm_elementwise_gelu_tanh_mul_bf16_fp32_v1",
    "sllm_elementwise_scalar_mul_bf16_fp32_v1",
    "sllm_elementwise_sigmoid_mul_bf16_fp32_v1",
    "sllm_elementwise_silu_mul_bf16_fp32_v1",
    "sllm_elementwise_tanh_softcap_bf16_fp32_v1",
    "sllm_embedding_gather_bf16_i32_v1",
    "sllm_gdn_projection_bundle_bf16_fp32_decode_v1",
    "sllm_gemma_causal_attention_online_softmax_gqa_bf16_v1",
    "sllm_kv_state_bf16_to_f16_token_major_v2",
    "sllm_kv_state_bf16_to_fp8_token_major_v1",
    "sllm_kv_state_bf16_to_nvfp4_token_major_v1",
    "sllm_linear_attention_causal_conv_silu_v1",
    "sllm_linear_attention_column_postprocess_v2",
    "sllm_linear_attention_column_preprocess_v2",
    "sllm_linear_attention_recurrent_column_state_v2",
    "sllm_linear_attention_recurrent_gated_norm_decode_pair_v1",
    "sllm_linear_attention_recurrent_gated_norm_v1",
    "sllm_matmul_bf16_fp32_decode_serial_rows_v1",
    "sllm_matmul_bf16_fp32_decode_serial_rows_wave64_v1",
    "sllm_matmul_bf16_fp32_decode_v4",
    "sllm_matmul_bf16_fp32_decode_wave64_v1",
    "sllm_matmul_bf16_fp32_prefill_short_serial_v1",
    "sllm_matmul_bf16_fp32_tiled16_v2",
    "sllm_matmul_bf16_fp32_v1",
    "sllm_matmul_bf16_to_fp8_outer_v1",
    "sllm_matmul_bf16_to_fp8_outer_v2",
    "sllm_matmul_bf16_to_mxfp4_block32_even_v1",
    "sllm_matmul_bf16_to_nvfp4_block16_v1",
    "sllm_matmul_fp32_to_bf16_short_mixed_v1",
    "sllm_matmul_fp8_outer_emulation_v1",
    "sllm_matmul_mxfp4_w4a4_block32_decode_v1",
    "sllm_matmul_mxfp4_w4a4_block32_prefill_v1",
    "sllm_matmul_nvfp4_block16_packed_dequant_v1",
    "sllm_matmul_nvfp4_block16_prefill_row8_tiled256_v2",
    "sllm_matmul_nvfp4_w4a4_block16_packed_v1",
    "sllm_mlp_gate_up_silu_bundle_bf16_fp32_decode_v1",
    "sllm_moe_down_combine_v1",
    "sllm_moe_route_bf16_stable_topk_v1",
    "sllm_moe_route_stable_group_v1",
    "sllm_moe_routed_gateup_v1",
    "sllm_moe_shared_gateup_v1",
    "sllm_rmsnorm_baseline_wave32_v1",
    "sllm_rmsnorm_baseline_wave64_v1",
    "sllm_rmsnorm_residual_fused_wave32_v1",
    "sllm_rmsnorm_residual_fused_wave64_v1",
    "sllm_rotary_split_half_bf16_fp32_v1",
    "sllm_token_selector_bf16_f32_mask_v1",
)
INTERNAL_RUNTIME_SYMBOLS = ("sllm_hip_kv_view_readback",)
CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS = (
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_138__device_stub__causal_attention_kernelILb0EEEvPKtPKvS5_S5_S5_PKfS7_Ptjmmmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_138__device_stub__causal_attention_kernelILb1EEEvPKtPKvS5_S5_S5_PKfS7_Ptjmmmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_combine_kernelEPfPKfPKtS5_PKmS5_jjmmjjjj",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_pack_kv_kernelEPKtS2_PtS3_Pmmjjj",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_144__device_stub__scaled_prefill_scatter_kernelEPKfPtjjjjjj",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_147__device_stub__scaled_prefill_pack_query_kernelEPKtPtPfjjjjjj",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_149__device_stub__scaled_prefill_softmax_fp16_kernelEPfPtS2_jmmjjPKff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_156__device_stub__causal_attention_decode_wave_split_kernelILb0EEEvPKtPKvS5_S5_S5_PKfS7_Ptmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_156__device_stub__causal_attention_decode_wave_split_kernelILb1EEEvPKtPKvS5_S5_S5_PKfS7_Ptmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_158__device_stub__causal_attention_prefill_gqa4_qtile4_kernelEPKtPKvS4_S4_S4_PKfS6_Ptjmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_158__device_stub__causal_attention_prefill_gqa4_shared_kernelEPKtPKvS4_S4_S4_PKfS6_Ptjmjjjjff",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_161__device_stub__causal_attention_long_prefill_v2_stage1_kernelEPKtS2_S2_jmmmjjPf",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_162__device_stub__causal_attention_long_prefill_v2_combine_kernelEPKfPtjjm",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage1_kernelILj16EEEvPKtS3_S3_PtmPf",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage1_kernelILj32EEEvPKtS3_S3_PtmPf",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage2_kernelILj16EEEvPKtPtjPKf",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_163__device_stub__causal_attention_decode_gqa4_split_stage2_kernelILj32EEEvPKtPtjPKf",
    "_ZN28sllm_causal_attention_kernel12_GLOBAL__N_166__device_stub__causal_attention_decode_wave_split_fp16_pair_kernelEPKtPKvS4_S4_S4_PKfS6_Ptmjjjjff",
)
EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS = (
    "__hipPopCallConfiguration",
    "__hipPushCallConfiguration",
    "__hipRegisterFatBinary",
    "__hipRegisterFunction",
    "__hipUnregisterFatBinary",
    "hipDeviceGetAttribute",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    "hipEventElapsedTime",
    "hipEventQuery",
    "hipEventRecord",
    "hipFree",
    "hipGetDeviceCount",
    "hipGetDevicePropertiesR0600",
    "hipGetErrorString",
    "hipGetLastError",
    "hipLaunchKernel",
    "hipMalloc",
    "hipMemAddressFree",
    "hipMemAddressReserve",
    "hipMemCreate",
    "hipMemGetAllocationGranularity",
    "hipMemGetInfo",
    "hipMemMap",
    "hipMemRelease",
    "hipMemRetainAllocationHandle",
    "hipMemSetAccess",
    "hipMemUnmap",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemset",
    "hipMemsetAsync",
    "hipSetDevice",
    "hipStreamCreateWithFlags",
    "hipStreamDestroy",
    "hipStreamSynchronize",
    "hipblasCreate",
    "hipblasDestroy",
    "hipblasGemmEx",
    "hipblasLtCreate",
    "hipblasLtDestroy",
    "hipblasLtMatmul",
    "hipblasLtMatmulAlgoGetHeuristic",
    "hipblasLtMatmulDescCreate",
    "hipblasLtMatmulDescDestroy",
    "hipblasLtMatmulDescSetAttribute",
    "hipblasLtMatmulPreferenceCreate",
    "hipblasLtMatmulPreferenceDestroy",
    "hipblasLtMatmulPreferenceSetAttribute",
    "hipblasLtMatrixLayoutCreate",
    "hipblasLtMatrixLayoutDestroy",
    "hipblasSetStream",
)
_MAX_SYMBOL_DIAGNOSTIC_ITEMS = 16
_MAX_SYMBOL_DIAGNOSTIC_NAME_LENGTH = 96


class RuntimeContractError(RuntimeError):
    pass


ProcessIdentity = tuple[int, int]
"""A Linux process identity: ``(pid, /proc/<pid>/stat start_time)``."""

ProcRecord = tuple[int, int, int, int]
"""A /proc sample: ``(parent_pid, process_group, rss_bytes, start_time)``."""

ProcSnapshot = dict[int, ProcRecord]


@dataclass(frozen=True)
class _DirectoryBinding:
    """An opened directory and the name which must continue to point to it."""

    path: Path
    parent_fd: int
    name: str
    fd: int


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise RuntimeContractError(f"JSON contract exceeds the bounded input size: {path}")
    try:
        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise RuntimeContractError(f"duplicate JSON key {key} in {path}")
                result[key] = item
            return result
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeContractError(f"JSON document is not an object: {path}")
    return value


def git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeContractError(f"git {' '.join(args)} exceeded its bounded inspection") from exc
    if result.returncode != 0:
        raise RuntimeContractError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    if len(result.stdout) + len(result.stderr) > 16 * 1024 * 1024:
        raise RuntimeContractError(f"git {' '.join(args)} produced unbounded inspection output")
    return result.stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    commit, tree = git(repo, "rev-parse", "HEAD"), git(repo, "rev-parse", "HEAD^{tree}")
    if not SHA40.fullmatch(commit) or not SHA40.fullmatch(tree):
        raise RuntimeContractError("checked-out identity is not a full immutable SHA/tree pair")
    return commit, tree


def require_clean_checkout(repo: Path) -> None:
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeContractError("strict H3 public-runtime compile rejects a dirty checkout")


def _absolute_path_without_resolution(path: Path) -> Path:
    """Make a lexical absolute path without following any symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path, label: str) -> Path:
    """Reject a symlink in every existing component, including the leaf."""

    absolute = _absolute_path_without_resolution(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            # The rest of the path cannot contain an existing symlink until
            # this missing component is created.  Callers re-check after mkdir.
            break
        except OSError as exc:
            raise RuntimeContractError(f"cannot inspect {label} path component: {current}") from exc
        if stat.S_ISLNK(mode):
            raise RuntimeContractError(f"{label} contains a symlink component: {current}")
    return absolute


def _require_within(path: Path, root: Path, label: str) -> Path:
    """Return a regular path only when it stays inside the canonical root."""

    lexical = _reject_symlink_components(path, label)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeContractError(f"{label} escapes the workspace root: {path}") from exc
    return lexical


def require_regular(path: Path, label: str, *, within: Path | None = None) -> None:
    if within is not None:
        path = _require_within(path, within, label)
    else:
        path = _reject_symlink_components(path, label)
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise RuntimeContractError(f"{label} is missing, symlinked, or not a regular file: {path}")


def _validate_output_directory(path: Path, repo: Path) -> Path:
    """Validate an output path before and after directory creation."""

    lexical = _reject_symlink_components(path, "output directory")
    if lexical == Path("/") or lexical == repo:
        raise RuntimeContractError("output directory must be private and outside the source tree")
    try:
        resolved = lexical.resolve(strict=False)
        repo_resolved = repo.resolve(strict=True)
        if resolved == repo_resolved or resolved.is_relative_to(repo_resolved):
            raise RuntimeContractError("output directory must be outside the source tree")
    except (OSError, RuntimeError) as exc:
        raise RuntimeContractError("output directory cannot be resolved safely") from exc
    if lexical.exists() and not lexical.is_dir():
        raise RuntimeContractError("output directory is not a directory")
    return lexical


def _same_directory_entry(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _verify_directory_bindings(bindings: list[_DirectoryBinding]) -> None:
    """Reject an ancestor or leaf replacement after its directory was opened."""

    for binding in bindings:
        try:
            path_stat = os.stat(binding.name, dir_fd=binding.parent_fd, follow_symlinks=False)
            fd_stat = os.fstat(binding.fd)
        except OSError as exc:
            raise RuntimeContractError(f"runner output path was replaced: {binding.path}") from exc
        if not stat.S_ISDIR(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
            raise RuntimeContractError(f"runner output path was replaced: {binding.path}")


def _open_bound_directory(
    path: Path,
    *,
    create_leaf: bool,
    repo: Path | None = None,
) -> tuple[Path, int, list[int], list[_DirectoryBinding]]:
    """Open a lexical directory using descriptor-relative no-follow traversal.

    The returned directory FD, rather than the pathname, is the authority for
    all later runner-owned publication.  Holding every component FD also
    makes an ancestor replacement harmless: the old inode remains the bound
    destination and the replacement is detected before publication.
    """

    if (
        sys.platform != "linux"
        or os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_TMPFILE")
    ):
        raise RuntimeContractError("runner publication requires Linux directory no-follow support")
    absolute = _absolute_path_without_resolution(path)
    if repo is not None:
        repo_absolute = _absolute_path_without_resolution(repo)
        if absolute == repo_absolute or absolute.is_relative_to(repo_absolute):
            raise RuntimeContractError("output directory must be outside the source tree")
    components = list(absolute.parts[1:])
    opened: list[int] = []
    bindings: list[_DirectoryBinding] = []
    try:
        current_fd = os.open("/", _DIRECTORY_FLAGS)
        opened.append(current_fd)
        for index, component in enumerate(components):
            parent_fd = current_fd
            current_path = Path(absolute.anchor).joinpath(*components[: index + 1])
            try:
                entry_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                if not create_leaf or index != len(components) - 1:
                    raise RuntimeContractError(f"runner output has a missing ancestor: {current_path}") from exc
                try:
                    os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except FileExistsError:
                    pass
                except OSError as mkdir_exc:
                    raise RuntimeContractError(f"runner output directory could not be created: {current_path}") from mkdir_exc
                try:
                    entry_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                except OSError as stat_exc:
                    raise RuntimeContractError(f"runner output directory disappeared after mkdir: {current_path}") from stat_exc
            except OSError as exc:
                raise RuntimeContractError(f"runner output ancestor cannot be inspected: {current_path}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise RuntimeContractError(f"runner output traverses a symlink: {current_path}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise RuntimeContractError(f"runner output component is not a directory: {current_path}")
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
                child_stat = os.fstat(child_fd)
            except OSError as exc:
                raise RuntimeContractError(f"runner output component cannot be opened safely: {current_path}") from exc
            opened.append(child_fd)
            if not _same_directory_entry(entry_stat, child_stat):
                raise RuntimeContractError(f"runner output component was replaced while opening: {current_path}")
            bindings.append(_DirectoryBinding(current_path, parent_fd, component, child_fd))
            current_fd = child_fd
        return absolute, current_fd, opened, bindings
    except Exception:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _open_bound_child_directory(
    parent_path: Path,
    parent_fd: int,
    name: str,
    bindings: list[_DirectoryBinding],
) -> tuple[Path, int]:
    """Create/open one private child directory relative to its bound parent."""

    try:
        entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeContractError(f"runner build directory could not be created: {parent_path / name}") from exc
    except OSError as exc:
        raise RuntimeContractError(f"runner build directory cannot be inspected: {parent_path / name}") from exc
    if stat.S_ISLNK(entry_stat.st_mode) or not stat.S_ISDIR(entry_stat.st_mode):
        raise RuntimeContractError(f"runner build directory is not a private directory: {parent_path / name}")
    try:
        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        child_stat = os.fstat(child_fd)
    except OSError as exc:
        if "child_fd" in locals():
            try:
                os.close(child_fd)
            except OSError:
                pass
        raise RuntimeContractError(f"runner build directory cannot be opened safely: {parent_path / name}") from exc
    if not _same_directory_entry(entry_stat, child_stat):
        os.close(child_fd)
        raise RuntimeContractError(f"runner build directory was replaced while opening: {parent_path / name}")
    bindings.append(_DirectoryBinding(parent_path / name, parent_fd, name, child_fd))
    return parent_path / name, child_fd


def _directory_name(path: Path, label: str) -> str:
    name = path.name
    if not name or name in {".", ".."} or "/" in name or "\x00" in name:
        raise RuntimeContractError(f"{label} is not a single local filename: {path}")
    return name


def _open_regular_input(directory_fd: int, name: str, label: str) -> tuple[int, os.stat_result]:
    try:
        fd = os.open(name, _INPUT_FLAGS, dir_fd=directory_fd)
        file_stat = os.fstat(fd)
    except OSError as exc:
        raise RuntimeContractError(f"{label} cannot be opened as an inode-bound input") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        os.close(fd)
        raise RuntimeContractError(f"{label} is not a regular file")
    return fd, file_stat


def _verify_input_entry(directory_fd: int, name: str, file_stat: os.stat_result, label: str) -> None:
    try:
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeContractError(f"{label} disappeared or was replaced") from exc
    if not stat.S_ISREG(path_stat.st_mode) or not _same_directory_entry(path_stat, file_stat):
        raise RuntimeContractError(f"{label} was replaced while being inspected")


def _require_bound_output(directory_fd: int, name: str, label: str, maximum: int) -> None:
    fd, file_stat = _open_regular_input(directory_fd, name, label)
    try:
        if file_stat.st_size < 1 or file_stat.st_size > maximum:
            raise RuntimeContractError(f"compiler output exceeds the bounded artifact size: {name}")
        _verify_input_entry(directory_fd, name, file_stat, label)
    finally:
        os.close(fd)


def _sha256_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)


def _open_anonymous_file(directory_fd: int) -> int:
    try:
        return os.open(".", _TMPFILE_FLAGS, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise RuntimeContractError("runner publication requires filesystem O_TMPFILE support") from exc


def _write_and_sync(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("short write while publishing runner output")
        view = view[written:]
    os.fsync(fd)


def _link_fd_no_replace(source_fd: int, directory_fd: int, name: str) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = libc.linkat
        linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        linkat.restype = ctypes.c_int
        result = linkat(source_fd, b"", directory_fd, os.fsencode(name), _AT_EMPTY_PATH)
    except (AttributeError, OSError) as exc:
        raise RuntimeContractError("runner publication lacks Linux linkat support") from exc
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise RuntimeContractError(f"runner output already contains {name}")
        raise RuntimeContractError(f"runner publication failed for {name}: errno {error_number}")


def _unlink_owned_file(fd: int, directory_fd: int, name: str) -> None:
    try:
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        fd_stat = os.fstat(fd)
    except (FileNotFoundError, OSError):
        return
    if not stat.S_ISREG(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
        return
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError:
        pass


def _verify_published_file(fd: int, directory_fd: int, name: str) -> None:
    try:
        path_stat = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        fd_stat = os.fstat(fd)
    except OSError as exc:
        raise RuntimeContractError(f"runner publication disappeared or was replaced: {name}") from exc
    if not stat.S_ISREG(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
        raise RuntimeContractError(f"runner publication was replaced or symlinked: {name}")


def _publish_bytes(directory_fd: int, bindings: list[_DirectoryBinding], name: str, payload: bytes) -> int:
    """Publish one immutable payload by descriptor-relative no-replace link."""

    _verify_directory_bindings(bindings)
    fd = _open_anonymous_file(directory_fd)
    linked = False
    try:
        _write_and_sync(fd, payload)
        _verify_directory_bindings(bindings)
        try:
            _link_fd_no_replace(fd, directory_fd, name)
        except Exception:
            # The linkat wrapper may report an exception after the kernel has
            # successfully linked the inode.  Ownership-by-inode cleanup is
            # therefore required on both sides of the call.
            _unlink_owned_file(fd, directory_fd, name)
            raise
        linked = True
        try:
            _verify_published_file(fd, directory_fd, name)
            _verify_directory_bindings(bindings)
        except Exception:
            _unlink_owned_file(fd, directory_fd, name)
            raise
        os.fsync(directory_fd)
        return fd
    except Exception:
        if linked:
            _unlink_owned_file(fd, directory_fd, name)
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _check_hashed_source_inventory(
    repo: Path,
    inventory: dict[str, Any],
    expected_paths: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    files = inventory["files"]
    paths = tuple(item["path"] for item in files)
    if paths != expected_paths or inventory["canonical_order"] != list(expected_paths):
        raise RuntimeContractError(f"{label} is missing, reordered, or contains an extra/duplicate source")
    observed: list[dict[str, str]] = []
    for item in files:
        path = repo / item["path"]
        require_regular(path, f"source {item['path']}", within=repo)
        digest = sha256_file(path)
        if digest != item["sha256"]:
            raise RuntimeContractError(f"source hash mismatch: {item['path']}")
        observed.append({"path": item["path"], "sha256": digest})
    if sha256_json(observed) != inventory["source_set_sha256"]:
        raise RuntimeContractError(f"canonical {label} hash is stale")
    return {"canonical_order": list(expected_paths), "source_set_sha256": inventory["source_set_sha256"], "files": observed}


def check_source_set(repo: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    return _check_hashed_source_inventory(repo, matrix["sources"], EXPECTED_SOURCE_PATHS, "public-runtime source set")


def check_direct_compile_sources(repo: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    return _check_hashed_source_inventory(
        repo,
        matrix["direct_compile_sources"],
        EXPECTED_DIRECT_COMPILE_SOURCE_PATHS,
        "direct compile source inventory",
    )


PUBLIC_RUNTIME_LINK_LIBRARIES = (
    "/opt/rocm/lib/libamdhip64.so",
    "/opt/rocm/lib/libhipblas.so",
    "/opt/rocm/lib/libhipblaslt.so",
    "/opt/rocm/lib/librocblas.so",
)


def validate_native_link_contract(cmake_text: str) -> None:
    expected = """if(SLLM_ENABLE_PUBLIC_HIP_RUNTIME)
        target_link_libraries(sllm_hip_stub PRIVATE
            \"${ROCM_PATH}/lib/libhipblas.so\"
            \"${ROCM_PATH}/lib/libhipblaslt.so\"
            \"${ROCM_PATH}/lib/librocblas.so\"
        )
    endif()"""
    if expected not in cmake_text:
        raise RuntimeContractError(
            "native public-runtime link contract must include hipBLAS then hipBLASLt then rocBLAS"
        )


def expected_build_commands() -> list[list[str]]:
    """Return the exact five argv templates for one generic public-runtime row."""

    return [
        [
            "/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17",
            "-I", "{repo}/include", "-I", "{repo}/native/hip/src", "--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "-o",
            "{build_dir}/hip-compile-probe-{target}.o", "-x", "hip", "-c",
            "{repo}/native/hip/src/hip_compile_probe.hip.cpp",
        ],
        [
            "/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17",
            "-I", "{repo}/include", "-I", "{repo}/native/hip/src", "--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "-o",
            "{build_dir}/public-runtime-{target}.o", "-x", "hip", "-c",
            "{repo}/native/hip/src/public_runtime.hip.cpp",
        ],
        [
            "/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17",
            "-I", "{repo}/include", "-I", "{repo}/native/hip/src", "--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "-pthread", "-o",
            "{build_dir}/rmsnorm-kernel-{target}.o", "-x", "hip", "-c",
            "{repo}/native/hip/src/rmsnorm_kernel.hip.cpp",
        ],
        [
            "/opt/rocm/bin/amdclang++", "-O3", "-DNDEBUG", "-std=gnu++17",
            "-I", "{repo}/include", "-I", "{repo}/native/hip/src", "--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "-pthread", "-o",
            "{build_dir}/rmsnorm-api-{target}.o", "-c", "{repo}/native/hip/src/rmsnorm_api.cpp",
        ],
        [
            "/opt/rocm/bin/amdclang++", "-O3", "-DNDEBUG", "--offload-arch={target}",
            "-mcode-object-version=6", "-mno-wavefrontsize64", "--hip-link", "--rtlib=compiler-rt",
            "-unwindlib=libgcc", "-pthread", "-nostartfiles",
            "{build_dir}/hip-compile-probe-{target}.o", "{build_dir}/public-runtime-{target}.o",
            "{build_dir}/rmsnorm-kernel-{target}.o", "{build_dir}/rmsnorm-api-{target}.o",
            "-D__HIP_ROCclr__=1", "-std=gnu++17", "-I", "{repo}/include", "-I", "{repo}/native/hip/src",
            "-x", "c++",
            *[f"{{repo}}/{path}" for path in PUBLIC_RUNTIME_API_SOURCE_PATHS if path != "native/hip/src/rmsnorm_api.cpp"],
            "-x", "hip",
            *[f"{{repo}}/{path}" for path in PUBLIC_RUNTIME_KERNEL_SOURCE_PATHS if path != "native/hip/src/rmsnorm_kernel.hip.cpp"],
            "-x", "none",
            "-o", "{build_dir}/public-runtime-{target}.elf", *PUBLIC_RUNTIME_LINK_LIBRARIES,
        ],
    ]


def validate_matrix(repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    validate_native_link_contract((repo / "native/hip/CMakeLists.txt").read_text(encoding="utf-8"))
    validate_public_symbol_contract(repo)
    matrix = read_json(repo / "ci/matrix/hip-runtime-compile-v1.json")
    toolchain = read_json(repo / "ci/toolchains/rocm-7.14.0.json")
    if set(matrix) != {"$schema", "schema_version", "matrix_id", "revision", "toolchain_id", "container", "sources", "direct_compile_sources", "public_abi_symbols", "targets", "rows"}:
        raise RuntimeContractError("public-runtime matrix has missing or unknown top-level fields")
    if matrix.get("schema_version") != "hip-runtime-compile-v1" or matrix.get("matrix_id") != "hip-runtime-compile-v1" or matrix.get("revision") != 8:
        raise RuntimeContractError("public-runtime matrix identity is invalid")
    if matrix.get("toolchain_id") != "rocm-7.14.0" or matrix.get("targets") != list(TARGETS):
        raise RuntimeContractError("public-runtime matrix is not bound to ROCm 7.14.0 and the exact two targets")
    image = toolchain.get("image", {})
    expected_image = {
        "repository": "docker.io/rocm/dev-ubuntu-24.04", "tag": "7.14.0-full",
        "manifest_digest": PINNED_IMAGE.split("@", 1)[1], "config_digest": PINNED_CONFIG,
        "platform": {"os": "linux", "architecture": "amd64"},
    }
    for key, value in expected_image.items():
        if image.get(key) != value:
            raise RuntimeContractError(f"toolchain image is not pinned for {key}")
    if toolchain.get("schema_version") != "rocm-toolchain-v1" or toolchain.get("toolchain_id") != "rocm-7.14.0":
        raise RuntimeContractError("ROCm toolchain manifest is not v1/7.14.0")
    if toolchain.get("rocm") != {"path": "/opt/rocm", "version": "7.14.0", "llvm_major": 23}:
        raise RuntimeContractError("ROCm root/version/LLVM tuple is not canonical")
    if matrix.get("container") != {"image_reference": PINNED_IMAGE, "image_config_digest": PINNED_CONFIG, "platform": {"os": "linux", "architecture": "amd64"}, "rocm_root": "/opt/rocm", "compiler": "/opt/rocm/bin/amdclang++", "llvm_major": 23}:
        raise RuntimeContractError("public-runtime matrix container/tool paths are not canonical")
    if matrix.get("public_abi_symbols") != list(sorted(PUBLIC_SYMBOLS)):
        raise RuntimeContractError("public-runtime matrix ABI symbol set is not canonical")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeContractError("public-runtime matrix must contain exactly two rows")
    by_id: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeContractError("public-runtime matrix row is not an object")
        target, row_id = row.get("target"), row.get("row_id")
        if target not in TARGETS or row_id != f"h3-public-{target}" or row_id in by_id:
            raise RuntimeContractError("public-runtime matrix has missing, duplicate, or unknown rows")
        if row.get("tier") != "tier_h3_public_runtime" or row.get("required") is not False or row.get("seed") != int(target[3:]):
            raise RuntimeContractError(f"{row_id} is not an explicit non-required exact row")
        if set(row) != {"row_id", "target", "tier", "required", "seed", "execution", "build", "resource", "output", "codegen"}:
            raise RuntimeContractError(f"{row_id} has missing or unknown fields")
        if row.get("execution") != {"mode": "compile-only", "compile_only": True, "requires_gpu": False, "requires_model": False, "network": False, "fallback_allowed": False, "execution_attempted": False, "gpu_execution": False}:
            raise RuntimeContractError(f"{row_id} is not compile-only and no-fallback")
        codegen = row.get("codegen", {})
        expected_codegen = {"target": target, "target_kind": "exact", "target_count": 1, "code_object_version": "V6", "wavefront_size": 32, "features": EXPECTED_FEATURES, "e_flags": E_FLAGS[target]}
        if codegen != expected_codegen:
            raise RuntimeContractError(f"{row_id} has a wrong target/codegen tuple")
        if row.get("resource") != {"max_rss_bytes": 4294967296, "max_output_bytes": 16777216, "timeout_seconds": 900, "max_output_file_bytes": 268435456}:
            raise RuntimeContractError(f"{row_id} resource bounds are not canonical")
        build = row.get("build", {})
        if build.get("generator") != "direct-amdclang++" or build.get("mode") != "compile-link" or build.get("build_type") != "Release" or build.get("language_standard") != "gnu++17":
            raise RuntimeContractError(f"{row_id} build mode is not direct Release amdclang++")
        if set(build) != {"generator", "mode", "build_type", "language_standard", "probe_source", "public_runtime_source", "public_runtime_header", "rmsnorm_kernel_source", "rmsnorm_kernel_header", "rmsnorm_api_source", "rmsnorm_api_header", "link_library", "commands"}:
            raise RuntimeContractError(f"{row_id} build fields are missing or unknown")
        if build.get("commands") != expected_build_commands():
            raise RuntimeContractError(f"{row_id} commands are not the exact five-command generic build")
        if build.get("probe_source") != "native/hip/src/hip_compile_probe.hip.cpp" or build.get("public_runtime_source") != "native/hip/src/public_runtime.hip.cpp" or build.get("public_runtime_header") != "native/hip/src/public_runtime_internal.hpp" or build.get("rmsnorm_kernel_source") != "native/hip/src/rmsnorm_kernel.hip.cpp" or build.get("rmsnorm_kernel_header") != "native/hip/src/rmsnorm_kernel_internal.hpp" or build.get("rmsnorm_api_source") != "native/hip/src/rmsnorm_api.cpp" or build.get("rmsnorm_api_header") != "native/hip/src/rmsnorm_api.hpp" or build.get("link_library") != "/opt/rocm/lib/libamdhip64.so":
            raise RuntimeContractError(f"{row_id} source paths are not the public-runtime tuple")
        if row.get("output") != EXPECTED_OUTPUT:
            raise RuntimeContractError(f"{row_id} output contract is not row-private and versioned")
        by_id[row_id] = row
    if set(by_id) != {f"h3-public-{target}" for target in TARGETS}:
        raise RuntimeContractError("public-runtime matrix target rows are incomplete")
    source_set = check_source_set(repo, matrix)
    check_direct_compile_sources(repo, matrix)
    return toolchain, matrix, by_id


def command_version(path: Path) -> str:
    try:
        result = subprocess.run([str(path), "--version"], text=True, capture_output=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeContractError(f"cannot query pinned tool {path}") from exc
    if result.returncode != 0 or len(result.stdout) + len(result.stderr) > 1024 * 1024:
        raise RuntimeContractError(f"cannot query pinned tool {path}")
    return (result.stdout or result.stderr).strip()


def inspect_toolchain(toolchain: dict[str, Any]) -> dict[str, Any]:
    paths = toolchain["paths"]
    required = {"rocm_root", "compiler", "hip_headers", "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy", "llvm_readobj"}
    if set(paths) != {"rocm_root", "compiler", "hip_headers", "hip_cmake_package", "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump"}:
        raise RuntimeContractError("toolchain path set is missing or has extra entries")
    for name in required:
        path = Path(paths[name])
        try:
            path.resolve().relative_to(Path("/opt/rocm").resolve())
        except ValueError as exc:
            raise RuntimeContractError(f"pinned ROCm path resolves outside /opt/rocm: {name}") from exc
        if not path.is_absolute() or not path.exists():
            raise RuntimeContractError(f"pinned ROCm path is missing/outside /opt/rocm: {name}")
    compiler = Path(paths["compiler"])
    if compiler.name != "amdclang++" or not os.access(compiler, os.X_OK) or not re.search(r"clang version 23\.", command_version(compiler), re.IGNORECASE):
        raise RuntimeContractError("pinned compiler is not executable AMD clang major 23")
    observed: dict[str, str] = {}
    for name in ("clang_offload_bundler", "llvm_objcopy", "llvm_readobj"):
        path = Path(paths[name])
        if not os.access(path, os.X_OK):
            raise RuntimeContractError(f"pinned LLVM tool is not executable: {name}")
        observed[name] = command_version(path)
    versions = [Path("/opt/rocm/.info/version"), Path("/opt/rocm/core-7.14/.info/version")]
    if not any(path.is_file() and "7.14.0" in path.read_text(encoding="utf-8").strip() for path in versions):
        raise RuntimeContractError("ROCm 7.14.0 was not observed at /opt/rocm")
    return {"toolchain_id": "rocm-7.14.0", "manifest_sha256": sha256_json(toolchain), "rocm": toolchain["rocm"], "compiler": toolchain["compiler"], "paths": paths, "observed": observed}


def network_isolated() -> bool:
    try:
        names = [name for _, name in socket.if_nameindex()]
    except OSError:
        return False
    if names != ["lo"]:
        return False
    try:
        route_lines = Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]
        return not any(line.split()[0] != "lo" for line in route_lines if line.split())
    except (OSError, UnicodeError):
        return False


def limit_address_space(limit: int) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


OUTPUT_LIMIT_EXIT = 125
RSS_LIMIT_EXIT = 126
DESCENDANT_EXIT = 127
_NATURAL_DRAIN_SECONDS = 0.25
_NATURAL_DRAIN_SELECT_SECONDS = 0.01
# A full /proc snapshot is not cancellable.  Reserve one natural-drain window
# before starting one when the argv hard deadline is the limiting deadline.
_NATURAL_DRAIN_SNAPSHOT_GUARD_SECONDS = _NATURAL_DRAIN_SECONDS
_TERMINATE_AND_REAP_SECONDS = 1.0
_EMERGENCY_CLEANUP_SECONDS = 1.0
_UNBOUND_PRIVATE_CLEANUP_SECONDS = 1.0
_POST_DEADLINE_REAP_SECONDS = 0.05
_CLEANUP_DIAGNOSTIC_BYTES = 2048

_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2
_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_ARCH_X86_64 = 0xC000003E
_SECCOMP_DATA_NR_OFFSET = 0
_SECCOMP_DATA_ARCH_OFFSET = 4
_SECCOMP_DATA_ARG0_OFFSET = 16
_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_JSET = 0x40
_BPF_K = 0x00
_BPF_RET = 0x06
_SYS_SETPGID = 109
_SYS_SETSID = 112
_SYS_UNSHARE = 272
_SYS_SETNS = 308
_SYS_CLONE = 56
_SYS_CLONE3 = 435
_CLONE_NEWNS = 0x00020000
_CLONE_NEWUTS = 0x04000000
_CLONE_NEWIPC = 0x08000000
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWPID = 0x20000000
_CLONE_NEWNET = 0x40000000
_CLONE_NEWCGROUP = 0x02000000
_CLONE_NEWTIME = 0x00000080
_CLONE_NAMESPACE_FLAGS = (
    _CLONE_NEWNS
    | _CLONE_NEWUTS
    | _CLONE_NEWIPC
    | _CLONE_NEWUSER
    | _CLONE_NEWPID
    | _CLONE_NEWNET
    | _CLONE_NEWCGROUP
    | _CLONE_NEWTIME
)
_SUBREAPER_LOCK = threading.RLock()
_LIBC: Any | None = None


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]

# AMDGPU ELF V6 e_flags fields.  The exact target occupies the low byte;
# feature states are encoded in the remaining fields.  The two H3 targets
# have no XNACK/SRAMECC support, so the observed zero fields mean
# ``unsupported``.  This is decoded from llvm-readobj output below; the
# matrix tuple is never used as a substitute for ELF evidence.
AMDGPU_MACH_MASK = 0x000000FF
AMDGPU_XNACK_MASK = 0x00000300
AMDGPU_SRAMECC_MASK = 0x00000C00
AMDGPU_GENERIC_VERSION_MASK = 0xFF000000
AMDGPU_GENERIC_VERSION_SHIFT = 24
AMDGPU_XNACK_STATES = {0x000: "unsupported", 0x100: "any", 0x200: "off", 0x300: "on"}
AMDGPU_SRAMECC_STATES = {0x000: "unsupported", 0x400: "any", 0x800: "off", 0xC00: "on"}


def _libc() -> Any:
    global _LIBC
    if _LIBC is None:
        try:
            _LIBC = ctypes.CDLL(None, use_errno=True)
            _LIBC.prctl.restype = ctypes.c_int
        except OSError as exc:
            raise RuntimeContractError("Linux child-subreaper support is unavailable") from exc
    return _LIBC


def _child_subreaper_enabled() -> bool:
    value = ctypes.c_int()
    result = _libc().prctl(_PR_GET_CHILD_SUBREAPER, ctypes.byref(value), 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise RuntimeContractError(f"cannot inspect Linux child-subreaper state: errno={error}")
    return bool(value.value)


def _set_child_subreaper(enabled: bool) -> None:
    result = _libc().prctl(_PR_SET_CHILD_SUBREAPER, int(enabled), 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise RuntimeContractError(f"cannot set Linux child-subreaper state: errno={error}")


def _require_linux_amd64() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeContractError("child containment requires the exact Linux/amd64 contract")


def _bpf(code: int, *, jt: int = 0, jf: int = 0, k: int = 0) -> _SockFilter:
    return _SockFilter(code, jt, jf, k)


def _containment_filter() -> list[_SockFilter]:
    """Build the inherited Linux/amd64 filter protecting the root session.

    The filter is intentionally narrow about ordinary process creation: clone
    remains available for threads and fork-like subprocess behavior, but all
    namespace-bearing clone flags are denied.  clone3 is made to look absent
    so glibc can use its normal clone fallback without allowing a user pointer
    to evade classic-BPF argument inspection.
    """

    blocked_errno = _SECCOMP_RET_ERRNO | errno.EPERM
    clone3_unavailable = _SECCOMP_RET_ERRNO | errno.ENOSYS
    program = [
        _bpf(_BPF_LD | _BPF_W | _BPF_ABS, k=_SECCOMP_DATA_ARCH_OFFSET),
        _bpf(_BPF_JMP | _BPF_JEQ | _BPF_K, jt=1, k=_SECCOMP_ARCH_X86_64),
        _bpf(_BPF_RET | _BPF_K, k=_SECCOMP_RET_KILL_PROCESS),
        _bpf(_BPF_LD | _BPF_W | _BPF_ABS, k=_SECCOMP_DATA_NR_OFFSET),
    ]
    for syscall_number in (_SYS_SETPGID, _SYS_SETSID, _SYS_UNSHARE, _SYS_SETNS):
        program.extend(
            (
                _bpf(_BPF_JMP | _BPF_JEQ | _BPF_K, jf=1, k=syscall_number),
                _bpf(_BPF_RET | _BPF_K, k=blocked_errno),
            )
        )
    program.extend(
        (
            _bpf(_BPF_JMP | _BPF_JEQ | _BPF_K, jf=1, k=_SYS_CLONE3),
            _bpf(_BPF_RET | _BPF_K, k=clone3_unavailable),
            _bpf(_BPF_JMP | _BPF_JEQ | _BPF_K, jf=3, k=_SYS_CLONE),
            _bpf(_BPF_LD | _BPF_W | _BPF_ABS, k=_SECCOMP_DATA_ARG0_OFFSET),
            _bpf(_BPF_JMP | _BPF_JSET | _BPF_K, jf=1, k=_CLONE_NAMESPACE_FLAGS),
            _bpf(_BPF_RET | _BPF_K, k=blocked_errno),
            _bpf(_BPF_RET | _BPF_K, k=_SECCOMP_RET_ALLOW),
        )
    )
    return program


def _seccomp_syscall_errno(syscall_number: int, *arguments: int) -> int | None:
    """Return errno for a raw syscall, or None when it succeeds."""

    syscall = getattr(_libc(), "syscall", None)
    if syscall is None:
        raise RuntimeContractError("Linux syscall self-test is unavailable")
    ctypes.set_errno(0)
    result = syscall(ctypes.c_long(syscall_number), *(ctypes.c_long(value) for value in arguments))
    if result != -1:
        return None
    return ctypes.get_errno()


def _containment_self_test() -> None:
    """Prove the inherited filter blocks escape operations before exec."""

    probe_pid = os.fork()
    if probe_pid == 0:
        try:
            try:
                os.setsid()
            except OSError as exc:
                if exc.errno != errno.EPERM:
                    os._exit(101)
            else:
                os._exit(102)
            try:
                os.setpgid(0, 0)
            except OSError as exc:
                if exc.errno != errno.EPERM:
                    os._exit(103)
            else:
                os._exit(104)
            if _seccomp_syscall_errno(_SYS_UNSHARE, 0) != errno.EPERM:
                os._exit(105)
            os._exit(0)
        except BaseException:
            os._exit(106)
    try:
        waited_pid, status = os.waitpid(probe_pid, 0)
    except (ChildProcessError, OSError) as exc:
        raise RuntimeContractError("child containment self-test could not reap its probe") from exc
    if waited_pid != probe_pid or not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeContractError("child containment self-test did not deny session/group/namespace escape")


def _install_child_containment() -> None:
    """Install an unprivileged, inherited containment boundary before exec."""

    _require_linux_amd64()
    try:
        if os.getsid(0) != os.getpid() or os.getpgid(0) != os.getpid():
            raise RuntimeContractError("private root session was not established before containment")
    except OSError as exc:
        raise RuntimeContractError("cannot verify private root session before containment") from exc
    if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise RuntimeContractError(f"cannot enable no_new_privs for child containment: errno={error}")
    filter_values = _containment_filter()
    filters = (_SockFilter * len(filter_values))(*filter_values)
    program = _SockFprog(len(filters), ctypes.cast(filters, ctypes.POINTER(_SockFilter)))
    if _libc().prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        error = ctypes.get_errno()
        raise RuntimeContractError(f"cannot install inherited child containment: errno={error}")
    _containment_self_test()


@contextmanager
def _process_observation_scope() -> Any:
    """Serialize subreaper use and capture a no-touch baseline.

    PR_SET_CHILD_SUBREAPER is process-global.  The reentrant lock makes nested
    runner calls preserve the outer state, while the baseline excludes every
    process that existed before this invocation from discovery and cleanup.
    """

    if sys.platform != "linux":
        raise RuntimeContractError("Linux /proc process observation is required")
    with _SUBREAPER_LOCK:
        previous = _child_subreaper_enabled()
        if not previous:
            _set_child_subreaper(True)
        try:
            baseline = _proc_snapshot()
            runner_identity = _snapshot_identity(baseline, os.getpid())
            if runner_identity is None:
                raise RuntimeContractError("/proc observation is unavailable for the runner process")
            yield _snapshot_identities(baseline)
        finally:
            if not previous:
                _set_child_subreaper(False)


def _proc_snapshot() -> ProcSnapshot:
    """Return pid -> (parent pid, process-group id, RSS bytes, start time).

    A process disappearing between the directory listing and either proc file
    read is a safe race and is ignored.  Failure to enumerate /proc or read a
    still-present entry is different: resource and descendant guarantees then
    cannot be made, so the runner fails closed.
    """

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        if page_size <= 0:
            raise ValueError("non-positive page size")
    except (OSError, ValueError) as exc:
        raise RuntimeContractError("/proc observation is unavailable: page size") from exc
    snapshot: ProcSnapshot = {}
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeContractError("/proc observation is unavailable: cannot enumerate /proc") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat = (entry / "stat").read_text(encoding="ascii")
            right_paren = stat.rfind(")")
            if right_paren < 0:
                raise ValueError("malformed /proc stat")
            fields = stat[right_paren + 2 :].split()
            if len(fields) < 20:
                raise ValueError("short /proc stat")
            parent_pid, process_group = int(fields[1]), int(fields[2])
            start_time = int(fields[19])
            statm = (entry / "statm").read_text(encoding="ascii").split()
            if len(statm) < 2:
                raise ValueError("short /proc statm")
            rss = int(statm[1]) * page_size
        except FileNotFoundError:
            # The PID exited after enumeration.  It cannot be killed by this
            # invocation, and treating the disappearance as safe avoids a
            # false failure in the normal proc race window.
            continue
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeContractError(f"/proc observation failed for {entry.name}") from exc
        snapshot[pid] = (parent_pid, process_group, rss, start_time)
    return snapshot


def _identity(pid: int, record: ProcRecord) -> ProcessIdentity:
    return pid, record[3]


def _snapshot_identity(snapshot: ProcSnapshot, pid: int) -> ProcessIdentity | None:
    record = snapshot.get(pid)
    return None if record is None else _identity(pid, record)


def _snapshot_identities(snapshot: ProcSnapshot) -> set[ProcessIdentity]:
    return {_identity(pid, record) for pid, record in snapshot.items()}


def _read_process_identity(pid: int) -> ProcessIdentity | None:
    """Read one process identity without treating a normal exit race as fatal."""

    if pid <= 1:
        return None
    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            raise ValueError("malformed /proc stat")
        fields = stat_text[right_paren + 2 :].split()
        if len(fields) < 20:
            raise ValueError("short /proc stat")
        return pid, int(fields[19])
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeContractError(f"cannot read process identity for PID {pid}") from exc


def _read_process_group(pid: int) -> int | None:
    """Read the process group for a just-verified root identity."""

    try:
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        right_paren = stat_text.rfind(")")
        if right_paren < 0:
            raise ValueError("malformed /proc stat")
        fields = stat_text[right_paren + 2 :].split()
        if len(fields) < 3:
            raise ValueError("short /proc stat")
        return int(fields[2])
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeContractError(f"cannot read process group for PID {pid}") from exc


def _root_identity_after_spawn(root_pid: int, baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity]) -> ProcessIdentity:
    """Bind the newly-created session leader before any signal can be sent."""

    root_identity = _read_process_identity(root_pid)
    if root_identity is None or root_identity in baseline_identities:
        raise RuntimeContractError("cannot bind the post-baseline root process identity")
    if _read_process_group(root_pid) != root_pid:
        raise RuntimeContractError("spawned root did not retain its private process group")
    return root_identity


def _root_exit_status_without_reap(process: subprocess.Popen[bytes], root_identity: ProcessIdentity) -> int | None:
    """Observe the direct child exit while preserving its private-group PID.

    Keeping the root as a zombie until the first natural-drain observations
    prevents its PID and therefore its process-group number from being reused
    before a just-forked helper can be identity-proven.
    """

    if process.returncode is not None:
        return process.returncode
    if _read_process_identity(root_identity[0]) != root_identity:
        raise RuntimeContractError("root identity disappeared before Popen reaped it")
    try:
        result = os.waitid(os.P_PID, root_identity[0], os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError as exc:
        raise RuntimeContractError("root wait status was consumed outside Popen") from exc
    except OSError as exc:
        raise RuntimeContractError("cannot observe root exit status without reaping") from exc
    if result is None or result.si_pid == 0:
        return None
    if result.si_code == os.CLD_EXITED:
        return result.si_status
    if result.si_code in {os.CLD_KILLED, os.CLD_DUMPED}:
        return -result.si_status
    raise RuntimeContractError("root returned an unsupported wait status")


def _tree_from_snapshot(
    snapshot: ProcSnapshot,
    root_identity: ProcessIdentity,
    known_identities: set[ProcessIdentity],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
) -> set[ProcessIdentity]:
    """Prove the current private tree from identity, ancestry, and group data.

    A direct ``parent_pid == runner_pid`` check is deliberately absent.  A
    child adopted by the subreaper is observable only after it is tied to the
    audited private process group or to an already-proven live ancestor.
    """

    runner_pid = os.getpid()
    root_pid = root_identity[0]
    identities = _snapshot_identities(snapshot)
    root_record = snapshot.get(root_pid)
    root_is_live_private = root_record is not None and _identity(root_pid, root_record) == root_identity and root_record[1] == root_pid
    known_private = {
        identity
        for identity in known_identities & identities
        if identity[0] != runner_pid and snapshot[identity[0]][1] == root_pid
    }
    # A stale PID/PGID must never establish a new private group.  Once the
    # leader exited, an already-proven, still-live member is the only anchor.
    private_group_proven = root_is_live_private or bool(known_private)
    tree: set[ProcessIdentity] = set(known_private)
    if root_is_live_private:
        tree.add(root_identity)

    changed = True
    while changed:
        changed = False
        tree_pids = {identity[0] for identity in tree}
        for pid, record in snapshot.items():
            identity = _identity(pid, record)
            if identity in baseline_identities or pid == runner_pid or identity in tree:
                continue
            parent_pid, process_group, _rss, _start_time = record
            if (private_group_proven and process_group == root_pid) or parent_pid in tree_pids:
                tree.add(identity)
                changed = True
    return tree


def _live_process_tree(
    root_identity: ProcessIdentity,
    known_identities: set[ProcessIdentity],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity] = frozenset(),
) -> tuple[set[ProcessIdentity], int]:
    """Find only the identity-proven private tree and sum its current RSS."""

    snapshot = _proc_snapshot()
    tree = _tree_from_snapshot(snapshot, root_identity, known_identities, baseline_identities)
    return tree, sum(snapshot[identity[0]][2] for identity in tree)


def _require_pidfd_signalling_preflight() -> None:
    """Prove pidfd signalling can be used before spawning an audited child."""

    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if not callable(pidfd_open) or not callable(pidfd_send_signal):
        raise RuntimeContractError("identity-safe pidfd signalling is unavailable")
    try:
        pidfd = pidfd_open(os.getpid())
    except OSError as exc:
        raise RuntimeContractError("cannot open a pidfd for the runner preflight") from exc
    try:
        os.close(pidfd)
    except OSError as exc:
        raise RuntimeContractError("cannot close the runner pidfd preflight descriptor") from exc


def _signal_process_identities(identities: set[ProcessIdentity], signum: signal.Signals) -> None:
    """Signal only identities revalidated through a pidfd, never a bare PID."""

    if not callable(getattr(os, "pidfd_open", None)) or not callable(getattr(signal, "pidfd_send_signal", None)):
        raise RuntimeContractError("identity-safe pidfd signalling is unavailable")
    for pid, start_time in sorted(identities, reverse=True):
        if pid <= 1 or pid == os.getpid():
            continue
        try:
            pidfd = os.pidfd_open(pid)
        except ProcessLookupError:
            continue
        except OSError as exc:
            raise RuntimeContractError(f"cannot open pidfd for audited PID {pid}") from exc
        try:
            if _read_process_identity(pid) != (pid, start_time):
                continue
            try:
                signal.pidfd_send_signal(pidfd, signum)
            except ProcessLookupError:
                continue
            except PermissionError as exc:
                raise RuntimeContractError(f"cannot signal audited PID {pid}") from exc
        finally:
            os.close(pidfd)


def _reap_process_identities(
    identities: set[ProcessIdentity],
    root_identity: ProcessIdentity,
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
) -> set[ProcessIdentity]:
    """Reap identity-verified adopted descendants; Popen alone owns root wait."""

    unresolved: set[ProcessIdentity] = set()
    for pid, start_time in sorted(identities):
        identity = pid, start_time
        if pid <= 1 or pid == os.getpid() or identity == root_identity or identity in baseline_identities:
            continue
        # An unreaped child keeps its PID allocated.  Therefore, after this
        # check, waitpid can only reap this identity or return ECHILD; it can
        # never consume a PID-reused foreign process.
        if _read_process_identity(pid) != identity:
            continue
        interrupted = 0
        while True:
            try:
                waited_pid, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                # The root may still own this known descendant.  Keep the
                # identity for a bounded retry after Popen reaps the root and
                # the kernel reparents the descendant to this subreaper.
                if _read_process_identity(pid) == identity:
                    unresolved.add(identity)
                break
            except InterruptedError:
                interrupted += 1
                if interrupted >= 8:
                    unresolved.add(identity)
                    break
                continue
            except OSError:
                break
            if waited_pid == 0:
                unresolved.add(identity)
                break
            if waited_pid == pid:
                break
    return unresolved


def _private_cleanup_snapshot() -> ProcSnapshot:
    """Read a minimal identity-bearing snapshot for emergency cleanup only."""

    processes: ProcSnapshot = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError as exc:
        raise RuntimeContractError("cannot enumerate /proc for private cleanup") from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            stat_text = (entry / "stat").read_text(encoding="ascii")
            right_paren = stat_text.rfind(")")
            if right_paren < 0:
                raise ValueError("malformed /proc stat")
            fields = stat_text[right_paren + 2 :].split()
            if len(fields) < 20:
                raise ValueError("short /proc stat")
            processes[pid] = int(fields[1]), int(fields[2]), 0, int(fields[19])
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError, ValueError) as exc:
            raise RuntimeContractError(f"private cleanup observation failed for {entry.name}") from exc
    return processes


def _cleanup_deadline_allows_scan(deadline: float) -> bool:
    """Return whether a new non-cancellable full ``/proc`` scan may begin.

    A scan that began before this deadline can still finish after it: walking
    ``/proc`` is not cancellable.  Cleanup must never start another scan once
    that bounded-grace deadline has elapsed.
    """

    return time.monotonic() < deadline


def _best_effort_post_deadline_reap(process: subprocess.Popen[bytes]) -> None:
    """Give Popen a short ownership-preserving reap opportunity without scanning."""

    try:
        process.wait(timeout=_POST_DEADLINE_REAP_SECONDS)
    except (OSError, subprocess.SubprocessError):
        pass


def _bounded_post_deadline_reap(
    process: subprocess.Popen[bytes],
    identities: set[ProcessIdentity],
    root_identity: ProcessIdentity,
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
    *,
    hard_deadline: float | None = None,
) -> set[ProcessIdentity]:
    """Retry only already-known adopted children without another proc scan.

    Signals and wait ownership are deliberately separate: the caller has
    already signalled any identities that may be terminated, while this small
    window lets a SIGKILL become waitable.  The direct root is always reaped by
    its Popen object; ``waitpid`` is restricted to the identity-proven adopted
    descendants.
    """

    deadline = time.monotonic() + _POST_DEADLINE_REAP_SECONDS
    if hard_deadline is not None:
        deadline = min(deadline, hard_deadline)
    remaining = set(identities)
    while True:
        remaining = _reap_process_identities(remaining, root_identity, baseline_identities)
        if not remaining and process.poll() is not None:
            return set()
        now = time.monotonic()
        if now >= deadline:
            return remaining
        if process.returncode is None:
            try:
                process.wait(timeout=min(0.01, max(0.0, deadline - now)))
            except (OSError, subprocess.SubprocessError):
                pass
        else:
            time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))


def _append_cleanup_note(original: BaseException, diagnostic: str) -> None:
    """Retain a cleanup diagnostic without replacing a caller BaseException."""

    add_note = getattr(original, "add_note", None)
    if not callable(add_note):
        return
    try:
        bounded = diagnostic.encode("utf-8", "replace")[:_CLEANUP_DIAGNOSTIC_BYTES].decode("utf-8", "replace")
        add_note(f"H3 runner cleanup diagnostic: {bounded}")
    except BaseException:
        # Diagnostics must not hide KeyboardInterrupt/SystemExit or their
        # original value if a non-standard exception rejects notes.
        pass


def _emergency_cleanup_after_observation_failure(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    output_limit: int,
    output_bytes: int,
    root_identity: ProcessIdentity,
    known_identities: set[ProcessIdentity],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
) -> bool:
    """Bound cleanup when /proc observation failed after the child spawned.

    The minimal cleanup scan retains ``start_time`` and proves descendants by
    private root group or already-proven ancestry.  An arbitrary child that
    the subreaper adopted is never included solely because its PPID is now the
    runner PID.
    """

    deadline = time.monotonic() + _EMERGENCY_CLEANUP_SECONDS
    remaining: set[ProcessIdentity] = set()
    quiet_rounds = 0
    cleanup_observation_failed = False
    cleanup_proven = False
    while _cleanup_deadline_allows_scan(deadline):
        try:
            snapshot = _private_cleanup_snapshot()
            remaining = _tree_from_snapshot(snapshot, root_identity, known_identities, baseline_identities)
            known_identities.update(remaining)
            _signal_process_identities(remaining, signal.SIGKILL)
            _reap_process_identities(remaining, root_identity, baseline_identities)
        except RuntimeContractError:
            cleanup_observation_failed = True
            break
        try:
            output_bytes, _ = _drain_streams(selector, buffers, output_limit, output_bytes)
        except (OSError, ValueError):
            break
        try:
            if not selector.get_map() and process.poll() is not None:
                quiet_rounds += 1
                if quiet_rounds >= 2:
                    cleanup_proven = True
                    break
            else:
                quiet_rounds = 0
            if time.monotonic() >= deadline:
                break
            selector.select(timeout=min(0.01, max(0.0, deadline - time.monotonic())))
        except (OSError, ValueError):
            break
    # A scan which was already under way may have crossed the deadline.  The
    # identities it yielded are still safe to signal/reap, but another full
    # scan would make the nominal grace unbounded and cannot prove cleanup.
    try:
        _signal_process_identities(known_identities, signal.SIGKILL)
        _reap_process_identities(known_identities, root_identity, baseline_identities)
    except RuntimeContractError:
        cleanup_observation_failed = True
    post_deadline_remaining = _bounded_post_deadline_reap(
        process,
        known_identities,
        root_identity,
        baseline_identities,
    )
    return cleanup_proven and not cleanup_observation_failed and not post_deadline_remaining and process.poll() is not None


def _close_streams(selector: selectors.BaseSelector) -> None:
    try:
        mapping = selector.get_map()
    except ValueError:
        return
    if mapping is None:
        return
    keys = list(mapping.values())
    for key in keys:
        try:
            selector.unregister(key.fd)
        except (KeyError, ValueError):
            pass
        try:
            os.close(key.fd)
        except OSError:
            pass
    selector.close()


def _close_process_pipes(process: subprocess.Popen[bytes]) -> None:
    """Close only the runner-owned ends of a Popen stdout/stderr pipe pair."""

    for pipe in (process.stdout, process.stderr):
        if pipe is None:
            continue
        try:
            pipe.close()
        except OSError:
            pass


def _unbound_private_group_identities(snapshot: ProcSnapshot, root_pid: int) -> set[ProcessIdentity]:
    """Return post-spawn members of the one Popen-owned private process group."""

    return {
        _identity(pid, record)
        for pid, record in snapshot.items()
        if pid > 1 and pid != os.getpid() and record[1] == root_pid
    }


def _reap_unbound_private_group_members(identities: set[ProcessIdentity]) -> None:
    """Boundedly reap only adopted members of the already-killed private group.

    A member first observed in this group cannot move to another group because
    the inherited containment blocks setpgid/setsid.  An unreaped child keeps
    its PID allocated, so the identity recheck makes waitpid safe here without
    promoting a bare PID into a general-purpose signalling target.
    """

    for pid, start_time in sorted(identities):
        identity = pid, start_time
        if _read_process_identity(pid) != identity:
            continue
        interrupted = 0
        while True:
            try:
                waited_pid, _status = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            except InterruptedError:
                interrupted += 1
                if interrupted >= 8:
                    break
                continue
            except OSError:
                break
            if waited_pid == 0 or waited_pid == pid:
                break


def _bounded_reap_unbound_private_group_members(identities: set[ProcessIdentity]) -> set[ProcessIdentity]:
    """Retry waitpid only for members already proven in the private group."""

    deadline = time.monotonic() + _POST_DEADLINE_REAP_SECONDS
    remaining = set(identities)
    while True:
        unresolved: set[ProcessIdentity] = set()
        for identity in remaining:
            _reap_unbound_private_group_members({identity})
            if _read_process_identity(identity[0]) == identity:
                unresolved.add(identity)
        remaining = unresolved
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(min(0.005, max(0.0, deadline - time.monotonic())))


def _cleanup_unbound_private_session(process: subprocess.Popen[bytes], selector: selectors.BaseSelector) -> None:
    """Kill/reap the Popen-owned session when root identity binding failed.

    This is the sole deliberately unbound signal path.  ``Popen`` returned a
    direct child that has not been reaped, ``start_new_session=True`` made its
    PID its private PGID, and the successful inherited containment forbids all
    descendants from setsid/setpgid.  Until ``process.wait`` reaps that direct
    child its PID cannot be reused; after that, any surviving same-PGID member
    retains the kernel's struct pid for the group.  Consequently this helper
    can use that exact private PGID without ever treating an arbitrary bare PID
    as a signal target.
    """

    root_pid = process.pid
    deadline = time.monotonic() + _UNBOUND_PRIVATE_CLEANUP_SECONDS
    failures: list[str] = []

    def kill_private_group() -> None:
        try:
            os.killpg(root_pid, signal.SIGKILL)
        except ProcessLookupError:
            # There are no group members left, which is a successful race.
            pass
        except OSError as exc:
            failures.append(f"cannot SIGKILL unbound private process group: {exc}")

    kill_private_group()
    try:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        failures.append("unbound root did not exit after private-group SIGKILL")
    except (OSError, subprocess.SubprocessError) as exc:
        failures.append(f"cannot reap unbound root through Popen.wait: {exc}")

    remaining: set[ProcessIdentity] = set()
    known_group_identities: set[ProcessIdentity] = set()
    cleanup_proven = False
    quiet_rounds = 0
    while _cleanup_deadline_allows_scan(deadline):
        try:
            snapshot = _private_cleanup_snapshot()
        except RuntimeContractError as exc:
            failures.append(str(exc))
            break
        remaining = _unbound_private_group_identities(snapshot, root_pid)
        known_group_identities.update(remaining)
        if not remaining:
            quiet_rounds += 1
            if quiet_rounds >= 2 and process.poll() is not None:
                cleanup_proven = True
                break
        else:
            quiet_rounds = 0
        kill_private_group()
        _reap_unbound_private_group_members(remaining)
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    # Do not begin a final scan after the deadline.  Any identities observed
    # before it remain safe to reap, but cleanup is fail-closed without a
    # timely second empty observation proving that the private group is gone.
    kill_private_group()
    _reap_unbound_private_group_members(remaining)
    _best_effort_post_deadline_reap(process)
    remaining = _bounded_reap_unbound_private_group_members(known_group_identities)

    _close_streams(selector)
    _close_process_pipes(process)
    if failures or not cleanup_proven or remaining or process.returncode is None:
        details = "; ".join(failures)
        if not cleanup_proven:
            details = f"{details}; bounded cleanup could not prove private-group removal".strip("; ")
        if remaining:
            details = f"{details}; residual private-group identities={sorted(remaining)}".strip("; ")
        if process.returncode is None:
            details = f"{details}; unbound root was not reaped through Popen".strip("; ")
        raise RuntimeContractError(f"unbound private-session cleanup failed: {details}")


def _drain_streams(
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    output_limit: int,
    output_bytes: int,
) -> tuple[int, bool]:
    """Drain ready pipe bytes without retaining more than the combined cap."""

    overflow = False
    for key, _ in selector.select(timeout=0):
        stream = key.data
        while True:
            remaining = output_limit - output_bytes
            read_size = min(64 * 1024, max(1, remaining + 1))
            try:
                chunk = os.read(key.fd, read_size)
            except BlockingIOError:
                break
            except OSError:
                try:
                    selector.unregister(key.fd)
                except (KeyError, ValueError):
                    pass
                break
            if not chunk:
                try:
                    selector.unregister(key.fd)
                except (KeyError, ValueError):
                    pass
                try:
                    os.close(key.fd)
                except OSError:
                    pass
                break
            if len(chunk) > remaining:
                if remaining > 0:
                    buffers[stream].extend(chunk[:remaining])
                    output_bytes += remaining
                overflow = True
                break
            buffers[stream].extend(chunk)
            output_bytes += len(chunk)
            # At the cap, one extra byte is read on the next iteration.  This
            # distinguishes exactly-at-cap output from an actual overflow.
        if overflow:
            break
    return output_bytes, overflow


def _natural_drain_and_reap(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    output_limit: int,
    output_bytes: int,
    root_identity: ProcessIdentity,
    known_identities: set[ProcessIdentity],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
    hard_deadline: float,
) -> tuple[bool, int, bool, bool]:
    """Bound the root-success race without intervening in natural exits.

    A compiler driver can return success while one of its helper processes is
    already on the way out.  That is not a leaked descendant and must not be
    turned into ``DESCENDANT_EXIT`` merely because one /proc sample caught it.
    During this short drain we only observe, drain inherited pipes, and reap
    PIDs already observed for this invocation.  A live descendant at the
    deadline is handed to the terminating cleanup path by the caller.
    """

    # waitid(WNOWAIT) observes the successful root without consuming its wait
    # status.  Reap that direct child immediately through Popen before the
    # natural-drain scans, so its zombie is never treated as a descendant and
    # its known adopted children can be reaped independently.
    now = time.monotonic()
    deadline = min(now + _NATURAL_DRAIN_SECONDS, hard_deadline)
    # Preserve the hard-deadline guard before touching the process object: a
    # non-cancellable observation must not be entered when there is no safe
    # room left, even if the caller is exercising this helper in isolation.
    if hard_deadline - time.monotonic() < _NATURAL_DRAIN_SNAPSHOT_GUARD_SECONDS:
        return False, output_bytes, False, True
    if process.poll() is None:
        wait_budget = min(_NATURAL_DRAIN_SECONDS, max(0.0, hard_deadline - now))
        if wait_budget <= 0:
            return False, output_bytes, False, True
        try:
            process.wait(timeout=wait_budget)
        except subprocess.TimeoutExpired:
            return False, output_bytes, False, True
    while True:
        now = time.monotonic()
        if now >= deadline:
            return False, output_bytes, False, deadline == hard_deadline
        # A /proc snapshot is a full directory walk and cannot be cancelled.
        # If the argv hard deadline has too little room for one, hand the
        # already-known identities to bounded termination immediately instead
        # of beginning a scan which could make the timeout decision late.
        if hard_deadline - now < _NATURAL_DRAIN_SNAPSHOT_GUARD_SECONDS:
            return False, output_bytes, False, True
        live, _ = _live_process_tree(root_identity, known_identities, baseline_identities)
        known_identities.update(live)
        now = time.monotonic()
        if now >= hard_deadline:
            return False, output_bytes, False, True
        _reap_process_identities(live, root_identity, baseline_identities)
        output_bytes, overflow = _drain_streams(selector, buffers, output_limit, output_bytes)
        if overflow:
            return False, output_bytes, True, False
        now = time.monotonic()
        if now >= hard_deadline:
            return False, output_bytes, False, True
        if not live and not selector.get_map():
            post_deadline_remaining = _bounded_post_deadline_reap(
                process,
                known_identities,
                root_identity,
                baseline_identities,
                hard_deadline=hard_deadline,
            )
            if post_deadline_remaining:
                # A child may become waitable just after the scan that showed
                # an empty tree.  Keep using the already-known identities and
                # the remaining natural-drain window; do not convert that
                # single observation race into DESCENDANT_EXIT.
                if time.monotonic() < deadline:
                    continue
                return False, output_bytes, False, hard_deadline <= time.monotonic()
            return True, output_bytes, False, False
        if now >= deadline:
            return False, output_bytes, False, deadline == hard_deadline
        try:
            selector.select(timeout=min(_NATURAL_DRAIN_SELECT_SECONDS, max(0.0, deadline - time.monotonic())))
        except (OSError, ValueError):
            return False, output_bytes, False, False


def _terminate_and_reap(
    process: subprocess.Popen[bytes],
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    output_limit: int,
    output_bytes: int,
    root_identity: ProcessIdentity,
    known_identities: set[ProcessIdentity],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
) -> tuple[int, int, bool]:
    """Bounded post-decision cleanup of identity-proven private processes."""

    cleanup_deadline = time.monotonic() + _TERMINATE_AND_REAP_SECONDS
    remaining: set[ProcessIdentity] = set()
    cleanup_proven = False
    empty_rounds = 0

    # Root identity is already bound, so it is safe to begin termination even
    # if a later full-tree observation cannot be started before the deadline.
    # Discover the still-private tree before the root can be reaped and its
    # private PGID loses its only identity-proven anchor.
    if _cleanup_deadline_allows_scan(cleanup_deadline):
        live, _ = _live_process_tree(root_identity, known_identities, baseline_identities)
        known_identities.update(live)
        remaining = live
    _signal_process_identities(known_identities, signal.SIGTERM)
    term_deadline = min(cleanup_deadline, time.monotonic() + 0.25)
    while _cleanup_deadline_allows_scan(term_deadline) and (process.returncode is None or selector.get_map()):
        if not _cleanup_deadline_allows_scan(cleanup_deadline):
            break
        live, _ = _live_process_tree(root_identity, known_identities, baseline_identities)
        known_identities.update(live)
        remaining = live
        _signal_process_identities(live, signal.SIGTERM)
        _reap_process_identities(live, root_identity, baseline_identities)
        output_bytes, _ = _drain_streams(selector, buffers, output_limit, output_bytes)
        try:
            selector.select(timeout=min(0.02, max(0.0, cleanup_deadline - time.monotonic())))
        except (OSError, ValueError):
            break

    # A command is not allowed to leave an identity-proven descendant behind,
    # including one that ignored SIGTERM.  Every scan in this phase is checked
    # against the one cleanup deadline before it begins.
    _signal_process_identities(known_identities, signal.SIGKILL)
    if process.returncode is None and _cleanup_deadline_allows_scan(cleanup_deadline):
        try:
            process.wait(timeout=min(0.25, max(0.0, cleanup_deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass

    # Two timely empty observations prove both that SIGKILL reached the
    # identity-proven tree and that no adopted private-group zombie remains.
    # A scan begun immediately before the deadline may finish late; it counts
    # as the one unavoidable scan, but no subsequent scan is started.
    while _cleanup_deadline_allows_scan(cleanup_deadline):
        live, _ = _live_process_tree(root_identity, known_identities, baseline_identities)
        known_identities.update(live)
        remaining = live
        _signal_process_identities(live, signal.SIGKILL)
        _reap_process_identities(live, root_identity, baseline_identities)
        output_bytes, _ = _drain_streams(selector, buffers, output_limit, output_bytes)
        if not remaining and process.poll() is not None:
            empty_rounds += 1
            if empty_rounds >= 2:
                cleanup_proven = True
                break
        else:
            empty_rounds = 0
        time.sleep(min(0.01, max(0.0, cleanup_deadline - time.monotonic())))

    # The deadline only forbids more full scans.  Already-proven identities
    # are still safe to signal/reap, and Popen remains the sole root waiter.
    _signal_process_identities(known_identities, signal.SIGKILL)
    post_deadline_remaining = _bounded_post_deadline_reap(
        process,
        known_identities,
        root_identity,
        baseline_identities,
    )

    # Do not block indefinitely on an escaped pipe writer.  All observed
    # descendants have received SIGKILL; closing our read ends also prevents a
    # cleanup failure from becoming an output-memory sink.  Output count is
    # carried through every cleanup drain, including SIGTERM flood output.
    for _ in range(20):
        if not selector.get_map():
            break
        output_bytes, _ = _drain_streams(selector, buffers, output_limit, output_bytes)
        try:
            selector.select(timeout=0.005)
        except (OSError, ValueError):
            break
    cleanup_failed = not cleanup_proven or bool(post_deadline_remaining) or process.poll() is None
    # A failed proof is escalated by the bound OSError/SubprocessError path.
    # Keep the selector open for that emergency pass; the outer finally still
    # closes it when the escalation finishes or raises.
    if not cleanup_failed:
        _close_streams(selector)
    return process.returncode if process.returncode is not None else 127, output_bytes, cleanup_failed


def _run_argv(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    rss_limit: int,
    output_limit: int,
    pass_fds: tuple[int, ...],
    baseline_identities: set[ProcessIdentity] | frozenset[ProcessIdentity],
) -> tuple[int, bytes, bytes, float, bool, int]:
    started = time.monotonic()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    selector = selectors.DefaultSelector()
    process: subprocess.Popen[bytes] | None = None
    root_identity: ProcessIdentity | None = None
    unbound_private_session_handled = False
    known_identities: set[ProcessIdentity] = set()
    output_bytes = 0
    max_rss = 0
    reason: str | None = None
    address_space_limit = max(rss_limit * 2, rss_limit + 256 * 1024 * 1024)

    def child_setup() -> None:
        # subprocess establishes start_new_session before invoking this hook.
        # The containment therefore covers every descendant from the instant
        # before exec, including a compiler that forks immediately on startup.
        limit_address_space(address_space_limit)
        _install_child_containment()

    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=pass_fds,
            # RLIMIT_AS is only a supplementary ceiling.  The live /proc RSS
            # sample below is the authoritative resource contract.
            preexec_fn=child_setup,
        )
        try:
            root_identity = _root_identity_after_spawn(process.pid, baseline_identities)
        except BaseException as bind_error:
            try:
                _cleanup_unbound_private_session(process, selector)
            except BaseException as cleanup_error:
                _append_cleanup_note(bind_error, f"unbound private-session cleanup failed: {cleanup_error}")
            finally:
                unbound_private_session_handled = True
            raise
        known_identities.add(root_identity)
        assert process.stdout is not None and process.stderr is not None
        for stream, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe.fileno(), selectors.EVENT_READ, stream)
        deadline = started + timeout
        root_code: int | None = None
        while True:
            if time.monotonic() >= deadline:
                reason = "timeout"
                _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                break
            live, current_rss = _live_process_tree(root_identity, known_identities, baseline_identities)
            known_identities.update(live)
            max_rss = max(max_rss, current_rss)
            root_code = _root_exit_status_without_reap(process, root_identity)
            if current_rss > rss_limit:
                reason = "rss"
            elif time.monotonic() >= deadline:
                reason = "timeout"
            elif root_code is not None and live - {root_identity}:
                if root_code == 0:
                    natural_complete, output_bytes, overflow, natural_timed_out = _natural_drain_and_reap(
                        process,
                        selector,
                        buffers,
                        output_limit,
                        output_bytes,
                        root_identity,
                        known_identities,
                        baseline_identities,
                        deadline,
                    )
                    if natural_complete:
                        process.wait()
                        break
                    reason = "output" if overflow else "timeout" if natural_timed_out else "descendants"
                else:
                    reason = "descendants"
            if reason is not None:
                _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                if cleanup_failed and reason != "timeout":
                    reason = "descendants"
                break
            if root_code is not None and not selector.get_map():
                process.wait()
                break
            wait_for = min(0.05, max(0.0, deadline - time.monotonic()))
            try:
                selector.select(timeout=wait_for)
            except (OSError, ValueError):
                reason = "pipe"
                _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                if cleanup_failed:
                    reason = "descendants"
                break
            output_bytes, overflow = _drain_streams(selector, buffers, output_limit, output_bytes)
            if overflow:
                reason = "output"
                _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                if cleanup_failed:
                    reason = "descendants"
                break
        return_code = process.returncode if process.returncode is not None else root_code if root_code is not None else 127
        if reason == "timeout":
            return_code = 124
        elif reason == "output":
            return_code = OUTPUT_LIMIT_EXIT
        elif reason == "rss":
            return_code = RSS_LIMIT_EXIT
        elif reason == "descendants":
            return_code = DESCENDANT_EXIT
        return return_code, bytes(buffers["stdout"]), bytes(buffers["stderr"]), time.monotonic() - started, reason == "timeout", max_rss
    except (OSError, subprocess.SubprocessError) as exc:
        emergency_cleanup_used = False
        if process is not None:
            if root_identity is None:
                if not unbound_private_session_handled:
                    raise RuntimeContractError("spawned root identity was unavailable for cleanup")
            else:
                termination_error: BaseException | None = None
                cleanup_failed = True
                try:
                    _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                except BaseException as cleanup_error:
                    termination_error = cleanup_error
                if termination_error is not None or cleanup_failed:
                    emergency_cleanup_used = True
                    emergency_error: BaseException | None = None
                    cleanup_ok = False
                    try:
                        cleanup_ok = _emergency_cleanup_after_observation_failure(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                    except BaseException as cleanup_error:
                        emergency_error = cleanup_error
                    if not cleanup_ok:
                        details = [f"original error: {exc}"]
                        if termination_error is not None:
                            details.append(f"terminate_and_reap raised: {termination_error}")
                        else:
                            details.append("terminate_and_reap reported cleanup_failed=True")
                        if emergency_error is not None:
                            details.append(f"emergency cleanup raised: {emergency_error}")
                        else:
                            details.append("emergency cleanup could not prove audited process removal")
                        detail = "; ".join(details).encode("utf-8", "replace")[:_CLEANUP_DIAGNOSTIC_BYTES].decode("utf-8", "replace")
                        raise RuntimeContractError(f"bound process cleanup-unproven (cleanup unproven): {detail}") from exc
        else:
            _close_streams(selector)
        diagnostic_text = str(exc)
        if emergency_cleanup_used:
            diagnostic_text += "; bounded emergency cleanup proved audited process removal"
        diagnostic = diagnostic_text.encode("utf-8", "replace")[:output_limit]
        return 127, b"", diagnostic, time.monotonic() - started, False, max_rss
    except RuntimeContractError:
        if process is not None:
            if root_identity is None:
                if not unbound_private_session_handled:
                    raise RuntimeContractError("emergency cleanup cannot bind the spawned root identity")
            else:
                cleanup_ok = _emergency_cleanup_after_observation_failure(process, selector, buffers, output_limit, output_bytes, root_identity, known_identities, baseline_identities)
                if not cleanup_ok:
                    raise RuntimeContractError("emergency cleanup could not prove that audited descendants were reaped")
        else:
            _close_streams(selector)
        raise
    except BaseException as original:
        # RuntimeContractError/OSError/SubprocessError have dedicated paths
        # above.  This branch is deliberately only for interrupts and other
        # BaseException subclasses raised after identity binding: cleanup must
        # be just as conservative, but must never replace the original value.
        if process is not None:
            if root_identity is None:
                if not unbound_private_session_handled:
                    try:
                        _cleanup_unbound_private_session(process, selector)
                    except BaseException as cleanup_error:
                        _append_cleanup_note(original, f"unbound cleanup failed: {cleanup_error}")
            else:
                cleanup_failed = True
                try:
                    _termination_code, output_bytes, cleanup_failed = _terminate_and_reap(
                        process,
                        selector,
                        buffers,
                        output_limit,
                        output_bytes,
                        root_identity,
                        known_identities,
                        baseline_identities,
                    )
                except BaseException as cleanup_error:
                    _append_cleanup_note(original, f"identity-safe cleanup raised: {cleanup_error}")
                if cleanup_failed:
                    try:
                        cleanup_ok = _emergency_cleanup_after_observation_failure(
                            process,
                            selector,
                            buffers,
                            output_limit,
                            output_bytes,
                            root_identity,
                            known_identities,
                            baseline_identities,
                        )
                    except BaseException as cleanup_error:
                        _append_cleanup_note(original, f"emergency cleanup raised: {cleanup_error}")
                    else:
                        if not cleanup_ok:
                            _append_cleanup_note(original, "emergency cleanup could not prove audited process removal")
        raise
    finally:
        if process is not None:
            _close_process_pipes(process)
        _close_streams(selector)


def run_argv(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    rss_limit: int,
    output_limit: int,
    pass_fds: tuple[int, ...] = (),
) -> tuple[int, bytes, bytes, float, bool, int]:
    if not argv or any(not isinstance(token, str) or not token or "\x00" in token for token in argv):
        raise RuntimeContractError("command is not a bounded argv-only vector")
    if timeout <= 0 or rss_limit <= 0 or output_limit <= 0:
        raise RuntimeContractError("command resource bounds must be positive")
    _require_linux_amd64()
    _require_pidfd_signalling_preflight()
    with _process_observation_scope() as baseline_identities:
        return _run_argv(
            argv,
            cwd=cwd,
            env=env,
            timeout=timeout,
            rss_limit=rss_limit,
            output_limit=output_limit,
            pass_fds=pass_fds,
            baseline_identities=baseline_identities,
        )


def section_sizes(output: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for block in re.findall(r"(?m)^\s*Section \{\n(.*?)^\s*\}", output, re.DOTALL):
        name = re.search(r"\bName: ([^\n]+)", block)
        size = re.search(r"\bSize: (0x[0-9a-fA-F]+|[0-9]+)", block)
        if name and size:
            found[re.sub(r"\s+\(\d+\)$", "", name.group(1).strip())] = int(size.group(1), 0)
    return found


def _symbol_blocks(output: str) -> list[str]:
    starts = list(re.finditer(r"(?m)^\s*Symbol \{\s*$", output))
    blocks: list[str] = []
    for start in starts:
        closing = re.search(r"(?m)^\s*}\s*$", output[start.end() :])
        if closing is None:
            raise RuntimeContractError("malformed ELF symbol record without a closing brace")
        blocks.append(output[start.end() : start.end() + closing.start()])
    return blocks


def _symbol_field(block: str, field: str, *, required: bool = True) -> str | None:
    matches = re.findall(rf"(?m)^\s*{re.escape(field)}:\s*(.*?)\s*$", block)
    if len(matches) > 1:
        raise RuntimeContractError(f"ELF symbol record has duplicate {field} fields")
    if not matches:
        if required:
            raise RuntimeContractError(f"ELF symbol record is missing {field}")
        return None
    return matches[0]


def _strip_symbol_index(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"\((?:0x[0-9a-fA-F]+|[0-9]+)\)", value):
        return ""
    return re.sub(r"\s+\((?:0x[0-9a-fA-F]+|[0-9]+)\)$", "", value)


def _symbol_binding_or_type(value: str, field: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\s+\((0x[0-9a-fA-F]+|[0-9]+)\)", value.strip())
    if not match:
        raise RuntimeContractError(f"ELF symbol {field} record is malformed: {value!r}")
    try:
        numeric = int(match.group(2), 0)
    except ValueError as exc:
        raise RuntimeContractError(f"ELF symbol {field} record has an invalid numeric value") from exc
    return match.group(1), numeric


_ELF_VISIBILITY_NUMBERS = {
    0: "STV_DEFAULT",
    1: "STV_INTERNAL",
    2: "STV_HIDDEN",
    3: "STV_PROTECTED",
}


def _symbol_other(block: str, name: str) -> tuple[int, str]:
    """Parse llvm-readobj's scalar and multiline Other spellings.

    ROCm 7.14 emits both Other: 0 and, for non-default visibility,
    Other [ (0x3) / STV_PROTECTED (0x3) / ].  The outer number, the
    symbolic value, and the bracket structure are all checked so malformed or
    partially observed records fail closed instead of being treated as
    default-visible.
    """

    starts = list(re.finditer(r"(?m)^\s*Other\s*(?::|\[)", block))
    if len(starts) != 1:
        raise RuntimeContractError(f"ELF symbol record has missing or duplicate Other fields: {name!r}")
    start = starts[0]
    tail = block[start.start() :]
    if tail.lstrip().startswith("Other:"):
        line = tail.splitlines()[0]
        match = re.fullmatch(r"\s*Other:\s*(0x[0-9a-fA-F]+|[0-9]+)\s*", line)
        if not match:
            raise RuntimeContractError(f"ELF symbol Other field is malformed: {line!r}")
        value = int(match.group(1), 0)
        visibility = _ELF_VISIBILITY_NUMBERS.get(value)
        if visibility is None:
            raise RuntimeContractError(f"ELF symbol Other field has an invalid visibility: {value}")
        return value, visibility

    closing = re.search(r"(?m)^\s*\]\s*$", tail)
    if closing is None:
        raise RuntimeContractError(f"ELF symbol Other visibility block is unterminated: {name!r}")
    other_text = tail[: closing.end()]
    match = re.fullmatch(
        r"(?ms)\s*Other\s*\[\s*\((0x[0-9a-fA-F]+|[0-9]+)\)\s*(.*?)\s*\]\s*",
        other_text,
    )
    if not match:
        raise RuntimeContractError(f"ELF symbol Other visibility block is malformed: {name!r}")
    value = int(match.group(1), 0)
    visibility_match = re.fullmatch(
        r"\s*(STV_(?:DEFAULT|INTERNAL|HIDDEN|PROTECTED))\s+\((0x[0-9a-fA-F]+|[0-9]+)\)\s*",
        match.group(2),
    )
    if visibility_match is None:
        raise RuntimeContractError(f"ELF symbol Other visibility entry is malformed: {name!r}")
    symbolic_visibility = visibility_match.group(1)
    symbolic_value = int(visibility_match.group(2), 0)
    if symbolic_value != value or _ELF_VISIBILITY_NUMBERS.get(value) != symbolic_visibility:
        raise RuntimeContractError(f"ELF symbol Other visibility number conflicts with its name: {name!r}")
    return value, symbolic_visibility


def _symbol_records(output: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for block in _symbol_blocks(output):
        name_value = _symbol_field(block, "Name")
        binding_value = _symbol_field(block, "Binding")
        type_value = _symbol_field(block, "Type")
        section_value = _symbol_field(block, "Section")
        assert name_value is not None and binding_value is not None and type_value is not None and section_value is not None
        name = _strip_symbol_index(name_value)
        other, visibility = _symbol_other(block, name)
        visibility_field = _symbol_field(block, "Visibility", required=False)
        if visibility_field is not None:
            visibility_name, visibility_number = _symbol_binding_or_type(visibility_field, "Visibility")
            field_visibility = {
                "Default": "STV_DEFAULT",
                "Internal": "STV_INTERNAL",
                "Hidden": "STV_HIDDEN",
                "Protected": "STV_PROTECTED",
            }.get(visibility_name)
            if field_visibility != visibility or visibility_number != other:
                raise RuntimeContractError(f"ELF symbol Visibility field conflicts with Other: {name!r}")
        binding, binding_number = _symbol_binding_or_type(binding_value, "Binding")
        symbol_type, type_number = _symbol_binding_or_type(type_value, "Type")
        section = _strip_symbol_index(section_value)
        if not section:
            raise RuntimeContractError("ELF symbol record has an empty section")
        records.append(
            {
                "name": name,
                "binding": binding,
                "binding_number": binding_number,
                "type": symbol_type,
                "type_number": type_number,
                "other": other,
                "visibility": visibility,
                "section": section,
                "defined": section not in {"Undefined", "UND", "0"},
            }
        )
    return records


def defined_symbols(output: str, *, reject_invalid_names: bool = True) -> list[str]:
    names: list[str] = []
    for record in _symbol_records(output):
        if not record["defined"]:
            continue
        value = record["name"]
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            if reject_invalid_names:
                raise RuntimeContractError(f"defined ELF symbol is not a C identifier: {value!r}")
            continue
        names.append(value)
    return sorted(names)


def _bounded_symbol_diagnostic(names: list[str] | set[str] | tuple[str, ...], counts: dict[str, int] | None = None) -> str:
    ordered = sorted(set(names))
    rendered: list[str] = []
    for name in ordered[:_MAX_SYMBOL_DIAGNOSTIC_ITEMS]:
        display = name if len(name) <= _MAX_SYMBOL_DIAGNOSTIC_NAME_LENGTH else name[: _MAX_SYMBOL_DIAGNOSTIC_NAME_LENGTH - 3] + "..."
        if counts is not None:
            display += f"x{counts[name]}"
        rendered.append(display)
    if len(ordered) > _MAX_SYMBOL_DIAGNOSTIC_ITEMS:
        rendered.append(f"...(+{len(ordered) - _MAX_SYMBOL_DIAGNOSTIC_ITEMS})")
    return ",".join(rendered)


def _require_host_symbols(output: str) -> list[str]:
    records = _symbol_records(output)
    expected_set = set(PUBLIC_SYMBOLS)
    kernel_set = set(KERNEL_SYMBOLS)
    internal_set = set(INTERNAL_RUNTIME_SYMBOLS)
    probe_name = "sllm_hip_compile_probe"
    kernel_name = "sllm_rmsnorm_baseline_wave32_v1"
    probe_records = [record for record in records if record["name"] == probe_name]
    kernel_records = [record for record in records if record["name"] == kernel_name]
    unknown = sorted(
        {
            record["name"]
            for record in records
            if record["name"].startswith("sllm_")
            and record["name"] not in expected_set
            and record["name"] not in kernel_set
            and record["name"] not in internal_set
            and record["name"] != probe_name
        }
    )
    if unknown:
        raise RuntimeContractError(f"linked host ELF contains unknown sllm symbols: {_bounded_symbol_diagnostic(unknown)}")
    # amdclang++ emits these compiler-owned host entry shims whenever a HIP
    # kernel is linked. They are bound to the exact audited kernel closure and
    # are not the forbidden public_runtime_stub translation unit. Every other
    # symbol containing "stub" remains a hard failure.
    compiler_stub_symbols = {f"__device_stub__{name}" for name in (probe_name, *KERNEL_SYMBOLS)}
    compiler_stub_symbols.update(CAUSAL_ATTENTION_DEVICE_STUB_SYMBOLS)
    stub_symbols = sorted(
        {
            record["name"]
            for record in records
            if "stub" in record["name"].lower()
            and record["name"] not in compiler_stub_symbols
        }
    )
    if stub_symbols:
        raise RuntimeContractError(f"linked host ELF contains stub symbols: {_bounded_symbol_diagnostic(stub_symbols)}")
    undefined_sllm = sorted({record["name"] for record in records if record["name"].startswith("sllm_") and not record["defined"]})
    if undefined_sllm:
        raise RuntimeContractError(f"linked host ELF contains undefined sllm symbols: {_bounded_symbol_diagnostic(undefined_sllm)}")
    expected_host_hip = set(EXPECTED_HOST_HIP_UNDEFINED_SYMBOLS)
    undefined_host_hip_counts: dict[str, int] = {}
    defined_expected_host_hip: set[str] = set()
    for record in records:
        name = record["name"]
        if name in expected_host_hip and record["defined"]:
            defined_expected_host_hip.add(name)
        if not record["defined"] and name.startswith(("hip", "__hip")):
            undefined_host_hip_counts[name] = undefined_host_hip_counts.get(name, 0) + 1
    missing_host_hip = sorted(name for name in expected_host_hip if undefined_host_hip_counts.get(name, 0) == 0)
    extra_host_hip = sorted(name for name in undefined_host_hip_counts if name not in expected_host_hip)
    duplicate_host_hip = sorted(name for name, count in undefined_host_hip_counts.items() if count > 1)
    if missing_host_hip or extra_host_hip or duplicate_host_hip or defined_expected_host_hip:
        details: list[str] = []
        if missing_host_hip:
            details.append(f"missing={_bounded_symbol_diagnostic(missing_host_hip)}")
        if extra_host_hip:
            details.append(f"extra={_bounded_symbol_diagnostic(extra_host_hip)}")
        if duplicate_host_hip:
            details.append(f"duplicate={_bounded_symbol_diagnostic(duplicate_host_hip, undefined_host_hip_counts)}")
        if defined_expected_host_hip:
            details.append(f"defined={_bounded_symbol_diagnostic(defined_expected_host_hip)}")
        raise RuntimeContractError("linked host ELF HIP/compiler-runtime undefined symbol multiset mismatch: " + "; ".join(details))
    counts: dict[str, int] = {name: 0 for name in PUBLIC_SYMBOLS}
    for record in records:
        name = record["name"]
        if name not in expected_set:
            continue
        counts[name] += 1
        if not record["defined"]:
            raise RuntimeContractError(f"linked host ELF contains an undefined public symbol: {name}")
        if record["binding"] != "Global" or record["binding_number"] != 1:
            raise RuntimeContractError(f"linked host ELF public symbol is not global: {name}")
        if record["type"] != "Function" or record["type_number"] != 2:
            raise RuntimeContractError(f"linked host ELF public symbol is not a function: {name}")
        if record["visibility"] != "STV_DEFAULT" or record["other"] != 0:
            raise RuntimeContractError(f"linked host ELF public symbol is not default-visible: {name}")
        if record["section"] != ".text":
            raise RuntimeContractError(f"linked host ELF public symbol is not defined in .text: {name}")
    duplicate = sorted(name for name, count in counts.items() if count > 1)
    if duplicate:
        raise RuntimeContractError(f"linked host ELF contains duplicate public symbols: {_bounded_symbol_diagnostic(duplicate)}")
    missing = sorted(name for name, count in counts.items() if count != 1)
    if missing:
        raise RuntimeContractError(f"linked host ELF is missing public ABI symbols: {_bounded_symbol_diagnostic(missing)}")
    if len(probe_records) != 1:
        raise RuntimeContractError("linked host ELF must contain exactly one linked probe object symbol")
    probe = probe_records[0]
    if (
        not probe["defined"]
        or probe["binding"] != "Global"
        or probe["binding_number"] != 1
        or probe["type"] != "Object"
        or probe["type_number"] != 1
        or probe["visibility"] != "STV_DEFAULT"
        or probe["other"] != 0
        or probe["section"] != ".data.rel.ro"
    ):
        raise RuntimeContractError("linked host ELF probe symbol is not the compiler-generated host object role")
    if len(kernel_records) != 1:
        raise RuntimeContractError("linked host ELF must contain exactly one linked RMSNorm kernel symbol")
    kernel = kernel_records[0]
    if (
        not kernel["defined"]
        or kernel["binding"] != "Global"
        or kernel["binding_number"] != 1
        or kernel["type"] != "Object"
        or kernel["type_number"] != 1
        or kernel["visibility"] != "STV_DEFAULT"
        or kernel["other"] != 0
        or kernel["section"] != ".data.rel.ro"
    ):
        raise RuntimeContractError("linked host ELF RMSNorm kernel registration is not the exact linked definition")
    return sorted(expected_set)


def _require_device_symbols(output: str) -> list[str]:
    records = _symbol_records(output)
    probe_name = "sllm_hip_compile_probe"
    metadata_name = f"{probe_name}.kd"
    cuid_pattern = re.compile(r"__hip_cuid_(?:[0-9a-f]{15}|[0-9a-f]{16})")
    allowed_defined_names = {probe_name, metadata_name, "_DYNAMIC"}
    unknown_sllm = sorted(
        {
            record["name"]
            for record in records
            if record["name"].startswith("sllm_")
            and record["name"] not in {probe_name, metadata_name}
        }
    )
    if unknown_sllm:
        raise RuntimeContractError(f"device object contains unknown sllm symbols: {','.join(unknown_sllm)}")
    for record in records:
        name = record["name"]
        if not record["defined"] and name != "":
            raise RuntimeContractError(f"device object contains an unexpected undefined symbol: {name!r}")
        if record["defined"] and name not in allowed_defined_names and not cuid_pattern.fullmatch(name):
            raise RuntimeContractError(f"device object contains an unexpected defined symbol: {name!r}")
        if name == "_DYNAMIC" and (
            record["binding"] != "Local"
            or record["binding_number"] != 0
            or record["type"] != "None"
            or record["type_number"] != 0
            or record["section"] != ".dynamic"
            or record["visibility"] != "STV_HIDDEN"
        ):
            raise RuntimeContractError("device _DYNAMIC symbol is not the linker-generated role")
    probe_records = [record for record in records if record["name"] == probe_name]
    metadata_records = [record for record in records if record["name"] == metadata_name]
    cuid_records = [record for record in records if cuid_pattern.fullmatch(record["name"])]
    dynamic_records = [record for record in records if record["name"] == "_DYNAMIC"]
    if len(probe_records) != 1 or len(metadata_records) != 1 or len(cuid_records) != 1 or len(dynamic_records) != 1:
        raise RuntimeContractError("device object must contain exactly one probe, .kd metadata, CUID, and linker _DYNAMIC symbol")
    probe = probe_records[0]
    if (
        not probe["defined"]
        or probe["binding"] != "Global"
        or probe["binding_number"] != 1
        or probe["type"] != "Function"
        or probe["type_number"] != 2
        or probe["section"] != ".text"
        or probe["visibility"] not in {"STV_DEFAULT", "STV_PROTECTED"}
    ):
        raise RuntimeContractError("device probe function is not the exact AMDGPU kernel role")
    metadata = metadata_records[0]
    if (
        not metadata["defined"]
        or metadata["binding"] != "Global"
        or metadata["binding_number"] != 1
        or metadata["type"] != "Object"
        or metadata["type_number"] != 1
        or metadata["section"] != ".rodata"
        or metadata["visibility"] not in {"STV_DEFAULT", "STV_PROTECTED"}
    ):
        raise RuntimeContractError("device .kd symbol is not the exact compiler metadata role")
    cuid = cuid_records[0]
    if (
        not cuid["defined"]
        or cuid["binding"] != "Global"
        or cuid["binding_number"] != 1
        or cuid["type"] != "Object"
        or cuid["type_number"] != 1
        or cuid["section"] != ".bss"
        or cuid["visibility"] != "STV_DEFAULT"
        or cuid["other"] != 0
    ):
        raise RuntimeContractError("device CUID symbol is not the exact compiler-generated role")
    return [probe_name]


def readobj(path: Path, tool: Path, *, cwd: Path, row: dict[str, Any], pass_fds: tuple[int, ...] = ()) -> str:
    code, stdout, stderr, _, timed_out, rss = run_argv([str(tool), "--file-headers", "--sections", "--symbols", "--notes", str(path)], cwd=cwd, env=os.environ.copy(), timeout=60, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=pass_fds)
    if code != 0 or timed_out or len(stdout) + len(stderr) > row["resource"]["max_output_bytes"] or rss > row["resource"]["max_rss_bytes"]:
        raise RuntimeContractError(f"llvm-readobj failed or exceeded bounds for {path.name}: {stderr.decode('utf-8', 'replace')[-2000:]}")
    return stdout.decode("utf-8", "replace")


def inspect_host(path: Path, readobj_tool: Path, row: dict[str, Any], bundles: list[str] | None = None, *, cwd: Path | None = None, pass_fds: tuple[int, ...] = ()) -> dict[str, Any]:
    output = readobj(path, readobj_tool, cwd=path.parent if cwd is None else cwd, row=row, pass_fds=pass_fds)
    if "elf64-x86-64" not in output.lower() and not re.search(r"\bArch: x86_64\b", output):
        raise RuntimeContractError("linked public-runtime ELF is not an x86-64 host ELF")
    sections = section_sizes(output)
    if ".hip_fatbin" not in sections or ".text" not in sections or sections[".text"] < 1:
        raise RuntimeContractError("linked host ELF does not prove non-empty .text and .hip_fatbin sections")
    if bundles is None:
        raise RuntimeContractError("host bundle evidence must come from the observed offload bundle list")
    expected_bundles = [BUNDLE_IDS[row["target"]], HOST_BUNDLE_ID]
    if bundles != expected_bundles or len(set(bundles)) != len(bundles):
        raise RuntimeContractError("observed host bundle list is not the exact device-plus-host tuple")
    observed_symbols = _require_host_symbols(output)
    return {"format": "ELF64", "machine": "X86_64", "sections": {".text": {"present": ".text" in sections, "size_bytes": sections.get(".text", 0)}, ".hip_fatbin": {"present": True, "size_bytes": sections[".hip_fatbin"]}}, "bundles": list(bundles), "public_symbols": [{"name": name, "defined": True} for name in observed_symbols], "probe_symbol": {"name": "sllm_hip_compile_probe", "defined": True}, "kernel_symbol": {"name": "sllm_rmsnorm_baseline_wave32_v1", "defined": True}, "stub_symbols": []}


def inspect_device(path: Path, readobj_tool: Path, row: dict[str, Any], *, cwd: Path | None = None, pass_fds: tuple[int, ...] = ()) -> dict[str, Any]:
    output = readobj(path, readobj_tool, cwd=path.parent if cwd is None else cwd, row=row, pass_fds=pass_fds)
    if "elf64-amdgpu" not in output.lower() and not re.search(r"\bArch: amdgcn\b", output):
        raise RuntimeContractError("extracted device object is not AMDGPU ELF")
    header = output.split("Sections [", 1)[0]
    abi_values = re.findall(r"(?m)^\s*ABIVersion:\s*(\d+)\s*$", header)
    flag_values = re.findall(r"(?m)^\s*Flags\s+\[\s*\(0x([0-9a-fA-F]+)\)", header)
    target_values = re.findall(r"(?m)^\s*amdhsa\.target:\s*([^\s]+)\s*$", output)
    wave_values = re.findall(r"(?m)^\s*\.wavefront_size:\s*(\d+)\s*$", output)
    if len(abi_values) != 1 or len(flag_values) != 1 or len(target_values) != 1 or len(wave_values) != 1:
        raise RuntimeContractError("device object does not prove exactly one ABI, flags, target note, and wavefront")
    observed_flags = f"0x{int(flag_values[0], 16):08x}"
    expected_target = row["target"]
    expected_target_note = f"amdgcn-amd-amdhsa--{expected_target}"
    if abi_values[0] != "4" or observed_flags != E_FLAGS[expected_target] or target_values[0] != expected_target_note or wave_values[0] != "32":
        raise RuntimeContractError("device object target/V6/wave32/e_flags mismatch")

    observed_flag_value = int(flag_values[0], 16)
    if observed_flag_value & AMDGPU_MACH_MASK != int(E_FLAGS[expected_target], 16) & AMDGPU_MACH_MASK:
        raise RuntimeContractError("device object e_flags machine target is not the exact row target")
    symbolic_machines = re.findall(
        r"EF_AMDGPU_MACH_AMDGCN_([A-Za-z0-9_]+)\s+\(0x[0-9a-fA-F]+\)",
        header,
    )
    if symbolic_machines and (len(symbolic_machines) != 1 or symbolic_machines[0].lower() != expected_target.lower()):
        raise RuntimeContractError("device object e_flags symbolic target conflicts with the exact row target")
    try:
        observed_features = {
            "xnack": AMDGPU_XNACK_STATES[observed_flag_value & AMDGPU_XNACK_MASK],
            "sramecc": AMDGPU_SRAMECC_STATES[observed_flag_value & AMDGPU_SRAMECC_MASK],
            "generic_processor_version": (observed_flag_value & AMDGPU_GENERIC_VERSION_MASK) >> AMDGPU_GENERIC_VERSION_SHIFT,
        }
    except KeyError as exc:
        raise RuntimeContractError("device object e_flags contains an invalid AMDGPU feature state") from exc
    if observed_features != EXPECTED_FEATURES:
        raise RuntimeContractError("device object e_flags does not prove the canonical unsupported feature tuple")

    # ROCm 7.14's llvm-readobj prints the feature states through e_flags, not
    # as separate AMDGPU note keys.  If a tool version also emits explicit
    # fields, accept only one canonical value and reject every conflict.
    explicit_features: dict[str, list[str]] = {name: [] for name in EXPECTED_FEATURES}
    for name, value in re.findall(
        r"(?im)^\s*(xnack|sramecc|generic_processor_version)\s*[:=]\s*([^\s]+)\s*$",
        output,
    ):
        explicit_features[name].append(value.lower())
    for name, values in explicit_features.items():
        if values and (len(values) != 1 or values[0] != str(EXPECTED_FEATURES[name]).lower()):
            raise RuntimeContractError(f"device object has missing/conflicting {name} feature evidence")

    sections = section_sizes(output)
    symbols = _require_device_symbols(output)
    if ".text" not in sections:
        raise RuntimeContractError("device object is not the compile probe code object")
    return {"format": "ELF64", "machine": "AMDGPU", "target": expected_target, "ei_abiversion": 4, "e_flags": observed_flags, "code_object_version": "V6", "wavefront_size": 32, "features": observed_features, "sections": {".text": {"present": True, "size_bytes": sections[".text"]}}, "symbols": [{"name": name, "defined": True} for name in symbols], "source_attribution": "hip_compile_probe.hip.cpp"}


def _publication_context(
    path: Path,
    directory_fd: int | None,
    bindings: list[_DirectoryBinding] | None,
) -> tuple[Path, int, list[int], list[_DirectoryBinding], bool]:
    """Use a supplied bound directory or safely bind the pathname's parent."""

    if (directory_fd is None) != (bindings is None):
        raise RuntimeContractError("directory FD and binding set must be supplied together")
    if directory_fd is not None and bindings is not None:
        return _absolute_path_without_resolution(path.parent), directory_fd, [], bindings, False
    parent_path, parent_fd, opened, parent_bindings = _open_bound_directory(path.parent, create_leaf=False)
    return parent_path, parent_fd, opened, parent_bindings, True


def digest_record(
    path: Path,
    *,
    directory_fd: int | None = None,
    bindings: list[_DirectoryBinding] | None = None,
) -> dict[str, Any]:
    """Hash one compiler artifact from an inode-bound read-only descriptor."""

    path = _absolute_path_without_resolution(path)
    directory_path, parent_fd, opened, directory_bindings, owns_directory = _publication_context(path, directory_fd, bindings)
    name = _directory_name(path, "output")
    sidecar_name = name + ".sha256"
    input_fd: int | None = None
    sidecar_fd: int | None = None
    try:
        _verify_directory_bindings(directory_bindings)
        input_fd, file_stat = _open_regular_input(parent_fd, name, f"output {name}")
        if file_stat.st_size < 1 or file_stat.st_size > 268435456:
            raise RuntimeContractError(f"output exceeds the bounded artifact size: {name}")
        digest = _sha256_fd(input_fd)
        _verify_input_entry(parent_fd, name, file_stat, f"output {name}")
        sidecar_payload = f"{digest}  {name}\n".encode("ascii")
        sidecar_fd = _publish_bytes(parent_fd, directory_bindings, sidecar_name, sidecar_payload)
        sidecar_path = directory_path / sidecar_name
        return {"path": str(directory_path / name), "size_bytes": file_stat.st_size, "sha256": digest, "sidecar_path": str(sidecar_path), "sidecar_sha256": sha256_bytes(sidecar_payload)}
    finally:
        if sidecar_fd is not None:
            try:
                os.close(sidecar_fd)
            except OSError:
                pass
        if input_fd is not None:
            try:
                os.close(input_fd)
            except OSError:
                pass
        if owns_directory:
            for fd in reversed(opened):
                try:
                    os.close(fd)
                except OSError:
                    pass


def write_json_with_sidecar(
    path: Path,
    document: dict[str, Any],
    *,
    directory_fd: int | None = None,
    bindings: list[_DirectoryBinding] | None = None,
) -> dict[str, Any]:
    """Publish JSON then its sidecar using bound, no-replace directory links."""

    path = _absolute_path_without_resolution(path)
    directory_path, parent_fd, opened, directory_bindings, owns_directory = _publication_context(path, directory_fd, bindings)
    name = _directory_name(path, "JSON output")
    payload = canonical_bytes(document)
    payload_sha256 = sha256_bytes(payload)
    sidecar_payload = f"{payload_sha256}  {name}\n".encode("ascii")
    payload_fd: int | None = None
    sidecar_fd: int | None = None
    try:
        payload_fd = _publish_bytes(parent_fd, directory_bindings, name, payload)
        try:
            sidecar_fd = _publish_bytes(parent_fd, directory_bindings, name + ".sha256", sidecar_payload)
        except Exception:
            _unlink_owned_file(payload_fd, parent_fd, name)
            raise
        return {"path": str(directory_path / name), "size_bytes": len(payload), "sha256": payload_sha256, "sidecar_path": str(directory_path / (name + ".sha256")), "sidecar_sha256": sha256_bytes(sidecar_payload)}
    finally:
        for fd in (sidecar_fd, payload_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if owns_directory:
            for fd in reversed(opened):
                try:
                    os.close(fd)
                except OSError:
                    pass


def execution_environment(args: argparse.Namespace) -> dict[str, Any]:
    if not args.strict_ci or not args.pinned_container or args.observed_image_reference != PINNED_IMAGE or args.observed_image_config_digest != PINNED_CONFIG:
        raise RuntimeContractError("public-runtime H3 requires the pinned official ROCm container")
    if not network_isolated():
        raise RuntimeContractError("public-runtime H3 requires a network-none namespace containing only lo")
    return {"mode": "required-ci", "execution_scope": "official-container", "container_image_reference": PINNED_IMAGE, "observed_image_config_digest": PINNED_CONFIG, "pinned_container": True, "identity_verified": True, "network_isolated": True}


def render_commands(row: dict[str, Any], repo: Path, build_dir: Path) -> list[list[str]]:
    replacements = {"{repo}": str(repo), "{build_dir}": str(build_dir), "{target}": row["target"]}
    rendered: list[list[str]] = []
    for template in row["build"]["commands"]:
        command: list[str] = []
        for token in template:
            value = token
            for marker, replacement in replacements.items():
                value = value.replace(marker, replacement)
            if "{" in value or "}" in value:
                raise RuntimeContractError("build argv contains an unresolved placeholder")
            command.append(value)
        rendered.append(command)
    return rendered


def run_commands(
    commands: list[list[str]],
    row: dict[str, Any],
    repo: Path,
    env: dict[str, str],
    *,
    pass_fds: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    started_all = time.monotonic()
    for index, command in enumerate(commands, 1):
        remaining = row["resource"]["timeout_seconds"] - (time.monotonic() - started_all)
        if remaining <= 0:
            raise RuntimeContractError("compile row timeout exhausted before all argv commands completed")
        started = utc_now()
        code, stdout, stderr, elapsed, timed_out, rss = run_argv(command, cwd=repo, env=env, timeout=remaining, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=pass_fds)
        output_bytes = len(stdout) + len(stderr)
        errors: list[str] = []
        if code != 0: errors.append(f"exit_code={code}")
        if timed_out: errors.append("timed_out=true")
        if code == OUTPUT_LIMIT_EXIT: errors.append("output_limit_exceeded=true")
        if code == RSS_LIMIT_EXIT: errors.append("rss_limit_exceeded=true")
        if code == DESCENDANT_EXIT: errors.append("descendant_cleanup_failed=true")
        if output_bytes > row["resource"]["max_output_bytes"]: errors.append("output_limit_exceeded=true")
        if rss > row["resource"]["max_rss_bytes"]: errors.append("rss_limit_exceeded=true")
        finished = utc_now()
        state = "PASS" if not errors else "FAIL"
        steps.append({"step_id": f"{row['row_id']}.compile-{index}", "state": state, "argv": command, "exit_code": code, "started_at": iso(started), "finished_at": iso(finished), "duration_seconds": round(elapsed, 6), "stdout_sha256": sha256_bytes(stdout), "stderr_sha256": sha256_bytes(stderr), "diagnostic": "; ".join(errors), "resource": {"output_bytes": output_bytes, "output_limit_bytes": row["resource"]["max_output_bytes"], "max_rss_bytes": rss, "max_rss_limit_bytes": row["resource"]["max_rss_bytes"], "timed_out": timed_out}})
        if errors:
            detail = (stderr or stdout).decode("utf-8", "replace")[-3000:]
            raise RuntimeContractError(f"compile command {index} failed: {'; '.join(errors)} {detail}".strip())
    return steps


def scope() -> dict[str, Any]:
    return {"public_runtime_stub_linked": False, "compile_only": True, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "support_claim": False, "numerics_verified": False, "performance_verified": False}


def report_base(row: dict[str, Any], commit: str, tree: str, args: argparse.Namespace, matrix: dict[str, Any], environment: dict[str, Any], state: str, started: datetime, finished: datetime, steps: list[dict[str, Any]], diagnostics: list[str], metadata_record: dict[str, Any] | None = None, hashes: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"schema_version": "hip-runtime-public-report-v1", "report_id": f"h3-public-runtime.{row['target']}.{args.run_id}.{args.run_attempt}", "row_id": row["row_id"], "target": row["target"], "state": state, "required": False, "evidence_mode": "required-ci", "run": {"run_id": str(args.run_id), "run_attempt": args.run_attempt}, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit, "git_tree_oid": tree, "candidate": {"commit_sha": commit, "tree_oid": tree, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit}, "toolchain_id": "rocm-7.14.0", "matrix_id": matrix["matrix_id"], "matrix_manifest_sha256": sha256_json(matrix), "scope": scope(), "execution_environment": environment, "compile_only_contract": "compile-only; no GPU/support/model/network/fallback evidence", "steps": steps, "diagnostics": diagnostics, "metadata": metadata_record, "hashes": hashes or {}, "started_at": iso(started), "finished_at": iso(finished), "duration_seconds": round((finished - started).total_seconds(), 6), "no_output_execution": True}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--row", choices=("h3-public-gfx1030", "h3-public-gfx1201"), required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local-h3-public-runtime"))
    result.add_argument("--run-attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    result.add_argument("--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha", default=os.environ.get("REVIEWED_SHA"))
    result.add_argument("--tested-sha", "--expected-tested-sha", dest="tested_sha", default=os.environ.get("TESTED_SHA"))
    result.add_argument("--workflow-sha", "--expected-workflow-sha", dest="workflow_sha", default=os.environ.get("WORKFLOW_SHA"))
    result.add_argument("--tree-oid", "--expected-tree-oid", dest="tree_oid", default=os.environ.get("TREE_OID"))
    result.add_argument("--strict-ci", action="store_true")
    result.add_argument("--pinned-container", action="store_true")
    result.add_argument("--observed-image-reference")
    result.add_argument("--observed-image-config-digest")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = _absolute_path_without_resolution(args.output_dir)
    started = utc_now()
    steps: list[dict[str, Any]] = []
    diagnostics: list[str] = []
    metadata_record: dict[str, Any] | None = None
    hashes: dict[str, Any] = {}
    commit = tree = "0" * 40
    row: dict[str, Any] = {"row_id": args.row, "target": args.row.removeprefix("h3-public-")}
    environment: dict[str, Any] = {"mode": "required-ci", "execution_scope": "official-container", "container_image_reference": PINNED_IMAGE, "observed_image_config_digest": PINNED_CONFIG, "pinned_container": True, "identity_verified": False, "network_isolated": False}
    output_fd: int | None = None
    build_fd: int | None = None
    opened_output_fds: list[int] = []
    output_bindings: list[_DirectoryBinding] = []
    build_bindings: list[_DirectoryBinding] = []
    try:
        repo_input = _reject_symlink_components(args.repo, "repository")
        repo = repo_input.resolve(strict=True)
        if not repo.is_dir():
            raise RuntimeContractError("repository path is not a directory")
        if args.run_attempt < 1 or not RUN_ID.fullmatch(str(args.run_id)):
            raise RuntimeContractError("run identity is invalid")
        commit, tree = git_identity(repo)
        for name, supplied in (("reviewed_sha", args.reviewed_sha), ("tested_sha", args.tested_sha), ("workflow_sha", args.workflow_sha)):
            if supplied is not None and (not SHA40.fullmatch(supplied) or supplied != commit):
                raise RuntimeContractError(f"{name} does not equal checked-out commit SHA")
        if args.tree_oid is not None and (not SHA40.fullmatch(args.tree_oid) or args.tree_oid != tree):
            raise RuntimeContractError("tree OID does not equal checked-out HEAD tree")
        require_clean_checkout(repo)
        environment = execution_environment(args)
        toolchain, matrix, rows = validate_matrix(repo)
        row = rows[args.row]
        source_set = check_source_set(repo, matrix)
        direct_compile_source_set = check_direct_compile_sources(repo, matrix)
        h3_toolchain = inspect_toolchain(toolchain)
        output_dir = _validate_output_directory(args.output_dir, repo)
        output_dir, output_fd, opened_output_fds, output_bindings = _open_bound_directory(output_dir, create_leaf=True, repo=repo)
        try:
            if os.listdir(output_fd):
                raise RuntimeContractError("output directory must be empty; stale artifacts are rejected")
        except OSError as exc:
            raise RuntimeContractError("output directory cannot be listed by descriptor") from exc
        build_dir, build_fd = _open_bound_child_directory(output_dir, output_fd, row["output"]["build_directory_pattern"], output_bindings)
        build_bindings = list(output_bindings)
        output_bindings = output_bindings[:-1]
        # The child process receives only this directory FD and addresses it
        # through /proc/self/fd.  A replaced output/build pathname therefore
        # cannot redirect compiler-created files to an external directory.
        build_exec_dir = Path(f"/proc/self/fd/{build_fd}")
        commands = render_commands(row, repo, build_exec_dir)
        env = {"PATH": os.environ.get("PATH", ""), "ROCM_PATH": "/opt/rocm", "HIP_PATH": "/opt/rocm", "HOME": "/tmp", "LANG": "C", "LC_ALL": "C"}
        steps = run_commands(commands, row, repo, env, pass_fds=(build_fd,))
        compiler_paths = {key: Path(value) for key, value in toolchain["paths"].items()}
        probe_object = build_dir / row["output"]["probe_object_pattern"].replace("{target}", row["target"])
        public_object = build_dir / row["output"]["public_runtime_object_pattern"].replace("{target}", row["target"])
        kernel_object = build_dir / row["output"]["rmsnorm_kernel_object_pattern"].replace("{target}", row["target"])
        api_object = build_dir / row["output"]["rmsnorm_api_object_pattern"].replace("{target}", row["target"])
        host_elf = build_dir / row["output"]["host_elf_pattern"].replace("{target}", row["target"])
        probe_fatbin = build_dir / row["output"]["probe_fatbin_pattern"].replace("{target}", row["target"])
        device_object = build_dir / row["output"]["device_object_pattern"].replace("{target}", row["target"])
        probe_name = probe_object.name
        public_name = public_object.name
        host_name = host_elf.name
        probe_fatbin_name = probe_fatbin.name
        device_name = device_object.name
        kernel_name = kernel_object.name
        api_name = api_object.name
        for name in (probe_name, public_name, kernel_name, api_name, host_name):
            _require_bound_output(build_fd, name, f"compiler output {name}", row["resource"]["max_output_file_bytes"])
        probe_exec = build_exec_dir / probe_name
        public_exec = build_exec_dir / public_name
        kernel_exec = build_exec_dir / kernel_name
        api_exec = build_exec_dir / api_name
        host_exec = build_exec_dir / host_name
        probe_fatbin_exec = build_exec_dir / probe_fatbin_name
        device_exec = build_exec_dir / device_name
        public_object_report = readobj(public_exec, compiler_paths["llvm_readobj"], cwd=repo, row=row, pass_fds=(build_fd,))
        if ".hip_fatbin" in section_sizes(public_object_report):
            raise RuntimeContractError("public runtime object contains device code; probe-only attribution cannot be proven")
        kernel_object_report = readobj(kernel_exec, compiler_paths["llvm_readobj"], cwd=repo, row=row, pass_fds=(build_fd,))
        if ".hip_fatbin" not in section_sizes(kernel_object_report):
            raise RuntimeContractError("RMSNorm kernel object does not contain its HIP device bundle")
        api_object_report = readobj(api_exec, compiler_paths["llvm_readobj"], cwd=repo, row=row, pass_fds=(build_fd,))
        if ".hip_fatbin" in section_sizes(api_object_report):
            raise RuntimeContractError("RMSNorm API object unexpectedly contains device code")
        objcopy = [str(compiler_paths["llvm_objcopy"]), f"--dump-section=.hip_fatbin={probe_fatbin_exec}", str(probe_exec)]
        code, out, err, _, timed, rss = run_argv(objcopy, cwd=repo, env=env, timeout=60, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=(build_fd,))
        if code != 0 or timed or rss > row["resource"]["max_rss_bytes"]:
            raise RuntimeContractError(f"probe .hip_fatbin extraction failed: {err.decode('utf-8', 'replace')[-2000:]}")
        _require_bound_output(build_fd, probe_fatbin_name, "probe fatbin", row["resource"]["max_output_file_bytes"])
        bundler = compiler_paths["clang_offload_bundler"]
        list_code, list_out, list_err, _, list_timed, _ = run_argv([str(bundler), "--list", "--type=o", f"--input={probe_fatbin_exec}"], cwd=repo, env=env, timeout=60, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=(build_fd,))
        bundles = [line.strip() for line in list_out.decode("utf-8", "replace").splitlines() if line.strip()]
        if list_code != 0 or list_timed or bundles != [BUNDLE_IDS[row["target"]], HOST_BUNDLE_ID]:
            raise RuntimeContractError(f"probe fatbin bundles are not exact device+host bundles: {bundles} {list_err.decode('utf-8', 'replace')[-1000:]}")
        unbundle = [str(bundler), "--unbundle", "--type=o", f"--targets={BUNDLE_IDS[row['target']]}", f"--input={probe_fatbin_exec}", f"--output={device_exec}"]
        code, _, err, _, timed, rss = run_argv(unbundle, cwd=repo, env=env, timeout=60, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=(build_fd,))
        if code != 0 or timed or rss > row["resource"]["max_rss_bytes"]:
            raise RuntimeContractError(f"probe-only device extraction failed: {err.decode('utf-8', 'replace')[-2000:]}")
        _require_bound_output(build_fd, device_name, "extracted device object", row["resource"]["max_output_file_bytes"])

        host_fatbin_fd, host_fatbin_name = tempfile.mkstemp(prefix=f"sllm-h3-host-{row['target']}-", suffix=".fatbin")
        os.close(host_fatbin_fd)
        host_fatbin = Path(host_fatbin_name)
        try:
            # Host evidence must be derived from the linked ELF that is being
            # audited.  The probe object has its own identical-looking fatbin,
            # but reusing its bundle list would not prove that the linker
            # retained the exact device+host bundle pair in the final host
            # artifact.
            host_objcopy = [str(compiler_paths["llvm_objcopy"]), f"--dump-section=.hip_fatbin={host_fatbin}", str(host_exec)]
            code, _out, err, _, timed, rss = run_argv(host_objcopy, cwd=repo, env=env, timeout=60, rss_limit=row["resource"]["max_rss_bytes"], output_limit=row["resource"]["max_output_bytes"], pass_fds=(build_fd,))
            if code != 0 or timed or rss > row["resource"]["max_rss_bytes"]:
                raise RuntimeContractError(f"linked host .hip_fatbin extraction failed: {err.decode('utf-8', 'replace')[-2000:]}")
            require_regular(host_fatbin, "linked host fatbin")
            host_list_code, host_list_out, host_list_err, _, host_list_timed, host_list_rss = run_argv(
                [str(bundler), "--list", "--type=o", f"--input={host_fatbin}"],
                cwd=repo,
                env=env,
                timeout=60,
                rss_limit=row["resource"]["max_rss_bytes"],
                output_limit=row["resource"]["max_output_bytes"],
                pass_fds=(build_fd,),
            )
            host_bundles = [line.strip() for line in host_list_out.decode("utf-8", "replace").splitlines() if line.strip()]
            if host_list_code != 0 or host_list_timed or host_list_rss > row["resource"]["max_rss_bytes"] or host_bundles != [BUNDLE_IDS[row["target"]], HOST_BUNDLE_ID]:
                raise RuntimeContractError(f"linked host ELF fatbin bundles are not exact device+host bundles: {host_bundles} {host_list_err.decode('utf-8', 'replace')[-1000:]}")
            if host_bundles != bundles:
                raise RuntimeContractError("probe and linked host fatbins disagree about the exact bundle tuple")
            host_report = inspect_host(host_exec, compiler_paths["llvm_readobj"], row, host_bundles, cwd=repo, pass_fds=(build_fd,))
        finally:
            try:
                host_fatbin.unlink()
            except FileNotFoundError:
                pass
        device_report = inspect_device(device_exec, compiler_paths["llvm_readobj"], row, cwd=repo, pass_fds=(build_fd,))
        output_hashes = {"probe_object": digest_record(probe_object, directory_fd=build_fd, bindings=build_bindings), "public_runtime_object": digest_record(public_object, directory_fd=build_fd, bindings=build_bindings), "rmsnorm_kernel_object": digest_record(kernel_object, directory_fd=build_fd, bindings=build_bindings), "rmsnorm_api_object": digest_record(api_object, directory_fd=build_fd, bindings=build_bindings), "host_elf": digest_record(host_elf, directory_fd=build_fd, bindings=build_bindings), "probe_fatbin": digest_record(probe_fatbin, directory_fd=build_fd, bindings=build_bindings), "device_object": digest_record(device_object, directory_fd=build_fd, bindings=build_bindings)}
        metadata_finished = utc_now()
        metadata = {"schema_version": "hip-runtime-artifact-v1", "metadata_id": f"h3-public-runtime-artifact-{row['target']}", "matrix_row_id": row["row_id"], "target": row["target"], "candidate": {"commit_sha": commit, "tree_oid": tree, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit}, "run": {"run_id": str(args.run_id), "run_attempt": args.run_attempt}, "toolchain_id": "rocm-7.14.0", "matrix_id": matrix["matrix_id"], "toolchain_manifest_sha256": sha256_json(toolchain), "matrix_manifest_sha256": sha256_json(matrix), "image": {"reference": PINNED_IMAGE, "config_digest": PINNED_CONFIG, "platform": {"os": "linux", "architecture": "amd64"}}, "resolved_paths": {key: toolchain["paths"][key] for key in ("rocm_root", "compiler", "hip_headers", "device_libraries", "hip_runtime", "clang_offload_bundler", "llvm_objcopy", "llvm_readobj")}, "source_set": source_set, "direct_compile_source_set": direct_compile_source_set, "codegen": row["codegen"], "build": {"output_directory": str(output_dir), "build_directory": str(build_dir), "probe_source": str(repo / row["build"]["probe_source"]), "public_runtime_source": str(repo / row["build"]["public_runtime_source"]), "public_runtime_header": str(repo / row["build"]["public_runtime_header"]), "rmsnorm_kernel_source": str(repo / row["build"]["rmsnorm_kernel_source"]), "rmsnorm_kernel_header": str(repo / row["build"]["rmsnorm_kernel_header"]), "rmsnorm_api_source": str(repo / row["build"]["rmsnorm_api_source"]), "rmsnorm_api_header": str(repo / row["build"]["rmsnorm_api_header"]), "link_library": row["build"]["link_library"], "probe_object": str(probe_object), "public_runtime_object": str(public_object), "rmsnorm_kernel_object": str(kernel_object), "rmsnorm_api_object": str(api_object), "host_elf": str(host_elf), "probe_fatbin": str(probe_fatbin), "device_object": str(device_object), "commands": commands, "generator": "direct-amdclang++", "mode": "compile-link", "build_type": "Release", "language_standard": "gnu++17", "source_tree_output": False}, "host_elf": host_report, "device_code_object": device_report, "public_abi_symbols": list(PUBLIC_SYMBOLS), "scope": scope(), "execution_environment": environment, "hashes": output_hashes, "timestamps": {"created_at": iso(started), "started_at": iso(started), "finished_at": iso(metadata_finished)}, "duration_seconds": round((metadata_finished - started).total_seconds(), 6)}
        metadata_path = output_dir / "hip-runtime-artifact.json"
        metadata_record = write_json_with_sidecar(metadata_path, metadata, directory_fd=output_fd, bindings=output_bindings)
        hashes = dict(output_hashes)
        hashes["metadata"] = metadata_record
        finished = utc_now()
        report = report_base(row, commit, tree, args, matrix, environment, "PASS", started, finished, steps, [], {"path": metadata_path.name, "sha256": metadata_record["sha256"], "sidecar_sha256": metadata_record["sidecar_sha256"]}, hashes)
        report_record = write_json_with_sidecar(output_dir / "report.json", report, directory_fd=output_fd, bindings=output_bindings)
        report["report_sidecar_sha256"] = report_record["sidecar_sha256"]
        # The report sidecar digest is deliberately external: putting it into
        # report.json would make the report hash self-referential.
        print(json.dumps({"row_id": row["row_id"], "state": "PASS", "output_dir": str(output_dir), "report_sha256": report_record["sha256"]}, sort_keys=True))
        return 0
    except (RuntimeContractError, OSError, KeyError, ValueError, StopIteration) as exc:
        diagnostics.append(str(exc))
        finished = utc_now()
        try:
            repo_for_output = _reject_symlink_components(args.repo, "repository").resolve(strict=True)
            _validate_output_directory(output_dir, repo_for_output)
            if output_fd is None:
                output_dir, output_fd, opened_output_fds, output_bindings = _open_bound_directory(output_dir, create_leaf=True, repo=repo_for_output)
            report = report_base(row, commit, tree, args, {"matrix_id": "hip-runtime-compile-v1"}, environment, "FAIL", started, finished, steps, diagnostics, metadata_record, hashes)
            write_json_with_sidecar(output_dir / "report.json", report, directory_fd=output_fd, bindings=output_bindings)
        except (OSError, RuntimeContractError) as write_error:
            print(f"H3 public-runtime runner: cannot write report: {write_error}", file=sys.stderr)
        print(f"{args.row}: FAIL: {exc}", file=sys.stderr)
        return 1
    finally:
        if build_fd is not None:
            try:
                os.close(build_fd)
            except OSError:
                pass
        for fd in reversed(opened_output_fds):
            try:
                os.close(fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
