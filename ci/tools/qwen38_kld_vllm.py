#!/usr/bin/env python3
"""Dump full-vocabulary, teacher-forced Qwen3.8 logits with vLLM.

The input manifest is shared with ``sllm-qwen38-kld-dump``::

    {"schema_version": "qwen38-kld-manifest-v1", "cases": [
        {"id": "case-0", "token_ids": [1, 2, 3],
         "positions": [0, 1, 2]}
    ]}

For every requested input position ``i`` this writes the raw logits produced
after consuming ``token_ids[:i + 1]``.  vLLM normally exposes prompt scores as
log-probabilities.  The adapter's V1 worker hook captures the model output
before vLLM's prompt-logprob normalization and top-k materialization, so no
top-k truncation or large Python dictionary is involved.  A harmless sentinel
token is appended to each prompt so the last requested input position also has
a hidden-state row.  The generated sentinel successor is ignored.

The report is printed as JSON on stdout.  Logits are raw little-endian FP32,
row-major files named ``<case-id>.f32`` under ``--output-dir``.  This format is
deliberately identical to the native KLD dump adapter and avoids serializing a
248,320-wide matrix as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import struct
import sys
import tempfile
import threading
import types
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "qwen38-kld-vllm-logit-dump-v1"
INPUT_SCHEMA_VERSION = "qwen38-kld-manifest-v1"
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CASES = 256
MAX_CASE_TOKENS = 262_144
KV_CACHE_DTYPE_CHOICES = (
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
)
FLA_AUTOTUNE_MODULES = (
    "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
    "vllm.model_executor.layers.fla.ops.solve_tril",
    "vllm.model_executor.layers.fla.ops.chunk_delta_h",
    "vllm.model_executor.layers.fla.ops.chunk_o",
    "vllm.model_executor.layers.fla.ops.wy_fast",
    "vllm.model_executor.layers.fla.ops.cumsum",
    "vllm.model_executor.layers.fla.ops.l2norm",
)
_E5M2_WRITER_PATCH_INSTALLED = False
_E5M2_WRITER_PATCH_LOCK = threading.RLock()
_E5M2_CACHE_QUANT_PATCH_INSTALLED = False


# The ROCm image currently used for this study is vLLM 0.21.0.  Its V1
# prompt-logprob worker predates the ``raw_logits`` branch and unconditionally
# normalizes prompt scores.  Keep the worker shim importable by vLLM's child
# processes, while retaining a host-side fallback so AST/compile checks need no
# vLLM installation.
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
try:  # pragma: no cover - exercised inside the vLLM runtime image
    from vllm.v1.worker.gpu_worker import Worker as _VllmGpuWorker
except ImportError:  # pragma: no cover - normal host validation path
    _VllmGpuWorker = object  # type: ignore[assignment,misc]


class RawLogitWorker(_VllmGpuWorker):
    """Stream V1 prompt logits before the output processor builds Python maps."""

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        restore_quant_guard = None
        if os.environ.get("SLLM_VLLM_FIX_FP8_E5M2_WRITER") == "1":
            restore_quant_guard = _install_fp8_e5m2_cache_quant_bypass()
        try:
            result = super().load_model(*args, **kwargs)
        finally:
            if restore_quant_guard is not None:
                restore_quant_guard()
        runner = getattr(self, "model_runner", None)
        if runner is None or not hasattr(runner, "_get_prompt_logprobs_dict"):
            raise RuntimeError("vLLM V1 worker does not expose prompt-logprob hook")
        runner._sllm_raw_logit_state = None
        runner._get_prompt_logprobs_dict = types.MethodType(
            _stream_prompt_logprobs, runner
        )
        if os.environ.get("SLLM_VLLM_FLA_AUTOTUNE", "single") == "single":
            _limit_fla_autotune_configs()
        if os.environ.get("SLLM_VLLM_FIX_FP8_E5M2_WRITER") == "1":
            _install_fp8_e5m2_writer_patch()
        return result

    def initialize_from_config(self, kv_cache_config: Any) -> None:
        result = super().initialize_from_config(kv_cache_config)
        _write_runtime_kv_metadata(self)
        return result


class AdapterError(RuntimeError):
    """A fail-closed input, vLLM, or output contract error."""


def _limit_fla_autotune_configs() -> list[dict[str, Any]]:
    """Bound first-run FLA compilation to one valid Triton configuration.

    The ROCm image has no persistent autotune database for the first Qwen3.5
    profile.  A full FLA search can compile dozens of AMDGPU candidates and
    take many minutes.  Every candidate implements the same kernel math; the
    first declared config is sufficient for a correctness capture.  This only
    changes the autotuner search set in the worker process and is recorded in
    the output manifest.
    """

    details: list[dict[str, Any]] = []
    for module_name in FLA_AUTOTUNE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for value in vars(module).values():
            # Heuristics wrappers are commonly applied outside Autotuner;
            # walk the function chain until the actual search object is found.
            seen: set[int] = set()
            while id(value) not in seen and value is not None:
                seen.add(id(value))
                if value.__class__.__name__ == "Autotuner" and hasattr(value, "configs"):
                    configs_before = len(value.configs)
                    if configs_before > 1:
                        selected = value.configs[0]
                        value.configs = value.configs[:1]
                        details.append(
                            {
                                "module": module_name,
                                "kernel": getattr(value.base_fn, "__name__", "unknown"),
                                "configs_before": configs_before,
                                "configs_after": len(value.configs),
                                "selected_kwargs": dict(getattr(selected, "kwargs", {})),
                                "selected_num_warps": getattr(selected, "num_warps", None),
                                "selected_num_stages": getattr(selected, "num_stages", None),
                            }
                        )
                    break
                value = getattr(value, "fn", None)
    report_name = os.environ.get("SLLM_VLLM_FLA_AUTOTUNE_REPORT")
    if report_name:
        report_path = Path(report_name)
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=report_path.parent,
                prefix=f".{report_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(details, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.replace(temporary, report_path)
        except OSError:
            if "temporary" in locals():
                temporary.unlink(missing_ok=True)
    return details


def _install_fp8_e5m2_writer_patch() -> None:
    """Opt in to an E5M2 dtype for the ROCm Triton cache writer only.

    The platform-wide FP8 dtype remains E4M3, which is required by the model's
    weight kernels.  The existing ROCm attention writer is wrapped at its
    function boundary and sees a temporary platform proxy only while handling
    an ``fp8_e5m2`` cache update.  This is deliberately opt-in because it is a
    local compatibility experiment for the vLLM image, not a global platform
    policy.
    """

    global _E5M2_WRITER_PATCH_INSTALLED
    if _E5M2_WRITER_PATCH_INSTALLED:
        return
    try:
        import torch
        import vllm.v1.attention.backends.rocm_attn as rocm_attn
    except ImportError as exc:
        raise RuntimeError("ROCm attention writer is unavailable for E5M2 fix") from exc
    original = getattr(rocm_attn, "triton_reshape_and_cache_flash", None)
    if original is None:
        raise RuntimeError("ROCm attention writer does not expose Triton cache hook")
    if getattr(original, "_sllm_e5m2_writer_patch", False):
        _E5M2_WRITER_PATCH_INSTALLED = True
        return
    base_platform = original.__globals__.get("current_platform")
    if base_platform is None:
        raise RuntimeError("ROCm attention writer has no platform binding")

    class _WriterPlatformProxy:
        def __getattr__(self, name: str) -> Any:
            if name == "fp8_dtype":
                return lambda: torch.float8_e5m2
            return getattr(base_platform, name)

    proxy = _WriterPlatformProxy()
    patched_globals = dict(original.__globals__)
    patched_globals["current_platform"] = proxy
    e5m2_original = types.FunctionType(
        original.__code__,
        patched_globals,
        getattr(original, "__name__", "triton_reshape_and_cache_flash"),
        getattr(original, "__defaults__", None),
        getattr(original, "__closure__", None),
    )
    e5m2_original.__kwdefaults__ = getattr(original, "__kwdefaults__", None)

    def _e5m2_writer(*args: Any, **kwargs: Any) -> Any:
        kv_cache_dtype = kwargs.get("kv_cache_dtype")
        if kv_cache_dtype is None and len(args) > 5:
            kv_cache_dtype = args[5]
        if kv_cache_dtype != "fp8_e5m2":
            return original(*args, **kwargs)
        with _E5M2_WRITER_PATCH_LOCK:
            return e5m2_original(*args, **kwargs)

    _e5m2_writer.__name__ = getattr(original, "__name__", "triton_reshape_and_cache_flash")
    _e5m2_writer.__qualname__ = getattr(original, "__qualname__", _e5m2_writer.__name__)
    _e5m2_writer.__doc__ = getattr(original, "__doc__", None)
    _e5m2_writer._sllm_e5m2_writer_patch = True  # type: ignore[attr-defined]
    rocm_attn.triton_reshape_and_cache_flash = _e5m2_writer
    _E5M2_WRITER_PATCH_INSTALLED = True


def _install_fp8_e5m2_cache_quant_bypass() -> Any:
    """Temporarily bypass vLLM's FP8-checkpoint/E5M2 constructor guard.

    The guard only rejects the combination before creating the checkpoint KV
    scale parameters.  Presenting E4M3 to that one constructor call allows the
    same scale setup, then restoring the layer's requested E5M2 string leaves
    the later cache spec, writer, and reader paths on E5M2.  The patch is
    installed only during model construction and is restored even on failure.
    """

    global _E5M2_CACHE_QUANT_PATCH_INSTALLED
    if _E5M2_CACHE_QUANT_PATCH_INSTALLED:
        return lambda: None
    try:
        import vllm.model_executor.layers.attention.attention as attention_module
    except ImportError as exc:
        raise RuntimeError("vLLM Attention module is unavailable for E5M2 guard fix") from exc
    original = getattr(attention_module, "_init_kv_cache_quant", None)
    if original is None:
        raise RuntimeError("vLLM Attention module has no KV quantization initializer")

    def _e5m2_cache_quant_bypass(layer: Any, quant_config: Any, prefix: str) -> Any:
        if getattr(layer, "kv_cache_dtype", None) != "fp8_e5m2":
            return original(layer, quant_config, prefix)
        previous = layer.kv_cache_dtype
        layer.kv_cache_dtype = "fp8_e4m3"
        try:
            return original(layer, quant_config, prefix)
        finally:
            layer.kv_cache_dtype = previous

    _e5m2_cache_quant_bypass.__name__ = getattr(
        original, "__name__", "_init_kv_cache_quant"
    )
    _e5m2_cache_quant_bypass.__doc__ = getattr(original, "__doc__", None)
    attention_module._init_kv_cache_quant = _e5m2_cache_quant_bypass
    _E5M2_CACHE_QUANT_PATCH_INSTALLED = True

    def _restore() -> None:
        global _E5M2_CACHE_QUANT_PATCH_INSTALLED
        attention_module._init_kv_cache_quant = original
        _E5M2_CACHE_QUANT_PATCH_INSTALLED = False

    return _restore


def _write_runtime_kv_metadata(worker: Any) -> None:
    """Record the allocated KV storage and attention encoding in rank zero.

    vLLM represents FP8 KV pages as byte storage and separately carries a
    dtype string through the attention backend.  Recording both is necessary
    for this ROCm image: its writer uses the platform FP8 dtype while the
    reader has a separate branch for the ``fp8_e5m2`` spelling.
    """

    report_name = os.environ.get("SLLM_VLLM_KV_RUNTIME_REPORT")
    if not report_name:
        return
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        if int(get_tensor_model_parallel_rank()) != 0:
            return
    except (ImportError, RuntimeError, AssertionError):
        return

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("vLLM worker did not expose model_runner for KV metadata")
    cache_config = getattr(runner, "cache_config", None)
    requested = str(getattr(cache_config, "cache_dtype", "unknown"))
    model_config = getattr(runner, "model_config", None)
    model_dtype = str(getattr(model_config, "dtype", "unknown"))
    if model_dtype.startswith("torch."):
        model_dtype = model_dtype.removeprefix("torch.")
    runner_dtype = str(getattr(runner, "kv_cache_dtype", "unknown"))
    if runner_dtype.startswith("torch."):
        runner_dtype = runner_dtype.removeprefix("torch.")

    attention_dtype_strings: set[str] = set()
    spec_dtypes: set[str] = set()
    quant_modes: set[str] = set()
    backend_names: set[str] = set()
    layer_impls: set[str] = set()
    for group_list in getattr(runner, "attn_groups", []):
        for group in group_list:
            backend = getattr(group, "backend", None)
            if backend is not None:
                backend_names.add(
                    f"{backend.__module__}.{backend.__qualname__}"
                )
            spec = getattr(group, "kv_cache_spec", None)
            if spec is not None:
                spec_dtype = str(getattr(spec, "dtype", "unknown"))
                spec_dtypes.add(spec_dtype.removeprefix("torch."))
                mode = getattr(spec, "kv_quant_mode", None)
                if mode is not None:
                    quant_modes.add(str(getattr(mode, "name", mode)))
            for layer_name in getattr(group, "layer_names", []):
                layer = getattr(runner.compilation_config, "static_forward_context", {}).get(
                    layer_name
                )
                impl = getattr(layer, "impl", None)
                if impl is None:
                    continue
                impl_name = f"{impl.__class__.__module__}.{impl.__class__.__qualname__}"
                layer_impls.add(impl_name)
                value = getattr(impl, "kv_cache_dtype", None)
                if value is not None:
                    attention_dtype_strings.add(str(value))

    storage_tensors: list[dict[str, Any]] = []
    seen_data: set[int] = set()
    for cache in getattr(runner, "kv_caches", []):
        if cache is None or not hasattr(cache, "dtype"):
            continue
        data_ptr = int(cache.data_ptr()) if hasattr(cache, "data_ptr") else id(cache)
        if data_ptr in seen_data:
            continue
        seen_data.add(data_ptr)
        storage_tensors.append(
            {
                "dtype": str(cache.dtype).removeprefix("torch."),
                "element_size_bytes": int(cache.element_size()),
                "shape": [int(dim) for dim in cache.shape],
                "device": str(cache.device),
            }
        )
        if len(storage_tensors) >= 4:
            break

    platform_fp8_dtype = None
    try:
        from vllm.platforms import current_platform

        platform_fp8_dtype = str(current_platform.fp8_dtype()).removeprefix("torch.")
    except (ImportError, RuntimeError, AttributeError):
        pass
    is_fp8 = requested.startswith("fp8") or any(
        value.startswith("fp8") for value in attention_dtype_strings
    )
    e5m2_fix_enabled = os.environ.get("SLLM_VLLM_FIX_FP8_E5M2_WRITER") == "1"
    writer_encoding = platform_fp8_dtype if is_fp8 else runner_dtype
    if requested == "fp8_e5m2" and e5m2_fix_enabled:
        writer_encoding = "float8_e5m2"
    if requested in ("fp8", "fp8_e4m3"):
        reader_encoding = platform_fp8_dtype
    elif requested == "fp8_e5m2":
        reader_encoding = "float8_e5m2"
    else:
        reader_encoding = runner_dtype
    alias_warning = bool(
        requested == "fp8_e5m2"
        and platform_fp8_dtype is not None
        and platform_fp8_dtype != "float8_e5m2"
        and not e5m2_fix_enabled
    )
    payload = {
        "schema_version": "qwen38-vllm-kv-runtime-v1",
        "cache_config_dtype": requested,
        "model_dtype": model_dtype,
        "runner_kv_cache_torch_dtype": runner_dtype,
        "attention_kv_cache_dtype_strings": sorted(attention_dtype_strings),
        "attention_spec_dtypes": sorted(spec_dtypes),
        "attention_kv_quant_modes": sorted(quant_modes),
        "attention_backends": sorted(backend_names),
        "attention_impls": sorted(layer_impls),
        "storage_tensors": storage_tensors,
        "platform_fp8_dtype": platform_fp8_dtype,
        "fp8_writer_encoding": writer_encoding,
        "fp8_reader_encoding": reader_encoding,
        "fp8_e5m2_writer_fix_opt_in": e5m2_fix_enabled,
        "fp8_e5m2_writer_patch_installed": _E5M2_WRITER_PATCH_INSTALLED,
        "fp8_e5m2_checkpoint_quant_bypass_opt_in": e5m2_fix_enabled,
        "fp8_e5m2_platform_alias_warning": alias_warning,
    }
    report_path = Path(report_name)
    temporary: Path | None = None
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, report_path)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _stream_prompt_logprobs(
    runner: Any,
    hidden_states: Any,
    num_scheduled_tokens: dict[str, int],
) -> dict[str, Any]:
    """Write dense raw logits from the worker and return no prompt maps.

    vLLM 0.21's normal prompt-logprob output converts every row into Python
    dictionaries, which is prohibitive for long Qwen vocabularies.  This hook
    runs at the same point as the built-in prompt-logprob code, but writes the
    model's pre-normalization logits directly to a preallocated shared file.
    The control file is created by the parent process one case at a time.
    """

    control_name = os.environ.get("SLLM_VLLM_RAW_DUMP_CONTROL")
    if not control_name:
        return {}
    control_path = Path(control_name)
    try:
        control = json.loads(control_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        # The parent creates the control file only immediately before enqueue.
        # An absent file is therefore the idle state between cases.
        return {}
    if control.get("state") != "RUNNING":
        return {}
    output_name = control.get("output_file")
    positions = control.get("positions")
    generation = control.get("generation")
    expected_bytes = control.get("expected_bytes")
    if (
        not isinstance(output_name, str)
        or not isinstance(positions, list)
        or not positions
        or not isinstance(generation, str)
        or not isinstance(expected_bytes, int)
    ):
        raise RuntimeError("malformed SLLM_VLLM_RAW_DUMP_CONTROL")
    if any(not isinstance(pos, int) or pos < 0 for pos in positions):
        raise RuntimeError("invalid raw-logit positions in control file")
    from vllm.distributed import get_tensor_model_parallel_rank

    # compute_logits may perform an all-gather, so every TP rank must execute
    # it.  Only TP rank zero owns the dense host file and performs writes.
    is_writer = int(get_tensor_model_parallel_rank()) == 0
    output_path = Path(output_name)
    state = getattr(runner, "_sllm_raw_logit_state", None)
    # A request performs one extra decode step after prefill.  Once the last
    # prompt row is written, the completion marker prevents that decode call
    # from trying to create the same O_EXCL file again.
    if is_writer and state is None and Path(f"{output_path}.complete").is_file():
        return {}
    if is_writer and (state is None or state.get("generation") != generation):
        if state is not None:
            try:
                os.close(state["fd"])
            except OSError:
                pass
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(output_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.ftruncate(fd, expected_bytes)
        state = {
            "generation": generation,
            "fd": fd,
            "output_path": output_path,
            "positions": {int(pos): index for index, pos in enumerate(positions)},
            "written": set(),
            "vocab_size": None,
            "expected_bytes": expected_bytes,
        }
        runner._sllm_raw_logit_state = state

    # This mirrors vLLM's V1 implementation's chunk accounting.  The hidden
    # rows in a prefill chunk correspond to positions start_idx..start_idx+n-1;
    # the final prompt token is represented by the appended sentinel.
    for req_id, num_tokens in num_scheduled_tokens.items():
        if num_tokens <= 0:
            continue
        request = runner.requests[req_id]
        prompt_ids = request.prompt_token_ids
        if prompt_ids is None:
            continue
        start_idx = request.num_computed_tokens
        num_remaining = len(prompt_ids) - (start_idx + 1)
        num_logits = min(num_tokens, num_remaining)
        if num_logits <= 0:
            continue
        req_idx = runner.input_batch.req_id_to_index[req_id]
        offset = int(runner.query_start_loc.np[req_idx].item())
        logits = runner.model.compute_logits(
            hidden_states[offset : offset + num_logits]
        )
        if logits is None:
            if is_writer:
                raise RuntimeError("vLLM rank zero returned no prompt logits")
            continue
        if logits.ndim != 2 or logits.shape[0] != num_logits:
            raise RuntimeError("vLLM returned malformed prompt logits")
        vocab_size = int(logits.shape[1])
        if not is_writer:
            continue
        if state["vocab_size"] is None:
            state["vocab_size"] = vocab_size
        elif state["vocab_size"] != vocab_size:
            raise RuntimeError("vLLM prompt-logit vocabulary changed mid-capture")
        import torch

        cpu_logits = logits.to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(cpu_logits).all()):
            raise RuntimeError("vLLM returned non-finite raw prompt logits")
        row_bytes = vocab_size * 4
        for local_index in range(num_logits):
            position = start_idx + local_index
            output_index = state["positions"].get(position)
            if output_index is None or position in state["written"]:
                continue
            payload = cpu_logits[local_index].numpy().tobytes(order="C")
            if len(payload) != row_bytes:
                raise RuntimeError("vLLM raw prompt-logit row has unexpected byte size")
            written = os.pwrite(state["fd"], payload, output_index * row_bytes)
            if written != len(payload):
                raise RuntimeError("short write while storing vLLM raw logits")
            state["written"].add(position)

    if is_writer and len(state["written"]) == len(state["positions"]):
        os.fsync(state["fd"])
        os.close(state["fd"])
        marker = Path(f"{state['output_path']}.complete")
        temporary_marker: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=marker.parent,
                prefix=f".{marker.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                json.dump(
                    {
                        "state": "COMPLETE",
                        "generation": generation,
                        "bytes": state["expected_bytes"],
                        "rows": len(state["positions"]),
                        "vocab_size": state["vocab_size"],
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
                temporary_marker = Path(stream.name)
            os.replace(temporary_marker, marker)
            temporary_marker = None
        finally:
            if temporary_marker is not None:
                temporary_marker.unlink(missing_ok=True)
        runner._sllm_raw_logit_state = None
    return {}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise AdapterError(f"manifest must be a regular file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AdapterError(f"cannot read manifest {path}: {exc}") from exc
    if not data or len(data) > MAX_MANIFEST_BYTES:
        raise AdapterError(
            f"manifest size must be in 1..={MAX_MANIFEST_BYTES} bytes"
        )
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"manifest JSON is invalid: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise AdapterError(
            f"manifest schema_version must be {INPUT_SCHEMA_VERSION!r}"
        )
    unknown_root_fields = set(value) - {"schema_version", "cases"}
    if unknown_root_fields:
        raise AdapterError(
            f"manifest has unknown fields: {sorted(unknown_root_fields)}"
        )
    cases = value.get("cases")
    if not isinstance(cases, list) or not 0 < len(cases) <= MAX_CASES:
        raise AdapterError(f"manifest cases must contain 1..={MAX_CASES} entries")

    seen: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise AdapterError(f"case {index} must be an object")
        unknown = set(case) - {"id", "token_ids", "positions"}
        if unknown:
            raise AdapterError(f"case {index} has unknown fields: {sorted(unknown)}")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise AdapterError(f"case {index} id must be non-empty and unique")
        if case_id in {".", ".."} or Path(case_id).name != case_id:
            raise AdapterError(
                f"case {index} id must be a single safe output filename component"
            )
        seen.add(case_id)
        token_ids = case.get("token_ids")
        if not isinstance(token_ids, list) or not 0 < len(token_ids) <= MAX_CASE_TOKENS:
            raise AdapterError(
                f"case {case_id} token_ids must contain 1..={MAX_CASE_TOKENS} entries"
            )
        if any(
            not _is_int(token) or token < 0 or token > 2_147_483_647
            for token in token_ids
        ):
            raise AdapterError(f"case {case_id} contains an invalid token ID")
        positions = case.get("positions")
        if positions is not None:
            if not isinstance(positions, list) or not positions:
                raise AdapterError(f"case {case_id} positions must be non-empty")
            if any(
                not _is_int(position)
                or position < 0
                or position >= len(token_ids)
                for position in positions
            ):
                raise AdapterError(
                    f"case {case_id} positions must be unique values in "
                    f"0..{len(token_ids) - 1}"
                )
            if len(set(positions)) != len(positions):
                raise AdapterError(f"case {case_id} positions must be unique")
    return value, data


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _token_ids_sha256(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(struct.pack("<i", token_id))
    return f"sha256:{digest.hexdigest()}"


def _canonical_cache_dtype(value: Any) -> str | None:
    """Normalize vLLM's resolved cache dtype for the output contract."""

    if value is None:
        return None
    value = getattr(value, "value", value)
    text = str(value).lower().removeprefix("torch.")
    aliases = {
        "float8_e4m3fn": "fp8_e4m3",
        "float8_e4m3fnuz": "fp8_e4m3",
        "float8_e5m2": "fp8_e5m2",
        "float8_e4m3": "fp8_e4m3",
    }
    return aliases.get(text, text)


def _resolved_cache_dtype(llm: Any) -> str:
    """Resolve vLLM's ``auto`` cache dtype to the model compute dtype.

    The ROCm Triton cache writer in the vLLM 0.21 image requires the literal
    ``auto`` marker for native (FP16/BF16) cache storage.  CacheConfig keeps
    that marker instead of replacing it with a torch dtype, so the effective
    dtype is the loaded model dtype in this case.
    """
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    cache_config = getattr(config, "cache_config", None)
    value = getattr(cache_config, "cache_dtype", None)
    if value is None:
        value = getattr(cache_config, "kv_cache_dtype", None)
    resolved = _canonical_cache_dtype(value)
    if resolved is None:
        raise AdapterError(
            "vLLM did not expose the resolved KV cache dtype after model initialization"
        )
    if resolved == "auto":
        model_config = getattr(llm, "model_config", None)
        model_dtype = _canonical_cache_dtype(
            getattr(model_config, "dtype", None)
        )
        if model_dtype in ("float16", "bfloat16", "float32"):
            return model_dtype
        raise AdapterError(
            "vLLM left KV cache dtype at auto but model dtype is unavailable or "
            f"unsupported for native cache resolution: {model_dtype!r}"
        )
    return resolved


def _engine_cache_dtype(args: argparse.Namespace) -> str:
    """Return the cache dtype spelling accepted by the target vLLM backend."""
    # vLLM 0.21's ROCm Triton reshape-and-cache path accepts ``auto`` for a
    # native FP16/BF16 cache, but asserts if the equivalent explicit string is
    # passed.  Keep the user's requested dtype in the report while using the
    # backend-compatible spelling here.
    if args.kv_cache_dtype in ("float16", "bfloat16"):
        return "auto"
    return args.kv_cache_dtype


def _config_fingerprint(model_root: Path) -> tuple[str | None, str | None]:
    """Return model config digest and a best-effort local model identity.

    Hashing every 30 GB FP8 shard before a run is needlessly expensive.  The
    config digest is always recorded; callers can supply the immutable model
    fingerprint from their model lock with ``--model-fingerprint``.
    """

    config = model_root / "config.json"
    if config.is_file() and not config.is_symlink():
        try:
            config_digest = _sha256_bytes(config.read_bytes())
        except OSError:
            config_digest = None
    else:
        config_digest = None
    try:
        entries = []
        for path in sorted(model_root.iterdir()):
            if path.is_file() and not path.is_symlink():
                entries.append(f"{path.name}\0{path.stat().st_size}\n")
        inventory = "".join(entries).encode("utf-8")
        inventory_digest = _sha256_bytes(inventory)
    except OSError:
        inventory_digest = None
    return config_digest, inventory_digest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", "--model", dest="model_root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=1)
    parser.add_argument("--dtype", default="auto", choices=("auto", "bfloat16", "float16"))
    parser.add_argument(
        "--kv-cache-dtype",
        default="auto",
        choices=KV_CACHE_DTYPE_CHOICES,
        help="vLLM KV cache dtype; auto is resolved from the loaded model",
    )
    parser.add_argument(
        "--attention-backend",
        default=None,
        help="optional vLLM attention backend (for example, the radiance R4D backend)",
    )
    parser.add_argument(
        "--mamba-cache-dtype",
        default=None,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="optional Mamba convolution-state cache dtype",
    )
    parser.add_argument(
        "--mamba-ssm-cache-dtype",
        default=None,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="optional Mamba SSM-state cache dtype",
    )
    parser.add_argument(
        "--mamba-cache-mode",
        default=None,
        choices=("none", "all", "align"),
        help="optional Mamba cache strategy",
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=0.0,
        help="offload this many GiB of weights to host RAM when a single GPU cannot load FP8 weights",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=32,
        help="prefill chunk bound; small chunks cap the worker's full-vocab staging buffer",
    )
    parser.add_argument(
        "--fla-autotune",
        choices=("single", "auto"),
        default="single",
        help="FLA Triton search mode; single bounds first-run compile time, auto benchmarks all candidates",
    )
    parser.add_argument(
        "--fix-fp8-e5m2-writer",
        action="store_true",
        help="opt in to E5M2 encoding in the ROCm Triton KV writer experiment",
    )
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--sentinel-token-id", type=int, default=0)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-repository", default="Qwen/Qwen3.8-27B-FP8")
    parser.add_argument("--model-fingerprint", default=None)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        help="opt into prefix caching (disabled by default for independent KLD rows)",
    )
    return parser


def _make_llm(args: argparse.Namespace) -> Any:
    # Keep imports out of module scope so host-side AST/compile validation does
    # not require a CUDA/ROCm PyTorch installation.
    # The V2 runner in older ROCm images has a separate prompt worker which
    # always normalizes prompt scores.  V1 plus RawLogitWorker is the only
    # supported path for exact raw-logit capture in this adapter.
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    module_dir = str(Path(__file__).resolve().parent)
    python_path = os.environ.get("PYTHONPATH", "")
    if module_dir not in python_path.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            module_dir if not python_path else module_dir + os.pathsep + python_path
        )
    try:
        from vllm import LLM
    except ImportError as exc:
        raise AdapterError(
            "vLLM is unavailable; run this adapter inside a compatible vLLM "
            "ROCm environment (the repository checkout is reference-only)"
        ) from exc

    kwargs: dict[str, Any] = {
        "model": str(args.model_root),
        "runner": "generate",
        "tokenizer": str(args.model_root),
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        # max_logprobs=-1 is required because the default OpenAI cap is 20.
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
        "enforce_eager": args.enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_num_seqs": args.max_num_seqs,
        "disable_log_stats": True,
        "kv_cache_dtype": _engine_cache_dtype(args),
        # The KLD corpus is text-only.  Qwen3.5's tower is marked missing when
        # all image/video limits are zero, avoiding its vision weights and
        # encoder allocations while retaining the language model wrapper.
        "language_model_only": True,
        # WorkerWrapperBase resolves worker_cls with ``module.Class`` syntax
        # (unlike logits_processors, which use ``module:Class``).
        "worker_cls": "qwen38_kld_vllm.RawLogitWorker",
        # Independent teacher-forced cases must not reuse a prior request's KV
        # blocks.  Prefix caching is available only as an explicit diagnostic
        # opt-in because it is a KV-cache variant, not the baseline.
        "enable_prefix_caching": args.enable_prefix_caching,
    }
    if args.revision is not None:
        kwargs["revision"] = args.revision
        kwargs["tokenizer_revision"] = args.revision
    if args.max_model_len is not None:
        kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    # These EngineArgs fields were added for hybrid Mamba/attention models and
    # are used by the radiance 0.27.x image.  Keep them absent unless the
    # caller explicitly requests a serving profile so older vLLM images retain
    # the adapter's original constructor defaults.
    optional_engine_args = {
        "attention_backend": args.attention_backend,
        "mamba_cache_dtype": args.mamba_cache_dtype,
        "mamba_ssm_cache_dtype": args.mamba_ssm_cache_dtype,
        "mamba_cache_mode": args.mamba_cache_mode,
    }
    kwargs.update(
        {
            name: value
            for name, value in optional_engine_args.items()
            if value is not None
        }
    )
    try:
        return LLM(**kwargs)
    except TypeError as exc:
        # Older vLLM releases may not expose one of the optional kwargs.  Do
        # not silently retry with logprobs_mode/max_logprobs removed: that
        # would turn a full raw-logit capture into a top-k capture.
        raise AdapterError(
            f"vLLM LLM constructor rejected the requested raw-logit contract: {exc}"
        ) from exc


def _capture_case(
    llm: Any,
    case: dict[str, Any],
    output_dir: Path,
    vocab_size: int,
    sentinel_token_id: int,
) -> dict[str, Any]:
    case_id = case["id"]
    token_ids = [int(token) for token in case["token_ids"]]
    requested_positions = case.get("positions")
    positions = (
        [int(position) for position in requested_positions]
        if requested_positions is not None
        else list(range(len(token_ids)))
    )
    # vLLM's worker hook returns rows in the manifest's position order via
    # pwrite offsets, including a sparse/non-monotonic request.
    prompt_ids = [*token_ids, sentinel_token_id]
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        ignore_eos=True,
        # The RawLogitWorker hook consumes the hidden states directly.  A
        # zero-width prompt request keeps the normal output processor from
        # materializing a Python map for every vocabulary row.
        prompt_logprobs=0,
        logprobs=None,
        detokenize=False,
    )
    control_name = os.environ.get("SLLM_VLLM_RAW_DUMP_CONTROL")
    if not control_name:
        raise AdapterError("SLLM_VLLM_RAW_DUMP_CONTROL was not configured")
    control_path = Path(control_name)
    expected_bytes = len(positions) * vocab_size * 4
    output_path = output_dir / f"{case_id}.f32"
    marker_path = Path(f"{output_path}.complete")
    if output_path.exists() or output_path.is_symlink() or marker_path.exists():
        raise AdapterError(f"refusing to overwrite existing raw logits: {output_path}")
    control_payload = {
        "state": "RUNNING",
        "generation": f"{case_id}:{_token_ids_sha256(token_ids)}",
        "output_file": str(output_path),
        "positions": positions,
        "expected_bytes": expected_bytes,
    }
    if control_path.exists() or control_path.is_symlink():
        raise AdapterError(f"raw-logit control file already exists: {control_path}")
    control_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_control: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=control_path.parent,
            prefix=f".{control_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(control_payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_control = Path(stream.name)
        os.replace(temporary_control, control_path)
        temporary_control = None
        outputs = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            sampling_params=params,
            use_tqdm=False,
        )
    except Exception as exc:  # noqa: BLE001 - report backend diagnostics
        raise AdapterError(f"case {case_id} vLLM execution failed: {exc}") from exc
    finally:
        if temporary_control is not None:
            temporary_control.unlink(missing_ok=True)
        control_path.unlink(missing_ok=True)
    if len(outputs) != 1:
        raise AdapterError(f"case {case_id} produced {len(outputs)} outputs; expected one")

    # The worker creates and fills this path.  Keeping the check here makes a
    # missing/partial worker capture fail closed even if generation returned.
    if output_path.is_symlink() or not output_path.is_file():
        raise AdapterError(f"worker did not produce raw logits file: {output_path}")
    if marker_path.is_symlink() or not marker_path.is_file():
        raise AdapterError(f"worker did not publish a complete marker: {marker_path}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"raw-logit completion marker is invalid: {marker_path}") from exc
    if (
        not isinstance(marker, dict)
        or marker.get("state") != "COMPLETE"
        or marker.get("generation") != control_payload["generation"]
        or marker.get("bytes") != expected_bytes
        or marker.get("rows") != len(positions)
        or marker.get("vocab_size") != vocab_size
    ):
        raise AdapterError(f"raw-logit completion marker does not match case {case_id}")
    try:
        actual_bytes = output_path.stat().st_size
    except OSError as exc:
        raise AdapterError(f"cannot stat {output_path}: {exc}") from exc
    if actual_bytes != expected_bytes:
        raise AdapterError(
            f"case {case_id} wrote {actual_bytes} bytes; expected {expected_bytes}"
        )
    try:
        marker_path.unlink()
    except OSError as exc:
        raise AdapterError(f"cannot remove raw-logit completion marker {marker_path}: {exc}") from exc
    return {
        "id": case_id,
        "token_ids": token_ids,
        "token_ids_sha256": _token_ids_sha256(token_ids),
        "positions": positions,
        "rows": len(positions),
        "vocab_size": vocab_size,
        "logits_dtype": "FP32 little-endian row-major",
        "logits_file": output_path.name,
        "logits_file_sha256": _sha256_file(output_path),
        "logits_file_bytes": actual_bytes,
        "shape": [len(positions), vocab_size],
        "dtype": "float32",
        "input_token_ids_sha256": _token_ids_sha256(token_ids),
        "nonfinite_count": 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, manifest_bytes = _read_manifest(args.manifest)
    if args.tensor_parallel_size < 1:
        raise AdapterError("--tensor-parallel-size must be positive")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        raise AdapterError("--gpu-memory-utilization must be in (0, 1]")
    if args.cpu_offload_gb < 0.0:
        raise AdapterError("--cpu-offload-gb must be non-negative")
    if args.fix_fp8_e5m2_writer and args.kv_cache_dtype != "fp8_e5m2":
        raise AdapterError(
            "--fix-fp8-e5m2-writer requires --kv-cache-dtype fp8_e5m2"
        )
    model_root = args.model_root.resolve()
    if not model_root.is_dir():
        raise AdapterError(f"model root is not a directory: {model_root}")
    if args.sentinel_token_id < 0:
        raise AdapterError("--sentinel-token-id must be non-negative")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.output_dir / "manifest.json"
    if output_manifest.exists() or output_manifest.is_symlink():
        raise AdapterError(f"refusing to overwrite output manifest: {output_manifest}")
    config_sha256, inventory_sha256 = _config_fingerprint(model_root)
    control_path = args.output_dir / ".vllm-raw-logit-control.json"
    autotune_report_path = args.output_dir / ".fla-autotune.json"
    kv_runtime_report_path = args.output_dir / ".kv-cache-runtime.json"
    if control_path.exists() or control_path.is_symlink():
        raise AdapterError(f"refusing to overwrite raw-logit control file: {control_path}")
    if autotune_report_path.exists() or autotune_report_path.is_symlink():
        raise AdapterError(
            f"refusing to overwrite FLA autotune report: {autotune_report_path}"
        )
    if kv_runtime_report_path.exists() or kv_runtime_report_path.is_symlink():
        raise AdapterError(
            f"refusing to overwrite KV runtime report: {kv_runtime_report_path}"
        )
    os.environ["SLLM_VLLM_RAW_DUMP_CONTROL"] = str(control_path.resolve())
    os.environ["SLLM_VLLM_FLA_AUTOTUNE"] = args.fla_autotune
    os.environ["SLLM_VLLM_FLA_AUTOTUNE_REPORT"] = str(autotune_report_path.resolve())
    os.environ["SLLM_VLLM_KV_RUNTIME_REPORT"] = str(kv_runtime_report_path.resolve())
    os.environ["SLLM_VLLM_FIX_FP8_E5M2_WRITER"] = (
        "1" if args.fix_fp8_e5m2_writer else "0"
    )

    max_prompt_len = max(len(case["token_ids"]) for case in manifest["cases"]) + 1
    model_max_len = args.max_model_len
    if model_max_len is not None and model_max_len < max_prompt_len + 1:
        raise AdapterError(
            f"--max-model-len={model_max_len} is too short for the appended sentinel "
            f"and one ignored successor (need at least {max_prompt_len + 1})"
        )
    # Bound vLLM's KV allocation to this capture's actual context.  The model
    # advertises a 262k context, which would otherwise reserve unnecessary KV
    # memory on a multi-GPU FP8 run.
    if model_max_len is None:
        args.max_model_len = max_prompt_len + 1
    llm = _make_llm(args)
    if not kv_runtime_report_path.is_file() or kv_runtime_report_path.is_symlink():
        raise AdapterError("vLLM worker produced no KV runtime metadata report")
    try:
        kv_runtime_metadata = json.loads(
            kv_runtime_report_path.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError("KV runtime metadata report is invalid") from exc
    if not isinstance(kv_runtime_metadata, dict):
        raise AdapterError("KV runtime metadata report has an invalid shape")
    fla_autotune_details: list[dict[str, Any]] | None = None
    if args.fla_autotune == "single":
        if autotune_report_path.is_file() and not autotune_report_path.is_symlink():
            try:
                parsed_autotune = json.loads(autotune_report_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AdapterError("FLA autotune report is invalid") from exc
            if not isinstance(parsed_autotune, list) or not all(
                isinstance(item, dict) for item in parsed_autotune
            ):
                raise AdapterError("FLA autotune report has an invalid shape")
            fla_autotune_details = parsed_autotune
        else:
            raise AdapterError("single FLA autotune mode produced no configuration report")
    model_config = getattr(llm, "model_config", None)
    resolved_kv_dtype = _resolved_cache_dtype(llm)
    vocab_size = None
    if model_config is not None:
        get_vocab_size = getattr(model_config, "get_vocab_size", None)
        if callable(get_vocab_size):
            vocab_size = int(get_vocab_size())
    if not vocab_size or vocab_size < 2:
        raise AdapterError("vLLM did not expose a valid model vocabulary size")
    if args.sentinel_token_id >= vocab_size:
        raise AdapterError(
            f"sentinel token {args.sentinel_token_id} is outside vocabulary {vocab_size}"
        )
    for case in manifest["cases"]:
        if any(int(token) >= vocab_size for token in case["token_ids"]):
            raise AdapterError(f"case {case['id']} has a token outside vocab {vocab_size}")

    case_reports = [
        _capture_case(
            llm,
            case,
            args.output_dir,
            vocab_size,
            args.sentinel_token_id,
        )
        for case in manifest["cases"]
    ]
    try:
        import vllm

        vllm_version = getattr(vllm, "__version__", "unknown")
    except ImportError:  # pragma: no cover - _make_llm already imports it
        vllm_version = "unknown"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "engine": "vLLM",
        "engine_id": "vllm",
        "vllm_version": vllm_version,
        "model": args.model_repository,
        "model_repository": args.model_repository,
        "model_root": str(model_root),
        "model_revision": args.revision or "local",
        "model_fingerprint": args.model_fingerprint or config_sha256 or "unknown",
        "model_config_sha256": config_sha256,
        "model_inventory_sha256": inventory_sha256,
        "tensor_parallel_size": args.tensor_parallel_size,
        "tp": args.tensor_parallel_size,
        "dtype": args.dtype,
        "cpu_offload_gb": args.cpu_offload_gb,
        "model_dtype": _canonical_cache_dtype(getattr(model_config, "dtype", None)),
        "kv_cache_dtype_requested": args.kv_cache_dtype,
        "kv_cache_dtype_engine_arg": _engine_cache_dtype(args),
        "attention_backend_requested": args.attention_backend,
        "mamba_cache_dtype_requested": args.mamba_cache_dtype,
        "mamba_ssm_cache_dtype_requested": args.mamba_ssm_cache_dtype,
        "mamba_cache_mode_requested": args.mamba_cache_mode,
        "kv": resolved_kv_dtype,
        "kv_cache_encoding": resolved_kv_dtype,
        "kv_cache_dtype_resolved": resolved_kv_dtype,
        "kv_cache_runtime": kv_runtime_metadata,
        "fp8_e5m2_writer_fix_opt_in": args.fix_fp8_e5m2_writer,
        "fp8_e5m2_checkpoint_quant_bypass_opt_in": args.fix_fp8_e5m2_writer,
        "language_model_only": True,
        "chunk_size": args.max_num_batched_tokens or 1024,
        "logprobs_mode": "raw_logits",
        "prompt_logprobs_requested": 0,
        "raw_logits_capture": "V1 worker hidden-state hook before prompt-logprob normalization",
        "fla_autotune": args.fla_autotune,
        "fla_autotune_details": fla_autotune_details,
        "vocab_size": vocab_size,
        "sentinel_token_id": args.sentinel_token_id,
        "position_semantics": "logits after token_ids[:position+1], predicting next token",
        "context_limit": max_prompt_len - 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256_bytes(manifest_bytes),
        "output_dir": str(args.output_dir.resolve()),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
            if os.environ.get(name) is not None
        },
        "cases": case_reports,
    }
    report["output_manifest"] = str(output_manifest.resolve())
    try:
        output_manifest.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise AdapterError(f"cannot write output manifest {output_manifest}: {exc}") from exc
    return report


def main() -> int:
    args = _build_parser().parse_args()
    try:
        report = run(args)
    except (AdapterError, OSError, ValueError) as exc:
        print(f"qwen38 vLLM logit dump failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
