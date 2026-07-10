#!/usr/bin/env python3
"""Export the fixed Qwen3-14B-FP8 M=8 full-model vLLM oracle."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import functools
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.qwen3_full_model_oracle.v1"
DEFAULT_MODEL = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)
DEFAULT_OUTPUT = Path("/tmp/ullm-qwen3-14b-fp8-vllm-oracle-m8-v0.1")
DEFAULT_PYTHON = Path(
    "/home/homelab1/coding-local/ultimateLLM/"
    "uLLM-project/build/envs/vllm-rocm-nightly/bin/python"
)
TOKEN_IDS = tuple(range(1, 9))
TOP_K = 10
KV_CACHE_MEMORY_BYTES = 64 * 1024 * 1024
EXPECTED_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
EXPECTED_REVISION_FILES = {
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
EXPECTED_CHECKPOINT_FILES = (
    ("config.json", 896, "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793"),
    (
        "model.safetensors.index.json",
        62044,
        "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
    ),
    (
        "model-00001-of-00004.safetensors",
        4922397616,
        "2c2f93f7639950a7246c54457482696b94aa0e6b1f49d2169f0422f56c1ed370",
    ),
    (
        "model-00002-of-00004.safetensors",
        4955472248,
        "7831581bc7d03d77707df3ef10b8d90ee1998ee890ea0020b4a62d27079925ba",
    ),
    (
        "model-00003-of-00004.safetensors",
        4892558664,
        "d57d1788fb339440b12c6917f7f88e18a5cb76e20f0bfacadd9e4e70a49b2a2a",
    ),
    (
        "model-00004-of-00004.safetensors",
        1555824768,
        "b4bf668aa6f8535dd467a9a3339116b536682b4241972054b783d514cbe84e50",
    ),
    (
        "tokenizer_config.json",
        9732,
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    ),
)
EXPECTED_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "max_position_embeddings": 40960,
    "max_window_layers": 40,
    "model_type": "qwen3",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}

# Imported only after the no-clobber and fixed-checkpoint preflight checks.
np: Any = None
torch: Any = None
LLM: Any = None
SamplingParams: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_unchanged(label: str, before: Any, after: Any) -> None:
    if before != after:
        raise RuntimeError(f"{label} changed while the oracle was being exported")


def ensure_output_available(output_dir: Path) -> None:
    if os.path.lexists(output_dir):
        raise SystemExit(f"refusing to overwrite existing output: {output_dir}")


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing a raced path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"refusing to overwrite raced output: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def checkpoint_revision(model_dir: Path) -> dict[str, Any]:
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    revisions: dict[str, str] = {}
    if metadata_dir.is_dir():
        for path in sorted(metadata_dir.glob("*.metadata")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                revisions[path.name.removesuffix(".metadata")] = lines[0]
    unique = sorted(set(revisions.values()))
    return {
        "revision": unique[0] if len(unique) == 1 else None,
        "per_file_revisions": revisions,
        "revision_consistent": len(unique) == 1,
    }


def validate_revision_contract(revision: dict[str, Any]) -> None:
    per_file = revision.get("per_file_revisions")
    if (
        revision.get("revision") != EXPECTED_REVISION
        or revision.get("revision_consistent") is not True
        or not isinstance(per_file, dict)
        or set(per_file) != EXPECTED_REVISION_FILES
        or any(value != EXPECTED_REVISION for value in per_file.values())
    ):
        raise SystemExit("checkpoint revision metadata does not match the fixed revision set")


def verify_model_contract(
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to load fixed model config: {error}") from error
    if config != EXPECTED_CONFIG:
        raise SystemExit("model config does not match the fixed Qwen3-14B-FP8 contract")

    records = []
    for name, expected_bytes, expected_sha256 in EXPECTED_CHECKPOINT_FILES:
        path = model_dir / name
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise SystemExit(f"missing fixed checkpoint file {name}: {error}") from error
        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"checkpoint size mismatch for {name}: {actual_bytes} != {expected_bytes}"
            )
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise SystemExit(f"checkpoint SHA-256 mismatch for {name}")
        records.append({"file": name, "bytes": actual_bytes, "sha256": actual_sha256})

    revision = checkpoint_revision(model_dir)
    validate_revision_contract(revision)
    return config, revision, records


def load_runtime_dependencies() -> None:
    global LLM, SamplingParams, np, torch
    try:
        import numpy as numpy_module
        import torch as torch_module
        from vllm import LLM as llm_class
        from vllm import SamplingParams as sampling_params_class
    except ImportError as error:
        raise SystemExit(f"vLLM oracle dependencies are unavailable: {error}") from error
    np = numpy_module
    torch = torch_module
    LLM = llm_class
    SamplingParams = sampling_params_class


def tensor_health(tensor_f32: Any) -> dict[str, Any]:
    flat = tensor_f32.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    result: dict[str, Any] = {
        "elements": int(flat.numel()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(flat).sum().item()),
        "inf_count": int(torch.isinf(flat).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std_population": float(finite_values.std(unbiased=False).item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
                "max_abs": float(finite_values.abs().max().item()),
            }
        )
    return result


def write_f32_tensor(path: Path, tensor: Any) -> dict[str, Any]:
    source_dtype = str(tensor.dtype)
    host = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    array = host.numpy().astype("<f4", copy=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {
        "file": str(path.name if path.parent.name != "layers" else Path("layers") / path.name),
        "shape": [int(value) for value in host.shape],
        "storage_dtype": "float32_le",
        "source_dtype": source_dtype,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "health": tensor_health(host),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def visible_gpu() -> dict[str, Any]:
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"oracle requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    props = torch.cuda.get_device_properties(0)
    gfx = str(getattr(props, "gcnArchName", ""))
    capability = list(torch.cuda.get_device_capability(0))
    if gfx != "gfx1201" or capability != [12, 0]:
        raise RuntimeError(
            f"oracle requires R9700/gfx1201 compute 12.0, got "
            f"{props.name}/{gfx}/{capability}"
        )
    return {
        "visible_device_index": 0,
        "name": str(props.name),
        "gfx": gfx,
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": capability,
        "rocr_visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES"),
    }


def install_capture(model: Any, work_dir: str) -> dict[str, Any]:
    root = Path(work_dir)
    layers = list(model.model.layers)
    if len(layers) != 40:
        raise RuntimeError(f"expected 40 decoder layers, found {len(layers)}")

    model._ullm_oracle_layer_records = {}
    model._ullm_oracle_handles = []

    def capture_materialized_layer(layer_index: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    f"layer {layer_index} materialization hook expected (normed, residual)"
                )
            logical_output = output[1]
            if not torch.is_tensor(logical_output):
                raise RuntimeError(f"layer {layer_index} residual output is not a tensor")
            records = model._ullm_oracle_layer_records
            if layer_index in records:
                raise RuntimeError(f"layer {layer_index} was captured more than once")
            record = write_f32_tensor(
                root / "layers" / f"layer-{layer_index:02d}-output.f32",
                logical_output,
            )
            record["layer_index"] = layer_index
            record["semantic"] = (
                "post_mlp_residual_output_materialized_by_next_fused_rms_norm"
            )
            records[layer_index] = record

        return hook

    for next_layer_index in range(1, len(layers)):
        handle = layers[next_layer_index].input_layernorm.register_forward_hook(
            capture_materialized_layer(next_layer_index - 1)
        )
        model._ullm_oracle_handles.append(handle)

    def capture_final_norm(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            raise RuntimeError("final norm hook expected (normed, residual)")
        final_hidden, layer_39_output = output[0], output[1]
        if not torch.is_tensor(final_hidden) or not torch.is_tensor(layer_39_output):
            raise RuntimeError("final norm hook outputs are not tensors")
        records = model._ullm_oracle_layer_records
        if 39 in records:
            raise RuntimeError("layer 39 was captured more than once")
        layer_record = write_f32_tensor(
            root / "layers" / "layer-39-output.f32", layer_39_output
        )
        layer_record["layer_index"] = 39
        layer_record["semantic"] = (
            "post_mlp_residual_output_materialized_by_final_fused_rms_norm"
        )
        records[39] = layer_record
        model._ullm_oracle_final_hidden = final_hidden.detach().clone()
        model._ullm_oracle_final_hidden_record = write_f32_tensor(
            root / "final-hidden.f32", final_hidden
        )
        model._ullm_oracle_final_hidden_record["semantic"] = (
            "post_final_rms_norm_pre_lm_head"
        )

    model._ullm_oracle_handles.append(
        model.model.norm.register_forward_hook(capture_final_norm)
    )

    quant_method = model.model.layers[0].self_attn.qkv_proj.quant_method
    rope_parameters = getattr(model.config, "rope_parameters", None) or {}
    rope_theta = getattr(model.config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta")
    if rope_theta is None:
        raise RuntimeError("model config does not expose rope_theta")
    return {
        "model_class": type(model).__name__,
        "decoder_layer_class": type(layers[0]).__name__,
        "final_norm_class": type(model.model.norm).__name__,
        "lm_head_class": type(model.lm_head).__name__,
        "decoder_layer_count": len(layers),
        "quant_config_class": type(model.quant_config).__name__,
        "quant_config_repr": repr(model.quant_config),
        "qkv_quant_method_class": type(quant_method).__name__,
        "config_hidden_size": int(model.config.hidden_size),
        "config_vocab_size": int(model.config.vocab_size),
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "rms_norm_eps": float(model.config.rms_norm_eps),
        "rope_theta": float(rope_theta),
        "head_dim": int(model.config.head_dim),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
    }


def collect_logits(model: Any, work_dir: str) -> dict[str, Any]:
    records = model._ullm_oracle_layer_records
    if sorted(records) != list(range(40)):
        missing = sorted(set(range(40)) - set(records))
        raise RuntimeError(f"missing layer captures: {missing}")
    hidden = model._ullm_oracle_final_hidden
    with torch.inference_mode():
        logits = model.compute_logits(hidden)
    if logits is None:
        raise RuntimeError("model.compute_logits returned None")
    logits_record = write_f32_tensor(Path(work_dir) / "logits.f32", logits)
    logits_record["semantic"] = "raw_pre_softmax_logits_for_each_prompt_position"
    logits_f32 = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    topk = []
    for position in range(logits_f32.shape[0]):
        row = logits_f32[position].numpy()
        token_ids = np.arange(row.size, dtype=np.int64)
        indices = np.lexsort((token_ids, -row))[:TOP_K]
        values = row[indices]
        topk.append(
            {
                "position": position,
                "token_ids": [int(value) for value in indices],
                "logits": [float(value) for value in values],
                "top1_top2_margin": float(values[0] - values[1]),
            }
        )
    for handle in model._ullm_oracle_handles:
        handle.remove()
    return {
        "layers": [records[index] for index in range(40)],
        "final_hidden": model._ullm_oracle_final_hidden_record,
        "logits": logits_record,
        "top_k": TOP_K,
        "topk_tie_breaker": "token_id_ascending",
        "topk_by_position": topk,
    }


def write_rerun_files(
    work_dir: Path,
    output_dir: Path,
    model_dir: Path,
    exporter_bytes: bytes,
    exporter_sha256: str,
) -> None:
    copied_script = work_dir / "export_oracle.py"
    copied_script.write_bytes(exporter_bytes)
    require_unchanged(
        "captured exporter bytes",
        exporter_sha256,
        sha256_file(copied_script),
    )
    copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR)
    rerun_output = str(output_dir) + "-rerun"
    arguments = ["--model-dir", str(model_dir), "--output-dir", rerun_output]
    shell_arguments = " ".join(shlex.quote(part) for part in arguments)
    rerun = work_dir / "rerun-command.sh"
    rerun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "export ROCR_VISIBLE_DEVICES=1\n"
        "export VLLM_ENABLE_V1_MULTIPROCESSING=0\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'exec {shlex.quote(str(DEFAULT_PYTHON))} "$SCRIPT_DIR/export_oracle.py" '
        f"{shell_arguments}\n",
        encoding="ascii",
    )
    rerun.chmod(rerun.stat().st_mode | stat.S_IXUSR)


def artifact_manifest(work_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(work_dir.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "file": str(path.relative_to(work_dir)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def main() -> int:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    exporter_path = Path(__file__).resolve()
    exporter_bytes = exporter_path.read_bytes()
    exporter_sha256 = sha256_bytes(exporter_bytes)

    # This check intentionally precedes all heavyweight imports and model hashing.
    ensure_output_available(output_dir)
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if os.environ.get("ROCR_VISIBLE_DEVICES") != "1":
        raise SystemExit("ROCR_VISIBLE_DEVICES must be exactly 1 for the local R9700")
    if Path(sys.executable).resolve() != DEFAULT_PYTHON.resolve():
        raise SystemExit(f"oracle must run with the fixed interpreter: {DEFAULT_PYTHON}")

    initial_model_snapshot = verify_model_contract(model_dir)
    config, revision, file_hashes = initial_model_snapshot
    load_runtime_dependencies()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )
    llm: Any = None
    success = False
    try:
        gpu = visible_gpu()
        llm = LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="auto",
            quantization="fp8",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=len(TOKEN_IDS) + 1,
            max_num_seqs=1,
            max_num_batched_tokens=len(TOKEN_IDS),
            kv_cache_memory_bytes=KV_CACHE_MEMORY_BYTES,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=True,
            seed=0,
        )
        model_info = llm.apply_model(
            functools.partial(install_capture, work_dir=str(work_dir))
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            ignore_eos=True,
            logprobs=TOP_K,
        )
        requests = llm.generate(
            [{"prompt_token_ids": list(TOKEN_IDS)}], sampling, use_tqdm=False
        )
        oracle = llm.apply_model(
            functools.partial(collect_logits, work_dir=str(work_dir))
        )[0]
        generated_token_ids = [int(value) for value in requests[0].outputs[0].token_ids]
        expected_top1 = int(oracle["topk_by_position"][-1]["token_ids"][0])
        if generated_token_ids != [expected_top1]:
            raise RuntimeError(
                f"sampler token {generated_token_ids} != final-logits top1 {expected_top1}"
            )

        final_model_snapshot = verify_model_contract(model_dir)
        require_unchanged(
            "checkpoint files and revision metadata",
            initial_model_snapshot,
            final_model_snapshot,
        )
        require_unchanged(
            "exporter script",
            exporter_sha256,
            sha256_bytes(exporter_path.read_bytes()),
        )
        write_rerun_files(
            work_dir,
            output_dir,
            model_dir,
            exporter_bytes,
            exporter_sha256,
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": {
                "name": "Qwen/Qwen3-14B-FP8",
                "local_dir": str(model_dir),
                "revision": revision,
                "checkpoint_files": file_hashes,
                "config": config,
            },
            "input": {
                "token_ids": list(TOKEN_IDS),
                "position_ids": list(range(len(TOKEN_IDS))),
                "attention": "causal",
                "bos_inserted": False,
                "chat_template_applied": False,
                "eos_ignored_for_single_sampler_cross_check": True,
                "eos_token_id": int(config["eos_token_id"]),
            },
            "execution": {
                "backend": "vLLM",
                "runner": "generate",
                "dtype": "bfloat16",
                "quantization": config["quantization_config"],
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "max_model_len": len(TOKEN_IDS) + 1,
                "max_num_seqs": 1,
                "max_num_batched_tokens": len(TOKEN_IDS),
                "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
                "enforce_eager": True,
                "enable_prefix_caching": False,
                "async_scheduling": False,
                "seed": 0,
                "v1_multiprocessing": False,
                "model_info": model_info,
            },
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "packages": {
                    name: package_version(name)
                    for name in [
                        "vllm",
                        "torch",
                        "transformers",
                        "safetensors",
                        "accelerate",
                        "triton",
                        "numpy",
                    ]
                },
                "torch_git_version": torch.version.git_version,
                "torch_hip_version": torch.version.hip,
                "rocm_version_file": (
                    Path("/opt/rocm/.info/version").read_text(encoding="ascii").strip()
                    if Path("/opt/rocm/.info/version").exists()
                    else None
                ),
                "gpu": gpu,
            },
            "semantics": {
                "layer_output": (
                    "logical post-MLP residual stream, captured as the second output of "
                    "the following fused RMSNorm (or final RMSNorm for layer 39)"
                ),
                "final_hidden": "post-final-RMSNorm, immediately before lm_head",
                "logits": (
                    "raw lm_head logits before softmax; row i predicts the token after "
                    "the prefix ending at input position i"
                ),
                "lm_head_tied": False,
                "lm_head_bias": False,
            },
            "oracle": oracle,
            "sampler_cross_check": {
                "generated_token_ids": generated_token_ids,
                "final_position_top1_token_id": expected_top1,
                "matches": True,
            },
            "artifact_files_excluding_metadata": artifact_manifest(work_dir),
        }
        (work_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rename_noreplace(work_dir, output_dir)
        success = True
        print(
            json.dumps(
                {
                    "output_dir": str(output_dir),
                    "layer_count": len(oracle["layers"]),
                    "final_hidden_sha256": oracle["final_hidden"]["sha256"],
                    "logits_sha256": oracle["logits"]["sha256"],
                    "final_topk": oracle["topk_by_position"][-1],
                    "generated_token_ids": generated_token_ids,
                },
                sort_keys=True,
            )
        )
    finally:
        if llm is not None:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                try:
                    engine.shutdown()
                except Exception as error:
                    print(f"shutdown warning: {error!r}", file=sys.stderr)
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as error:
            print(f"distributed cleanup warning: {error!r}", file=sys.stderr)
        del llm
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not success and work_dir.exists():
            shutil.rmtree(work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
