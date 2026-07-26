#!/usr/bin/env python3
"""Small, independent HF-to-uLLM architecture trace comparator.

This tool deliberately does not import the repository's FP32 corpus, campaign,
candidate, authorization, or release code.  It has two independent boundaries:

* ``capture-hf`` runs a local Hugging Face checkpoint on CPU, captures the
  embedding, every text-decoder layer output, final norm, and last-token logits.
* ``compare`` compares that reference with a trace emitted by a uLLM debug
  runner.  The candidate writer only needs NumPy and the documented on-disk
  format below; it does not need to link this script.

Trace directory format (schema ``ullm.architecture_trace.v1``)::

    trace/
      metadata.json
      tensors.npz

``metadata.json`` has ``steps`` entries with a stable ``id`` such as
``step-0000``.  ``tensors.npz`` contains one C-contiguous F32 array per
``step``/``tensor`` pair, named ``<step-id>__<tensor-name>``.  Required tensor
names are ``embedding``, ``layer-0000`` ... ``layer-NNNN``, ``final-norm``,
and ``logits-last``.  uLLM candidates must use the identical input token IDs,
step order, and tensor shapes.  The comparison report makes the first failing
layer/step explicit.

The normal reference scope is intentionally tiny: a short prompt plus a few
greedy decode steps, CPU-only, with a bounded Torch thread count.  It is an
architecture bring-up diagnostic, not a performance benchmark or a production
release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = "ullm.architecture_trace.v1"
DEFAULT_ATOL = 5.0e-5
DEFAULT_RTOL = 5.0e-4
DEFAULT_RELATIVE_FLOOR = 1.0e-4
DEFAULT_L2_REL_MAX = 1.0e-4
MAX_PROMPT_TOKENS = 64
MAX_NEW_TOKENS = 4


class TraceError(RuntimeError):
    """A deterministic error suitable for a short bring-up report."""


@dataclass(frozen=True)
class TensorComparison:
    key: str
    count: int
    failed_count: int
    max_abs_error: float
    max_relative_error: float
    l2_relative_error: float
    allowed_max: float
    passed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture-hf", help="capture an HF CPU reference trace")
    capture.add_argument("--model-dir", type=Path, required=True)
    prompt = capture.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="plain text prompt; tokenized locally without network access")
    prompt.add_argument("--token-ids", help="comma-separated explicit token IDs")
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--new-tokens", type=int, default=2)
    capture.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    capture.add_argument("--threads", type=int, default=8)
    capture.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    capture.add_argument(
        "--allow-quantized-reference",
        action="store_true",
        help="acknowledge that the input checkpoint itself is quantized (Qwen3-14B-FP8 baseline only)",
    )

    compare = subparsers.add_parser("compare", help="compare an HF trace and a uLLM trace")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--report", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    compare.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    compare.add_argument("--relative-floor", type=float, default=DEFAULT_RELATIVE_FLOOR)
    compare.add_argument("--l2-relative-max", type=float, default=DEFAULT_L2_REL_MAX)

    selftest = subparsers.add_parser("self-test", help="exercise serialization and failure localization")
    selftest.add_argument("--output", type=Path, required=True)

    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TraceError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise TraceError(f"{path} must contain a JSON object")
    return value


def parse_token_ids(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            raise TraceError("--token-ids contains an empty item")
        try:
            value = int(part)
        except ValueError as error:
            raise TraceError(f"invalid token ID {part!r}") from error
        if value < 0:
            raise TraceError(f"token ID must be non-negative, got {value}")
        values.append(value)
    if not values:
        raise TraceError("--token-ids requires at least one ID")
    return values


def first_tensor(value: object, label: str):
    """Extract a Tensor from standard HF module outputs without model-specific casts."""
    import torch

    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item, label)
            except TraceError:
                continue
    if hasattr(value, "last_hidden_state"):
        candidate = getattr(value, "last_hidden_state")
        if torch.is_tensor(candidate):
            return candidate
    raise TraceError(f"{label} hook output did not contain a Tensor")


def tensor_f32(value: object, label: str) -> np.ndarray:
    tensor = first_tensor(value, label)
    import torch

    # Explicitly materialize F32 after the device transfer rather than relying
    # on the model's parameter dtype.
    output = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    array = output.numpy()
    if not np.isfinite(array).all():
        raise TraceError(f"{label} contains NaN or infinity")
    return array


def resolve_text_model(model: object) -> object:
    """Find the text decoder underneath causal and multimodal HF wrappers."""
    candidates = [model]
    if hasattr(model, "model"):
        candidates.append(getattr(model, "model"))
    for candidate in list(candidates):
        if hasattr(candidate, "language_model"):
            candidates.append(getattr(candidate, "language_model"))
    for candidate in candidates:
        if hasattr(candidate, "layers") and hasattr(candidate, "embed_tokens") and hasattr(candidate, "norm"):
            return candidate
    names = ", ".join(type(candidate).__name__ for candidate in candidates)
    raise TraceError(f"cannot locate text decoder layers below {names}")


def model_class_for_config(config: object):
    """Resolve the smallest public AutoModel entry point that can load the config."""
    import transformers

    architectures = list(getattr(config, "architectures", []) or [])
    if any("ForConditionalGeneration" in name for name in architectures):
        for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
            candidate = getattr(transformers, name, None)
            if candidate is not None:
                return candidate
        raise TraceError(
            "this Transformers installation has no public multimodal auto-model class; "
            "upgrade Transformers or use a matching environment"
        )
    candidate = getattr(transformers, "AutoModelForCausalLM", None)
    if candidate is None:
        raise TraceError("Transformers has no AutoModelForCausalLM")
    return candidate


def torch_dtype(name: str):
    import torch

    return torch.float32 if name == "float32" else torch.bfloat16


def model_token_ids(args: argparse.Namespace, model_dir: Path) -> list[int]:
    if args.token_ids is not None:
        values = parse_token_ids(args.token_ids)
    else:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            values = list(tokenizer(args.prompt, add_special_tokens=True)["input_ids"])
        except Exception as error:  # pragma: no cover - depends on optional tokenizer extras
            raise TraceError(
                "local tokenizer load failed; install its tokenizer dependency or use --token-ids. "
                f"Original error: {error}"
            ) from error
    if len(values) > args.max_prompt_tokens:
        values = values[-args.max_prompt_tokens :]
    if not values:
        raise TraceError("prompt tokenization produced zero tokens")
    return [int(value) for value in values]


def config_weight_format(config: dict[str, Any]) -> str:
    quantization = config.get("quantization_config")
    if isinstance(quantization, dict):
        method = quantization.get("quant_method", "unknown")
        return f"quantized:{method}"
    return str(config.get("dtype", config.get("torch_dtype", "unspecified")))


def output_key(step_id: str, tensor_name: str) -> str:
    return f"{step_id}__{tensor_name}"


def sorted_tensor_names(names: Iterable[str]) -> list[str]:
    def key(name: str) -> tuple[int, int | str]:
        if name == "embedding":
            return (0, 0)
        if name.startswith("layer-"):
            return (1, int(name.removeprefix("layer-")))
        if name == "final-norm":
            return (2, 0)
        if name == "logits-last":
            return (3, 0)
        return (4, name)

    return sorted(names, key=key)


def capture_hf(args: argparse.Namespace) -> int:
    import torch
    import transformers
    from transformers import AutoConfig

    if args.threads < 1 or args.threads > 32:
        raise TraceError("--threads must be in 1..32 to keep this diagnostic lightweight")
    if args.new_tokens < 1 or args.new_tokens > MAX_NEW_TOKENS:
        raise TraceError(f"--new-tokens must be in 1..{MAX_NEW_TOKENS}")
    if args.max_prompt_tokens < 1 or args.max_prompt_tokens > MAX_PROMPT_TOKENS:
        raise TraceError(f"--max-prompt-tokens must be in 1..{MAX_PROMPT_TOKENS}")
    if args.output.exists():
        raise TraceError(f"output already exists; refusing to overwrite {args.output}")

    model_dir = args.model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise TraceError(f"missing {config_path}")
    raw_config = json_load(config_path)
    weight_format = config_weight_format(raw_config)
    if weight_format.startswith("quantized:") and not args.allow_quantized_reference:
        raise TraceError(
            "checkpoint declares quantization_config; use an unquantized source for the strict FP32 path, "
            "or explicitly pass --allow-quantized-reference for trace-plumbing validation"
        )

    # Do not inherit a busy host's large OpenMP setting.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Some parent processes set this before importing us.  Intra-op remains
        # bounded, which is the meaningful control for this script.
        pass

    token_ids = model_token_ids(args, model_dir)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    model_class = model_class_for_config(config)
    load_start = time.monotonic()
    try:
        model = model_class.from_pretrained(
            str(model_dir),
            local_files_only=True,
            dtype=torch_dtype(args.dtype),
        )
    except TypeError:
        # Older Transformers used ``torch_dtype``.  Keep the fallback local and
        # explicit so a trace records exactly which version was used.
        model = model_class.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch_dtype(args.dtype),
        )
    model.eval()
    text_model = resolve_text_model(model)
    layers = list(getattr(text_model, "layers"))
    if not layers:
        raise TraceError("text decoder has zero layers")

    captured: dict[str, np.ndarray] = {}
    handles: list[object] = []

    def install_hook(name: str, module: object) -> None:
        def hook(_module: object, _inputs: object, output: object) -> None:
            captured[name] = tensor_f32(output, name)

        handles.append(module.register_forward_hook(hook))

    install_hook("embedding", getattr(text_model, "embed_tokens"))
    for index, layer in enumerate(layers):
        install_hook(f"layer-{index:04d}", layer)
    install_hook("final-norm", getattr(text_model, "norm"))

    try:
        step_arrays: dict[str, np.ndarray] = {}
        step_metadata: list[dict[str, Any]] = []
        generated: list[int] = []
        next_input = torch.tensor([token_ids], dtype=torch.long)
        past_key_values: object | None = None
        for step_index in range(args.new_tokens):
            step_id = f"step-{step_index:04d}"
            captured.clear()
            started = time.monotonic()
            with torch.inference_mode():
                outputs = model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            elapsed = time.monotonic() - started
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise TraceError("model output has no logits")
            logits_last = logits[:, -1, :].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if not np.isfinite(logits_last).all():
                raise TraceError("logits-last contains NaN or infinity")
            expected_names = ["embedding", *[f"layer-{i:04d}" for i in range(len(layers))], "final-norm"]
            missing = [name for name in expected_names if name not in captured]
            if missing:
                raise TraceError(f"{step_id} did not capture required tensors: {missing}")
            for name in expected_names:
                step_arrays[output_key(step_id, name)] = captured[name]
            step_arrays[output_key(step_id, "logits-last")] = logits_last
            greedy = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            generated.append(greedy)
            step_metadata.append(
                {
                    "id": step_id,
                    "input_token_ids": [int(value) for value in next_input.reshape(-1).tolist()],
                    "greedy_next_token_id": greedy,
                    "elapsed_seconds": elapsed,
                    "tensor_names": expected_names + ["logits-last"],
                    "tensor_shapes": {
                        name: list(step_arrays[output_key(step_id, name)].shape)
                        for name in expected_names + ["logits-last"]
                    },
                }
            )
            past_key_values = getattr(outputs, "past_key_values", None)
            if step_index < args.new_tokens - 1:
                if past_key_values is None:
                    raise TraceError("model did not return past_key_values for decode step")
                next_input = torch.tensor([[greedy]], dtype=torch.long)
    finally:
        for handle in handles:
            handle.remove()

    args.output.mkdir(parents=True, exist_ok=False)
    tensor_path = args.output / "tensors.npz"
    np.savez_compressed(tensor_path, **step_arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "producer": "huggingface-cpu-reference",
        "model_dir": str(model_dir),
        "config_sha256": sha256_file(config_path),
        "architectures": list(getattr(config, "architectures", []) or []),
        "model_type": str(getattr(config, "model_type", "")),
        "weight_format": weight_format,
        "allow_quantized_reference": bool(args.allow_quantized_reference),
        "compute_dtype": args.dtype,
        "device": "cpu",
        "torch_threads": args.threads,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "initial_token_ids": token_ids,
        "generated_token_ids": generated,
        "load_and_run_elapsed_seconds": time.monotonic() - load_start,
        "steps": step_metadata,
        "tensors_file": tensor_path.name,
        "tensors_sha256": sha256_file(tensor_path),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"captured {len(layers)} layers x {len(step_metadata)} CPU steps to {args.output} "
        f"(load+run {metadata['load_and_run_elapsed_seconds']:.1f}s)"
    )
    return 0


def load_trace(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = root.expanduser().resolve()
    metadata_path = root / "metadata.json"
    metadata = json_load(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise TraceError(f"{metadata_path} has unsupported schema {metadata.get('schema_version')!r}")
    tensor_name = metadata.get("tensors_file")
    if not isinstance(tensor_name, str) or not tensor_name:
        raise TraceError(f"{metadata_path} has no tensors_file")
    tensor_path = root / tensor_name
    if not tensor_path.is_file():
        raise TraceError(f"missing trace tensor file {tensor_path}")
    expected_sha = metadata.get("tensors_sha256")
    if isinstance(expected_sha, str) and sha256_file(tensor_path) != expected_sha:
        raise TraceError(f"trace tensor checksum mismatch for {tensor_path}")
    try:
        with np.load(tensor_path, allow_pickle=False) as loaded:
            arrays = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    except (OSError, ValueError) as error:
        raise TraceError(f"cannot read {tensor_path}: {error}") from error
    steps = metadata.get("steps")
    if not isinstance(steps, list) or not steps:
        raise TraceError(f"{metadata_path} must contain nonempty steps")
    expected_keys: set[str] = set()
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("id"), str):
            raise TraceError(f"{metadata_path} has invalid step metadata")
        names = step.get("tensor_names")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise TraceError(f"{metadata_path} step {step.get('id')!r} has invalid tensor_names")
        expected_keys.update(output_key(step["id"], name) for name in names)
    missing = sorted(expected_keys.difference(arrays))
    extra = sorted(set(arrays).difference(expected_keys))
    if missing or extra:
        raise TraceError(f"trace tensor set differs from metadata: missing={missing}, extra={extra}")
    for key, array in arrays.items():
        if array.dtype != np.float32:
            raise TraceError(f"{key} is {array.dtype}; candidate and reference tensors must be F32")
        if not array.flags.c_contiguous:
            raise TraceError(f"{key} is not C-contiguous")
        if not np.isfinite(array).all():
            raise TraceError(f"{key} contains NaN or infinity")
    return metadata, arrays


def compare_array(
    key: str,
    actual: np.ndarray,
    reference: np.ndarray,
    atol: float,
    rtol: float,
    relative_floor: float,
    l2_relative_max: float,
) -> TensorComparison:
    if actual.shape != reference.shape:
        raise TraceError(f"{key}: shape mismatch candidate={list(actual.shape)} reference={list(reference.shape)}")
    diff = actual.astype(np.float64) - reference.astype(np.float64)
    abs_error = np.abs(diff)
    allowed = atol + rtol * np.abs(reference.astype(np.float64))
    relative = abs_error / np.maximum(np.abs(reference.astype(np.float64)), relative_floor)
    l2_denominator = max(float(np.linalg.norm(reference.astype(np.float64).ravel())), relative_floor)
    l2_relative = float(np.linalg.norm(diff.ravel()) / l2_denominator)
    failed = int(np.count_nonzero(abs_error > allowed))
    return TensorComparison(
        key=key,
        count=int(actual.size),
        failed_count=failed,
        max_abs_error=float(np.max(abs_error)),
        max_relative_error=float(np.max(relative)),
        l2_relative_error=l2_relative,
        allowed_max=float(np.max(allowed)),
        passed=failed == 0 and l2_relative <= l2_relative_max,
    )


def compare_traces(args: argparse.Namespace) -> int:
    if args.atol < 0 or args.rtol < 0 or args.relative_floor <= 0 or args.l2_relative_max < 0:
        raise TraceError("comparison tolerances must be non-negative; --relative-floor must be positive")
    if args.report.exists():
        raise TraceError(f"report already exists; refusing to overwrite {args.report}")
    reference_metadata, reference_arrays = load_trace(args.reference)
    candidate_metadata, candidate_arrays = load_trace(args.candidate)
    for field in (
        "architectures",
        "model_type",
        "config_sha256",
        "initial_token_ids",
        "generated_token_ids",
    ):
        if reference_metadata.get(field) != candidate_metadata.get(field):
            raise TraceError(
                f"metadata mismatch for {field}: candidate={candidate_metadata.get(field)!r} "
                f"reference={reference_metadata.get(field)!r}"
            )
    if set(reference_arrays) != set(candidate_arrays):
        raise TraceError("candidate and reference arrays have different names")
    comparisons = [
        compare_array(
            key,
            candidate_arrays[key],
            reference_arrays[key],
            args.atol,
            args.rtol,
            args.relative_floor,
            args.l2_relative_max,
        )
        for key in sorted(reference_arrays)
    ]
    failed = [item for item in comparisons if not item.passed]
    report = {
        "schema_version": "ullm.architecture_trace_comparison.v1",
        "status": "pass" if not failed else "fail",
        "reference": str(args.reference.expanduser().resolve()),
        "candidate": str(args.candidate.expanduser().resolve()),
        "tolerances": {
            "atol": args.atol,
            "rtol": args.rtol,
            "relative_floor": args.relative_floor,
            "l2_relative_max": args.l2_relative_max,
        },
        "first_failure": failed[0].key if failed else None,
        "comparison_count": len(comparisons),
        "failure_count": len(failed),
        "comparisons": [item.__dict__ for item in comparisons],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"comparison={report['status']} tensors={len(comparisons)} first_failure={report['first_failure']}")
    return 0 if not failed else 2


def write_trace(root: Path, producer: str, arrays: dict[str, np.ndarray], layer_count: int) -> None:
    root.mkdir(parents=True, exist_ok=False)
    step_id = "step-0000"
    tensor_path = root / "tensors.npz"
    np.savez_compressed(tensor_path, **arrays)
    names = sorted_tensor_names(name.removeprefix(f"{step_id}__") for name in arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "producer": producer,
        "architectures": ["SyntheticForSelfTest"],
        "model_type": "synthetic",
        "initial_token_ids": [1, 2, 3],
        "generated_token_ids": [4],
        "tensors_file": tensor_path.name,
        "tensors_sha256": sha256_file(tensor_path),
        "steps": [
            {
                "id": step_id,
                "input_token_ids": [1, 2, 3],
                "greedy_next_token_id": 4,
                "tensor_names": names,
                "tensor_shapes": {name: list(arrays[output_key(step_id, name)].shape) for name in names},
            }
        ],
        "self_test_layer_count": layer_count,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def self_test(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise TraceError(f"output already exists; refusing to overwrite {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.output / "reference"
    candidate = args.output / "candidate"
    rng = np.random.default_rng(20260726)
    step = "step-0000"
    names = ["embedding", "layer-0000", "layer-0001", "layer-0002", "layer-0003", "final-norm", "logits-last"]
    arrays = {
        output_key(step, name): np.ascontiguousarray(rng.standard_normal((1, 3, 8), dtype=np.float32))
        for name in names
    }
    arrays[output_key(step, "logits-last")] = np.ascontiguousarray(rng.standard_normal((1, 17), dtype=np.float32))
    write_trace(reference, "synthetic-reference", arrays, layer_count=4)
    candidate_arrays = {name: np.array(value, copy=True, order="C") for name, value in arrays.items()}
    candidate_arrays[output_key(step, "layer-0003")][0, 1, 2] += np.float32(1.0)
    write_trace(candidate, "synthetic-uLLM-candidate", candidate_arrays, layer_count=4)
    report = args.output / "negative-report.json"
    compare_args = argparse.Namespace(
        reference=reference,
        candidate=candidate,
        report=report,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        relative_floor=DEFAULT_RELATIVE_FLOOR,
        l2_relative_max=DEFAULT_L2_REL_MAX,
    )
    status = compare_traces(compare_args)
    result = json_load(report)
    if status != 2 or result.get("first_failure") != "step-0000__layer-0003":
        raise TraceError("self-test did not reject and localize the deliberate layer-0003 error")
    print(f"self-test=pass negative_failure={result['first_failure']}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        if args.command == "capture-hf":
            return capture_hf(args)
        if args.command == "compare":
            return compare_traces(args)
        if args.command == "self-test":
            return self_test(args)
        raise AssertionError(f"unknown command {args.command}")
    except TraceError as error:
        print(f"architecture_hf_trace: error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
