#!/usr/bin/env python3
"""Run a bounded CPU-only BF16 Qwen3.5-9B source oracle.

This exporter loads one local checkpoint, executes the short token-ID cases,
and writes only final hidden samples, final-logit samples, exact greedy IDs,
and top-k summaries.  It never writes a complete logit row/matrix and never
uses a network or a GPU.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as oracle  # noqa: E402

_capture_spec = importlib.util.spec_from_file_location(
    "capture_qwen35_aq4_p2_oracle",
    Path(__file__).resolve().parent / "capture-qwen35-aq4-p2-oracle.py",
)
if _capture_spec is None or _capture_spec.loader is None:
    raise RuntimeError("cannot load the strict oracle capture helper")
capture_qwen35_aq4_p2_oracle = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(capture_qwen35_aq4_p2_oracle)


DEFAULT_MODEL = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B")
DEFAULT_CASES = ROOT / "tests/fixtures/qwen35-aq4-p2-oracle/cases.json"
DEFAULT_OUTPUT = Path("/tmp/qwen35-9b-bf16-source-oracle-v1")
TOP_K = 10
HIDDEN_SAMPLE_INDICES = (0, 1, 1024, 2048, 4095)
LOGIT_SAMPLE_COUNT = 32
MIN_AVAILABLE_HEADROOM = 1.5


class ExportError(ValueError):
    pass


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _load_cases(path: Path) -> list[dict[str, Any]]:
    raw = oracle.load_json(path)
    if not isinstance(raw, dict) or set(raw) != {"cases"} or not isinstance(raw["cases"], list):
        raise ExportError("cases JSON must contain exactly a cases array")
    if not raw["cases"] or len(raw["cases"]) > oracle.MAX_CASES:
        raise ExportError("cases exceed bounded limits")
    cases = []
    seen: set[str] = set()
    for index, case in enumerate(raw["cases"]):
        if not isinstance(case, dict) or set(case) != {"case_id", "prompt_token_ids", "step_count"}:
            raise ExportError(f"cases[{index}] keys differ")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ExportError("case IDs must be unique non-empty strings")
        seen.add(case_id)
        token_ids = case["prompt_token_ids"]
        if not isinstance(token_ids, list) or not token_ids or len(token_ids) > 4096:
            raise ExportError(f"cases[{index}] token IDs are invalid")
        for token_id in token_ids:
            oracle.integer(token_id, f"cases[{index}].prompt_token_ids", minimum=0)
        step_count = oracle.integer(case["step_count"], f"cases[{index}].step_count", minimum=1)
        if step_count > oracle.MAX_STEPS:
            raise ExportError("step_count exceeds bounded limits")
        cases.append({"case_id": case_id, "prompt_token_ids": token_ids, "step_count": step_count})
    return cases


def _checkpoint_bytes(model_dir: Path) -> int:
    config = oracle.load_json(model_dir / "model.safetensors.index.json")
    if not isinstance(config, dict) or not isinstance(config.get("weight_map"), dict):
        raise ExportError("source safetensors index has no weight map")
    shards = sorted(set(config["weight_map"].values()))
    if any(not isinstance(name, str) for name in shards):
        raise ExportError("source safetensors index contains invalid shard name")
    total = 0
    for name in shards:
        path = oracle.safe_relative(model_dir, name, "checkpoint shard")
        total += path.stat().st_size
    return total


def _preflight(model_dir: Path) -> dict[str, Any]:
    checkpoint_bytes = _checkpoint_bytes(model_dir)
    memory = shutil.disk_usage(model_dir)
    del memory
    meminfo = {line.split(":", 1)[0]: int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line and line.split(":", 1)[0] in {"MemTotal", "MemAvailable"}}
    available = meminfo.get("MemAvailable", 0)
    required = int(checkpoint_bytes * MIN_AVAILABLE_HEADROOM)
    result = {"checkpoint_bytes": checkpoint_bytes, "mem_total_bytes": meminfo.get("MemTotal"), "mem_available_bytes": available, "required_headroom_bytes": required, "headroom_factor": MIN_AVAILABLE_HEADROOM, "status": "passed" if available >= required else "blocked"}
    if result["status"] != "passed":
        raise ExportError(f"CPU memory preflight failed: available={available} required={required}")
    return result


def _topk(logits: torch.Tensor, count: int = TOP_K) -> list[dict[str, Any]]:
    # This is one final-token logit vector, not a sequence/vocabulary matrix.
    values = logits.detach().to(dtype=torch.float32, device="cpu").flatten()
    if values.numel() <= count:
        indices = torch.arange(values.numel(), dtype=torch.int64)
    else:
        indices = torch.topk(values, k=count, largest=True, sorted=False).indices
    pairs = [(int(index), float(values[index])) for index in indices.tolist()]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return [{"token_id": token_id, "logit": logit} for token_id, logit in pairs[:count]]


def _sample_tensor(tensor: torch.Tensor, indices: tuple[int, ...]) -> dict[str, Any]:
    flat = tensor.detach().flatten()
    if flat.numel() <= max(indices):
        raise ExportError(f"tensor is smaller than required bounded sample: {flat.numel()}")
    selected = flat[list(indices)].to(dtype=torch.float32, device="cpu").tolist()
    return {"dtype": "f32", "indices": list(indices), "shape": [int(flat.numel())], "values": [float(value) for value in selected]}


def _record(model: Any, input_ids: list[int], case_id: str, step: int, past_key_values: Any) -> tuple[dict[str, Any], Any]:
    token_tensor = torch.tensor([input_ids], dtype=torch.long, device="cpu")
    with torch.inference_mode():
        base = model.model(input_ids=token_tensor, past_key_values=past_key_values, use_cache=True, return_dict=True)
        hidden = base.last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden.unsqueeze(1))[:, -1, :]
    topk = _topk(logits)
    greedy = topk[0]["token_id"]
    # Sample the final-logit row only at bounded positions.  Top-k indices are
    # included so the validator can re-check the reported greedy decision.
    sample_indices = tuple(sorted(set(LOGIT_SAMPLE_COUNT and [item["token_id"] for item in topk] + list(range(LOGIT_SAMPLE_COUNT)))))
    if len(sample_indices) > oracle.MAX_SAMPLE_VALUES:
        raise ExportError("bounded logit sample exceeds contract")
    record = {
        "case_id": case_id,
        "step": step,
        "greedy_token_id": greedy,
        "hidden_sample": _sample_tensor(hidden, HIDDEN_SAMPLE_INDICES),
        "logit_sample": _sample_tensor(logits, sample_indices),
        "topk": topk,
    }
    oracle.validate_payload_record(record, f"record {case_id}/{step}")
    next_past = getattr(base, "past_key_values", None)
    del token_tensor, base, hidden, logits
    return record, next_past


def _run_model(model_dir: Path, cases: list[dict[str, Any]], payload_path: Path) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if torch.cuda.is_available():
        raise ExportError("GPU is visible; source exporter is CPU-only")
    torch.set_num_threads(1)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise ExportError(f"installed transformers runtime is unavailable: {error}") from error
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None)
    model.eval()
    rows = 0
    with payload_path.open("w", encoding="utf-8") as output:
        for case in cases:
            past = None
            input_ids = list(case["prompt_token_ids"])
            try:
                for step in range(case["step_count"]):
                    record, past = _record(model, input_ids, case["case_id"], step, past)
                    encoded = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
                    output.write(encoded)
                    rows += 1
                    input_ids = [record["greedy_token_id"]]
            finally:
                del past
                gc.collect()
    del model
    gc.collect()
    return {"rows": rows, "elapsed_seconds": time.monotonic() - started}


def export(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = args.model_dir
    cases = _load_cases(args.cases)
    preflight = _preflight(model_dir)
    if args.output.exists() or os.path.lexists(args.output):
        raise ExportError(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.incomplete-", dir=args.output.parent))
    try:
        payload = temporary / "source-payload.jsonl"
        run = _run_model(model_dir, cases, payload)
        # Reuse the strict capture contract; it computes tokenizer/model/shard
        # identities and verifies the bounded payload independently.
        capture_args = type("CaptureArgs", (), {"output": args.output, "cases": args.cases, "payload": payload, "kind": "source", "source_root": model_dir, "evidence_class": "production"})()
        manifest = capture_qwen35_aq4_p2_oracle.capture(capture_args)
        manifest_path = args.output / "manifest.json"
        manifest = oracle.load_json(manifest_path)
        manifest["runtime"] = {"runtime": "transformers.AutoModelForCausalLM", "transformers": package_version("transformers"), "torch": package_version("torch"), "safetensors": package_version("safetensors"), "python": platform.python_version(), "device": "cpu", "dtype": "bfloat16", "low_cpu_mem_usage": False, "low_cpu_mem_usage_blocker": "accelerate package is unavailable in the installed environment", "preflight": preflight, "run": run}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        oracle.validate_manifest(args.output, expected_kind="source")
        runtime_path = args.output / "runtime.json"
        runtime_path.write_text(json.dumps(manifest["runtime"], ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checksum_lines = []
        for path in sorted(args.output.iterdir()):
            if path.name == "SHA256SUMS" or not path.is_file() or path.is_symlink():
                continue
            checksum_lines.append(f"{oracle.sha256_file(path)}  {path.name}\n")
        (args.output / "SHA256SUMS").write_text("".join(checksum_lines), encoding="ascii")
        return manifest
    except Exception:
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = export(args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (ExportError, oracle.OracleError, OSError, RuntimeError, ValueError) as error:
        print(f"Qwen3.5 BF16 source oracle export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
