#!/usr/bin/env python3
"""Capture and analyze bounded Qwen3.5 source/AQ4 differential traces.

The source capture is CPU-only and uses forward hooks that retain only fixed
coordinate samples and streaming summary statistics for embedding, every
decoder layer, final norm, and LM head.  The analyzer accepts a matching AQ4
trace (or an endpoint adapter made from the existing path payload) and reports
the first stage whose bounded sample diverges.  It never stores a full hidden
state or vocabulary row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import qwen35_aq4_p2_oracle as ORACLE  # noqa: E402

SCHEMA = "ullm.qwen35_aq4_differential_trace.v1"
ANALYSIS_SCHEMA = "ullm.qwen35_aq4_differential_analysis.v1"
HIDDEN_COORDINATES = (0, 1, 1024, 2048, 4095)
LOGIT_COORDINATES = tuple(range(32))
TOP_K = 10
MAX_JSON_BYTES = 64 * 1024 * 1024


class TraceError(ValueError):
    pass


def _sha(path: Path) -> str:
    return ORACLE.sha256_file(path)


def _load_json(path: Path) -> Any:
    value = ORACLE.load_json(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _context_hash(token_ids: list[int]) -> str:
    return ORACLE.canonical_token_ids_hash(token_ids)


def _finite(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise TraceError("non-finite trace sample")
    return result


def _sample(values: torch.Tensor, coordinates: tuple[int, ...]) -> dict[str, Any]:
    flat = values.detach().to(dtype=torch.float32, device="cpu").reshape(-1)
    if flat.numel() <= max(coordinates):
        raise TraceError(f"trace vector has {flat.numel()} elements, expected coordinate {max(coordinates)}")
    selected = _finite(flat[list(coordinates)].tolist())
    return {
        "coordinates": list(coordinates),
        "elements": int(flat.numel()),
        "values": selected,
        "max_abs": float(flat.abs().max().item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
    }


def _tensor_from_hook_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = _tensor_from_hook_output(value)
            if tensor is not None:
                return tensor
    for name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


class _HookCapture:
    def __init__(self) -> None:
        self.samples: dict[tuple[str, int | None], dict[str, Any]] = {}
        self.handles: list[Any] = []

    def install(self, module: Any, stage: str, layer_index: int | None = None) -> None:
        key = (stage, layer_index)

        def hook(_: Any, __: tuple[Any, ...], output: Any) -> None:
            tensor = _tensor_from_hook_output(output)
            if tensor is None:
                raise TraceError(f"{stage} hook returned no tensor")
            vector = tensor.reshape(-1, tensor.shape[-1])[-1]
            coordinates = LOGIT_COORDINATES if stage == "lm_head" else HIDDEN_COORDINATES
            self.samples[key] = _sample(vector, coordinates)

        self.handles.append(module.register_forward_hook(hook))

    def clear(self) -> None:
        self.samples.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _stage_records(capture: _HookCapture) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for key in [("embedding", None)] + [("decoder_layer", index) for index in range(32)] + [("final_norm", None), ("lm_head", None)]:
        sample = capture.samples.get(key)
        if sample is None:
            raise TraceError(f"missing stage sample: {key[0]}:{key[1]}")
        stage, layer_index = key
        item: dict[str, Any] = {"stage": stage, "sample": sample}
        if layer_index is not None:
            item["layer_index"] = layer_index
        ordered.append(item)
    return ordered


def _load_cases(path: Path) -> list[dict[str, Any]]:
    value = _load_json(path)
    if not isinstance(value, dict) or set(value) != {"cases"} or not isinstance(value["cases"], list):
        raise TraceError("cases must contain only a cases array")
    cases = []
    for case in value["cases"]:
        if not isinstance(case, dict) or set(case) != {"case_id", "prompt_token_ids", "step_count"}:
            raise TraceError("case fields differ")
        token_ids = [ORACLE.integer(token, "prompt token", minimum=0) for token in case["prompt_token_ids"]]
        step_count = ORACLE.integer(case["step_count"], "step count", minimum=1)
        cases.append({"case_id": case["case_id"], "prompt_token_ids": token_ids, "step_count": step_count})
    return cases


def capture_source(model_dir: Path, cases_path: Path, output: Path) -> dict[str, Any]:
    if output.exists() or os.path.lexists(output):
        raise TraceError(f"refusing to overwrite output: {output}")
    if torch.cuda.is_available():
        raise TraceError("source trace is CPU-only, but CUDA is visible")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from transformers import AutoModelForCausalLM

    cases = _load_cases(cases_path)
    source_manifest_path = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2/manifest.json"
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None)
    model.eval()
    capture = _HookCapture()
    capture.install(model.model.embed_tokens, "embedding")
    for index, layer in enumerate(model.model.layers[:32]):
        capture.install(layer, "decoder_layer", index)
    capture.install(model.model.norm, "final_norm")
    capture.install(model.lm_head, "lm_head")
    output.mkdir(parents=True)
    payload_path = output / "payload.jsonl"
    rows = 0
    started = time.monotonic()
    try:
        with payload_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                past = None
                input_ids = list(case["prompt_token_ids"])
                for step in range(case["step_count"]):
                    capture.clear()
                    input_hash = _context_hash(input_ids)
                    tokens = torch.tensor([input_ids], dtype=torch.long, device="cpu")
                    with torch.inference_mode():
                        base = model.model(input_ids=tokens, past_key_values=past, use_cache=True, return_dict=True)
                        hidden = base.last_hidden_state[:, -1, :]
                        logits = model.lm_head(hidden.unsqueeze(1))[:, -1, :]
                    stages = _stage_records(capture)
                    greedy = int(torch.argmax(logits.reshape(-1)).item())
                    row = {"case_id": case["case_id"], "step": step, "context_length": len(input_ids), "context_token_ids_sha256": input_hash, "stages": stages, "greedy_token_id": greedy}
                    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                    rows += 1
                    past = getattr(base, "past_key_values", None)
                    input_ids = [greedy]
                    del tokens, base, hidden, logits
                del past
    finally:
        capture.close()
        del model
    manifest = {"schema_version": SCHEMA, "mode": "source_cpu", "model_dir": str(model_dir.resolve()), "cases_path": str(cases_path.resolve()), "source_manifest_sha256": _sha(source_manifest_path) if source_manifest_path.is_file() else None, "rows": rows, "stage_contract": {"decoder_layers": 32, "hidden_coordinates": list(HIDDEN_COORDINATES), "logit_coordinates": list(LOGIT_COORDINATES)}, "runtime": {"device": "cpu", "dtype": "bfloat16", "model_loads": 1, "elapsed_seconds": time.monotonic() - started}}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "runtime.json", manifest["runtime"])
    sums = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS" and path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
    return manifest


def _trace_rows(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
        raise TraceError("trace manifest schema differs")
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with (root / "payload.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (row["case_id"], row["step"])
            if key in result:
                raise TraceError(f"duplicate trace row: {key}")
            result[key] = row
    return result


def _oracle_context_rows(root: Path, cases_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    cases = {case["case_id"]: case for case in _load_cases(cases_path)}
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    sequences: dict[str, list[int]] = {case_id: [] for case_id in cases}
    for line in (root / "payload.jsonl").read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        case_id = value["case_id"]
        case = cases.get(case_id)
        if case is None:
            raise TraceError(f"oracle payload case is not in cases: {case_id}")
        step = int(value["step"])
        context = case["prompt_token_ids"] + sequences[case_id]
        rows[(case_id, step)] = {"case_id": case_id, "step": step, "context_length": len(context), "context_token_ids_sha256": _context_hash(context), "oracle_row": value}
        sequences[case_id].append(int(value["greedy_token_id"]))
    return rows


def endpoint_trace_from_path(path_payload: Path, output: Path, source_root: Path, cases_path: Path) -> dict[str, Any]:
    """Adapt immutable path payload into an endpoint-only trace for diagnosis."""
    if output.exists() or os.path.lexists(output):
        raise TraceError(f"refusing to overwrite output: {output}")
    try:
        source_rows = _trace_rows(source_root)
    except TraceError:
        source_rows = _oracle_context_rows(source_root, cases_path)
    output.mkdir(parents=True)
    rows = 0
    with path_payload.open(encoding="utf-8") as source, (output / "payload.jsonl").open("w", encoding="utf-8") as target:
        for line in source:
            row = json.loads(line)
            key = (row["case_id"], row["step"])
            source_row = source_rows.get(key)
            if source_row is None:
                raise TraceError(f"path endpoint row lacks source context: {key}")
            hidden = row["hidden_sample"]
            logits = row["logit_sample"]
            stages = [
                {"stage": "final_norm", "sample": {"coordinates": hidden["indices"], "elements": hidden["shape"][0], "values": hidden["values"]}},
                {"stage": "lm_head", "sample": {"coordinates": logits["indices"], "elements": logits["shape"][0], "values": logits["values"]}},
            ]
            adapted = {"case_id": row["case_id"], "step": row["step"], "context_length": source_row["context_length"], "context_token_ids_sha256": source_row["context_token_ids_sha256"], "stages": stages, "greedy_token_id": row["greedy_token_id"]}
            target.write(json.dumps(adapted, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
            rows += 1
    manifest = {"schema_version": SCHEMA, "mode": "path_endpoint_only", "source_trace_root": str(source_root.resolve()), "rows": rows, "stage_contract": {"available": ["final_norm", "lm_head"]}}
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "runtime.json", {"device": "unknown", "rows": rows})
    sums = []
    for path in sorted(output.iterdir()):
        if path.name != "SHA256SUMS" and path.is_file():
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
    return manifest


def _stage_key(stage: dict[str, Any]) -> tuple[str, int | None]:
    return stage["stage"], stage.get("layer_index")


def analyze(source_root: Path, path_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() or os.path.lexists(output):
        raise TraceError(f"refusing to overwrite output: {output}")
    source = _trace_rows(source_root)
    path = _trace_rows(path_root)
    reports: list[dict[str, Any]] = []
    for key in sorted(source):
        left = source[key]
        right = path.get(key)
        if right is None:
            reports.append({"case_id": key[0], "step": key[1], "status": "missing_path_row"})
            continue
        if left["context_token_ids_sha256"] != right.get("context_token_ids_sha256"):
            reports.append({"case_id": key[0], "step": key[1], "status": "context_mismatch"})
            continue
        left_stages = {_stage_key(stage): stage for stage in left.get("stages", [])}
        right_stages = {_stage_key(stage): stage for stage in right.get("stages", [])}
        first: dict[str, Any] | None = None
        missing: list[dict[str, Any]] = []
        for stage_key, source_stage in left_stages.items():
            target_stage = right_stages.get(stage_key)
            if target_stage is None:
                missing.append({"stage": stage_key[0], "layer_index": stage_key[1]})
                continue
            source_sample = source_stage["sample"]
            target_sample = target_stage["sample"]
            source_values = dict(zip(source_sample["coordinates"], source_sample["values"], strict=True))
            target_values = dict(zip(target_sample["coordinates"], target_sample["values"], strict=True))
            coordinates = sorted(set(source_values) & set(target_values))
            if not coordinates:
                first = {"stage": stage_key[0], "layer_index": stage_key[1], "reason": "sample_coordinate_intersection_empty"}
                break
            max_abs = max(abs(float(source_values[index]) - float(target_values[index])) for index in coordinates)
            if first is None and max_abs > 0.0:
                first = {"stage": stage_key[0], "layer_index": stage_key[1], "reason": "sample_value_mismatch", "max_abs": max_abs, "coordinate_count": len(coordinates)}
        if missing:
            diagnosis = "inconclusive_missing_intermediate_aq4_trace"
        elif first is None:
            diagnosis = "no_bounded_stage_mismatch"
        elif first["stage"] == "embedding":
            diagnosis = "embedding_or_input_mapping"
        elif first["stage"] == "decoder_layer":
            diagnosis = f"decoder_layer_{first['layer_index']}"
        elif first["stage"] == "final_norm":
            diagnosis = "final_norm"
        else:
            diagnosis = "lm_head"
        reports.append({"case_id": key[0], "step": key[1], "status": "analyzed", "diagnosis": diagnosis, "first_mismatch": first, "missing_stages": missing, "greedy_exact": left.get("greedy_token_id") == right.get("greedy_token_id")})
    result = {"schema_version": ANALYSIS_SCHEMA, "source_root": str(source_root.resolve()), "path_root": str(path_root.resolve()), "stage_contract": {"embedding": True, "decoder_layers": 32, "final_norm": True, "lm_head": True}, "reports": reports, "overall_diagnosis": "inconclusive_missing_intermediate_aq4_trace" if any(item.get("diagnosis") == "inconclusive_missing_intermediate_aq4_trace" for item in reports) else "analyzed"}
    output.mkdir(parents=True)
    _write_json(output / "analysis.json", result)
    (output / "SHA256SUMS").write_text(f"{hashlib.sha256((output / 'analysis.json').read_bytes()).hexdigest()}  analysis.json\n", encoding="ascii")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("--model-dir", type=Path, required=True)
    source.add_argument("--cases", type=Path, required=True)
    source.add_argument("--output", type=Path, required=True)
    endpoint = sub.add_parser("path-endpoint")
    endpoint.add_argument("--payload", type=Path, required=True)
    endpoint.add_argument("--source-trace", type=Path, required=True)
    endpoint.add_argument("--cases", type=Path, required=True)
    endpoint.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("analyze")
    compare.add_argument("--source-trace", type=Path, required=True)
    compare.add_argument("--path-trace", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "source":
            result = capture_source(args.model_dir, args.cases, args.output)
        elif args.command == "path-endpoint":
            result = endpoint_trace_from_path(args.payload, args.output, args.source_trace, args.cases)
        else:
            result = analyze(args.source_trace, args.path_trace, args.output)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (OSError, RuntimeError, TraceError, ORACLE.OracleError, ValueError) as error:
        print(f"Qwen3.5 AQ4 differential trace failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
