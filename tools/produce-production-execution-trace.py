#!/usr/bin/env python3
"""Build a strict execution trace from a bounded executor-record sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_BYTES = 4 * 1024 * 1024
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
FORBIDDEN = re.compile(r"^(?:prompt_text|prompt_or_token_content|prompt_token_ids|response_text|response_body|token_ids|generated_text|request_id|api_key|authorization|account_id|client_address|raw_headers)$", re.I)


class ProducerError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o002:
        raise ProducerError(f"identity path is unavailable: {path}")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(value: Any, *, manifest_path: Path, label: str, root: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ProducerError(f"{label} path is missing")
    raw = Path(value)
    if ".." in raw.parts:
        raise ProducerError(f"{label} path escapes its manifest root")
    base = root if root is not None else manifest_path.parent
    lexical = raw if raw.is_absolute() else base / raw
    cursor = lexical
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ProducerError(f"{label} path contains a symlink component")
        cursor = cursor.parent
    candidate = lexical.resolve(strict=False)
    return candidate


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES or path.stat().st_mode & 0o002:
        raise ProducerError(f"{label} must be a bounded regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProducerError(f"cannot parse {label}: {error}") from error


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ProducerError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ProducerError(f"non-finite JSON value: {value}")


def reject_private(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if FORBIDDEN.fullmatch(key):
                raise ProducerError(f"forbidden executor fact field: {path}.{key}")
            reject_private(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_private(child, f"{path}[{index}]")


def manifest_identity(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    public = manifest.get("public", {}); fmt = manifest.get("format", {}); worker = manifest.get("worker", {}); product = manifest.get("product", {}); package = product.get("package", {})
    product_root = resolve_manifest_path(product.get("root", "."), manifest_path=path, label="product root")
    package_path = resolve_manifest_path(package.get("manifest_path", ""), manifest_path=path, root=product_root, label="package manifest")
    receipt = resolve_manifest_path(manifest.get("promotion", {}).get("receipt", ""), manifest_path=path, label="promotion receipt")
    worker_path = resolve_manifest_path(worker.get("binary", ""), manifest_path=path, label="worker binary")
    package_hash = digest(package_path); receipt_hash = digest(receipt); worker_hash = digest(worker_path)
    package_declared = package.get("manifest_sha256")
    if package_declared is not None and package_declared != package_hash:
        raise ProducerError("manifest package digest differs")
    worker_declared = worker.get("binary_sha256")
    if worker_declared is not None and worker_declared != worker_hash:
        raise ProducerError("manifest worker digest differs")
    receipt_declared = manifest.get("promotion", {}).get("receipt_sha256")
    if receipt_declared is not None and receipt_declared != receipt_hash:
        raise ProducerError("manifest receipt digest differs")
    product_identity = {"id": public.get("id"), "revision": public.get("revision"), "root": str(product_root), "package_manifest_sha256": package_hash}
    artifact = product.get("artifact") if isinstance(product.get("artifact"), dict) else {}
    if artifact.get("manifest_sha256") is not None:
        artifact_manifest = resolve_manifest_path(artifact.get("manifest_path", ""), manifest_path=path, root=product_root, label="artifact manifest")
        if digest(artifact_manifest) != artifact.get("manifest_sha256"):
            raise ProducerError("manifest artifact digest differs")
    if artifact.get("content_sha256") is not None:
        artifact_doc = load_json(artifact_manifest, "artifact manifest") if artifact.get("manifest_sha256") is not None else {}
        integrity = artifact_doc.get("integrity", {}) if isinstance(artifact_doc, dict) else {}
        content_path = artifact.get("content_path") or (integrity.get("content_path") if isinstance(integrity, dict) else None) or artifact_doc.get("content_path")
        if not content_path:
            raise ProducerError("artifact content digest has no declared content path")
        artifact_content = resolve_manifest_path(content_path, manifest_path=path, root=product_root, label="artifact content")
        if digest(artifact_content) != artifact.get("content_sha256"):
            raise ProducerError("artifact content digest differs")
    return {
        "model": {"id": public.get("id"), "revision": public.get("revision"), "format_id": fmt.get("format_id"), "implementation_id": fmt.get("implementation_id")},
        "served_model_manifest_sha256": digest(path),
        "worker": {"protocol": worker.get("protocol"), "binary_sha256": worker_hash},
        "product": {"id": public.get("id"), "revision": public.get("revision"), "identity_sha256": hashlib.sha256(canonical(product_identity)).hexdigest(), "promotion_receipt_sha256": receipt_hash},
        "artifact": {"manifest_sha256": artifact.get("manifest_sha256"), "content_sha256": artifact.get("content_sha256")},
        "package": {"manifest_sha256": package_hash},
    }


def phase_transform(phases: Any, executor: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(phases, list) or not phases:
        raise ProducerError("executor record must provide nonempty phases")
    result = []
    for phase in phases:
        # The record and trace use the same normative shape; no producer-only
        # labels are allowed to leak into the trace.
        fields = {"phase_id", "kind", "executor_id", "executor_version", "prefill_mode", "chunk_width_tokens", "actual_token_batch_width", "actual_request_batch_width", "request_count", "input_token_count", "output_token_count", "cached_prefix_token_count", "context_tokens_before", "context_tokens_after", "wall_time_ms"}
        if not isinstance(phase, dict) or set(phase) != fields:
            raise ProducerError("executor phase fields differ from normative schema")
        if phase["executor_id"] != executor["id"] or phase["executor_version"] != executor["version"]:
            raise ProducerError("phase executor identity differs")
        result.append(phase)
    return result


def normalize_scope(value: Any, server: Any) -> str:
    if value in {"worker", "direct_worker", "full_model"}:
        if isinstance(server, dict):
            raise ProducerError("direct worker cannot claim a server boundary")
        return "full_model"
    if value == "production_server":
        if not isinstance(server, dict) or server.get("ready_observed") is not True or server.get("release_observed") is not True:
            raise ProducerError("production_server requires an observed ready/release boundary")
        return value
    if value == "component":
        return value
    raise ProducerError("executor record scope is invalid")


def build_trace(args: argparse.Namespace, manifest: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    reject_private(facts)
    required = {"schema_version", "trace_id", "status", "scope", "graph", "executor", "request_summary", "phases", "operator_resolutions", "fallback", "memory", "state_commit", "server", "failure"}
    if not isinstance(facts, dict) or set(facts) != required or facts["schema_version"] != "ullm.production_executor_record.v1":
        raise ProducerError("executor record root fields differ")
    trace_id = facts["trace_id"] if isinstance(facts["trace_id"], str) else args.trace_id
    if not isinstance(trace_id, str) or ID_RE.fullmatch(trace_id) is None:
        raise ProducerError("trace_id is invalid")
    scope = normalize_scope(facts["scope"], facts["server"])
    identity = manifest_identity(manifest, args.manifest)
    graph_input = facts["graph"]
    if not isinstance(graph_input, dict): raise ProducerError("graph is required")
    graph = {}
    for name, schema_id in (("model_graph", "ullm.model_graph.v0.1"), ("state_schema", "ullm.state_schema.v0.1")):
        item = graph_input.get(name)
        if not isinstance(item, dict) or set(item) - {"schema_id", "schema_version", "source", "canonical"} != set() or not isinstance(item.get("canonical"), (dict, list)):
            raise ProducerError(f"graph.{name}.canonical is required")
        if item.get("source", "adapter_derived") not in {"serialized", "adapter_derived"}:
            raise ProducerError(f"graph.{name}.source is invalid")
        graph[name] = {"schema_id": item.get("schema_id", schema_id), "schema_version": item.get("schema_version", "0.1"), "sha256": hashlib.sha256(canonical(item["canonical"])).hexdigest(), "source": item.get("source", "adapter_derived")}
    compatibility_inputs = graph_input.get("compatibility_inputs", {"backend": facts["executor"].get("backend"), "format_id": identity["model"]["format_id"], "layout": "row_major_grouped"})
    graph["compatibility_key_sha256"] = hashlib.sha256(canonical({"model_graph": graph["model_graph"]["sha256"], "state_schema": graph["state_schema"]["sha256"], "compatibility_inputs": compatibility_inputs})).hexdigest()
    executor = facts["executor"]
    if not isinstance(executor, dict) or set(executor) != {"id", "version", "mode", "backend", "device"}: raise ProducerError("executor fields differ")
    request = facts["request_summary"]
    if not isinstance(request, dict) or set(request) != {"fixture_id", "request_count", "prompt_token_count", "cached_prefix_token_count", "generated_token_count", "context_tokens_at_decode_start", "prompt_or_token_content_recorded"}: raise ProducerError("request_summary fields differ")
    phases = phase_transform(facts["phases"], executor)
    operators = facts["operator_resolutions"]
    fallback = facts["fallback"]; memory = facts["memory"]; state = facts["state_commit"]
    if not isinstance(operators, list) or not isinstance(fallback, dict) or not isinstance(memory, dict) or not isinstance(state, dict): raise ProducerError("executor facts contain invalid execution fields")
    if scope != "production_server": server = None
    else: server = facts["server"]
    trace = {"schema_version": "ullm.production_execution_trace.v1", "trace_id": trace_id, "status": facts["status"], "scope": scope, "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"), "producer": {"id": "ullm-aq4-production-trace-producer", "version": "0.2.0", "binary_sha256": identity["worker"]["binary_sha256"], "verified": True}, "identity": identity, "graph": graph, "executor": executor, "request_summary": request, "phases": phases, "operator_resolutions": operators, "fallback": fallback, "memory": memory, "state_commit": state, "aggregation": {"is_aggregated": False, "source_trace_sha256s": [], "component_trace_count": 0, "full_model_trace_count": 0, "coverage": scope}, "server": server, "verification": {"producer_verified": True, "independent_validation": {"status": "not_run", "validator_id": None, "validator_version": None, "report_sha256": None, "failure_codes": []}}, "failure": facts["failure"]}
    raw = canonical(trace)
    if len(raw) > MAX_BYTES: raise ProducerError("trace exceeds 4 MiB")
    return trace


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink(): raise ProducerError(f"refusing to overwrite {path}")
    if path.parent.stat().st_mode & 0o002: raise ProducerError("trace destination directory is world-writable")
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--executor-record", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--binding-output", type=Path, required=True); parser.add_argument("--trace-id")
    args = parser.parse_args(argv)
    try:
        manifest = load_json(args.manifest, "manifest"); facts = load_json(args.executor_record, "executor record")
        trace = build_trace(args, manifest, facts)
        raw = (json.dumps(trace, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        atomic_write(args.output, raw)
        binding = {"schema_version": "ullm.production_executor_trace_binding.v1", "trace_id": trace["trace_id"], "trace_sha256": hashlib.sha256(raw).hexdigest(), "executor_record_sha256": digest(args.executor_record)}
        atomic_write(args.binding_output, (json.dumps(binding, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        print(json.dumps({"status": "ok", "trace_id": trace["trace_id"], "trace_sha256": binding["trace_sha256"]})); return 0
    except (ProducerError, OSError, ValueError) as error:
        print(f"production trace failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
