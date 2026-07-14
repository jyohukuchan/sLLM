#!/usr/bin/env python3
"""Fail-closed independent validator for ``ullm.production_execution_trace.v1``.

The producer is intentionally treated as untrusted input.  This validator
reconstructs the trace from the bounded executor record, manifest and binding
sidecar and only then permits an independently-finalized report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

MAX_BYTES = 4 * 1024 * 1024
MAX_DEPTH = 24
MAX_NODES = 32_768
MAX_ARRAY = {"phases": 4096, "operator_resolutions": 16_384, "events": 4096, "source_trace_sha256s": 4096}
SAFE_INT = 9_007_199_254_740_991
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
FORBIDDEN = re.compile(r"^(?:prompt_text|prompt_or_token_content|prompt_token_ids|response_text|response_body|token_ids|generated_text|request_id|api_key|authorization|account_id|client_address|raw_headers)$", re.I)


class ValidationError(ValueError):
    pass


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValidationError("duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON number: {value}")


def load(path: Path, label: str) -> tuple[Any, bytes]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValidationError(f"{label} must be a bounded regular non-symlink file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{label} is not strict UTF-8 JSON") from error
    return value, raw


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"identity path is unavailable: {path}")
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(value: Any, *, manifest_path: Path, label: str, root: Path | None = None) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{label} path is missing")
    raw = Path(value)
    if ".." in raw.parts:
        raise ValidationError(f"{label} path escapes its manifest root")
    base = root if root is not None else manifest_path.parent
    lexical = raw if raw.is_absolute() else base / raw
    cursor = lexical
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise ValidationError(f"{label} path contains a symlink component")
        cursor = cursor.parent
    candidate = lexical
    candidate = candidate.resolve(strict=False)
    return candidate


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def check_tree(value: Any, path: str = "root", depth: int = 0, nodes: list[int] | None = None) -> None:
    if nodes is None:
        nodes = [0]
    if depth > MAX_DEPTH:
        raise ValidationError(f"JSON nesting exceeds {MAX_DEPTH} at {path}")
    nodes[0] += 1
    if nodes[0] > MAX_NODES:
        raise ValidationError("JSON node bound exceeded")
    if isinstance(value, dict):
        for key, child in value.items():
            if len(key.encode("utf-8")) > 16_384 or any(ord(ch) < 0x20 for ch in key):
                raise ValidationError(f"invalid object key at {path}")
            check_tree(child, f"{path}.{key}", depth + 1, nodes)
    elif isinstance(value, list):
        check_array_bound(path, value)
        for index, child in enumerate(value):
            check_tree(child, f"{path}[{index}]", depth + 1, nodes)
    elif isinstance(value, str):
        if len(value.encode("utf-8")) > 16_384 or any(ord(ch) < 0x20 for ch in value):
            raise ValidationError(f"invalid string at {path}")
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, int):
        if value < 0 or value > SAFE_INT:
            raise ValidationError(f"negative or unsafe integer at {path}")
    elif isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            raise ValidationError(f"non-finite or negative number at {path}")
    else:
        raise ValidationError(f"unsupported JSON value at {path}")


def check_array_bound(path: str, value: list[Any]) -> None:
    for field, limit in MAX_ARRAY.items():
        if path == f"root.{field}" or path.endswith(f".{field}"):
            if len(value) > limit:
                raise ValidationError(f"{path} exceeds {limit} entries")


def reject_forbidden(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if FORBIDDEN.fullmatch(key):
                raise ValidationError(f"forbidden privacy field at {path}.{key}")
            reject_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden(child, f"{path}[{index}]")


def exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValidationError(f"{label} fields differ")
    return value


def string(value: Any, label: str, *, identifier: bool = False, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or (identifier and ID_RE.fullmatch(value) is None):
        raise ValidationError(f"{label} is invalid")


def digest(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} is not a lowercase SHA-256")


def nonnegative(value: Any, label: str, *, integer: bool = True, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if integer:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > SAFE_INT:
            raise ValidationError(f"{label} must be a safe nonnegative integer")
    elif not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValidationError(f"{label} must be a finite nonnegative number")


def validate_graph(graph: Any, facts_graph: Any) -> None:
    exact(graph, {"model_graph", "state_schema", "compatibility_key_sha256"}, "graph")
    exact(graph["model_graph"], {"schema_id", "schema_version", "sha256", "source"}, "graph.model_graph")
    exact(graph["state_schema"], {"schema_id", "schema_version", "sha256", "source"}, "graph.state_schema")
    for name in ("model_graph", "state_schema"):
        item = graph[name]
        string(item["schema_id"], f"graph.{name}.schema_id", identifier=True)
        string(item["schema_version"], f"graph.{name}.schema_version", identifier=True)
        digest(item["sha256"], f"graph.{name}.sha256")
        if item["source"] not in {"serialized", "adapter_derived"}:
            raise ValidationError(f"graph.{name}.source is invalid")
        if not isinstance(facts_graph, dict) or not isinstance(facts_graph.get(name), dict):
            raise ValidationError(f"executor graph.{name} canonical input is absent")
        canonical_value = facts_graph[name].get("canonical")
        if set(facts_graph[name]) != {"schema_id", "schema_version", "source", "canonical"}:
            raise ValidationError(f"executor graph.{name} fields differ")
        if facts_graph[name]["schema_id"] != item["schema_id"] or facts_graph[name]["schema_version"] != item["schema_version"] or facts_graph[name]["source"] != item["source"]:
            raise ValidationError(f"graph.{name} schema/source differs from executor record")
        if canonical_value is None or sha(canonical(canonical_value)) != item["sha256"]:
            raise ValidationError(f"graph.{name}.sha256 does not match canonical input")
    digest(graph["compatibility_key_sha256"], "graph.compatibility_key_sha256")
    if not isinstance(facts_graph, dict) or set(facts_graph) != {"model_graph", "state_schema", "compatibility_inputs"} or not isinstance(facts_graph.get("compatibility_inputs"), dict):
        raise ValidationError("executor graph compatibility inputs are absent or unknown")
    backend = facts_graph["compatibility_inputs"]
    expected = sha(canonical({"model_graph": graph["model_graph"]["sha256"], "state_schema": graph["state_schema"]["sha256"], "compatibility_inputs": backend}))
    if graph["compatibility_key_sha256"] != expected:
        raise ValidationError("graph compatibility key differs")


def validate_executor(executor: Any) -> None:
    exact(executor, {"id", "version", "mode", "backend", "device"}, "executor")
    for name in ("id", "version", "mode", "backend"):
        string(executor[name], f"executor.{name}", identifier=True)
    exact(executor["device"], {"runtime_device_index", "name", "architecture"}, "executor.device")
    nonnegative(executor["device"]["runtime_device_index"], "executor.device.runtime_device_index")
    string(executor["device"]["name"], "executor.device.name")
    string(executor["device"]["architecture"], "executor.device.architecture", identifier=True)


def validate_request(request: Any) -> None:
    exact(request, {"fixture_id", "request_count", "prompt_token_count", "cached_prefix_token_count", "generated_token_count", "context_tokens_at_decode_start", "prompt_or_token_content_recorded"}, "request_summary")
    string(request["fixture_id"], "request_summary.fixture_id", identifier=True, nullable=True)
    for name in ("request_count", "prompt_token_count", "cached_prefix_token_count", "generated_token_count", "context_tokens_at_decode_start"):
        nonnegative(request[name], f"request_summary.{name}")
    if request["request_count"] < 1 or request["prompt_or_token_content_recorded"] is not False:
        raise ValidationError("request summary is invalid or records private content")


def validate_phases(phases: Any, request: dict[str, Any], executor: dict[str, Any]) -> None:
    if not isinstance(phases, list) or not phases:
        raise ValidationError("phases must be nonempty")
    seen: set[str] = set()
    total_in = total_out = 0
    previous_after = 0
    for phase in phases:
        exact(phase, {"phase_id", "kind", "executor_id", "executor_version", "prefill_mode", "chunk_width_tokens", "actual_token_batch_width", "actual_request_batch_width", "request_count", "input_token_count", "output_token_count", "cached_prefix_token_count", "context_tokens_before", "context_tokens_after", "wall_time_ms"}, "phase")
        string(phase["phase_id"], "phase.phase_id", identifier=True)
        if phase["phase_id"] in seen:
            raise ValidationError("duplicate phase_id")
        seen.add(phase["phase_id"])
        if phase["kind"] not in {"cold_prefill", "cached_prefix_prefill", "decode"}:
            raise ValidationError("phase kind is invalid")
        expected_mode = {"cold_prefill": "cold", "cached_prefix_prefill": "cached_prefix", "decode": None}[phase["kind"]]
        if phase["prefill_mode"] != expected_mode or phase["executor_id"] != executor["id"] or phase["executor_version"] != executor["version"]:
            raise ValidationError("phase executor or prefill mode differs")
        for name in ("chunk_width_tokens", "actual_token_batch_width", "actual_request_batch_width", "request_count", "input_token_count", "output_token_count", "cached_prefix_token_count", "context_tokens_before", "context_tokens_after"):
            nonnegative(phase[name], f"phase.{name}")
        nonnegative(phase["wall_time_ms"], "phase.wall_time_ms", integer=False)
        if min(phase["chunk_width_tokens"], phase["actual_token_batch_width"], phase["actual_request_batch_width"]) <= 0:
            raise ValidationError("zero-width phase")
        if phase["context_tokens_before"] != previous_after:
            raise ValidationError("phase context transition does not chain")
        if phase["context_tokens_after"] < phase["context_tokens_before"] + phase["output_token_count"]:
            raise ValidationError("phase context transition is invalid")
        previous_after = phase["context_tokens_after"]
        total_in += phase["input_token_count"]
        total_out += phase["output_token_count"]
    if total_out != request["generated_token_count"] or previous_after < request["context_tokens_at_decode_start"]:
        raise ValidationError("phase/request counters do not reconcile")


def validate_operator(operator: Any, phase_kinds: set[str]) -> None:
    exact(operator, {"phase_kind", "operator_instance_id", "op_kind", "implementation_id", "implementation_version", "resolution_status", "backend", "device", "formats", "shape_bucket", "selection_reason", "architecture_constraint", "workspace", "invocation_count"}, "operator_resolution")
    for name in ("phase_kind", "operator_instance_id", "op_kind", "implementation_id", "implementation_version", "backend"):
        string(operator[name], f"operator.{name}", identifier=True)
    if operator["phase_kind"] not in phase_kinds or operator["resolution_status"] not in {"selected", "fallback", "unsupported", "fail_closed"}:
        raise ValidationError("operator resolution status or phase is invalid")
    string(operator["device"], "operator.device")
    exact(operator["formats"], {"weight", "activation", "state", "layout"}, "operator.formats")
    for name in operator["formats"]:
        string(operator["formats"][name], f"operator.formats.{name}", identifier=True, nullable=name == "state")
    exact(operator["shape_bucket"], {"id", "dimensions"}, "operator.shape_bucket")
    string(operator["shape_bucket"]["id"], "operator.shape_bucket.id", identifier=True)
    if not isinstance(operator["shape_bucket"]["dimensions"], list) or len(operator["shape_bucket"]["dimensions"]) > 16:
        raise ValidationError("operator shape dimensions are invalid")
    for dimension in operator["shape_bucket"]["dimensions"]:
        exact(dimension, {"name", "value"}, "operator.shape_bucket.dimension")
        string(dimension["name"], "operator.shape_bucket.dimension.name", identifier=True)
        nonnegative(dimension["value"], "operator.shape_bucket.dimension.value")
    exact(operator["selection_reason"], {"kind", "candidate_count", "score", "priority", "matched_constraints"}, "operator.selection_reason")
    if operator["selection_reason"]["kind"] not in {"exact_match", "highest_specificity_priority", "generic_fallback", "workspace_limited_fallback", "unsupported", "fail_closed"}:
        raise ValidationError("operator selection reason is invalid")
    for name in ("candidate_count", "score", "priority"):
        nonnegative(operator["selection_reason"][name], f"operator.selection_reason.{name}")
    if not isinstance(operator["selection_reason"]["matched_constraints"], list) or len(operator["selection_reason"]["matched_constraints"]) > 8:
        raise ValidationError("operator matched constraints are invalid")
    for constraint in operator["selection_reason"]["matched_constraints"]:
        string(constraint, "operator.selection_reason.matched_constraints", identifier=True)
    constraint = operator["architecture_constraint"]
    if constraint is not None:
        exact(constraint, {"model_arch", "gpu_arch", "gpu_name"}, "operator.architecture_constraint")
        for name in constraint:
            string(constraint[name], f"operator.architecture_constraint.{name}", nullable=True)
    exact(operator["workspace"], {"planned_bytes", "observed_peak_bytes"}, "operator.workspace")
    nonnegative(operator["workspace"]["planned_bytes"], "operator.workspace.planned_bytes")
    nonnegative(operator["workspace"]["observed_peak_bytes"], "operator.workspace.observed_peak_bytes", nullable=True)
    nonnegative(operator["invocation_count"], "operator.invocation_count")
    if operator["invocation_count"] == 0:
        raise ValidationError("operator invocation count must be positive")


def validate_trace(trace: dict[str, Any], manifest: dict[str, Any], manifest_raw: bytes, manifest_path: Path, facts: dict[str, Any], binding: dict[str, Any], trace_raw: bytes, facts_raw: bytes) -> dict[str, Any]:
    exact(trace, {"schema_version", "trace_id", "status", "scope", "created_at", "producer", "identity", "graph", "executor", "request_summary", "phases", "operator_resolutions", "fallback", "memory", "state_commit", "aggregation", "server", "verification", "failure"}, "trace")
    if trace["schema_version"] != "ullm.production_execution_trace.v1" or trace["status"] not in {"ok", "unsupported", "oom", "failed", "skipped"} or trace["scope"] not in {"component", "full_model", "production_server"}:
        raise ValidationError("trace schema/status/scope is invalid")
    string(trace["trace_id"], "trace_id", identifier=True)
    if TIMESTAMP_RE.fullmatch(trace["created_at"]) is None:
        raise ValidationError("created_at must be RFC3339 UTC")
    exact(trace["producer"], {"id", "version", "binary_sha256", "verified"}, "producer")
    string(trace["producer"]["id"], "producer.id", identifier=True); string(trace["producer"]["version"], "producer.version", identifier=True); digest(trace["producer"]["binary_sha256"], "producer.binary_sha256")
    if not isinstance(trace["producer"]["verified"], bool): raise ValidationError("producer.verified is invalid")
    identity = exact(trace["identity"], {"model", "served_model_manifest_sha256", "worker", "product", "artifact", "package"}, "identity")
    exact(identity["model"], {"id", "revision", "format_id", "implementation_id"}, "identity.model")
    exact(identity["worker"], {"protocol", "binary_sha256"}, "identity.worker")
    exact(identity["product"], {"id", "revision", "identity_sha256", "promotion_receipt_sha256"}, "identity.product")
    exact(identity["artifact"], {"manifest_sha256", "content_sha256"}, "identity.artifact")
    exact(identity["package"], {"manifest_sha256"}, "identity.package")
    for name in ("model", "worker", "product"):
        for key, value in identity[name].items():
            if key.endswith("sha256") or key == "identity_sha256": digest(value, f"identity.{name}.{key}", nullable=trace["scope"] == "component")
            elif name != "product" or key in {"id", "revision"}: string(value, f"identity.{name}.{key}", identifier=True)
    for key in ("manifest_sha256", "content_sha256"): digest(identity["artifact"][key], f"identity.artifact.{key}", nullable=True)
    digest(identity["served_model_manifest_sha256"], "identity.served_model_manifest_sha256", nullable=trace["scope"] == "component")
    if identity["served_model_manifest_sha256"] is not None and sha(manifest_raw) != identity["served_model_manifest_sha256"]: raise ValidationError("active manifest hash differs")
    public, fmt, worker = manifest.get("public", {}), manifest.get("format", {}), manifest.get("worker", {})
    if identity["model"]["id"] != public.get("id") or identity["model"]["revision"] != public.get("revision") or identity["model"]["format_id"] != fmt.get("format_id") or identity["model"]["implementation_id"] != fmt.get("implementation_id"): raise ValidationError("model identity differs")
    binary = resolve_manifest_path(worker.get("binary", ""), manifest_path=manifest_path, label="worker binary")
    observed_worker_sha = file_sha(binary)
    if worker.get("binary_sha256") is not None and worker.get("binary_sha256") != observed_worker_sha: raise ValidationError("manifest declared worker digest differs")
    if identity["worker"]["binary_sha256"] is not None:
        if observed_worker_sha != identity["worker"]["binary_sha256"] or identity["worker"]["binary_sha256"] != trace["producer"]["binary_sha256"]: raise ValidationError("worker binary binding differs")
    if worker.get("protocol") != identity["worker"]["protocol"]: raise ValidationError("worker protocol differs")
    product = manifest.get("product", {}) if isinstance(manifest.get("product"), dict) else {}
    package = product.get("package", {}) if isinstance(product.get("package"), dict) else {}
    # The manifest itself is supplied separately to validate_trace; relative
    # paths resolve from the current manifest directory in the caller.
    product_root = resolve_manifest_path(product.get("root", "."), manifest_path=manifest_path, label="product root")
    package_path = resolve_manifest_path(package.get("manifest_path", ""), manifest_path=manifest_path, root=product_root, label="package manifest")
    if file_sha(package_path) != identity["package"]["manifest_sha256"]: raise ValidationError("package manifest binding differs")
    receipt_path = resolve_manifest_path(manifest.get("promotion", {}).get("receipt", ""), manifest_path=manifest_path, label="promotion receipt")
    observed_receipt_sha = file_sha(receipt_path)
    if manifest.get("promotion", {}).get("receipt_sha256") is not None and manifest.get("promotion", {}).get("receipt_sha256") != observed_receipt_sha: raise ValidationError("manifest declared receipt digest differs")
    if observed_receipt_sha != identity["product"]["promotion_receipt_sha256"]: raise ValidationError("promotion receipt binding differs")
    if package.get("manifest_sha256") is not None and package.get("manifest_sha256") != identity["package"]["manifest_sha256"]: raise ValidationError("manifest declared package digest differs")
    expected_product = {"id": public.get("id"), "revision": public.get("revision"), "root": str(product_root), "package_manifest_sha256": identity["package"]["manifest_sha256"]}
    if sha(canonical(expected_product)) != identity["product"]["identity_sha256"]: raise ValidationError("product identity digest differs")
    manifest_artifact = product.get("artifact") if isinstance(product.get("artifact"), dict) else {}
    for name in ("manifest_sha256", "content_sha256"):
        if identity["artifact"][name] != manifest_artifact.get(name): raise ValidationError(f"artifact identity differs at {name}")
    validate_graph(trace["graph"], facts.get("graph"))
    validate_executor(trace["executor"]); validate_request(trace["request_summary"]); validate_phases(trace["phases"], trace["request_summary"], trace["executor"])
    phase_kinds = {item["kind"] for item in trace["phases"]}
    if not isinstance(trace["operator_resolutions"], list) or not trace["operator_resolutions"]: raise ValidationError("operator_resolutions must be nonempty")
    for operator in trace["operator_resolutions"]: validate_operator(operator, phase_kinds)
    exact(trace["fallback"], {"fallback_count", "unexpected_fallback_count", "unsupported_count", "fail_closed_count", "events"}, "fallback")
    counts = {"fallback": 0, "unexpected": 0, "unsupported": 0, "fail_closed": 0}
    for event in trace["fallback"]["events"]:
        exact(event, {"phase_kind", "op_kind", "from_implementation_id", "to_implementation_id", "reason_code", "classification"}, "fallback.event")
        if event["classification"] not in {"expected", "unexpected", "unsupported", "fail_closed"}: raise ValidationError("fallback classification invalid")
        for name in ("phase_kind", "op_kind", "from_implementation_id", "to_implementation_id", "reason_code"): string(event[name], f"fallback.{name}", identifier=True)
        counts["fallback"] += 1; counts[event["classification"]] += 1
    for name in ("fallback_count", "unexpected_fallback_count", "unsupported_count", "fail_closed_count"): nonnegative(trace["fallback"][name], f"fallback.{name}")
    if trace["fallback"]["fallback_count"] != counts["fallback"] or trace["fallback"]["unexpected_fallback_count"] != counts["unexpected"] or trace["fallback"]["unsupported_count"] != counts["unsupported"] or trace["fallback"]["fail_closed_count"] != counts["fail_closed"]: raise ValidationError("fallback counts do not reconcile")
    for operator in trace["operator_resolutions"]:
        if operator["resolution_status"] != "selected" and not any(event["phase_kind"] == operator["phase_kind"] and event["op_kind"] == operator["op_kind"] and event["to_implementation_id"] == operator["implementation_id"] for event in trace["fallback"]["events"]): raise ValidationError("non-selected operator has no fallback event")
    nonselected = {(item["phase_kind"], item["op_kind"], item["implementation_id"]) for item in trace["operator_resolutions"] if item["resolution_status"] != "selected"}
    event_targets = {(item["phase_kind"], item["op_kind"], item["to_implementation_id"]) for item in trace["fallback"]["events"]}
    if nonselected != event_targets:
        raise ValidationError("fallback events do not reconstruct non-selected operators exactly")
    if any(item["resolution_status"] in {"unsupported", "fail_closed"} for item in trace["operator_resolutions"]) and trace["status"] == "ok":
        raise ValidationError("unsupported/fail-closed operator cannot be reported as ok")
    memory = exact(trace["memory"], {"vram_capacity_bytes", "resident_bytes", "persistent_state_bytes", "planned_temporary_bytes", "planned_total_bytes", "planned_headroom_bytes", "observed_peak_bytes", "observed_headroom_bytes", "observer", "oom"}, "memory")
    for name in ("vram_capacity_bytes", "resident_bytes", "persistent_state_bytes", "planned_temporary_bytes", "planned_total_bytes", "planned_headroom_bytes"): nonnegative(memory[name], f"memory.{name}")
    for name in ("observed_peak_bytes", "observed_headroom_bytes"): nonnegative(memory[name], f"memory.{name}", nullable=True)
    exact(memory["observer"], {"kind", "sample_count", "complete"}, "memory.observer"); string(memory["observer"]["kind"], "memory.observer.kind", identifier=True); nonnegative(memory["observer"]["sample_count"], "memory.observer.sample_count")
    if not isinstance(memory["observer"]["complete"], bool): raise ValidationError("memory.observer.complete is invalid")
    if memory["planned_total_bytes"] != memory["resident_bytes"] + memory["persistent_state_bytes"] + memory["planned_temporary_bytes"] or memory["planned_headroom_bytes"] != memory["vram_capacity_bytes"] - memory["planned_total_bytes"]: raise ValidationError("planned memory arithmetic differs")
    if memory["observed_peak_bytes"] is not None and memory["observed_headroom_bytes"] != memory["vram_capacity_bytes"] - memory["observed_peak_bytes"]: raise ValidationError("observed memory arithmetic differs")
    if memory["oom"] is not None:
        exact(memory["oom"], {"stage", "reason_code", "planned_bytes", "observed_peak_bytes"}, "memory.oom")
        string(memory["oom"]["stage"], "memory.oom.stage", identifier=True); string(memory["oom"]["reason_code"], "memory.oom.reason_code", identifier=True); nonnegative(memory["oom"]["planned_bytes"], "memory.oom.planned_bytes"); nonnegative(memory["oom"]["observed_peak_bytes"], "memory.oom.observed_peak_bytes", nullable=True)
    elif trace["status"] == "oom": raise ValidationError("OOM status must retain memory.oom object")
    if trace["status"] == "ok" and (not memory["observer"]["complete"] or memory["observed_peak_bytes"] is None or memory["observed_headroom_bytes"] is None): raise ValidationError("successful trace has incomplete memory observation")
    state = exact(trace["state_commit"], {"prepared_batch_count", "committed_batch_count", "discarded_batch_count", "stale_nonce_count", "cancelled_batch_count", "error_batch_count", "reset"}, "state_commit")
    for name in state:
        if name != "reset": nonnegative(state[name], f"state_commit.{name}")
    exact(state["reset"], {"required", "attempted", "complete", "failed"}, "state_commit.reset")
    for name in state["reset"]:
        if not isinstance(state["reset"][name], bool): raise ValidationError("state reset flag is invalid")
    if state["prepared_batch_count"] != state["committed_batch_count"] + state["discarded_batch_count"]: raise ValidationError("state batch counters do not reconcile")
    if state["reset"]["required"] and (not state["reset"]["attempted"] or not state["reset"]["complete"] or state["reset"]["failed"]): raise ValidationError("required reset is incomplete")
    aggregation = exact(trace["aggregation"], {"is_aggregated", "source_trace_sha256s", "component_trace_count", "full_model_trace_count", "coverage"}, "aggregation")
    if not isinstance(aggregation["is_aggregated"], bool) or not isinstance(aggregation["source_trace_sha256s"], list) or len(set(aggregation["source_trace_sha256s"])) != len(aggregation["source_trace_sha256s"]): raise ValidationError("aggregation sources are invalid or duplicated")
    for source in aggregation["source_trace_sha256s"]: digest(source, "aggregation.source_trace_sha256")
    for name in ("component_trace_count", "full_model_trace_count"): nonnegative(aggregation[name], f"aggregation.{name}")
    if aggregation["coverage"] != trace["scope"]: raise ValidationError("aggregation coverage differs from scope")
    if trace["scope"] == "production_server":
        server = exact(trace["server"], {"transport", "protocol", "observation", "request_trace_count", "request_count", "ready_observed", "release_observed", "gateway", "openwebui_observed"}, "server")
        for name in ("transport", "protocol", "gateway"): string(server[name], f"server.{name}", identifier=True)
        if server["observation"] not in {"per_request", "run_summary"}: raise ValidationError("server observation is invalid")
        for name in ("request_trace_count", "request_count"): nonnegative(server[name], f"server.{name}")
        if server["observation"] == "per_request" and (server["request_trace_count"] != 1 or server["request_count"] != 1): raise ValidationError("per-request server counts differ")
        if server["ready_observed"] is not True or server["release_observed"] is not True: raise ValidationError("server boundary was not observed")
        if not isinstance(server["openwebui_observed"], bool): raise ValidationError("server observation flag invalid")
    elif trace["server"] is not None: raise ValidationError("non-server trace has server object")
    if trace["scope"] != facts.get("scope") and not (facts.get("scope") in {"worker", "direct_worker"} and trace["scope"] == "full_model"): raise ValidationError("trace/facts scope differs")
    exact(binding, {"schema_version", "trace_id", "trace_sha256", "executor_record_sha256"}, "trace binding")
    if binding["schema_version"] != "ullm.production_executor_trace_binding.v1" or binding["trace_id"] != trace["trace_id"] or binding["trace_sha256"] != sha(trace_raw) or binding["executor_record_sha256"] != sha(facts_raw): raise ValidationError("trace binding differs")
    exact(trace["verification"], {"producer_verified", "independent_validation"}, "verification"); exact(trace["verification"]["independent_validation"], {"status", "validator_id", "validator_version", "report_sha256", "failure_codes"}, "independent_validation")
    if trace["verification"]["producer_verified"] != trace["producer"]["verified"]: raise ValidationError("verification producer flag differs")
    independent = trace["verification"]["independent_validation"]
    if independent["status"] not in {"not_run", "valid", "invalid"} or not isinstance(independent["failure_codes"], list) or len(independent["failure_codes"]) > 128: raise ValidationError("independent validation fields are invalid")
    if independent["status"] == "not_run" and (independent["validator_id"] is not None or independent["validator_version"] is not None or independent["report_sha256"] is not None or independent["failure_codes"]): raise ValidationError("not_run validation must be empty")
    if independent["status"] == "valid":
        string(independent["validator_id"], "independent_validation.validator_id", identifier=True); string(independent["validator_version"], "independent_validation.validator_version", identifier=True); digest(independent["report_sha256"], "independent_validation.report_sha256")
        if independent["failure_codes"]: raise ValidationError("valid validation has failure codes")
    if trace["failure"] is None:
        if trace["status"] != "ok": raise ValidationError("non-ok trace has no failure")
    else:
        exact(trace["failure"], {"class", "stage", "reason_code", "message"}, "failure")
        if trace["status"] == "ok" or trace["failure"]["class"] not in {"unsupported", "oom", "execution", "validation", "skipped"}: raise ValidationError("failure/status mismatch")
        for name in ("class", "stage", "reason_code", "message"): string(trace["failure"][name], f"failure.{name}", identifier=name != "message")
    exact(facts, {"schema_version", "trace_id", "status", "scope", "graph", "executor", "request_summary", "phases", "operator_resolutions", "fallback", "memory", "state_commit", "server", "failure"}, "executor record")
    if facts["schema_version"] != "ullm.production_executor_record.v1" or facts["trace_id"] != trace["trace_id"] or facts["status"] != trace["status"] or facts["failure"] != trace["failure"]: raise ValidationError("executor record identity differs")
    if trace["executor"] != facts["executor"] or trace["request_summary"] != facts["request_summary"] or trace["phases"] != facts["phases"] or trace["operator_resolutions"] != facts["operator_resolutions"] or trace["fallback"] != facts["fallback"] or trace["memory"] != facts["memory"] or trace["state_commit"] != facts["state_commit"] or trace["server"] != facts["server"]:
        raise ValidationError("trace execution fields were not reconstructed from executor record")
    report = {"schema_version": "ullm.production_execution_trace_validator.v1", "status": "valid", "trace_sha256": sha(trace_raw), "executor_record_sha256": sha(facts_raw), "scope": trace["scope"], "promotion_eligible": trace["scope"] == "production_server" and trace["status"] == "ok" and not trace["fallback"]["unexpected_fallback_count"] and not trace["fallback"]["unsupported_count"] and not trace["fallback"]["fail_closed_count"] and trace["verification"]["independent_validation"]["status"] == "valid"}
    return report


def atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink(): raise ValidationError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.incomplete")
    with temporary.open("xb") as target:
        target.write(raw); target.flush(); os.fsync(target.fileno())
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--executor-record", type=Path, required=True); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--output", type=Path); parser.add_argument("--report", type=Path, help="detached validator report to bind when trace carries independent_validation=valid"); parser.add_argument("--verified-trace", type=Path); parser.add_argument("--verified-binding", type=Path)
    args = parser.parse_args(argv)
    try:
        trace, trace_raw = load(args.trace, "trace"); manifest, manifest_raw = load(args.manifest, "manifest"); facts, facts_raw = load(args.executor_record, "executor record"); binding, _ = load(args.binding, "binding")
        check_tree(trace); reject_forbidden(trace)
        check_tree(facts); reject_forbidden(facts)
        check_tree(manifest); reject_forbidden(manifest)
        check_tree(binding); reject_forbidden(binding)
        if trace.get("verification", {}).get("independent_validation", {}).get("status") == "valid":
            if args.report is None:
                raise ValidationError("valid independent_validation requires --report detached report")
            detached, detached_raw = load(args.report, "detached validator report")
            check_tree(detached); reject_forbidden(detached)
            exact(detached, {"schema_version", "status", "trace_sha256", "executor_record_sha256", "scope", "promotion_eligible"}, "detached validator report")
            expected_report_sha = trace["verification"]["independent_validation"].get("report_sha256")
            if sha(detached_raw) != expected_report_sha or detached.get("status") != "valid":
                raise ValidationError("detached validator report digest/status differs")
        report = validate_trace(trace, manifest, manifest_raw, args.manifest.resolve(), facts, binding, trace_raw, facts_raw)
        if (args.verified_trace is None) != (args.verified_binding is None): raise ValidationError("verified trace and binding must be supplied together")
        if args.verified_trace is not None and args.output is None:
            raise ValidationError("verified trace requires --output detached validator report")
        report_raw = (json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if args.output: atomic_write(args.output, report_raw)
        report_digest = sha(report_raw)
        if args.verified_trace is not None:
            verified = json.loads(json.dumps(trace)); verified["verification"]["independent_validation"] = {"status": "valid", "validator_id": "ullm-production-execution-trace-validator", "validator_version": "0.2.0", "report_sha256": report_digest, "failure_codes": []}
            verified_raw = (json.dumps(verified, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8")
            atomic_write(args.verified_trace, verified_raw)
            verified_binding = {"schema_version": "ullm.production_executor_trace_binding.v1", "trace_id": verified["trace_id"], "trace_sha256": sha(verified_raw), "executor_record_sha256": sha(facts_raw)}
            atomic_write(args.verified_binding, (json.dumps(verified_binding, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            report["verified_trace_sha256"] = sha(verified_raw)
        print(json.dumps(report, ensure_ascii=True, sort_keys=True))
        return 0
    except (ValidationError, OSError, ValueError) as error:
        print(f"trace validation failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
