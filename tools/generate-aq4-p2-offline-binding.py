#!/usr/bin/env python3
"""Generate and validate the offline AQ4 P2 binding-input bundle.

The bundle is deliberately an input bundle, not a performance or promotion
artifact.  It binds the current served-model identity to the P0 snapshot and
the live P1 trace, extracts the P1 canonical graph/state without changing
their values, and records that the production all-M=1 path oracle has not run.
No source-model revision is copied into ``model_identity.json``.

This tool performs no GPU work, starts no worker, and does not call a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Callable


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_IDENTITY_FIELDS = ("id", "revision", "format_id", "implementation_id")
REQUIRED_CORRECTNESS_FIELDS = (
    "max_hidden_relative_l2",
    "max_hidden_max_abs",
    "max_logits_relative_l2",
    "max_logits_max_abs",
    "minimum_top_k_overlap",
)
ROOT_FIELDS = {
    "schema_version",
    "status",
    "promotion_eligible",
    "model_identity",
    "graph",
    "state",
    "inputs",
    "source_oracle",
    "path_oracle",
    "correctness_thresholds",
    "artifacts",
    "promotion",
}
GRAPH_FIELDS = {"canonical", "schema_id", "schema_version", "source"}
TRACE_GRAPH_FIELDS = {"schema_id", "schema_version", "sha256", "source"}
MODEL_GRAPH_CANONICAL_FIELDS = {
    "block_size",
    "cache_blocks",
    "context_length",
    "format_id",
    "hidden_size",
    "layers",
    "model_id",
    "terminal_components",
    "vocab_size",
}
STATE_CANONICAL_FIELDS = {
    "request_state",
    "reset_scope",
    "resident_weights_reloaded_per_request",
    "transaction",
}
LAYER_FIELDS = {"kind", "layer_index", "tensor_count"}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HASH_BYTES = 64 * 1024 * 1024
MAX_SUMS_BYTES = 1024 * 1024
EXPECTED_BUNDLE_FILES = {
    "SHA256SUMS",
    "binding-inputs.json",
    "correctness-threshold-audit.json",
    "graph.json",
    "hash-manifest.json",
    "model_identity.json",
    "state.json",
    "validation-report.json",
}
VALIDATION_REPORT_FIELDS = {
    "schema_version",
    "status",
    "promotion_eligible",
    "checked",
    "artifact_directory",
    "artifacts",
    "path_oracle_status",
    "correctness_threshold_status",
    "blocking_reasons",
    "promotion_reason",
}
VALIDATION_CHECKS = [
    "active/P0/P1 four-field model identity equality",
    "P1 canonical graph/state extraction and trace digest/schema/source equality",
    "duplicate and unknown-field rejection contract",
    "source oracle role separation",
    "same-artifact all-M=1 production path oracle not-run contract",
    "correctness threshold audit remains blocked",
    "generated artifact hashes",
]
VALIDATION_BLOCKING_REASONS = [
    "same-artifact all-M=1 production path oracle was not executed",
    "numerical correctness thresholds are not available in the normative plan/spec",
]

# Tests replace this hook to force a rename, rewrite, or append at a precise
# point in the fd-based read.  Production callers leave it as ``None``.
_READ_TEST_HOOK: Callable[[Path, str], None] | None = None


class BindingError(ValueError):
    """Raised when an offline binding input is malformed or inconsistent."""


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise BindingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> Any:
    raise BindingError(f"non-finite JSON value: {value}")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_components(path: Path, label: str, *, directory: bool) -> Path:
    """Reject every symlink component without resolving through it."""

    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    require(bool(parts), f"{label} path is invalid")
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise BindingError(f"{label} path component is unavailable: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise BindingError(f"{label} path contains a symlink component: {current}")
        leaf = index == len(parts) - 1
        if not leaf and not stat.S_ISDIR(metadata.st_mode):
            raise BindingError(f"{label} parent component is not a directory: {current}")
        if leaf:
            expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
            require(expected, f"{label} must be a regular {'directory' if directory else 'file'}")
    return absolute


def _same_identity(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _stable_read(path: Path, label: str, *, maximum: int) -> bytes:
    """Read a bounded regular file through one fd and detect TOCTOU changes."""

    absolute = _validate_components(path, label, directory=False)
    try:
        pre_open = os.stat(absolute, follow_symlinks=False)
    except OSError as error:
        raise BindingError(f"{label} is unavailable before open") from error
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise BindingError(f"{label} cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        path_before = os.stat(absolute, follow_symlinks=False)
        require(stat.S_ISREG(before.st_mode), f"{label} fd is not a regular file")
        require(_same_identity(pre_open, before), f"{label} changed while opening")
        require(_same_identity(before, path_before), f"{label} changed before read")
        require(before.st_size <= maximum, f"{label} exceeds bounded size")
        if _READ_TEST_HOOK is not None:
            _READ_TEST_HOOK(absolute, "after_open")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            require(total <= maximum, f"{label} exceeds bounded size")
        if _READ_TEST_HOOK is not None:
            _READ_TEST_HOOK(absolute, "after_read")
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(absolute, follow_symlinks=False)
        except OSError as error:
            raise BindingError(f"{label} path disappeared during read") from error
        require(_same_identity(before, after), f"{label} changed during read")
        require(_same_identity(before, path_after), f"{label} path changed during read")
        require(total == before.st_size, f"{label} size changed during read")
        _validate_components(absolute, label, directory=False)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json(path: Path, label: str) -> Any:
    """Read one stable JSON file, rejecting duplicates and non-finite values."""

    try:
        raw = _stable_read(path, label, maximum=MAX_JSON_BYTES)
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BindingError(f"invalid {label}: {error}") from error
    _reject_nonfinite(value, label)
    return value


def _reject_nonfinite(value: Any, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BindingError(f"non-finite number in {label}")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{label}[{index}]")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path, label: str) -> str:
    return sha_bytes(_stable_read(path, label, maximum=MAX_HASH_BYTES))


def regular(path: Path, label: str) -> Path:
    return _validate_components(path, label, directory=False)


def regular_directory(path: Path, label: str) -> Path:
    return _validate_components(path, label, directory=True)


def ensure_directory(path: Path, label: str) -> Path:
    """Create a directory only after validating every existing parent."""

    absolute = _lexical_absolute(path)
    cursor = absolute
    missing: list[Path] = []
    while not cursor.exists() and not cursor.is_symlink():
        missing.append(cursor)
        cursor = cursor.parent
    regular_directory(cursor, f"{label} existing parent")
    for item in reversed(missing):
        os.mkdir(item, 0o755)
        regular_directory(item, label)
    return regular_directory(absolute, label)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def exact(value: Any, fields: set[str] | tuple[str, ...], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    expected = set(fields)
    require(set(value) == expected, f"{label} fields differ")
    return value


def nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f"{label} must be a non-empty string")
    return value


def hash_field(value: Any, label: str) -> str:
    require(isinstance(value, str) and HASH_RE.fullmatch(value) is not None, f"{label} must be a lowercase SHA-256")
    return value


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_new(path: Path, value: bytes) -> None:
    """Atomically create a new file; never overwrite prior evidence."""

    path = _lexical_absolute(path)
    regular_directory(path.parent, "artifact parent")
    if os.path.lexists(path):
        raise BindingError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.incomplete")
    if os.path.lexists(temporary):
        raise BindingError(f"refusing to overwrite {temporary}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o644)
    try:
        view = memoryview(value)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short write for {path.name}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    regular(path, path.name)


def active_identity(manifest: dict[str, Any], manifest_path: Path) -> tuple[dict[str, str], str]:
    public = manifest.get("public")
    fmt = manifest.get("format")
    require(isinstance(public, dict), "active.public must be an object")
    require(isinstance(fmt, dict), "active.format must be an object")
    identity = {
        "id": nonempty_string(public.get("id"), "active.public.id"),
        "revision": nonempty_string(public.get("revision"), "active.public.revision"),
        "format_id": nonempty_string(fmt.get("format_id"), "active.format.format_id"),
        "implementation_id": nonempty_string(fmt.get("implementation_id"), "active.format.implementation_id"),
    }
    return identity, sha_file(manifest_path, "active manifest")


def validate_model_identity(value: Any, label: str = "model_identity") -> dict[str, str]:
    result = exact(value, MODEL_IDENTITY_FIELDS, label)
    for field in MODEL_IDENTITY_FIELDS:
        nonempty_string(result[field], f"{label}.{field}")
    # The four-field file must not smuggle an upstream checkpoint revision into
    # the served identity.  An upstream revision belongs only to source_oracle.
    require("upstream_id" not in result and "upstream_revision" not in result, f"{label} mixes source identity")
    return {field: result[field] for field in MODEL_IDENTITY_FIELDS}


def _validate_model_canonical(value: Any, identity: dict[str, str], label: str) -> None:
    graph = exact(value, MODEL_GRAPH_CANONICAL_FIELDS, label)
    require(graph["model_id"] == identity["id"], f"{label}.model_id differs from active identity")
    require(graph["format_id"] == identity["format_id"], f"{label}.format_id differs from active identity")
    require(isinstance(graph["context_length"], int) and not isinstance(graph["context_length"], bool), f"{label}.context_length is invalid")
    require(isinstance(graph["vocab_size"], int) and not isinstance(graph["vocab_size"], bool), f"{label}.vocab_size is invalid")
    layers = graph["layers"]
    require(isinstance(layers, list) and layers, f"{label}.layers must be a non-empty array")
    for index, layer in enumerate(layers):
        child = exact(layer, LAYER_FIELDS, f"{label}.layers[{index}]")
        require(isinstance(child["layer_index"], int) and not isinstance(child["layer_index"], bool), f"{label}.layers[{index}].layer_index is invalid")
        require(isinstance(child["tensor_count"], int) and child["tensor_count"] >= 0, f"{label}.layers[{index}].tensor_count is invalid")
        require(isinstance(child["kind"], list) and all(isinstance(kind, str) for kind in child["kind"]), f"{label}.layers[{index}].kind is invalid")
    require(isinstance(graph["terminal_components"], list), f"{label}.terminal_components is invalid")


def _validate_state_canonical(value: Any, label: str) -> None:
    state = exact(value, STATE_CANONICAL_FIELDS, label)
    require(isinstance(state["request_state"], list), f"{label}.request_state is invalid")
    require(isinstance(state["transaction"], list), f"{label}.transaction is invalid")
    require(isinstance(state["reset_scope"], str), f"{label}.reset_scope is invalid")
    require(isinstance(state["resident_weights_reloaded_per_request"], bool), f"{label}.resident_weights_reloaded_per_request is invalid")


def _validate_graph_inputs(
    record: dict[str, Any], trace: dict[str, Any], identity: dict[str, str], active_sha: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph = exact(record.get("graph"), {"model_graph", "state_schema"}, "executor_record.graph")
    trace_graph = exact(trace.get("graph"), {"model_graph", "state_schema", "compatibility_key_sha256"}, "trace.graph")
    extracted: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for name in ("model_graph", "state_schema"):
        wrapper = exact(graph.get(name), GRAPH_FIELDS, f"executor_record.graph.{name}")
        trace_wrapper = exact(trace_graph.get(name), TRACE_GRAPH_FIELDS, f"trace.graph.{name}")
        canonical_value = wrapper["canonical"]
        require(isinstance(canonical_value, dict), f"executor_record.graph.{name}.canonical must be an object")
        if name == "model_graph":
            _validate_model_canonical(canonical_value, identity, f"graph.{name}.canonical")
        else:
            _validate_state_canonical(canonical_value, f"graph.{name}.canonical")
        digest = sha_bytes(canonical(canonical_value))
        require(digest == trace_wrapper["sha256"], f"{name} canonical hash differs from P1 trace")
        require(wrapper["schema_id"] == trace_wrapper["schema_id"], f"{name} schema_id differs from P1 trace")
        require(wrapper["schema_version"] == trace_wrapper["schema_version"], f"{name} schema_version differs from P1 trace")
        require(wrapper["source"] == trace_wrapper["source"], f"{name} source differs from P1 trace")
        extracted[name] = canonical_value
        metadata[name] = {
            "schema_id": wrapper["schema_id"],
            "schema_version": wrapper["schema_version"],
            "source": wrapper["source"],
            "sha256": digest,
            "trace_sha256": trace_wrapper["sha256"],
        }
    require(trace.get("identity", {}).get("served_model_manifest_sha256") == active_sha, "P1 trace manifest identity differs from active manifest")
    require(trace_graph["compatibility_key_sha256"] and HASH_RE.fullmatch(trace_graph["compatibility_key_sha256"]) is not None, "P1 compatibility digest is invalid")
    return extracted["model_graph"], extracted["state_schema"], metadata["model_graph"], metadata["state_schema"]


def validate_inputs(
    active_path: Path,
    p0_path: Path,
    record_path: Path,
    trace_path: Path,
    source_path: Path,
    template_path: Path,
    spec_path: Path,
) -> dict[str, Any]:
    """Validate source inputs and return the values needed to build artifacts."""

    active_path = regular(active_path, "active manifest")
    p0_path = regular(p0_path, "P0 snapshot")
    record_path = regular(record_path, "P1 executor record")
    trace_path = regular(trace_path, "P1 trace")
    source_path = regular(source_path, "source-oracle manifest")
    template_path = regular(template_path, "threshold policy template")
    spec_path = regular(spec_path, "prefill validation specification")
    active = load_json(active_path, "active manifest")
    p0 = load_json(p0_path, "P0 snapshot")
    record = load_json(record_path, "P1 executor record")
    trace = load_json(trace_path, "P1 trace")
    source = load_json(source_path, "source oracle")
    template = load_json(template_path, "threshold policy template")
    active_model, active_sha = active_identity(active, active_path)
    require(p0.get("schema_version") == "ullm.aq4_production_optimization_p0.v1", "P0 snapshot schema differs")
    p0_identity = exact(p0.get("identity", {}).get("model"), MODEL_IDENTITY_FIELDS, "P0 identity.model")
    require(validate_model_identity(p0_identity, "P0 identity.model") == active_model, "P0 model identity differs from active manifest")
    require(p0.get("identity", {}).get("manifest", {}).get("sha256") == active_sha, "P0 manifest hash differs from active manifest")
    require(record.get("schema_version") == "ullm.production_executor_record.v1", "P1 executor record schema differs")
    require(record.get("status") == "ok" and record.get("scope") == "production_server", "P1 executor record must be live production_server ok")
    require(trace.get("schema_version") == "ullm.production_execution_trace.v1", "P1 trace schema differs")
    require(trace.get("status") == "ok" and trace.get("scope") == "production_server", "P1 trace must be live production_server ok")
    require(record.get("trace_id") == trace.get("trace_id"), "P1 record/trace id differs")
    trace_model = exact(trace.get("identity", {}).get("model"), MODEL_IDENTITY_FIELDS, "P1 trace identity.model")
    require(validate_model_identity(trace_model, "P1 trace identity.model") == active_model, "P1 trace model identity differs from active manifest")
    require(trace.get("identity", {}).get("worker", {}).get("binary_sha256") == active.get("worker", {}).get("binary_sha256"), "P1 worker identity differs from active manifest")
    model_graph, state_schema, model_meta, state_meta = _validate_graph_inputs(record, trace, active_model, active_sha)
    require(source.get("schema_version") == "ullm.qwen35_aq4_source_oracle.v1", "source oracle schema differs")
    require(source.get("oracle_kind") == "independent_source", "source oracle must be independent_source")
    require(source.get("status") == "available", "source oracle is not available")
    source_identity = source.get("identity")
    require(isinstance(source_identity, dict), "source oracle identity is missing")
    source_model_id = nonempty_string(source_identity.get("model_id"), "source oracle identity.model_id")
    source_revision = nonempty_string(source_identity.get("model_revision"), "source oracle identity.model_revision")
    require(template.get("schema_version") == "ullm.aq4_production_p2_threshold_policy.v1", "threshold template schema differs")
    correctness = template.get("correctness_thresholds")
    require(isinstance(correctness, dict), "threshold template correctness_thresholds is missing")
    for field in REQUIRED_CORRECTNESS_FIELDS:
        require(field in correctness, f"threshold template missing {field}")
        require(correctness[field] is None, f"threshold template unexpectedly binds {field}; refusing to invent or adopt it")
    require(spec_path.is_file(), "prefill validation specification is unavailable")
    return {
        "active_model": active_model,
        "active_sha": active_sha,
        "p0_sha": sha_file(p0_path, "P0 snapshot"),
        "record_sha": sha_file(record_path, "P1 executor record"),
        "trace_sha": sha_file(trace_path, "P1 trace"),
        "source_sha": sha_file(source_path, "source oracle"),
        "template_sha": sha_file(template_path, "threshold policy template"),
        "spec_sha": sha_file(spec_path, "prefill validation specification"),
        "model_graph": model_graph,
        "state_schema": state_schema,
        "model_meta": model_meta,
        "state_meta": state_meta,
        "source_model_id": source_model_id,
        "source_revision": source_revision,
        "source_status": source.get("status"),
        "source_oracle_kind": source.get("oracle_kind"),
        "input_paths": {
            "active_manifest": str(active_path),
            "p0_snapshot": str(p0_path),
            "p1_executor_record": str(record_path),
            "p1_trace": str(trace_path),
            "source_oracle_manifest": str(source_path),
            "threshold_policy_template": str(template_path),
            "prefill_validation_spec": str(spec_path),
        },
    }


def _threshold_audit(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "ullm.aq4_production_p2_correctness_threshold_audit.v1",
        "status": "blocked",
        "decision": "BLOCKED",
        "promotion_eligible": False,
        "required_fields": list(REQUIRED_CORRECTNESS_FIELDS),
        "values": None,
        "normative_sources": [
            {"path": "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json", "sha256": values["template_sha"]},
            {"path": "docs/specs/prefill-validation-v0.1.md", "sha256": values["spec_sha"]},
        ],
        "finding": "The existing plan/spec define qualitative correctness gates but do not define numerical values for the five required fields.",
        "blocking_reasons": [
            "no approved numerical correctness thresholds are present in the existing plan/spec artifacts",
            "this generator refuses to invent threshold values",
            "P2 execution and promotion must remain blocked until a reviewed threshold policy is bound",
        ],
    }


def _validation_report(values: dict[str, Any], output: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "ullm.aq4_production_p2_offline_binding_validation.v1",
        "status": "valid",
        "promotion_eligible": False,
        "checked": VALIDATION_CHECKS,
        "artifact_directory": str(output),
        "artifacts": artifacts,
        "path_oracle_status": "not_run",
        "correctness_threshold_status": "blocked",
        "blocking_reasons": VALIDATION_BLOCKING_REASONS,
        "promotion_reason": "offline binding inputs are not promotion evidence",
    }


def generate(args: argparse.Namespace) -> Path:
    output = ensure_directory(args.output_dir, "output directory")
    values = validate_inputs(
        args.active_manifest,
        args.p0_snapshot,
        args.p1_executor_record,
        args.p1_trace,
        args.source_oracle,
        args.threshold_template,
        args.prefill_spec,
    )
    model = values["active_model"]
    graph = values["model_graph"]
    state = values["state_schema"]
    audit = _threshold_audit(values)
    model_path = output / "model_identity.json"
    graph_path = output / "graph.json"
    state_path = output / "state.json"
    audit_path = output / "correctness-threshold-audit.json"
    write_new(model_path, json_bytes(model))
    write_new(graph_path, json_bytes(graph))
    write_new(state_path, json_bytes(state))
    write_new(audit_path, json_bytes(audit))
    core = {
        "model_identity": {"path": model_path.name, "sha256": sha_file(model_path, "model identity"), "type": "model_identity"},
        "graph": {"path": graph_path.name, "sha256": sha_file(graph_path, "graph"), "type": "model_graph"},
        "state": {"path": state_path.name, "sha256": sha_file(state_path, "state"), "type": "state_schema"},
        "correctness_threshold_audit": {"path": audit_path.name, "sha256": sha_file(audit_path, "threshold audit"), "type": "correctness_threshold_audit"},
    }
    binding = {
        "schema_version": "ullm.aq4_production_p2_offline_binding_inputs.v1",
        "status": "blocked",
        "promotion_eligible": False,
        "model_identity": model,
        "graph": {"artifact": core["graph"], **values["model_meta"]},
        "state": {"artifact": core["state"], **values["state_meta"]},
        "inputs": {
            name: {"path": path, "sha256": digest}
            for name, path, digest in (
                ("active_manifest", values["input_paths"]["active_manifest"], values["active_sha"]),
                ("p0_snapshot", values["input_paths"]["p0_snapshot"], values["p0_sha"]),
                ("p1_executor_record", values["input_paths"]["p1_executor_record"], values["record_sha"]),
                ("p1_trace", values["input_paths"]["p1_trace"], values["trace_sha"]),
            )
        },
        "source_oracle": {
            "path": values["input_paths"]["source_oracle_manifest"],
            "sha256": values["source_sha"],
            "oracle_kind": values["source_oracle_kind"],
            "status": values["source_status"],
            "source_model_id": values["source_model_id"],
            "source_model_revision": values["source_revision"],
            "production_path_oracle_substitute": False,
        },
        "path_oracle": {
            "kind": "same-artifact-all-m1",
            "status": "not_run",
            "required": True,
            "reason": "P2 offline binding generation performs no production execution",
            "source_oracle_substitute": False,
            "p1_trace_substitute": False,
            "fixture_substitute": False,
            "production_path_oracle": False,
        },
        "correctness_thresholds": {
            "status": "blocked",
            "artifact": core["correctness_threshold_audit"],
            "template_path": values["input_paths"]["threshold_policy_template"],
            "template_sha256": values["template_sha"],
            "required_fields": list(REQUIRED_CORRECTNESS_FIELDS),
        },
        "artifacts": core,
        "promotion": {
            "eligible": False,
            "reason": "offline binding inputs are not production performance or promotion evidence",
            "path_oracle_not_run": True,
            "thresholds_blocked": True,
        },
    }
    # Build the report before binding-inputs so binding can include its final
    # digest without creating a self-hash cycle.
    report_path = output / "validation-report.json"
    report_artifacts = dict(core)
    report_artifacts["validation_report"] = {"path": report_path.name, "sha256": None, "type": "validation_report"}
    report = _validation_report(values, output, report_artifacts)
    report_raw = json_bytes(report)
    report_artifacts["validation_report"]["sha256"] = sha_bytes(report_raw)
    binding["artifacts"] = report_artifacts
    binding_raw = json_bytes(binding)
    write_new(report_path, report_raw)
    write_new(output / "binding-inputs.json", binding_raw)
    all_artifacts = {
        path.name: {"bytes": os.lstat(regular(path, path.name)).st_size, "sha256": sha_file(path, path.name)}
        for path in (model_path, graph_path, state_path, audit_path, report_path, output / "binding-inputs.json")
    }
    hash_manifest = {
        "schema_version": "ullm.aq4_production_p2_hash_manifest.v1",
        "status": "complete",
        "self_included": False,
        "artifacts": [{"path": name, **all_artifacts[name]} for name in sorted(all_artifacts)],
    }
    hash_path = output / "hash-manifest.json"
    write_new(hash_path, json_bytes(hash_manifest))
    sums_path = output / "SHA256SUMS"
    sums = "".join(
        f"{sha_file(output / name, name)}  {name}\n"
        for name in sorted(EXPECTED_BUNDLE_FILES - {"SHA256SUMS"})
    )
    write_new(sums_path, sums.encode("ascii"))
    # Validate the final bundle, including hash-manifest and SHA256SUMS.
    validate_bundle(output)
    return output


def _artifact_link(value: Any, output: Path, label: str, *, expected_path: str, expected_type: str) -> Path:
    link = exact(value, {"path", "sha256", "type"}, label)
    name = nonempty_string(link["path"], f"{label}.path")
    require(name == expected_path and Path(name).name == name, f"{label} path differs")
    require(link["type"] == expected_type, f"{label} type differs")
    path = output / name
    hash_field(link["sha256"], f"{label}.sha256")
    require(sha_file(path, label) == link["sha256"], f"{label} hash differs")
    return path


def _bundle_file_set(output: Path) -> set[str]:
    """Return the exact regular-file set, rejecting symlinks and directories."""

    before = os.stat(output, follow_symlinks=False)
    names: set[str] = set()
    with os.scandir(output) as entries:
        for entry in entries:
            require(not entry.is_symlink(), f"unexpected symlink in bundle: {entry.name}")
            metadata = entry.stat(follow_symlinks=False)
            require(stat.S_ISREG(metadata.st_mode), f"unexpected non-file in bundle: {entry.name}")
            require(entry.name not in names, f"duplicate bundle entry: {entry.name}")
            names.add(entry.name)
    after = os.stat(output, follow_symlinks=False)
    require(_same_identity(before, after), "output directory changed during enumeration")
    return names


def _validate_validation_report(
    report: Any,
    output: Path,
    binding_artifacts: dict[str, Any],
) -> None:
    document = exact(report, VALIDATION_REPORT_FIELDS, "validation report")
    require(document["schema_version"] == "ullm.aq4_production_p2_offline_binding_validation.v1", "validation report schema differs")
    require(document["status"] == "valid" and document["promotion_eligible"] is False, "validation report status differs")
    require(document["checked"] == VALIDATION_CHECKS, "validation report checks differ")
    require(document["artifact_directory"] == str(output), "validation report artifact directory differs")
    require(document["path_oracle_status"] == "not_run", "validation report path oracle status differs")
    require(document["correctness_threshold_status"] == "blocked", "validation report correctness status differs")
    require(document["blocking_reasons"] == VALIDATION_BLOCKING_REASONS, "validation report blocking reasons differ")
    require(document["promotion_reason"] == "offline binding inputs are not promotion evidence", "validation report promotion reason differs")
    artifacts = exact(
        document["artifacts"],
        {"model_identity", "graph", "state", "correctness_threshold_audit", "validation_report"},
        "validation report artifacts",
    )
    for name in ("model_identity", "graph", "state", "correctness_threshold_audit"):
        require(artifacts[name] == binding_artifacts[name], f"validation report {name} link differs")
    self_link = exact(artifacts["validation_report"], {"path", "sha256", "type"}, "validation report self link")
    require(
        self_link == {"path": "validation-report.json", "sha256": None, "type": "validation_report"},
        "validation report self link differs",
    )


def validate_bundle(output_dir: Path) -> dict[str, Any]:
    """Validate a generated bundle; fail closed on any mismatch or tamper."""

    output = regular_directory(output_dir, "output directory")
    require(_bundle_file_set(output) == EXPECTED_BUNDLE_FILES, "bundle file set differs")
    binding_path = output / "binding-inputs.json"
    binding = load_json(binding_path, "binding-inputs")
    require(set(binding) == ROOT_FIELDS, "binding-inputs root fields differ")
    require(binding["schema_version"] == "ullm.aq4_production_p2_offline_binding_inputs.v1", "binding schema differs")
    require(binding["status"] == "blocked" and binding["promotion_eligible"] is False, "binding must remain blocked and ineligible")
    model = validate_model_identity(load_json(output / "model_identity.json", "model identity"))
    require(model == binding["model_identity"], "model identity binding differs")
    artifacts = exact(binding["artifacts"], {"model_identity", "graph", "state", "correctness_threshold_audit", "validation_report"}, "binding artifacts")
    graph_path = _artifact_link(artifacts["graph"], output, "graph artifact", expected_path="graph.json", expected_type="model_graph")
    state_path = _artifact_link(artifacts["state"], output, "state artifact", expected_path="state.json", expected_type="state_schema")
    model_path = _artifact_link(artifacts["model_identity"], output, "model identity artifact", expected_path="model_identity.json", expected_type="model_identity")
    audit_path = _artifact_link(artifacts["correctness_threshold_audit"], output, "threshold audit artifact", expected_path="correctness-threshold-audit.json", expected_type="correctness_threshold_audit")
    report_path = _artifact_link(artifacts["validation_report"], output, "validation report artifact", expected_path="validation-report.json", expected_type="validation_report")
    _validate_validation_report(load_json(report_path, "validation report"), output, artifacts)
    audit = load_json(audit_path, "threshold audit")
    require(set(audit) == {"schema_version", "status", "decision", "promotion_eligible", "required_fields", "values", "normative_sources", "finding", "blocking_reasons"}, "threshold audit fields differ")
    require(audit["status"] == "blocked" and audit["decision"] == "BLOCKED" and audit["promotion_eligible"] is False and audit["values"] is None, "threshold audit is not blocked")
    require(audit["required_fields"] == list(REQUIRED_CORRECTNESS_FIELDS), "threshold audit field list differs")
    path_oracle = exact(binding["path_oracle"], {"kind", "status", "required", "reason", "source_oracle_substitute", "p1_trace_substitute", "fixture_substitute", "production_path_oracle"}, "path_oracle")
    require(path_oracle == {
        "kind": "same-artifact-all-m1",
        "status": "not_run",
        "required": True,
        "reason": path_oracle["reason"],
        "source_oracle_substitute": False,
        "p1_trace_substitute": False,
        "fixture_substitute": False,
        "production_path_oracle": False,
    }, "same-artifact all-M=1 path oracle contract differs")
    source = exact(binding["source_oracle"], {"path", "sha256", "oracle_kind", "status", "source_model_id", "source_model_revision", "production_path_oracle_substitute"}, "source_oracle")
    source_path = regular(Path(source["path"]), "source oracle")
    require(sha_file(source_path, "source oracle") == source["sha256"], "source oracle hash differs")
    source_doc = load_json(source_path, "source oracle")
    source_doc_identity = source_doc.get("identity", {})
    require(source_doc.get("schema_version") == "ullm.qwen35_aq4_source_oracle.v1" and source_doc.get("oracle_kind") == "independent_source" and source_doc.get("status") == "available", "source oracle document role differs")
    require(source["oracle_kind"] == "independent_source" and source["status"] == "available" and source["production_path_oracle_substitute"] is False, "source oracle role differs")
    require(source_doc_identity.get("model_id") == source["source_model_id"] and source_doc_identity.get("model_revision") == source["source_model_revision"], "source oracle identity differs")
    inputs = exact(binding["inputs"], {"active_manifest", "p0_snapshot", "p1_executor_record", "p1_trace"}, "inputs")
    for name, link in inputs.items():
        link_obj = exact(link, {"path", "sha256"}, f"inputs.{name}")
        path = regular(Path(link_obj["path"]), f"inputs.{name}")
        require(sha_file(path, f"inputs.{name}") == link_obj["sha256"], f"inputs.{name} hash differs")
    active = load_json(Path(inputs["active_manifest"]["path"]), "active manifest")
    active_model, active_sha = active_identity(active, regular(Path(inputs["active_manifest"]["path"]), "active manifest"))
    require(active_model == model and active_sha == inputs["active_manifest"]["sha256"], "active identity differs")
    p0 = load_json(Path(inputs["p0_snapshot"]["path"]), "P0 snapshot")
    require(validate_model_identity(p0["identity"]["model"], "P0 identity.model") == model, "P0 identity differs")
    record = load_json(Path(inputs["p1_executor_record"]["path"]), "P1 executor record")
    trace = load_json(Path(inputs["p1_trace"]["path"]), "P1 trace")
    model_graph, state_schema, model_meta, state_meta = _validate_graph_inputs(record, trace, model, active_sha)
    require(load_json(graph_path, "graph") == model_graph and load_json(state_path, "state") == state_schema, "standalone graph/state values differ")
    require(binding["graph"] == {"artifact": binding["artifacts"]["graph"], **model_meta}, "graph binding differs")
    require(binding["state"] == {"artifact": binding["artifacts"]["state"], **state_meta}, "state binding differs")
    correctness_binding = exact(binding["correctness_thresholds"], {"status", "artifact", "template_path", "template_sha256", "required_fields"}, "correctness_thresholds")
    require(correctness_binding["status"] == "blocked", "correctness thresholds are not blocked")
    require(correctness_binding["artifact"] == artifacts["correctness_threshold_audit"], "correctness threshold audit link differs")
    require(correctness_binding["required_fields"] == list(REQUIRED_CORRECTNESS_FIELDS), "correctness threshold required fields differ")
    template_path = regular(Path(correctness_binding["template_path"]), "threshold policy template")
    require(sha_file(template_path, "threshold policy template") == correctness_binding["template_sha256"], "threshold template hash differs")
    require(exact(binding["promotion"], {"eligible", "reason", "path_oracle_not_run", "thresholds_blocked"}, "promotion") == {
        "eligible": False,
        "reason": "offline binding inputs are not production performance or promotion evidence",
        "path_oracle_not_run": True,
        "thresholds_blocked": True,
    }, "promotion binding differs")
    # Hash manifest intentionally excludes itself.  Every listed artifact is
    # checked for duplicate paths, unknown files, bytes, and digest tampering.
    hash_manifest = load_json(output / "hash-manifest.json", "hash manifest")
    require(set(hash_manifest) == {"schema_version", "status", "self_included", "artifacts"}, "hash manifest fields differ")
    require(hash_manifest["schema_version"] == "ullm.aq4_production_p2_hash_manifest.v1" and hash_manifest["status"] == "complete" and hash_manifest["self_included"] is False, "hash manifest header differs")
    listed = hash_manifest["artifacts"]
    require(isinstance(listed, list), "hash manifest artifacts must be an array")
    names: set[str] = set()
    for item in listed:
        child = exact(item, {"path", "bytes", "sha256"}, "hash manifest artifact")
        name = nonempty_string(child["path"], "hash manifest path")
        require(Path(name).name == name and name not in names, f"duplicate or invalid hash manifest path: {name}")
        names.add(name)
        target = output / name
        metadata = os.lstat(regular(target, f"hash manifest artifact {name}"))
        require(child["bytes"] == metadata.st_size, f"hash manifest byte count differs: {name}")
        require(child["sha256"] == sha_file(target, name), f"hash manifest digest differs: {name}")
    expected_names = {"binding-inputs.json", "correctness-threshold-audit.json", "graph.json", "model_identity.json", "state.json", "validation-report.json"}
    require(names == expected_names, "hash manifest artifact set differs")
    sums_path = output / "SHA256SUMS"
    try:
        lines = _stable_read(sums_path, "SHA256SUMS", maximum=MAX_SUMS_BYTES).decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise BindingError("SHA256SUMS is not ASCII") from error
    require(len(lines) == len(expected_names) + 1, "SHA256SUMS must include hash-manifest.json and all generated artifacts")
    sums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ")
        require(len(parts) == 2, "invalid SHA256SUMS line")
        digest, name = parts
        hash_field(digest, f"SHA256SUMS.{name}")
        require(Path(name).name == name and name not in sums, f"duplicate or invalid SHA256SUMS path: {name}")
        sums[name] = digest
    require(set(sums) == expected_names | {"hash-manifest.json"}, "SHA256SUMS artifact set differs")
    for name, digest in sums.items():
        require(digest == sha_file(output / name, name), f"SHA256SUMS digest differs: {name}")
    return {"status": "valid", "promotion_eligible": False, "path_oracle_status": "not_run", "correctness_threshold_status": "blocked"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate an existing output directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-manifest", type=Path, default=Path("/etc/ullm/served-models/active.json"))
    parser.add_argument("--p0-snapshot", type=Path)
    parser.add_argument("--p1-executor-record", type=Path)
    parser.add_argument("--p1-trace", type=Path)
    parser.add_argument("--source-oracle", type=Path)
    parser.add_argument("--threshold-template", type=Path, default=Path(__file__).resolve().parents[1] / "benchmarks/workloads/aq4-production-opt-p2-threshold-policy-template-v0.1.json")
    parser.add_argument("--prefill-spec", type=Path, default=Path(__file__).resolve().parents[1] / "docs/specs/prefill-validation-v0.1.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.validate:
            result = validate_bundle(args.output_dir)
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        else:
            required = {"--p0-snapshot": args.p0_snapshot, "--p1-executor-record": args.p1_executor_record, "--p1-trace": args.p1_trace, "--source-oracle": args.source_oracle}
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise BindingError("generation requires " + ", ".join(missing))
            output = generate(args)
            print(json.dumps({"status": "blocked", "promotion_eligible": False, "output_dir": str(output)}, ensure_ascii=True, sort_keys=True))
        return 0
    except (BindingError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"AQ4 P2 offline binding failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
