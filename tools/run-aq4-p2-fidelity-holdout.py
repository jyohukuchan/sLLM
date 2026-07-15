#!/usr/bin/env python3
"""Run the frozen AQ4 P2 holdout exactly once.

``preflight`` is CPU-only and emits an immutable, hash-bound command plan.  ``execute``
consumes that plan, creates a one-shot attempt marker before starting the existing Rust
capture binary, and publishes either an immutable failure receipt or an immutable holdout
result.  The runner never derives or changes the calibration envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ROWS = 24
TOP_K = 10
RUN_SCHEMA = "ullm.aq4_p2_fidelity_holdout_run.v1"
PREFLIGHT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_preflight.v1"
ATTEMPT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_attempt.v1"
FAILURE_SCHEMA = "ullm.aq4_p2_fidelity_holdout_failure.v1"
RESULT_SCHEMA = "ullm.aq4_p2_fidelity_holdout_result.v1"
RECEIPT_SCHEMA = "ullm.aq4_p2_fidelity_freeze_receipt.v1"
METRICS_SCHEMA = "ullm.aq4_p2_fidelity_calibration_metrics.v1"
HEX = set("0123456789abcdef")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("aq4_fidelity_holdout_protocol", "generate-aq4-p2-fidelity-holdout.py")
SPLIT = _load("aq4_fidelity_holdout_split_validator", "validate-aq4-p2-fidelity-holdout.py")
CAPTURE = _load("aq4_fidelity_holdout_capture", "capture-qwen35-aq4-fidelity.py")
FULL_COMPARE = _load("aq4_fidelity_holdout_compare", "compare-qwen35-aq4-p2-calibration.py")


class HoldoutError(ValueError):
    pass


def _sha(path: Path, label: str, limit: int | None = None) -> str:
    _regular(path, label, limit=limit)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _no_symlink_components(path: Path, label: str, *, missing_leaf: bool = False) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:], 1):
        current /= component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if missing_leaf and index == len(absolute.parts) - 1:
                return
            raise HoldoutError(f"{label} path component is unavailable: {current}")
        if stat.S_ISLNK(info.st_mode):
            raise HoldoutError(f"{label} path component is a symlink: {current}")


def _regular(path: Path, label: str, *, limit: int | None = None, missing: bool = False) -> None:
    _no_symlink_components(path, label, missing_leaf=missing)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise HoldoutError(f"{label} metadata unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise HoldoutError(f"{label} must be a regular file")
    if info.st_nlink != 1:
        raise HoldoutError(f"{label} must have exactly one hard link")
    if limit is not None and info.st_size > limit:
        raise HoldoutError(f"{label} exceeds bounded size")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label, limit=64 * 1024 * 1024)
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants)
    except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
        raise HoldoutError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise HoldoutError(f"{label} root must be an object")
    return value, raw


def _atomic_json(path: Path, value: Any, label: str) -> str:
    if os.path.lexists(path):
        raise HoldoutError(f"refusing to overwrite {label}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlink_components(path.parent, f"{label} parent")
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    if os.path.lexists(temporary):
        raise HoldoutError(f"incomplete {label} already exists")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(encoded).hexdigest()


def _sha_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in HEX for char in value):
        raise HoldoutError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _identity_file(path: Path, label: str, expected_sha: str | None = None) -> dict[str, Any]:
    _regular(path, label)
    info = os.lstat(path)
    digest = _sha(path, label)
    if expected_sha is not None and digest != _sha_value(expected_sha, f"expected {label} SHA"):
        raise HoldoutError(f"{label} SHA differs")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": info.st_size, "mode": f"{stat.S_IMODE(info.st_mode):04o}", "nlink": info.st_nlink}


def _load_rows(path: Path, subset: str) -> list[dict[str, Any]]:
    _regular(path, f"{subset} cases", limit=16 * 1024 * 1024)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or len(line) > 64 * 1024:
            raise HoldoutError(f"{subset} row {number} is empty or oversized")
        try:
            value = json.loads(line, object_pairs_hook=PROTOCOL.pairs, parse_constant=PROTOCOL.no_constants)
        except (UnicodeError, json.JSONDecodeError, PROTOCOL.ProtocolError) as error:
            raise HoldoutError(f"invalid {subset} row {number}: {error}") from error
        if not isinstance(value, dict) or value.get("case_id") in seen:
            raise HoldoutError(f"{subset} rows contain duplicate case_id")
        if value.get("subset") != subset or value.get("step") != 0 or value.get("row_count") != 1:
            raise HoldoutError(f"{subset} row contract differs: {value.get('case_id')}")
        seen.add(value.get("case_id"))
        rows.append(value)
    if len(rows) != MAX_ROWS:
        raise HoldoutError(f"{subset} rows must contain exactly {MAX_ROWS} entries")
    return rows


def _freeze(split_root: Path, freeze_receipt: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    for name in ("split-manifest.json", "policy.json", "calibration-cases.jsonl", "holdout-cases.jsonl", "SHA256SUMS"):
        _regular(split_root / name, name)
    _regular(freeze_receipt, "freeze receipt")
    try:
        result = SPLIT.validate(split_root, freeze_receipt)
    except Exception as error:
        raise HoldoutError(f"split/freeze validation failed: {error}") from error
    manifest, manifest_raw = PROTOCOL.load(split_root / "split-manifest.json", "split manifest")
    policy, policy_raw = PROTOCOL.load(split_root / "policy.json", "policy")
    receipt, receipt_raw = PROTOCOL.load(freeze_receipt, "freeze receipt")
    if receipt.get("schema_version") != RECEIPT_SCHEMA or receipt.get("status") != "frozen_calibration_envelope":
        raise HoldoutError("freeze receipt is not a frozen calibration envelope")
    if receipt.get("holdout_status") != "not_started" or receipt.get("holdout_evaluations_remaining") != 1:
        raise HoldoutError("holdout has already been evaluated or is not frozen")
    holdout_path = split_root / "holdout-cases.jsonl"
    calibration_path = split_root / "calibration-cases.jsonl"
    _load_rows(calibration_path, "calibration")
    _load_rows(holdout_path, "holdout")
    shas = {
        "split_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "policy_sha256": hashlib.sha256(policy_raw).hexdigest(),
        "calibration_cases_sha256": _sha(calibration_path, "calibration cases"),
        "holdout_cases_sha256": _sha(holdout_path, "holdout cases"),
        "freeze_receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
    }
    if receipt.get("split_manifest_sha256") != shas["split_manifest_sha256"] or receipt.get("policy_sha256") != shas["policy_sha256"]:
        raise HoldoutError("freeze receipt split/policy binding differs")
    if manifest.get("calibration_sha256") != shas["calibration_cases_sha256"] or manifest.get("holdout_sha256") != shas["holdout_cases_sha256"]:
        raise HoldoutError("split manifest cases binding differs")
    return manifest, policy, receipt, result, shas


def _actual_verified(path: Path) -> dict[str, Any]:
    identity = _identity_file(path, "actual-verified receipt")
    value, _ = _read_json(path, "actual-verified receipt")
    if value.get("status") != "actual_verified":
        raise HoldoutError("actual-verified receipt status differs")
    return {"path": identity["path"], "sha256": identity["sha256"]}


def _artifact_identity(root: Path, kind: str, rows: list[dict[str, Any]], subset: str) -> dict[str, Any]:
    try:
        artifact = CAPTURE._artifact(root, kind)
    except Exception as error:
        raise HoldoutError(f"{kind} artifact validation failed: {error}") from error
    manifest = artifact["manifest"]
    if manifest.get("subset") not in (None, subset):
        raise HoldoutError(f"{kind} artifact subset differs")
    expected = {(row["case_id"], 0): row for row in rows}
    if set(artifact["rows"]) != set(expected):
        raise HoldoutError(f"{kind} artifact must contain exactly the holdout 24 rows")
    for key, row in artifact["rows"].items():
        expected_row = expected[key]
        if row.get("step") != 0 or row.get("case_id") != key[0] or row.get("input_token_ids_sha256") != expected_row.get("context_token_ids_sha256"):
            raise HoldoutError(f"{kind} artifact row identity differs: {key[0]}")
    return artifact


def _runtime_identity(active: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    runtime = active["manifest"].get("runtime", {})
    nested = runtime.get("runtime", {}) if isinstance(runtime, dict) else {}
    required = {
        "served_model_manifest_sha256": expected["served_model_manifest_sha256"],
        "package_manifest_sha256": expected["package_manifest_sha256"],
        "worker_binary_sha256": expected["worker_binary_sha256"],
        "selected_cases_sha256": expected["holdout_cases_sha256"],
        "split_manifest_sha256": expected["split_manifest_sha256"],
        "policy_sha256": expected["policy_sha256"],
        "holdout_cases_sha256": expected["holdout_cases_sha256"],
        "quantized_artifact_revision": expected["quantized_artifact_revision"],
    }
    for field, value in required.items():
        if nested.get(field) != value:
            raise HoldoutError(f"active runtime identity differs: {field}")
    if nested.get("one_process") is not True or nested.get("one_model_load") is not True or nested.get("gpu_parallelism") != 1 or runtime.get("model_loads") != 1:
        raise HoldoutError("active runtime one-process/model-load/GPU-parallelism contract differs")
    if nested.get("device", {}).get("architecture") != expected["device_architecture"]:
        raise HoldoutError("active device architecture differs")
    if nested.get("selected_subset") != "holdout":
        raise HoldoutError("active selected subset is not holdout")
    if expected.get("device_id") is not None and nested.get("device", {}).get("device_id") != expected["device_id"]:
        raise HoldoutError("active device identity differs")
    if nested.get("build_sha256") != expected["build_sha256"]:
        raise HoldoutError("active capture build SHA differs")
    source_identity = expected.get("source_identity", {})
    if nested.get("upstream_model_revision") != source_identity.get("model_revision") or nested.get("tokenizer_aggregate_sha256") != source_identity.get("tokenizer", {}).get("aggregate_sha256"):
        raise HoldoutError("active upstream source identity differs")
    source_checkpoint_sha = source_identity.get("source_checkpoint", {}).get("aggregate_sha256")
    if source_checkpoint_sha is not None and nested.get("source_checkpoint_aggregate_sha256") != source_checkpoint_sha:
        raise HoldoutError("active source checkpoint identity differs")
    return nested


def _source_active_identity(source: dict[str, Any], active: dict[str, Any]) -> dict[str, Any]:
    left = source["manifest"].get("identity", {})
    right = active["manifest"].get("identity", {})
    if left.get("model_id") != right.get("model_id") or left.get("model_revision") != right.get("model_revision") or left.get("tokenizer", {}).get("aggregate_sha256") != right.get("tokenizer", {}).get("aggregate_sha256"):
        raise HoldoutError("source/active source identity differs")
    return {"model_id": left.get("model_id"), "model_revision": left.get("model_revision"), "tokenizer_aggregate_sha256": left.get("tokenizer", {}).get("aggregate_sha256")}


def _compare(source: dict[str, Any], active: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    metrics_rows: list[dict[str, Any]] = []
    aggregate = {name: [] for name in PROTOCOL.METRICS}
    with FULL_COMPARE._VALIDATOR.stable_fd(source["hidden"], "source hidden") as (source_hidden, _), FULL_COMPARE._VALIDATOR.stable_fd(source["logits"], "source logits") as (source_logits, _), FULL_COMPARE._VALIDATOR.stable_fd(active["hidden"], "active hidden") as (active_hidden, _), FULL_COMPARE._VALIDATOR.stable_fd(active["logits"], "active logits") as (active_logits, _):
        for split_row in sorted(rows, key=lambda item: item["case_id"]):
            key = (split_row["case_id"], 0)
            left = source["rows"][key]
            right = active["rows"][key]
            if left.get("input_token_ids_sha256") != split_row.get("context_token_ids_sha256") or right.get("input_token_ids_sha256") != split_row.get("context_token_ids_sha256"):
                raise HoldoutError(f"input identity differs: {split_row['case_id']}")
            source_top = [item["token_id"] for item in left["topk"]]
            active_top = [item["token_id"] for item in right["topk"]]
            hidden = CAPTURE._stream_stats(CAPTURE._chunks(source_hidden, left["hidden"]["offset_bytes"], CAPTURE.HIDDEN_SIZE, source["chunk_elements"]), CAPTURE._chunks(active_hidden, right["hidden"]["offset_bytes"], CAPTURE.HIDDEN_SIZE, active["chunk_elements"]), CAPTURE.HIDDEN_SIZE)
            logits = CAPTURE._stream_stats(CAPTURE._chunks(source_logits, left["logits"]["offset_bytes"], CAPTURE.VOCAB_SIZE, source["chunk_elements"]), CAPTURE._chunks(active_logits, right["logits"]["offset_bytes"], CAPTURE.VOCAB_SIZE, active["chunk_elements"]), CAPTURE.VOCAB_SIZE)
            values = {
                "token_agreement_rate": float(left["greedy_token_id"] == right["greedy_token_id"]),
                "topk_overlap_rate_k10": len(set(source_top) & set(active_top)) / TOP_K,
                "logits_cosine": logits["cosine"], "logits_relative_l2": logits["relative_l2"],
                "hidden_cosine": hidden["cosine"], "hidden_relative_l2": hidden["relative_l2"],
                "hidden_max_abs": hidden["max_abs"],
                "bf16_top1_retained_in_aq4_top10_rate": float(left["greedy_token_id"] in active_top),
            }
            for name, value in values.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise HoldoutError(f"non-finite holdout metric: {split_row['case_id']}.{name}")
                numeric = float(value)
                if name in {"token_agreement_rate", "topk_overlap_rate_k10", "bf16_top1_retained_in_aq4_top10_rate"} and not 0.0 <= numeric <= 1.0:
                    raise HoldoutError(f"holdout metric outside [0,1]: {split_row['case_id']}.{name}")
                if name in {"logits_cosine", "hidden_cosine"} and not -1.0 <= numeric <= 1.0:
                    raise HoldoutError(f"holdout cosine outside [-1,1]: {split_row['case_id']}.{name}")
                if name not in {"token_agreement_rate", "topk_overlap_rate_k10", "bf16_top1_retained_in_aq4_top10_rate", "logits_cosine", "hidden_cosine"} and numeric < 0.0:
                    raise HoldoutError(f"holdout metric is negative: {split_row['case_id']}.{name}")
                if name in {"logits_relative_l2", "hidden_relative_l2"} and float(value) > 1.0:
                    raise HoldoutError(f"pathological relative-L2 > 1: {split_row['case_id']}.{name}")
                aggregate[name].append(float(value))
            metrics_rows.append({"case_id": split_row["case_id"], "case_sha256": split_row["case_sha256"], "fixture_sha256": split_row["fixture_sha256"], "prompt_token_ids_sha256": split_row["prompt_token_ids_sha256"], "context_token_ids_sha256": split_row["context_token_ids_sha256"], "prompt_tokens": split_row["prompt_tokens"], "context_tokens": split_row["context_tokens"], "baseline_mode": split_row["baseline_mode"], "prefill_requested_m": split_row["prefill_requested_m"], "resolved_m": split_row["resolved_m"], "step": 0, "row_count": 1, "greedy": {"source": left["greedy_token_id"], "active": right["greedy_token_id"], "exact": left["greedy_token_id"] == right["greedy_token_id"]}, "ordered_top10": {"source": source_top, "active": active_top, "exact": source_top == active_top, "overlap": values["topk_overlap_rate_k10"]}, "metrics": values})
    means = {name: (max(values) if PROTOCOL.METRICS[name]["role"] == "diagnostic_only" else sum(values) / len(values)) for name, values in aggregate.items()}
    return metrics_rows, means


def _decision(means: dict[str, float], receipt: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    bounds = receipt.get("derived_bounds")
    if not isinstance(bounds, dict) or set(bounds) != set(PROTOCOL.METRICS):
        raise HoldoutError("freeze receipt bounds are incomplete")
    checks: dict[str, Any] = {}
    go = True
    for name, spec in PROTOCOL.METRICS.items():
        item = bounds[name]
        observed = means[name]
        if spec["role"] == "diagnostic_only":
            checks[name] = {"observed": observed, "bound": None, "pass": True}
            continue
        bound = float(item["bound"])
        passed = observed >= bound if spec["direction"] == "higher" else observed <= bound
        checks[name] = {"observed": observed, "bound": bound, "pass": passed}
        go = go and passed
    return ("go" if go else "no_go"), checks


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    manifest, policy, receipt, _validated, shas = _freeze(args.split_root, args.freeze_receipt)
    actual = _actual_verified(args.actual_verified_receipt)
    source_manifest = args.source_artifact / "manifest.json"
    source_identity = _identity_file(source_manifest, "source artifact manifest")
    split_rows = _load_rows(args.split_root / "holdout-cases.jsonl", "holdout")
    source = _artifact_identity(args.source_artifact, "independent_source_full", split_rows, "holdout")
    source_cases_path = Path(source["manifest"]["cases"]["path"])
    if not source_cases_path.is_absolute():
        source_cases_path = (args.source_artifact / source_cases_path).resolve()
    source_cases_sha = _sha(source_cases_path, "source cases")
    source_case_value, _ = _read_json(source_cases_path, "source cases")
    if source_case_value.get("schema_version") != "ullm.qwen35_aq4_source_calibration_cases.v1" or len(source_case_value.get("cases", [])) != MAX_ROWS:
        raise HoldoutError("source cases schema/count differs")
    if {item.get("case_id") for item in source_case_value["cases"]} != {item["case_id"] for item in split_rows}:
        raise HoldoutError("source cases are not exactly the holdout cases")
    if source["manifest"]["cases"].get("sha256") != source_cases_sha:
        raise HoldoutError("source artifact cases SHA differs")
    capture_identity = _identity_file(args.capture_binary, "capture binary", args.expected_capture_binary_sha256)
    served_identity = _identity_file(args.served_model_manifest, "served model manifest", args.expected_served_model_manifest_sha256)
    if capture_identity["mode"] not in {"0555", "0755", "0775"}:
        raise HoldoutError("capture binary mode is not executable")
    expected = {**shas, "served_model_manifest_sha256": served_identity["sha256"], "package_manifest_sha256": _sha_value(args.expected_package_manifest_sha256, "package manifest SHA"), "worker_binary_sha256": _sha_value(args.expected_worker_binary_sha256, "worker binary SHA"), "capture_binary_sha256": capture_identity["sha256"], "build_sha256": _sha_value(args.expected_build_sha256, "build SHA"), "device_architecture": args.expected_device_architecture, "device_id": args.expected_device_id, "quantized_artifact_revision": args.expected_quantized_artifact_revision}
    if source["manifest"].get("runtime", {}).get("runtime", {}).get("selected_subset") not in (None, "holdout"):
        raise HoldoutError("source artifact selected subset is not holdout")
    command = [str(args.capture_binary.resolve()), "--served-model-manifest", str(args.served_model_manifest.resolve()), "--split-root", str(args.split_root.resolve()), "--source", str(args.source_artifact.resolve()), "--cases-file", str(source_cases_path.resolve()), "--output", str(args.active_output.resolve()), "--subset", "holdout", "--device-index", str(args.device_index), "--chunk-elements", str(args.chunk_elements), "--expected-split-manifest-sha256", shas["split_manifest_sha256"], "--expected-policy-sha256", shas["policy_sha256"], "--expected-calibration-cases-sha256", shas["calibration_cases_sha256"], "--expected-holdout-cases-sha256", shas["holdout_cases_sha256"], "--expected-served-model-manifest-sha256", expected["served_model_manifest_sha256"], "--expected-package-manifest-sha256", expected["package_manifest_sha256"], "--expected-worker-binary-sha256", expected["worker_binary_sha256"], "--expected-guard-sha256", _sha_value(args.expected_guard_sha256, "guard SHA"), "--expected-device-architecture", args.expected_device_architecture, "--expected-quantized-artifact-revision", args.expected_quantized_artifact_revision]
    if args.active_output.exists() or os.path.lexists(args.active_output):
        raise HoldoutError("active output already exists; overwrite is forbidden")
    plan = {"schema_version": PREFLIGHT_SCHEMA, "status": "ready_for_execute", "promotion_eligible": False, "subset": "holdout", "row_count": MAX_ROWS, "strata": {"count": 8, "rows_per_stratum": 3}, "split_manifest_sha256": shas["split_manifest_sha256"], "policy_sha256": shas["policy_sha256"], "calibration_cases_sha256": shas["calibration_cases_sha256"], "holdout_cases_sha256": shas["holdout_cases_sha256"], "freeze_receipt_sha256": shas["freeze_receipt_sha256"], "freeze_receipt_path": str(args.freeze_receipt.resolve()), "actual_verified_receipt": actual, "source_artifact": {"path": str(args.source_artifact.resolve()), "manifest_sha256": source_identity["sha256"], "cases_sha256": source_cases_sha, "identity": source["manifest"].get("identity")}, "identity": {**expected, "served_model_manifest_path": served_identity["path"], "source_identity": source["manifest"].get("identity")}, "execution_contract": {"one_process": True, "one_model_load": True, "gpu_parallelism": 1, "timeout_seconds": args.timeout_seconds, "capture_binary": capture_identity, "command": command, "command_sha256": hashlib.sha256(json.dumps(command, separators=(",", ":")).encode()).hexdigest()}, "paths": {"split_root": str(args.split_root.resolve()), "source_artifact": str(args.source_artifact.resolve()), "active_output": str(args.active_output.resolve()), "attempt_marker": str((args.output.parent / "attempt.json").resolve())}, "frozen_bounds": receipt["derived_bounds"]}
    _atomic_json(args.output, plan, "preflight")
    return {"status": "ok", "preflight": str(args.output), "preflight_sha256": _sha(args.output, "preflight")}


def _failure(path: Path, plan: dict[str, Any], preflight_sha: str, kind: str, detail: str, exit_code: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"schema_version": FAILURE_SCHEMA, "status": "holdout_failed", "holdout_status": "failed", "holdout_evaluations_remaining": 1, "attempt_consumed": True, "failure_kind": kind, "detail": detail, "preflight_sha256": preflight_sha, "split_manifest_sha256": plan["split_manifest_sha256"], "policy_sha256": plan["policy_sha256"], "calibration_cases_sha256": plan["calibration_cases_sha256"], "holdout_cases_sha256": plan["holdout_cases_sha256"], "freeze_receipt_sha256": plan["freeze_receipt_sha256"], "actual_verified_receipt": plan["actual_verified_receipt"], "identity": plan["identity"], "immutable": True}
    if exit_code is not None:
        value["exit_code"] = exit_code
    _atomic_json(path, value, "failure receipt")
    return value


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    plan, plan_raw = _read_json(args.preflight, "preflight")
    if plan.get("schema_version") != PREFLIGHT_SCHEMA or plan.get("status") != "ready_for_execute" or plan.get("subset") != "holdout" or plan.get("row_count") != MAX_ROWS:
        raise HoldoutError("preflight schema/status differs")
    preflight_sha = hashlib.sha256(plan_raw).hexdigest()
    attempt_path = Path(plan["paths"]["attempt_marker"])
    if os.path.lexists(attempt_path):
        raise HoldoutError("attempt marker already exists; retry is forbidden")
    marker = {"schema_version": ATTEMPT_SCHEMA, "status": "started", "preflight_sha256": preflight_sha, "started_unix": time.time(), "command_sha256": plan["execution_contract"]["command_sha256"]}
    _atomic_json(attempt_path, marker, "attempt marker")
    stdout_path = args.receipt_output.parent / "capture.stdout.log"
    stderr_path = args.receipt_output.parent / "capture.stderr.log"
    if os.path.lexists(stdout_path) or os.path.lexists(stderr_path):
        return _failure(args.receipt_output, plan, preflight_sha, "partial", "capture logs already exist")
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    out_fd = os.open(stdout_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    err_fd = os.open(stderr_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    env = os.environ.copy()
    env.update({"ROCR_VISIBLE_DEVICES": str(args.device_index), "HIP_VISIBLE_DEVICES": str(args.device_index), "CUDA_VISIBLE_DEVICES": str(args.device_index)})
    try:
        try:
            process = subprocess.Popen(plan["execution_contract"]["command"], stdin=subprocess.DEVNULL, stdout=out_fd, stderr=err_fd, env=env, start_new_session=True)
        except OSError as error:
            return _failure(args.receipt_output, plan, preflight_sha, "nonzero", f"capture could not start: {error}")
        try:
            process.wait(timeout=float(plan["execution_contract"]["timeout_seconds"]))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            return _failure(args.receipt_output, plan, preflight_sha, "timeout", "capture exceeded the frozen timeout")
    finally:
        os.close(out_fd)
        os.close(err_fd)
    if process.returncode != 0:
        kind = "oom" if process.returncode in (-signal.SIGKILL, 137) else "nonzero"
        return _failure(args.receipt_output, plan, preflight_sha, kind, "capture returned nonzero", process.returncode)
    try:
        actual_identity = _actual_verified(Path(plan["actual_verified_receipt"]["path"]))
    except Exception as error:
        return _failure(args.receipt_output, plan, preflight_sha, "partial", f"actual-verified receipt unavailable: {error}")
    if actual_identity != plan["actual_verified_receipt"]:
        return _failure(args.receipt_output, plan, preflight_sha, "partial", "actual-verified receipt changed after preflight")
    active_root = Path(plan["paths"]["active_output"])
    if not active_root.is_dir() or active_root.is_symlink():
        return _failure(args.receipt_output, plan, preflight_sha, "partial", "capture output directory is missing")
    try:
        holdout_rows = _load_rows(Path(plan["paths"]["split_root"]) / "holdout-cases.jsonl", "holdout")
        if _sha(Path(plan["paths"]["split_root"]) / "holdout-cases.jsonl", "holdout cases") != plan["holdout_cases_sha256"] or _sha(Path(plan["paths"]["split_root"]) / "calibration-cases.jsonl", "calibration cases") != plan["calibration_cases_sha256"] or _sha(Path(plan["paths"]["split_root"]) / "policy.json", "policy") != plan["policy_sha256"] or _sha(Path(plan["paths"]["split_root"]) / "split-manifest.json", "split manifest") != plan["split_manifest_sha256"]:
            raise HoldoutError("split identity changed after preflight")
        source_manifest_sha = _sha(Path(plan["paths"]["source_artifact"]) / "manifest.json", "source artifact manifest")
        if source_manifest_sha != plan["source_artifact"]["manifest_sha256"]:
            raise HoldoutError("source artifact changed after preflight")
        source = _artifact_identity(Path(plan["paths"]["source_artifact"]), "independent_source_full", holdout_rows, "holdout")
        source_cases_path = Path(source["manifest"]["cases"]["path"])
        if _sha(source_cases_path, "source cases") != plan["source_artifact"]["cases_sha256"]:
            raise HoldoutError("source cases changed after preflight")
        active = _artifact_identity(active_root, "aq4_target", holdout_rows, "holdout")
        _runtime_identity(active, plan["identity"])
        source_identity = _source_active_identity(source, active)
        rows, means = _compare(source, active, holdout_rows)
        freeze, freeze_raw = _read_json(Path(plan["freeze_receipt_path"]), "freeze receipt")
        if hashlib.sha256(freeze_raw).hexdigest() != plan["freeze_receipt_sha256"] or freeze.get("status") != "frozen_calibration_envelope" or freeze.get("holdout_status") != "not_started" or freeze.get("holdout_evaluations_remaining") != 1:
            raise HoldoutError("freeze receipt changed or is no longer executable")
        decision, checks = _decision(means, freeze)
    except Exception as error:
        return _failure(args.receipt_output, plan, preflight_sha, "partial", str(error))
    result = {"schema_version": RESULT_SCHEMA, "status": "holdout_result", "decision": decision, "holdout_status": "complete", "holdout_evaluations_remaining": 0, "holdout_evaluation_count": 1, "promotion_eligible": False, "attempt_consumed": True, "preflight_sha256": preflight_sha, "split_manifest_sha256": plan["split_manifest_sha256"], "policy_sha256": plan["policy_sha256"], "calibration_cases_sha256": plan["calibration_cases_sha256"], "holdout_cases_sha256": plan["holdout_cases_sha256"], "freeze_receipt_sha256": plan["freeze_receipt_sha256"], "actual_verified_receipt": plan["actual_verified_receipt"], "source_artifact_manifest_sha256": source["manifest_sha256"], "active_artifact_manifest_sha256": active["manifest_sha256"], "identity": {**plan["identity"], "source_identity": source_identity}, "execution_contract": {"one_process": True, "one_model_load": True, "gpu_parallelism": 1, "active_model_loads": active["manifest"].get("runtime", {}).get("model_loads")}, "metrics": {"row_count": len(rows), "means": means, "checks": checks, "rows": rows}, "immutable": True}
    _atomic_json(args.receipt_output, result, "holdout result")
    return {"status": "ok", "decision": decision, "receipt": str(args.receipt_output), "receipt_sha256": _sha(args.receipt_output, "holdout result")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--split-root", type=Path, required=True)
    pre.add_argument("--freeze-receipt", type=Path, required=True)
    pre.add_argument("--actual-verified-receipt", type=Path, required=True)
    pre.add_argument("--source-artifact", type=Path, required=True)
    pre.add_argument("--capture-binary", type=Path, required=True)
    pre.add_argument("--served-model-manifest", type=Path, required=True)
    pre.add_argument("--active-output", type=Path, required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--expected-served-model-manifest-sha256", required=True)
    pre.add_argument("--expected-package-manifest-sha256", required=True)
    pre.add_argument("--expected-worker-binary-sha256", required=True)
    pre.add_argument("--expected-capture-binary-sha256", required=True)
    pre.add_argument("--expected-build-sha256", required=True)
    pre.add_argument("--expected-guard-sha256", required=True)
    pre.add_argument("--expected-device-architecture", required=True)
    pre.add_argument("--expected-device-id")
    pre.add_argument("--expected-quantized-artifact-revision", required=True)
    pre.add_argument("--device-index", type=int, default=0)
    pre.add_argument("--chunk-elements", type=int, default=65536)
    pre.add_argument("--timeout-seconds", type=float, default=3600.0)
    exe = commands.add_parser("execute")
    exe.add_argument("--preflight", type=Path, required=True)
    exe.add_argument("--receipt-output", type=Path, required=True)
    exe.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        value = _preflight(args) if args.command == "preflight" else _execute(args)
        print(json.dumps(value, ensure_ascii=True, sort_keys=True))
        return 0
    except (HoldoutError, OSError, ValueError) as error:
        print(f"AQ4 P2 fidelity holdout runner failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
