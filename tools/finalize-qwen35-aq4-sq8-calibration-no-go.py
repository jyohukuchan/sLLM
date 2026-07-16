#!/usr/bin/env python3
"""Publish or validate immutable SQ8 calibration rejection evidence.

This artifact is deliberately not a metrics envelope, freeze receipt, or
holdout attempt ledger.  It records the pre-aggregation relative-L2 rejection
after independently revalidating the existing plan, capture, target, and
comparison evidence.  It never starts a model, GPU, service, or subprocess.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SCHEMA = "ullm.qwen35_aq4_sq8_fidelity_calibration_rejection_evidence.v1"
RECEIPT_NAME = "calibration-no-go-evidence.json"
SUMS_NAME = "SHA256SUMS"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = _load("sq8_no_go_protocol", "qwen35_aq4_sq8_fidelity_protocol.py")
VALIDATOR = _load("sq8_no_go_validator", "validate-qwen35-aq4-p2-full-calibration.py")
COMPARATOR = _load("sq8_no_go_comparator", "compare-qwen35-aq4-p2-calibration.py")


class NoGoEvidenceError(ValueError):
    pass


def _sha(path: Path, label: str) -> str:
    return VALIDATOR.sha256_file(path, label)


def _object(path: Path, label: str) -> dict[str, Any]:
    value, _ = PROTOCOL.load_json(path, label)
    if not isinstance(value, dict):
        raise NoGoEvidenceError(f"{label} must be an object")
    return value


def _exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise NoGoEvidenceError(f"{label} has unknown or missing fields")
    return value


def _hex(value: Any, label: str, *, forty: bool = False) -> str:
    pattern = HEX40 if forty else HEX64
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise NoGoEvidenceError(f"{label} is not a canonical digest")
    return value


def _ref(path: Path, label: str) -> dict[str, str]:
    resolved = path.resolve()
    if path != resolved or path.is_symlink() or not path.is_file():
        raise NoGoEvidenceError(f"{label} must be a canonical regular file")
    return {"path": str(resolved), "sha256": _sha(resolved, label)}


def _check_ref(value: Any, label: str) -> Path:
    ref = _exact(value, {"path", "sha256"}, f"{label} reference")
    digest = _hex(ref["sha256"], f"{label} SHA")
    path = Path(ref["path"])
    if not path.is_absolute() or path != path.resolve() or path.is_symlink() or not path.is_file():
        raise NoGoEvidenceError(f"{label} path differs")
    if _sha(path, label) != digest:
        raise NoGoEvidenceError(f"{label} SHA differs")
    return path


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise NoGoEvidenceError(f"{label} is not finite")
    number = float(value)
    if minimum is not None and number < minimum:
        raise NoGoEvidenceError(f"{label} is below its domain")
    return number


def _derive_observed(rows: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != PROTOCOL.MAX_ROWS:
        raise NoGoEvidenceError("comparison must contain exactly 24 rows")
    logits: list[float] = []
    hidden: list[float] = []
    greedy_mismatch = 0
    overlaps: list[int] = []
    for key in sorted(rows):
        row = rows[key]
        logits.append(_finite(row.get("logits", {}).get("relative_l2"), f"{key} logits relative-L2", minimum=0.0))
        hidden.append(_finite(row.get("hidden", {}).get("relative_l2"), f"{key} hidden relative-L2", minimum=0.0))
        greedy = row.get("greedy")
        if not isinstance(greedy, dict) or set(greedy) != {"source", "target"} or type(greedy["source"]) is not int or type(greedy["target"]) is not int:
            raise NoGoEvidenceError("comparison greedy identity differs")
        greedy_mismatch += int(greedy["source"] != greedy["target"])
        overlap = row.get("top_k_overlap")
        if type(overlap) is not int or not 0 <= overlap <= 10:
            raise NoGoEvidenceError("comparison top-k overlap differs")
        overlaps.append(overlap)
    ceiling = 1.0
    logits_over = sum(value > ceiling for value in logits)
    hidden_over = sum(value > ceiling for value in hidden)
    if logits_over + hidden_over == 0:
        raise NoGoEvidenceError("comparison does not trigger the frozen relative-L2 rejection")
    return {
        "row_count": PROTOCOL.MAX_ROWS,
        "nonfinite_rows": 0,
        "logits_relative_l2": {"count_above_ceiling": logits_over, "minimum": min(logits), "mean": sum(logits) / len(logits), "maximum": max(logits)},
        "hidden_relative_l2": {"count_above_ceiling": hidden_over, "minimum": min(hidden), "mean": sum(hidden) / len(hidden), "maximum": max(hidden)},
        "greedy_mismatch_rows": greedy_mismatch,
        "minimum_top_k_overlap": min(overlaps),
    }


def _validate_capture_history(capture: dict[str, Any], lineage: dict[str, Any]) -> None:
    expected = {"gate_script", "gate_log", "gate_exit_code", "capture_completed", "post_capture_failure", "staged_binary_receipt", "staged_binary", "offline_target_validation", "service_before", "stopped_stable2", "service_restore"}
    _exact(capture, expected, "capture history")
    if type(capture["gate_exit_code"]) is not int or capture["gate_exit_code"] != 2 or capture["capture_completed"] is not True or capture["post_capture_failure"] != "validator_cli_missing_required_artifact_argument":
        raise NoGoEvidenceError("capture gate terminal state differs")
    paths = {name: _check_ref(capture[name], name.replace("_", " ")) for name in expected - {"gate_exit_code", "capture_completed", "post_capture_failure"}}
    gate = paths["gate_log"].read_text(encoding="utf-8")
    required_gate = (
        "capture_finish=",
        "manifest.json: OK",
        "rows.jsonl: OK",
        "vectors/hidden.f32le: OK",
        "vectors/logits.f32le: OK",
        "error: the following arguments are required: --artifact",
        "gate_finish=",
        "rc=2",
    )
    if any(token not in gate for token in required_gate):
        raise NoGoEvidenceError("capture gate log does not prove the post-capture CLI failure")
    script = paths["gate_script"].read_text(encoding="utf-8")
    capture_commit = _hex(lineage["capture_commit"], "capture commit", forty=True)
    capture_tree = _hex(lineage["capture_tree"], "capture tree", forty=True)
    if f"rev-parse HEAD)\" = {capture_commit}" not in script or f"rev-parse HEAD^{{tree}})\" = {capture_tree}" not in script:
        raise NoGoEvidenceError("capture script commit/tree binding differs")
    staged = _object(paths["staged_binary_receipt"], "staged binary receipt")
    _exact(staged, {"schema_version", "status", "execution_contract", "source", "staged"}, "staged binary receipt")
    staged_value = staged["staged"]
    if staged.get("schema_version") != "ullm.aq4_fidelity_capture_staged_binary.v1" or staged.get("status") != "ready" or not isinstance(staged_value, dict) or staged_value.get("path") != str(paths["staged_binary"]) or staged_value.get("sha256") != capture["staged_binary"]["sha256"] or staged_value.get("mode") != "0555" or staged_value.get("nlink") != 1:
        raise NoGoEvidenceError("staged capture binary identity differs")
    info = paths["staged_binary"].stat()
    if stat.S_IMODE(info.st_mode) != 0o555 or info.st_nlink != 1 or info.st_size != staged_value.get("bytes"):
        raise NoGoEvidenceError("staged capture binary filesystem identity differs")
    offline = _object(paths["offline_target_validation"], "offline target validation")
    report = offline.get("result", {}).get("report")
    if offline.get("command") != "validate-target" or offline.get("status") != "ok" or offline.get("result", {}).get("validator_modified_artifact") is not False or not isinstance(report, dict) or report.get("status") != "valid" or report.get("row_count") != PROTOCOL.MAX_ROWS or report.get("nonfinite_rows") != 0:
        raise NoGoEvidenceError("offline target validation history differs")
    stopped = _object(paths["stopped_stable2"], "stopped stable2")
    observations = stopped.get("observations")
    if stopped.get("schema_version") != "ullm.aq4_fidelity_stopped_stable2.v1" or stopped.get("status") != "passed" or not isinstance(observations, list) or len(observations) != 2:
        raise NoGoEvidenceError("stopped stable2 history differs")
    for observation in observations:
        owners = observation.get("owners", {})
        service = observation.get("service", {})
        if owners.get("worker_pids") != [] or owners.get("amd_pids") != [] or owners.get("kfd_pids") != [] or service.get("active") is not False or service.get("running") is not False:
            raise NoGoEvidenceError("stopped stable2 was not GPU-exclusive")
    restore = _object(paths["service_restore"], "service restore")
    service = restore.get("service")
    owners = restore.get("owners")
    worker = service.get("worker_pid") if isinstance(service, dict) else None
    if restore.get("schema_version") != "ullm.aq4_fidelity_service_restore.v1" or restore.get("status") != "passed" or not isinstance(service, dict) or service.get("active") is not True or service.get("running") is not True or service.get("healthy") is not True or service.get("nrestarts") != 0 or not isinstance(worker, int) or worker <= 0 or not isinstance(owners, dict) or owners.get("worker_pids") != [worker] or owners.get("amd_pids") != [worker] or owners.get("kfd_pids") != [worker]:
        raise NoGoEvidenceError("service restore evidence differs")
    before = _object(paths["service_before"], "service before")
    before_service = before.get("service")
    if before.get("schema_version") != "ullm.aq4_fidelity_service_snapshot.v1" or before.get("status") != "ready" or not isinstance(before_service, dict) or before_service.get("active") is not True or before_service.get("running") is not True or before_service.get("healthy") is not True or before_service.get("nrestarts") != 0:
        raise NoGoEvidenceError("pre-capture service snapshot differs")


def _validate_receipt(value: dict[str, Any]) -> dict[str, Any]:
    keys = {"schema_version", "status", "reason", "policy", "observed", "state", "bindings", "capture", "lineage"}
    _exact(value, keys, "calibration rejection evidence")
    if value.get("schema_version") != SCHEMA or value.get("status") != "calibration_rejected_no_go" or value.get("reason") != "relative_l2_pathological_rejection_before_aggregation":
        raise NoGoEvidenceError("calibration rejection state differs")
    policy = _exact(value.get("policy"), {"path", "sha256", "ceiling", "action"}, "rejection policy")
    policy_path = Path(policy["path"])
    if policy.get("ceiling") != 1.0 or policy.get("action") != "reject any observed relative-L2 > 1 before aggregation" or _sha(policy_path, "policy") != policy.get("sha256"):
        raise NoGoEvidenceError("relative-L2 rejection policy differs")
    policy_value = _object(policy_path, "policy")
    if policy_value.get("relative_l2_rejection") != {"ceiling": 1.0, "action": policy["action"], "reason": "relative-L2 above 100 percent is a predeclared pathological-drift rejection; this structural check is distinct from raw hidden max-abs, which has no natural scale"}:
        raise NoGoEvidenceError("policy file rejection contract differs")
    state = _exact(value.get("state"), {"metrics_published", "freeze_published", "holdout_status", "holdout_evaluations_remaining", "retry_permitted", "holdout_executed"}, "terminal state")
    expected_state = {"metrics_published": False, "freeze_published": False, "holdout_status": "not_started", "holdout_evaluations_remaining": 1, "retry_permitted": False, "holdout_executed": False}
    if state != expected_state:
        raise NoGoEvidenceError("terminal holdout state differs")
    bindings = _exact(value.get("bindings"), {"plan", "actual_receipt", "source_artifact", "target_artifact", "comparison"}, "evidence bindings")
    plan_path = _check_ref(bindings["plan"], "plan")
    plan, _ = PROTOCOL._check_plan(plan_path)
    if plan.get("status") != "ready_for_calibration" or plan.get("preflight_only") is not False or plan.get("holdout_state") != {"status": "not_started", "evaluations_remaining": 1, "retry_permitted": False}:
        raise NoGoEvidenceError("plan terminal state differs")
    actual_path = _check_ref(bindings["actual_receipt"], "actual receipt")
    if str(actual_path) != plan["identity"]["sq8_receipt_path"] or bindings["actual_receipt"]["sha256"] != plan["identity"]["sq8_receipt_sha256"]:
        raise NoGoEvidenceError("actual receipt/plan binding differs")
    evidence_refs: dict[str, dict[str, str]] = {}
    for name in ("source_artifact", "target_artifact", "comparison"):
        artifact = _exact(bindings[name], {"root", "manifest_sha256", "sha256sums_sha256"}, f"{name} binding")
        root = Path(artifact["root"])
        if not root.is_absolute() or root != root.resolve() or not root.is_dir() or root.is_symlink():
            raise NoGoEvidenceError(f"{name} root differs")
        if _sha(root / "manifest.json", f"{name} manifest") != artifact["manifest_sha256"] or _sha(root / SUMS_NAME, f"{name} SHA256SUMS") != artifact["sha256sums_sha256"]:
            raise NoGoEvidenceError(f"{name} hashes differ")
        evidence_refs[name] = {"root": str(root), "manifest_sha256": artifact["manifest_sha256"]}
    for name in ("source_artifact", "target_artifact"):
        report = VALIDATOR.validate(Path(bindings[name]["root"]))
        if report.get("status") != "valid" or report.get("row_count") != PROTOCOL.MAX_ROWS or report.get("nonfinite_rows") != 0:
            raise NoGoEvidenceError(f"{name} validation differs")
    rows = PROTOCOL._metrics_evidence(evidence_refs, plan["identity"])
    observed = _derive_observed(rows)
    if value.get("observed") != observed:
        raise NoGoEvidenceError("recorded rejection observations differ")
    lineage = _exact(value.get("lineage"), {"capture_commit", "capture_tree", "validator_fix_commit", "validator_fix_tree", "tensor_authority_commit", "tensor_authority_tree", "finalizer_commit", "finalizer_tree", "finalizer_tool"}, "commit lineage")
    for name in ("capture_commit", "capture_tree", "validator_fix_commit", "validator_fix_tree", "tensor_authority_commit", "tensor_authority_tree", "finalizer_commit", "finalizer_tree"):
        _hex(lineage[name], name.replace("_", " "), forty=True)
    _check_ref(lineage["finalizer_tool"], "finalizer tool")
    _validate_capture_history(value.get("capture"), lineage)
    return {"status": "valid_no_go", "observed": observed, "holdout_status": "not_started", "holdout_evaluations_remaining": 1}


def _artifact_binding(root: Path, label: str) -> dict[str, str]:
    resolved = root.resolve()
    if root != resolved or root.is_symlink() or not root.is_dir():
        raise NoGoEvidenceError(f"{label} root differs")
    return {"root": str(resolved), "manifest_sha256": _sha(resolved / "manifest.json", f"{label} manifest"), "sha256sums_sha256": _sha(resolved / SUMS_NAME, f"{label} SHA256SUMS")}


def build(args: argparse.Namespace) -> dict[str, Any]:
    plan, _ = PROTOCOL._check_plan(args.plan)
    evidence_refs = {
        "source_artifact": {key: value for key, value in _artifact_binding(args.source, "source artifact").items() if key != "sha256sums_sha256"},
        "target_artifact": {key: value for key, value in _artifact_binding(args.target, "target artifact").items() if key != "sha256sums_sha256"},
        "comparison": {key: value for key, value in _artifact_binding(args.comparison, "comparison").items() if key != "sha256sums_sha256"},
    }
    rows = PROTOCOL._metrics_evidence(evidence_refs, plan["identity"])
    policy_path = Path(plan["identity"]["policy_path"])
    lineage = {
        "capture_commit": args.capture_commit,
        "capture_tree": args.capture_tree,
        "validator_fix_commit": args.validator_fix_commit,
        "validator_fix_tree": args.validator_fix_tree,
        "tensor_authority_commit": args.tensor_authority_commit,
        "tensor_authority_tree": args.tensor_authority_tree,
        "finalizer_commit": args.finalizer_commit,
        "finalizer_tree": args.finalizer_tree,
        "finalizer_tool": _ref(Path(__file__).resolve(), "finalizer tool"),
    }
    capture = {
        "gate_script": _ref(args.gate_script, "capture gate script"),
        "gate_log": _ref(args.gate_log, "capture gate log"),
        "gate_exit_code": 2,
        "capture_completed": True,
        "post_capture_failure": "validator_cli_missing_required_artifact_argument",
        "staged_binary_receipt": _ref(args.staged_binary_receipt, "staged binary receipt"),
        "staged_binary": _ref(args.staged_binary, "staged binary"),
        "offline_target_validation": _ref(args.offline_target_validation, "offline target validation"),
        "service_before": _ref(args.service_before, "service before"),
        "stopped_stable2": _ref(args.stopped_stable2, "stopped stable2"),
        "service_restore": _ref(args.service_restore, "service restore"),
    }
    value = {
        "schema_version": SCHEMA,
        "status": "calibration_rejected_no_go",
        "reason": "relative_l2_pathological_rejection_before_aggregation",
        "policy": {"path": str(policy_path.resolve()), "sha256": _sha(policy_path, "policy"), "ceiling": 1.0, "action": "reject any observed relative-L2 > 1 before aggregation"},
        "observed": _derive_observed(rows),
        "state": {"metrics_published": False, "freeze_published": False, "holdout_status": "not_started", "holdout_evaluations_remaining": 1, "retry_permitted": False, "holdout_executed": False},
        "bindings": {
            "plan": _ref(args.plan, "plan"),
            "actual_receipt": _ref(Path(plan["identity"]["sq8_receipt_path"]), "actual receipt"),
            "source_artifact": _artifact_binding(args.source, "source artifact"),
            "target_artifact": _artifact_binding(args.target, "target artifact"),
            "comparison": _artifact_binding(args.comparison, "comparison"),
        },
        "capture": capture,
        "lineage": lineage,
    }
    _validate_receipt(value)
    return value


def publish(value: dict[str, Any], output: Path) -> dict[str, Any]:
    parent = output.resolve().parent
    if output != output.resolve() or output.exists() or output.is_symlink():
        raise NoGoEvidenceError("output must be a missing canonical path")
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=parent))
    try:
        receipt = temporary / RECEIPT_NAME
        raw = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode("ascii")
        fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
        try:
            os.write(fd, raw)
            os.fsync(fd)
        finally:
            os.close(fd)
        receipt_sha = hashlib.sha256(raw).hexdigest()
        sums = temporary / SUMS_NAME
        line = f"{receipt_sha}  {RECEIPT_NAME}\n".encode("ascii")
        fd = os.open(sums, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(temporary, 0o555)
        COMPARATOR.publish_noreplace(temporary, output)
        return {"status": "published_no_go", "receipt_sha256": receipt_sha, "sha256sums_sha256": _sha(output / SUMS_NAME, "NO-GO SHA256SUMS")}
    finally:
        if temporary.exists():
            os.chmod(temporary, 0o755)
            shutil.rmtree(temporary, ignore_errors=True)


def validate_package(root: Path) -> dict[str, Any]:
    if root != root.resolve() or root.is_symlink() or not root.is_dir() or stat.S_IMODE(root.stat().st_mode) != 0o555:
        raise NoGoEvidenceError("NO-GO package directory identity differs")
    files = {path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()}
    if files != {RECEIPT_NAME, SUMS_NAME} or any(path.is_symlink() or not path.is_file() for path in root.iterdir()):
        raise NoGoEvidenceError("NO-GO package inventory differs")
    for name in files:
        info = (root / name).stat()
        if stat.S_IMODE(info.st_mode) != 0o444 or info.st_nlink != 1:
            raise NoGoEvidenceError("NO-GO package file identity differs")
    receipt_sha = _sha(root / RECEIPT_NAME, "NO-GO receipt")
    expected_sums = f"{receipt_sha}  {RECEIPT_NAME}\n"
    if (root / SUMS_NAME).read_text(encoding="ascii") != expected_sums:
        raise NoGoEvidenceError("NO-GO SHA256SUMS differs")
    result = _validate_receipt(_object(root / RECEIPT_NAME, "NO-GO receipt"))
    return {**result, "receipt_sha256": receipt_sha, "sha256sums_sha256": _sha(root / SUMS_NAME, "NO-GO SHA256SUMS")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    finalize = sub.add_parser("finalize")
    for name in ("plan", "source", "target", "comparison", "gate_script", "gate_log", "staged_binary_receipt", "staged_binary", "offline_target_validation", "service_before", "stopped_stable2", "service_restore"):
        finalize.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    for name in ("capture_commit", "capture_tree", "validator_fix_commit", "validator_fix_tree", "tensor_authority_commit", "tensor_authority_tree", "finalizer_commit", "finalizer_tree"):
        finalize.add_argument(f"--{name.replace('_', '-')}", required=True)
    finalize.add_argument("--output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = publish(build(args), args.output) if args.command == "finalize" else validate_package(args.evidence)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (NoGoEvidenceError, PROTOCOL.ProtocolError, VALIDATOR.ValidationError, OSError, ValueError, KeyError) as error:
        print(f"SQ8 calibration NO-GO evidence failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
