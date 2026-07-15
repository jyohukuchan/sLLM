#!/usr/bin/env python3
"""Collect a read-only profile quiet window and prepare one exact operator command."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
P2 = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2"
P3 = ROOT / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3"
MAINTENANCE = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
SOURCE = Path(__file__).resolve()
PROFILE_READY_ROOT = P2 / "resident-one-case-smoke-profile-ready-v6"
PROFILE_READY = PROFILE_READY_ROOT / "ready-binding.json"
QUIET_ROOT = P2 / "resident-one-case-smoke-profile-quiet-window-v12"
OPERATOR_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v7"
MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v6"
OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v7"
ACTUAL_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v7"
PREVIOUS_OPERATOR_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v6"
PYTHON = Path("/usr/bin/python3.12")
QUIET_SCHEMA = "ullm.aq4_p3_profile_quiet_window.v12"
OPERATOR_SCHEMA = "ullm.aq4_p3_profile_operator_command.v7"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_INTERVAL = 5.0
DEFAULT_MAXIMUM = 900.0
DEFAULT_MINIMUM_SPAN = 130.0
DEFAULT_REQUIRED_SAMPLES = 27

# Filled only after the fresh profile-ready-v6 commit is final.  The collector
# rejects any other ready artifact rather than silently following a moving path.
READY_ARTIFACT_COMMIT = "ff15b75ceed5e7b7eabe376e27859106694c285f"
READY_BINDING_SHA256 = "0fca1b6d5b561b582bd0e59c33d88e558be785faa026a058a1a8d3e9d3b4e54e"
READY_SHA256SUMS_SHA256 = "49c5535e617db3598029e0968e253a8771ad3489bc683cf541d11c54d13a1ccc"


class OperatorError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode("ascii") + b"\n"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes(), parse_constant=lambda item: (_ for _ in ()).throw(OperatorError(f"non-finite {label}: {item}")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OperatorError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise OperatorError(f"{label} root is not an object")
    return value


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=ROOT, text=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or completed.stderr:
        raise OperatorError(f"Git command failed: {' '.join(args)}")
    return completed.stdout.strip()


def verify_sums(root: Path) -> dict[str, Any]:
    metadata = root.lstat()
    if root.is_symlink() or not root.is_dir() or stat.S_IMODE(metadata.st_mode) != 0o555:
        raise OperatorError(f"sealed root differs: {root}")
    sums = root / "SHA256SUMS"
    if sums.is_symlink() or not sums.is_file() or stat.S_IMODE(sums.lstat().st_mode) != 0o444 or sums.lstat().st_nlink != 1:
        raise OperatorError(f"SHA256SUMS contract differs: {root}")
    declared: dict[str, str] = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        if SHA_RE.fullmatch(digest) is None or name in declared or "/" in name:
            raise OperatorError(f"SHA256SUMS syntax differs: {root}")
        declared[name] = digest
    expected = {item.name for item in root.iterdir()} - {"SHA256SUMS"}
    if set(declared) != expected:
        raise OperatorError(f"SHA256SUMS coverage differs: {root}")
    members: dict[str, Any] = {}
    for name in sorted(expected):
        path = root / name
        child = path.lstat()
        if not stat.S_ISREG(child.st_mode) or child.st_nlink != 1 or stat.S_IMODE(child.st_mode) != 0o444 or sha_file(path) != declared[name]:
            raise OperatorError(f"sealed member differs: {path}")
        members[name] = {"path": str(path), "sha256": declared[name], "mode": "0444", "nlink": 1, "size": child.st_size}
    return {"root": str(root), "mode": "0555", "sha256sums_sha256": sha_file(sums), "members": members}


def ready_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    if not READY_ARTIFACT_COMMIT or not READY_BINDING_SHA256 or not READY_SHA256SUMS_SHA256:
        raise OperatorError("profile-ready-v6 authority pins are not finalized")
    inventory = verify_sums(PROFILE_READY_ROOT)
    if sha_file(PROFILE_READY) != READY_BINDING_SHA256 or inventory["sha256sums_sha256"] != READY_SHA256SUMS_SHA256:
        raise OperatorError("profile-ready-v6 hashes differ")
    path = str(PROFILE_READY.relative_to(ROOT))
    if git("rev-parse", f"{READY_ARTIFACT_COMMIT}:{path}") != git("hash-object", str(PROFILE_READY)):
        raise OperatorError("profile-ready-v6 Git blob differs")
    return load(PROFILE_READY, "profile ready binding"), inventory


def fresh_paths(ready: dict[str, Any]) -> list[Path]:
    binding = ready.get("launcher_binding")
    if not isinstance(binding, dict):
        raise OperatorError("ready launcher binding is missing")
    profile = binding.get("profile_diagnostic", {}).get("output", {})
    paths = [
        Path(str(binding.get("runner_output", ""))),
        Path(str(binding.get("evidence_output", ""))),
        MAINTENANCE_EVIDENCE,
        Path(str(profile.get("directory", ""))),
        Path(str(profile.get("artifact", ""))),
        Path(str(profile.get("directory", ""))) / "rocprof.stdout",
        Path(str(profile.get("directory", ""))) / "rocprof.stderr",
        OPERATOR_RESULT,
        ACTUAL_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9 or any(not path.is_absolute() or ".." in path.parts for path in paths):
        raise OperatorError("fresh output set differs")
    return paths


def root_set() -> list[Path]:
    return [
        ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v6",
        P2 / "resident-one-case-smoke-execute-binding-v6",
        P2 / "resident-one-case-smoke-ready-v6",
        P2 / "resident-one-case-smoke-ready-dry-run-v6",
        PROFILE_READY_ROOT,
        P2 / "resident-one-case-smoke-profile-ready-dry-run-v6",
    ]


def trusted_source_snapshot(ready: dict[str, Any]) -> list[dict[str, Any]]:
    trust = load(PROFILE_READY_ROOT / "harness-trust.json", "profile harness trust")
    qa = load(PROFILE_READY_ROOT / "qa-attestation.json", "profile QA attestation")
    binding = ready["launcher_binding"]
    specifications: list[tuple[Path, str, str]] = [
        (Path(trust["path"]), trust["sha256"], trust["commit"]),
        (ROOT / "tools/launch-aq4-p2-resident-smoke.py", qa["launcher"]["sha256"], qa["launcher"]["commit"]),
        (Path(ready["profile_diagnostic"]["capture_tool"]["path"]), qa["capture_tool"]["sha256"], qa["capture_tool"]["commit"]),
        (Path(binding["R"]["path"]), binding["R"]["sha256"], binding["R"]["commit"]),
        (Path(binding["validator"]["path"]), binding["validator"]["sha256"], binding["validator"]["commit"]),
        (Path(binding["resident"]["path"]), binding["resident"]["sha256"], binding["resident"]["commit"]),
    ]
    for suite in qa["automated_tests"]["suites"]:
        for item in suite["files"]:
            path = ROOT / item["path"]
            specifications.append((path, sha_file(path), item["source_commit"]))
    specifications.append((SOURCE, sha_file(SOURCE), git("rev-parse", "HEAD")))
    unique: dict[str, tuple[Path, str, str]] = {}
    for path, expected_sha, commit in specifications:
        key = str(path)
        if key in unique and unique[key][1:] != (expected_sha, commit):
            raise OperatorError(f"trusted source authority conflicts: {path}")
        unique[key] = (path, expected_sha, commit)
    records: list[dict[str, Any]] = []
    for path, expected_sha, commit in unique.values():
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1 or sha_file(path) != expected_sha:
            raise OperatorError(f"trusted source differs: {path}")
        relative = str(path.relative_to(ROOT))
        current_blob = git("hash-object", str(path))
        if git("rev-parse", f"{commit}:{relative}") != current_blob:
            raise OperatorError(f"trusted source Git authority differs: {path}")
        records.append({"path": str(path), "sha256": expected_sha, "commit": commit, "git_blob": current_blob, "identity": [metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]})
    return sorted(records, key=lambda item: item["path"])


def relevant_snapshot(ready: dict[str, Any]) -> dict[str, Any]:
    roots = [verify_sums(root) for root in root_set()]
    records: list[dict[str, Any]] = []
    for root in roots:
        for member in root["members"].values():
            path = Path(member["path"])
            metadata = path.lstat()
            records.append({"path": str(path), "sha256": member["sha256"], "identity": [metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]})
        sums = Path(root["root"]) / "SHA256SUMS"
        metadata = sums.lstat()
        records.append({"path": str(sums), "sha256": root["sha256sums_sha256"], "identity": [metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]})
    sources = trusted_source_snapshot(ready)
    records.extend(sources)
    absent = {str(path): not path.exists() and not path.is_symlink() for path in fresh_paths(ready)}
    records.sort(key=lambda item: item["path"])
    return {"root_count": len(roots), "file_count": len(records), "trusted_source_count": len(sources), "byte_aggregate_sha256": sha_bytes(canonical([{"path": item["path"], "sha256": item["sha256"]} for item in records])), "identity_aggregate_sha256": sha_bytes(canonical(records)), "fresh_absence": absent, "all_required_absent": all(absent.values())}


def load_maintenance() -> Any:
    spec = importlib.util.spec_from_file_location("aq4_profile_operator_maintenance", MAINTENANCE)
    if spec is None or spec.loader is None:
        raise OperatorError("maintenance import failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def targeted_processes() -> list[dict[str, Any]]:
    own = {os.getpid(), os.getppid()}
    markers = ("--mode\x00execute", "rocprofv3", "capture-aq4-p3-diagnostic-profile.py", "--confirm-one-case")
    found: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal() or int(entry.name) in own:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()[:65536]
        except OSError:
            continue
        text = raw.decode("utf-8", "replace")
        if any(marker in text for marker in markers):
            found.append({"pid": int(entry.name), "cmdline_sha256": sha_bytes(raw), "matched": [marker for marker in markers if marker in text]})
    return sorted(found, key=lambda item: item["pid"])


def capture_snapshot(ready: dict[str, Any], maintenance: Any | None = None) -> dict[str, Any]:
    maintenance = load_maintenance() if maintenance is None else maintenance
    running = maintenance.capture_running(maintenance.default_dependencies())
    lock_metadata = Path(running["lock"]["path"]).lstat()
    relevant = relevant_snapshot(ready)
    formal = running["health"]["formal"]
    stable = {
        "head": git("rev-parse", "HEAD"),
        "tree": git("write-tree"),
        "service": running["service"],
        "worker": running["worker"],
        "gpu": running["gpu"],
        "owners": {"amd_smi": running["owners"]["amd_smi"], "kfd": running["owners"]["kfd"]},
        "lock": {"path": running["lock"]["path"], "busy": running["lock"]["busy"], "identity": [lock_metadata.st_dev, lock_metadata.st_ino, lock_metadata.st_mode, lock_metadata.st_nlink, lock_metadata.st_size]},
        "hashes": running["hashes"],
        "formal_health_sha256": sha_bytes(canonical({key: formal[key] for key in ("container", "curl", "docker", "endpoints", "process_counts", "secret_material_recorded")})),
        "relevant": relevant,
    }
    processes = targeted_processes()
    clean = relevant["all_required_absent"] and not processes and running["owners"]["amd_smi"] == [running["worker"]["pid"]] and running["owners"]["kfd"] == [running["worker"]["pid"]] and formal.get("secret_material_recorded") is False
    return {"captured_unix_ns": time.time_ns(), "captured_monotonic_ns": time.monotonic_ns(), **stable, "targeted_processes": processes, "blocking_identity_sha256": sha_bytes(canonical(stable)), "clean": clean}


def monitor(ready: dict[str, Any], capture: Callable[[dict[str, Any]], dict[str, Any]], sleep: Callable[[float], None], *, interval: float, maximum: float, minimum_span: float, required: int) -> dict[str, Any]:
    if interval < 0.0 or maximum <= 0.0 or minimum_span < 0.0 or required < 2:
        raise OperatorError("quiet window policy is invalid")
    started_mono = time.monotonic_ns(); started_unix = time.time_ns()
    samples: list[dict[str, Any]] = []; streak: list[dict[str, Any]] = []; resets: list[dict[str, Any]] = []
    while (time.monotonic_ns() - started_mono) / 1e9 <= maximum:
        sample = capture(ready); samples.append(sample)
        if not sample.get("clean"):
            if streak:
                resets.append({"sample_index": len(samples) - 1, "reason": "sample_not_clean"})
            streak = []
        elif streak and sample["blocking_identity_sha256"] != streak[-1]["blocking_identity_sha256"]:
            resets.append({"sample_index": len(samples) - 1, "reason": "blocking_identity_changed"}); streak = [sample]
        else:
            streak.append(sample)
        span = 0.0 if len(streak) < 2 else (streak[-1]["captured_monotonic_ns"] - streak[0]["captured_monotonic_ns"]) / 1e9
        if len(streak) >= required and span >= minimum_span:
            confirmation = capture(ready)
            passed = confirmation.get("clean") is True and confirmation.get("blocking_identity_sha256") == streak[-1]["blocking_identity_sha256"]
            return {"samples": samples, "streak": streak, "resets": resets, "confirmation": confirmation, "passed": passed, "span_seconds": span, "started_monotonic_ns": started_mono, "started_unix_ns": started_unix, "finished_monotonic_ns": time.monotonic_ns()}
        sleep(interval)
    return {"samples": samples, "streak": streak, "resets": resets, "confirmation": None, "passed": False, "span_seconds": 0.0, "started_monotonic_ns": started_mono, "started_unix_ns": started_unix, "finished_monotonic_ns": time.monotonic_ns()}


def write_sealed(root: Path, name: str, value: dict[str, Any]) -> None:
    if root.exists() or root.is_symlink():
        raise OperatorError(f"output already exists: {root}")
    root.mkdir(parents=True, mode=0o755)
    raw = pretty(value); (root / name).write_bytes(raw)
    (root / "SHA256SUMS").write_text(f"{sha_bytes(raw)}  {name}\n", encoding="ascii")
    os.chmod(root / name, 0o444); os.chmod(root / "SHA256SUMS", 0o444); os.chmod(root, 0o555)


def collect_quiet(output: Path = QUIET_ROOT, *, interval: float = DEFAULT_INTERVAL, maximum: float = DEFAULT_MAXIMUM, minimum_span: float = DEFAULT_MINIMUM_SPAN, required: int = DEFAULT_REQUIRED_SAMPLES) -> dict[str, Any]:
    ready, inventory = ready_authority()
    result = monitor(ready, capture_snapshot, time.sleep, interval=interval, maximum=maximum, minimum_span=minimum_span, required=required)
    value = {"schema_version": QUIET_SCHEMA, "status": "go" if result["passed"] and not result["resets"] else "no_go", "decision": "GO" if result["passed"] and not result["resets"] else "NO_GO", "captured_unix_ns": time.time_ns(), "policy": {"interval_seconds": interval, "maximum_monitoring_seconds": maximum, "minimum_sample_span_seconds": minimum_span, "required_consecutive_clean_samples": required, "reset_count_required": 0}, "binding": {"ready_artifact_commit": READY_ARTIFACT_COMMIT, "ready_binding_sha256": READY_BINDING_SHA256, "ready_inventory": inventory}, "samples": result["samples"], "resets": result["resets"], "confirmation": result["confirmation"], "summary": {"sample_count": len(result["samples"]), "final_streak_samples": len(result["streak"]), "final_streak_span_seconds": result["span_seconds"], "reset_count": len(result["resets"]), "confirmation_passed": result["passed"], "fresh_outputs_absent": bool(result["streak"] and result["streak"][-1]["relevant"]["all_required_absent"])}, "timing": {"monitor_started_unix_ns": result["started_unix_ns"], "monitor_started_monotonic_ns": result["started_monotonic_ns"], "monitor_finished_monotonic_ns": result["finished_monotonic_ns"]}, "read_only": True, "actual_executed": False, "gpu_command_executed": False, "service_touched": False, "secret_material_recorded": False}
    write_sealed(output, "quiet-window.json", value)
    validate_quiet(output)
    return value


def validate_quiet(root: Path = QUIET_ROOT) -> dict[str, Any]:
    inventory = verify_sums(root); value = load(root / "quiet-window.json", "quiet window")
    if value.get("schema_version") != QUIET_SCHEMA or value.get("status") != "go" or value.get("decision") != "GO" or value.get("resets") != [] or value.get("read_only") is not True or value.get("actual_executed") is not False or value.get("gpu_command_executed") is not False or value.get("service_touched") is not False or value.get("secret_material_recorded") is not False:
        raise OperatorError("quiet window decision/safety differs")
    summary = value.get("summary", {})
    policy = value.get("policy", {})
    if summary.get("final_streak_samples", 0) < policy.get("required_consecutive_clean_samples", DEFAULT_REQUIRED_SAMPLES) or summary.get("final_streak_span_seconds", 0.0) < policy.get("minimum_sample_span_seconds", DEFAULT_MINIMUM_SPAN) or summary.get("confirmation_passed") is not True or summary.get("fresh_outputs_absent") is not True:
        raise OperatorError("quiet window final streak differs")
    return {"value": value, "inventory": inventory}


def actual_argv() -> list[str]:
    return [str(PYTHON), str(MAINTENANCE), "--mode", "execute", "--profile-diagnostic", "--ready-artifact", str(PROFILE_READY), "--evidence-output", str(MAINTENANCE_EVIDENCE), "--confirm-one-case"]


def prepare_operator(output: Path = OPERATOR_ROOT) -> dict[str, Any]:
    ready, ready_inventory = ready_authority(); quiet = validate_quiet(QUIET_ROOT); previous = verify_sums(PREVIOUS_OPERATOR_ROOT)
    fresh = fresh_paths(ready)
    if any(path.exists() or path.is_symlink() for path in fresh):
        raise OperatorError("operator fresh outputs are not absent")
    argv = actual_argv()
    manifest: dict[str, Any] = {"schema_version": OPERATOR_SCHEMA, "status": "audited_ready_for_single_explicit_profile_diagnostic", "argv": argv, "command_sha256": sha_bytes(canonical(argv)), "authorization": {"maximum_invocations": 1, "explicit_confirmation_flag_count": argv.count("--confirm-one-case"), "profile_diagnostic_flag_count": argv.count("--profile-diagnostic"), "ready_artifact_flag_count": argv.count("--ready-artifact"), "evidence_output_flag_count": argv.count("--evidence-output"), "quiet_window_status_required": "go", "quiet_window_decision_required": "GO"}, "execution": {"argument_count": len(argv), "shell": False, "working_directory": str(ROOT), "same_pty_sudo_cache_required": True, "external_service_stop_required": True, "maximum_invocations": 1, "output_no_reuse": True, "operator_must_use_manifest_argv_exactly": True, "requires_fresh_output_recheck_immediately_before_execution": True, "promotion_eligible": False, "measurement_eligible": False}, "inputs": {"profile_ready": {"artifact_commit": READY_ARTIFACT_COMMIT, "ready_binding_sha256": READY_BINDING_SHA256, "inventory": ready_inventory}, "quiet_window": {"path": str(QUIET_ROOT / "quiet-window.json"), "sha256": sha_file(QUIET_ROOT / "quiet-window.json"), "decision": quiet["value"]["decision"], "status": quiet["value"]["status"]}, "historical_operator_v6": previous}, "fresh_outputs": [{"path": str(path), "absent": True} for path in fresh], "quiet_final_streak": quiet["value"]["summary"], "failure_contract": {"retry_forbidden": True, "preserve_operator_stdout_stderr": True, "preserve_maintenance_launcher_capture_and_ready_audits": True, "immutable_failure_capture_before_reporting": True, "outer_restore_in_finally": True, "restore_timeout_seconds": ready.get("maintenance", {}).get("restore_poll", {}).get("timeout_seconds"), "restore_requires_active_running_new_epoch_nrestarts_zero_worker_lock_gpu_kfd_formal_health_and_hashes": True, "children_remaining_must_be_empty": True}, "target_runner_manifest": {"schema_version": "ullm.aq4_p3_profile_target_command.v1", "fresh_per_execution": True, "generated_by": "launcher_after_live_preflight", "maximum_invocations": 1, "static_manifest_present": False}, "pre_execution_audit": {"quiet_window": "passed", "fresh_outputs": "9/9 absent", "historical_operator_v6": "immutable_readback", "actual_executed": False}, "actual_executed": False, "gpu_command_executed": False, "service_touched": False, "secret_material_embedded": False, "manifest_sha256": None}
    manifest["manifest_sha256"] = sha_bytes(canonical(manifest))
    write_sealed(output, "command-manifest.json", manifest); validate_operator(output)
    return manifest


def validate_operator(root: Path = OPERATOR_ROOT) -> dict[str, Any]:
    inventory = verify_sums(root); value = load(root / "command-manifest.json", "operator manifest")
    clone = json.loads(json.dumps(value)); declared = clone.get("manifest_sha256"); clone["manifest_sha256"] = None
    if value.get("schema_version") != OPERATOR_SCHEMA or declared != sha_bytes(canonical(clone)) or value.get("argv") != actual_argv() or value.get("command_sha256") != sha_bytes(canonical(actual_argv())):
        raise OperatorError("operator manifest semantic binding differs")
    failure = value.get("failure_contract", {})
    execution = value.get("execution", {})
    if value.get("authorization", {}).get("maximum_invocations") != 1 or execution.get("maximum_invocations") != 1 or execution.get("shell") is not False or execution.get("outer_restore_in_finally") is True or value.get("actual_executed") is not False or value.get("gpu_command_executed") is not False or value.get("service_touched") is not False or value.get("secret_material_embedded") is not False or len(value.get("fresh_outputs", [])) != 9 or not all(item.get("absent") is True for item in value["fresh_outputs"]):
        raise OperatorError("operator authorization/safety differs")
    if failure.get("retry_forbidden") is not True or failure.get("outer_restore_in_finally") is not True or failure.get("restore_timeout_seconds") != 120.0 or failure.get("children_remaining_must_be_empty") is not True:
        raise OperatorError("operator failure/restore contract differs")
    return {"value": value, "inventory": inventory}


def audit_current() -> dict[str, Any]:
    ready, _ = ready_authority()
    snapshot = capture_snapshot(ready)
    return {
        "status": "clean" if snapshot["clean"] else "blocked",
        "service": snapshot["service"],
        "worker": snapshot["worker"],
        "lock": snapshot["lock"],
        "gpu": snapshot["gpu"],
        "owners": snapshot["owners"],
        "hashes": snapshot["hashes"],
        "formal_health_sha256": snapshot["formal_health_sha256"],
        "targeted_processes": snapshot["targeted_processes"],
        "fresh_outputs_absent": snapshot["relevant"]["all_required_absent"],
        "actual_executed": False,
        "service_touched": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    quiet = sub.add_parser("collect-quiet"); quiet.add_argument("--output", type=Path, default=QUIET_ROOT); quiet.add_argument("--interval", type=float, default=DEFAULT_INTERVAL); quiet.add_argument("--maximum", type=float, default=DEFAULT_MAXIMUM); quiet.add_argument("--minimum-span", type=float, default=DEFAULT_MINIMUM_SPAN); quiet.add_argument("--required-samples", type=int, default=DEFAULT_REQUIRED_SAMPLES)
    sub.add_parser("validate-quiet").add_argument("--root", type=Path, default=QUIET_ROOT)
    sub.add_parser("prepare-operator").add_argument("--output", type=Path, default=OPERATOR_ROOT)
    sub.add_parser("validate-operator").add_argument("--root", type=Path, default=OPERATOR_ROOT)
    sub.add_parser("audit-current")
    sub.add_parser("print-actual")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-quiet": result = collect_quiet(args.output, interval=args.interval, maximum=args.maximum, minimum_span=args.minimum_span, required=args.required_samples)
        elif args.command == "validate-quiet": result = validate_quiet(args.root)["value"]
        elif args.command == "prepare-operator": result = prepare_operator(args.output)
        elif args.command == "validate-operator": result = validate_operator(args.root)["value"]
        elif args.command == "audit-current": result = audit_current()
        else: result = {"argv": actual_argv(), "shell": False, "maximum_invocations": 1, "actual_executed": False}
        if args.command == "audit-current":
            print(json.dumps(result, sort_keys=True))
        else:
            print(json.dumps({"status": result.get("status", "prepared"), "decision": result.get("decision"), "actual_executed": result.get("actual_executed", False), "argv": result.get("argv") if args.command == "print-actual" else None}, sort_keys=True))
        return 0
    except (OperatorError, OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError) as error:
        print(f"AQ4 P3 profile operator {args.command} failed: {error}", file=sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
