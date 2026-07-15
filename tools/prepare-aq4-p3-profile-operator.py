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
PROFILE_READY_ROOT = P2 / "resident-one-case-smoke-profile-ready-v11"
PROFILE_READY = PROFILE_READY_ROOT / "ready-binding.json"
QUIET_ROOT = P2 / "resident-one-case-smoke-profile-quiet-window-v14"
OPERATOR_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v9"
MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v8"
OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v9"
ACTUAL_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v9"
PROFILE_RUNTIME = P2 / "resident-one-case-smoke-profile-execute-v8"
PROFILE_EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-profile-execute-evidence-v8"
PROFILE_CAPTURE = P3 / "aq4-p3-diagnostic-rocprof-capture-v8"
PREVIOUS_OPERATOR_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v8"
EXECUTE_BINDING_ROOT = P2 / "resident-one-case-smoke-execute-binding-v8"
PYTHON = Path("/usr/bin/python3.12")
QUIET_SCHEMA = "ullm.aq4_p3_profile_quiet_window.v14"
OPERATOR_SCHEMA = "ullm.aq4_p3_profile_operator_command.v9"
OPERATOR_RESULT_SCHEMA = "ullm.aq4_p3_profile_operator_result.v9"
ACTUAL_AUDIT_SCHEMA = "ullm.aq4_p3_profile_actual_audit.v9"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_INTERVAL = 5.0
DEFAULT_MAXIMUM = 900.0
DEFAULT_MINIMUM_SPAN = 130.0
DEFAULT_REQUIRED_SAMPLES = 27

# Filled only after the fresh profile-ready-v11 commit is final.  The collector
# rejects any other ready artifact rather than silently following a moving path.
READY_ARTIFACT_COMMIT = "abcf95ad3e56010b3e5f8b38c883c25bf5e2c780"
READY_ARTIFACT_TREE = "5f140564964883a67c2c2d8af066e8eecb935b37"
READY_BINDING_SHA256 = "ef23daf6b8166abc98fa0a72a0eeeae86ab24b5b1747ff0018c4240398ba0c18"
READY_SHA256SUMS_SHA256 = "7bb6a891969ef73a3024aec370c8e38a245bb95e21711e0f1b6068cdfabf9217"
EXECUTE_BINDING_ARTIFACT_COMMIT = "ee7333cdbc1da23f24295fe6d32462feebc6467f"


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
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise OperatorError(f"SHA256SUMS syntax differs: {root}") from error
        relative = Path(name)
        if SHA_RE.fullmatch(digest) is None or not name or name in declared or relative.is_absolute() or ".." in relative.parts or name == "SHA256SUMS":
            raise OperatorError(f"SHA256SUMS syntax differs: {root}")
        declared[name] = digest
    expected: set[str] = set()
    for item in root.rglob("*"):
        relative = str(item.relative_to(root))
        metadata = item.lstat()
        if item.is_symlink() or (not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(metadata.st_mode)):
            raise OperatorError(f"sealed member differs: {item}")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o555:
                raise OperatorError(f"sealed directory differs: {item}")
        elif relative != "SHA256SUMS":
            expected.add(relative)
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


def verify_inventory_commit(root: Path, inventory: dict[str, Any], commit: str) -> None:
    paths = [root / "SHA256SUMS", *(Path(item["path"]) for item in inventory["members"].values())]
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if git("rev-parse", f"{commit}:{relative}") != git("hash-object", str(path)):
            raise OperatorError(f"sealed Git authority differs: {path}")


def ready_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    if not READY_ARTIFACT_COMMIT or not READY_ARTIFACT_TREE or not READY_BINDING_SHA256 or not READY_SHA256SUMS_SHA256:
        raise OperatorError("profile-ready-v11 authority pins are not finalized")
    inventory = verify_sums(PROFILE_READY_ROOT)
    if sha_file(PROFILE_READY) != READY_BINDING_SHA256 or inventory["sha256sums_sha256"] != READY_SHA256SUMS_SHA256:
        raise OperatorError("profile-ready-v11 hashes differ")
    if git("rev-parse", f"{READY_ARTIFACT_COMMIT}^{{tree}}") != READY_ARTIFACT_TREE:
        raise OperatorError("profile-ready-v11 Git tree differs")
    verify_inventory_commit(PROFILE_READY_ROOT, inventory, READY_ARTIFACT_COMMIT)
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
        EXECUTE_BINDING_ROOT,
        P2 / "resident-one-case-smoke-ready-v6",
        P2 / "resident-one-case-smoke-ready-dry-run-v6",
        PROFILE_READY_ROOT,
        P2 / "resident-one-case-smoke-profile-ready-dry-run-v11",
    ]


def trusted_source_snapshot(ready: dict[str, Any]) -> list[dict[str, Any]]:
    trust = load(PROFILE_READY_ROOT / "harness-trust.json", "profile harness trust")
    qa = load(PROFILE_READY_ROOT / "qa-attestation.json", "profile QA attestation")
    binding = ready["launcher_binding"]
    specifications: list[tuple[Path, str, str, str | None]] = [
        (Path(trust["path"]), trust["sha256"], trust["commit"], trust["git_blob"]),
        (ROOT / "tools/launch-aq4-p2-resident-smoke.py", qa["launcher"]["sha256"], qa["launcher"]["commit"], None),
        (Path(ready["profile_diagnostic"]["capture_tool"]["path"]), qa["capture_tool"]["sha256"], qa["capture_tool"]["commit"], ready["profile_diagnostic"]["capture_tool"]["git_blob"]),
        (Path(binding["R"]["path"]), binding["R"]["sha256"], binding["R"]["commit"], binding["R"]["git_blob"]),
        (Path(binding["validator"]["path"]), binding["validator"]["sha256"], binding["validator"]["commit"], binding["validator"]["git_blob"]),
        (Path(binding["resident"]["path"]), binding["resident"]["sha256"], binding["resident"]["commit"], None),
    ]
    for suite in qa["automated_tests"]["suites"]:
        for item in suite["files"]:
            path = ROOT / item["path"]
            specifications.append((path, sha_file(path), item["source_commit"], item["git_blob"]))
    specifications.append((SOURCE, sha_file(SOURCE), git("rev-parse", "HEAD"), None))
    unique: dict[str, tuple[Path, str, str, str | None]] = {}
    for path, expected_sha, source_commit, expected_blob in specifications:
        key = str(path)
        if key in unique and unique[key][1:] != (expected_sha, source_commit, expected_blob):
            raise OperatorError(f"trusted source authority conflicts: {path}")
        unique[key] = (path, expected_sha, source_commit, expected_blob)
    records: list[dict[str, Any]] = []
    for path, expected_sha, source_commit, expected_blob in unique.values():
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1 or sha_file(path) != expected_sha:
            raise OperatorError(f"trusted source differs: {path}")
        relative = str(path.relative_to(ROOT))
        current_blob = git("hash-object", str(path))
        artifact_commit = git("log", "-1", "--format=%H", "--", relative)
        if not artifact_commit or git("rev-parse", f"{artifact_commit}:{relative}") != current_blob or (expected_blob is not None and expected_blob != current_blob):
            raise OperatorError(f"trusted source Git authority differs: {path}")
        records.append({"path": str(path), "sha256": expected_sha, "source_commit": source_commit, "artifact_commit": artifact_commit, "git_blob": current_blob, "identity": [metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_nlink, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns]})
    return sorted(records, key=lambda item: item["path"])


def relevant_snapshot(ready: dict[str, Any]) -> dict[str, Any]:
    roots = [verify_sums(root) for root in root_set()]
    execute_inventory = next(item for item in roots if item["root"] == str(EXECUTE_BINDING_ROOT))
    verify_inventory_commit(EXECUTE_BINDING_ROOT, execute_inventory, EXECUTE_BINDING_ARTIFACT_COMMIT)
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


def seal_existing(root: Path) -> dict[str, Any]:
    if (root / "SHA256SUMS").exists() or (root / "SHA256SUMS").is_symlink():
        return verify_sums(root)
    if root.is_symlink() or not root.is_dir():
        raise OperatorError(f"evidence root differs: {root}")
    members: list[Path] = []
    directories: list[Path] = []
    for item in root.rglob("*"):
        metadata = item.lstat()
        if item.is_symlink() or (not stat.S_ISREG(metadata.st_mode) and not stat.S_ISDIR(metadata.st_mode)):
            raise OperatorError(f"evidence member differs: {item}")
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(item)
        else:
            members.append(item)
    members.sort(key=lambda item: str(item.relative_to(root)))
    if not members:
        raise OperatorError(f"evidence root is empty: {root}")
    lines: list[str] = []
    for path in members:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OperatorError(f"evidence member differs: {path}")
        lines.append(f"{sha_file(path)}  {path.relative_to(root)}\n")
    sums = root / "SHA256SUMS"
    sums.write_text("".join(lines), encoding="ascii")
    for path in [*members, sums]:
        os.chmod(path, 0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        os.chmod(path, 0o555)
    os.chmod(root, 0o555)
    return verify_sums(root)


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
    manifest: dict[str, Any] = {"schema_version": OPERATOR_SCHEMA, "status": "audited_ready_for_single_explicit_profile_diagnostic", "argv": argv, "command_sha256": sha_bytes(canonical(argv)), "authorization": {"maximum_invocations": 1, "explicit_confirmation_flag_count": argv.count("--confirm-one-case"), "profile_diagnostic_flag_count": argv.count("--profile-diagnostic"), "ready_artifact_flag_count": argv.count("--ready-artifact"), "evidence_output_flag_count": argv.count("--evidence-output"), "quiet_window_status_required": "go", "quiet_window_decision_required": "GO"}, "execution": {"argument_count": len(argv), "shell": False, "working_directory": str(ROOT), "same_pty_sudo_cache_required": True, "external_service_stop_required": True, "maximum_invocations": 1, "output_no_reuse": True, "operator_must_use_manifest_argv_exactly": True, "requires_fresh_output_recheck_immediately_before_execution": True, "promotion_eligible": False, "measurement_eligible": False}, "inputs": {"profile_ready": {"artifact_commit": READY_ARTIFACT_COMMIT, "ready_binding_sha256": READY_BINDING_SHA256, "inventory": ready_inventory}, "quiet_window": {"path": str(QUIET_ROOT / "quiet-window.json"), "sha256": sha_file(QUIET_ROOT / "quiet-window.json"), "decision": quiet["value"]["decision"], "status": quiet["value"]["status"]}, "historical_operator_v8": previous}, "fresh_outputs": [{"path": str(path), "absent": True} for path in fresh], "quiet_final_streak": quiet["value"]["summary"], "failure_contract": {"retry_forbidden": True, "preserve_operator_stdout_stderr": True, "preserve_maintenance_launcher_capture_and_ready_audits": True, "immutable_failure_capture_before_reporting": True, "outer_restore_in_finally": True, "restore_timeout_seconds": ready.get("maintenance", {}).get("restore_poll", {}).get("timeout_seconds"), "restore_requires_active_running_new_epoch_nrestarts_zero_worker_lock_gpu_kfd_formal_health_and_hashes": True, "children_remaining_must_be_empty": True}, "target_runner_manifest": {"schema_version": "ullm.aq4_p3_profile_target_command.v1", "fresh_per_execution": True, "generated_by": "launcher_after_live_preflight", "maximum_invocations": 1, "static_manifest_present": False}, "pre_execution_audit": {"quiet_window": "passed", "fresh_outputs": "9/9 absent", "historical_operator_v8": "immutable_readback", "actual_executed": False}, "actual_executed": False, "gpu_command_executed": False, "service_touched": False, "secret_material_embedded": False, "manifest_sha256": None}
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


def stream_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha_file(path)}


def optional_stream(root: Path, value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not isinstance(value.get("file"), str):
        return None
    relative = Path(value["file"])
    if relative.is_absolute() or ".." in relative.parts:
        raise OperatorError("subprocess stream binding differs")
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise OperatorError("subprocess stream evidence differs")
    return stream_record(path)


def seal_optional(root: Path, *, required: bool) -> dict[str, Any] | None:
    if not root.exists() and not root.is_symlink():
        if required:
            raise OperatorError(f"required evidence root is missing: {root}")
        return None
    return seal_existing(root)


def validate_actual_documents(result: dict[str, Any], audit: dict[str, Any]) -> None:
    clone = json.loads(json.dumps(audit)); declared = clone.get("audit_sha256"); clone["audit_sha256"] = None
    returncode = result.get("returncode")
    succeeded = type(returncode) is int and returncode == 0
    expected_result = "passed" if succeeded else "failed"
    expected_audit = "passed_immutable_evidence_preserved_restore_passed" if succeeded else "failed_immutable_evidence_preserved_restore_passed"
    if result.get("schema_version") != OPERATOR_RESULT_SCHEMA or result.get("status") != expected_result or type(returncode) is not int or result.get("invocation_count") != 1 or result.get("maximum_invocations") != 1 or result.get("shell") is not False or result.get("retry_performed") is not False or result.get("actual_executed") is not True or result.get("secret_material_recorded") is not False:
        raise OperatorError("operator result semantic boundary differs")
    execution = audit.get("execution", {})
    profile = audit.get("profile_artifacts", {})
    if audit.get("schema_version") != ACTUAL_AUDIT_SCHEMA or declared != sha_bytes(canonical(clone)) or audit.get("status") != expected_audit or execution.get("returncode") != returncode or execution.get("invocation_count") != 1 or execution.get("maximum_invocations") != 1 or execution.get("shell") is not False or execution.get("retry_performed") is not False or audit.get("restore", {}).get("passed") is not True or audit.get("package_integrity", {}).get("full_hash_count") != 1 or audit.get("cleanup", {}).get("residual_targeted_processes") != [] or audit.get("actual_executed") is not True or audit.get("retry_performed") is not False or audit.get("secret_material_recorded") is not False:
        raise OperatorError("actual audit semantic boundary differs")
    if succeeded and (audit.get("failure") is not None or profile.get("status") != "complete_diagnostic" or profile.get("measurement_eligible") is not False or profile.get("promotion_eligible") is not False):
        raise OperatorError("successful actual audit outcome differs")
    if not succeeded and (not isinstance(audit.get("failure"), dict) or profile.get("status") != "failure_evidence_only"):
        raise OperatorError("failed actual audit outcome differs")


def finalize_actual(*, returncode: int, start_unix_ns: int, end_unix_ns: int) -> dict[str, Any]:
    if type(returncode) is not int or start_unix_ns <= 0 or end_unix_ns <= start_unix_ns:
        raise OperatorError("actual execution boundary differs")
    manifest = validate_operator()["value"]
    quiet = validate_quiet()["value"]
    stdout_path = OPERATOR_RESULT / "operator.stdout.bin"
    stderr_path = OPERATOR_RESULT / "operator.stderr.bin"
    if ACTUAL_AUDIT.exists() or ACTUAL_AUDIT.is_symlink() or OPERATOR_RESULT.is_symlink() or not OPERATOR_RESULT.is_dir() or not stdout_path.is_file() or stdout_path.is_symlink() or not stderr_path.is_file() or stderr_path.is_symlink() or (OPERATOR_RESULT / "operator-result.json").exists() or (OPERATOR_RESULT / "SHA256SUMS").exists():
        raise OperatorError("operator raw stream state differs")
    stdout_value = load(stdout_path, "operator stdout")
    succeeded = returncode == 0
    outcome_status = "passed" if succeeded else "failed"
    if stdout_value.get("status") != outcome_status or stdout_value.get("mode") != "execute" or stdout_value.get("evidence") != str(MAINTENANCE_EVIDENCE / "launcher-evidence.json"):
        raise OperatorError("operator raw outcome differs")

    maintenance_inventory = verify_sums(MAINTENANCE_EVIDENCE)
    maintenance = load(MAINTENANCE_EVIDENCE / "launcher-evidence.json", "maintenance evidence")
    if maintenance.get("status") != outcome_status or maintenance.get("mode") != "execute":
        raise OperatorError("maintenance outcome boundary differs")
    if succeeded and maintenance.get("failure") is not None:
        raise OperatorError("successful maintenance recorded a failure")
    if not succeeded and not isinstance(maintenance.get("failure"), dict):
        raise OperatorError("failed maintenance omitted its failure")
    package = maintenance.get("package_integrity", {})
    restore = maintenance.get("restore", {})
    cleanup = maintenance.get("lock_substrate_cleanup", {})
    if package.get("full_hash_count") != 1 or package.get("full_content", {}).get("passed") is not True or package.get("integrity_identity", {}).get("passed") is not True:
        raise OperatorError("package exact-one integrity differs")
    if restore.get("passed") is not True or restore.get("duration_ns", 120_000_000_001) > 120_000_000_000 or restore.get("final_metadata_recheck", {}).get("within_absolute_deadline") is not True:
        raise OperatorError("outer-finally restore differs")
    if cleanup.get("passed") is not True or cleanup.get("runner_children") != [] or cleanup.get("holder_pids") != []:
        raise OperatorError("lock/residual cleanup differs")

    execute_inventory = seal_optional(PROFILE_EXECUTE_EVIDENCE, required=succeeded)
    runtime_inventory = seal_optional(PROFILE_RUNTIME, required=succeeded)
    capture_inventory = seal_optional(PROFILE_CAPTURE, required=succeeded)
    launcher_path = PROFILE_EXECUTE_EVIDENCE / "launcher-evidence.json"
    runtime_summary_path = PROFILE_RUNTIME / "resident-batch.summary.json"
    driver_process_path = PROFILE_RUNTIME / "resident-batch.driver-process.json"
    capture_artifact_path = PROFILE_CAPTURE / "capture-artifact.json"
    capture_failure_path = PROFILE_CAPTURE / "capture-failure.json"
    launcher = load(launcher_path, "launcher evidence") if launcher_path.is_file() else None
    runtime_summary = load(runtime_summary_path, "runtime summary") if runtime_summary_path.is_file() else None
    driver_process = load(driver_process_path, "driver process") if driver_process_path.is_file() else None
    capture_artifact = load(capture_artifact_path, "capture artifact") if capture_artifact_path.is_file() else None
    capture_failure = load(capture_failure_path, "capture failure") if capture_failure_path.is_file() else None
    if succeeded:
        if not isinstance(launcher, dict) or launcher.get("status") != "passed" or launcher.get("runner", {}).get("exit_code") != 0:
            raise OperatorError("successful launcher boundary differs")
        if not isinstance(runtime_summary, dict) or runtime_summary.get("status") != "complete" or runtime_summary.get("resident_model_loads") != 1 or not isinstance(driver_process, dict) or driver_process.get("cleanup", {}).get("passed") is not True:
            raise OperatorError("successful runtime boundary differs")
        if not isinstance(capture_artifact, dict) or capture_artifact.get("status") != "complete_diagnostic" or capture_artifact.get("measurement_eligible") is not False or capture_artifact.get("promotion_eligible") is not False or capture_failure is not None:
            raise OperatorError("successful capture boundary differs")
    else:
        if isinstance(launcher, dict) and (launcher.get("status") != "failed" or launcher.get("failure", {}).get("children_remaining") != []):
            raise OperatorError("failed launcher boundary differs")
        if isinstance(driver_process, dict) and driver_process.get("cleanup", {}).get("passed") is not True:
            raise OperatorError("failed runtime cleanup differs")
        if isinstance(capture_failure, dict) and (capture_failure.get("status") != "failed" or capture_failure.get("children_remaining") != [] or capture_failure.get("process_group_cleanup_complete") is not True):
            raise OperatorError("capture failure boundary differs")

    post = capture_snapshot(load(PROFILE_READY, "profile ready binding"))
    running = {"service": post["service"], "worker": post["worker"], "gpu": post["gpu"], "owners": post["owners"], "lock": post["lock"], "hashes": post["hashes"], "formal_health_sha256": post["formal_health_sha256"], "targeted_processes": post["targeted_processes"]}
    pre = quiet["confirmation"]
    if post["service"].get("active_state") != "active" or post["service"].get("sub_state") != "running" or post["service"].get("nrestarts") != 0 or post["service"].get("main_pid") == pre["service"]["main_pid"] or post["worker"]["pid"] == pre["worker"]["pid"]:
        raise OperatorError("post-restore service epoch differs")
    if post["owners"] != {"amd_smi": [post["worker"]["pid"]], "kfd": [post["worker"]["pid"]]} or post["lock"].get("busy") is not True or post["targeted_processes"]:
        raise OperatorError("post-restore owner/residual state differs")
    if post["hashes"] != pre["hashes"] or post["formal_health_sha256"] != pre["formal_health_sha256"]:
        raise OperatorError("post-restore health/hash state differs")

    operator_result = {
        "schema_version": OPERATOR_RESULT_SCHEMA,
        "status": outcome_status,
        "authority_commit": manifest["inputs"]["profile_ready"]["artifact_commit"],
        "operator_manifest_commit": git("log", "-1", "--format=%H", "--", str((OPERATOR_ROOT / "command-manifest.json").relative_to(ROOT))),
        "manifest_file_sha256": sha_file(OPERATOR_ROOT / "command-manifest.json"),
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "command_sha256": manifest["command_sha256"],
        "argument_count": len(manifest["argv"]),
        "working_directory": str(ROOT),
        "shell": False,
        "same_pty_sudo_cache": True,
        "maximum_invocations": 1,
        "invocation_count": 1,
        "retry_performed": False,
        "returncode": returncode,
        "canonical_start_unix_ns": start_unix_ns,
        "canonical_end_unix_ns": end_unix_ns,
        "elapsed_ns": end_unix_ns - start_unix_ns,
        "preflight": {"passed": True, "fresh_outputs_absent": 9, "service_main_pid": pre["service"]["main_pid"], "worker_pid": pre["worker"]["pid"], "amd_smi_owners": pre["owners"]["amd_smi"], "kfd_owners": pre["owners"]["kfd"], "formal_health_sha256": pre["formal_health_sha256"], "targeted_external_processes": 0},
        "stdout": stream_record(stdout_path),
        "stderr": stream_record(stderr_path),
        "actual_executed": True,
        "secret_material_recorded": False,
    }
    raw = pretty(operator_result)
    (OPERATOR_RESULT / "operator-result.json").write_bytes(raw)
    operator_inventory = seal_existing(OPERATOR_RESULT)

    trace_files = [] if capture_inventory is None else [item for item in capture_inventory["members"].values() if item["path"].endswith(".csv")]
    maintenance_failure = maintenance.get("failure") if isinstance(maintenance.get("failure"), dict) else {}
    launcher_failure = launcher.get("failure") if isinstance(launcher, dict) and isinstance(launcher.get("failure"), dict) else {}
    failure = None if succeeded else {
        "returncode": returncode,
        "maintenance_stage": maintenance_failure.get("stage"),
        "maintenance_reason": maintenance_failure.get("reason"),
        "launcher_stage": launcher_failure.get("stage"),
        "launcher_reason": launcher_failure.get("reason"),
        "capture_failure_schema": capture_failure.get("schema_version") if isinstance(capture_failure, dict) else None,
        "capture_failure_reason": capture_failure.get("reason") if isinstance(capture_failure, dict) else None,
        "capture_failure_sha256": sha_file(capture_failure_path) if capture_failure_path.is_file() else None,
        "ready_candidate_reason_code": capture_failure.get("ready_candidate_audit", {}).get("reason_code") if isinstance(capture_failure, dict) else None,
    }
    runner = launcher.get("runner", {}) if isinstance(launcher, dict) else {}
    validator = launcher.get("validator", {}) if isinstance(launcher, dict) else {}
    capture_process = maintenance.get("capture", {}) if isinstance(maintenance.get("capture"), dict) else {}
    audit = {
        "schema_version": ACTUAL_AUDIT_SCHEMA,
        "status": "passed_immutable_evidence_preserved_restore_passed" if succeeded else "failed_immutable_evidence_preserved_restore_passed",
        "authority_commit": operator_result["operator_manifest_commit"],
        "manifest_file_sha256": operator_result["manifest_file_sha256"],
        "execution": {key: operator_result[key] for key in ("argument_count", "working_directory", "shell", "same_pty_sudo_cache", "maximum_invocations", "invocation_count", "retry_performed", "returncode", "canonical_start_unix_ns", "canonical_end_unix_ns", "elapsed_ns")},
        "failure": failure,
        "all_returncodes_and_streams": {"operator": {"returncode": returncode, "stdout": operator_result["stdout"], "stderr": operator_result["stderr"]}, "runner": {"returncode": runner.get("exit_code"), "stdout": optional_stream(PROFILE_EXECUTE_EVIDENCE, runner.get("stdout")), "stderr": optional_stream(PROFILE_EXECUTE_EVIDENCE, runner.get("stderr"))}, "validator": {"returncode": validator.get("exit_code"), "stdout": optional_stream(PROFILE_EXECUTE_EVIDENCE, validator.get("stdout")), "stderr": optional_stream(PROFILE_EXECUTE_EVIDENCE, validator.get("stderr"))}, "rocprof": {"returncode": capture_process.get("exit_code"), "stdout": stream_record(PROFILE_CAPTURE / "rocprof.stdout") if (PROFILE_CAPTURE / "rocprof.stdout").is_file() else None, "stderr": stream_record(PROFILE_CAPTURE / "rocprof.stderr") if (PROFILE_CAPTURE / "rocprof.stderr").is_file() else None}},
        "package_integrity": package,
        "restore": restore,
        "post_health": running,
        "cleanup": {"capture_children_remaining": capture_failure.get("children_remaining") if isinstance(capture_failure, dict) else [], "capture_process_group_cleanup_complete": capture_failure.get("process_group_cleanup_complete") if isinstance(capture_failure, dict) else True, "launcher_children_remaining": launcher_failure.get("children_remaining", []), "launcher_cleanup_passed": launcher_failure.get("cleanup_passed", True), "driver_cleanup_passed": driver_process.get("cleanup", {}).get("passed") if isinstance(driver_process, dict) else None, "residual_targeted_processes": post["targeted_processes"], "trusted_lock_substrate_cleanup_passed": cleanup["passed"], "trusted_lock_substrate_holder_pids": cleanup["holder_pids"], "retry_forbidden_and_not_performed": True},
        "profile_artifacts": {"status": "complete_diagnostic" if succeeded else "failure_evidence_only", "measurement_eligible": False, "promotion_eligible": False, "trace_csv_count": len(trace_files), "trace_csv_bytes": sum(item["size"] for item in trace_files), "capture_artifact": capture_inventory["members"].get("capture-artifact.json") if capture_inventory is not None else None, "capture_failure": capture_inventory["members"].get("capture-failure.json") if capture_inventory is not None else None, "runtime_summary": runtime_inventory["members"].get("resident-batch.summary.json") if runtime_inventory is not None else None},
        "evidence": {"maintenance": maintenance_inventory, "execute": execute_inventory, "runtime": runtime_inventory, "capture": capture_inventory, "operator_result": operator_inventory},
        "actual_executed": True,
        "retry_performed": False,
        "secret_material_recorded": False,
        "audit_sha256": None,
    }
    audit["audit_sha256"] = sha_bytes(canonical(audit))
    validate_actual_documents(operator_result, audit)
    write_sealed(ACTUAL_AUDIT, "actual-audit.json", audit)
    validate_actual()
    return audit


def validate_actual() -> dict[str, Any]:
    result_inventory = verify_sums(OPERATOR_RESULT)
    audit_inventory = verify_sums(ACTUAL_AUDIT)
    result = load(OPERATOR_RESULT / "operator-result.json", "operator result")
    audit = load(ACTUAL_AUDIT / "actual-audit.json", "actual audit")
    validate_actual_documents(result, audit)
    verify_sums(MAINTENANCE_EVIDENCE)
    for root in (PROFILE_EXECUTE_EVIDENCE, PROFILE_RUNTIME, PROFILE_CAPTURE):
        if root.exists() or root.is_symlink():
            verify_sums(root)
    return {"result": result, "audit": audit, "result_inventory": result_inventory, "audit_inventory": audit_inventory}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    quiet = sub.add_parser("collect-quiet"); quiet.add_argument("--output", type=Path, default=QUIET_ROOT); quiet.add_argument("--interval", type=float, default=DEFAULT_INTERVAL); quiet.add_argument("--maximum", type=float, default=DEFAULT_MAXIMUM); quiet.add_argument("--minimum-span", type=float, default=DEFAULT_MINIMUM_SPAN); quiet.add_argument("--required-samples", type=int, default=DEFAULT_REQUIRED_SAMPLES)
    sub.add_parser("validate-quiet").add_argument("--root", type=Path, default=QUIET_ROOT)
    sub.add_parser("prepare-operator").add_argument("--output", type=Path, default=OPERATOR_ROOT)
    sub.add_parser("validate-operator").add_argument("--root", type=Path, default=OPERATOR_ROOT)
    sub.add_parser("audit-current")
    sub.add_parser("print-actual")
    final = sub.add_parser("finalize-actual"); final.add_argument("--returncode", type=int, required=True); final.add_argument("--start-unix-ns", type=int, required=True); final.add_argument("--end-unix-ns", type=int, required=True)
    sub.add_parser("validate-actual")
    args = parser.parse_args(argv)
    try:
        if args.command == "collect-quiet": result = collect_quiet(args.output, interval=args.interval, maximum=args.maximum, minimum_span=args.minimum_span, required=args.required_samples)
        elif args.command == "validate-quiet": result = validate_quiet(args.root)["value"]
        elif args.command == "prepare-operator": result = prepare_operator(args.output)
        elif args.command == "validate-operator": result = validate_operator(args.root)["value"]
        elif args.command == "audit-current": result = audit_current()
        elif args.command == "finalize-actual": result = finalize_actual(returncode=args.returncode, start_unix_ns=args.start_unix_ns, end_unix_ns=args.end_unix_ns)
        elif args.command == "validate-actual": result = validate_actual()["audit"]
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
