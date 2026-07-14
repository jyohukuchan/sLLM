#!/usr/bin/env python3
"""Immutable L launcher for the AQ4 P2 resident one-case smoke trust chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
BINDING_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v4"
BINDING_MANIFEST = BINDING_ROOT / "binding-manifest.json"
RUNNER = BINDING_ROOT / "trusted-runner.py"
VALIDATOR = ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py"
PYTHON = Path("/usr/bin/python3.12")
RESIDENT_DRIVER = INPUT_ROOT / "resident-driver"
SERVED_MANIFEST = Path("/etc/ullm/served-models/active.json")
LOCK_PATH = Path("/run/ullm/r9700.lock")
RUNNER_OUTPUT = Path("/tmp/ullm-aq4-p2-resident-smoke-L-dry-run")

INPUT_ROOT_DEVICE = 66306
INPUT_ROOT_INODE = 10512713
INPUT_FINGERPRINT_SHA = "9e2be7a00fb7cb4c085dc1bc3e8892d36bc8187a3f3e37bb802c97e0302a673a"
BINDING_ROOT_DEVICE = 66306
BINDING_ROOT_INODE = 10512778
BINDING_MANIFEST_SHA = "9fc63f2dbd759fd7d5176c0e1b421fb6c7e601d1e9e14825cc27ab85f3cddde1"
BINDING_PLAN_SHA = "bc449219b5e32882d1bca4663abf1eac631dd59c5d0503f1ee287d76ebeabd9c"
RUNNER_COMMIT = "e9065925d7b5af0352cb8dfd454a7e106abd7172"
RUNNER_TREE = "9f2ff38d06d5ea5724a6e84af1c00d2b8147f241"
RUNNER_GIT_BLOB = "9c097d1a97af3e15ca695c6da08b1e2928d08df7"
RUNNER_SHA = "3140574c4f50f9b09aeb3780e400cbf8020ecf1c4ff69da685622858128f33cc"
VALIDATOR_COMMIT = "2e39b7851b856ab067686249ce2d6284484c53d4"
VALIDATOR_TREE = "6f468b4d9c79b664c4cc2ea12c4a889ebe85f8b1"
VALIDATOR_GIT_BLOB = "108b50eeb0c3abfc991a7273a93cff6e06fde766"
VALIDATOR_SHA = "43de32a5a9533c2714085303f80446b6a2f96f191c59830a30ecc73adef95597"
PYTHON_SHA = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
RESIDENT_COMMIT = "319d6187b29e877536aa5dfe80c02bde0c77ed7a"
RESIDENT_SHA = "62f720835de60a61bad0a9aab5b80d778624d4d97ef5c8998e179418dab730f1"
SERVED_SHA = "feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44"
DEVICE_INDEX = 1
CASE_ID = "p2-representative-full_model-cold_prefill-cold_batched-n128-m128-r9700-rdna4-aq4_0_target"
CASE_SHA = "d83a420476bde889c7c8014d7982fd52e0f61ab09b888f66415d0ac9fb443ae7"

INPUT_MEMBER_SHA = {
    "SHA256SUMS": "ccf4857a0a33e0c669a105fad199ef8ec3caf3483ba49f872f27a4b5ffb1f4f9",
    "SUPERSEDED-0fd7993.json": "b3eb5e1c5242830187ce7925185e945c1d210c971ffa129273666e4c7b2bec72",
    "bundle.json": "6041c47bb69efc3bb3dd35e8537b02165a34892ccf48192779b9c33af725a748",
    "case-binding.json": "1c8cf17475c0840900ebcc5cd9334d4ebe76c1bd354aaa5106cb875efa1da8b5",
    "dry-run.json": "14b147bb2c4bfd1acad7fde09021c82547eab87e8b49adaf05093afd43d6c669",
    "fake-ready.json": "a26daf1a51499714e6e484694b2a94c107c02e8a604df3ae86de6ebbb0d7d54d",
    "fixture-index.json": "4bcc02ac22bfd19a55913943e5f28dc690c5917b0743b0b5f679c4a5610d353a",
    "fixture.json": "a61c977a7671e7e3d141b87fc84e20e9957be71706cface1988d03054f2dad50",
    "identity.json": "883de9cdb773d83b71e7ea570a84ad9c1c8b93c15b11ef1307ec00e1f94ca741",
    "launch-command.json": "12a1dc385e4e2dc1ee8910a901766a0b6614208ae60dbdcfc2cf1c1557636958",
    "official-case.json": "8f0d27ea03b995cfb26b4e3d5d4424a54a3a563bbcbbb046eb9feb70a1385d5d",
    "package-manifest.json": "a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad",
    "policy.json": "21dff8ecdbc17a1cd86a458fe7f8771eed0cdd18577a5f0fb6c7b96310a2de16",
    "preflight.json": "294ddf1771251c4b1954ea663d73e85821749119da2a4f6c7528fdae510bbc6e",
    "resident-driver": RESIDENT_SHA,
    "runner-dry-run-evidence.json": "6d791cec5a79171a69540896df0974f3e95fc8297d6e059fa58b41ea81326550",
    "served-model.json": SERVED_SHA,
    "trust-roots.json": "8159476b86ebda6694963df3c973c01f26edd9f75db1e3c11ecc01958c87189f",
    "trusted-runner.py": "e7dae31c64b3844a09fbba7ef36bbae7834e21d5d217bad679dd50bdf314ff02",
}
BINDING_MEMBER_SHA = {
    "binding-manifest.json": BINDING_MANIFEST_SHA,
    "runner-plan.json": BINDING_PLAN_SHA,
    "runner-subprocess-evidence.json": "c0cd1c6258070792de8704e456ff3c2125089cdad78a34c8d964d6ec7bc42fb7",
    "trusted-runner.py": RUNNER_SHA,
    "trusted-validator.py": VALIDATOR_SHA,
    "validator-report.json": "a6af7c425935971d1ec8be878888922c319222f3b900afad5a1a9421216f84d2",
    "SHA256SUMS": "180e60b788696381a8edb317a9059ea2282f0d060b663fb9016b60851bb4bb62",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_BYTES = 64 * 1024 * 1024
CHUNK = 1024 * 1024
_AFTER_VALIDATOR_HOOK: Callable[[], None] | None = None
_BEFORE_FINAL_VERIFY_HOOK: Callable[[], None] | None = None


class LauncherError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise LauncherError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda item: (_ for _ in ()).throw(LauncherError(f"non-finite JSON: {item}")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LauncherError(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise LauncherError(f"{label} root must be an object")
    return value


def file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def reject_symlink_components(path: Path, label: str, *, allow_missing_leaf: bool = False) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise LauncherError(f"{label} must be absolute without parent traversal")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:], 1):
        current /= part
        try:
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise LauncherError(f"{label} has a symlink component: {current}")
        except FileNotFoundError:
            if allow_missing_leaf and index == len(path.parts) - 1:
                return
            raise LauncherError(f"{label} component is missing: {current}")


def read_regular(path: Path, label: str, *, maximum: int | None = MAX_BYTES) -> tuple[bytes, tuple[int, ...]]:
    reject_symlink_components(path, label)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise LauncherError(f"{label} must be a single-link regular file")
    if maximum is not None and before.st_size > maximum:
        raise LauncherError(f"{label} exceeds size bound")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        if file_identity(os.fstat(descriptor)) != file_identity(before):
            raise LauncherError(f"{label} changed while opening")
        while chunk := os.read(descriptor, CHUNK):
            digest.update(chunk)
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if file_identity(after) != file_identity(before) or file_identity(path.lstat()) != file_identity(before):
        raise LauncherError(f"{label} changed while reading")
    return b"".join(chunks), file_identity(before)


def sha_file(path: Path, label: str) -> tuple[str, tuple[int, ...]]:
    raw, identity = read_regular(path, label, maximum=None)
    return sha_bytes(raw), identity


class Snapshot:
    def __init__(self) -> None:
        self.files: dict[Path, tuple[int, ...]] = {}
        self.directories: dict[Path, tuple[int, ...]] = {}

    def file(self, path: Path, expected_sha: str, label: str) -> bytes:
        raw, identity = read_regular(path, label, maximum=None)
        if sha_bytes(raw) != expected_sha:
            raise LauncherError(f"{label} SHA differs")
        self.files[path] = identity
        return raw

    def directory(self, path: Path, device: int, inode: int, label: str) -> None:
        reject_symlink_components(path, label)
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_dev != device or metadata.st_ino != inode:
            raise LauncherError(f"{label} identity differs")
        self.directories[path] = file_identity(metadata)

    def verify(self) -> None:
        for path, expected in self.files.items():
            if file_identity(path.lstat()) != expected:
                raise LauncherError(f"late replacement detected: {path}")
        for path, expected in self.directories.items():
            if file_identity(path.lstat()) != expected:
                raise LauncherError(f"late directory replacement detected: {path}")


def validate_binding_manifest(raw: bytes) -> dict[str, Any]:
    manifest = parse_json(raw, "B binding manifest")
    exact = {"schema_version", "status", "promotion", "launch_eligible", "requires_immutable_launcher", "predecessor", "trust_roots", "input_root", "outputs", "execution", "cycle_control", "next_stage"}
    if set(manifest) != exact:
        raise LauncherError("B binding manifest exact schema differs")
    if manifest.get("schema_version") != "ullm.aq4_p2_resident_smoke_binding.v4" or manifest.get("status") != "prepared_not_executed" or manifest.get("promotion") is not False:
        raise LauncherError("B binding status/promotion differs")
    if manifest.get("launch_eligible") is not False or manifest.get("requires_immutable_launcher") is not True:
        raise LauncherError("B binding L boundary differs")
    roots = manifest.get("trust_roots")
    if not isinstance(roots, dict) or roots.get("source_commit") != RUNNER_COMMIT or roots.get("source_tree") != RUNNER_TREE or roots.get("runner") != {"git_blob": RUNNER_GIT_BLOB, "sha256": RUNNER_SHA}:
        raise LauncherError("B runner trust root differs")
    validator = roots.get("validator")
    if not isinstance(validator, dict) or validator.get("source_commit") != VALIDATOR_COMMIT or validator.get("source_tree") != VALIDATOR_TREE or validator.get("git_blob") != VALIDATOR_GIT_BLOB or validator.get("sha256") != VALIDATOR_SHA or validator.get("execution_path") != str(VALIDATOR):
        raise LauncherError("B validator trust root differs")
    resident = roots.get("resident_driver")
    if not isinstance(resident, dict) or resident.get("normative_commit") != RESIDENT_COMMIT or resident.get("binary_sha256") != RESIDENT_SHA or resident.get("blob_unchanged") is not True:
        raise LauncherError("B resident trust root differs")
    input_root = manifest.get("input_root")
    if not isinstance(input_root, dict) or input_root.get("sha256") != INPUT_FINGERPRINT_SHA or input_root.get("directory") != {"path": str(INPUT_ROOT), "device": INPUT_ROOT_DEVICE, "inode": INPUT_ROOT_INODE}:
        raise LauncherError("B input root fingerprint differs")
    members = input_root.get("members")
    if not isinstance(members, dict) or set(members) != set(INPUT_MEMBER_SHA):
        raise LauncherError("B exact19 member coverage differs")
    for name, digest in INPUT_MEMBER_SHA.items():
        record = members.get(name)
        if not isinstance(record, dict) or record.get("path") != str(INPUT_ROOT / name) or record.get("sha256") != digest or record.get("type") != "regular_file" or record.get("nlink") != 1:
            raise LauncherError(f"B input member differs: {name}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs.get("runner_plan_sha256") != BINDING_PLAN_SHA:
        raise LauncherError("B runner plan binding differs")
    if manifest.get("predecessor") != {"commit": "791a20c", "status": "SUPERSEDED", "execution_eligible": False}:
        raise LauncherError("B predecessor differs")
    return manifest


def validator_argv() -> list[str]:
    return [str(PYTHON), str(VALIDATOR), "validate-binding", "--binding", str(BINDING_ROOT), "--validator-source-commit", VALIDATOR_COMMIT, "--validator-sha256", VALIDATOR_SHA]


def runner_argv() -> list[str]:
    return [
        str(PYTHON), str(RUNNER),
        "--expanded", str(INPUT_ROOT / "case-binding.json"), "--fixture-index", str(INPUT_ROOT / "fixture-index.json"),
        "--identity", str(INPUT_ROOT / "identity.json"), "--preflight", str(INPUT_ROOT / "preflight.json"),
        "--policy", str(INPUT_ROOT / "policy.json"), "--bundle-root", str(INPUT_ROOT),
        "--trusted-validator", str(VALIDATOR), "--trusted-validator-sha256", VALIDATOR_SHA,
        "--output-dir", str(RUNNER_OUTPUT), "--run-id", "p2-r9700-resident-one-case-smoke-binding-v4-validate",
        "--baseline-kind", "active-production", "--lock-path", str(LOCK_PATH), "--one-case-smoke", "--dry-run",
    ]


def validate_constants(snapshot: Snapshot) -> dict[str, Any]:
    snapshot.directory(INPUT_ROOT, INPUT_ROOT_DEVICE, INPUT_ROOT_INODE, "input root")
    snapshot.directory(BINDING_ROOT, BINDING_ROOT_DEVICE, BINDING_ROOT_INODE, "B root")
    if {entry.name for entry in INPUT_ROOT.iterdir()} != set(INPUT_MEMBER_SHA):
        raise LauncherError("input root exact19 coverage differs")
    for name, digest in INPUT_MEMBER_SHA.items():
        snapshot.file(INPUT_ROOT / name, digest, f"input/{name}")
    if {entry.name for entry in BINDING_ROOT.iterdir()} != set(BINDING_MEMBER_SHA):
        raise LauncherError("B root exact member coverage differs")
    binding_raw = b""
    for name, digest in BINDING_MEMBER_SHA.items():
        raw = snapshot.file(BINDING_ROOT / name, digest, f"B/{name}")
        if name == "binding-manifest.json":
            binding_raw = raw
    snapshot.file(PYTHON, PYTHON_SHA, "Python")
    snapshot.file(VALIDATOR, VALIDATOR_SHA, "validator")
    snapshot.file(RUNNER, RUNNER_SHA, "runner")
    snapshot.file(RESIDENT_DRIVER, RESIDENT_SHA, "resident driver")
    snapshot.file(SERVED_MANIFEST, SERVED_SHA, "served manifest")
    reject_symlink_components(LOCK_PATH, "device lock", allow_missing_leaf=True)
    case = parse_json(snapshot.file(INPUT_ROOT / "case-binding.json", INPUT_MEMBER_SHA["case-binding.json"], "case binding"), "case binding")
    cases = case.get("cases")
    if not isinstance(cases, list) or len(cases) != 1 or cases[0].get("case_id") != CASE_ID or cases[0].get("case_sha256") != CASE_SHA or cases[0].get("device", {}).get("runtime_device_index") != DEVICE_INDEX:
        raise LauncherError("pinned case/device differs")
    return validate_binding_manifest(binding_raw)


def atomic_write(directory: Path, name: str, raw: bytes, mode: int = 0o444) -> None:
    path = directory / name
    temporary = directory / f".{name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:offset + CHUNK])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, mode)
    try:
        os.link(temporary, path, follow_symlinks=False)
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def process_record(argv: list[str], completed: subprocess.CompletedProcess[bytes], prefix: str, output: Path) -> dict[str, Any]:
    atomic_write(output, f"{prefix}.stdout.bin", completed.stdout)
    atomic_write(output, f"{prefix}.stderr.bin", completed.stderr)
    return {
        "argv": argv, "exit_code": completed.returncode,
        "stdout": {"file": f"{prefix}.stdout.bin", "sha256": sha_bytes(completed.stdout)},
        "stderr": {"file": f"{prefix}.stderr.bin", "sha256": sha_bytes(completed.stderr)},
    }


def validate_validator_report(raw: bytes) -> dict[str, Any]:
    report = parse_json(raw, "validator report")
    expected = {"status": "prepared_not_executed", "promotion": False, "run_id": "p2-r9700-resident-one-case-smoke-binding-v4"}
    if report != expected:
        raise LauncherError("validator report/root/B binding differs")
    return report


def validate_runner_plan(raw: bytes) -> dict[str, Any]:
    if sha_bytes(raw) != BINDING_PLAN_SHA:
        raise LauncherError("runner dry-run plan differs from B")
    plan = parse_json(raw, "runner plan")
    expected = {"case_count": 1, "transaction_count": 12, "warmup_runs": 2, "measured_runs": 10, "smoke_only": True, "promotion_eligible": False}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise LauncherError("runner one-case plan facts differ")
    validation = plan.get("validation")
    if not isinstance(validation, dict) or validation.get("root_contract") != "ullm.aq4_p2_resident_smoke_bundle_root.v4" or validation.get("trusted_bundle_validator", {}).get("source") != {"path": str(VALIDATOR), "sha256": VALIDATOR_SHA}:
        raise LauncherError("runner validator/root report differs")
    return plan


def make_evidence(mode: str, self_sha: str) -> dict[str, Any]:
    return {
        "schema_version": "ullm.aq4_p2_resident_smoke_immutable_launcher.v1", "status": "failed", "mode": mode,
        "promotion": False, "self": {"path": str(Path(__file__).resolve()), "sha256": self_sha},
        "constants": {
            "input_root": {"path": str(INPUT_ROOT), "fingerprint_sha256": INPUT_FINGERPRINT_SHA, "member_count": 19},
            "B": {"path": str(BINDING_ROOT), "manifest_sha256": BINDING_MANIFEST_SHA},
            "R": {"path": str(RUNNER), "commit": RUNNER_COMMIT, "tree": RUNNER_TREE, "git_blob": RUNNER_GIT_BLOB, "sha256": RUNNER_SHA},
            "validator": {"path": str(VALIDATOR), "commit": VALIDATOR_COMMIT, "tree": VALIDATOR_TREE, "git_blob": VALIDATOR_GIT_BLOB, "sha256": VALIDATOR_SHA},
            "resident": {"path": str(RESIDENT_DRIVER), "commit": RESIDENT_COMMIT, "sha256": RESIDENT_SHA, "served_manifest": str(SERVED_MANIFEST), "served_sha256": SERVED_SHA, "device_index": DEVICE_INDEX, "lock_path": str(LOCK_PATH)},
            "case": {"case_id": CASE_ID, "case_sha256": CASE_SHA},
        },
        "sequence": [], "process_counts": {"launcher_validator": 0, "runner": 0, "runner_internal_validator": 0, "fake_driver": 0},
        "validator": None, "runner": None, "result": None, "failure": None,
        "safety": {"gpu_command_executed": False, "model_load_executed": False, "service_touched": False, "service_stopped": False},
    }


def finalize_output(output: Path, evidence: dict[str, Any]) -> None:
    atomic_write(output, "launcher-evidence.json", pretty(evidence))
    names = sorted(entry.name for entry in output.iterdir() if entry.name != "SHA256SUMS")
    lines = []
    for name in names:
        digest, _ = sha_file(output / name, f"launcher output {name}")
        lines.append(f"{digest}  {name}\n")
    atomic_write(output, "SHA256SUMS", "".join(lines).encode("ascii"))
    os.chmod(output, 0o555)


def launch(mode: str, output: Path, *, run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run) -> tuple[int, dict[str, Any]]:
    reject_symlink_components(output, "launcher evidence output", allow_missing_leaf=True)
    if output.exists() or output.is_symlink():
        raise LauncherError(f"launcher evidence output already exists: {output}")
    output.mkdir(mode=0o700, parents=False)
    self_sha, _ = sha_file(Path(__file__).resolve(), "launcher self")
    evidence = make_evidence(mode, self_sha)
    snapshot = Snapshot()
    stage = "constants"
    runner_started = False
    try:
        if mode != "dry-run":
            raise LauncherError("actual execution is disabled; only dry-run is authorized")
        validate_constants(snapshot)
        snapshot.file(Path(__file__).resolve(), self_sha, "launcher self")
        stage = "validator"
        validator_command = validator_argv()
        validator_completed = run(validator_command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence["process_counts"]["launcher_validator"] = 1
        evidence["sequence"].append("validator")
        evidence["validator"] = process_record(validator_command, validator_completed, "validator", output)
        if validator_completed.returncode != 0 or validator_completed.stderr:
            raise LauncherError("trusted validator subprocess rejected root/B")
        report = validate_validator_report(validator_completed.stdout)
        evidence["validator"]["report"] = report
        evidence["validator"]["report_sha256"] = sha_bytes(canonical(report))
        if _AFTER_VALIDATOR_HOOK is not None:
            _AFTER_VALIDATOR_HOOK()
        snapshot.verify()
        stage = "runner"
        if RUNNER_OUTPUT.exists() or RUNNER_OUTPUT.is_symlink():
            raise LauncherError(f"runner output already exists: {RUNNER_OUTPUT}")
        runner_command = runner_argv()
        runner_started = True
        runner_completed = run(runner_command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence["process_counts"]["runner"] = 1
        evidence["sequence"].append("runner")
        evidence["runner"] = process_record(runner_command, runner_completed, "runner", output)
        if runner_completed.returncode != 0 or runner_completed.stderr:
            raise LauncherError("trusted runner subprocess failed")
        plan_path = RUNNER_OUTPUT / "resident-batch.plan.json"
        plan_raw, _ = read_regular(plan_path, "runner result plan")
        validate_runner_plan(plan_raw)
        atomic_write(output, "runner-plan.json", plan_raw)
        evidence["runner"]["plan"] = {"file": "runner-plan.json", "sha256": sha_bytes(plan_raw)}
        evidence["result"] = {"kind": "dry_run_plan", "sha256": sha_bytes(plan_raw), "B_plan_match": True}
        evidence["process_counts"]["runner_internal_validator"] = 1
        evidence["process_counts"]["fake_driver"] = 1
        if _BEFORE_FINAL_VERIFY_HOOK is not None:
            _BEFORE_FINAL_VERIFY_HOOK()
        snapshot.verify()
        evidence["status"] = "passed"
        return_code = 0
    except (LauncherError, OSError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        evidence["failure"] = {"stage": stage, "reason": str(error), "runner_started": runner_started}
        return_code = 1
    finally:
        if RUNNER_OUTPUT.exists() and not RUNNER_OUTPUT.is_symlink():
            shutil.rmtree(RUNNER_OUTPUT)
        finalize_output(output, evidence)
    return return_code, evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dry-run", "execute"), default="dry-run")
    parser.add_argument("--evidence-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        code, evidence = launch(args.mode, args.evidence_output)
        print(json.dumps({"status": evidence["status"], "mode": evidence["mode"], "evidence": str(args.evidence_output / "launcher-evidence.json")}, sort_keys=True))
        return code
    except (LauncherError, OSError, ValueError) as error:
        print(f"AQ4 P2 immutable launcher failed before evidence creation: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
