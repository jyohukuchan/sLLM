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
PREPARED_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v2"
BINDING_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v7"
MAINTENANCE = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
SOURCE = Path(__file__).resolve()
PROFILE_READY_ROOT = P2 / "resident-one-case-smoke-profile-ready-v16"
PROFILE_READY = PROFILE_READY_ROOT / "ready-binding.json"
PROFILE_READY_DRY_RUN_ROOT = P2 / "resident-one-case-smoke-profile-ready-dry-run-v16"
HISTORICAL_READY_V15_ROOT = P2 / "resident-one-case-smoke-profile-ready-v15"
HISTORICAL_READY_V15 = HISTORICAL_READY_V15_ROOT / "ready-binding.json"
HISTORICAL_READY_DRY_RUN_V15_ROOT = P2 / "resident-one-case-smoke-profile-ready-dry-run-v15"
QUIET_ROOT = P2 / "resident-one-case-smoke-profile-quiet-window-v19"
OPERATOR_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v14"
MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v11"
OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v14"
ACTUAL_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v14"
PROFILE_RUNTIME = P2 / "resident-one-case-smoke-profile-execute-v10"
PROFILE_EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-profile-execute-evidence-v10"
PROFILE_CAPTURE = P3 / "aq4-p3-diagnostic-rocprof-capture-v10"
OFFLINE_CAPTURE_ROOT = P3 / "aq4-p3-diagnostic-rocprof-capture-offline-reassembly-v11"
OFFLINE_EVIDENCE_ROOT = P2 / "resident-one-case-smoke-profile-maintenance-offline-reassembly-evidence-v11"
PREVIOUS_QUIET_V18_ROOT = P2 / "resident-one-case-smoke-profile-quiet-window-v18"
PREVIOUS_OPERATOR_V13_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v13"
PREVIOUS_OPERATOR_ROOT = PREVIOUS_OPERATOR_V13_ROOT
PREVIOUS_OPERATOR_V12_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v12"
PREVIOUS_OPERATOR_V11_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v11"
PREVIOUS_OPERATOR_V10_ROOT = P2 / "resident-one-case-smoke-profile-operator-command-v10"
PREVIOUS_OPERATOR_RESULT_V10 = P2 / "resident-one-case-smoke-profile-operator-result-v10"
PREVIOUS_ACTUAL_AUDIT_V10 = P2 / "resident-one-case-smoke-profile-actual-audit-v10"
ACTUAL_V11_MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v9"
ACTUAL_V11_OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v11"
ACTUAL_V11_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v11"
ACTUAL_V11_PROFILE_RUNTIME = P2 / "resident-one-case-smoke-profile-execute-v9"
ACTUAL_V11_PROFILE_EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-profile-execute-evidence-v9"
ACTUAL_V11_PROFILE_CAPTURE = P3 / "aq4-p3-diagnostic-rocprof-capture-v9"
PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v10"
PREVIOUS_ACTUAL_V12_PROFILE_RUNTIME = P2 / "resident-one-case-smoke-profile-execute-v9"
PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-profile-execute-evidence-v9"
PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE = P3 / "aq4-p3-diagnostic-rocprof-capture-v9"
PREVIOUS_ACTUAL_V12_OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v12"
PREVIOUS_ACTUAL_V12_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v12"
PREVIOUS_V13_PROFILE_READY = P2 / "resident-one-case-smoke-profile-ready-v16/ready-binding.json"
PREVIOUS_V13_PYTHON = Path("/usr/bin/python3.12")
PREVIOUS_V13_MAINTENANCE = ROOT / "tools/run-aq4-p2-resident-smoke-maintenance.py"
PREVIOUS_V13_MAINTENANCE_EVIDENCE = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v11"
PREVIOUS_V13_PROFILE_RUNTIME = P2 / "resident-one-case-smoke-profile-execute-v10"
PREVIOUS_V13_PROFILE_EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-profile-execute-evidence-v10"
PREVIOUS_V13_PROFILE_CAPTURE = P3 / "aq4-p3-diagnostic-rocprof-capture-v10"
PREVIOUS_V13_OPERATOR_RESULT = P2 / "resident-one-case-smoke-profile-operator-result-v13"
PREVIOUS_V13_ACTUAL_AUDIT = P2 / "resident-one-case-smoke-profile-actual-audit-v13"
EXECUTE_BINDING_ROOT = P2 / "resident-one-case-smoke-execute-binding-v10"
EXECUTE_RUNTIME = P2 / "resident-one-case-smoke-execute-v10"
EXECUTE_EVIDENCE = P2 / "resident-one-case-smoke-execute-evidence-v10"
PYTHON = Path("/usr/bin/python3.12")
QUIET_SCHEMA = "ullm.aq4_p3_profile_quiet_window.v19"
OPERATOR_SCHEMA = "ullm.aq4_p3_profile_operator_command.v14"
OPERATOR_RESULT_SCHEMA = "ullm.aq4_p3_profile_operator_result.v14"
ACTUAL_AUDIT_SCHEMA = "ullm.aq4_p3_profile_actual_audit.v14"
PREVIOUS_QUIET_V18_SCHEMA = "ullm.aq4_p3_profile_quiet_window.v18"
PREVIOUS_OPERATOR_V13_SCHEMA = "ullm.aq4_p3_profile_operator_command.v13"
PREVIOUS_OPERATOR_V13_COMMIT = "764045355ee06c3b5c53f296d4bcbe47e1495ece"
PREVIOUS_OPERATOR_V13_TREE = "cb73e9c7c34c884eac567510f6d89da238b57a49"
PREVIOUS_OPERATOR_V13_ROOT_TREE = "d187b2902aa9f83503c17d6c0c8665210744f2e0"
PREVIOUS_OPERATOR_V13_MANIFEST_SHA256 = "78168089ff34e2eb8560bcaa85c94f49c0f3ae23ee4a614f0d0fc7e077a0d4f0"
PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256 = "42c8498adc6c8f97382ef17421d3145a14d50126a549a66d0693f114f8cad313"
PREVIOUS_OPERATOR_V13_SUMS_SHA256 = "1c157f9d864b4e75d62e2acc7b5b5189b1765e3795b3109ef4e815df26b87fd6"
PREVIOUS_OPERATOR_V13_COMMAND_SHA256 = "5693d75b17f91187b6841566815ad717d001a91280d651860aa127dc20277079"
PREVIOUS_QUIET_V18_COMMIT = "cb774ac0090380d4fff5b613a942fad9b3d106c8"
PREVIOUS_QUIET_V18_TREE = "add160bacc5f372cd21bbaa6840ebcb1735c94f4"
PREVIOUS_QUIET_V18_ROOT_TREE = "18c7e4c0c83142bab61be025022e77696c259ea7"
PREVIOUS_QUIET_V18_JSON_SHA256 = "0fb7e3346e7f38d0b9d844d3bac2815b533945eb7d25b3981ac3d5542eb36e00"
PREVIOUS_QUIET_V18_SUMS_SHA256 = "081e220fd195c3576eeced4d59464c309be4d1304bb5cfbc771cbe197c59608b"
PREVIOUS_OPERATOR_V12_SCHEMA = "ullm.aq4_p3_profile_operator_command.v12"
PREVIOUS_OPERATOR_RESULT_V12_SCHEMA = "ullm.aq4_p3_profile_operator_result.v12"
PREVIOUS_ACTUAL_AUDIT_V12_SCHEMA = "ullm.aq4_p3_profile_actual_audit.v12"
PREVIOUS_OPERATOR_V12_COMMIT = "2185ac90f7188402c60280e87b8eded3cbfc65e8"
PREVIOUS_OPERATOR_V12_TREE = "a6eef3569f960f909e1e07b78cb465152fe288a7"
PREVIOUS_OPERATOR_V12_MANIFEST_SHA256 = "5712168a29d708d0ce7578d81f15089fb1dbed400dbba84e55887a4ee0348944"
PREVIOUS_OPERATOR_V12_SUMS_SHA256 = "641d83c39957967fdcb39abedea901b11bd8eb214fe587f8f08cf9a0a858f396"
PREVIOUS_OPERATOR_V12_SEMANTIC_SHA256 = "5f9b5a8758fe1dd22446f88c140a5bed5738de440f253eedba2cb5a0668f5b27"
PREVIOUS_ACTUAL_V12_COMMIT = "44617f7fd46c39f71f04502b248739cc116fe095"
PREVIOUS_ACTUAL_V12_TREE = "813c4ffc88fb58cf8764b91d3c80cea9ef351f0f"
PREVIOUS_ACTUAL_V12_FILE_COUNT = 35
PREVIOUS_OPERATOR_V11_SCHEMA = "ullm.aq4_p3_profile_operator_command.v11"
PREVIOUS_OPERATOR_RESULT_V11_SCHEMA = "ullm.aq4_p3_profile_operator_result.v11"
PREVIOUS_ACTUAL_AUDIT_V11_SCHEMA = "ullm.aq4_p3_profile_actual_audit.v11"
PREVIOUS_OPERATOR_V11_COMMIT = "637ca8ed26e8cbb1200656ba4fb6ef1676b8282f"
PREVIOUS_OPERATOR_V11_TREE = "578f720472e0eef5b5607321e7a21df04fc72cf6"
PREVIOUS_OPERATOR_V11_MANIFEST_SHA256 = "4597826e0c876e3b51c756f65c99c2bb43ee395504b7fe9767eb324db1706102"
PREVIOUS_OPERATOR_V11_SUMS_SHA256 = "a3fcc93e45071224e880449e48e5471134f9f82a1f0dd6c8e77446f4f24e11d6"
PREVIOUS_OPERATOR_V11_SEMANTIC_SHA256 = "623730860c878b7652138bf54b8582677c48a346544244d0ee327b811d4b9387"
ACTUAL_V11_COMMIT = "854e5a348bd3c0f442f2371a0d3619308bce3b95"
ACTUAL_V11_TREE = "147bd97b595d8cea268c193e09e5c817ef6bdacc"
ACTUAL_V11_FILE_COUNT = 8
PREVIOUS_OPERATOR_V10_SCHEMA = "ullm.aq4_p3_profile_operator_command.v10"
PREVIOUS_OPERATOR_V10_COMMIT = "d278a2ba71a0f30c56c7af8927990eb4d6ac1e26"
PREVIOUS_OPERATOR_V10_TREE = "5a4d1b0a3a0e30c4befaef2f6e2cf355b3af3484"
PREVIOUS_OPERATOR_V10_MANIFEST_SHA256 = "05f457d3cf17cc57db50add9456714407c2a442b94f9a3aa567e5d594cc64cff"
PREVIOUS_OPERATOR_V10_SUMS_SHA256 = "7cd59f443e66667ba05fc7e1e2fb95326f8b60eda62ce2a3987d367bba8821c3"
HISTORICAL_ACTUAL_V9_COMMIT = "00358807d7f400d621c11e20b942ecd4fbbd656f"
HISTORICAL_ACTUAL_V9_TREE = "6f0f61be424057a9fd8ca3c455d565e6dc3a6c08"
HISTORICAL_ACTUAL_V9_FILE_COUNT = 35
HISTORICAL_OPERATOR_MANIFEST_V9_COMMIT = "2df19a16723df952c0be58a5cff4a1d86bb80d99"
HISTORICAL_OPERATOR_RESULT_V9_SCHEMA = "ullm.aq4_p3_profile_operator_result.v9"
HISTORICAL_ACTUAL_AUDIT_V9_SCHEMA = "ullm.aq4_p3_profile_actual_audit.v9"
HISTORICAL_MAINTENANCE_EVIDENCE_V8 = P2 / "resident-one-case-smoke-profile-maintenance-evidence-v8"
HISTORICAL_PROFILE_RUNTIME_V8 = P2 / "resident-one-case-smoke-profile-execute-v8"
HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8 = P2 / "resident-one-case-smoke-profile-execute-evidence-v8"
HISTORICAL_PROFILE_CAPTURE_V8 = P3 / "aq4-p3-diagnostic-rocprof-capture-v8"
HISTORICAL_OPERATOR_RESULT_V9 = P2 / "resident-one-case-smoke-profile-operator-result-v9"
HISTORICAL_ACTUAL_AUDIT_V9 = P2 / "resident-one-case-smoke-profile-actual-audit-v9"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
DEFAULT_INTERVAL = 5.0
DEFAULT_MAXIMUM = 900.0
DEFAULT_MINIMUM_SPAN = 130.0
DEFAULT_REQUIRED_SAMPLES = 27

READY_ARTIFACT_COMMIT = "09324284ab27d61642f126d8e052fa05c1cbb3cf"
READY_ARTIFACT_TREE = "984136dfc469d15394f00bba8e1adfca742ad30f"
READY_ROOT_TREE = "232c82816dcabf35a8861434b0af612ab2ced59c"
READY_BINDING_SHA256 = "54c218a203a19643eae8983bfb2ac84b8132341dc04d42e7dc30f080ea02e42d"
READY_SHA256SUMS_SHA256 = "76fc710b78d384c890f3a3c7c21dad7ed912299a21e20e431085e0d319e78686"
READY_DRY_RUN_ROOT_TREE = "461bdeb0449c474be8d791a9297853b038b8dc4e"
READY_DRY_RUN_EVIDENCE_SHA256 = "82d7a17c5d71c9e0e4019280a1f14a75c569ad8faa6bb2b00702088dd0d93f17"
READY_DRY_RUN_SHA256SUMS_SHA256 = "d6b11306091f1132ef1485bf01ee67005b5335ccb32060b6f25987fd8a7c7fc8"
HISTORICAL_READY_V15_COMMIT = "b39e21822db40e7fd5060da66db885b3a9ff0b8a"
HISTORICAL_READY_V15_TREE = "4daa8f0cafe93274aeddd902bea58727633b3080"
HISTORICAL_READY_V15_ROOT_TREE = "8045019bc2346efccc3c37781fc8bd6280e95dac"
HISTORICAL_READY_V15_BINDING_SHA256 = "4c2c2079fd428c8db156e36d0513726ae49e372927770d4d9aba0a0172b4497b"
HISTORICAL_READY_DRY_RUN_V15_ROOT_TREE = "b375ac9a0e55b738715dd637d38b864ccf6a2204"
HISTORICAL_READY_DRY_RUN_V15_SUMS_SHA256 = "86ab1e7714e05951a17e6a7584bf6183f68a1e009f289751810025f36329ec67"
HISTORICAL_READY_DRY_RUN_V15_EVIDENCE_SHA256 = "743941cfa6c580d9f6fc786a37b9e270f5ee0f8764bb8ffcbceefb0c79f535fd"
CURRENT_MAINTENANCE_COMMIT = "c4fe279e6c0bf9a8899c2cd36642f45bf145fe8f"
CURRENT_MAINTENANCE_TREE = "49685f2b9194d6128d8e92ad04d52c01540eed38"
CURRENT_MAINTENANCE_BLOB = "53ad6ab6eeec43eb77478397ad0fcd8c09caa45b"
CURRENT_MAINTENANCE_SHA256 = "4330469041c664454165844e2f1de452f207ddd27814876d4f35caf9775698c4"
CURRENT_MAINTENANCE_TEST_COMMIT = "6af8dfa47968fed55b1f198bb03409f496bfb6c1"
CURRENT_MAINTENANCE_TEST_BLOB = "6b77b2228f581e788d724aa1b540d611dd7b3b3b"
CURRENT_MAINTENANCE_TEST_SHA256 = "896d78be1430569ef911c25482761d85f60e9584604e5b229320689b98ca7f3b"
CURRENT_CAPTURE_COMMIT = "eb00cbd83b90d6fd8d519f6662ddea16d5f4438c"
CURRENT_CAPTURE_TREE = "545511060d95a02d69f4164d35bb56d89c22ea59"
CURRENT_CAPTURE_BLOB = "91f243ff5dcc0c36c63e471ac7c4581c74535a2f"
CURRENT_CAPTURE_SHA256 = "e326fb5c9f5ff04290fe0c37cfd25ad7e1e37bd7f76b5d7a62002465b9965df4"
OFFLINE_ARTIFACT_COMMIT = "aa26f4e85dbdf2bc000c32a9869fc22b6597e888"
OFFLINE_ARTIFACT_TREE = "79446c68a4e0c4b4782a37d6a48646f8583d92f3"
OFFLINE_CAPTURE_TREE = "141261cd4007e83d6a6ce5be36d404a374c1edfd"
OFFLINE_CAPTURE_SUMS_SHA256 = "5ae70242a6943e9ace4e4b64e0e5f0b81eac8d02335114ceb7c3a0b4c330bdbf"
OFFLINE_CAPTURE_ARTIFACT_SHA256 = "ea0c73aa03dc89e173a274790a050a48fb26384990d14f4bbb4b326aa0179e28"
OFFLINE_EVIDENCE_TREE = "e3d87086b30406abd051eddbb4eddc77af5332ce"
OFFLINE_EVIDENCE_SUMS_SHA256 = "02972beff0ef06a310ad19ad9c7ddf62f69196ac09fa6a231a3e136ccb51f2c7"
OFFLINE_EVIDENCE_JSON_SHA256 = "7b294b6c3d8beb28cb3e9682d852d23cb9e3c91a2433128aa380101c32639a36"
EXECUTE_BINDING_ARTIFACT_COMMIT = "2b477ed0dd1344d368e684e413cb756706af22f3"
EXECUTE_BINDING_ARTIFACT_TREE = "bcc014d925e9d5c6b334496f6060959fe343decb"
EXECUTE_BINDING_ROOT_TREE = "0a1ea5664829bb7257bff097c551c3f625aeef6a"
EXECUTE_BINDING_SHA256SUMS_SHA256 = "059ab6bab846f94b511a3d602a8cca350a328cf11b7dcf0f50a5ae8407b698de"
EXECUTE_BINDING_MANIFEST_SHA256 = "6fb8e61d4460ab89fdd643e917c7c20d1ddd9a68b1292703f0a2bd4d86ecef06"
EXECUTE_LAUNCHER_COMMIT = "fc4559ee4fb8c7c1e62353fb3978a1a1e0a7d86d"
EXECUTE_LAUNCHER_TREE = "a5f938243463e36e401787aa62dfa6a5ef46e125"
EXECUTE_LAUNCHER_BLOB = "debace42c2063c476a9db3dcfe7fdf480bdf5088"
EXECUTE_LAUNCHER_SHA256 = "5197efa84ec98343dda9438e4c0bc31e144765ce686a4b41199f1ae0315de8a6"
EXECUTE_LAUNCHER_TRUST_SHA256 = "33182ae19350cc7ed0a8fe3b439746a81996dc70a5d6d355fb0aac323e75dd6c"


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


def git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise OperatorError(f"Git command failed: {' '.join(args)}")
    return completed.stdout


def sealed_mode_manifest(root: Path) -> dict[str, set[str]]:
    """Return immutable member roles for a known sealed artifact path."""
    if root == PREPARED_ROOT:
        return {"executable": {"resident-driver"}}
    if root == BINDING_ROOT:
        return {"executable": set()}
    return {"executable": set()}


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
    mode_manifest = sealed_mode_manifest(root)
    executable_members = mode_manifest["executable"]
    if not executable_members.issubset(expected):
        raise OperatorError(f"sealed role coverage differs: {root}")
    members: dict[str, Any] = {}
    for name in sorted(expected):
        path = root / name
        child = path.lstat()
        expected_mode = 0o555 if name in executable_members else 0o444
        if not stat.S_ISREG(child.st_mode) or child.st_nlink != 1 or stat.S_IMODE(child.st_mode) != expected_mode or sha_file(path) != declared[name]:
            raise OperatorError(f"sealed member differs: {path}")
        members[name] = {"path": str(path), "sha256": declared[name], "mode": f"0{expected_mode:o}", "nlink": 1, "size": child.st_size}
    return {"root": str(root), "mode": "0555", "sha256sums_sha256": sha_file(sums), "members": members}


def verify_inventory_commit(root: Path, inventory: dict[str, Any], commit: str) -> None:
    paths = [root / "SHA256SUMS", *(Path(item["path"]) for item in inventory["members"].values())]
    for path in paths:
        relative = str(path.relative_to(ROOT))
        if git("rev-parse", f"{commit}:{relative}") != git("hash-object", str(path)):
            raise OperatorError(f"sealed Git authority differs: {path}")


def previous_authorization_v10_fresh_paths() -> list[Path]:
    paths = [
        PROFILE_RUNTIME,
        PROFILE_EXECUTE_EVIDENCE,
        MAINTENANCE_EVIDENCE,
        PROFILE_CAPTURE,
        PROFILE_CAPTURE / "capture-artifact.json",
        PROFILE_CAPTURE / "rocprof.stdout",
        PROFILE_CAPTURE / "rocprof.stderr",
        PREVIOUS_OPERATOR_RESULT_V10,
        PREVIOUS_ACTUAL_AUDIT_V10,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("previous operator-v10 fresh output set differs")
    return paths


def previous_authorization_v10_state() -> dict[str, Any]:
    inventory = verify_sums(PREVIOUS_OPERATOR_V10_ROOT)
    manifest_path = PREVIOUS_OPERATOR_V10_ROOT / "command-manifest.json"
    if (
        inventory["sha256sums_sha256"] != PREVIOUS_OPERATOR_V10_SUMS_SHA256
        or sha_file(manifest_path) != PREVIOUS_OPERATOR_V10_MANIFEST_SHA256
        or git("rev-parse", f"{PREVIOUS_OPERATOR_V10_COMMIT}^{{tree}}")
        != PREVIOUS_OPERATOR_V10_TREE
    ):
        raise OperatorError("previous operator-v10 authority differs")
    verify_inventory_commit(
        PREVIOUS_OPERATOR_V10_ROOT,
        inventory,
        PREVIOUS_OPERATOR_V10_COMMIT,
    )
    root_relative = str(PREVIOUS_OPERATOR_V10_ROOT.relative_to(ROOT))
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                PREVIOUS_OPERATOR_V10_COMMIT,
                "--",
                root_relative,
            ).splitlines(),
        )
    )
    expected = {
        f"{root_relative}/SHA256SUMS",
        f"{root_relative}/command-manifest.json",
    }
    if observed != expected:
        raise OperatorError("previous operator-v10 Git file coverage differs")

    value = load(manifest_path, "previous operator-v10 manifest")
    clone = json.loads(json.dumps(value))
    declared = clone.get("manifest_sha256")
    clone["manifest_sha256"] = None
    authorization = value.get("authorization", {})
    execution = value.get("execution", {})
    if (
        value.get("schema_version") != PREVIOUS_OPERATOR_V10_SCHEMA
        or declared != sha_bytes(canonical(clone))
        or value.get("argv") != actual_argv()
        or value.get("command_sha256") != sha_bytes(canonical(actual_argv()))
        or authorization.get("maximum_invocations") != 1
        or authorization.get("explicit_confirmation_flag_count") != 1
        or authorization.get("profile_diagnostic_flag_count") != 1
        or authorization.get("ready_artifact_flag_count") != 1
        or authorization.get("evidence_output_flag_count") != 1
        or execution.get("maximum_invocations") != 1
        or execution.get("shell") is not False
        or execution.get("requires_fresh_output_recheck_immediately_before_execution")
        is not True
        or value.get("actual_executed") is not False
        or value.get("gpu_command_executed") is not False
        or value.get("service_touched") is not False
        or value.get("secret_material_embedded") is not False
    ):
        raise OperatorError("previous operator-v10 semantic authority differs")

    paths = previous_authorization_v10_fresh_paths()
    declared_outputs = value.get("fresh_outputs")
    if (
        not isinstance(declared_outputs, list)
        or declared_outputs
        != [{"path": str(path), "absent": True} for path in paths]
    ):
        raise OperatorError("previous operator-v10 fresh authorization differs")
    present = [path.exists() or path.is_symlink() for path in paths]
    if any(present):
        raise OperatorError("previous operator-v10 partial outputs are present")
    state = [
        {"path": str(path), "present": observed}
        for path, observed in zip(paths, present, strict=True)
    ]
    return {
        "state": "authorized_not_invoked_preflight_blocked",
        "authorization_commit": PREVIOUS_OPERATOR_V10_COMMIT,
        "authorization_tree": PREVIOUS_OPERATOR_V10_TREE,
        "manifest_file_sha256": PREVIOUS_OPERATOR_V10_MANIFEST_SHA256,
        "manifest_semantic_sha256": declared,
        "inventory": inventory,
        "fresh_outputs": state,
        "invocation_count": 0,
        "maximum_invocations": 1,
        "result_present": False,
        "audit_present": False,
        "actual_executed": False,
    }


def previous_operator_v11_argv() -> list[str]:
    return [
        str(PYTHON),
        str(MAINTENANCE),
        "--mode",
        "execute",
        "--profile-diagnostic",
        "--ready-artifact",
        str(P2 / "resident-one-case-smoke-profile-ready-v12/ready-binding.json"),
        "--evidence-output",
        str(ACTUAL_V11_MAINTENANCE_EVIDENCE),
        "--confirm-one-case",
    ]


def previous_operator_v11_fresh_paths() -> list[Path]:
    paths = [
        ACTUAL_V11_PROFILE_RUNTIME,
        ACTUAL_V11_PROFILE_EXECUTE_EVIDENCE,
        ACTUAL_V11_MAINTENANCE_EVIDENCE,
        ACTUAL_V11_PROFILE_CAPTURE,
        ACTUAL_V11_PROFILE_CAPTURE / "capture-artifact.json",
        ACTUAL_V11_PROFILE_CAPTURE / "rocprof.stdout",
        ACTUAL_V11_PROFILE_CAPTURE / "rocprof.stderr",
        ACTUAL_V11_OPERATOR_RESULT,
        ACTUAL_V11_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9:
        raise OperatorError("previous operator-v11 fresh output set differs")
    return paths


def previous_operator_v11_state() -> dict[str, Any]:
    inventory = verify_sums(PREVIOUS_OPERATOR_V11_ROOT)
    manifest_path = PREVIOUS_OPERATOR_V11_ROOT / "command-manifest.json"
    if (
        inventory["sha256sums_sha256"] != PREVIOUS_OPERATOR_V11_SUMS_SHA256
        or sha_file(manifest_path) != PREVIOUS_OPERATOR_V11_MANIFEST_SHA256
        or git("rev-parse", f"{PREVIOUS_OPERATOR_V11_COMMIT}^{{tree}}")
        != PREVIOUS_OPERATOR_V11_TREE
    ):
        raise OperatorError("previous operator-v11 authority differs")
    verify_inventory_commit(
        PREVIOUS_OPERATOR_V11_ROOT,
        inventory,
        PREVIOUS_OPERATOR_V11_COMMIT,
    )
    relative = str(PREVIOUS_OPERATOR_V11_ROOT.relative_to(ROOT))
    observed = set(filter(None, git("ls-tree", "-r", "--name-only", PREVIOUS_OPERATOR_V11_COMMIT, "--", relative).splitlines()))
    if observed != {f"{relative}/SHA256SUMS", f"{relative}/command-manifest.json"}:
        raise OperatorError("previous operator-v11 Git file coverage differs")

    value = load(manifest_path, "previous operator-v11 manifest")
    clone = json.loads(json.dumps(value))
    declared = clone.get("manifest_sha256")
    clone["manifest_sha256"] = None
    authorization = value.get("authorization", {})
    execution = value.get("execution", {})
    argv = previous_operator_v11_argv()
    if (
        value.get("schema_version") != PREVIOUS_OPERATOR_V11_SCHEMA
        or declared != PREVIOUS_OPERATOR_V11_SEMANTIC_SHA256
        or declared != sha_bytes(canonical(clone))
        or value.get("argv") != argv
        or value.get("command_sha256") != sha_bytes(canonical(argv))
        or authorization.get("maximum_invocations") != 1
        or authorization.get("explicit_confirmation_flag_count") != 1
        or authorization.get("profile_diagnostic_flag_count") != 1
        or authorization.get("ready_artifact_flag_count") != 1
        or authorization.get("evidence_output_flag_count") != 1
        or execution.get("maximum_invocations") != 1
        or execution.get("shell") is not False
        or execution.get("requires_fresh_output_recheck_immediately_before_execution") is not True
        or value.get("actual_executed") is not False
        or value.get("gpu_command_executed") is not False
        or value.get("service_touched") is not False
        or value.get("secret_material_embedded") is not False
        or value.get("fresh_outputs")
        != [{"path": str(path), "absent": True} for path in previous_operator_v11_fresh_paths()]
    ):
        raise OperatorError("previous operator-v11 semantic authority differs")

    previous_v10 = value.get("inputs", {}).get("previous_operator_v10", {})
    historical_v9 = value.get("inputs", {}).get("historical_actual_v9", {})
    if (
        previous_v10.get("state") != "authorized_not_invoked_preflight_blocked"
        or previous_v10.get("authorization_commit") != PREVIOUS_OPERATOR_V10_COMMIT
        or previous_v10.get("authorization_tree") != PREVIOUS_OPERATOR_V10_TREE
        or previous_v10.get("manifest_file_sha256") != PREVIOUS_OPERATOR_V10_MANIFEST_SHA256
        or previous_v10.get("invocation_count") != 0
        or previous_v10.get("maximum_invocations") != 1
        or previous_v10.get("result_present") is not False
        or previous_v10.get("audit_present") is not False
        or historical_v9.get("state") != "executed_sealed"
        or historical_v9.get("artifact_commit") != HISTORICAL_ACTUAL_V9_COMMIT
        or historical_v9.get("artifact_tree") != HISTORICAL_ACTUAL_V9_TREE
        or historical_v9.get("invocation_count") != 1
        or historical_v9.get("retry_performed") is not False
    ):
        raise OperatorError("previous operator-v11 historical final-state binding differs")
    return {
        "state": "authorized_then_invoked_once_pre_stop_failed",
        "authorization_commit": PREVIOUS_OPERATOR_V11_COMMIT,
        "authorization_tree": PREVIOUS_OPERATOR_V11_TREE,
        "manifest_file_sha256": PREVIOUS_OPERATOR_V11_MANIFEST_SHA256,
        "manifest_semantic_sha256": declared,
        "inventory": inventory,
        "historical_operator_v10": previous_v10,
        "historical_actual_v9": historical_v9,
        "maximum_invocations": 1,
    }


def previous_operator_v12_argv() -> list[str]:
    return [
        str(PYTHON),
        str(MAINTENANCE),
        "--mode",
        "execute",
        "--profile-diagnostic",
        "--ready-artifact",
        str(P2 / "resident-one-case-smoke-profile-ready-v14/ready-binding.json"),
        "--evidence-output",
        str(PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE),
        "--confirm-one-case",
    ]


def previous_operator_v12_fresh_paths() -> list[Path]:
    paths = [
        PREVIOUS_ACTUAL_V12_PROFILE_RUNTIME,
        PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "capture-artifact.json",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stdout",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stderr",
        PREVIOUS_ACTUAL_V12_OPERATOR_RESULT,
        PREVIOUS_ACTUAL_V12_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("previous operator-v12 fresh output set differs")
    return paths


def previous_operator_v12_state() -> dict[str, Any]:
    inventory = verify_sums(PREVIOUS_OPERATOR_V12_ROOT)
    manifest_path = PREVIOUS_OPERATOR_V12_ROOT / "command-manifest.json"
    if (
        inventory["sha256sums_sha256"] != PREVIOUS_OPERATOR_V12_SUMS_SHA256
        or sha_file(manifest_path) != PREVIOUS_OPERATOR_V12_MANIFEST_SHA256
        or git("rev-parse", f"{PREVIOUS_OPERATOR_V12_COMMIT}^{{tree}}")
        != PREVIOUS_OPERATOR_V12_TREE
    ):
        raise OperatorError("previous operator-v12 authority differs")
    verify_inventory_commit(
        PREVIOUS_OPERATOR_V12_ROOT,
        inventory,
        PREVIOUS_OPERATOR_V12_COMMIT,
    )
    relative = str(PREVIOUS_OPERATOR_V12_ROOT.relative_to(ROOT))
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                PREVIOUS_OPERATOR_V12_COMMIT,
                "--",
                relative,
            ).splitlines(),
        )
    )
    if observed != {
        f"{relative}/SHA256SUMS",
        f"{relative}/command-manifest.json",
    }:
        raise OperatorError("previous operator-v12 Git file coverage differs")

    value = load(manifest_path, "previous operator-v12 manifest")
    clone = json.loads(json.dumps(value))
    declared = clone.get("manifest_sha256")
    clone["manifest_sha256"] = None
    authorization = value.get("authorization", {})
    execution = value.get("execution", {})
    argv = previous_operator_v12_argv()
    if (
        value.get("schema_version") != PREVIOUS_OPERATOR_V12_SCHEMA
        or declared != PREVIOUS_OPERATOR_V12_SEMANTIC_SHA256
        or declared != sha_bytes(canonical(clone))
        or value.get("argv") != argv
        or value.get("command_sha256") != sha_bytes(canonical(argv))
        or authorization.get("maximum_invocations") != 1
        or authorization.get("explicit_confirmation_flag_count") != 1
        or authorization.get("profile_diagnostic_flag_count") != 1
        or authorization.get("ready_artifact_flag_count") != 1
        or authorization.get("evidence_output_flag_count") != 1
        or execution.get("maximum_invocations") != 1
        or execution.get("shell") is not False
        or execution.get("requires_fresh_output_recheck_immediately_before_execution")
        is not True
        or value.get("actual_executed") is not False
        or value.get("gpu_command_executed") is not False
        or value.get("service_touched") is not False
        or value.get("secret_material_embedded") is not False
        or value.get("fresh_outputs")
        != [
            {"path": str(path), "absent": True}
            for path in previous_operator_v12_fresh_paths()
        ]
    ):
        raise OperatorError("previous operator-v12 semantic authority differs")

    inputs = value.get("inputs", {})
    previous_v11 = inputs.get("previous_operator_v11", {})
    actual_v11 = inputs.get("actual_v11", {})
    historical_v9 = inputs.get("historical_actual_v9", {})
    pre_audit = value.get("pre_execution_audit", {})
    if (
        previous_v11.get("state")
        != "authorized_then_invoked_once_pre_stop_failed"
        or previous_v11.get("authorization_commit")
        != PREVIOUS_OPERATOR_V11_COMMIT
        or actual_v11.get("state") != "pre_stop_failed_sealed"
        or actual_v11.get("artifact_commit") != ACTUAL_V11_COMMIT
        or historical_v9.get("state") != "executed_sealed"
        or historical_v9.get("artifact_commit") != HISTORICAL_ACTUAL_V9_COMMIT
        or pre_audit.get("previous_operator_v11")
        != "authorized_then_invoked_once_pre_stop_failed"
        or pre_audit.get("actual_v11") != "pre_stop_failed_sealed"
        or pre_audit.get("historical_actual_v9") != "executed_sealed"
        or pre_audit.get("actual_executed") is not False
    ):
        raise OperatorError("previous operator-v12 historical binding differs")
    return {
        "state": "authorized_sealed",
        "authorization_commit": PREVIOUS_OPERATOR_V12_COMMIT,
        "authorization_tree": PREVIOUS_OPERATOR_V12_TREE,
        "manifest_file_sha256": PREVIOUS_OPERATOR_V12_MANIFEST_SHA256,
        "manifest_semantic_sha256": declared,
        "inventory": inventory,
        "historical_operator_v11": previous_v11,
        "historical_actual_v11": actual_v11,
        "historical_actual_v9": historical_v9,
        "maximum_invocations": 1,
        "actual_executed": False,
    }


def previous_quiet_v18_authority() -> dict[str, Any]:
    inventory = verify_sums(PREVIOUS_QUIET_V18_ROOT)
    relative = str(PREVIOUS_QUIET_V18_ROOT.relative_to(ROOT))
    quiet_path = PREVIOUS_QUIET_V18_ROOT / "quiet-window.json"
    if (
        git("rev-parse", f"{PREVIOUS_QUIET_V18_COMMIT}^{{tree}}")
        != PREVIOUS_QUIET_V18_TREE
        or git("rev-parse", f"{PREVIOUS_QUIET_V18_COMMIT}:{relative}")
        != PREVIOUS_QUIET_V18_ROOT_TREE
        or inventory["sha256sums_sha256"] != PREVIOUS_QUIET_V18_SUMS_SHA256
        or sha_file(quiet_path) != PREVIOUS_QUIET_V18_JSON_SHA256
    ):
        raise OperatorError("previous quiet-v18 authority differs")
    verify_inventory_commit(
        PREVIOUS_QUIET_V18_ROOT,
        inventory,
        PREVIOUS_QUIET_V18_COMMIT,
    )
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                PREVIOUS_QUIET_V18_COMMIT,
                "--",
                relative,
            ).splitlines(),
        )
    )
    if observed != {
        f"{relative}/SHA256SUMS",
        f"{relative}/quiet-window.json",
    }:
        raise OperatorError("previous quiet-v18 Git file coverage differs")
    value = load(quiet_path, "previous quiet-v18")
    policy = value.get("policy", {})
    summary = value.get("summary", {})
    if (
        value.get("schema_version") != PREVIOUS_QUIET_V18_SCHEMA
        or value.get("status") != "go"
        or value.get("decision") != "GO"
        or value.get("resets") != []
        or policy
        != {
            "interval_seconds": 5.0,
            "maximum_monitoring_seconds": 900.0,
            "minimum_sample_span_seconds": 130.0,
            "required_consecutive_clean_samples": 27,
            "reset_count_required": 0,
        }
        or summary.get("sample_count") != 27
        or summary.get("final_streak_samples") != 27
        or summary.get("final_streak_span_seconds", 0.0) < 130.0
        or summary.get("reset_count") != 0
        or summary.get("confirmation_passed") is not True
        or summary.get("fresh_outputs_absent") is not True
        or value.get("read_only") is not True
        or value.get("actual_executed") is not False
        or value.get("gpu_command_executed") is not False
        or value.get("service_touched") is not False
        or value.get("secret_material_recorded") is not False
    ):
        raise OperatorError("previous quiet-v18 semantic authority differs")
    return {
        "status": value["status"],
        "decision": value["decision"],
        "artifact_commit": PREVIOUS_QUIET_V18_COMMIT,
        "artifact_tree": PREVIOUS_QUIET_V18_TREE,
        "root_tree": PREVIOUS_QUIET_V18_ROOT_TREE,
        "json_sha256": PREVIOUS_QUIET_V18_JSON_SHA256,
        "inventory": inventory,
        "summary": summary,
    }


def previous_operator_v13_argv() -> list[str]:
    return [
        str(PREVIOUS_V13_PYTHON),
        str(PREVIOUS_V13_MAINTENANCE),
        "--mode",
        "execute",
        "--profile-diagnostic",
        "--ready-artifact",
        str(PREVIOUS_V13_PROFILE_READY),
        "--evidence-output",
        str(PREVIOUS_V13_MAINTENANCE_EVIDENCE),
        "--confirm-one-case",
    ]


def previous_operator_v13_fresh_paths() -> list[Path]:
    paths = [
        PREVIOUS_V13_PROFILE_RUNTIME,
        PREVIOUS_V13_PROFILE_EXECUTE_EVIDENCE,
        PREVIOUS_V13_MAINTENANCE_EVIDENCE,
        PREVIOUS_V13_PROFILE_CAPTURE,
        PREVIOUS_V13_PROFILE_CAPTURE / "capture-artifact.json",
        PREVIOUS_V13_PROFILE_CAPTURE / "rocprof.stdout",
        PREVIOUS_V13_PROFILE_CAPTURE / "rocprof.stderr",
        PREVIOUS_V13_OPERATOR_RESULT,
        PREVIOUS_V13_ACTUAL_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("previous operator-v13 fresh output set differs")
    return paths


def previous_authorization_v13_state() -> dict[str, Any]:
    inventory = verify_sums(PREVIOUS_OPERATOR_V13_ROOT)
    manifest_path = PREVIOUS_OPERATOR_V13_ROOT / "command-manifest.json"
    relative = str(PREVIOUS_OPERATOR_V13_ROOT.relative_to(ROOT))
    if (
        git("rev-parse", f"{PREVIOUS_OPERATOR_V13_COMMIT}^{{tree}}")
        != PREVIOUS_OPERATOR_V13_TREE
        or git("rev-parse", f"{PREVIOUS_OPERATOR_V13_COMMIT}:{relative}")
        != PREVIOUS_OPERATOR_V13_ROOT_TREE
        or inventory["sha256sums_sha256"] != PREVIOUS_OPERATOR_V13_SUMS_SHA256
        or sha_file(manifest_path) != PREVIOUS_OPERATOR_V13_MANIFEST_SHA256
    ):
        raise OperatorError("previous operator-v13 authority differs")
    verify_inventory_commit(
        PREVIOUS_OPERATOR_V13_ROOT,
        inventory,
        PREVIOUS_OPERATOR_V13_COMMIT,
    )
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                PREVIOUS_OPERATOR_V13_COMMIT,
                "--",
                relative,
            ).splitlines(),
        )
    )
    if observed != {
        f"{relative}/SHA256SUMS",
        f"{relative}/command-manifest.json",
    }:
        raise OperatorError("previous operator-v13 Git file coverage differs")

    value = load(manifest_path, "previous operator-v13 manifest")
    clone = json.loads(json.dumps(value))
    declared = clone.get("manifest_sha256")
    clone["manifest_sha256"] = None
    authorization = value.get("authorization", {})
    execution = value.get("execution", {})
    argv = previous_operator_v13_argv()
    paths = previous_operator_v13_fresh_paths()
    if (
        value.get("schema_version") != PREVIOUS_OPERATOR_V13_SCHEMA
        or value.get("status")
        != "audited_ready_for_single_explicit_profile_diagnostic"
        or declared != PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256
        or declared != sha_bytes(canonical(clone))
        or value.get("argv") != argv
        or value.get("command_sha256") != PREVIOUS_OPERATOR_V13_COMMAND_SHA256
        or value.get("command_sha256") != sha_bytes(canonical(argv))
        or authorization.get("maximum_invocations") != 1
        or authorization.get("explicit_confirmation_flag_count") != 1
        or authorization.get("profile_diagnostic_flag_count") != 1
        or authorization.get("ready_artifact_flag_count") != 1
        or authorization.get("evidence_output_flag_count") != 1
        or execution.get("maximum_invocations") != 1
        or execution.get("shell") is not False
        or execution.get("requires_fresh_output_recheck_immediately_before_execution")
        is not True
        or value.get("fresh_outputs")
        != [{"path": str(path), "absent": True} for path in paths]
        or value.get("actual_executed") is not False
        or value.get("gpu_command_executed") is not False
        or value.get("service_touched") is not False
        or value.get("secret_material_embedded") is not False
    ):
        raise OperatorError("previous operator-v13 semantic authority differs")

    quiet = previous_quiet_v18_authority()
    quiet_input = value.get("inputs", {}).get("quiet_window", {})
    previous_v12 = value.get("inputs", {}).get("previous_operator_v12", {})
    previous_actual = value.get("inputs", {}).get("previous_actual_v12", {})
    if (
        quiet_input
        != {
            "path": str(PREVIOUS_QUIET_V18_ROOT / "quiet-window.json"),
            "sha256": PREVIOUS_QUIET_V18_JSON_SHA256,
            "decision": "GO",
            "status": "go",
        }
        or value.get("quiet_final_streak") != quiet["summary"]
        or previous_v12.get("state") != "authorized_sealed"
        or previous_v12.get("authorization_commit")
        != PREVIOUS_OPERATOR_V12_COMMIT
        or previous_actual.get("state") != "executed_sealed"
        or previous_actual.get("artifact_commit") != PREVIOUS_ACTUAL_V12_COMMIT
        or previous_actual.get("artifact_tree") != PREVIOUS_ACTUAL_V12_TREE
        or previous_actual.get("file_count") != PREVIOUS_ACTUAL_V12_FILE_COUNT
        or previous_actual.get("invocation_count") != 1
        or previous_actual.get("maximum_invocations") != 1
        or previous_actual.get("retry_performed") is not False
    ):
        raise OperatorError("previous operator-v13 input authority differs")

    present = [path.exists() or path.is_symlink() for path in paths]
    if any(present):
        raise OperatorError("previous operator-v13 partial outputs are present")
    state = [
        {"path": str(path), "present": observed}
        for path, observed in zip(paths, present, strict=True)
    ]
    return {
        "state": "authorized_not_invoked_preflight_blocked",
        "reason": "external_owner_after_seal_before_invocation",
        "authorization_commit": PREVIOUS_OPERATOR_V13_COMMIT,
        "authorization_tree": PREVIOUS_OPERATOR_V13_TREE,
        "authorization_root_tree": PREVIOUS_OPERATOR_V13_ROOT_TREE,
        "manifest_file_sha256": PREVIOUS_OPERATOR_V13_MANIFEST_SHA256,
        "manifest_semantic_sha256": declared,
        "command_sha256": PREVIOUS_OPERATOR_V13_COMMAND_SHA256,
        "inventory": inventory,
        "quiet_v18": quiet,
        "fresh_outputs": state,
        "invocation_count": 0,
        "maximum_invocations": 1,
        "result_present": False,
        "audit_present": False,
        "actual_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
    }


def verify_current_source_authority(
    path: Path,
    commit: str,
    tree: str,
    blob: str,
    raw_sha256: str,
) -> None:
    relative = str(path.relative_to(ROOT))
    if (
        path.is_symlink()
        or not path.is_file()
        or path.lstat().st_nlink != 1
        or sha_file(path) != raw_sha256
        or git("rev-parse", f"{commit}^{{tree}}") != tree
        or git("rev-parse", f"{commit}:{relative}") != blob
        or git("hash-object", str(path)) != blob
        or git("log", "-1", "--format=%H", "--", relative) != commit
    ):
        raise OperatorError(f"current source authority differs: {path}")


def historical_ready_v15_authority() -> dict[str, Any]:
    inventory = verify_sums(HISTORICAL_READY_V15_ROOT)
    dry_inventory = verify_sums(HISTORICAL_READY_DRY_RUN_V15_ROOT)
    relative = str(HISTORICAL_READY_V15_ROOT.relative_to(ROOT))
    dry_relative = str(HISTORICAL_READY_DRY_RUN_V15_ROOT.relative_to(ROOT))
    if (
        sha_file(HISTORICAL_READY_V15) != HISTORICAL_READY_V15_BINDING_SHA256
        or dry_inventory["sha256sums_sha256"]
        != HISTORICAL_READY_DRY_RUN_V15_SUMS_SHA256
        or sha_file(
            HISTORICAL_READY_DRY_RUN_V15_ROOT / "launcher-evidence.json"
        )
        != HISTORICAL_READY_DRY_RUN_V15_EVIDENCE_SHA256
        or git("rev-parse", f"{HISTORICAL_READY_V15_COMMIT}^{{tree}}")
        != HISTORICAL_READY_V15_TREE
        or git("rev-parse", f"{HISTORICAL_READY_V15_COMMIT}:{relative}")
        != HISTORICAL_READY_V15_ROOT_TREE
        or git("rev-parse", f"{HISTORICAL_READY_V15_COMMIT}:{dry_relative}")
        != HISTORICAL_READY_DRY_RUN_V15_ROOT_TREE
    ):
        raise OperatorError("historical profile-ready-v15 authority differs")
    verify_inventory_commit(
        HISTORICAL_READY_V15_ROOT,
        inventory,
        HISTORICAL_READY_V15_COMMIT,
    )
    verify_inventory_commit(
        HISTORICAL_READY_DRY_RUN_V15_ROOT,
        dry_inventory,
        HISTORICAL_READY_V15_COMMIT,
    )
    maintenance = load_maintenance()
    value = maintenance.load_ready_artifact(HISTORICAL_READY_V15)
    if value != load(HISTORICAL_READY_V15, "historical profile ready binding"):
        raise OperatorError("historical profile-ready-v15 readback differs")
    dry = load(
        HISTORICAL_READY_DRY_RUN_V15_ROOT / "launcher-evidence.json",
        "historical profile ready dry-run evidence",
    )
    if (
        dry.get("status") != "passed"
        or dry.get("mode") != "dry-run"
        or dry.get("gpu_command_executed") is not False
        or dry.get("service_touched") is not False
        or not isinstance(dry.get("process_counts"), dict)
        or any(count != 0 for count in dry["process_counts"].values())
    ):
        raise OperatorError("historical profile-ready-v15 dry-run differs")
    return value


def ready_authority() -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = verify_sums(PROFILE_READY_ROOT)
    dry_inventory = verify_sums(PROFILE_READY_DRY_RUN_ROOT)
    ready_relative = str(PROFILE_READY_ROOT.relative_to(ROOT))
    dry_relative = str(PROFILE_READY_DRY_RUN_ROOT.relative_to(ROOT))
    if (
        sha_file(PROFILE_READY) != READY_BINDING_SHA256
        or inventory["sha256sums_sha256"] != READY_SHA256SUMS_SHA256
        or sha_file(PROFILE_READY_DRY_RUN_ROOT / "launcher-evidence.json")
        != READY_DRY_RUN_EVIDENCE_SHA256
        or dry_inventory["sha256sums_sha256"]
        != READY_DRY_RUN_SHA256SUMS_SHA256
    ):
        raise OperatorError("profile-ready-v16 hashes differ")
    if (
        git("rev-parse", f"{READY_ARTIFACT_COMMIT}^{{tree}}")
        != READY_ARTIFACT_TREE
        or git("rev-parse", f"{READY_ARTIFACT_COMMIT}:{ready_relative}")
        != READY_ROOT_TREE
        or git("rev-parse", f"{READY_ARTIFACT_COMMIT}:{dry_relative}")
        != READY_DRY_RUN_ROOT_TREE
    ):
        raise OperatorError("profile-ready-v16 Git tree differs")
    verify_inventory_commit(PROFILE_READY_ROOT, inventory, READY_ARTIFACT_COMMIT)
    verify_inventory_commit(
        PROFILE_READY_DRY_RUN_ROOT,
        dry_inventory,
        READY_ARTIFACT_COMMIT,
    )
    ready = load(PROFILE_READY, "profile ready binding")
    dry = load(
        PROFILE_READY_DRY_RUN_ROOT / "launcher-evidence.json",
        "profile ready dry-run evidence",
    )
    profile_output = ready.get("profile_diagnostic", {}).get("output", {})
    launcher_binding = ready.get("launcher_binding", {})
    if (
        ready.get("schema_version")
        != "ullm.aq4_p2_resident_smoke_ready_binding.v1"
        or ready.get("status") != "ready_for_one_case"
        or ready.get("actual_eligible") is not True
        or ready.get("authorization", {}).get("run_id")
        != "p2-r9700-resident-one-case-smoke-profile-diagnostic-v10"
        or launcher_binding.get("runner_output") != str(PROFILE_RUNTIME)
        or launcher_binding.get("evidence_output")
        != str(PROFILE_EXECUTE_EVIDENCE)
        or profile_output.get("directory") != str(PROFILE_CAPTURE)
        or profile_output.get("artifact")
        != str(PROFILE_CAPTURE / "capture-artifact.json")
        or profile_output.get("must_not_exist_before_capture") is not True
        or dry.get("status") != "passed"
        or dry.get("mode") != "dry-run"
        or dry.get("gpu_command_executed") is not False
        or dry.get("service_touched") is not False
        or not isinstance(dry.get("process_counts"), dict)
        or any(count != 0 for count in dry["process_counts"].values())
    ):
        raise OperatorError("profile-ready-v16 semantic authority differs")
    trust = load(PROFILE_READY_ROOT / "harness-trust.json", "profile harness trust")
    qa = load(PROFILE_READY_ROOT / "qa-attestation.json", "profile QA attestation")
    maintenance_test = next(
        (
            item
            for suite in qa.get("automated_tests", {}).get("suites", [])
            for item in suite.get("files", [])
            if item.get("path")
            == "tests/test_aq4_p2_resident_smoke_maintenance.py"
        ),
        {},
    )
    if (
        trust.get("commit") != CURRENT_MAINTENANCE_COMMIT
        or trust.get("tree") != CURRENT_MAINTENANCE_TREE
        or trust.get("git_blob") != CURRENT_MAINTENANCE_BLOB
        or trust.get("sha256") != CURRENT_MAINTENANCE_SHA256
        or qa.get("automated_tests", {}).get("aggregate")
        != {
            "distinct_test_file_count": 13,
            "collected": 690,
            "passed": 690,
            "failed": 0,
            "deselected": 0,
        }
        or maintenance_test.get("source_commit")
        != CURRENT_MAINTENANCE_TEST_COMMIT
        or maintenance_test.get("git_blob") != CURRENT_MAINTENANCE_TEST_BLOB
        or maintenance_test.get("collected") != 170
        or maintenance_test.get("passed") != 170
    ):
        raise OperatorError("profile-ready-v16 current source/QA authority differs")
    verify_current_source_authority(
        MAINTENANCE,
        CURRENT_MAINTENANCE_COMMIT,
        CURRENT_MAINTENANCE_TREE,
        CURRENT_MAINTENANCE_BLOB,
        CURRENT_MAINTENANCE_SHA256,
    )
    verify_current_source_authority(
        ROOT / "tests/test_aq4_p2_resident_smoke_maintenance.py",
        CURRENT_MAINTENANCE_TEST_COMMIT,
        git("rev-parse", f"{CURRENT_MAINTENANCE_TEST_COMMIT}^{{tree}}"),
        CURRENT_MAINTENANCE_TEST_BLOB,
        CURRENT_MAINTENANCE_TEST_SHA256,
    )
    if load_maintenance().load_ready_artifact(PROFILE_READY) != ready:
        raise OperatorError("profile-ready-v16 formal readback differs")
    return ready, inventory


def offline_reassembly_authority() -> dict[str, Any]:
    capture_inventory = verify_sums(OFFLINE_CAPTURE_ROOT)
    evidence_inventory = verify_sums(OFFLINE_EVIDENCE_ROOT)
    capture_relative = str(OFFLINE_CAPTURE_ROOT.relative_to(ROOT))
    evidence_relative = str(OFFLINE_EVIDENCE_ROOT.relative_to(ROOT))
    if (
        git("rev-parse", f"{OFFLINE_ARTIFACT_COMMIT}^{{tree}}")
        != OFFLINE_ARTIFACT_TREE
        or git("rev-parse", f"{OFFLINE_ARTIFACT_COMMIT}:{capture_relative}")
        != OFFLINE_CAPTURE_TREE
        or git("rev-parse", f"{OFFLINE_ARTIFACT_COMMIT}:{evidence_relative}")
        != OFFLINE_EVIDENCE_TREE
        or capture_inventory["sha256sums_sha256"]
        != OFFLINE_CAPTURE_SUMS_SHA256
        or evidence_inventory["sha256sums_sha256"]
        != OFFLINE_EVIDENCE_SUMS_SHA256
        or sha_file(OFFLINE_CAPTURE_ROOT / "capture-artifact.json")
        != OFFLINE_CAPTURE_ARTIFACT_SHA256
        or sha_file(OFFLINE_EVIDENCE_ROOT / "offline-reassembly.json")
        != OFFLINE_EVIDENCE_JSON_SHA256
    ):
        raise OperatorError("offline reassembly-v11 authority differs")
    verify_inventory_commit(
        OFFLINE_CAPTURE_ROOT,
        capture_inventory,
        OFFLINE_ARTIFACT_COMMIT,
    )
    verify_inventory_commit(
        OFFLINE_EVIDENCE_ROOT,
        evidence_inventory,
        OFFLINE_ARTIFACT_COMMIT,
    )
    verify_current_source_authority(
        ROOT / "tools/capture-aq4-p3-diagnostic-profile.py",
        CURRENT_CAPTURE_COMMIT,
        CURRENT_CAPTURE_TREE,
        CURRENT_CAPTURE_BLOB,
        CURRENT_CAPTURE_SHA256,
    )
    value = load_maintenance().validate_profile_offline_reassembly()
    if (
        value.get("status") != "offline_reassembled_sealed"
        or value.get("generator", {}).get("commit")
        != CURRENT_MAINTENANCE_COMMIT
        or value.get("source_actual_seal", {}).get("member_count") != 35
        or value.get("output", {}).get("root") != str(OFFLINE_CAPTURE_ROOT)
        or value.get("execution")
        != {
            "offline_assemble_calls": 1,
            "workload_processes": 0,
            "rocprof_processes": 0,
            "gpu_commands": 0,
            "service_operations": 0,
            "operator_invocations": 0,
            "actual_invocations": 0,
            "model_loads": 0,
        }
    ):
        raise OperatorError("offline reassembly-v11 formal readback differs")
    return {
        "value": value,
        "artifact_commit": OFFLINE_ARTIFACT_COMMIT,
        "artifact_tree": OFFLINE_ARTIFACT_TREE,
        "file_count": sum(
            len(inventory["members"]) + 1
            for inventory in (capture_inventory, evidence_inventory)
        ),
        "capture_inventory": capture_inventory,
        "evidence_inventory": evidence_inventory,
    }


def current_fresh_paths() -> list[Path]:
    paths = [
        PROFILE_RUNTIME,
        PROFILE_EXECUTE_EVIDENCE,
        MAINTENANCE_EVIDENCE,
        PROFILE_CAPTURE,
        PROFILE_CAPTURE / "capture-artifact.json",
        PROFILE_CAPTURE / "rocprof.stdout",
        PROFILE_CAPTURE / "rocprof.stderr",
        OPERATOR_RESULT,
        ACTUAL_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("fresh output set differs")
    return paths


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
    if paths != current_fresh_paths():
        raise OperatorError("fresh output set differs")
    return paths


def root_set() -> list[Path]:
    return [
        PREPARED_ROOT,
        BINDING_ROOT,
        EXECUTE_BINDING_ROOT,
        P2 / "resident-one-case-smoke-ready-v6",
        P2 / "resident-one-case-smoke-ready-dry-run-v6",
        HISTORICAL_READY_V15_ROOT,
        HISTORICAL_READY_DRY_RUN_V15_ROOT,
        PROFILE_READY_ROOT,
        PROFILE_READY_DRY_RUN_ROOT,
        OFFLINE_CAPTURE_ROOT,
        OFFLINE_EVIDENCE_ROOT,
    ]


def trusted_operator_source_record(path: Path = SOURCE) -> dict[str, Any]:
    metadata = path.lstat()
    relative = str(path.relative_to(ROOT))
    artifact_commit = git("log", "-1", "--format=%H", "--", relative)
    if not artifact_commit or GIT_OID_RE.fullmatch(artifact_commit) is None:
        raise OperatorError("operator source last-change commit differs")
    source_commit = artifact_commit
    source_tree = git("rev-parse", f"{source_commit}^{{tree}}")
    committed_blob = git("rev-parse", f"{source_commit}:{relative}")
    current_blob = git("hash-object", str(path))
    raw = path.read_bytes()
    committed_raw = git_bytes("show", f"{source_commit}:{relative}")
    if (
        path.is_symlink()
        or not path.is_file()
        or metadata.st_nlink != 1
        or source_commit != artifact_commit
        or committed_blob != current_blob
        or committed_raw != raw
        or sha_bytes(committed_raw) != sha_bytes(raw)
    ):
        raise OperatorError("operator source last-change authority differs")
    return {
        "path": str(path),
        "sha256": sha_bytes(raw),
        "source_commit": source_commit,
        "artifact_commit": artifact_commit,
        "source_tree": source_tree,
        "git_blob": current_blob,
        "identity": [
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        ],
    }


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
    records.append(trusted_operator_source_record())
    return sorted(records, key=lambda item: item["path"])


def relevant_snapshot(ready: dict[str, Any]) -> dict[str, Any]:
    roots = [verify_sums(root) for root in root_set()]
    execute_inventory = next(item for item in roots if item["root"] == str(EXECUTE_BINDING_ROOT))
    execute_relative = str(EXECUTE_BINDING_ROOT.relative_to(ROOT))
    if (
        git("rev-parse", f"{EXECUTE_BINDING_ARTIFACT_COMMIT}^{{tree}}")
        != EXECUTE_BINDING_ARTIFACT_TREE
        or git(
            "rev-parse",
            f"{EXECUTE_BINDING_ARTIFACT_COMMIT}:{execute_relative}",
        )
        != EXECUTE_BINDING_ROOT_TREE
        or execute_inventory["sha256sums_sha256"]
        != EXECUTE_BINDING_SHA256SUMS_SHA256
        or sha_file(EXECUTE_BINDING_ROOT / "execute-binding.json")
        != EXECUTE_BINDING_MANIFEST_SHA256
        or sha_file(EXECUTE_BINDING_ROOT / "launcher-trust.json")
        != EXECUTE_LAUNCHER_TRUST_SHA256
    ):
        raise OperatorError("execute-binding-v10 authority differs")
    verify_inventory_commit(EXECUTE_BINDING_ROOT, execute_inventory, EXECUTE_BINDING_ARTIFACT_COMMIT)
    execute_binding = load(
        EXECUTE_BINDING_ROOT / "execute-binding.json",
        "execute-binding-v10 manifest",
    )
    launcher_trust = load(
        EXECUTE_BINDING_ROOT / "launcher-trust.json",
        "execute-binding-v10 launcher trust",
    )
    if (
        execute_binding.get("schema_version")
        != "ullm.aq4_p2_resident_smoke_execute_binding.v1"
        or execute_binding.get("run_id")
        != "p2-r9700-resident-one-case-smoke-execute-v10"
        or execute_binding.get("actual_eligible") is not False
        or execute_binding.get("runner_output") != str(EXECUTE_RUNTIME)
        or execute_binding.get("evidence_output")
        != str(EXECUTE_EVIDENCE)
        or launcher_trust.get("commit") != EXECUTE_LAUNCHER_COMMIT
        or launcher_trust.get("tree") != EXECUTE_LAUNCHER_TREE
        or launcher_trust.get("git_blob") != EXECUTE_LAUNCHER_BLOB
        or launcher_trust.get("sha256") != EXECUTE_LAUNCHER_SHA256
        or git("rev-parse", f"{EXECUTE_LAUNCHER_COMMIT}^{{tree}}")
        != EXECUTE_LAUNCHER_TREE
        or git(
            "rev-parse",
            f"{EXECUTE_LAUNCHER_COMMIT}:tools/launch-aq4-p2-resident-smoke.py",
        )
        != EXECUTE_LAUNCHER_BLOB
        or launcher_trust.get("actual_eligible") is not False
    ):
        raise OperatorError("execute-binding-v10 semantic authority differs")
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
    previous = previous_authorization_v13_state()
    if previous.get("state") != "authorized_not_invoked_preflight_blocked":
        raise OperatorError("previous operator-v13 final state differs")
    previous_actual = previous_actual_v12_state()
    if previous_actual.get("state") != "executed_sealed":
        raise OperatorError("previous actual-v12 final state differs")
    historical_ready = historical_ready_v15_authority()
    offline = offline_reassembly_authority()
    if offline.get("file_count") != 42:
        raise OperatorError("offline reassembly-v11 file coverage differs")
    records.sort(key=lambda item: item["path"])
    return {"root_count": len(roots), "file_count": len(records), "trusted_source_count": len(sources), "byte_aggregate_sha256": sha_bytes(canonical([{"path": item["path"], "sha256": item["sha256"]} for item in records])), "identity_aggregate_sha256": sha_bytes(canonical(records)), "fresh_absence": absent, "all_required_absent": all(absent.values()), "previous_operator_v13": previous, "previous_actual_v12": previous_actual, "historical_ready_v15": {"status": historical_ready["status"], "actual_eligible": historical_ready["actual_eligible"], "artifact_commit": HISTORICAL_READY_V15_COMMIT}, "offline_reassembly_v11": offline}


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
    ready, ready_inventory = ready_authority(); quiet = validate_quiet(QUIET_ROOT); previous = previous_authorization_v13_state()
    if previous.get("state") != "authorized_not_invoked_preflight_blocked" or previous.get("reason") != "external_owner_after_seal_before_invocation":
        raise OperatorError("previous operator-v13 final state differs")
    previous_actual = previous_actual_v12_state()
    if previous_actual.get("state") != "executed_sealed":
        raise OperatorError("previous actual-v12 final state differs")
    historical_ready = historical_ready_v15_authority()
    offline = offline_reassembly_authority()
    fresh = fresh_paths(ready)
    if any(path.exists() or path.is_symlink() for path in fresh):
        raise OperatorError("operator fresh outputs are not absent")
    argv = actual_argv()
    manifest: dict[str, Any] = {"schema_version": OPERATOR_SCHEMA, "status": "audited_ready_for_single_explicit_profile_diagnostic", "argv": argv, "command_sha256": sha_bytes(canonical(argv)), "authorization": {"maximum_invocations": 1, "explicit_confirmation_flag_count": argv.count("--confirm-one-case"), "profile_diagnostic_flag_count": argv.count("--profile-diagnostic"), "ready_artifact_flag_count": argv.count("--ready-artifact"), "evidence_output_flag_count": argv.count("--evidence-output"), "quiet_window_status_required": "go", "quiet_window_decision_required": "GO"}, "execution": {"argument_count": len(argv), "shell": False, "working_directory": str(ROOT), "same_pty_sudo_cache_required": True, "external_service_stop_required": True, "maximum_invocations": 1, "output_no_reuse": True, "operator_must_use_manifest_argv_exactly": True, "requires_fresh_output_recheck_immediately_before_execution": True, "promotion_eligible": False, "measurement_eligible": False}, "inputs": {"profile_ready": {"artifact_commit": READY_ARTIFACT_COMMIT, "ready_binding_sha256": READY_BINDING_SHA256, "inventory": ready_inventory}, "historical_ready_v15": {"artifact_commit": HISTORICAL_READY_V15_COMMIT, "status": historical_ready["status"], "actual_eligible": historical_ready["actual_eligible"]}, "offline_reassembly_v11": offline, "quiet_window": {"path": str(QUIET_ROOT / "quiet-window.json"), "sha256": sha_file(QUIET_ROOT / "quiet-window.json"), "decision": quiet["value"]["decision"], "status": quiet["value"]["status"]}, "previous_operator_v13": previous, "previous_actual_v12": previous_actual}, "fresh_outputs": [{"path": str(path), "absent": True} for path in fresh], "quiet_final_streak": quiet["value"]["summary"], "failure_contract": {"retry_forbidden": True, "preserve_operator_stdout_stderr": True, "preserve_maintenance_launcher_capture_and_ready_audits": True, "immutable_failure_capture_before_reporting": True, "outer_restore_in_finally": True, "restore_timeout_seconds": ready.get("maintenance", {}).get("restore_poll", {}).get("timeout_seconds"), "restore_requires_active_running_new_epoch_nrestarts_zero_worker_lock_gpu_kfd_formal_health_and_hashes": True, "children_remaining_must_be_empty": True}, "target_runner_manifest": {"schema_version": "ullm.aq4_p3_profile_target_command.v1", "fresh_per_execution": True, "generated_by": "launcher_after_live_preflight", "maximum_invocations": 1, "static_manifest_present": False}, "pre_execution_audit": {"quiet_window": "passed", "fresh_outputs": "9/9 absent", "historical_ready_v15": "validated_historical", "offline_reassembly_v11": "offline_reassembled_sealed", "previous_operator_v13": "authorized_not_invoked_preflight_blocked", "previous_operator_v13_reason": "external_owner_after_seal_before_invocation", "previous_actual_v12": "executed_sealed", "actual_executed": False}, "actual_executed": False, "gpu_command_executed": False, "service_touched": False, "secret_material_embedded": False, "manifest_sha256": None}
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
    if value.get("authorization", {}).get("maximum_invocations") != 1 or execution.get("maximum_invocations") != 1 or execution.get("shell") is not False or execution.get("outer_restore_in_finally") is True or value.get("actual_executed") is not False or value.get("gpu_command_executed") is not False or value.get("service_touched") is not False or value.get("secret_material_embedded") is not False or value.get("fresh_outputs") != [{"path": str(path), "absent": True} for path in current_fresh_paths()]:
        raise OperatorError("operator authorization/safety differs")
    if failure.get("retry_forbidden") is not True or failure.get("outer_restore_in_finally") is not True or failure.get("restore_timeout_seconds") != 120.0 or failure.get("children_remaining_must_be_empty") is not True:
        raise OperatorError("operator failure/restore contract differs")
    inputs = value.get("inputs", {})
    previous = inputs.get("previous_operator_v13", {})
    previous_actual = inputs.get("previous_actual_v12", {})
    historical_ready = inputs.get("historical_ready_v15", {})
    offline = inputs.get("offline_reassembly_v11", {})
    pre_audit = value.get("pre_execution_audit", {})
    if (
        previous.get("state") != "authorized_not_invoked_preflight_blocked"
        or previous.get("reason")
        != "external_owner_after_seal_before_invocation"
        or previous.get("authorization_commit") != PREVIOUS_OPERATOR_V13_COMMIT
        or previous.get("authorization_tree") != PREVIOUS_OPERATOR_V13_TREE
        or previous.get("authorization_root_tree")
        != PREVIOUS_OPERATOR_V13_ROOT_TREE
        or previous.get("manifest_file_sha256")
        != PREVIOUS_OPERATOR_V13_MANIFEST_SHA256
        or previous.get("manifest_semantic_sha256")
        != PREVIOUS_OPERATOR_V13_SEMANTIC_SHA256
        or previous.get("command_sha256")
        != PREVIOUS_OPERATOR_V13_COMMAND_SHA256
        or previous.get("invocation_count") != 0
        or previous.get("maximum_invocations") != 1
        or previous.get("result_present") is not False
        or previous.get("audit_present") is not False
        or previous.get("actual_executed") is not False
        or previous.get("gpu_command_executed") is not False
        or previous.get("service_touched") is not False
        or len(previous.get("fresh_outputs", [])) != 9
        or not all(
            item.get("present") is False
            for item in previous.get("fresh_outputs", [])
        )
        or previous.get("quiet_v18", {}).get("artifact_commit")
        != PREVIOUS_QUIET_V18_COMMIT
        or previous.get("quiet_v18", {}).get("root_tree")
        != PREVIOUS_QUIET_V18_ROOT_TREE
        or previous.get("quiet_v18", {}).get("json_sha256")
        != PREVIOUS_QUIET_V18_JSON_SHA256
        or previous_actual.get("state") != "executed_sealed"
        or previous_actual.get("artifact_commit") != PREVIOUS_ACTUAL_V12_COMMIT
        or previous_actual.get("artifact_tree") != PREVIOUS_ACTUAL_V12_TREE
        or previous_actual.get("file_count") != PREVIOUS_ACTUAL_V12_FILE_COUNT
        or previous_actual.get("invocation_count") != 1
        or previous_actual.get("maximum_invocations") != 1
        or previous_actual.get("retry_performed") is not False
        or previous_actual.get("previous_operator_v12", {}).get("authorization_commit")
        != PREVIOUS_OPERATOR_V12_COMMIT
        or historical_ready
        != {
            "artifact_commit": HISTORICAL_READY_V15_COMMIT,
            "status": "ready_for_one_case",
            "actual_eligible": True,
        }
        or offline.get("artifact_commit") != OFFLINE_ARTIFACT_COMMIT
        or offline.get("artifact_tree") != OFFLINE_ARTIFACT_TREE
        or offline.get("file_count") != 42
        or offline.get("value", {}).get("status")
        != "offline_reassembled_sealed"
        or pre_audit.get("historical_ready_v15") != "validated_historical"
        or pre_audit.get("offline_reassembly_v11")
        != "offline_reassembled_sealed"
        or pre_audit.get("previous_operator_v13")
        != "authorized_not_invoked_preflight_blocked"
        or pre_audit.get("previous_operator_v13_reason")
        != "external_owner_after_seal_before_invocation"
        or pre_audit.get("previous_actual_v12") != "executed_sealed"
        or pre_audit.get("fresh_outputs") != "9/9 absent"
        or pre_audit.get("actual_executed") is not False
    ):
        raise OperatorError("operator previous/final-state binding differs")
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


def finalizer_source_authority() -> dict[str, Any]:
    relative = str(SOURCE.relative_to(ROOT))
    commit = git("log", "-1", "--format=%H", "--", relative)
    blob = git("rev-parse", f"{commit}:{relative}")
    if git("hash-object", str(SOURCE)) != blob:
        raise OperatorError("finalizer source is not committed authority")
    return {
        "role": "existing_evidence_recovery_only_not_execution_authority",
        "path": str(SOURCE),
        "commit": commit,
        "git_blob": blob,
        "sha256": sha_file(SOURCE),
    }


def validate_finalizer_source_authority(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "path",
        "commit",
        "git_blob",
        "sha256",
    }:
        raise OperatorError("finalizer source authority shape differs")
    path = Path(str(value["path"]))
    try:
        relative = str(path.relative_to(ROOT))
    except ValueError as error:
        raise OperatorError("finalizer source authority path differs") from error
    if (
        value["role"] != "existing_evidence_recovery_only_not_execution_authority"
        or path != SOURCE
        or GIT_OID_RE.fullmatch(str(value["commit"])) is None
        or GIT_OID_RE.fullmatch(str(value["git_blob"])) is None
        or SHA_RE.fullmatch(str(value["sha256"])) is None
        or git("rev-parse", f"{value['commit']}:{relative}") != value["git_blob"]
    ):
        raise OperatorError("finalizer source Git authority differs")
    completed = subprocess.run(
        ["git", "cat-file", "blob", value["git_blob"]],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr or sha_bytes(completed.stdout) != value["sha256"]:
        raise OperatorError("finalizer source blob authority differs")


def pre_stop_noop_failure_record(
    maintenance: dict[str, Any],
    inventory: dict[str, Any],
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    if evidence_root is None:
        evidence_root = MAINTENANCE_EVIDENCE
    failure = maintenance.get("failure")
    restore = maintenance.get("restore")
    counts = maintenance.get("process_counts")
    safety = maintenance.get("safety")
    expected_counts = {
        "sudo": 1,
        "sudo_keepalive": 0,
        "systemctl_stop": 0,
        "launcher": 0,
        "systemctl_start": 0,
        "capture_tool": 0,
        "rocprof": 0,
        "docker": 0,
        "docker_exec": 0,
        "container_curl": 0,
        "container_curl_total": 0,
        "container_curl_version": 0,
        "container_curl_endpoint": 0,
        "stopped_gate_polls": 0,
        "stopped_gate_probe_commands": 0,
    }
    if (
        failure
        != {
            "stage": "pre-stop-snapshot",
            "reason": "restored worker does not uniquely own target GPU",
            "launcher_started": False,
        }
        or restore
        != {"attempted": False, "error": None, "passed": True, "post_start": None}
        or counts != expected_counts
        or safety
        != {
            "service_touched": False,
            "service_stopped": False,
            "gpu_command_executed": False,
            "model_load_executed": False,
        }
        or maintenance.get("pre_stop") is not None
        or maintenance.get("stopped_gates") is not None
        or maintenance.get("stopped_gate_poll") is not None
        or maintenance.get("lock_substrate") is not None
        or maintenance.get("lock_substrate_cleanup") is not None
        or maintenance.get("launcher") is not None
        or maintenance.get("capture") is not None
        or maintenance.get("sequence") != ["sudo-prevalidate"]
        or maintenance.get("secret_material_recorded") is not False
    ):
        raise OperatorError("pre-stop no-op restore evidence differs")
    member = inventory.get("members", {}).get("launcher-evidence.json", {})
    if member.get("sha256") != sha_file(evidence_root / "launcher-evidence.json"):
        raise OperatorError("pre-stop failure evidence hash differs")
    return {
        "source": "sealed_maintenance_failure",
        "maintenance_evidence_path": member["path"],
        "maintenance_evidence_sha256": member["sha256"],
        "stage": failure["stage"],
        "reason": failure["reason"],
        "launcher_started": False,
        "owner_identity_evidence": "unavailable_not_recorded_by_pre_stop_probe",
        "normative_external_owner_pids": None,
        "post_hoc_owner_diagnostics_normative": False,
    }


def capture_recovery_snapshot(ready: dict[str, Any]) -> dict[str, Any]:
    maintenance = load_maintenance()
    running = maintenance.capture_running(maintenance.default_dependencies())
    formal = running["health"]["formal"]
    lock_metadata = Path(running["lock"]["path"]).lstat()
    return {
        "phase": "post_actual_evidence_recovery",
        "source": "fresh_read_only_phase_aware_probe",
        "previous_authorization_source": "sealed_operator_manifest_no_live_absence_recheck",
        "actual_outputs_permitted": True,
        "service": running["service"],
        "worker": running["worker"],
        "gpu": running["gpu"],
        "owners": {
            "amd_smi": running["owners"]["amd_smi"],
            "kfd": running["owners"]["kfd"],
        },
        "lock": {
            "path": running["lock"]["path"],
            "busy": running["lock"]["busy"],
            "identity": [
                lock_metadata.st_dev,
                lock_metadata.st_ino,
                lock_metadata.st_mode,
                lock_metadata.st_nlink,
                lock_metadata.st_size,
            ],
        },
        "hashes": running["hashes"],
        "formal_health_sha256": sha_bytes(
            canonical(
                {
                    key: formal[key]
                    for key in (
                        "container",
                        "curl",
                        "docker",
                        "endpoints",
                        "process_counts",
                        "secret_material_recorded",
                    )
                }
            )
        ),
        "targeted_processes": targeted_processes(),
        "read_only": True,
        "service_touched": False,
        "gpu_workload_executed": False,
    }


def validate_actual_documents(
    result: dict[str, Any],
    audit: dict[str, Any],
    *,
    result_schema: str = OPERATOR_RESULT_SCHEMA,
    audit_schema: str = ACTUAL_AUDIT_SCHEMA,
) -> None:
    clone = json.loads(json.dumps(audit)); declared = clone.get("audit_sha256"); clone["audit_sha256"] = None
    returncode = result.get("returncode")
    succeeded = type(returncode) is int and returncode == 0
    expected_result = "passed" if succeeded else "failed"
    expected_audit = "passed_immutable_evidence_preserved_restore_passed" if succeeded else "failed_immutable_evidence_preserved_restore_passed"
    if result.get("schema_version") != result_schema or result.get("status") != expected_result or type(returncode) is not int or result.get("invocation_count") != 1 or result.get("maximum_invocations") != 1 or result.get("shell") is not False or result.get("retry_performed") is not False or result.get("actual_executed") is not True or result.get("secret_material_recorded") is not False:
        raise OperatorError("operator result semantic boundary differs")
    execution = audit.get("execution", {})
    profile = audit.get("profile_artifacts", {})
    if audit.get("schema_version") != audit_schema or declared != sha_bytes(canonical(clone)) or audit.get("status") != expected_audit or execution.get("returncode") != returncode or execution.get("invocation_count") != 1 or execution.get("maximum_invocations") != 1 or execution.get("shell") is not False or execution.get("retry_performed") is not False or audit.get("restore", {}).get("passed") is not True or audit.get("package_integrity", {}).get("full_hash_count") != 1 or audit.get("cleanup", {}).get("residual_targeted_processes") != [] or audit.get("actual_executed") is not True or audit.get("retry_performed") is not False or audit.get("secret_material_recorded") is not False:
        raise OperatorError("actual audit semantic boundary differs")
    if succeeded and (audit.get("failure") is not None or profile.get("status") != "complete_diagnostic" or profile.get("measurement_eligible") is not False or profile.get("promotion_eligible") is not False):
        raise OperatorError("successful actual audit outcome differs")
    if not succeeded and (not isinstance(audit.get("failure"), dict) or profile.get("status") != "failure_evidence_only"):
        raise OperatorError("failed actual audit outcome differs")
    if result_schema == OPERATOR_RESULT_SCHEMA and audit_schema == ACTUAL_AUDIT_SCHEMA:
        authority = result.get("finalizer_authority")
        restore = audit.get("restore", {})
        classification = audit.get("restore_classification")
        recovery = audit.get("recovery_snapshot", {})
        cleanup = audit.get("cleanup", {})
        if (
            not isinstance(authority, dict)
            or authority != audit.get("finalizer_authority")
            or authority.get("role")
            != "existing_evidence_recovery_only_not_execution_authority"
            or authority.get("path") != str(SOURCE)
            or GIT_OID_RE.fullmatch(str(authority.get("commit", ""))) is None
            or GIT_OID_RE.fullmatch(str(authority.get("git_blob", ""))) is None
            or SHA_RE.fullmatch(str(authority.get("sha256", ""))) is None
            or recovery.get("source") != "fresh_read_only_phase_aware_probe"
            or recovery.get("previous_authorization_source")
            != "sealed_operator_manifest_no_live_absence_recheck"
            or recovery.get("actual_outputs_permitted") is not True
            or recovery.get("targeted_processes") != []
            or recovery.get("read_only") is not True
            or recovery.get("service_touched") is not False
            or recovery.get("gpu_workload_executed") is not False
        ):
            raise OperatorError("actual finalizer/recovery authority differs")
        if restore.get("attempted") is False:
            failure_snapshot = audit.get("pre_stop_failure_snapshot", {})
            if (
                classification != "pre_stop_untouched_same_epoch"
                or restore
                != {"attempted": False, "error": None, "passed": True, "post_start": None}
                or cleanup.get("trusted_lock_substrate_cleanup_required") is not False
                or failure_snapshot.get("stage") != "pre-stop-snapshot"
                or failure_snapshot.get("reason")
                != "restored worker does not uniquely own target GPU"
                or failure_snapshot.get("owner_identity_evidence")
                != "unavailable_not_recorded_by_pre_stop_probe"
                or failure_snapshot.get("normative_external_owner_pids") is not None
                or failure_snapshot.get("post_hoc_owner_diagnostics_normative") is not False
            ):
                raise OperatorError("actual pre-stop no-op restore binding differs")
        elif (
            restore.get("attempted") is not True
            or classification != "outer_finally_restored_new_epoch"
            or audit.get("pre_stop_failure_snapshot") is not None
            or cleanup.get("trusted_lock_substrate_cleanup_required") is not True
        ):
            raise OperatorError("actual touched restore binding differs")


def actual_v11_state() -> dict[str, Any]:
    sealed_roots = {
        "maintenance": ACTUAL_V11_MAINTENANCE_EVIDENCE,
        "operator_result": ACTUAL_V11_OPERATOR_RESULT,
        "actual_audit": ACTUAL_V11_AUDIT,
    }
    forbidden_roots = [
        ACTUAL_V11_PROFILE_RUNTIME,
        ACTUAL_V11_PROFILE_EXECUTE_EVIDENCE,
        ACTUAL_V11_PROFILE_CAPTURE,
    ]
    sealed_present = [root.exists() or root.is_symlink() for root in sealed_roots.values()]
    forbidden_present = [root.exists() or root.is_symlink() for root in forbidden_roots]
    if not any(sealed_present) and not any(forbidden_present):
        return {
            "state": "not_executed",
            "artifact_commit": ACTUAL_V11_COMMIT,
            "artifact_tree": ACTUAL_V11_TREE,
            "actual_executed": False,
        }
    if not all(sealed_present) or any(forbidden_present):
        raise OperatorError("actual-v11 state is partial or mixed")

    inventories = {name: verify_sums(root) for name, root in sealed_roots.items()}
    if git("rev-parse", f"{ACTUAL_V11_COMMIT}^{{tree}}") != ACTUAL_V11_TREE:
        raise OperatorError("actual-v11 Git tree differs")
    expected: set[str] = set()
    for inventory in inventories.values():
        root = Path(inventory["root"])
        verify_inventory_commit(root, inventory, ACTUAL_V11_COMMIT)
        expected.add(str((root / "SHA256SUMS").relative_to(ROOT)))
        expected.update(
            str(Path(member["path"]).relative_to(ROOT))
            for member in inventory["members"].values()
        )
    relative_roots = [str(root.relative_to(ROOT)) for root in sealed_roots.values()]
    observed = set(filter(None, git("ls-tree", "-r", "--name-only", ACTUAL_V11_COMMIT, "--", *relative_roots).splitlines()))
    if observed != expected or len(expected) != ACTUAL_V11_FILE_COUNT:
        raise OperatorError("actual-v11 Git file coverage differs")

    maintenance = load(
        ACTUAL_V11_MAINTENANCE_EVIDENCE / "launcher-evidence.json",
        "actual-v11 maintenance evidence",
    )
    result = load(
        ACTUAL_V11_OPERATOR_RESULT / "operator-result.json",
        "actual-v11 operator result",
    )
    audit = load(ACTUAL_V11_AUDIT / "actual-audit.json", "actual-v11 audit")
    validate_actual_documents(
        result,
        audit,
        result_schema=PREVIOUS_OPERATOR_RESULT_V11_SCHEMA,
        audit_schema=PREVIOUS_ACTUAL_AUDIT_V11_SCHEMA,
    )
    previous = previous_operator_v11_state()
    if (
        result.get("returncode") != 1
        or result.get("operator_manifest_commit") != PREVIOUS_OPERATOR_V11_COMMIT
        or result.get("authority_commit") != "5456117e223653155897eaab9c176a2424198250"
        or result.get("manifest_file_sha256") != PREVIOUS_OPERATOR_V11_MANIFEST_SHA256
        or result.get("manifest_semantic_sha256") != PREVIOUS_OPERATOR_V11_SEMANTIC_SHA256
        or audit.get("authority_commit") != PREVIOUS_OPERATOR_V11_COMMIT
        or audit.get("manifest_file_sha256") != PREVIOUS_OPERATOR_V11_MANIFEST_SHA256
    ):
        raise OperatorError("actual-v11 command/result authority differs")
    stdout_path = ACTUAL_V11_OPERATOR_RESULT / "operator.stdout.bin"
    stderr_path = ACTUAL_V11_OPERATOR_RESULT / "operator.stderr.bin"
    _stream_record_matches(result.get("stdout"), stdout_path, "actual-v11 operator stdout")
    _stream_record_matches(result.get("stderr"), stderr_path, "actual-v11 operator stderr")
    streams = audit.get("all_returncodes_and_streams", {})
    if (
        streams.get("operator")
        != {"returncode": 1, "stdout": stream_record(stdout_path), "stderr": stream_record(stderr_path)}
        or any(streams.get(name) != {"returncode": None, "stdout": None, "stderr": None} for name in ("rocprof", "runner", "validator"))
    ):
        raise OperatorError("actual-v11 exact subprocess boundary differs")

    failure_snapshot = pre_stop_noop_failure_record(
        maintenance,
        inventories["maintenance"],
        ACTUAL_V11_MAINTENANCE_EVIDENCE,
    )
    package = maintenance.get("package_integrity", {})
    if (
        package.get("full_hash_count") != 1
        or package.get("full_content", {}).get("passed") is not True
        or package.get("tree_identity", {}).get("stable_across_full_hash") is not True
        or package != audit.get("package_integrity")
        or audit.get("pre_stop_failure_snapshot") != failure_snapshot
        or audit.get("restore")
        != {"attempted": False, "error": None, "passed": True, "post_start": None}
        or audit.get("restore_classification") != "pre_stop_untouched_same_epoch"
    ):
        raise OperatorError("actual-v11 pre-stop sealed evidence differs")

    authority = result.get("finalizer_authority")
    validate_finalizer_source_authority(authority)
    recovery = audit.get("recovery_snapshot", {})
    post_health = audit.get("post_health", {})
    if (
        authority != audit.get("finalizer_authority")
        or authority.get("commit") != "370ab8cff2fc745d85657260329a80fab21b0acb"
        or recovery.get("source") != "fresh_read_only_phase_aware_probe"
        or recovery.get("previous_authorization_source") != "sealed_operator_manifest_no_live_absence_recheck"
        or recovery.get("actual_outputs_permitted") is not True
        or recovery.get("phase") != "post_actual_evidence_recovery"
        or recovery.get("read_only") is not True
        or recovery.get("service_touched") is not False
        or recovery.get("gpu_workload_executed") is not False
        or recovery.get("targeted_processes") != []
        or recovery.get("service", {}).get("active_state") != "active"
        or recovery.get("service", {}).get("sub_state") != "running"
        or recovery.get("service", {}).get("nrestarts") != 0
        or recovery.get("owners", {}).get("amd_smi") != [recovery.get("worker", {}).get("pid")]
        or recovery.get("owners", {}).get("kfd") != [recovery.get("worker", {}).get("pid")]
        or recovery.get("lock", {}).get("busy") is not True
        or any(recovery.get(key) != post_health.get(key) for key in ("service", "worker", "gpu", "owners", "lock", "hashes", "formal_health_sha256", "targeted_processes"))
    ):
        raise OperatorError("actual-v11 finalizer recovery authority differs")
    cleanup = audit.get("cleanup", {})
    profile = audit.get("profile_artifacts", {})
    if (
        cleanup.get("retry_forbidden_and_not_performed") is not True
        or cleanup.get("residual_targeted_processes") != []
        or cleanup.get("launcher_children_remaining") != []
        or cleanup.get("capture_children_remaining") != []
        or profile
        != {
            "status": "failure_evidence_only",
            "runtime_summary": None,
            "capture_artifact": None,
            "capture_failure": None,
            "trace_csv_count": 0,
            "trace_csv_bytes": 0,
            "measurement_eligible": False,
            "promotion_eligible": False,
        }
        or audit.get("evidence", {}).get("maintenance") != inventories["maintenance"]
        or audit.get("evidence", {}).get("operator_result") != inventories["operator_result"]
        or any(audit.get("evidence", {}).get(name) is not None for name in ("execute", "runtime", "capture"))
    ):
        raise OperatorError("actual-v11 no-touch final evidence differs")
    return {
        "state": "pre_stop_failed_sealed",
        "artifact_commit": ACTUAL_V11_COMMIT,
        "artifact_tree": ACTUAL_V11_TREE,
        "file_count": len(expected),
        "returncode": 1,
        "invocation_count": 1,
        "maximum_invocations": 1,
        "retry_performed": False,
        "previous_operator_v11": previous,
        "inventories": inventories,
        "actual_executed": True,
    }


def historical_actual_v9_fresh_paths() -> list[Path]:
    paths = [
        HISTORICAL_PROFILE_RUNTIME_V8,
        HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8,
        HISTORICAL_MAINTENANCE_EVIDENCE_V8,
        HISTORICAL_PROFILE_CAPTURE_V8,
        HISTORICAL_PROFILE_CAPTURE_V8 / "capture-artifact.json",
        HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stdout",
        HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stderr",
        HISTORICAL_OPERATOR_RESULT_V9,
        HISTORICAL_ACTUAL_AUDIT_V9,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("historical actual-v9 path set differs")
    return paths


def _stream_record_matches(record: Any, path: Path, label: str) -> None:
    if not isinstance(record, dict) or record != stream_record(path):
        raise OperatorError(f"historical actual-v9 {label} stream differs")


def _historical_actual_v9_commit_authority(
    inventories: dict[str, dict[str, Any]],
) -> None:
    if (
        git("rev-parse", f"{HISTORICAL_ACTUAL_V9_COMMIT}^{{tree}}")
        != HISTORICAL_ACTUAL_V9_TREE
    ):
        raise OperatorError("historical actual-v9 Git tree differs")
    expected: set[str] = set()
    for inventory in inventories.values():
        root = Path(inventory["root"])
        verify_inventory_commit(root, inventory, HISTORICAL_ACTUAL_V9_COMMIT)
        expected.add(str((root / "SHA256SUMS").relative_to(ROOT)))
        expected.update(
            str(Path(member["path"]).relative_to(ROOT))
            for member in inventory["members"].values()
        )
    roots = [str(Path(inventory["root"]).relative_to(ROOT)) for inventory in inventories.values()]
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                HISTORICAL_ACTUAL_V9_COMMIT,
                "--",
                *roots,
            ).splitlines(),
        )
    )
    if (
        expected != observed
        or len(expected) != HISTORICAL_ACTUAL_V9_FILE_COUNT
    ):
        raise OperatorError("historical actual-v9 Git file coverage differs")


def historical_actual_v9_state() -> dict[str, Any]:
    fresh = historical_actual_v9_fresh_paths()
    present = [path.exists() or path.is_symlink() for path in fresh]
    state = [
        {"path": str(path), "present": observed}
        for path, observed in zip(fresh, present, strict=True)
    ]
    if not any(present):
        return {
            "state": "not_executed",
            "artifact_commit": HISTORICAL_ACTUAL_V9_COMMIT,
            "artifact_tree": HISTORICAL_ACTUAL_V9_TREE,
            "fresh_outputs": state,
            "actual_executed": False,
        }

    required = {
        HISTORICAL_MAINTENANCE_EVIDENCE_V8,
        HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8,
        HISTORICAL_PROFILE_RUNTIME_V8,
        HISTORICAL_PROFILE_CAPTURE_V8,
        HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stdout",
        HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stderr",
        HISTORICAL_OPERATOR_RESULT_V9,
        HISTORICAL_ACTUAL_AUDIT_V9,
        HISTORICAL_PROFILE_CAPTURE_V8 / "capture-failure.json",
    }
    if (
        any(not path.exists() or path.is_symlink() for path in required)
        or (HISTORICAL_PROFILE_CAPTURE_V8 / "capture-artifact.json").exists()
        or (HISTORICAL_PROFILE_CAPTURE_V8 / "capture-artifact.json").is_symlink()
    ):
        raise OperatorError("historical actual-v9 state is partial or mixed")

    roots = {
        "maintenance": HISTORICAL_MAINTENANCE_EVIDENCE_V8,
        "execute": HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8,
        "runtime": HISTORICAL_PROFILE_RUNTIME_V8,
        "capture": HISTORICAL_PROFILE_CAPTURE_V8,
        "operator_result": HISTORICAL_OPERATOR_RESULT_V9,
        "actual_audit": HISTORICAL_ACTUAL_AUDIT_V9,
    }
    inventories = {name: verify_sums(root) for name, root in roots.items()}
    result = load(
        HISTORICAL_OPERATOR_RESULT_V9 / "operator-result.json",
        "historical operator-v9 result",
    )
    audit = load(
        HISTORICAL_ACTUAL_AUDIT_V9 / "actual-audit.json",
        "historical actual-v9 audit",
    )
    validate_actual_documents(
        result,
        audit,
        result_schema=HISTORICAL_OPERATOR_RESULT_V9_SCHEMA,
        audit_schema=HISTORICAL_ACTUAL_AUDIT_V9_SCHEMA,
    )
    if (
        result.get("returncode") != 1
        or result.get("operator_manifest_commit")
        != HISTORICAL_OPERATOR_MANIFEST_V9_COMMIT
        or audit.get("authority_commit")
        != HISTORICAL_OPERATOR_MANIFEST_V9_COMMIT
    ):
        raise OperatorError("historical actual-v9 result authority differs")

    operator_stdout = HISTORICAL_OPERATOR_RESULT_V9 / "operator.stdout.bin"
    operator_stderr = HISTORICAL_OPERATOR_RESULT_V9 / "operator.stderr.bin"
    rocprof_stdout = HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stdout"
    rocprof_stderr = HISTORICAL_PROFILE_CAPTURE_V8 / "rocprof.stderr"
    _stream_record_matches(result.get("stdout"), operator_stdout, "operator stdout")
    _stream_record_matches(result.get("stderr"), operator_stderr, "operator stderr")
    streams = audit.get("all_returncodes_and_streams", {})
    operator = streams.get("operator", {})
    rocprof = streams.get("rocprof", {})
    if operator.get("returncode") != 1 or rocprof.get("returncode") != 1:
        raise OperatorError("historical actual-v9 subprocess returncodes differ")
    _stream_record_matches(operator.get("stdout"), operator_stdout, "audit operator stdout")
    _stream_record_matches(operator.get("stderr"), operator_stderr, "audit operator stderr")
    _stream_record_matches(rocprof.get("stdout"), rocprof_stdout, "audit rocprof stdout")
    _stream_record_matches(rocprof.get("stderr"), rocprof_stderr, "audit rocprof stderr")
    launcher = load(
        HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8 / "launcher-evidence.json",
        "historical actual-v9 launcher evidence",
    )
    for name in ("runner", "validator"):
        process = launcher.get(name, {})
        audited_process = streams.get(name, {})
        if audited_process.get("returncode") != process.get("exit_code"):
            raise OperatorError("historical actual-v9 subprocess returncodes differ")
        for direction in ("stdout", "stderr"):
            expected_stream = optional_stream(
                HISTORICAL_PROFILE_EXECUTE_EVIDENCE_V8,
                process.get(direction),
            )
            if audited_process.get(direction) != expected_stream:
                raise OperatorError(
                    f"historical actual-v9 {name} {direction} stream differs"
                )

    maintenance = load(
        HISTORICAL_MAINTENANCE_EVIDENCE_V8 / "launcher-evidence.json",
        "historical actual-v9 maintenance evidence",
    )
    counts = maintenance.get("process_counts", {})
    capture = maintenance.get("capture", {})
    if (
        maintenance.get("status") != "failed"
        or maintenance.get("mode") != "execute"
        or maintenance.get("secret_material_recorded") is not False
        or any(counts.get(name) != 1 for name in ("capture_tool", "launcher", "rocprof"))
        or capture.get("capture_tool_invocations") != 1
        or capture.get("rocprof_invocations") != 1
    ):
        raise OperatorError("historical actual-v9 exact-one maintenance differs")

    failure_path = HISTORICAL_PROFILE_CAPTURE_V8 / "capture-failure.json"
    failure = load(failure_path, "historical actual-v9 capture failure")
    if (
        failure.get("schema_version")
        != "ullm.aq4_p3_diagnostic_rocprof_failure.v2"
        or failure.get("status") != "failed"
        or failure.get("children_remaining") != []
        or failure.get("process_group_cleanup_complete") is not True
        or audit.get("failure", {}).get("capture_failure_sha256")
        != sha_file(failure_path)
    ):
        raise OperatorError("historical actual-v9 capture failure differs")
    failure_streams = failure.get("streams", {})
    for name, path in (("rocprof.stdout", rocprof_stdout), ("rocprof.stderr", rocprof_stderr)):
        expected = stream_record(path)
        if failure_streams.get(name) != {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            raise OperatorError("historical actual-v9 capture stream differs")

    embedded = audit.get("evidence", {})
    for name in ("maintenance", "execute", "runtime", "capture", "operator_result"):
        if embedded.get(name) != inventories[name]:
            raise OperatorError("historical actual-v9 embedded inventory differs")
    _historical_actual_v9_commit_authority(inventories)
    return {
        "state": "executed_sealed",
        "artifact_commit": HISTORICAL_ACTUAL_V9_COMMIT,
        "artifact_tree": HISTORICAL_ACTUAL_V9_TREE,
        "file_count": sum(
            len(inventory["members"]) + 1 for inventory in inventories.values()
        ),
        "fresh_outputs": state,
        "outcome": result["status"],
        "returncode": result["returncode"],
        "invocation_count": result["invocation_count"],
        "maximum_invocations": result["maximum_invocations"],
        "retry_performed": result["retry_performed"],
        "inventories": inventories,
        "actual_executed": True,
    }


def previous_actual_v12_fresh_paths() -> list[Path]:
    paths = [
        PREVIOUS_ACTUAL_V12_PROFILE_RUNTIME,
        PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "capture-artifact.json",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stdout",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stderr",
        PREVIOUS_ACTUAL_V12_OPERATOR_RESULT,
        PREVIOUS_ACTUAL_V12_AUDIT,
    ]
    if len({str(path) for path in paths}) != 9 or any(
        not path.is_absolute() or ".." in path.parts for path in paths
    ):
        raise OperatorError("previous actual-v12 path set differs")
    return paths


def _previous_actual_v12_commit_authority(
    inventories: dict[str, dict[str, Any]],
) -> None:
    if (
        git("rev-parse", f"{PREVIOUS_ACTUAL_V12_COMMIT}^{{tree}}")
        != PREVIOUS_ACTUAL_V12_TREE
    ):
        raise OperatorError("previous actual-v12 Git tree differs")
    expected: set[str] = set()
    for inventory in inventories.values():
        root = Path(inventory["root"])
        verify_inventory_commit(root, inventory, PREVIOUS_ACTUAL_V12_COMMIT)
        expected.add(str((root / "SHA256SUMS").relative_to(ROOT)))
        expected.update(
            str(Path(member["path"]).relative_to(ROOT))
            for member in inventory["members"].values()
        )
    roots = [
        str(Path(inventory["root"]).relative_to(ROOT))
        for inventory in inventories.values()
    ]
    observed = set(
        filter(
            None,
            git(
                "ls-tree",
                "-r",
                "--name-only",
                PREVIOUS_ACTUAL_V12_COMMIT,
                "--",
                *roots,
            ).splitlines(),
        )
    )
    if expected != observed or len(expected) != PREVIOUS_ACTUAL_V12_FILE_COUNT:
        raise OperatorError("previous actual-v12 Git file coverage differs")


def _previous_actual_v12_stream_matches(
    record: Any,
    path: Path,
    label: str,
) -> None:
    if not isinstance(record, dict) or record != stream_record(path):
        raise OperatorError(f"previous actual-v12 {label} stream differs")


def previous_actual_v12_state() -> dict[str, Any]:
    fresh = previous_actual_v12_fresh_paths()
    present = [path.exists() or path.is_symlink() for path in fresh]
    state = [
        {"path": str(path), "present": observed}
        for path, observed in zip(fresh, present, strict=True)
    ]
    if not any(present):
        return {
            "state": "not_executed",
            "artifact_commit": PREVIOUS_ACTUAL_V12_COMMIT,
            "artifact_tree": PREVIOUS_ACTUAL_V12_TREE,
            "fresh_outputs": state,
            "actual_executed": False,
        }

    required = {
        PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE,
        PREVIOUS_ACTUAL_V12_PROFILE_RUNTIME,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE,
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stdout",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stderr",
        PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "capture-failure.json",
        PREVIOUS_ACTUAL_V12_OPERATOR_RESULT,
        PREVIOUS_ACTUAL_V12_AUDIT,
    }
    capture_artifact = PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "capture-artifact.json"
    if (
        any(not path.exists() or path.is_symlink() for path in required)
        or capture_artifact.exists()
        or capture_artifact.is_symlink()
    ):
        raise OperatorError("previous actual-v12 state is partial or mixed")

    roots = {
        "maintenance": PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE,
        "execute": PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE,
        "runtime": PREVIOUS_ACTUAL_V12_PROFILE_RUNTIME,
        "capture": PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE,
        "operator_result": PREVIOUS_ACTUAL_V12_OPERATOR_RESULT,
        "actual_audit": PREVIOUS_ACTUAL_V12_AUDIT,
    }
    inventories = {name: verify_sums(root) for name, root in roots.items()}
    result = load(
        PREVIOUS_ACTUAL_V12_OPERATOR_RESULT / "operator-result.json",
        "previous operator-v12 result",
    )
    audit = load(
        PREVIOUS_ACTUAL_V12_AUDIT / "actual-audit.json",
        "previous actual-v12 audit",
    )
    validate_actual_documents(
        result,
        audit,
        result_schema=PREVIOUS_OPERATOR_RESULT_V12_SCHEMA,
        audit_schema=PREVIOUS_ACTUAL_AUDIT_V12_SCHEMA,
    )
    previous = previous_operator_v12_state()
    if (
        result.get("returncode") != 1
        or result.get("operator_manifest_commit") != PREVIOUS_OPERATOR_V12_COMMIT
        or result.get("manifest_file_sha256")
        != previous["manifest_file_sha256"]
        or result.get("manifest_semantic_sha256")
        != previous["manifest_semantic_sha256"]
        or result.get("command_sha256")
        != sha_bytes(canonical(previous_operator_v12_argv()))
        or audit.get("authority_commit") != PREVIOUS_OPERATOR_V12_COMMIT
        or audit.get("manifest_file_sha256")
        != previous["manifest_file_sha256"]
    ):
        raise OperatorError("previous actual-v12 command/result authority differs")

    operator_stdout = PREVIOUS_ACTUAL_V12_OPERATOR_RESULT / "operator.stdout.bin"
    operator_stderr = PREVIOUS_ACTUAL_V12_OPERATOR_RESULT / "operator.stderr.bin"
    rocprof_stdout = PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stdout"
    rocprof_stderr = PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "rocprof.stderr"
    _previous_actual_v12_stream_matches(
        result.get("stdout"),
        operator_stdout,
        "operator stdout",
    )
    _previous_actual_v12_stream_matches(
        result.get("stderr"),
        operator_stderr,
        "operator stderr",
    )
    streams = audit.get("all_returncodes_and_streams", {})
    operator = streams.get("operator", {})
    rocprof = streams.get("rocprof", {})
    if operator.get("returncode") != 1 or rocprof.get("returncode") != 1:
        raise OperatorError("previous actual-v12 subprocess returncodes differ")
    _previous_actual_v12_stream_matches(
        operator.get("stdout"),
        operator_stdout,
        "audit operator stdout",
    )
    _previous_actual_v12_stream_matches(
        operator.get("stderr"),
        operator_stderr,
        "audit operator stderr",
    )
    _previous_actual_v12_stream_matches(
        rocprof.get("stdout"),
        rocprof_stdout,
        "audit rocprof stdout",
    )
    _previous_actual_v12_stream_matches(
        rocprof.get("stderr"),
        rocprof_stderr,
        "audit rocprof stderr",
    )

    launcher = load(
        PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE / "launcher-evidence.json",
        "previous actual-v12 launcher evidence",
    )
    for name in ("runner", "validator"):
        process = launcher.get(name, {})
        audited_process = streams.get(name, {})
        if audited_process.get("returncode") != process.get("exit_code"):
            raise OperatorError("previous actual-v12 subprocess returncodes differ")
        for direction in ("stdout", "stderr"):
            expected_stream = optional_stream(
                PREVIOUS_ACTUAL_V12_PROFILE_EXECUTE_EVIDENCE,
                process.get(direction),
            )
            if audited_process.get(direction) != expected_stream:
                raise OperatorError(
                    f"previous actual-v12 {name} {direction} stream differs"
                )

    maintenance = load(
        PREVIOUS_ACTUAL_V12_MAINTENANCE_EVIDENCE / "launcher-evidence.json",
        "previous actual-v12 maintenance evidence",
    )
    counts = maintenance.get("process_counts", {})
    capture = maintenance.get("capture", {})
    if (
        maintenance.get("status") != "failed"
        or maintenance.get("mode") != "execute"
        or maintenance.get("secret_material_recorded") is not False
        or any(
            counts.get(name) != 1
            for name in ("capture_tool", "launcher", "rocprof")
        )
        or capture.get("capture_tool_invocations") != 1
        or capture.get("rocprof_invocations") != 1
    ):
        raise OperatorError("previous actual-v12 exact-one maintenance differs")

    failure_path = PREVIOUS_ACTUAL_V12_PROFILE_CAPTURE / "capture-failure.json"
    failure = load(failure_path, "previous actual-v12 capture failure")
    if (
        failure.get("schema_version")
        != "ullm.aq4_p3_diagnostic_rocprof_failure.v2"
        or failure.get("status") != "failed"
        or failure.get("children_remaining") != []
        or failure.get("process_group_cleanup_complete") is not True
        or audit.get("failure", {}).get("capture_failure_sha256")
        != sha_file(failure_path)
    ):
        raise OperatorError("previous actual-v12 capture failure differs")
    failure_streams = failure.get("streams", {})
    for name, path in (
        ("rocprof.stdout", rocprof_stdout),
        ("rocprof.stderr", rocprof_stderr),
    ):
        expected = stream_record(path)
        if failure_streams.get(name) != {
            "bytes": expected["bytes"],
            "sha256": expected["sha256"],
        }:
            raise OperatorError("previous actual-v12 capture stream differs")

    embedded = audit.get("evidence", {})
    for name in ("maintenance", "execute", "runtime", "capture", "operator_result"):
        if embedded.get(name) != inventories[name]:
            raise OperatorError("previous actual-v12 embedded inventory differs")
    _previous_actual_v12_commit_authority(inventories)
    return {
        "state": "executed_sealed",
        "artifact_commit": PREVIOUS_ACTUAL_V12_COMMIT,
        "artifact_tree": PREVIOUS_ACTUAL_V12_TREE,
        "file_count": sum(
            len(inventory["members"]) + 1 for inventory in inventories.values()
        ),
        "fresh_outputs": state,
        "outcome": result["status"],
        "returncode": result["returncode"],
        "invocation_count": result["invocation_count"],
        "maximum_invocations": result["maximum_invocations"],
        "retry_performed": result["retry_performed"],
        "previous_operator_v12": previous,
        "inventories": inventories,
        "actual_executed": True,
    }


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
    cleanup_value = maintenance.get("lock_substrate_cleanup")
    cleanup = cleanup_value if isinstance(cleanup_value, dict) else {}
    if package.get("full_hash_count") != 1 or package.get("full_content", {}).get("passed") is not True or package.get("integrity_identity", {}).get("passed") is not True:
        raise OperatorError("package exact-one integrity differs")
    no_op_restore = restore.get("attempted") is False
    pre_stop_failure_snapshot = None
    if no_op_restore:
        if succeeded:
            raise OperatorError("successful execution cannot use pre-stop no-op restore")
        pre_stop_failure_snapshot = pre_stop_noop_failure_record(
            maintenance,
            maintenance_inventory,
        )
        if any(
            root.exists() or root.is_symlink()
            for root in (PROFILE_EXECUTE_EVIDENCE, PROFILE_RUNTIME, PROFILE_CAPTURE)
        ):
            raise OperatorError("pre-stop failure produced downstream artifacts")
    else:
        if (
            restore.get("attempted") is not True
            or restore.get("passed") is not True
            or restore.get("duration_ns", 120_000_000_001) > 120_000_000_000
            or restore.get("final_metadata_recheck", {}).get("within_absolute_deadline")
            is not True
        ):
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

    post = capture_recovery_snapshot(load(PROFILE_READY, "profile ready binding"))
    running = {"service": post["service"], "worker": post["worker"], "gpu": post["gpu"], "owners": post["owners"], "lock": post["lock"], "hashes": post["hashes"], "formal_health_sha256": post["formal_health_sha256"], "targeted_processes": post["targeted_processes"]}
    pre = quiet["confirmation"]
    if post["owners"] != {"amd_smi": [post["worker"]["pid"]], "kfd": [post["worker"]["pid"]]} or post["lock"].get("busy") is not True or post["targeted_processes"]:
        raise OperatorError("post-restore owner/residual state differs")
    if no_op_restore:
        if (
            post["service"] != pre["service"]
            or post["worker"] != pre["worker"]
            or post["gpu"] != pre["gpu"]
            or post["owners"] != pre["owners"]
            or post["lock"] != pre["lock"]
            or post["hashes"] != pre["hashes"]
            or post["formal_health_sha256"] != pre["formal_health_sha256"]
            or post["service"].get("active_state") != "active"
            or post["service"].get("sub_state") != "running"
            or post["service"].get("nrestarts") != 0
        ):
            raise OperatorError("pre-stop no-op recovery epoch differs")
    else:
        if post["service"].get("active_state") != "active" or post["service"].get("sub_state") != "running" or post["service"].get("nrestarts") != 0 or post["service"].get("main_pid") == pre["service"]["main_pid"] or post["worker"]["pid"] == pre["worker"]["pid"]:
            raise OperatorError("post-restore service epoch differs")
        if post["hashes"] != pre["hashes"] or post["formal_health_sha256"] != pre["formal_health_sha256"]:
            raise OperatorError("post-restore health/hash state differs")

    finalizer_authority = finalizer_source_authority()

    operator_result = {
        "schema_version": OPERATOR_RESULT_SCHEMA,
        "status": outcome_status,
        "authority_commit": manifest["inputs"]["profile_ready"]["artifact_commit"],
        "operator_manifest_commit": git("log", "-1", "--format=%H", "--", str((OPERATOR_ROOT / "command-manifest.json").relative_to(ROOT))),
        "manifest_file_sha256": sha_file(OPERATOR_ROOT / "command-manifest.json"),
        "manifest_semantic_sha256": manifest["manifest_sha256"],
        "command_sha256": manifest["command_sha256"],
        "finalizer_authority": finalizer_authority,
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
        "finalizer_authority": finalizer_authority,
        "execution": {key: operator_result[key] for key in ("argument_count", "working_directory", "shell", "same_pty_sudo_cache", "maximum_invocations", "invocation_count", "retry_performed", "returncode", "canonical_start_unix_ns", "canonical_end_unix_ns", "elapsed_ns")},
        "failure": failure,
        "all_returncodes_and_streams": {"operator": {"returncode": returncode, "stdout": operator_result["stdout"], "stderr": operator_result["stderr"]}, "runner": {"returncode": runner.get("exit_code"), "stdout": optional_stream(PROFILE_EXECUTE_EVIDENCE, runner.get("stdout")), "stderr": optional_stream(PROFILE_EXECUTE_EVIDENCE, runner.get("stderr"))}, "validator": {"returncode": validator.get("exit_code"), "stdout": optional_stream(PROFILE_EXECUTE_EVIDENCE, validator.get("stdout")), "stderr": optional_stream(PROFILE_EXECUTE_EVIDENCE, validator.get("stderr"))}, "rocprof": {"returncode": capture_process.get("exit_code"), "stdout": stream_record(PROFILE_CAPTURE / "rocprof.stdout") if (PROFILE_CAPTURE / "rocprof.stdout").is_file() else None, "stderr": stream_record(PROFILE_CAPTURE / "rocprof.stderr") if (PROFILE_CAPTURE / "rocprof.stderr").is_file() else None}},
        "package_integrity": package,
        "restore": restore,
        "restore_classification": "pre_stop_untouched_same_epoch" if no_op_restore else "outer_finally_restored_new_epoch",
        "pre_stop_failure_snapshot": pre_stop_failure_snapshot,
        "recovery_snapshot": post,
        "post_health": running,
        "cleanup": {"capture_children_remaining": capture_failure.get("children_remaining") if isinstance(capture_failure, dict) else [], "capture_process_group_cleanup_complete": capture_failure.get("process_group_cleanup_complete") if isinstance(capture_failure, dict) else True, "launcher_children_remaining": launcher_failure.get("children_remaining", []), "launcher_cleanup_passed": launcher_failure.get("cleanup_passed", True), "driver_cleanup_passed": driver_process.get("cleanup", {}).get("passed") if isinstance(driver_process, dict) else None, "residual_targeted_processes": post["targeted_processes"], "trusted_lock_substrate_cleanup_required": not no_op_restore, "trusted_lock_substrate_cleanup_passed": True if no_op_restore else cleanup["passed"], "trusted_lock_substrate_holder_pids": [] if no_op_restore else cleanup["holder_pids"], "retry_forbidden_and_not_performed": True},
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
    manifest = validate_operator()["value"]
    result_inventory = verify_sums(OPERATOR_RESULT)
    audit_inventory = verify_sums(ACTUAL_AUDIT)
    result = load(OPERATOR_RESULT / "operator-result.json", "operator result")
    audit = load(ACTUAL_AUDIT / "actual-audit.json", "actual audit")
    validate_actual_documents(result, audit)
    manifest_commit = git(
        "log",
        "-1",
        "--format=%H",
        "--",
        str((OPERATOR_ROOT / "command-manifest.json").relative_to(ROOT)),
    )
    if (
        result.get("operator_manifest_commit") != manifest_commit
        or result.get("manifest_file_sha256")
        != sha_file(OPERATOR_ROOT / "command-manifest.json")
        or result.get("manifest_semantic_sha256") != manifest.get("manifest_sha256")
        or result.get("command_sha256") != manifest.get("command_sha256")
        or audit.get("authority_commit") != manifest_commit
        or audit.get("authority_commit") != result.get("operator_manifest_commit")
        or audit.get("manifest_file_sha256")
        != result.get("manifest_file_sha256")
    ):
        raise OperatorError("actual operator manifest binding differs")
    validate_finalizer_source_authority(result.get("finalizer_authority"))
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
