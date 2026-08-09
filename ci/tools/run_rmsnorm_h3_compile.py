#!/usr/bin/env python3
"""Run the dedicated RMSNorm H3 compile/link/extract/inspect contract.

This runner deliberately has no HIP runtime entry point.  It only invokes the
pinned compiler and LLVM inspection tools, and it emits a row-private evidence
bundle after every required check has passed.  The strict mode is intended for
the fixed container workflow; ``--non-strict-local`` is an explicit, isolated
developer mode for a dirty checkout with the same compile-only checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ("gfx1030", "gfx1201")
ROWS = tuple(f"h3-rmsnorm-{target}" for target in TARGETS)
E_FLAGS = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}
PINNED_IMAGE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
PINNED_CONFIG = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
COMPILER = "/opt/rocm/bin/amdclang++"
ROCM_ROOT = "/opt/rocm"
LLVM_TOOLS = {
    "clang_offload_bundler": "/opt/rocm/lib/llvm/bin/clang-offload-bundler",
    "llvm_objcopy": "/opt/rocm/lib/llvm/bin/llvm-objcopy",
    "llvm_readobj": "/opt/rocm/lib/llvm/bin/llvm-readobj",
}
EXPECTED_FEATURES = {"xnack": "unsupported", "sramecc": "unsupported", "generic_processor_version": 0}
LOGICAL_KERNEL = "rmsnorm.baseline.wave32.v1"
DEVICE_SYMBOL = "sllm_rmsnorm_baseline_wave32_v1"
HOST_BUNDLE_ID = "host-x86_64-unknown-linux-gnu-"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ROOT_NAME = re.compile(r"^sllm-rmsnorm-h3-[A-Za-z0-9_.-]+$")
_SCHEMA_FILES = {
    "compile": "ci/schema/rmsnorm-h3-compile-v1.schema.json",
    "artifact": "ci/schema/rmsnorm-h3-artifact-v1.schema.json",
    "report": "ci/schema/rmsnorm-h3-report-v1.schema.json",
    "aggregate": "ci/schema/rmsnorm-h3-aggregate-v1.schema.json",
}
_EXPECTED_SOURCE_SETS = {
    "device": (
        "include/sllm/hip.h",
        "native/hip/src/rmsnorm_kernel.hip.cpp",
        "native/hip/src/rmsnorm_kernel_internal.hpp",
    ),
    "host_abi": (
        "native/hip/src/public_runtime.hip.cpp",
        "native/hip/src/public_runtime_internal.hpp",
        "native/hip/src/rmsnorm_api.cpp",
        "native/hip/src/rmsnorm_api.hpp",
    ),
    "binding_build": (
        "crates/sllm-hip-sys/build.rs",
        "crates/sllm-hip-sys/src/bindings.rs",
        "crates/sllm-hip/src/rmsnorm.rs",
        "crates/sllm-core/src/op.rs",
        "native/hip/CMakeLists.txt",
    ),
    "ci_contract": (
        "ci/schema/rmsnorm-h3-compile-v1.schema.json",
        "ci/schema/rmsnorm-h3-artifact-v1.schema.json",
        "ci/schema/rmsnorm-h3-report-v1.schema.json",
        "ci/schema/rmsnorm-h3-aggregate-v1.schema.json",
        "ci/tools/validate_rmsnorm_h3_contracts.py",
        "ci/tools/run_rmsnorm_h3_compile.py",
        "ci/tools/aggregate_rmsnorm_h3_results.py",
        "ci/tests/test_rmsnorm_h3_contracts.py",
        "ci/tests/test_rmsnorm_h3_runner.py",
        "ci/tests/test_rmsnorm_h3_aggregate.py",
        ".github/workflows/rmsnorm-h3-compile.yml",
    ),
}
PUBLIC_ABI_SYMBOLS = (
    "sllm_backend_probe", "sllm_buffer_copy_d2h", "sllm_buffer_copy_h2d", "sllm_buffer_create",
    "sllm_buffer_release", "sllm_buffer_size", "sllm_completion_query", "sllm_completion_read",
    "sllm_completion_release", "sllm_completion_wait", "sllm_context_create", "sllm_context_probe",
    "sllm_context_release", "sllm_device_count", "sllm_device_query", "sllm_event_create",
    "sllm_event_release", "sllm_get_abi_version", "sllm_query_version", "sllm_queue_create",
    "sllm_queue_release", "sllm_rmsnorm_execute", "sllm_rmsnorm_plan_release", "sllm_rmsnorm_prepare",
)
SOURCE_SYMBOL_MAP = [
    {"path": "include/sllm/hip.h", "symbol": "sllm_rmsnorm_execute", "role": "declaration"},
    {"path": "native/hip/src/public_runtime.hip.cpp", "symbol": "sllm_rmsnorm_execute", "role": "definition"},
    {"path": "crates/sllm-hip-sys/src/bindings.rs", "symbol": "sllm_rmsnorm_execute", "role": "abi-binding"},
    {"path": "crates/sllm-hip/src/rmsnorm.rs", "symbol": "PreparedRmsNorm::execute", "role": "wrapper"},
    {"path": "native/hip/src/rmsnorm_kernel.hip.cpp", "symbol": DEVICE_SYMBOL, "role": "device-definition"},
]


class ContractError(RuntimeError):
    """Raised for any contract, identity, tool, or artifact violation."""


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON document is not an object: {path}")
    return value


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ContractError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def identity(repo: Path) -> tuple[str, str, bool]:
    commit = git(repo, "rev-parse", "HEAD")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    if not SHA40.fullmatch(commit) or not SHA40.fullmatch(tree):
        raise ContractError("Git identity is not a complete lowercase SHA")
    clean = not bool(git(repo, "status", "--porcelain=v1", "--untracked-files=all"))
    return commit, tree, clean


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def reject_symlink_components(path: Path, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} contains a symlink component: {current}")


def require_regular(path: Path, label: str) -> None:
    reject_symlink_components(path, label)
    if not path.exists() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} is missing, symlinked, or not a regular file: {path}")


def require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ContractError(f"{label} is not a lowercase SHA-256")
    return value


def check_source_sets(repo: Path, matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    observed_sets: dict[str, dict[str, Any]] = {}
    source_sets = matrix.get("source_sets")
    if not isinstance(source_sets, dict) or set(source_sets) != set(_EXPECTED_SOURCE_SETS):
        raise ContractError("RMSNorm source-set inventory is missing or has unknown sets")
    for set_name, expected_paths in _EXPECTED_SOURCE_SETS.items():
        inventory = source_sets[set_name]
        if not isinstance(inventory, dict) or inventory.get("canonical_order") != list(expected_paths):
            raise ContractError(f"{set_name} source order is missing, reordered, duplicated, or extra")
        files = inventory.get("files")
        if not isinstance(files, list) or len(files) != len(expected_paths):
            raise ContractError(f"{set_name} source inventory count is wrong")
        observed: list[dict[str, str]] = []
        for item, expected_path in zip(files, expected_paths):
            if not isinstance(item, dict) or set(item) != {"path", "sha256"} or item["path"] != expected_path:
                raise ContractError(f"{set_name} source entry is malformed or out of order")
            require_regular(repo / expected_path, f"source {expected_path}")
            digest = sha256_file(repo / expected_path)
            if item["sha256"] != digest:
                raise ContractError(f"source hash drift: {expected_path}")
            observed.append({"path": expected_path, "sha256": digest})
        if inventory.get("source_set_sha256") != sha256_json(observed):
            raise ContractError(f"canonical {set_name} source-set hash is stale")
        observed_sets[set_name] = {"canonical_order": list(expected_paths), "files": observed, "source_set_sha256": sha256_json(observed)}
    return observed_sets


def check_symbol_map(repo: Path, matrix: dict[str, Any]) -> None:
    if matrix.get("source_symbol_map") != SOURCE_SYMBOL_MAP:
        raise ContractError("RMSNorm source-symbol map is missing, reordered, or changed")
    checks = {
        "include/sllm/hip.h": re.compile(r"\bsllm_rmsnorm_execute\s*\("),
        "native/hip/src/public_runtime.hip.cpp": re.compile(r'extern\s+"C"\s+sllm_status_t\s+sllm_rmsnorm_execute\s*\('),
        "crates/sllm-hip-sys/src/bindings.rs": re.compile(r"\bpub\s+fn\s+sllm_rmsnorm_execute\s*\("),
        "crates/sllm-hip/src/rmsnorm.rs": re.compile(r"impl\s+PreparedRmsNorm\b[\s\S]*?\bpub\s+fn\s+execute\s*\("),
        "native/hip/src/rmsnorm_kernel.hip.cpp": re.compile(r"\bsllm_rmsnorm_baseline_wave32_v1\s*\("),
    }
    for path, expression in checks.items():
        text = (repo / path).read_text(encoding="utf-8")
        count = len(expression.findall(text))
        if count != 1:
            raise ContractError(f"source-symbol map occurrence count for {path} is {count}, expected 1")


def validate_matrix(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    matrix_path = repo / "ci/matrix/rmsnorm-h3-compile-v1.json"
    matrix = read_json(matrix_path)
    expected_top = {"$schema", "schema_version", "matrix_id", "revision", "suite_id", "tier", "toolchain_id", "container", "workflow", "source_sets", "source_symbol_map", "public_abi_symbols", "logical_kernel", "device_symbol", "case_manifest", "rows"}
    if set(matrix) != expected_top:
        raise ContractError("RMSNorm matrix has missing or unknown top-level fields")
    if matrix["$schema"] != "https://sllm-project.local/ci/schema/rmsnorm-h3-compile-v1.schema.json" or matrix["schema_version"] != "rmsnorm-h3-compile-v1" or matrix["matrix_id"] != "rmsnorm-h3-compile-v1" or matrix["revision"] != 1:
        raise ContractError("RMSNorm matrix identity is invalid")
    if matrix["suite_id"] != "h3-rmsnorm-compile-only" or matrix["tier"] != "tier_h3_rmsnorm" or matrix["toolchain_id"] != "rocm-7.14.0":
        raise ContractError("RMSNorm matrix suite/tier/toolchain is not fixed")
    expected_container = {"image_reference": PINNED_IMAGE, "image_config_digest": PINNED_CONFIG, "platform": {"os": "linux", "architecture": "amd64"}, "rocm_root": "/opt/rocm", "compiler": COMPILER, "llvm_major": 23}
    if matrix["container"] != expected_container:
        raise ContractError("RMSNorm matrix container/toolchain tuple drifted")
    if matrix["workflow"]["path"] != ".github/workflows/rmsnorm-h3-compile.yml":
        raise ContractError("RMSNorm workflow path is not dedicated")
    require_regular(repo / matrix["workflow"]["path"], "RMSNorm workflow")
    if matrix["workflow"]["sha256"] != sha256_file(repo / matrix["workflow"]["path"]):
        raise ContractError("RMSNorm workflow SHA-256 is stale")
    if matrix["public_abi_symbols"] != list(PUBLIC_ABI_SYMBOLS) or matrix["logical_kernel"] != LOGICAL_KERNEL or matrix["device_symbol"] != DEVICE_SYMBOL:
        raise ContractError("RMSNorm public ABI or kernel identity is not canonical")
    expected_case = {"id": "rmsnorm-h3-compile-link-extract-inspect-v1", "selected_count": 1, "collected_count": 1, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fake_hip": False, "emulation": False}
    if matrix["case_manifest"] != expected_case:
        raise ContractError("RMSNorm case selection/count contract drifted")
    rows = matrix["rows"]
    if not isinstance(rows, list) or len(rows) != 2 or {row.get("row_id") for row in rows} != set(ROWS):
        raise ContractError("RMSNorm matrix must contain exactly the two exact rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = row.get("target")
        expected_id = f"h3-rmsnorm-{target}"
        if row.get("row_id") != expected_id or target not in TARGETS or row.get("tier") != "tier_h3_rmsnorm" or row.get("required") is not False or row.get("seed") != int(target[3:]):
            raise ContractError("RMSNorm row identity/tier/requiredness is invalid")
        if set(row) != {"row_id", "target", "tier", "required", "seed", "execution", "build", "resource", "output", "codegen"}:
            raise ContractError(f"{expected_id} has missing or unknown fields")
        expected_execution = {"mode": "compile-link-extract-inspect-only", "compile_only": True, "requires_gpu": False, "requires_model": False, "network": False, "fallback_allowed": False, "execution_attempted": False, "gpu_execution": False, "fake_hip": False, "emulation": False}
        if row["execution"] != expected_execution:
            raise ContractError(f"{expected_id} permits execution, model, fallback, fake HIP, or emulation")
        expected_codegen = {"target": target, "target_kind": "exact", "target_count": 1, "code_object_version": "V6", "wavefront_size": 32, "features": EXPECTED_FEATURES, "e_flags": E_FLAGS[target]}
        if row["codegen"] != expected_codegen:
            raise ContractError(f"{expected_id} has wrong exact target/codegen tuple")
        if row["resource"] != {"max_rss_bytes": 4294967296, "max_output_bytes": 16777216, "timeout_seconds": 900, "max_output_file_bytes": 268435456}:
            raise ContractError(f"{expected_id} resource bounds drifted")
        build = row["build"]
        if build.get("generator") != "direct-amdclang++" or build.get("mode") != "compile-link-extract-inspect" or build.get("build_type") != "Release" or build.get("language_standard") != "gnu++17" or build.get("sources") != ["native/hip/src/rmsnorm_kernel.hip.cpp", "native/hip/src/public_runtime.hip.cpp", "native/hip/src/rmsnorm_api.cpp"] or len(build.get("commands", [])) != 4:
            raise ContractError(f"{expected_id} build contract is not the dedicated RMSNorm tuple")
        for command in build["commands"]:
            if not command or command[0] != COMPILER or any(any(token in value for token in (";", "&&", "||", "`", "$(")) for value in command):
                raise ContractError(f"{expected_id} has a shell or unpinned compiler command")
            if sum(value == "--offload-arch={target}" for value in command) != 1 or "-mcode-object-version=6" not in command or "-mno-wavefrontsize64" not in command:
                raise ContractError(f"{expected_id} does not pin one V6 wave32 target")
            if any(target in value for value in command):
                raise ContractError(f"{expected_id} embeds an additional literal target")
        if "--hip-link" not in build["commands"][-1] or "--rtlib=compiler-rt" not in build["commands"][-1] or "-unwindlib=libgcc" not in build["commands"][-1] or "-pthread" not in build["commands"][-1]:
            raise ContractError(f"{expected_id} link command is not the pinned HIP link contract")
        if row["output"] != {"root_prefix": "/tmp/sllm-rmsnorm-h3-", "directory_pattern": "h3-rmsnorm-{target}", "host_elf_pattern": "host-bundle-{target}.elf", "device_object_pattern": "device-code-object-{target}.elf", "artifact_metadata": "rmsnorm-h3-artifact.json", "report": "rmsnorm-h3-report.json"}:
            raise ContractError(f"{expected_id} output contract drifted")
        by_id[expected_id] = row
    source_sets = check_source_sets(repo, matrix)
    check_symbol_map(repo, matrix)
    return read_json(repo / "ci/toolchains/rocm-7.14.0.json"), matrix, by_id


def command_version(path: Path) -> str:
    result = subprocess.run([str(path), "--version"], text=True, capture_output=True, check=False, timeout=30)
    if result.returncode != 0 or len(result.stdout) + len(result.stderr) > 1024 * 1024:
        raise ContractError(f"cannot query tool version: {path}")
    return (result.stdout or result.stderr).strip()


def inspect_toolchain(toolchain: dict[str, Any]) -> dict[str, Any]:
    if toolchain.get("schema_version") != "rocm-toolchain-v1" or toolchain.get("toolchain_id") != "rocm-7.14.0":
        raise ContractError("ROCm toolchain manifest is not the pinned 7.14.0 manifest")
    paths = toolchain.get("paths", {})
    for name, expected in {"rocm_root": ROCM_ROOT, "compiler": COMPILER, **LLVM_TOOLS}.items():
        if paths.get(name) != expected:
            raise ContractError(f"toolchain path drifted: {name}")
        path = Path(expected)
        if name == "rocm_root":
            if not path.is_dir() or path.is_symlink():
                raise ContractError(f"toolchain root is missing or symlinked: {expected}")
        else:
            if not path.exists() or not path.is_file():
                raise ContractError(f"toolchain executable is missing: {expected}")
            try:
                path.resolve(strict=True).relative_to(Path(ROCM_ROOT).resolve(strict=True))
            except ValueError as exc:
                raise ContractError(f"toolchain executable resolves outside /opt/rocm: {expected}") from exc
        if name != "rocm_root" and not os.access(path, os.X_OK):
            raise ContractError(f"toolchain executable is not executable: {expected}")
    compiler_version = command_version(Path(COMPILER))
    if not re.search(r"clang version 23\.", compiler_version, re.IGNORECASE):
        raise ContractError("ROCm compiler is not LLVM major 23")
    if not any(path.is_file() and "7.14.0" in path.read_text(encoding="utf-8").strip() for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm/core-7.14/.info/version"))):
        raise ContractError("ROCm 7.14.0 was not observed under /opt/rocm")
    return {"id": "rocm-7.14.0", "rocm_version": "7.14.0", "llvm_major": 23, "compiler": COMPILER, "compiler_version": compiler_version, "paths": {**LLVM_TOOLS}}


def network_isolated() -> bool:
    try:
        names = [name for _, name in socket.if_nameindex()]
        if names != ["lo"]:
            return False
        route_lines = Path("/proc/net/route").read_text(encoding="ascii").splitlines()[1:]
        return not any(line.split()[0] != "lo" for line in route_lines if line.split())
    except (OSError, UnicodeError):
        return False


def render_commands(row: dict[str, Any], repo: Path, build_dir: Path) -> list[list[str]]:
    result: list[list[str]] = []
    replacements = {"{repo}": str(repo), "{build_dir}": str(build_dir), "{target}": row["target"]}
    for template in row["build"]["commands"]:
        command: list[str] = []
        for token in template:
            value = token
            for placeholder, replacement in replacements.items():
                value = value.replace(placeholder, replacement)
            if "{" in value or "}" in value:
                raise ContractError("unresolved build command placeholder")
            command.append(value)
        result.append(command)
    return result


def _kill_process_group(process: subprocess.Popen[bytes]) -> bool:
    cleaned = True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        cleaned = False
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            cleaned = False
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            cleaned = False
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    except OSError:
        cleaned = False
    else:
        cleaned = False
    return cleaned


def run_process(argv: list[str], *, cwd: Path, timeout: int, output_limit: int) -> dict[str, Any]:
    if not argv or argv[0] != COMPILER and not argv[0].startswith("/opt/rocm/lib/llvm/bin/"):
        raise ContractError(f"child command is outside the pinned compiler/LLVM toolchain: {argv[0] if argv else '<empty>'}")
    if any(any(part in token for part in (";", "&&", "||", "`", "$(")) for token in argv):
        raise ContractError("child command contains shell syntax")
    started = time.monotonic()
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, env={"PATH": "/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C", "LC_ALL": "C"})
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        cleanup = _kill_process_group(process)
        stdout, stderr = process.communicate()
        if not cleanup:
            raise ContractError("timed-out child process cleanup was not proven") from exc
    duration = round(time.monotonic() - started, 6)
    if len(stdout) + len(stderr) > output_limit:
        raise ContractError("child output exceeded the fail-closed limit")
    exit_code = process.returncode
    if timed_out or exit_code != 0:
        message = stderr.decode("utf-8", "replace")[-2000:]
        raise ContractError(f"child failed ({exit_code}, timed_out={timed_out}): {message}")
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        cleanup_proven = True
    except OSError:
        cleanup_proven = False
    else:
        cleanup_proven = False
    if not cleanup_proven:
        raise ContractError("successful child process group cleanup was not proven")
    return {"argv": argv, "state": "PASS", "exit_code": 0, "timed_out": False, "duration_seconds": duration, "stdout_sha256": sha256_bytes(stdout), "stderr_sha256": sha256_bytes(stderr), "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "_stdout": stdout, "_stderr": stderr}


def section_sizes(readobj_output: str) -> dict[str, int]:
    sections: dict[str, int] = {}
    for match in re.finditer(r"Name:\s+(\S+)[\s\S]{0,1200}?Size:\s+(0x[0-9a-fA-F]+|[0-9]+)", readobj_output):
        name = match.group(1)
        if name in {".text", ".hip_fatbin", ".kd"}:
            sections[name] = int(match.group(2), 0)
    return sections


def symbol_names(readobj_output: str) -> list[str]:
    names: list[str] = []
    for value in re.findall(r"(?m)^\s*Name:\s*(\S.*)$", readobj_output):
        name = re.sub(r"\s+\([0-9]+\)$", "", value.strip())
        if name:
            names.append(name)
    return names


def run_readobj(path: Path, tool: Path, row: dict[str, Any], cwd: Path) -> tuple[str, dict[str, Any]]:
    step = run_process([str(tool), "--file-headers", "--sections", "--symbols", "--notes", str(path)], cwd=cwd, timeout=60, output_limit=row["resource"]["max_output_bytes"])
    output = step.pop("_stdout").decode("utf-8", "replace")
    step.pop("_stderr", None)
    return output, step


def inspect_host(path: Path, output: str, row: dict[str, Any], bundles: list[str]) -> dict[str, Any]:
    lowered = output.lower()
    if "elf64-x86-64" not in lowered and not re.search(r"\bArch:\s*x86_64\b", output):
        raise ContractError("host bundle is not an ELF64 x86-64 host ELF")
    sections = section_sizes(output)
    if sections.get(".text", 0) < 1 or sections.get(".hip_fatbin", 0) < 1:
        raise ContractError("host ELF does not contain non-empty .text and .hip_fatbin")
    expected_bundles = [f"hipv4-amdgcn-amd-amdhsa--{row['target']}", HOST_BUNDLE_ID]
    if bundles != expected_bundles or len(set(bundles)) != 2:
        raise ContractError("host ELF bundle list is not the exact target plus host tuple")
    names = symbol_names(output)
    sllm_names = [name for name in names if name.startswith("sllm_")]
    allowed = set(PUBLIC_ABI_SYMBOLS) | {DEVICE_SYMBOL}
    if any(name not in allowed for name in sllm_names):
        raise ContractError("host ELF contains an unexpected sllm symbol")
    for name in PUBLIC_ABI_SYMBOLS:
        if sllm_names.count(name) != 1:
            raise ContractError(f"host ELF public ABI symbol count is not exactly one: {name}")
    return {"format": "ELF64", "machine": "X86_64", "sections": {".text": sections[".text"], ".hip_fatbin": sections[".hip_fatbin"]}, "bundles": bundles, "public_symbols": [{"name": name, "defined": True} for name in PUBLIC_ABI_SYMBOLS], "stub_symbols": []}


def inspect_device(path: Path, output: str, row: dict[str, Any]) -> dict[str, Any]:
    if "elf64-amdgpu" not in output.lower() and not re.search(r"\bArch:\s*amdgcn\b", output):
        raise ContractError("extracted device object is not an AMDGPU ELF")
    header = output.split("Sections [", 1)[0]
    abi = re.findall(r"(?m)^\s*ABIVersion:\s*(\d+)\s*$", header)
    flags = re.findall(r"(?m)^\s*Flags\s+\[\s*\(0x([0-9a-fA-F]+)\)", header)
    targets = re.findall(r"(?m)^\s*amdhsa\.target:\s*(\S+)\s*$", output)
    waves = re.findall(r"(?m)^\s*\.wavefront_size:\s*(\d+)\s*$", output)
    if len(abi) != 1 or len(flags) != 1 or len(targets) != 1 or len(waves) != 1:
        raise ContractError("device ELF does not prove exactly one ABI, e_flags, target, and wavefront")
    observed_flags = f"0x{int(flags[0], 16):08x}"
    if abi[0] != "4" or observed_flags != E_FLAGS[row["target"]] or targets[0] != f"amdgcn-amd-amdhsa--{row['target']}" or waves[0] != "32":
        raise ContractError("device ELF target/V6/wave32/e_flags mismatch")
    value = int(flags[0], 16)
    xnack = {0x000: "unsupported", 0x100: "any", 0x200: "off", 0x300: "on"}.get(value & 0x300)
    sramecc = {0x000: "unsupported", 0x400: "any", 0x800: "off", 0xC00: "on"}.get(value & 0xC00)
    features = {"xnack": xnack, "sramecc": sramecc, "generic_processor_version": (value >> 24) & 0xFF}
    if features != EXPECTED_FEATURES:
        raise ContractError("device ELF feature fields are not the exact unsupported tuple")
    sections = section_sizes(output)
    if sections.get(".text", 0) < 1:
        raise ContractError("device ELF has no non-empty .text")
    names = symbol_names(output)
    relevant = [name for name in names if name == DEVICE_SYMBOL or name.startswith(DEVICE_SYMBOL + ".") or name.startswith("sllm_")]
    kd = [name for name in relevant if name == DEVICE_SYMBOL + ".kd"]
    if relevant.count(DEVICE_SYMBOL) != 1 or len(kd) != 1 or any(name not in {DEVICE_SYMBOL, DEVICE_SYMBOL + ".kd"} for name in relevant):
        raise ContractError("device ELF does not contain exactly the RMSNorm kernel and .kd symbols")
    return {"format": "ELF64", "machine": "AMDGPU", "target": row["target"], "ei_abiversion": 4, "e_flags": observed_flags, "code_object_version": "V6", "wavefront_size": 32, "features": features, "sections": {".text": sections[".text"], ".kd": 1}, "symbols": [{"name": DEVICE_SYMBOL, "defined": True}, {"name": DEVICE_SYMBOL + ".kd", "defined": True}], "source_attribution": "rmsnorm_kernel.hip.cpp"}


def write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite output: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    if sidecar.exists() or sidecar.is_symlink():
        raise ContractError(f"refusing to overwrite sidecar: {sidecar}")
    sidecar.write_text(f"{sha256_file(path)}  {path.name}\n", encoding="ascii")
    return sha256_file(sidecar)


def copy_exclusive(source: Path, destination: Path) -> None:
    require_regular(source, f"build output {source.name}")
    if destination.exists() or destination.is_symlink():
        raise ContractError(f"refusing to overwrite artifact: {destination}")
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)


def file_record(path: Path, sidecar_sha256: str) -> dict[str, Any]:
    return {"path": path.name, "sha256": sha256_file(path), "sidecar_sha256": sidecar_sha256, "size_bytes": path.stat().st_size}


def output_root(path: Path) -> Path:
    absolute = _absolute(path)
    reject_symlink_components(absolute, "output root")
    if absolute.parent != Path("/tmp") or not ROOT_NAME.fullmatch(absolute.name):
        raise ContractError("output must be a new direct child of /tmp with the sllm-rmsnorm-h3 prefix")
    if absolute.exists() or absolute.is_symlink():
        raise ContractError("output root already exists; overwrite is forbidden")
    absolute.mkdir(mode=0o700)
    return absolute


def scope() -> dict[str, Any]:
    return {"compile_only": True, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "fake_hip": False, "emulation": False, "support_claim": False, "numerics_verified": False, "performance_verified": False}


def run_row(args: argparse.Namespace) -> dict[str, Any]:
    repo = _absolute(Path(args.repo))
    if not repo.is_dir() or repo.is_symlink():
        raise ContractError("repository must be a regular non-symlink directory")
    if bool(args.strict_ci) == bool(args.non_strict_local):
        raise ContractError("choose exactly one of --strict-ci or --non-strict-local")
    if args.strict_ci and not args.pinned_container:
        raise ContractError("strict H3 requires --pinned-container")
    commit, tree, clean = identity(repo)
    if args.strict_ci and not clean:
        raise ContractError("strict RMSNorm H3 rejects a dirty checkout")
    for name, value in (("reviewed_sha", args.reviewed_sha), ("tested_sha", args.tested_sha), ("workflow_sha", args.workflow_sha), ("tree_oid", args.tree_oid)):
        if value is not None and not SHA40.fullmatch(value):
            raise ContractError(f"{name} must be a complete lowercase SHA")
    if args.strict_ci and (args.reviewed_sha, args.tested_sha, args.workflow_sha, args.tree_oid) != (commit, commit, commit, tree):
        raise ContractError("strict RMSNorm H3 requires exact commit/tree identity arguments")
    if args.reviewed_sha not in (None, commit) or args.tested_sha not in (None, commit) or args.workflow_sha not in (None, commit) or args.tree_oid not in (None, tree):
        raise ContractError("candidate identity does not equal checked-out HEAD")
    matrix_argument = _absolute(args.matrix if args.matrix.is_absolute() else repo / args.matrix)
    if matrix_argument != repo / "ci/matrix/rmsnorm-h3-compile-v1.json":
        raise ContractError("dedicated runner refuses a non-canonical matrix path")
    toolchain, matrix, rows = validate_matrix(repo)
    if args.row not in rows:
        raise ContractError(f"unknown RMSNorm row: {args.row}")
    row = rows[args.row]
    source_sets = check_source_sets(repo, matrix)
    symbols = SOURCE_SYMBOL_MAP
    compiler = inspect_toolchain(toolchain)
    if args.strict_ci:
        if args.observed_image_reference != PINNED_IMAGE or args.observed_image_config_digest != PINNED_CONFIG or os.environ.get("SLLM_H3_NETWORK_DISABLED") != "1":
            raise ContractError("strict RMSNorm H3 container identity/network boundary is not proven")
        if not network_isolated():
            raise ContractError("strict RMSNorm H3 network isolation is not proven")
    elif args.observed_image_reference is not None or args.observed_image_config_digest is not None:
        raise ContractError("local-nonstrict mode cannot claim a pinned container observation")
    root = output_root(Path(args.output_dir))
    row_dir = root / row["output"]["directory_pattern"].replace("{target}", row["target"])
    row_dir.mkdir(mode=0o700)
    build_dir = Path(tempfile.mkdtemp(prefix="sllm-rmsnorm-h3-build-"))
    started = iso_now()
    steps: list[dict[str, Any]] = []
    try:
        commands = render_commands(row, repo, build_dir)
        for index, command in enumerate(commands, 1):
            result = run_process(command, cwd=repo, timeout=row["resource"]["timeout_seconds"], output_limit=row["resource"]["max_output_bytes"])
            result.pop("_stdout", None)
            result.pop("_stderr", None)
            steps.append({"step_id": f"compile-link-{index}", **result})
        host_build = build_dir / row["output"]["host_elf_pattern"].replace("{target}", row["target"])
        if not host_build.exists():
            raise ContractError("link command did not produce the host ELF")
        host_fatbin = build_dir / "host.fatbin"
        result = run_process([LLVM_TOOLS["llvm_objcopy"], f"--dump-section=.hip_fatbin={host_fatbin}", str(host_build)], cwd=repo, timeout=60, output_limit=row["resource"]["max_output_bytes"])
        result.pop("_stdout", None)
        result.pop("_stderr", None)
        steps.append({"step_id": "extract-host-fatbin", **result})
        require_regular(host_fatbin, "host .hip_fatbin")
        bundler = LLVM_TOOLS["clang_offload_bundler"]
        bundle_list = run_process([bundler, "--list", "--type=o", f"--input={host_fatbin}"], cwd=repo, timeout=60, output_limit=row["resource"]["max_output_bytes"])
        bundle_stdout = bundle_list.pop("_stdout").decode("utf-8", "replace")
        bundle_list.pop("_stderr", None)
        steps.append({"step_id": "inspect-bundle-list", **bundle_list})
        bundles = [line.strip() for line in bundle_stdout.splitlines() if line.strip()]
        expected_bundles = [f"hipv4-amdgcn-amd-amdhsa--{row['target']}", HOST_BUNDLE_ID]
        if bundles != expected_bundles:
            raise ContractError(f"host ELF bundles are not exact: {bundles}")
        device_build = build_dir / row["output"]["device_object_pattern"].replace("{target}", row["target"])
        result = run_process([bundler, "--unbundle", "--type=o", f"--targets={expected_bundles[0]}", f"--input={host_fatbin}", f"--output={device_build}"], cwd=repo, timeout=60, output_limit=row["resource"]["max_output_bytes"])
        result.pop("_stdout", None)
        result.pop("_stderr", None)
        steps.append({"step_id": "extract-device-code-object", **result})
        require_regular(device_build, "extracted device code object")
        host_output, host_step = run_readobj(host_build, Path(LLVM_TOOLS["llvm_readobj"]), row, repo)
        steps.append({"step_id": "inspect-host-elf", **host_step})
        host_inspection = inspect_host(host_build, host_output, row, bundles)
        device_output, device_step = run_readobj(device_build, Path(LLVM_TOOLS["llvm_readobj"]), row, repo)
        steps.append({"step_id": "inspect-device-elf", **device_step})
        device_inspection = inspect_device(device_build, device_output, row)
        copy_exclusive(host_build, row_dir / host_build.name)
        host_sidecar_sha = write_sidecar(row_dir / host_build.name)
        copy_exclusive(device_build, row_dir / device_build.name)
        device_sidecar_sha = write_sidecar(row_dir / device_build.name)
        host_file = file_record(row_dir / host_build.name, host_sidecar_sha)
        device_file = file_record(row_dir / device_build.name, device_sidecar_sha)
        schema_digests = {name: sha256_file(repo / path) for name, path in _SCHEMA_FILES.items()}
        workflow_hash = sha256_file(repo / matrix["workflow"]["path"])
        machine = platform.machine().lower()
        if machine in {"x86_64", "amd64"}:
            machine = "amd64"
        environment = {"image_reference": PINNED_IMAGE if args.strict_ci else "local-rocm-7.14", "image_config_digest": PINNED_CONFIG if args.strict_ci else "unobserved", "platform": {"os": "linux", "architecture": machine}, "pinned": bool(args.strict_ci), "network_isolated": bool(args.strict_ci and network_isolated())}
        process = {"expected_children": 1, "observed_children": 1, "commands_expected": len(steps), "commands_executed": len(steps), "exit_codes": [0] * len(steps), "timed_out": False, "crashed": False, "cleanup_proven": True}
        metadata = {"schema_version": "rmsnorm-h3-artifact-v1", "artifact_id": f"rmsnorm-h3-artifact-{row['target']}", "suite_id": matrix["suite_id"], "tier": matrix["tier"], "state": "PASS", "required": False, "row_id": row["row_id"], "target": row["target"], "reviewed_sha": args.reviewed_sha or commit, "tested_sha": args.tested_sha or commit, "workflow_sha": args.workflow_sha or commit, "git_tree_oid": args.tree_oid or tree, "worktree_clean": clean, "matrix_id": matrix["matrix_id"], "matrix_manifest_sha256": sha256_json(matrix), "workflow_file_sha256": workflow_hash, "schema_digests": schema_digests, "source_sets": source_sets, "source_symbol_map": symbols, "toolchain": compiler, "container": environment, "codegen": row["codegen"], "logical_kernel": LOGICAL_KERNEL, "device_symbol": DEVICE_SYMBOL, "host_elf": {"file": host_file, **host_inspection}, "device_code_object": {"file": device_file, **device_inspection}, "case_manifest": {"id": matrix["case_manifest"]["id"], "selected_count": 1, "collected_count": 1}, "scope": scope(), "process": process, "timestamps": {"started_at": started, "finished_at": iso_now()}}
        metadata_path = row_dir / row["output"]["artifact_metadata"]
        write_json(metadata_path, metadata)
        metadata_sidecar_sha = write_sidecar(metadata_path)
        report = {"schema_version": "rmsnorm-h3-report-v1", "report_id": f"rmsnorm-h3-report-{row['target']}", "suite_id": matrix["suite_id"], "tier": matrix["tier"], "state": "PASS", "required": False, "evidence_mode": "required-ci" if args.strict_ci else "local-nonstrict", "run_id": str(args.run_id), "run_attempt": args.run_attempt, "reviewed_sha": args.reviewed_sha or commit, "tested_sha": args.tested_sha or commit, "workflow_sha": args.workflow_sha or commit, "git_tree_oid": args.tree_oid or tree, "worktree_clean": clean, "matrix_id": matrix["matrix_id"], "matrix_manifest_sha256": sha256_json(matrix), "workflow_file_sha256": workflow_hash, "schema_digests": schema_digests, "row_id": row["row_id"], "target": row["target"], "source_sets": source_sets, "source_symbol_map": symbols, "toolchain": {"id": compiler["id"], "compiler": compiler["compiler"], "compiler_version": compiler["compiler_version"], "llvm_major": compiler["llvm_major"]}, "container": environment, "codegen": row["codegen"], "logical_kernel": LOGICAL_KERNEL, "device_symbol": DEVICE_SYMBOL, "scope": scope(), "case_manifest": {"id": matrix["case_manifest"]["id"], "selected_count": 1, "collected_count": 1}, "process": process, "artifact": {"metadata": metadata_path.name, "metadata_sha256": sha256_file(metadata_path), "metadata_sidecar_sha256": metadata_sidecar_sha, "host_elf_sha256": host_file["sha256"], "device_code_object_sha256": device_file["sha256"], "host_elf_sidecar_sha256": host_file["sidecar_sha256"], "device_code_object_sidecar_sha256": device_file["sidecar_sha256"]}, "steps": [{"step_id": step["step_id"], "state": "PASS", "exit_code": 0} for step in steps], "diagnostics": [], "timestamps": {"started_at": started, "finished_at": iso_now()}}
        report_path = row_dir / row["output"]["report"]
        write_json(report_path, report)
        write_sidecar(report_path)
        expected_names = {host_file["path"], host_file["path"] + ".sha256", device_file["path"], device_file["path"] + ".sha256", metadata_path.name, metadata_path.name + ".sha256", report_path.name, report_path.name + ".sha256"}
        actual_names = {entry.name for entry in row_dir.iterdir()}
        if actual_names != expected_names:
            raise ContractError("row output contains stale, missing, duplicate, or unknown entries")
        return report
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--matrix", type=Path, default=Path("ci/matrix/rmsnorm-h3-compile-v1.json"))
    result.add_argument("--row", required=True, choices=ROWS)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--reviewed-sha")
    result.add_argument("--tested-sha")
    result.add_argument("--workflow-sha")
    result.add_argument("--tree-oid")
    result.add_argument("--run-id", default="local")
    result.add_argument("--run-attempt", type=int, default=1)
    result.add_argument("--strict-ci", action="store_true")
    result.add_argument("--non-strict-local", action="store_true")
    result.add_argument("--pinned-container", action="store_true")
    result.add_argument("--observed-image-reference")
    result.add_argument("--observed-image-config-digest")
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        if not RUN_ID.fullmatch(str(args.run_id)) or args.run_attempt < 1:
            raise ContractError("run identity is invalid")
        run_row(args)
        print(f"RMSNorm H3 {args.row}: PASS (compile/link/extract/inspect only; no GPU execution)")
        return 0
    except (ContractError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"RMSNorm H3: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
