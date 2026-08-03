#!/usr/bin/env python3
"""Run exactly one immutable, compile/link-only Phase 2 H3 row."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
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
BUNDLE_IDS = {target: f"hipv4-amdgcn-amd-amdhsa--{target}" for target in TARGETS}
E_FLAGS = {"gfx1030": "0x00000036", "gfx1201": "0x0000004e"}
FEATURES = {
    "xnack": "unsupported",
    "sramecc": "unsupported",
    "generic_processor_version": 0,
}
DIRECT_BUILD = {
    "driver": "/opt/rocm/bin/amdclang++",
    "mode": "direct-compile-link",
    "build_type": "Release",
    "timeout_seconds": 900,
    "source_relative_path": "native/hip/src/hip_compile_probe.hip.cpp",
    "object_pattern": "hip-compile-probe-{target}.o",
    "link_output_pattern": "hip-compile-probe-{target}.elf",
    "commands": [
        ["/opt/rocm/bin/amdclang++", "-D__HIP_ROCclr__=1", "-O3", "-DNDEBUG", "-std=gnu++17", "--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64", "-o", "{build_dir}/hip-compile-probe-{target}.o", "-x", "hip", "-c", "{source_path}"],
        ["/opt/rocm/bin/amdclang++", "-O3", "-DNDEBUG", "--offload-arch={target}", "-mcode-object-version=6", "-mno-wavefrontsize64", "--hip-link", "--rtlib=compiler-rt", "-unwindlib=libgcc", "{build_dir}/hip-compile-probe-{target}.o", "-o", "{build_dir}/hip-compile-probe-{target}.elf", "/opt/rocm/lib/libamdhip64.so"],
    ],
}
PINNED_IMAGE_REFERENCE = "docker.io/rocm/dev-ubuntu-24.04@sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7"
PINNED_IMAGE_CONFIG_DIGEST = "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf"
ZERO_SHA = "0" * 64
GIT_VERSION_FALLBACK = "unavailable"
RTF_UP = 0x1
RTF_REJECT = 0x200
IPV4_ROUTE_HEADER = ("Iface", "Destination", "Gateway", "Flags", "RefCnt", "Use", "Metric", "Mask", "MTU", "Window", "IRTT")
DEFAULT_H3_PATHS = {
    "rocm_root": "/opt/rocm",
    "compiler": "/opt/rocm/bin/amdclang++",
    "hip_headers": "/opt/rocm/include",
    "hip_cmake_package": "/opt/rocm/lib/cmake/hip",
    "device_libraries": "/opt/rocm/amdgcn/bitcode",
    "hip_runtime": "/opt/rocm/lib/libamdhip64.so",
    "clang_offload_bundler": "/opt/rocm/lib/llvm/bin/clang-offload-bundler",
    "llvm_objcopy": "/opt/rocm/lib/llvm/bin/llvm-objcopy",
    "llvm_readobj": "/opt/rocm/lib/llvm/bin/llvm-readobj",
    "llvm_objdump": "/opt/rocm/lib/llvm/bin/llvm-objdump",
}
DEFAULT_H3_TOOLCHAIN = {
    "toolchain_id": "rocm-7.14.0",
    "manifest_sha256": ZERO_SHA,
    "rocm": {"path": "/opt/rocm", "version": "7.14.0", "llvm_major": 23},
    "compiler": {
        "name": "amdclang++",
        "path": "/opt/rocm/bin/amdclang++",
        "version": "23.0.0git",
        "llvm_major": 23,
    },
    "paths": DEFAULT_H3_PATHS,
    "observed": {
        "compiler_version": "unavailable",
        "llvm_major": 23,
        "tools": {
            "clang_offload_bundler": "unavailable",
            "llvm_objcopy": "unavailable",
            "llvm_readobj": "unavailable",
            "llvm_objdump": "unavailable",
        },
    },
}


class H3Error(RuntimeError):
    pass


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


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise H3Error(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise H3Error(f"JSON document is not an object: {path}")
    return value


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise H3Error(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def git_identity(repo: Path) -> tuple[str, str]:
    commit = git_output(repo, "rev-parse", "HEAD")
    tree = git_output(repo, "rev-parse", "HEAD^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise H3Error("git identity is not an immutable SHA/tree pair")
    return commit, tree


def worktree_clean(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise H3Error(f"cannot inspect worktree: {result.stderr.strip()}")
    return not result.stdout.strip()


def command_version(command: Path, *args: str) -> str:
    result = subprocess.run([str(command), *args], text=True, capture_output=True, check=False)
    output = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not output:
        raise H3Error(f"{command} {' '.join(args)} failed")
    return output


def observe_git_version() -> str:
    """Observe the runner's Git without making it part of the ROCm toolchain."""

    git = shutil.which("git")
    if git is None:
        return GIT_VERSION_FALLBACK
    try:
        return command_version(Path(git), "--version")
    except (H3Error, OSError):
        return GIT_VERSION_FALLBACK


def within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def validate_manifests(repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    toolchain = read_json(repo / "ci/toolchains/rocm-7.14.0.json")
    matrix = read_json(repo / "ci/matrix/hip-compile-v1.json")
    if toolchain.get("schema_version") != "rocm-toolchain-v1" or toolchain.get("toolchain_id") != "rocm-7.14.0":
        raise H3Error("toolchain manifest is not the pinned ROCm 7.14.0 contract")
    image = toolchain.get("image", {})
    if image.get("manifest_digest") != "sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7" or image.get("config_digest") != "sha256:4c91c0d850e38a40fd669dd043ab42e9bad9a2b8a38e3f873c5a4eaced9f28cf":
        raise H3Error("toolchain image is not immutably pinned")
    if image.get("platform") != {"os": "linux", "architecture": "amd64"}:
        raise H3Error("toolchain image platform is not linux/amd64")
    if toolchain.get("rocm") != {"path": "/opt/rocm", "version": "7.14.0", "llvm_major": 23}:
        raise H3Error("ROCm release/root/LLVM tuple is not canonical")
    if matrix.get("schema_version") != "hip-compile-v1" or matrix.get("matrix_id") != "hip-compile-v1":
        raise H3Error("H3 matrix identity is invalid")
    if matrix.get("toolchain_id") != toolchain["toolchain_id"] or matrix.get("targets") != list(TARGETS):
        raise H3Error("H3 matrix is not bound to the exact two target set")
    rows = matrix.get("rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise H3Error("H3 matrix must contain exactly two rows")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise H3Error("H3 row is not an object")
        target = row.get("target")
        row_id = row.get("row_id")
        if target not in TARGETS or row_id != f"h3-{target}" or target in seen:
            raise H3Error("H3 matrix has a missing, duplicate, unknown, or swapped exact row")
        seen.add(target)
        if row.get("tier") != "tier_h3" or row.get("required") is not False:
            raise H3Error(f"H3 row is not explicitly non-required: {row_id}")
        if row.get("execution") != {"mode": "compile-only", "requires_gpu": False, "requires_model": False, "network": False, "fallback_allowed": False}:
            raise H3Error(f"H3 row execution scope is not compile-only: {row_id}")
        if row.get("direct_build") != DIRECT_BUILD:
            raise H3Error(f"H3 row direct amdclang++ contract is not canonical: {row_id}")
        if row.get("resource") != {"max_rss_bytes": 4294967296, "max_output_bytes": 16777216}:
            raise H3Error(f"H3 row resource contract is not canonical: {row_id}")
        if row.get("output") != {"root_prefix": "/tmp/sllm-h3-", "directory_pattern": "h3-{target}", "artifact_pattern": "device-code-object-{target}.elf"}:
            raise H3Error(f"H3 row output contract is not canonical: {row_id}")
        if row.get("codegen") != {"target": target, "target_kind": "exact", "target_count": 1, "code_object_version": "V6", "wavefront_size": 32, "features": FEATURES}:
            raise H3Error(f"H3 row codegen tuple is not canonical: {row_id}")
    return toolchain, matrix


def inspect_toolchain(toolchain: dict[str, Any]) -> dict[str, Any]:
    manifest_root = Path(toolchain["rocm"]["path"])
    if not manifest_root.is_absolute() or manifest_root.resolve() != Path("/opt/rocm").resolve():
        raise H3Error("ROCm root is not the canonical /opt/rocm")
    paths = toolchain["paths"]
    resolved_paths: dict[str, Path] = {}
    for name, value in paths.items():
        path = Path(value)
        if not path.is_absolute() or not within(path, manifest_root) or not path.exists():
            raise H3Error(f"toolchain path is missing or outside ROCM_PATH: {name}={value}")
        resolved_paths[name] = path
    compiler = resolved_paths["compiler"]
    if compiler.name != "amdclang++" or not os.access(compiler, os.X_OK):
        raise H3Error("ROCm compiler entry point is not executable amdclang++")
    compiler_version = command_version(compiler, "--version")
    if not re.search(r"(?:AMD )?clang version 23\.", compiler_version, re.IGNORECASE):
        raise H3Error(f"compiler is not LLVM 23: {compiler_version}")
    tool_versions: dict[str, str] = {}
    for name in ("clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump"):
        path = resolved_paths[name]
        if not os.access(path, os.X_OK):
            raise H3Error(f"LLVM inspector is not executable: {name}")
        version = command_version(path, "--version")
        if not re.search(r"(?:LLVM|clang(?:-[A-Za-z-]+)?) version 23\.", version, re.IGNORECASE):
            raise H3Error(f"LLVM inspector is not major 23: {name}={version}")
        tool_versions[name] = version
    version_files = [manifest_root / ".info/version", manifest_root / "core-7.14/.info/version"]
    actual_versions = [path.read_text(encoding="utf-8").strip() for path in version_files if path.is_file()]
    if "7.14.0" not in actual_versions:
        raise H3Error("ROCm release version was not observed as 7.14.0")
    if os.environ.get("ROCM_PATH") not in (None, str(manifest_root)):
        raise H3Error("ROCM_PATH environment disagrees with the pinned root")
    return {
        "toolchain_id": toolchain["toolchain_id"],
        "manifest_sha256": sha256_json(toolchain),
        "rocm": toolchain["rocm"],
        "compiler": toolchain["compiler"],
        "paths": paths,
        "observed": {
            "compiler_version": compiler_version,
            "llvm_major": 23,
            "tools": tool_versions,
        },
    }


def child_max_rss_bytes() -> int:
    """Return the OS-reported high-water RSS of this runner's child processes."""

    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    # Linux reports ru_maxrss in KiB.  H3 is explicitly Linux-only.
    return int(usage.ru_maxrss) * 1024


def limit_address_space(limit_bytes: int) -> None:
    """Apply the matrix address-space limit to the command process group."""

    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def read_command(
    path: Path,
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str],
    address_space_limit_bytes: int,
) -> tuple[int, bytes, bytes, float, bool, int]:
    started = time.monotonic()
    before_rss = child_max_rss_bytes()
    try:
        process = subprocess.Popen(
            [str(path), *args],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            preexec_fn=lambda: limit_address_space(address_space_limit_bytes),
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return (
                process.returncode,
                stdout,
                stderr,
                time.monotonic() - started,
                False,
                max(before_rss, child_max_rss_bytes()),
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return (
                124,
                stdout or b"",
                stderr or b"",
                time.monotonic() - started,
                True,
                max(before_rss, child_max_rss_bytes()),
            )
    except OSError as exc:
        return (
            127,
            b"",
            str(exc).encode("utf-8", "replace"),
            time.monotonic() - started,
            False,
            max(before_rss, child_max_rss_bytes()),
        )


def section_sizes(output: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for block in re.findall(r"(?m)^\s*Section \{\n(.*?)^\s*\}", output, re.DOTALL):
        name = re.search(r"\bName: ([^\n]+)", block)
        size = re.search(r"\bSize: (0x[0-9a-fA-F]+|[0-9]+)", block)
        if name and size:
            section_name = re.sub(r"\s+\(\d+\)$", "", name.group(1).strip())
            result[section_name] = int(size.group(1), 0)
    return result


def defined_symbols(output: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in re.findall(r"(?m)^\s*Symbol \{\n(.*?)^\s*\}", output, re.DOTALL):
        name = re.search(r"\bName: ([^\n]+)", block)
        section = re.search(r"\bSection: ([^\n]+)", block)
        if not name or not section:
            continue
        symbol = re.sub(r"\s+\(\d+\)$", "", name.group(1).strip())
        section_name = section.group(1).strip()
        if symbol and section_name not in {"Undefined", "UND", "0"}:
            normalized = re.sub(r"[^A-Za-z0-9_]", "_", symbol)
            if re.match(r"^[A-Za-z_]", normalized):
                result.append({"name": normalized, "defined": True})
    unique = {item["name"]: item for item in result}
    return [unique[name] for name in sorted(unique)]


def inspect_elf(path: Path, target: str, tools: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    readobj = tools["llvm_readobj"]
    result = subprocess.run([str(readobj), "--file-headers", "--sections", "--symbols", "--notes", str(path)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise H3Error(f"llvm-readobj failed for {path.name}: {result.stderr.strip()}")
    output = result.stdout
    sections = section_sizes(output)
    if "elf64-x86-64" in output.lower() or re.search(r"\bArch: x86_64\b", output):
        if ".hip_fatbin" not in sections:
            raise H3Error("host ELF does not contain an observed .hip_fatbin section")
        host = {
            "format": "ELF64",
            "machine": "X86_64",
            "bundles": [],
            "sections": {
                ".text": {"present": ".text" in sections, "size_bytes": sections.get(".text", 0)},
                ".hip_fatbin": {"present": True, "size_bytes": sections[".hip_fatbin"]},
            },
        }
        return host, {"raw": output, "sections": sections}
    if not ("elf64-amdgpu" in output.lower() or re.search(r"\bArch: amdgcn\b", output)):
        raise H3Error(f"ELF machine is neither X86_64 host nor AMDGPU device: {path.name}")
    abi = re.search(r"\bABIVersion: (\d+)", output)
    flags_area = output[output.find("Flags"):] if "Flags" in output else output
    flags = re.search(r"\bFlags\s+\[\s*\(0x([0-9a-fA-F]+)\)", flags_area)
    notes = output[output.find("Notes"):] if "Notes" in output else output
    metadata_target = re.search(r"\bamdhsa\.target:\s*amdgcn-amd-amdhsa--([A-Za-z0-9_-]+)", notes)
    wavefront_size = re.search(r"\.wavefront_size:\s*(\d+)", notes)
    if not abi or not flags or not metadata_target or not wavefront_size:
        raise H3Error("device ELF metadata did not prove ABI, e_flags, target, and wavefront size")
    if metadata_target.group(1) != target or int(wavefront_size.group(1)) != 32:
        raise H3Error("device ELF metadata target or measured wavefront size does not match the exact H3 row")
    observed_flags = f"0x{int(flags.group(1), 16):08x}"
    if int(abi.group(1)) != 4 or observed_flags != E_FLAGS[target]:
        raise H3Error("device ELF ABI/e_flags do not prove the pinned Code Object V6 target")
    device = {
        "format": "ELF64",
        "machine": "AMDGPU",
        "target": target,
        "ei_abiversion": int(abi.group(1)),
        "e_flags": observed_flags,
        "code_object_version": "V6",
        "wavefront_size": 32,
        "features": FEATURES,
        "sections": {".text": {"present": ".text" in sections, "size_bytes": sections.get(".text", 0)}},
        "symbols": defined_symbols(output),
    }
    if device["e_flags"] != E_FLAGS[target] or device["ei_abiversion"] != 4 or device["code_object_version"] != "V6" or not device["sections"][".text"]["present"] or not device["symbols"]:
        raise H3Error("device ELF metadata does not match the exact target contract")
    return device, {"raw": output, "sections": sections}


def make_host_toolchain(h3: dict[str, Any]) -> dict[str, Any]:
    git_version = observe_git_version()
    return {
        "python": platform.python_version(), "platform": platform.platform(aliased=True), "system": platform.system(), "machine": platform.machine(),
        "git": git_version, "rustc_dev": "not-applicable", "cargo_dev": "not-applicable", "rustc_msrv": "not-applicable", "cargo_msrv": "not-applicable",
        "clang_format": "not-applicable", "cmake": "not-applicable", "host_packages": {"h3": "compile-only"}, "h3": h3,
    }


def render_direct_commands(row: dict[str, Any], target: str, build_dir: Path, source_path: Path) -> list[list[str]]:
    """Render only the row-private output, exact target, and selected source path."""

    direct = row["direct_build"]
    replacements = {
        "{target}": target,
        "{build_dir}": str(build_dir),
        "{source_path}": str(source_path),
    }
    commands: list[list[str]] = []
    for template in direct["commands"]:
        command = []
        for token in template:
            rendered = token
            for placeholder, value in replacements.items():
                rendered = rendered.replace(placeholder, value)
            if "{" in rendered or "}" in rendered:
                raise H3Error("direct build command has an unresolved template placeholder")
            command.append(rendered)
        commands.append(command)
    return commands


def parse_route_number(value: str, *, base: int, width: int, field: str, line_number: int) -> int:
    pattern = rf"[0-9A-Fa-f]{{{width}}}" if base == 16 else r"[0-9]+"
    if not re.fullmatch(pattern, value):
        raise H3Error(f"malformed /proc route {field} at line {line_number}")
    return int(value, base)


def validate_ipv4_routes(text: str, interfaces: list[str]) -> None:
    lines = text.splitlines()
    if not lines or tuple(lines[0].split()) != IPV4_ROUTE_HEADER:
        raise H3Error("malformed /proc/net/route header")
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 11:
            raise H3Error(f"malformed /proc/net/route at line {line_number}")
        interface = fields[0]
        if interface not in interfaces:
            raise H3Error(f"/proc/net/route references an unavailable interface: {interface}")
        destination = parse_route_number(fields[1], base=16, width=8, field="IPv4 destination", line_number=line_number)
        parse_route_number(fields[2], base=16, width=8, field="IPv4 gateway", line_number=line_number)
        flags = parse_route_number(fields[3], base=16, width=4, field="IPv4 flags", line_number=line_number)
        for field, value in zip(("IPv4 RefCnt", "IPv4 Use", "IPv4 Metric"), fields[4:7], strict=True):
            parse_route_number(value, base=10, width=0, field=field, line_number=line_number)
        mask = parse_route_number(fields[7], base=16, width=8, field="IPv4 mask", line_number=line_number)
        for field, value in zip(("IPv4 MTU", "IPv4 Window", "IPv4 IRTT"), fields[8:11], strict=True):
            parse_route_number(value, base=10, width=0, field=field, line_number=line_number)
        if destination == 0 and mask == 0 and flags & RTF_UP and not flags & RTF_REJECT and interface != "lo":
            raise H3Error("required CI network namespace has a usable non-loopback IPv4 default route")


def validate_ipv6_routes(text: str, interfaces: list[str]) -> None:
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if len(fields) != 10:
            raise H3Error(f"malformed /proc/net/ipv6_route at line {line_number}")
        interface = fields[9]
        if interface not in interfaces:
            raise H3Error(f"/proc/net/ipv6_route references an unavailable interface: {interface}")
        destination = parse_route_number(fields[0], base=16, width=32, field="IPv6 destination", line_number=line_number)
        destination_prefix = parse_route_number(fields[1], base=16, width=2, field="IPv6 destination prefix", line_number=line_number)
        parse_route_number(fields[2], base=16, width=32, field="IPv6 source", line_number=line_number)
        source_prefix = parse_route_number(fields[3], base=16, width=2, field="IPv6 source prefix", line_number=line_number)
        parse_route_number(fields[4], base=16, width=32, field="IPv6 next hop", line_number=line_number)
        parse_route_number(fields[5], base=16, width=8, field="IPv6 metric", line_number=line_number)
        parse_route_number(fields[6], base=16, width=8, field="IPv6 reference count", line_number=line_number)
        parse_route_number(fields[7], base=16, width=8, field="IPv6 use count", line_number=line_number)
        flags = parse_route_number(fields[8], base=16, width=8, field="IPv6 flags", line_number=line_number)
        if destination_prefix > 128 or source_prefix > 128:
            raise H3Error(f"malformed /proc/net/ipv6_route prefix at line {line_number}")
        if destination == 0 and destination_prefix == 0 and not flags & RTF_REJECT and interface != "lo":
            raise H3Error("required CI network namespace has a usable non-loopback IPv6 default route")


def assert_required_network_isolation() -> None:
    """Fail closed unless the process sees Docker's network-none namespace.

    This checks the namespace visible to the runner, not a host-level security
    proof.  The workflow supplies the actual boundary with ``--network none``.
    """

    if os.environ.get("SLLM_H3_NETWORK_DISABLED") != "1":
        raise H3Error("required CI requires SLLM_H3_NETWORK_DISABLED=1")
    try:
        interfaces = sorted(name for _index, name in socket.if_nameindex())
    except OSError as exc:
        raise H3Error(f"cannot inspect network interfaces: {exc}") from exc
    if interfaces != ["lo"]:
        raise H3Error(f"required CI network namespace interfaces are not exactly ['lo']: {interfaces}")
    try:
        ipv4_routes = Path("/proc/net/route").read_text(encoding="ascii")
        ipv6_routes = Path("/proc/net/ipv6_route").read_text(encoding="ascii")
    except OSError as exc:
        raise H3Error(f"cannot inspect network routes: {exc}") from exc
    validate_ipv4_routes(ipv4_routes, interfaces)
    validate_ipv6_routes(ipv6_routes, interfaces)


def execution_environment(args: argparse.Namespace, evidence_mode: str) -> dict[str, Any]:
    if evidence_mode == "required-ci":
        if not args.pinned_container:
            raise H3Error("required CI requires --pinned-container")
        if args.observed_image_reference != PINNED_IMAGE_REFERENCE:
            raise H3Error("required CI observed image reference is not the exact pinned ROCm digest")
        if args.observed_image_config_digest != PINNED_IMAGE_CONFIG_DIGEST:
            raise H3Error("required CI observed image config digest is not the exact pinned ROCm image")
        assert_required_network_isolation()
        return {
            "mode": "required-ci",
            "execution_scope": "official-container",
            "container_image_reference": PINNED_IMAGE_REFERENCE,
            "observed_image_config_digest": PINNED_IMAGE_CONFIG_DIGEST,
            "pinned_container": True,
            "identity_verified": True,
            "network_isolated": True,
        }
    if args.pinned_container or args.observed_image_reference is not None or args.observed_image_config_digest is not None:
        raise H3Error("local development runs must not claim official container image evidence")
    return {
        "mode": "local-development",
        "execution_scope": "local-system",
        "container_image_reference": None,
        "observed_image_config_digest": None,
        "pinned_container": False,
        "identity_verified": False,
        "network_isolated": False,
    }


def report_payload(row: dict[str, Any], target: str, candidate: tuple[str, str], *, run_id: str, run_attempt: int, evidence_mode: str, execution_env: dict[str, Any], started: datetime, finished: datetime, commands: list[list[str]], steps: list[dict[str, Any]], h3_toolchain: dict[str, Any], artifact_info: dict[str, Any], scope: dict[str, Any], diagnostics: list[str]) -> dict[str, Any]:
    selected = len(steps)
    failed = sum(step["state"] != "PASS" for step in steps)
    passed = selected - failed
    output_bytes = sum(step["resource"]["output_bytes"] for step in steps)
    captured_bytes = sum(step["resource"]["captured_output_bytes"] for step in steps)
    rss = max((step["resource"]["max_rss_bytes"] for step in steps), default=0)
    state = "PASS" if not diagnostics and len(steps) == len(commands) and failed == 0 else ("INFRA_ERROR" if any(step["state"] == "INFRA_ERROR" for step in steps) else "FAIL")
    commit, tree = candidate
    host_toolchain = make_host_toolchain(h3_toolchain)
    return {
        "schema_version": "test-result-v1", "result_id": f"h3-{target}.{run_id}.{run_attempt}", "suite_id": f"h3-{target}", "tier": "tier_h3", "state": state, "required": False, "evidence_mode": evidence_mode,
        "run_id": run_id, "run_attempt": run_attempt, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit, "git_tree_oid": tree, "worktree_clean": evidence_mode == "required-ci",
        "matrix_manifest_sha256": row["_matrix_manifest_sha256"], "matrix_row_id": row["row_id"], "tuple_digest": sha256_json({key: value for key, value in row.items() if not key.startswith("_")}),
        "command": commands, "command_sha256": sha256_json(commands), "toolchain": host_toolchain, "toolchain_sha256": sha256_json(host_toolchain),
        "artifact": {"content_sha256": artifact_info["content_sha256"], "manifest_sha256": artifact_info["metadata_sha256"]}, "h3_artifact": artifact_info, "h3_scope": scope,
        "created_at": iso(started), "started_at": iso(started), "finished_at": iso(finished), "duration_seconds": round((finished - started).total_seconds(), 6), "seed": row["seed"],
        "counts": {"collected": selected, "selected": selected, "passed": passed, "failed": failed, "skipped": 0, "deselected": 0},
        "resource": {"wall_time_limit_seconds": 900, "wall_time_breach": False, "max_rss_bytes": rss, "max_rss_limit_bytes": row["resource"]["max_rss_bytes"], "rss_breach": rss > row["resource"]["max_rss_bytes"], "runner_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "fixture_size_bytes": 0, "fixture_size_limit_bytes": 1, "fixture_size_breach": False, "output_bytes": output_bytes, "captured_output_bytes": captured_bytes, "row_output_limit_bytes": row["resource"]["max_output_bytes"], "output_breach": output_bytes > row["resource"]["max_output_bytes"], "address_space_limit_bytes": row["resource"]["max_rss_bytes"], "commands_expected": len(commands), "commands_executed": len(steps), "commands_complete": len(steps) == len(commands), "network_isolated": execution_env["network_isolated"], "network_guard_strategies": ["container-network-none" if execution_env["network_isolated"] else "not-isolated-local-development"]},
        "cases": [{"case_id": step["step_id"], **{key: value for key, value in step.items() if key != "step_id"}} for step in steps], "steps": steps,
        "diagnostic": {"message": "H3 compile/link/artifact identity passed" if state == "PASS" else "H3 compile/link/artifact identity did not pass", "errors": diagnostics, "warnings": ["compile-only evidence; no GPU, model, numerics, performance, or support claim"], "output_dir": row.get("_output_directory", ""), "network_disabled": execution_env["network_isolated"], "model_disabled": True, "gpu_fallback_disabled": True, "network_guard_self_test": execution_env["network_isolated"]},
        "execution_environment": execution_env,
    }


def write_json_sidecar(path: Path, document: dict[str, Any]) -> str:
    data = canonical_bytes(document)
    path.write_bytes(data)
    digest = sha256_bytes(data)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", choices=("h3-gfx1030", "h3-gfx1201"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local-h3"))
    parser.add_argument("--run-attempt", type=int, default=int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    parser.add_argument("--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha", default=os.environ.get("REVIEWED_SHA"))
    parser.add_argument("--tested-sha", "--expected-tested-sha", dest="tested_sha", default=os.environ.get("TESTED_SHA"))
    parser.add_argument("--workflow-sha", "--expected-workflow-sha", dest="workflow_sha", default=os.environ.get("WORKFLOW_SHA"))
    parser.add_argument("--strict-ci", action="store_true")
    parser.add_argument("--allow-dirty-local", action="store_true")
    parser.add_argument("--pinned-container", action="store_true")
    parser.add_argument("--observed-image-reference")
    parser.add_argument("--observed-image-config-digest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = args_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = now()
    target = args.row.removeprefix("h3-")
    diagnostics: list[str] = []
    steps: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    artifact_info = {"target": target, "size_bytes": 0, "content_sha256": ZERO_SHA, "metadata_sha256": ZERO_SHA, "metadata_sidecar_sha256": ZERO_SHA, "artifact_sidecar_sha256": ZERO_SHA}
    scope = {"compile_only": True, "link_verified": False, "gpu_execution": False, "execution_attempted": False, "numerics_verified": False, "model_verified": False, "performance_verified": False, "support_claim": False, "network_used": False, "model_used": False, "cpu_fallback_used": False}
    execution_env = {"mode": "local-development", "execution_scope": "local-system", "container_image_reference": None, "observed_image_config_digest": None, "pinned_container": False, "identity_verified": False, "network_isolated": False}
    h3_toolchain: dict[str, Any] = dict(DEFAULT_H3_TOOLCHAIN)
    try:
        repo = args.repo.resolve()
        if args.run_attempt < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", str(args.run_id)):
            raise H3Error("run identity is invalid")
        if args.row not in ("h3-gfx1030", "h3-gfx1201"):
            raise H3Error("runner accepts exactly one known H3 row")
        commit, tree = git_identity(repo)
        expected = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha}
        if args.strict_ci and not all(expected.values()):
            raise H3Error("strict CI requires all reviewed/tested/workflow SHA values")
        for name, value in expected.items():
            if value is None:
                expected[name] = commit
            elif value != commit or not re.fullmatch(r"[0-9a-f]{40}", value):
                raise H3Error(f"{name} is stale or not the checked-out immutable commit")
        clean = worktree_clean(repo)
        if args.strict_ci and not clean:
            raise H3Error("strict CI rejects a dirty checkout")
        if not args.strict_ci and not clean and not args.allow_dirty_local:
            raise H3Error("dirty local run requires --allow-dirty-local")
        evidence_mode = "required-ci" if args.strict_ci else "local-development"
        execution_env = execution_environment(args, evidence_mode)
        toolchain, matrix = validate_manifests(repo)
        row = next(row for row in matrix["rows"] if row["row_id"] == args.row)
        row = dict(row)
        row["_matrix_manifest_sha256"] = sha256_json(matrix)
        h3_toolchain = inspect_toolchain(toolchain)
        h3_toolchain["paths"] = toolchain["paths"]
        h3_toolchain = make_host_toolchain(h3_toolchain)["h3"]
        root = Path(toolchain["rocm"]["path"])
        tools = {key: Path(value) for key, value in toolchain["paths"].items() if key in {"clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump"}}
        output_root = Path(tempfile.mkdtemp(prefix=row["output"]["root_prefix"].removeprefix("/tmp/"), dir="/tmp"))
        build_dir = output_root / row["output"]["directory_pattern"].replace("{target}", target)
        build_dir.mkdir(parents=True)
        row["_output_directory"] = str(build_dir)
        source_path = repo / row["direct_build"]["source_relative_path"]
        if not source_path.is_file() or source_path.is_symlink():
            raise H3Error("direct compile probe source is missing or is not a regular file")
        commands = render_direct_commands(row, target, build_dir, source_path)
        env = os.environ.copy()
        env.update({"ROCM_PATH": str(root), "HIP_PATH": str(root), "SLLM_H3_NETWORK_DISABLED": env.get("SLLM_H3_NETWORK_DISABLED", "0")})
        start_monotonic = time.monotonic()
        for step_index, command in enumerate(commands):
            remaining = 900 - (time.monotonic() - start_monotonic)
            if remaining <= 0:
                diagnostics.append("H3 row timeout exhausted before all commands completed")
                break
            step_started = now()
            command_path = Path(command[0])
            code, stdout, stderr, elapsed, timed_out, max_rss_bytes = read_command(
                command_path,
                command[1:],
                cwd=repo,
                timeout=remaining,
                env=env,
                address_space_limit_bytes=row["resource"]["max_rss_bytes"],
            )
            output_bytes = len(stdout) + len(stderr)
            rss_breach = max_rss_bytes > row["resource"]["max_rss_bytes"]
            step_state = "PASS" if code == 0 and not timed_out and output_bytes <= row["resource"]["max_output_bytes"] and not rss_breach else ("INFRA_ERROR" if code == 127 else "FAIL")
            detail = (stderr or stdout).decode("utf-8", "replace")[-4000:]
            step_diagnostics: list[str] = []
            if code != 0: step_diagnostics.append(f"command exited {code}")
            if timed_out: step_diagnostics.append("command timed out")
            if output_bytes > row["resource"]["max_output_bytes"]: step_diagnostics.append("command output exceeded the row limit")
            if rss_breach: step_diagnostics.append("command maximum RSS exceeded the row limit")
            if step_diagnostics: diagnostics.append(f"{command[0]}: {'; '.join(step_diagnostics)} {detail}".strip())
            step_finished = now()
            steps.append({"step_id": f"h3-{target}.command-{step_index + 1}", "state": step_state, "started_at": iso(step_started), "finished_at": iso(step_finished), "duration_seconds": round(elapsed, 6), "exit_code": code, "stdout_sha256": sha256_bytes(stdout), "stderr_sha256": sha256_bytes(stderr), "diagnostic": "; ".join(step_diagnostics), "selection_required": True, "count_source": "validator-command", "counts": {"collected": 1, "selected": 1, "passed": 1 if step_state == "PASS" else 0, "failed": 1 if step_state != "PASS" else 0, "skipped": 0, "deselected": 0}, "resource": {"wall_time_limit_seconds": remaining, "timed_out": timed_out, "max_rss_bytes": max_rss_bytes, "max_rss_limit_bytes": row["resource"]["max_rss_bytes"], "rss_breach": rss_breach, "cpu_user_seconds": 0, "cpu_system_seconds": 0, "stdout_bytes": len(stdout), "stderr_bytes": len(stderr), "output_bytes": output_bytes, "stdout_captured_bytes": len(stdout), "stderr_captured_bytes": len(stderr), "captured_output_bytes": output_bytes, "output_limit_bytes": row["resource"]["max_output_bytes"], "output_breach": output_bytes > row["resource"]["max_output_bytes"], "network_isolated": execution_env["network_isolated"], "network_guard_strategy": "container-network-none" if execution_env["network_isolated"] else "not-isolated-local-development", "address_space_limit_bytes": row["resource"]["max_rss_bytes"], "address_space_limit_enforced": True}})
            if step_state != "PASS":
                break
        if len(steps) == len(commands) and all(step["state"] == "PASS" for step in steps):
            host_binary = build_dir / f"hip-compile-probe-{target}.elf"
            host_object = build_dir / f"hip-compile-probe-{target}.o"
            if not host_binary.is_file() or host_binary.is_symlink():
                raise H3Error("direct link command did not produce the expected host ELF")
            if not host_object.is_file() or host_object.is_symlink():
                raise H3Error("direct compile command did not produce the expected bundle-preserving host object")
            bundle_blob = build_dir / f"hip-compile-probe-{target}.fatbin"
            dump_result = subprocess.run([str(tools["llvm_objcopy"]), f"--dump-section=.hip_fatbin={bundle_blob}", str(host_object)], text=True, capture_output=True, check=False)
            if dump_result.returncode != 0 or not bundle_blob.is_file() or bundle_blob.is_symlink():
                raise H3Error(f"cannot extract .hip_fatbin from the host object: {dump_result.stderr.strip()}")
            list_result = subprocess.run([str(tools["clang_offload_bundler"]), "--list", "--type=o", f"--input={bundle_blob}"], text=True, capture_output=True, check=False)
            if list_result.returncode != 0:
                raise H3Error(f"clang-offload-bundler list failed: {list_result.stderr.strip()}")
            bundles = [line.strip() for line in list_result.stdout.splitlines() if line.strip()]
            expected_bundles = [BUNDLE_IDS[target], "host-x86_64-unknown-linux-gnu-"]
            if bundles != expected_bundles:
                raise H3Error(f"host object bundle list is not the exact device/host order: {bundles}")
            artifact_path = build_dir / row["output"]["artifact_pattern"].replace("{target}", target)
            unbundle = subprocess.run([str(tools["clang_offload_bundler"]), "--unbundle", "--type=o", f"--targets={BUNDLE_IDS[target]}", f"--input={bundle_blob}", f"--output={artifact_path}"], text=True, capture_output=True, check=False)
            if unbundle.returncode != 0 or not artifact_path.is_file() or artifact_path.is_symlink():
                raise H3Error(f"device code object extraction failed: {unbundle.stderr.strip()}")
            host_meta, _ = inspect_elf(host_object, target, {"llvm_readobj": tools["llvm_readobj"]})
            device_meta, _ = inspect_elf(artifact_path, target, {"llvm_readobj": tools["llvm_readobj"]})
            host_meta["bundles"] = [
                {"id": BUNDLE_IDS[target], "target": target},
                {"id": "host-x86_64-unknown-linux-gnu-", "target": "host"},
            ]
            metadata = {"schema_version": "hip-artifact-metadata-v1", "metadata_id": f"h3-artifact-{target}", "matrix_row_id": row["row_id"], "target": target, "candidate": {"commit_sha": commit, "tree_oid": tree, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit}, "toolchain_id": toolchain["toolchain_id"], "matrix_id": matrix["matrix_id"], "toolchain_manifest_sha256": sha256_json(toolchain), "matrix_manifest_sha256": sha256_json(matrix), "image": {key: toolchain["image"][key] for key in ("repository", "tag", "manifest_digest", "config_digest", "manifest_list_digest", "manifest_type", "platform")}, "resolved_paths": toolchain["paths"], "build": {"source_directory": str(repo), "source_path": str(source_path), "output_directory": str(build_dir), "object_path": str(host_object), "link_output_path": str(host_binary), "generator": "direct-amdclang++", "mode": "direct-compile-link", "build_type": "Release", "language_standard": "gnu++17", "output_directory_scope": "row-private", "source_tree_output": False, "shared_build_directory": False}, "codegen": row["codegen"], "artifact": {"path": str(artifact_path), "size_bytes": artifact_path.stat().st_size, "sha256": sha256_file(artifact_path)}, "host_bundle": host_meta, "device_code_object": device_meta, "scope": scope | {"link_verified": True}, "execution_environment": execution_env, "timestamps": {"created_at": iso(started), "started_at": iso(started), "finished_at": iso(now())}, "duration_seconds": round(time.monotonic() - start_monotonic, 6)}
            metadata_path = output_dir / "hip-artifact-metadata.json"
            metadata_sha = write_json_sidecar(metadata_path, metadata)
            metadata_sidecar_sha = sha256_file(metadata_path.with_name(metadata_path.name + ".sha256"))
            artifact_sidecar = output_dir / row["output"]["artifact_pattern"].replace("{target}", target)
            shutil.copy2(artifact_path, artifact_sidecar)
            artifact_sidecar_hash_path = artifact_sidecar.with_name(artifact_sidecar.name + ".sha256")
            artifact_sidecar_sha = sha256_file(artifact_sidecar)
            artifact_sidecar_hash_path.write_text(f"{artifact_sidecar_sha}  {artifact_sidecar.name}\n", encoding="utf-8")
            artifact_sidecar_sha = sha256_file(artifact_sidecar_hash_path)
            artifact_info = {"target": target, "size_bytes": artifact_sidecar.stat().st_size, "content_sha256": sha256_file(artifact_sidecar), "metadata_sha256": metadata_sha, "metadata_sidecar_sha256": metadata_sidecar_sha, "artifact_sidecar_sha256": artifact_sidecar_sha}
            scope = scope | {"link_verified": True}
        else:
            diagnostics.append("compile/link command result is incomplete or non-success")
    except (H3Error, OSError, StopIteration, ValueError, KeyError) as exc:
        diagnostics.append(str(exc))
    finished = now()
    try:
        commit, tree = git_identity(args.repo.resolve())
    except H3Error:
        commit, tree = "0" * 40, "0" * 40
    row_for_report = locals().get("row", {"row_id": args.row, "seed": 0, "resource": {"max_rss_bytes": 4294967296, "max_output_bytes": 16777216}, "_matrix_manifest_sha256": ZERO_SHA, "_output_directory": str(output_dir)})
    payload = report_payload(row_for_report, target, (commit, tree), run_id=str(args.run_id), run_attempt=args.run_attempt, evidence_mode=locals().get("evidence_mode", "local-development"), execution_env=execution_env, started=started, finished=finished, commands=commands or [["missing-h3-command"]], steps=steps, h3_toolchain=h3_toolchain, artifact_info=artifact_info, scope=scope, diagnostics=diagnostics)
    write_json_sidecar(output_dir / "report.json", payload)
    state = payload["state"]
    print(f"{args.row}: {state} output={output_dir}")
    return 0 if state == "PASS" else (2 if state == "INFRA_ERROR" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
