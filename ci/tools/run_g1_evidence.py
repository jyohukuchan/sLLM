#!/usr/bin/env python3
"""Run one fail-closed trusted-local G1 runtime evidence row.

The runner deliberately keeps the trust boundary small.  Git identity, GPU
routing, health, and process observations are provided by the existing G0
read-only path.  The only process started for evidence is the staged,
dedicated Rust evidence binary named by the G1 artifact metadata.
"""

from __future__ import annotations

import argparse
import copy
import errno
import fcntl
import json
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ContractError,
    ROOT,
    canonical_bytes,
    exact_sha,
    read_json,
    sha256_bytes,
    sha256_file,
    sha256_json,
)
from run_g0_preflight import (  # noqa: E402
    amd_smi_list_json,
    git_candidate,
    nonblocking_host_lock,
    observe_health,
    observe_processes,
    require_available_observation,
)
from validate_g0_contracts import (  # noqa: E402
    AMD_SMI_EXECUTABLE,
    path_outside_repo,
    reject_inherited_visibility_selectors,
    validate_health,
    validate_processes,
    validate_routing,
    validate_visibility_environment,
)
from validate_g1_contracts import (  # noqa: E402
    BINARY_NAME,
    EXPECTED_SIZES,
    METADATA_NAME,
    RUN_ID_TOKEN,
    validate_artifact_metadata,
    validate_g1_matrix,
    validate_report,
    validate_schema,
    row_by_id,
    _manifest_hashes,
)


ZERO_SHA = "0" * 64
ZERO_SHA40 = "0" * 40
REPORT_NAME = "report.json"
COMMAND = ["target/release/sllm-hip-evidence", "--timeout-ms", "1000"]
MAX_CAPTURED_OUTPUT = 1024 * 1024
CAPTURE_CHUNK_BYTES = 64 * 1024
TERM_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
REAP_GRACE_SECONDS = 2.0
METADATA_READ_LIMIT = 16 * 1024 * 1024
SYSFS_PCI_ROOT = Path("/sys/bus/pci/devices")
PRIVATE_TMP_PREFIX = "sllm-g1-"
ROCM_ROOT = "/opt/rocm"
ROCM_RELEASE = "7.14.0"
PINNED_PATH = "/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PINNED_LD_LIBRARY_PATH = "/opt/rocm/lib:/opt/rocm/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:/lib:/usr/lib"
REQUIRED_LOADER_LIBRARIES = ("libamdhip64.so.7", "libhsa-runtime64.so.1")
FORBIDDEN_LOADER_ENVIRONMENT = {
    "LD_PRELOAD",
    "LD_AUDIT",
    "ROCR_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES_MASK",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
}
LIBRARY_NAME_PATTERN = {
    name: re.compile(rf"^{re.escape(name)}(?:\.[0-9][A-Za-z0-9._-]*)?$")
    for name in REQUIRED_LOADER_LIBRARIES
}
ALLOWED_STATES = {
    "PASS",
    "FAIL",
    "UNAVAILABLE",
    "TIMEOUT",
    "CRASH",
    "SKIP",
    "QUARANTINED",
    "MISSING",
    "INFRA_ERROR",
}


def now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _path_has_symlink_component(path: Path, label: str) -> None:
    """Reject a path whose existing directory or file component is a symlink."""

    if not path.is_absolute() or "\x00" in str(path):
        raise ContractError(f"{label} must be an absolute path without NUL")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ContractError(f"{label} contains a symlink component")


def _private_tmp_root(path: Path, label: str) -> Path:
    _path_has_symlink_component(path, label)
    resolved = path.resolve(strict=False)
    try:
        relative = resolved.relative_to(Path("/tmp"))
    except ValueError as exc:
        raise ContractError(f"{label} must be staged below /tmp") from exc
    if len(relative.parts) < 1 or not relative.parts[0].startswith(PRIVATE_TMP_PREFIX):
        raise ContractError(f"{label} must be below a private {PRIVATE_TMP_PREFIX}* directory")
    root = Path("/tmp") / relative.parts[0]
    if not root.is_dir() or root.is_symlink():
        raise ContractError(f"{label} private staging root is missing or unsafe")
    if stat.S_IMODE(root.stat().st_mode) & 0o077:
        raise ContractError(f"{label} private staging root is accessible by group or other users")
    return root


def _safe_repo(repo: Path) -> Path:
    if not repo.is_absolute():
        raise ContractError("G1 repository path must be absolute")
    _path_has_symlink_component(repo, "G1 repository")
    if not repo.is_dir() or repo.is_symlink():
        raise ContractError("G1 repository path is missing, symlinked, or not a directory")
    resolved = repo.resolve(strict=True)
    if resolved != repo:
        raise ContractError("G1 repository path is not canonical")
    return resolved


@dataclass(frozen=True)
class FileSnapshot:
    """Identity and content observed through one no-follow regular fd."""

    identity: tuple[int, int, int, int, int, int, int]
    size_bytes: int
    sha256: str


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _open_nofollow_directory(path: Path, label: str) -> int:
    """Walk an absolute path using directory fds without following symlinks."""

    if not path.is_absolute() or "\x00" in str(path):
        raise ContractError(f"{label} must be an absolute path without NUL")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        current = os.open(path.anchor, flags)
        for component in path.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current)
            os.close(current)
            current = next_fd
        return current
    except OSError as exc:
        try:
            os.close(current)
        except (UnboundLocalError, OSError):
            pass
        raise ContractError(f"{label} directory is missing or contains a symlink") from exc


def _lstat_nofollow(path: Path, label: str) -> os.stat_result:
    parent_fd = _open_nofollow_directory(path.parent, label)
    try:
        try:
            value = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(f"{label} is missing or unsafe") from exc
    finally:
        os.close(parent_fd)
    if not stat.S_ISREG(value.st_mode):
        raise ContractError(f"{label} is not a regular file")
    return value


def _open_regular_nofollow(path: Path, label: str, *, writable: bool = False) -> tuple[int, os.stat_result]:
    parent_fd = _open_nofollow_directory(path.parent, label)
    flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        try:
            fd = os.open(path.name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ContractError(f"{label} is missing, symlinked, or inaccessible") from exc
    finally:
        os.close(parent_fd)
    try:
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode):
            raise ContractError(f"{label} is not a regular file")
        return fd, value
    except BaseException:
        os.close(fd)
        raise


def _hash_fd(fd: int, *, limit: int | None = None) -> tuple[str, bytes | None]:
    import hashlib

    digest = hashlib.sha256()
    captured = bytearray() if limit is not None else None
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, CAPTURE_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        if captured is not None:
            if len(captured) + len(chunk) > limit:
                raise ContractError("regular file exceeds the bounded metadata read limit")
            captured.extend(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return digest.hexdigest(), bytes(captured) if captured is not None else None


def _snapshot_open_fd(fd: int, before: os.stat_result, path: Path, label: str) -> FileSnapshot:
    before_path = _lstat_nofollow(path, label)
    if _stat_identity(before_path) != _stat_identity(before):
        raise ContractError(f"{label} was replaced before hashing")
    digest, _ = _hash_fd(fd)
    after = os.fstat(fd)
    after_path = _lstat_nofollow(path, label)
    if _stat_identity(after) != _stat_identity(before) or _stat_identity(after_path) != _stat_identity(after):
        raise ContractError(f"{label} changed while being hashed")
    return FileSnapshot(_stat_identity(after), after.st_size, digest)


def _snapshot_regular(path: Path, label: str) -> FileSnapshot:
    fd, before = _open_regular_nofollow(path, label)
    try:
        return _snapshot_open_fd(fd, before, path, label)
    finally:
        os.close(fd)


def _read_json_nofollow(path: Path, label: str) -> tuple[Any, FileSnapshot]:
    fd, before = _open_regular_nofollow(path, label)
    try:
        snapshot_before = _snapshot_open_fd(fd, before, path, label)
        os.lseek(fd, 0, os.SEEK_SET)
        body = bytearray()
        while True:
            chunk = os.read(fd, CAPTURE_CHUNK_BYTES)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > METADATA_READ_LIMIT:
                raise ContractError(f"{label} exceeds the bounded metadata read limit")
        after = os.fstat(fd)
        after_path = _lstat_nofollow(path, label)
        if (_stat_identity(after) != snapshot_before.identity or
                _stat_identity(after_path) != snapshot_before.identity):
            raise ContractError(f"{label} changed while being read")
    finally:
        os.close(fd)
    try:
        return json.loads(bytes(body).decode("utf-8")), FileSnapshot(
            snapshot_before.identity, snapshot_before.size_bytes, snapshot_before.sha256
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label} is not valid UTF-8 JSON") from exc


def _assert_snapshot(path: Path, expected: FileSnapshot, label: str) -> None:
    actual = _snapshot_regular(path, label)
    if actual != expected:
        raise ContractError(f"{label} identity or hash changed during validation")


def _open_output_exclusive(path: Path, label: str, mode: int) -> tuple[int, int]:
    parent_fd = _open_nofollow_directory(path.parent, label)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        try:
            fd = os.open(path.name, flags, mode, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise ContractError(f"{label} already exists") from exc
        except OSError as exc:
            raise ContractError(f"{label} could not be created exclusively") from exc
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode):
            os.close(fd)
            raise ContractError(f"{label} output is not a regular file")
        return fd, parent_fd
    except BaseException:
        os.close(parent_fd)
        raise


def _write_exclusive(path: Path, data: bytes, label: str, *, mode: int = 0o600) -> FileSnapshot:
    fd, parent_fd = _open_output_exclusive(path, label, mode)
    created = True
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
        value = os.fstat(fd)
        if not stat.S_ISREG(value.st_mode) or value.st_size != len(data):
            raise ContractError(f"{label} output changed while being written")
        snapshot = FileSnapshot(_stat_identity(value), value.st_size, sha256_bytes(data))
        if _stat_identity(_lstat_nofollow(path, label)) != snapshot.identity:
            raise ContractError(f"{label} output was replaced while being written")
        os.fsync(parent_fd)
        created = False
        return snapshot
    finally:
        os.close(fd)
        os.close(parent_fd)
        if created:
            try:
                cleanup_parent = _open_nofollow_directory(path.parent, label)
                try:
                    os.unlink(path.name, dir_fd=cleanup_parent)
                finally:
                    os.close(cleanup_parent)
            except OSError:
                pass


def _copy_regular(
    source: Path,
    destination: Path,
    label: str,
    *,
    expected: FileSnapshot | None = None,
) -> FileSnapshot:
    source_fd, source_before = _open_regular_nofollow(source, label)
    destination_fd: int | None = None
    destination_parent: int | None = None
    created = False
    try:
        source_snapshot = _snapshot_open_fd(source_fd, source_before, source, label)
        if expected is not None and source_snapshot != expected:
            raise ContractError(f"{label} changed before copying")
        destination_fd, destination_parent = _open_output_exclusive(destination, label, 0o700)
        created = True
        os.lseek(source_fd, 0, os.SEEK_SET)
        import hashlib
        copied_digest = hashlib.sha256()
        copied_size = 0
        while True:
            chunk = os.read(source_fd, CAPTURE_CHUNK_BYTES)
            if not chunk:
                break
            copied_digest.update(chunk)
            copied_size += len(chunk)
            offset = 0
            while offset < len(chunk):
                offset += os.write(destination_fd, chunk[offset:])
        os.fsync(destination_fd)
        source_after = os.fstat(source_fd)
        destination_after = os.fstat(destination_fd)
        source_path_after = _lstat_nofollow(source, label)
        if (_stat_identity(source_after) != source_snapshot.identity or
                _stat_identity(source_path_after) != source_snapshot.identity or
                copied_size != source_snapshot.size_bytes or
                copied_digest.hexdigest() != source_snapshot.sha256):
            raise ContractError(f"{label} was replaced or modified during copy")
        if not stat.S_ISREG(destination_after.st_mode) or destination_after.st_size != copied_size:
            raise ContractError(f"{label} destination is not a complete regular file")
        destination_snapshot = FileSnapshot(
            _stat_identity(destination_after), destination_after.st_size, copied_digest.hexdigest()
        )
        if _stat_identity(_lstat_nofollow(destination, label)) != destination_snapshot.identity:
            raise ContractError(f"{label} destination was replaced during copy")
        os.fsync(destination_parent)
        created = False
        return destination_snapshot
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if destination_parent is not None:
            os.close(destination_parent)
        os.close(source_fd)
        if created:
            try:
                cleanup_parent = _open_nofollow_directory(destination.parent, label)
                try:
                    os.unlink(destination.name, dir_fd=cleanup_parent)
                finally:
                    os.close(cleanup_parent)
            except OSError:
                pass


def safe_output_directory(output: Path, repo: Path, row: Mapping[str, Any]) -> Path:
    """Create a new private row directory and reject stale/unsafe output."""

    if not output.is_absolute():
        raise ContractError("G1 output directory must be absolute")
    _path_has_symlink_component(output, "G1 output directory")
    root = _private_tmp_root(output, "G1 output directory")
    resolved = output.resolve(strict=False)
    path_outside_repo(resolved, repo, "G1 output directory")
    if resolved.name != row["row_id"] or resolved.parent != root:
        raise ContractError("G1 output must be /tmp/sllm-g1-*/g1-<exact-row>")
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ContractError("G1 output directory is unsafe")
        if any(output.iterdir()):
            raise ContractError("G1 output directory must be new and empty")
    else:
        output.mkdir(mode=0o700)
    if output.resolve() != resolved:
        raise ContractError("G1 output directory changed during creation")
    return resolved


def _safe_staged_metadata(path: Path, row: Mapping[str, Any], repo: Path) -> Path:
    if not path.is_absolute():
        raise ContractError("G1 runtime metadata path must be absolute")
    _path_has_symlink_component(path, "G1 runtime metadata")
    root = _private_tmp_root(path, "G1 runtime metadata")
    resolved = path.resolve(strict=True)
    path_outside_repo(resolved, repo, "G1 runtime metadata")
    if resolved.name != METADATA_NAME or resolved.parent.name != row["row_id"]:
        raise ContractError("G1 runtime metadata is not staged in the exact row directory")
    if resolved.parent.parent != root:
        raise ContractError("G1 runtime metadata is not directly below its private staging root")
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError("G1 runtime metadata is not a regular non-symlink file")
    return resolved


def _safe_staged_artifact(
    declared_path: Any, metadata_path: Path, repo: Path
) -> Path:
    if not isinstance(declared_path, str) or not declared_path:
        raise ContractError("G1 runtime artifact path is missing")
    artifact = Path(declared_path)
    if not artifact.is_absolute():
        raise ContractError("G1 runtime artifact path must be absolute")
    _path_has_symlink_component(artifact, "G1 runtime artifact")
    resolved = artifact.resolve(strict=True)
    path_outside_repo(resolved, repo, "G1 runtime artifact")
    stage_root = metadata_path.parent.parent
    try:
        resolved.relative_to(stage_root)
    except ValueError as exc:
        raise ContractError("G1 runtime artifact is outside the private metadata staging root") from exc
    if resolved.name != BINARY_NAME or resolved.parts[-3:] != ("target", "release", BINARY_NAME):
        raise ContractError("G1 runtime artifact is not target/release/sllm-hip-evidence")
    if not resolved.is_file() or resolved.is_symlink():
        raise ContractError("G1 runtime artifact is not a regular non-symlink file")
    if not os.access(resolved, os.X_OK):
        raise ContractError("G1 runtime artifact is not executable")
    if resolved != artifact:
        raise ContractError("G1 runtime artifact path is not canonical")
    return resolved


def _write_sidecar(path: Path) -> str:
    digest = _snapshot_regular(path, f"{path.name} sidecar target").sha256
    sidecar = path.with_name(path.name + ".sha256")
    snapshot = _write_exclusive(
        sidecar,
        f"{digest}  {path.name}\n".encode("ascii"),
        f"sidecar for {path.name}",
        mode=0o600,
    )
    return snapshot.sha256


def _copy_staged_artifacts(
    output: Path,
    metadata_path: Path,
    artifact_path: Path,
    *,
    metadata_snapshot: FileSnapshot | None = None,
    artifact_snapshot: FileSnapshot | None = None,
) -> tuple[Path, Path]:
    output_metadata = output / METADATA_NAME
    output_artifact = output / BINARY_NAME
    _copy_regular(
        metadata_path,
        output_metadata,
        "G1 runtime metadata",
        expected=metadata_snapshot,
    )
    _copy_regular(
        artifact_path,
        output_artifact,
        "G1 runtime artifact",
        expected=artifact_snapshot,
    )
    _write_sidecar(output_metadata)
    _write_sidecar(output_artifact)
    return output_metadata, output_artifact


@dataclass(frozen=True)
class EvidenceExecution:
    state: str
    exit_code: int | None
    timed_out: bool
    crashed: bool
    stdout: bytes
    stderr: bytes
    duration_seconds: float
    payload: dict[str, Any] | None
    error: str | None
    runtime_binding: dict[str, Any] | None = None
    artifact_sha256: str | None = None
    cleanup_proven: bool = True


def _read_process_maps(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/maps").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def _loader_path_candidates(maps_text: str) -> dict[str, set[str]]:
    candidates = {name: set() for name in REQUIRED_LOADER_LIBRARIES}
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        mapped = fields[5]
        if not mapped.startswith("/") or mapped.endswith(" (deleted)"):
            continue
        mapped_path = os.path.realpath(mapped)
        basename = Path(mapped_path).name
        for name, pattern in LIBRARY_NAME_PATTERN.items():
            if pattern.fullmatch(basename):
                candidates[name].add(mapped_path)
    return candidates


def _observe_loader_paths(pid: int, observed: dict[str, set[str]]) -> None:
    candidates = _loader_path_candidates(_read_process_maps(pid))
    for name in REQUIRED_LOADER_LIBRARIES:
        observed[name].update(candidates[name])
        if len(observed[name]) > 1:
            raise ContractError(f"loader binding has duplicate paths for {name}")


def _read_rocm_release() -> str:
    """Read the release from the installed ROCm tree, independently of metadata."""

    candidates = (
        Path(ROCM_ROOT) / "core-7.14" / ".info" / "version",
        Path(ROCM_ROOT) / ".info" / "version",
    )
    for path in candidates:
        try:
            fd, before = _open_regular_nofollow(path, "ROCm release file")
        except ContractError:
            continue
        try:
            snapshot = _snapshot_open_fd(fd, before, path, "ROCm release file")
            os.lseek(fd, 0, os.SEEK_SET)
            body = bytearray()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > 128:
                    raise ContractError("ROCm release file is unexpectedly large")
            after = os.fstat(fd)
            if _stat_identity(after) != snapshot.identity:
                raise ContractError("ROCm release file changed while being read")
            return bytes(body).decode("ascii").strip()
        except (UnicodeDecodeError, OSError) as exc:
            raise ContractError("ROCm release file is not stable ASCII") from exc
        finally:
            os.close(fd)
    raise ContractError("ROCm 7.14.0 release file is missing")


def _runtime_binding(
    observed: Mapping[str, set[str]], environment: Mapping[str, str]
) -> dict[str, Any]:
    if set(environment) & FORBIDDEN_LOADER_ENVIRONMENT:
        raise ContractError("runtime loader environment contains an inherited selector")
    if environment.get("PATH") != PINNED_PATH or environment.get("LD_LIBRARY_PATH") != PINNED_LD_LIBRARY_PATH:
        raise ContractError("runtime loader environment is not the pinned minimal environment")
    loaded: dict[str, str] = {}
    for name in REQUIRED_LOADER_LIBRARIES:
        paths = observed.get(name, set())
        if len(paths) != 1:
            raise ContractError(f"runtime loader did not prove exactly one path for {name}")
        path = next(iter(paths))
        if not path.startswith(ROCM_ROOT + "/"):
            raise ContractError(f"runtime loader path for {name} is outside /opt/rocm")
        if not Path(path).is_absolute() or not LIBRARY_NAME_PATTERN[name].fullmatch(Path(path).name):
            raise ContractError(f"runtime loader path for {name} has the wrong soname")
        loaded[name] = path
    release = _read_rocm_release()
    if release != ROCM_RELEASE:
        raise ContractError(f"installed ROCm release is not {ROCM_RELEASE}")
    binding = {
        "rocm_root": ROCM_ROOT,
        "rocm_release": release,
        "path": PINNED_PATH,
        "ld_library_path": PINNED_LD_LIBRARY_PATH,
        "observation_method": "proc-pid-maps-poll-v1",
        "required_libraries": list(REQUIRED_LOADER_LIBRARIES),
        "loaded_libraries": loaded,
        "inherited_loader_environment": False,
    }
    _validate_runtime_binding(binding)
    return binding


def _validate_runtime_binding(binding: Any) -> None:
    if not isinstance(binding, dict) or set(binding) != {
        "rocm_root", "rocm_release", "path", "ld_library_path",
        "observation_method", "required_libraries", "loaded_libraries",
        "inherited_loader_environment",
    }:
        raise ContractError("runtime_binding has unknown or missing keys")
    if binding["rocm_root"] != ROCM_ROOT or binding["rocm_release"] != ROCM_RELEASE:
        raise ContractError("runtime_binding ROCm root or release is not canonical")
    if binding["path"] != PINNED_PATH or binding["ld_library_path"] != PINNED_LD_LIBRARY_PATH:
        raise ContractError("runtime_binding loader search path is not pinned")
    if binding["observation_method"] != "proc-pid-maps-poll-v1" or binding["inherited_loader_environment"] is not False:
        raise ContractError("runtime_binding observation or environment contract is invalid")
    if binding["required_libraries"] != list(REQUIRED_LOADER_LIBRARIES):
        raise ContractError("runtime_binding required library list drifted")
    loaded = binding["loaded_libraries"]
    if not isinstance(loaded, dict) or set(loaded) != set(REQUIRED_LOADER_LIBRARIES):
        raise ContractError("runtime_binding loaded library set is incomplete or over-broad")
    for name in REQUIRED_LOADER_LIBRARIES:
        path = loaded[name]
        if not isinstance(path, str) or not path.startswith(ROCM_ROOT + "/"):
            raise ContractError(f"runtime_binding path for {name} is not an absolute ROCm path")
        if not LIBRARY_NAME_PATTERN[name].fullmatch(Path(path).name):
            raise ContractError(f"runtime_binding path for {name} has the wrong soname")


def _process_group_gone(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return True
        return False
    proc_root = Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return False
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat_text = (entry / "stat").read_text(encoding="ascii")
            tail = stat_text.rsplit(")", 1)[1].split()
            if len(tail) >= 3 and int(tail[2]) == pgid:
                return False
        except (FileNotFoundError, PermissionError, OSError, ValueError, IndexError):
            continue
    return True


def _signal_group(pgid: int, process: subprocess.Popen[bytes], signum: signal.Signals) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return
    except OSError:
        try:
            if process.poll() is None:
                process.send_signal(signum)
        except OSError:
            pass


def _drain_capture(
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    streams: dict[int, tuple[str, Any]],
) -> bool:
    overflow = False
    for key, _mask in selector.select(timeout=0):
        fd = key.fd
        name, stream = streams[fd]
        try:
            chunk = os.read(fd, CAPTURE_CHUNK_BYTES)
        except BlockingIOError:
            continue
        except OSError:
            selector.unregister(fd)
            streams.pop(fd, None)
            stream.close()
            continue
        if not chunk:
            selector.unregister(fd)
            streams.pop(fd, None)
            stream.close()
            continue
        available = MAX_CAPTURED_OUTPUT - len(buffers[name])
        if len(chunk) > available:
            buffers[name].extend(chunk[:max(0, available)])
            overflow = True
        else:
            buffers[name].extend(chunk)
    return overflow


def _cleanup_process_group(
    process: subprocess.Popen[bytes],
    pgid: int,
    selector: selectors.BaseSelector,
    buffers: dict[str, bytearray],
    streams: dict[int, tuple[str, Any]],
) -> bool:
    def phase(signum: signal.Signals, duration: float) -> bool:
        _signal_group(pgid, process, signum)
        end = time.monotonic() + duration
        while time.monotonic() < end:
            _drain_capture(selector, buffers, streams)
            if process.poll() is not None and _process_group_gone(pgid) and not streams:
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    return False
                return True
            wait_for = min(0.05, max(0.0, end - time.monotonic()))
            if wait_for:
                selector.select(timeout=wait_for)
        _drain_capture(selector, buffers, streams)
        return process.poll() is not None and _process_group_gone(pgid) and not streams

    if phase(signal.SIGTERM, TERM_GRACE_SECONDS):
        return True
    if phase(signal.SIGKILL, KILL_GRACE_SECONDS):
        return True
    end = time.monotonic() + REAP_GRACE_SECONDS
    while time.monotonic() < end:
        _drain_capture(selector, buffers, streams)
        if process.poll() is not None and _process_group_gone(pgid) and not streams:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                return False
            return True
        selector.select(timeout=min(0.05, max(0.0, end - time.monotonic())))
    _drain_capture(selector, buffers, streams)
    return process.poll() is not None and _process_group_gone(pgid) and not streams


def _bounded_reap_without_capture(process: subprocess.Popen[bytes], pgid: int | None) -> bool:
    """Last-resort bounded reap for setup failures before selector registration."""

    if pgid is not None:
        _signal_group(pgid, process, signal.SIGTERM)
    else:
        try:
            process.terminate()
        except OSError:
            pass
    try:
        process.wait(timeout=TERM_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if pgid is not None:
            _signal_group(pgid, process, signal.SIGKILL)
        else:
            try:
                process.kill()
            except OSError:
                pass
        try:
            process.wait(timeout=KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            return False
    if process.poll() is None or pgid is None:
        return False
    return _process_group_gone(pgid)


def _valid_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_evidence_pass_payload(payload: Any) -> tuple[int, int, int, int]:
    if not isinstance(payload, dict):
        raise ContractError("dedicated G1 binary returned a non-object JSON value")
    expected_keys = {
        "schema_version", "state", "selected_backend", "fallback_used", "case_count",
        "allocation_count", "copy_count", "kernel_dispatch_count", "dispatch_count", "cases",
    }
    if set(payload) != expected_keys:
        raise ContractError("dedicated G1 binary returned an incomplete or over-broad diagnostic payload")
    if payload.get("schema_version") != "g1-report-v1":
        raise ContractError("dedicated G1 binary returned the wrong schema version")
    if payload.get("state") != "PASS":
        raise ContractError("dedicated G1 binary did not report PASS")
    if payload.get("selected_backend") != "hip" or payload.get("fallback_used") is not False:
        raise ContractError("dedicated G1 binary selected a fallback or non-HIP backend")
    if payload.get("case_count") != len(EXPECTED_SIZES):
        raise ContractError("dedicated G1 binary did not report exactly six cases")
    for field, expected in (
        ("allocation_count", 12), ("copy_count", 12),
        ("kernel_dispatch_count", 6), ("dispatch_count", 6),
    ):
        if payload.get(field) != expected:
            raise ContractError(f"dedicated G1 binary reported an incorrect {field}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or [case.get("size") for case in cases if isinstance(case, dict)] != list(EXPECTED_SIZES):
        raise ContractError("dedicated G1 binary did not return the exact ordered case sizes")
    allocations = copies = dispatches = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "size", "state", "byte_exact", "dispatch_count", "allocation_count",
            "copy_count", "timed_out", "fallback_used",
        }:
            raise ContractError("dedicated G1 binary case record is malformed")
        if case["state"] != "PASS" or case["byte_exact"] is not True:
            raise ContractError("dedicated G1 binary did not verify the diagnostic bytes exactly")
        if case["dispatch_count"] != 1 or case["timed_out"] is not False or case["fallback_used"] is not False:
            raise ContractError("dedicated G1 binary case must have exactly one dispatch")
        if case["allocation_count"] != 2 or case["copy_count"] != 2:
            raise ContractError("dedicated G1 binary case resource counts are not exact")
        allocations += case["allocation_count"]
        copies += case["copy_count"]
        dispatches += case["dispatch_count"]
    if (allocations, copies, dispatches) != (12, 12, 6):
        raise ContractError("dedicated G1 binary dispatch totals are inconsistent")
    return allocations, copies, dispatches, payload["dispatch_count"]


def _decode_payload(stdout: bytes) -> Any:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ContractError("dedicated G1 binary stdout is not UTF-8") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractError("dedicated G1 binary returned malformed JSON") from exc


def run_evidence_binary(
    artifact_path: Path,
    *,
    timeout_seconds: int,
    hip_visible_devices: str,
    expected_artifact_sha256: str | None = None,
) -> EvidenceExecution:
    """Execute the copied artifact with bounded capture and fail-closed cleanup."""

    argv = [str(artifact_path), *COMMAND[1:]]
    environment = {
        "HIP_VISIBLE_DEVICES": hip_visible_devices,
        "PATH": PINNED_PATH,
        "LD_LIBRARY_PATH": PINNED_LD_LIBRARY_PATH,
    }
    if set(environment) & FORBIDDEN_LOADER_ENVIRONMENT:
        return EvidenceExecution(
            "INFRA_ERROR", None, False, False, b"", b"", 0.0, None,
            "minimal runtime environment contains a forbidden inherited selector",
            cleanup_proven=False,
        )
    started = now()
    process: subprocess.Popen[bytes] | None = None
    selector: selectors.BaseSelector | None = None
    streams: dict[int, tuple[str, Any]] = {}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    artifact_fd: int | None = None
    artifact_snapshot: FileSnapshot | None = None
    pgid: int | None = None
    timed_out = False
    cleanup_required = False
    cleanup_reason: str | None = None
    output_overflow = False
    loader_error: str | None = None
    observed_loader_paths = {name: set() for name in REQUIRED_LOADER_LIBRARIES}
    cleanup_proven = True
    try:
        artifact_fd, opened = _open_regular_nofollow(artifact_path, "G1 executable artifact")
        artifact_snapshot = _snapshot_open_fd(artifact_fd, opened, artifact_path, "G1 executable artifact")
        if expected_artifact_sha256 is not None and artifact_snapshot.sha256 != expected_artifact_sha256:
            raise ContractError("G1 executable artifact hash changed before execution")
        if not os.access(f"/proc/self/fd/{artifact_fd}", os.X_OK):
            raise ContractError("G1 executable artifact is not executable")
        process = subprocess.Popen(
            argv,
            executable=f"/proc/self/fd/{artifact_fd}",
            cwd=Path("/"),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            pass_fds=(artifact_fd,),
        )
        os.close(artifact_fd)
        artifact_fd = None
        pgid = os.getpgid(process.pid)
        selector = selectors.DefaultSelector()
        if process.stdout is None or process.stderr is None:
            raise ContractError("dedicated G1 binary pipes were not created")
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            fd = stream.fileno()
            os.set_blocking(fd, False)
            streams[fd] = (name, stream)
            selector.register(fd, selectors.EVENT_READ)

        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        post_exit_deadline: float | None = None
        finished_normally = False
        while True:
            try:
                _observe_loader_paths(process.pid, observed_loader_paths)
            except ContractError as exc:
                loader_error = str(exc)
            if _drain_capture(selector, buffers, streams):
                output_overflow = True
            if output_overflow:
                cleanup_required = True
                cleanup_reason = "dedicated G1 binary exceeded the captured output limit"
                break
            if loader_error is not None:
                cleanup_required = True
                cleanup_reason = loader_error
                break
            if process.poll() is not None:
                if _process_group_gone(pgid):
                    if not streams:
                        try:
                            process.wait(timeout=0)
                        except subprocess.TimeoutExpired:
                            cleanup_required = True
                            cleanup_reason = "dedicated G1 binary could not be reaped"
                        else:
                            finished_normally = True
                        break
                    if post_exit_deadline is None:
                        post_exit_deadline = time.monotonic() + REAP_GRACE_SECONDS
                    if time.monotonic() >= post_exit_deadline:
                        cleanup_required = True
                        cleanup_reason = "dedicated G1 binary output pipes could not be closed"
                        break
                    selector.select(timeout=min(0.05, max(0.0, post_exit_deadline - time.monotonic())))
                    continue
                cleanup_required = True
                cleanup_reason = "dedicated G1 binary left a descendant in its process group"
                break
            if time.monotonic() >= deadline:
                timed_out = True
                cleanup_required = True
                cleanup_reason = "dedicated G1 binary timed out"
                break
            selector.select(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        if cleanup_required and process is not None and pgid is not None and selector is not None:
            cleanup_proven = _cleanup_process_group(process, pgid, selector, buffers, streams)
            if not cleanup_proven:
                cleanup_reason = f"{cleanup_reason}; process group could not be proven gone and reaped"
        if process is not None:
            exit_code = process.poll()
        else:
            exit_code = None
        stdout = bytes(buffers["stdout"])
        stderr = bytes(buffers["stderr"])
        crashed = exit_code is not None and exit_code < 0
        duration = max(0.0, (now() - started).total_seconds())
        if not cleanup_proven:
            return EvidenceExecution(
                "INFRA_ERROR", exit_code, timed_out, crashed, stdout, stderr, duration, None,
                cleanup_reason or "process cleanup could not be proven", artifact_sha256=(artifact_snapshot.sha256 if artifact_snapshot else None),
                cleanup_proven=False,
            )
        if cleanup_required:
            state = "TIMEOUT" if timed_out else "INFRA_ERROR" if output_overflow or loader_error else "INFRA_ERROR"
            return EvidenceExecution(
                state, exit_code, timed_out, crashed, stdout, stderr, duration, None,
                cleanup_reason, artifact_sha256=(artifact_snapshot.sha256 if artifact_snapshot else None),
            )
        if not finished_normally:
            return EvidenceExecution(
                "INFRA_ERROR", exit_code, timed_out, crashed, stdout, stderr, duration, None,
                "dedicated G1 binary did not finish normally", artifact_sha256=(artifact_snapshot.sha256 if artifact_snapshot else None),
            )
        runtime_binding = None
        if exit_code == 0:
            runtime_binding = _runtime_binding(observed_loader_paths, environment)
    except (OSError, subprocess.SubprocessError, ContractError) as exc:
        if process is not None and pgid is not None and selector is not None and process.poll() is None:
            cleanup_proven = _cleanup_process_group(process, pgid, selector, buffers, streams)
        elif process is not None:
            cleanup_proven = _bounded_reap_without_capture(process, pgid)
        else:
            cleanup_proven = True
        exit_code = process.poll() if process is not None else None
        stdout = bytes(buffers["stdout"])
        stderr = bytes(buffers["stderr"])
        return EvidenceExecution(
            "INFRA_ERROR", exit_code, timed_out, bool(exit_code is not None and exit_code < 0),
            stdout, stderr, max(0.0, (now() - started).total_seconds()), None, str(exc),
            artifact_sha256=(artifact_snapshot.sha256 if artifact_snapshot else None),
            cleanup_proven=cleanup_proven,
        )
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        if selector is not None:
            for fd, (_name, stream) in list(streams.items()):
                try:
                    selector.unregister(fd)
                except Exception:
                    pass
                try:
                    stream.close()
                except OSError:
                    pass
            selector.close()
    try:
        payload = _decode_payload(stdout)
    except ContractError as exc:
        return EvidenceExecution("INFRA_ERROR", exit_code, False, False, stdout, stderr, duration, None, str(exc), artifact_sha256=artifact_snapshot.sha256 if artifact_snapshot else None)
    if exit_code != 0:
        state = payload.get("state") if isinstance(payload, dict) else None
        if state not in ALLOWED_STATES or state == "PASS":
            state = "FAIL"
        reason = payload.get("reason") if isinstance(payload, dict) else None
        return EvidenceExecution(state, exit_code, False, False, stdout, stderr, duration, payload, str(reason or "dedicated G1 binary returned non-zero"), artifact_sha256=artifact_snapshot.sha256 if artifact_snapshot else None)
    try:
        _validate_evidence_pass_payload(payload)
    except ContractError as exc:
        return EvidenceExecution("INFRA_ERROR", exit_code, False, False, stdout, stderr, duration, payload if isinstance(payload, dict) else None, str(exc), artifact_sha256=artifact_snapshot.sha256 if artifact_snapshot else None)
    return EvidenceExecution("PASS", 0, False, False, stdout, stderr, duration, payload, None, runtime_binding=runtime_binding, artifact_sha256=artifact_snapshot.sha256 if artifact_snapshot else None)


def _report_health_observation(
    raw: Mapping[str, Any] | None, row: Mapping[str, Any], fallback_time: datetime
) -> dict[str, Any]:
    if raw is None:
        raise ContractError("G1 health observation is missing")
    if raw.get("source") != "amd-smi-sysfs-read-only-v1":
        raise ContractError("G1 health observation is not from the read-only G0 provider")
    if {
        "bdf": raw.get("bdf"), "uuid": raw.get("uuid"), "gcnArchName": raw.get("gcnArchName"),
    } != {"bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"]}:
        raise ContractError("G1 health observation is not bound to the canonical device")
    observed_at = str(raw["observed_at"])
    facts = raw["facts"]
    return {
        "available": True,
        "reliable": True,
        "source": "g0-read-only-health-v1",
        "observed_at": observed_at,
        "device": {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]},
        "state": "OK",
        "device_state": facts["device_state"],
        "runtime_status": facts["runtime_status"],
        "amdgpu_driver_bound": facts["amdgpu_driver_bound"],
        "ras_uncorrectable_count": facts["ras_uncorrectable_count"],
        "temperature_c": facts["temperature_c"],
    }


def _report_process_observation(
    raw: Mapping[str, Any] | None, row: Mapping[str, Any], fallback_time: datetime
) -> dict[str, Any]:
    if raw is None:
        raise ContractError("G1 process observation is missing")
    if raw.get("source") != "amd-smi-sysfs-read-only-v1":
        raise ContractError("G1 process observation is not from the read-only G0 provider")
    if {
        "bdf": raw.get("bdf"), "uuid": raw.get("uuid"), "gcnArchName": raw.get("gcnArchName"),
    } != {"bdf": row["bdf"], "uuid": row["uuid"], "gcnArchName": row["target"]}:
        raise ContractError("G1 process observation is not bound to the canonical device")
    return {
        "available": True,
        "reliable": True,
        "source": "g0-read-only-process-v1",
        "observed_at": str(raw["observed_at"]) if raw is not None else iso(fallback_time),
        "device": {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]},
        "state": "CLEAN",
        "gpu_processes": copy.deepcopy(raw["gpu_processes"]),
        "residual_runner_children": copy.deepcopy(raw["residual_runner_children"]),
    }


def _candidate_placeholder(args: argparse.Namespace) -> dict[str, Any]:
    def value(name: str) -> str:
        candidate = getattr(args, name, None)
        return candidate if isinstance(candidate, str) and len(candidate) == 40 and all(char in "0123456789abcdef" for char in candidate) else ZERO_SHA40

    return {
        "reviewed_sha": value("reviewed_sha"),
        "tested_sha": value("tested_sha"),
        "workflow_sha": value("workflow_sha"),
        "git_tree_oid": value("tree_oid"),
        "worktree_clean": True,
        "revision_input": "full-sha",
    }


def _execution_record(
    execution: EvidenceExecution | None, duration_seconds: float
) -> dict[str, Any]:
    if execution is None:
        return {
            "command": list(COMMAND),
            "command_sha256": sha256_json(COMMAND),
            "exit_code": None,
            "timed_out": False,
            "crashed": False,
            "stdout_sha256": ZERO_SHA,
            "stderr_sha256": ZERO_SHA,
            "duration_seconds": duration_seconds,
        }
    return {
        "command": list(COMMAND),
        "command_sha256": sha256_json(COMMAND),
        "exit_code": execution.exit_code,
        "timed_out": execution.timed_out,
        "crashed": execution.crashed,
        "stdout_sha256": sha256_bytes(execution.stdout),
        "stderr_sha256": sha256_bytes(execution.stderr),
        "duration_seconds": duration_seconds,
    }


def _artifact_record(
    *,
    metadata_path: Path,
    source_artifact_path: Path,
    staged_artifact_path: Path,
    metadata_summary: Mapping[str, Any] | None,
    manifest_hashes: Mapping[str, str],
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if metadata_summary is None:
        raise ContractError("G1 PASS report requires validated artifact metadata")
    metadata_sidecar = metadata_path.with_name(metadata_path.name + ".sha256")
    artifact_sidecar = staged_artifact_path.with_name(staged_artifact_path.name + ".sha256")
    metadata_snapshot = _snapshot_regular(metadata_path, "G1 output metadata")
    metadata_sidecar_snapshot = _snapshot_regular(metadata_sidecar, "G1 output metadata sidecar")
    source_artifact_snapshot = _snapshot_regular(source_artifact_path, "G1 source runtime artifact")
    staged_artifact_snapshot = _snapshot_regular(staged_artifact_path, "G1 output runtime artifact")
    artifact_sidecar_snapshot = _snapshot_regular(artifact_sidecar, "G1 output artifact sidecar")
    return {
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": metadata_snapshot.sha256,
        "metadata_sidecar_sha256": metadata_sidecar_snapshot.sha256,
        "artifact_path": str(source_artifact_path),
        "staged_artifact_path": str(staged_artifact_path.resolve()),
        "artifact_sha256": staged_artifact_snapshot.sha256,
        "artifact_sidecar_sha256": artifact_sidecar_snapshot.sha256,
        "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
        "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
        "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
        "target": row["target"],
        "row_id": row["row_id"],
        "h3_executable_used": False,
    }


def make_report(
    *,
    row: Mapping[str, Any],
    matrix: Mapping[str, Any],
    repo: Path,
    candidate: Mapping[str, Any],
    run_id: str,
    run_attempt: int,
    created_at: datetime,
    started_at: datetime,
    finished_at: datetime,
    state: str,
    error: str | None,
    metadata_path: Path,
    source_artifact_path: Path,
    staged_artifact_path: Path,
    metadata_summary: Mapping[str, Any] | None,
    execution: EvidenceExecution | None,
    health_pre: Mapping[str, Any] | None,
    health_post: Mapping[str, Any] | None,
    process_pre: Mapping[str, Any] | None,
    process_post: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if state != "PASS" or error is not None or execution is None:
        raise ContractError("G1 reports are PASS-only and require a successful execution")
    if execution.exit_code != 0 or execution.timed_out or execution.crashed:
        raise ContractError("G1 PASS report requires a clean, non-timeout execution")
    _validate_evidence_pass_payload(execution.payload)
    _validate_runtime_binding(execution.runtime_binding)
    if execution.artifact_sha256 is None:
        raise ContractError("G1 PASS report requires an execution artifact hash")
    if _snapshot_regular(staged_artifact_path, "G1 executed output artifact").sha256 != execution.artifact_sha256:
        raise ContractError("G1 output artifact changed after descriptor-bound execution")
    health_pre_bound = _report_health_observation(health_pre, row, created_at)
    health_post_bound = _report_health_observation(health_post, row, finished_at)
    process_pre_bound = _report_process_observation(process_pre, row, created_at)
    process_post_bound = _report_process_observation(process_post, row, finished_at)

    actual_allocations = actual_copies = actual_dispatches = 0
    actual_total_dispatches = 0
    if execution is not None and execution.payload is not None:
        cases_payload = execution.payload.get("cases")
        if isinstance(cases_payload, list):
            for case in cases_payload:
                if isinstance(case, dict):
                    actual_allocations += case.get("allocation_count", 0) if _valid_nonnegative_int(case.get("allocation_count")) else 0
                    actual_copies += case.get("copy_count", 0) if _valid_nonnegative_int(case.get("copy_count")) else 0
                    actual_dispatches += case.get("dispatch_count", 0) if _valid_nonnegative_int(case.get("dispatch_count")) else 0
        if _valid_nonnegative_int(execution.payload.get("dispatch_count")):
            actual_total_dispatches = execution.payload["dispatch_count"]
    actual_total_dispatches = max(actual_total_dispatches, actual_dispatches)
    if (actual_allocations, actual_copies, actual_dispatches, actual_total_dispatches) != (12, 12, 6, 6):
        raise ContractError("G1 execution payload totals are not exact")
    cases = []
    for size in EXPECTED_SIZES:
        case_payload = None
        if execution is not None and isinstance(execution.payload, dict) and isinstance(execution.payload.get("cases"), list):
            case_payload = next(
                (value for value in execution.payload["cases"] if isinstance(value, dict) and value.get("size") == size),
                None,
            )
        if not isinstance(case_payload, dict):
            raise ContractError(f"G1 execution payload is missing case size {size}")
        cases.append(
            {
                "size": size,
                "state": "PASS",
                "byte_exact": case_payload.get("byte_exact"),
                "allocation_count": 2,
                "copy_count": 2,
                "kernel_dispatch_count": 1,
                "dispatch_count": 1,
                "timed_out": execution.timed_out,
                "fallback_used": False,
            }
        )
    return {
        "schema_version": "g1-report-v1",
        "report_id": f"{row['row_id']}.{run_id}.{run_attempt}",
        "row_id": row["row_id"],
        "target": row["target"],
        "state": state,
        "required": True,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "candidate": dict(candidate),
        "artifact": _artifact_record(
            metadata_path=metadata_path,
            source_artifact_path=source_artifact_path,
            staged_artifact_path=staged_artifact_path,
            metadata_summary=metadata_summary,
            manifest_hashes=_manifest_hashes(repo),
            row=row,
        ),
        "execution": _execution_record(
            execution, max(0.0, (finished_at - started_at).total_seconds())
        ),
        "runtime_binding": copy.deepcopy(execution.runtime_binding),
        "scope": {
            "selected_backend": "hip",
            "fallback_allowed": False,
            "fallback_used": False,
            "model_used": False,
            "semantic_op_used": False,
            "byte_exact_verified": True,
            "semantic_numerics_verified": False,
            "allocation_count": actual_allocations,
            "copy_count": actual_copies,
            "kernel_dispatch_count": max(1, actual_dispatches),
            "dispatch_count": actual_total_dispatches,
        },
        "device": {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]},
        "created_at": iso(created_at),
        "started_at": iso(started_at),
        "finished_at": iso(finished_at),
        "duration_seconds": max(0.0, (finished_at - started_at).total_seconds()),
        "health_pre": health_pre_bound,
        "health_post": health_post_bound,
        "process_pre": process_pre_bound,
        "process_post": process_post_bound,
        "cases": cases,
        "error": error,
    }


def write_report(output: Path, report: Mapping[str, Any]) -> None:
    if not output.is_dir() or output.is_symlink():
        raise ContractError("G1 report output is not a regular directory")
    data = canonical_bytes(dict(report))
    report_path = output / REPORT_NAME
    _write_exclusive(report_path, data, "G1 report", mode=0o600)
    _write_sidecar(report_path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--row", choices=("g1-gfx1030", "g1-gfx1201"), required=True)
    result.add_argument("--runtime-metadata", "--artifact-metadata", dest="runtime_metadata", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--trusted-local", action="store_true")
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--reviewed-sha", required=True)
    result.add_argument("--tested-sha", required=True)
    result.add_argument("--workflow-sha", required=True)
    result.add_argument("--git-tree-oid", "--tree-oid", dest="tree_oid", required=True)
    return result


def _validate_identity_inputs(args: argparse.Namespace) -> None:
    if not args.trusted_local:
        raise ContractError("explicit --trusted-local execution mode is required")
    if not isinstance(args.run_id, str) or RUN_ID_TOKEN.fullmatch(args.run_id) is None:
        raise ContractError("G1 run_id is malformed")
    if isinstance(args.run_attempt, bool) or not isinstance(args.run_attempt, int) or args.run_attempt < 1:
        raise ContractError("G1 run_attempt must be positive")
    for name in ("reviewed_sha", "tested_sha", "workflow_sha", "tree_oid"):
        exact_sha(getattr(args, name), name)
    if len({args.reviewed_sha, args.tested_sha, args.workflow_sha}) != 1:
        raise ContractError("reviewed/tested/workflow SHA values differ")


def _validate_observation_pair(
    health_pre: Mapping[str, Any],
    health_post: Mapping[str, Any],
    process_pre: Mapping[str, Any],
    process_post: Mapping[str, Any],
    row: Mapping[str, Any],
) -> None:
    require_available_observation(health_pre, "pre-health")
    require_available_observation(health_post, "post-health")
    require_available_observation(process_pre, "pre-process")
    require_available_observation(process_post, "post-process")
    validate_health(health_pre, "pre", row)
    validate_health(health_post, "post", row)
    validate_processes(process_pre, "pre", row)
    validate_processes(process_post, "post", row)
    for field in (
        "device_state",
        "amdgpu_driver_bound",
        "runtime_status",
        "ras_uncorrectable_count",
        "sysfs_ras_uncorrectable_count",
    ):
        if health_pre["facts"][field] != health_post["facts"][field]:
            raise ContractError(f"G1 post-health fact changed: {field}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    created_at = now()
    execution_started = created_at
    execution_finished = created_at
    candidate = _candidate_placeholder(args)
    metadata_path = args.runtime_metadata.resolve(strict=False)
    artifact_path = (metadata_path.parent / BINARY_NAME).resolve()
    metadata_summary: dict[str, Any] | None = None
    execution: EvidenceExecution | None = None
    health_pre: Mapping[str, Any] | None = None
    health_post: Mapping[str, Any] | None = None
    process_pre: Mapping[str, Any] | None = None
    process_post: Mapping[str, Any] | None = None
    state = "INFRA_ERROR"
    error: str | None = None
    output: Path | None = None
    row: dict[str, Any] | None = None
    matrix: dict[str, Any] | None = None
    repo: Path | None = None
    output_metadata: Path | None = None
    output_artifact: Path | None = None
    source_metadata_snapshot: FileSnapshot | None = None
    source_metadata_sidecar_snapshot: FileSnapshot | None = None
    source_artifact_snapshot: FileSnapshot | None = None
    source_artifact_sidecar_snapshot: FileSnapshot | None = None
    try:
        repo = _safe_repo(args.repo)
        matrix = validate_g1_matrix(repo)
        row = row_by_id(matrix, args.row)
        output = safe_output_directory(args.output_dir, repo, row)
        _validate_identity_inputs(args)
        reject_inherited_visibility_selectors(
            {name: os.environ.get(name) for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES")}
        )
        observed_candidate = git_candidate(
            repo, args.reviewed_sha, args.tested_sha, args.workflow_sha
        )
        if observed_candidate["git_tree_oid"] != args.tree_oid:
            raise ContractError("candidate tree does not match the checked-out immutable tree")
        candidate = observed_candidate
        metadata_path = _safe_staged_metadata(args.runtime_metadata, row, repo)
        metadata, source_metadata_snapshot = _read_json_nofollow(metadata_path, "G1 runtime metadata")
        if not isinstance(metadata, dict):
            raise ContractError("G1 runtime metadata must be a JSON object")
        artifact_path = _safe_staged_artifact(metadata.get("artifact", {}).get("path"), metadata_path, repo)
        if str(artifact_path) != metadata["artifact"]["path"]:
            raise ContractError("G1 metadata artifact path is not canonical")
        source_metadata_sidecar_snapshot = _snapshot_regular(
            metadata_path.with_name(metadata_path.name + ".sha256"),
            "G1 source metadata sidecar",
        )
        source_artifact_snapshot = _snapshot_regular(artifact_path, "G1 source runtime artifact")
        source_artifact_sidecar_snapshot = _snapshot_regular(
            artifact_path.with_name(artifact_path.name + ".sha256"),
            "G1 source artifact sidecar",
        )
        identity = {
            **observed_candidate,
            "run_id": args.run_id,
            "run_attempt": args.run_attempt,
        }
        metadata_summary = validate_artifact_metadata(
            metadata, artifact_path, metadata_path, row, identity, repo
        )
        if source_metadata_snapshot is None or source_metadata_sidecar_snapshot is None or source_artifact_snapshot is None or source_artifact_sidecar_snapshot is None:
            raise ContractError("G1 source snapshots are incomplete")
        _assert_snapshot(metadata_path, source_metadata_snapshot, "G1 runtime metadata")
        _assert_snapshot(
            metadata_path.with_name(metadata_path.name + ".sha256"),
            source_metadata_sidecar_snapshot,
            "G1 source metadata sidecar",
        )
        _assert_snapshot(artifact_path, source_artifact_snapshot, "G1 source runtime artifact")
        _assert_snapshot(
            artifact_path.with_name(artifact_path.name + ".sha256"),
            source_artifact_sidecar_snapshot,
            "G1 source artifact sidecar",
        )
        output_metadata, output_artifact = _copy_staged_artifacts(
            output,
            metadata_path,
            artifact_path,
            metadata_snapshot=source_metadata_snapshot,
            artifact_snapshot=source_artifact_snapshot,
        )

        with nonblocking_host_lock(Path("/tmp/sllm-g0.lock")):
            routing = amd_smi_list_json(row, executable=AMD_SMI_EXECUTABLE)
            visibility = validate_visibility_environment(
                {
                    "HIP_VISIBLE_DEVICES": str(routing["hip_id"]),
                    "CUDA_VISIBLE_DEVICES": None,
                    "GPU_DEVICE_ORDINAL": None,
                }
            )
            validate_routing(routing, visibility, row)
            health_pre = observe_health(
                row, routing, amd_smi=AMD_SMI_EXECUTABLE, sysfs_root=SYSFS_PCI_ROOT
            )
            require_available_observation(health_pre, "pre-health")
            validate_health(health_pre, "pre", row)
            process_pre = observe_processes(row, routing, amd_smi=AMD_SMI_EXECUTABLE)
            require_available_observation(process_pre, "pre-process")
            validate_processes(process_pre, "pre", row)
            execution_started = now()
            execution = run_evidence_binary(
                output_artifact,
                timeout_seconds=row["timeout_seconds"],
                hip_visible_devices=str(routing["hip_id"]),
                expected_artifact_sha256=source_artifact_snapshot.sha256,
            )
            execution_finished = now()
            health_post = observe_health(
                row, routing, amd_smi=AMD_SMI_EXECUTABLE, sysfs_root=SYSFS_PCI_ROOT
            )
            process_post = observe_processes(row, routing, amd_smi=AMD_SMI_EXECUTABLE)
            _validate_observation_pair(health_pre, health_post, process_pre, process_post, row)
            _assert_snapshot(artifact_path, source_artifact_snapshot, "G1 source runtime artifact")
            if execution.artifact_sha256 is not None and execution.artifact_sha256 != _snapshot_regular(output_artifact, "G1 executed output artifact").sha256:
                raise ContractError("G1 executed output artifact hash changed after execution")
            state = execution.state
            error = execution.error
    except (ContractError, KeyError, OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
        error = str(exc)
        state = "INFRA_ERROR"

    finished_at = now()
    if execution is None:
        execution_started = created_at
        execution_finished = finished_at
    if row is None or matrix is None or repo is None or output is None:
        print(f"G1 evidence: INFRA_ERROR: {error or 'runner setup failed'}", file=sys.stderr)
        return 2
    if state == "PASS" and execution is not None and output_metadata is not None and output_artifact is not None:
        try:
            report = make_report(
                row=row, matrix=matrix, repo=repo, candidate=candidate, run_id=args.run_id,
                run_attempt=args.run_attempt, created_at=created_at,
                started_at=execution_started, finished_at=execution_finished,
                state=state, error=error, metadata_path=output_metadata,
                source_artifact_path=artifact_path, staged_artifact_path=output_artifact,
                metadata_summary=metadata_summary,
                execution=execution, health_pre=health_pre, health_post=health_post,
                process_pre=process_pre, process_post=process_post,
            )
            validate_report(report, row, identity, output_artifact, output_metadata, matrix, repo)
        except (ContractError, KeyError, OSError, TypeError, ValueError) as exc:
            state = "INFRA_ERROR"
            error = str(exc)
    if state != "PASS" or execution is None or output_metadata is None or output_artifact is None:
        print(f"G1 evidence: {state}: {error or 'G1 evidence did not pass'}", file=sys.stderr)
        return 2
    # A failure has no honest representation in the PASS-only report schema.
    # Do not manufacture healthy observations or canonical positive counts.
    report = make_report(
        row=row, matrix=matrix, repo=repo, candidate=candidate, run_id=args.run_id,
        run_attempt=args.run_attempt, created_at=created_at,
        started_at=execution_started, finished_at=execution_finished,
        state=state, error=error, metadata_path=output_metadata,
        source_artifact_path=artifact_path, staged_artifact_path=output_artifact,
        metadata_summary=metadata_summary, execution=execution,
        health_pre=health_pre, health_post=health_post,
        process_pre=process_pre, process_post=process_post,
    )
    try:
        validate_schema(report, read_json(repo / "ci/schema/g1-report-v1.schema.json"), "G1 report")
        write_report(output, report)
    except (ContractError, OSError, TypeError, ValueError) as exc:
        print(f"G1 evidence: result/output contract failure: {exc}", file=sys.stderr)
        return 3
    if state != "PASS":
        print(f"G1 evidence: {state}: {error or 'G1 evidence did not pass'}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
