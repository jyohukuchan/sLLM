"""Generic parent-issued exact action manifests and one-shot state.

This module deliberately contains no compiler, CMake, Cargo, or HIP policy.
Callers derive reviewed declarative recipes, turn an accepted observation into a
complete action manifest, and atomically consume that manifest before launch.
"""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
import secrets
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class ExactActionError(ValueError):
    """An exact-action manifest or state-machine invariant was violated."""


INPUT_VIEW_FD_BASE = 300
INPUT_VIEW_ALGORITHM = "sealed-input-view-v1"
INPUT_VIEW_SEALS = fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _directory_identity(path: Path, label: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        details = resolved.stat()
    except OSError as exc:
        raise ExactActionError(f"{label} directory is unavailable") from exc
    if path.is_symlink() or not resolved.is_dir():
        raise ExactActionError(f"{label} directory is not a non-symlink directory")
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "device": int(details.st_dev),
        "inode": int(details.st_ino),
    }


def file_record(path: Path, *, role: str, label: str) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        details = resolved.stat()
        data = resolved.read_bytes()
    except OSError as exc:
        raise ExactActionError(f"{label} is unavailable") from exc
    if path.is_symlink() or not resolved.is_file():
        raise ExactActionError(f"{label} is not a non-symlink regular file")
    return {
        "role": role,
        "path": str(path),
        "resolved_path": str(resolved),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
        "device": int(details.st_dev),
        "inode": int(details.st_ino),
    }


def implicit_record(*, role: str, value: bytes) -> dict[str, Any]:
    """Bind implicit configuration or response-file bytes, including empty data."""

    return {
        "role": role,
        "size_bytes": len(value),
        "sha256": sha256_bytes(value),
        "bytes_hex": value.hex(),
    }


def output_record(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise ExactActionError(f"{label} output path is malformed")
    parent = _directory_identity(path.parent, f"{label} output parent")
    if path.exists() or path.is_symlink():
        raise ExactActionError(f"{label} output already exists or is a symlink")
    return {"path": str(path), "parent": parent}


def _read_verified_descriptor(descriptor: int, record: Mapping[str, Any], label: str) -> bytes:
    """Read bytes only after checking the opened inode, then check it again."""

    try:
        before = os.fstat(descriptor)
        if (int(before.st_dev), int(before.st_ino), int(before.st_size)) != (
            int(record["device"]), int(record["inode"]), int(record["size_bytes"])
        ):
            raise ExactActionError(f"{label} inode or size changed before immutable snapshot")
        chunks: list[bytes] = []
        remaining = int(before.st_size)
        offset = 0
        while remaining:
            chunk = os.pread(descriptor, min(1024 * 1024, remaining), offset)
            if not chunk:
                raise ExactActionError(f"{label} ended before its validated size")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ExactActionError(f"{label} could not be read for immutable snapshot") from exc
    if (int(after.st_dev), int(after.st_ino), int(after.st_size)) != (
        int(record["device"]), int(record["inode"]), int(record["size_bytes"])
    ) or sha256_bytes(data) != record["sha256"]:
        raise ExactActionError(f"{label} bytes changed during immutable snapshot")
    return data


def _sealed_bytes(data: bytes, label: str) -> int:
    try:
        descriptor = os.memfd_create(label, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, INPUT_VIEW_SEALS)
        os.set_inheritable(descriptor, False)
        return descriptor
    except OSError as exc:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise ExactActionError(f"{label} could not be sealed into an immutable descriptor") from exc


def _open_input_record(record: Mapping[str, Any], label: str) -> tuple[int, bytes]:
    path = Path(str(record["path"]))
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ExactActionError(f"{label} pathname could not be opened without following a symlink") from exc
    try:
        return descriptor, _read_verified_descriptor(descriptor, record, label)
    except BaseException:
        os.close(descriptor)
        raise


def _argv_include_options(argv: Sequence[str]) -> list[tuple[int, str, str]]:
    result: list[tuple[int, str, str]] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in {"-I", "-isystem", "-iquote", "-idirafter"} and index + 1 < len(argv):
            result.append((index + 1, value, argv[index + 1]))
            index += 2
            continue
        for option in ("-I", "-isystem", "-iquote", "-idirafter"):
            if value.startswith(option) and len(value) > len(option):
                result.append((index, option, value[len(option):]))
                break
        index += 1
    return result


@dataclass
class ImmutableInputView:
    """A compiler argv view whose input bytes are sealed before spawn.

    Direct input operands use ``/proc/self/fd`` references to sealed memfds.
    Include search roots are private, read-only name maps whose leaves point to
    those same sealed descriptors.  The descriptors are duplicated into fixed
    child FDs by the final posix_spawn boundary, so pathname mutation after
    this object is built cannot change compiler input bytes.
    """

    argv: list[str]
    records: list[dict[str, Any]]
    include_directories: list[dict[str, str]]
    _descriptors: list[int]
    _fd_targets: list[int]
    _record_targets: list[int]
    _view_root: Path | None
    _closed: bool = False

    def spawn_file_actions(self) -> list[tuple[int, int, int]]:
        if self._closed:
            raise ExactActionError("immutable compiler input view is already closed")
        return [(os.POSIX_SPAWN_DUP2, source, target) for source, target in zip(self._descriptors, self._fd_targets, strict=True)]

    def transcript(self) -> dict[str, Any]:
        if self._closed:
            raise ExactActionError("immutable compiler input view is already closed")
        return {
            "algorithm": INPUT_VIEW_ALGORITHM,
            "argv": list(self.argv),
            "argv_sha256": sha256(self.argv),
            "inputs": [dict(record, view_fd=target) for record, target in zip(self.records, self._record_targets, strict=True)],
            "include_directories": [dict(item) for item in self.include_directories],
            "sealed": True,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in self._descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if self._view_root is not None:
            for directory in sorted((item for item in self._view_root.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
                try:
                    directory.chmod(0o700)
                except OSError:
                    pass
            try:
                self._view_root.chmod(0o700)
            except OSError:
                pass
            shutil.rmtree(self._view_root, ignore_errors=True)


def seal_input_view(manifest: Mapping[str, Any]) -> ImmutableInputView:
    """Atomically bind every declared input to validated bytes before spawn.

    The open/hash/fstat sequence rejects a post-validation pathname mutation;
    only after every input has passed is the compiler argv rewritten to the
    sealed descriptor view.  Implicit and response-file bytes are already
    embedded in the exact manifest and therefore need no pathname lookup.
    """

    checked = validate_manifest(manifest)
    descriptors: list[int] = []
    by_path: dict[str, tuple[int, bytes]] = {}
    records: list[dict[str, Any]] = []
    try:
        for index, source in enumerate(checked["inputs"]):
            key = str(source["path"])
            if key not in by_path:
                source_fd, data = _open_input_record(source, f"exact action input {index}")
                try:
                    sealed = _sealed_bytes(data, f"sllm-exact-input-{index}")
                finally:
                    os.close(source_fd)
                by_path[key] = (sealed, data)
                descriptors.append(sealed)
            sealed, _data = by_path[key]
            records.append(dict(source))
        target_base = max(INPUT_VIEW_FD_BASE, max(descriptors, default=0) + len(descriptors) + 8)
        fd_targets = [target_base + index for index in range(len(descriptors))]
        fd_by_path = {path: target for path, target in zip(by_path, fd_targets, strict=True)}
        rewritten = list(checked["argv"])
        for index, value in enumerate(rewritten):
            target = fd_by_path.get(value)
            if target is not None:
                rewritten[index] = f"/proc/self/fd/{target}"

        include_directories: list[dict[str, str]] = []
        view_root: Path | None = None
        include_options = _argv_include_options(rewritten)
        for option_index, option, raw_directory in include_options:
            original_directory = Path(raw_directory)
            if not original_directory.is_absolute():
                original_directory = Path(str(checked["cwd"]["path"])) / original_directory
            original_directory = Path(os.path.normpath(str(original_directory)))
            members: list[tuple[Path, int]] = []
            for record, target in zip(records, (fd_by_path[str(item["path"])] for item in records), strict=True):
                candidate = Path(str(record["path"]))
                try:
                    relative = candidate.relative_to(original_directory)
                except ValueError:
                    continue
                if relative.parts and all(part not in {"", ".", ".."} for part in relative.parts):
                    members.append((relative, target))
            if not members:
                continue
            if view_root is None:
                view_root = Path(tempfile.mkdtemp(prefix="sllm-exact-input-view-"))
            view_directory = view_root / f"include-{len(include_directories)}"
            view_directory.mkdir(mode=0o700)
            for relative, target in members:
                destination = view_directory / relative
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                os.symlink(f"/proc/self/fd/{target}", destination)
            for directory in sorted((item for item in view_directory.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
                directory.chmod(0o555)
            view_directory.chmod(0o555)
            view_path = str(view_directory)
            if option_index < len(rewritten) and rewritten[option_index] == raw_directory:
                rewritten[option_index] = view_path
            elif option_index < len(rewritten) and rewritten[option_index].startswith(option):
                rewritten[option_index] = option + view_path
            include_directories.append({"original": raw_directory, "view": view_path})
        if view_root is not None:
            view_root.chmod(0o555)
        record_targets = [fd_by_path[str(record["path"])] for record in records]
        return ImmutableInputView(rewritten, records, include_directories, descriptors, fd_targets, record_targets, view_root)
    except BaseException:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _environment_pairs(environment: Mapping[str, str]) -> list[list[str]]:
    if not isinstance(environment, Mapping):
        raise ExactActionError("action environment is not a mapping")
    pairs: list[list[str]] = []
    for key, value in environment.items():
        if not isinstance(key, str) or not isinstance(value, str) or not key or "\x00" in key or "\x00" in value:
            raise ExactActionError("action environment is malformed")
        pairs.append([key, value])
    pairs.sort(key=lambda item: item[0].encode("utf-8"))
    if len({item[0] for item in pairs}) != len(pairs):
        raise ExactActionError("action environment has duplicate keys")
    return pairs


def _manifest_digest(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("manifest_digest", None)
    return sha256(unsigned)


def _is_hex_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _validate_file_identity(record: Mapping[str, Any], *, require_role: bool, require_seals: bool, label: str) -> None:
    required = {"path", "resolved_path", "size_bytes", "sha256", "device", "inode"}
    if require_role:
        required.add("role")
    if require_seals:
        required.add("seals")
    if set(record) != required:
        raise ExactActionError(f"exact action {label} identity is not closed")
    if (
        not isinstance(record.get("path"), str)
        or not Path(str(record["path"])).is_absolute()
        or not isinstance(record.get("resolved_path"), str)
        or not Path(str(record["resolved_path"])).is_absolute()
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
        or not _is_hex_digest(record.get("sha256"))
        or any(isinstance(record.get(name), bool) or not isinstance(record.get(name), int) or record[name] < 1 for name in (("device", "inode", "seals") if require_seals else ("device", "inode")))
        or (require_role and (not isinstance(record.get("role"), str) or not record["role"] or "\x00" in record["role"]))
    ):
        raise ExactActionError(f"exact action {label} identity is malformed")


def _validate_directory_identity(record: Mapping[str, Any], label: str) -> None:
    if set(record) != {"path", "resolved_path", "device", "inode"}:
        raise ExactActionError(f"exact action {label} identity is malformed")
    if (
        not isinstance(record.get("path"), str)
        or not Path(str(record["path"])).is_absolute()
        or not isinstance(record.get("resolved_path"), str)
        or not Path(str(record["resolved_path"])).is_absolute()
        or any(isinstance(record.get(name), bool) or not isinstance(record.get(name), int) or record[name] < 1 for name in ("device", "inode"))
    ):
        raise ExactActionError(f"exact action {label} identity is malformed")


def _validate_implicit_record(record: Mapping[str, Any], label: str) -> None:
    if set(record) != {"role", "size_bytes", "sha256", "bytes_hex"}:
        raise ExactActionError(f"exact action {label} record is malformed")
    if (
        not isinstance(record.get("role"), str)
        or not record["role"]
        or "\x00" in record["role"]
        or isinstance(record.get("size_bytes"), bool)
        or not isinstance(record.get("size_bytes"), int)
        or record["size_bytes"] < 0
        or not _is_hex_digest(record.get("sha256"))
        or not isinstance(record.get("bytes_hex"), str)
    ):
        raise ExactActionError(f"exact action {label} record is malformed")
    try:
        value = bytes.fromhex(record["bytes_hex"])
    except ValueError as exc:
        raise ExactActionError(f"exact action {label} bytes are malformed") from exc
    if len(value) != record["size_bytes"] or sha256_bytes(value) != record["sha256"]:
        raise ExactActionError(f"exact action {label} bytes/digest differ")


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "action_id", "manifest_digest", "executable", "argv0", "argv", "cwd",
        "environment", "inputs", "implicit", "response_files", "outputs", "target",
        "occurrence_index", "occurrence_limit",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ExactActionError("exact action manifest is not closed")
    result = dict(manifest)
    if result["schema_version"] != "exact-parent-action-manifest-v1":
        raise ExactActionError("exact action manifest version is not canonical")
    for name in ("action_id", "manifest_digest"):
        if not _is_hex_digest(result.get(name)):
            raise ExactActionError(f"exact action {name} is malformed")
    executable = result.get("executable")
    if not isinstance(executable, Mapping):
        raise ExactActionError("exact action executable identity is not closed")
    _validate_file_identity(executable, require_role=False, require_seals=True, label="executable")
    if not isinstance(result.get("argv0"), str) or not result["argv0"] or "\x00" in result["argv0"]:
        raise ExactActionError("exact action argv0 is malformed")
    argv = result.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value or "\x00" in value for value in argv):
        raise ExactActionError("exact action argv is malformed")
    cwd = result.get("cwd")
    if not isinstance(cwd, Mapping):
        raise ExactActionError("exact action cwd identity is malformed")
    _validate_directory_identity(cwd, "cwd")
    environment = result.get("environment")
    if not isinstance(environment, list) or any(not isinstance(item, list) or len(item) != 2 or any(not isinstance(value, str) or "\x00" in value for value in item) or not item[0] for item in environment) or environment != sorted(environment, key=lambda item: item[0].encode("utf-8")) or len({item[0] for item in environment}) != len(environment):
        raise ExactActionError("exact action environment is not a canonical full environment")
    inputs = result.get("inputs")
    implicit = result.get("implicit")
    response_files = result.get("response_files")
    outputs = result.get("outputs")
    if not all(isinstance(values, list) for values in (inputs, implicit, response_files, outputs)):
        raise ExactActionError("exact action records are malformed")
    for record in inputs:
        if not isinstance(record, Mapping):
            raise ExactActionError("exact action input record is malformed")
        _validate_file_identity(record, require_role=True, require_seals=False, label="input")
    for name, values in (("implicit", implicit), ("response-file", response_files)):
        for record in values:
            if not isinstance(record, Mapping):
                raise ExactActionError(f"exact action {name} record is malformed")
            _validate_implicit_record(record, name)
    for record in outputs:
        if not isinstance(record, Mapping) or set(record) != {"path", "parent"} or not isinstance(record.get("path"), str) or not Path(str(record["path"])).is_absolute() or not isinstance(record.get("parent"), Mapping):
            raise ExactActionError("exact action output record is malformed")
        _validate_directory_identity(record["parent"], "output parent")
    if not isinstance(result.get("target"), str) or not result["target"]:
        raise ExactActionError("exact action target is malformed")
    if any(isinstance(result.get(name), bool) or not isinstance(result.get(name), int) or result[name] < 0 for name in ("occurrence_index", "occurrence_limit")) or result["occurrence_limit"] != 1 or result["occurrence_index"] != 0:
        raise ExactActionError("exact action occurrence bound is malformed")
    if _manifest_digest(result) != result["manifest_digest"]:
        raise ExactActionError("exact action manifest digest does not bind every field")
    return result


def make_manifest(
    *,
    executable: Mapping[str, Any],
    argv0: str,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
    implicit: Sequence[Mapping[str, Any]],
    response_files: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
    target: str,
) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "exact-parent-action-manifest-v1",
        "action_id": secrets.token_hex(32),
        "manifest_digest": "0" * 64,
        "executable": dict(executable),
        "argv0": argv0,
        "argv": list(argv),
        "cwd": _directory_identity(cwd, "action cwd"),
        "environment": _environment_pairs(environment),
        "inputs": [dict(record) for record in inputs],
        "implicit": [dict(record) for record in implicit],
        "response_files": [dict(record) for record in response_files],
        "outputs": [dict(record) for record in outputs],
        "target": target,
        "occurrence_index": 0,
        "occurrence_limit": 1,
    }
    document["manifest_digest"] = _manifest_digest(document)
    return validate_manifest(document)


def validate_live_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Re-read every filesystem identity that can affect an issued action.

    A manifest's digest proves what the parent issued; this check proves that
    those byte/path identities still hold immediately before an executor uses
    them.  The executor itself is separately supplied through a sealed file
    descriptor, so its descriptor identity is compared by that executor.
    """

    checked = validate_manifest(manifest)
    cwd = _directory_identity(Path(str(checked["cwd"]["path"])), "action cwd")
    if cwd != checked["cwd"]:
        raise ExactActionError("action cwd identity changed after issuance")
    for record in checked["inputs"]:
        current = file_record(Path(str(record["path"])), role=str(record["role"]), label="exact action input")
        if current != record:
            raise ExactActionError("action input identity changed after issuance")
    for record in checked["outputs"]:
        current = output_record(Path(str(record["path"])), label="exact action")
        if current != record:
            raise ExactActionError("action output parent identity changed after issuance")
    return checked


@dataclass
class _Issued:
    manifest: dict[str, Any]
    state: str
    issued_at_ns: int
    consumed_at_ns: int | None = None


class OneShotBroker:
    """Thread-safe issuance/consumption state for complete exact actions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_recipe: dict[str, str] = {}
        self._issued: dict[str, _Issued] = {}
        self._terminal = False

    def issue(self, recipe_key: str, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        checked = validate_manifest(manifest)
        if not recipe_key or "\x00" in recipe_key:
            raise ExactActionError("exact action recipe key is malformed")
        with self._lock:
            if self._terminal:
                raise ExactActionError("exact action broker is terminal and cannot issue actions")
            existing_id = self._by_recipe.get(recipe_key)
            if existing_id is not None:
                return dict(self._issued[existing_id].manifest), False
            self._by_recipe[recipe_key] = checked["action_id"]
            self._issued[checked["action_id"]] = _Issued(dict(checked), "issued", time.monotonic_ns())
            return dict(checked), True

    def consume(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        checked = validate_manifest(manifest)
        with self._lock:
            if self._terminal:
                raise ExactActionError("exact action broker is terminal and cannot consume actions")
            issued = self._issued.get(checked["action_id"])
            if issued is None or issued.manifest["manifest_digest"] != checked["manifest_digest"] or issued.manifest != checked:
                raise ExactActionError("exact action request is not an issued complete manifest")
            if issued.state != "issued":
                raise ExactActionError("exact action has already been consumed")
            issued.state = "consumed"
            issued.consumed_at_ns = time.monotonic_ns()
            return dict(issued.manifest)

    def terminal(self) -> None:
        with self._lock:
            self._terminal = True

    def transcript(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "action_id": item.manifest["action_id"],
                    "action_digest": item.manifest["manifest_digest"],
                    "state": item.state,
                    "issued_at_ns": item.issued_at_ns,
                    "consumed_at_ns": item.consumed_at_ns,
                    "manifest": dict(item.manifest),
                }
                for item in self._issued.values()
            ]
