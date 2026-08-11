#!/usr/bin/env python3
"""Fail-closed aggregate for the two-row H3 public-runtime compile contract."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from validate_h3_public_runtime_contracts import (
        ContractError, EXPECTED_ENVIRONMENT, EXPECTED_SCOPE, ROOT, SHA40,
        TARGETS, RuntimeContractError, canonical_bytes, git_identity, read_json, sha256_file,
        sha256_json, validate_metadata, validate_report, validate_static,
    )
except ImportError:  # pragma: no cover - package import path used by some test runners
    from ci.tools.validate_h3_public_runtime_contracts import (
        ContractError, EXPECTED_ENVIRONMENT, EXPECTED_SCOPE, ROOT, SHA40,
        TARGETS, RuntimeContractError, canonical_bytes, git_identity, read_json, sha256_file,
        sha256_json, validate_metadata, validate_report, validate_static,
    )

EXPECTED_ROWS = ("h3-public-gfx1030", "h3-public-gfx1201")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
MAX_NEEDS_BYTES = 16 * 1024
_AT_EMPTY_PATH = 0x1000
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_TMPFILE_FLAGS = (
    os.O_RDWR
    | getattr(os, "O_TMPFILE", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


@dataclass(frozen=True)
class _DirectoryBinding:
    """An opened directory and the name which must continue to point to it."""

    path: Path
    parent_fd: int
    name: str
    fd: int


def _absolute_lexical(path: Path) -> Path:
    """Make a path absolute without resolving any symlink."""

    return Path(os.path.abspath(path))


def _strict_path(
    path: Path,
    *,
    workspace_root: Path | None,
    label: str,
    kind: str,
    allow_missing_leaf: bool = False,
) -> Path:
    """Validate a local path without allowing symlink traversal or escape."""

    absolute = _absolute_lexical(path)
    boundary = _absolute_lexical(workspace_root) if workspace_root is not None else None
    if boundary is not None:
        if not boundary.is_dir() or boundary.is_symlink():
            raise ContractError(f"{label} workspace root is missing or symlinked")
        if not absolute.is_relative_to(boundary):
            raise ContractError(f"{label} escapes the checked-out workspace")

    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, component in enumerate(parts):
        current /= component
        is_leaf = index == len(parts) - 1
        if current.is_symlink():
            raise ContractError(f"{label} traverses a symlink component")
        if not current.exists():
            if allow_missing_leaf and is_leaf:
                return absolute
            raise ContractError(f"{label} has a missing path component")
        if not is_leaf and not current.is_dir():
            raise ContractError(f"{label} has a non-directory ancestor")

    if kind == "directory":
        if not absolute.is_dir() or absolute.is_symlink():
            if not (allow_missing_leaf and not absolute.exists()):
                raise ContractError(f"{label} must be a regular nonsymlink directory")
    elif kind == "file":
        if not absolute.is_file() or absolute.is_symlink():
            raise ContractError(f"{label} must be a local regular nonsymlink file")
    else:  # pragma: no cover - all callers use a closed set of kinds
        raise ContractError(f"{label} has an unsupported path kind")
    return absolute


def _read_exact_needs(path: Path, *, workspace_root: Path | None = None) -> dict[str, Any]:
    """Read the local, bounded, exact two-row needs contract."""

    path = _strict_path(
        path,
        workspace_root=workspace_root,
        label="needs JSON",
        kind="file",
    )
    try:
        if path.stat().st_size > MAX_NEEDS_BYTES:
            raise ContractError("needs JSON exceeds its bounded local size")
        needs = read_json(path)
    except (OSError, RuntimeContractError, ValueError) as exc:
        raise ContractError("needs JSON cannot be read as a bounded local file") from exc
    if not isinstance(needs, dict) or set(needs) != {"state", "rows"}:
        raise ContractError("needs JSON must have exactly state and rows fields")
    if needs != {"state": "PASS", "rows": list(EXPECTED_ROWS)}:
        raise ContractError("needs JSON does not prove the exact ordered two-row PASS input")
    return needs


def _same_directory_entry(left: os.stat_result, right: os.stat_result) -> bool:
    """Compare the identity of directory entries, not their mutable pathnames."""

    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
    )


def _verify_directory_bindings(bindings: list[_DirectoryBinding]) -> None:
    """Reject an ancestor or output-leaf replacement after it was opened."""

    for binding in bindings:
        try:
            path_stat = os.stat(binding.name, dir_fd=binding.parent_fd, follow_symlinks=False)
            fd_stat = os.fstat(binding.fd)
        except OSError as exc:
            raise ContractError(f"aggregate output path was replaced: {binding.path}") from exc
        if not stat.S_ISDIR(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
            raise ContractError(f"aggregate output path was replaced: {binding.path}")


def _open_validated_output_directory(
    output_dir: Path,
    *,
    workspace_root: Path | None,
) -> tuple[Path, int, list[int], list[_DirectoryBinding]]:
    """Open output and every ancestor with descriptor-relative no-follow operations.

    Only the final component may be absent.  The caller owns all returned file
    descriptors and must close them.  A descriptor remains the authority for
    every later operation; pathname checks below are only race detection.
    """

    if (
        sys.platform != "linux"
        or os.name != "posix"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_TMPFILE")
    ):
        raise ContractError("aggregate publication requires Linux directory no-follow support")

    absolute = _absolute_lexical(Path(output_dir))
    boundary: Path | None = None
    if workspace_root is not None:
        boundary = _absolute_lexical(Path(workspace_root))
        if not absolute.is_relative_to(boundary):
            raise ContractError("aggregate output escapes the checked-out workspace")

    boundary_parts = list(boundary.parts[1:]) if boundary is not None else []
    if boundary is None:
        output_parts = list(absolute.parts[1:])
    else:
        output_parts = list(absolute.relative_to(boundary).parts)
    components = boundary_parts + output_parts

    opened: list[int] = []
    bindings: list[_DirectoryBinding] = []
    try:
        current_fd = os.open("/", _DIRECTORY_FLAGS)
        opened.append(current_fd)
        for index, component in enumerate(components):
            is_output_leaf = bool(output_parts) and index == len(components) - 1
            parent_fd = current_fd
            path = Path(absolute.anchor).joinpath(*components[: index + 1])

            try:
                entry_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError as exc:
                if not is_output_leaf:
                    raise ContractError(f"aggregate output has a missing ancestor: {path}") from exc
                try:
                    os.mkdir(component, mode=0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                except OSError as mkdir_exc:
                    raise ContractError(f"aggregate output directory could not be created: {path}") from mkdir_exc
                try:
                    entry_stat = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                except OSError as stat_exc:
                    raise ContractError(f"aggregate output directory disappeared after mkdir: {path}") from stat_exc
            except OSError as exc:
                raise ContractError(f"aggregate output ancestor cannot be inspected: {path}") from exc

            if stat.S_ISLNK(entry_stat.st_mode):
                raise ContractError(f"aggregate output traverses a symlink: {path}")
            if not stat.S_ISDIR(entry_stat.st_mode):
                raise ContractError(f"aggregate output component is not a directory: {path}")
            try:
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            except OSError as exc:
                raise ContractError(f"aggregate output component cannot be opened safely: {path}") from exc
            opened.append(child_fd)
            try:
                child_stat = os.fstat(child_fd)
            except OSError as exc:
                raise ContractError(f"aggregate output component cannot be inspected safely: {path}") from exc
            if not _same_directory_entry(entry_stat, child_stat):
                raise ContractError(f"aggregate output component was replaced while opening: {path}")
            binding = _DirectoryBinding(path=path, parent_fd=parent_fd, name=component, fd=child_fd)
            bindings.append(binding)
            current_fd = child_fd

        return absolute, current_fd, opened, bindings
    except Exception:
        for fd in reversed(opened):
            try:
                os.close(fd)
            except OSError:
                pass
        raise


def _check_output_is_empty(
    output_path: Path,
    output_fd: int,
    bindings: list[_DirectoryBinding],
) -> None:
    """Check emptiness from the FD and use pathname observation only to detect swaps."""

    _verify_directory_bindings(bindings)
    try:
        # This observation is deliberately not trusted for publication.  It
        # preserves a deterministic race point for the legacy Path.iterdir
        # validation while the descriptor inventory below remains authoritative.
        path_entry = next(iter(output_path.iterdir()), None)
    except OSError as exc:
        raise ContractError("aggregate output directory cannot be inspected") from exc
    _verify_directory_bindings(bindings)
    if path_entry is not None:
        raise ContractError("aggregate output directory must be empty")
    try:
        fd_entries = os.listdir(output_fd)
    except OSError as exc:
        raise ContractError("aggregate output directory cannot be listed by descriptor") from exc
    _verify_directory_bindings(bindings)
    if fd_entries:
        raise ContractError("aggregate output directory must be empty")


def _open_anonymous_output_file(output_fd: int) -> int:
    """Create an unnamed file in the already validated output directory."""

    try:
        return os.open(".", _TMPFILE_FLAGS, 0o600, dir_fd=output_fd)
    except OSError as exc:
        raise ContractError("aggregate publication requires filesystem O_TMPFILE support") from exc


def _write_and_sync(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written <= 0:  # pragma: no cover - defensive guard for a broken descriptor
            raise OSError("short write while publishing aggregate")
        view = view[written:]
    os.fsync(fd)


def _link_fd_no_replace(source_fd: int, output_fd: int, name: str) -> None:
    """Atomically publish an open file descriptor without replacing a name."""

    try:
        libc = ctypes.CDLL(None, use_errno=True)
        linkat = libc.linkat
        linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        linkat.restype = ctypes.c_int
        result = linkat(source_fd, b"", output_fd, os.fsencode(name), _AT_EMPTY_PATH)
    except (AttributeError, OSError) as exc:
        raise ContractError("aggregate publication lacks Linux linkat support") from exc
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == getattr(os, "EEXIST", 17):
            raise ContractError(f"aggregate output already contains {name}")
        raise ContractError(f"aggregate publication failed for {name}: errno {error_number}")


def _verify_published_file(fd: int, output_fd: int, name: str) -> None:
    try:
        path_stat = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        fd_stat = os.fstat(fd)
    except OSError as exc:
        raise ContractError(f"published aggregate file disappeared or was replaced: {name}") from exc
    if not stat.S_ISREG(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
        raise ContractError(f"published aggregate file was replaced or symlinked: {name}")


def _unlink_owned_file(fd: int, output_fd: int, name: str) -> None:
    """Remove only a file still bound to our open inode during failure cleanup."""

    try:
        path_stat = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
        fd_stat = os.fstat(fd)
    except FileNotFoundError:
        return
    except OSError:
        return
    if not stat.S_ISREG(path_stat.st_mode) or not _same_directory_entry(path_stat, fd_stat):
        return
    try:
        os.unlink(name, dir_fd=output_fd)
    except OSError:
        pass


def write_summary(
    output_dir: Path,
    summary: dict[str, Any],
    *,
    workspace_root: Path | None = None,
) -> dict[str, str]:
    payload = canonical_bytes(summary)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    sidecar_payload = f"{payload_sha256}  aggregate.json\n".encode("ascii")
    sidecar_sha256 = hashlib.sha256(sidecar_payload).hexdigest()
    output_path, output_fd, opened_fds, bindings = _open_validated_output_directory(
        output_dir,
        workspace_root=workspace_root,
    )
    aggregate_fd: int | None = None
    sidecar_fd: int | None = None
    published: list[tuple[str, int]] = []
    try:
        _check_output_is_empty(output_path, output_fd, bindings)
        aggregate_fd = _open_anonymous_output_file(output_fd)
        sidecar_fd = _open_anonymous_output_file(output_fd)
        _write_and_sync(aggregate_fd, payload)
        _write_and_sync(sidecar_fd, sidecar_payload)
        _verify_directory_bindings(bindings)
        try:
            if os.listdir(output_fd):
                raise ContractError("aggregate output directory changed before publication")
        except OSError as exc:
            raise ContractError("aggregate output directory cannot be rechecked before publication") from exc

        # Publish the payload first.  A hard exit can therefore leave only the
        # payload; it must never leave an orphan sidecar that claims a missing
        # aggregate.json.
        for name, fd in (("aggregate.json", aggregate_fd), ("aggregate.json.sha256", sidecar_fd)):
            _verify_directory_bindings(bindings)
            try:
                _link_fd_no_replace(fd, output_fd, name)
            except Exception:
                # linkat may have succeeded before its wrapper raised.  The
                # open inode is the ownership proof for safe cleanup.
                _unlink_owned_file(fd, output_fd, name)
                raise
            published.append((name, fd))
            try:
                _verify_published_file(fd, output_fd, name)
            except Exception:
                _unlink_owned_file(fd, output_fd, name)
                raise
            expected = {item[0] for item in published}
            try:
                current = set(os.listdir(output_fd))
            except OSError as exc:
                raise ContractError("aggregate output directory cannot be checked after publication") from exc
            _verify_directory_bindings(bindings)
            if current != expected:
                raise ContractError("aggregate output directory changed during publication")
        os.fsync(output_fd)
        _verify_directory_bindings(bindings)
        return {"sha256": payload_sha256, "sidecar_sha256": sidecar_sha256}
    except Exception:
        for name, fd in reversed(published):
            _unlink_owned_file(fd, output_fd, name)
        try:
            if published:
                os.fsync(output_fd)
        except OSError:
            pass
        raise
    finally:
        for fd in (sidecar_fd, aggregate_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        for fd in reversed(opened_fds):
            try:
                os.close(fd)
            except OSError:
                pass


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    repo = _strict_path(
        args.repo,
        workspace_root=None,
        label="checked-out workspace",
        kind="directory",
    )
    artifact_dir = _strict_path(
        args.artifact_dir,
        workspace_root=repo,
        label="public-runtime artifact directory",
        kind="directory",
    )
    output_dir = _strict_path(
        args.output_dir,
        workspace_root=repo,
        label="aggregate output directory",
        kind="directory",
        allow_missing_leaf=True,
    )
    needs_path = getattr(args, "needs_json", None)
    if needs_path is None:
        raise ContractError("strict public-runtime aggregation requires an exact needs JSON")
    needs_path = _strict_path(
        needs_path,
        workspace_root=repo,
        label="needs JSON",
        kind="file",
    )
    commit, tree, clean = git_identity(repo)
    if not clean:
        raise ContractError("strict public-runtime aggregation rejects a dirty checkout")
    for name, value in (("reviewed SHA", args.reviewed_sha), ("tested SHA", args.tested_sha), ("workflow SHA", args.workflow_sha), ("tree OID", args.tree_oid)):
        expected = commit if name != "tree OID" else tree
        if not isinstance(value, str) or not SHA40.fullmatch(value) or value != expected:
            raise ContractError(f"{name} is not the checked-out immutable identity")
    if args.run_attempt < 1 or not RUN_ID.fullmatch(str(args.run_id)):
        raise ContractError("aggregate run identity is invalid")
    needs = _read_exact_needs(needs_path, workspace_root=repo)
    toolchain, matrix, rows = validate_static(repo)
    row_dirs = sorted(path for path in artifact_dir.iterdir())
    if {path.name for path in row_dirs} != set(EXPECTED_ROWS) or any(not path.is_dir() or path.is_symlink() for path in row_dirs):
        raise ContractError("aggregate requires exactly the gfx1030 and gfx1201 row directories")
    if output_dir == artifact_dir or output_dir.is_relative_to(artifact_dir):
        raise ContractError("aggregate output must be private and outside the row artifact directory")
    summaries: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for row_id in EXPECTED_ROWS:
        row_dir = artifact_dir / row_id
        metadata_path = row_dir / "hip-runtime-artifact.json"
        report_path = row_dir / "report.json"
        metadata = validate_metadata(metadata_path, repo, expected_sha=commit, expected_tree=tree, artifact_root=row_dir)
        report, report_sha, report_sidecar_sha = validate_report(report_path, metadata, repo, row_dir)
        if report["state"] != "PASS" or metadata["matrix_row_id"] != row_id:
            raise ContractError(f"{row_id} is not a PASS row")
        if metadata["run"] != {"run_id": str(args.run_id), "run_attempt": args.run_attempt}:
            raise ContractError(f"{row_id} run identity differs from aggregate inputs")
        identities.append(metadata["candidate"])
        summaries.append({"row_id": row_id, "target": metadata["target"], "state": "PASS", "report_sha256": report_sha, "report_sidecar_sha256": report_sidecar_sha, "metadata_sha256": sha256_file(metadata_path), "metadata_sidecar_sha256": sha256_file(metadata_path.with_name(metadata_path.name + ".sha256")), "device_object_sha256": metadata["hashes"]["device_object"]["sha256"]})
        expected_root = {"hip-runtime-artifact.json", "hip-runtime-artifact.json.sha256", "report.json", "report.json.sha256"}
        expected_build = {name for record in metadata["hashes"].values() for name in (Path(record["path"]).name, Path(record["sidecar_path"]).name)}
        actual_root = {path.name for path in row_dir.iterdir() if path.is_file() or path.is_symlink()}
        build_dir = row_dir / "build"
        actual_build = {path.name for path in build_dir.iterdir()} if build_dir.is_dir() and not build_dir.is_symlink() else set()
        if actual_root != expected_root or actual_build != expected_build:
            raise ContractError(f"{row_id} artifact directory has missing, extra, or symlinked outputs")
    if len(summaries) != 2 or {item["row_id"] for item in summaries} != set(EXPECTED_ROWS):
        raise ContractError("aggregate did not collect exactly two unique PASS rows")
    if any(identity != identities[0] for identity in identities[1:]):
        raise ContractError("row candidate identities differ")
    if identities[0] != {"commit_sha": commit, "tree_oid": tree, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit}:
        raise ContractError("row identity is not the expected immutable candidate")
    summary = {"schema_version": "hip-runtime-aggregate-v1", "aggregate_id": f"h3-public-runtime-aggregate.{args.run_id}.{args.run_attempt}", "state": "PASS", "required": False, "evidence_mode": "required-ci", "run_id": str(args.run_id), "run_attempt": args.run_attempt, "reviewed_sha": commit, "tested_sha": commit, "workflow_sha": commit, "git_tree_oid": tree, "toolchain_id": toolchain["toolchain_id"], "toolchain_manifest_sha256": sha256_json(toolchain), "matrix_id": matrix["matrix_id"], "matrix_manifest_sha256": sha256_json(matrix), "expected_rows": list(EXPECTED_ROWS), "scope": {key: EXPECTED_SCOPE[key] for key in ("public_runtime_stub_linked", "compile_only", "execution_attempted", "gpu_execution", "model_used", "network_used", "fallback_allowed", "fallback_used", "cpu_fallback_used", "support_claim")}, "rows": summaries, "errors": []}
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha", required=True)
    result.add_argument("--tested-sha", "--expected-tested-sha", dest="tested_sha", required=True)
    result.add_argument("--workflow-sha", "--expected-workflow-sha", dest="workflow_sha", required=True)
    result.add_argument("--tree-oid", "--expected-tree-oid", dest="tree_oid", required=True)
    result.add_argument("--needs-json", type=Path)
    result.add_argument("--strict-ci", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if not args.strict_ci:
            raise ContractError("public-runtime aggregation requires --strict-ci")
        summary = aggregate(args)
        workspace = _strict_path(
            args.repo,
            workspace_root=None,
            label="checked-out workspace",
            kind="directory",
        )
        output_dir = _strict_path(
            args.output_dir,
            workspace_root=workspace,
            label="aggregate output directory",
            kind="directory",
            allow_missing_leaf=True,
        )
        hashes = write_summary(output_dir, summary, workspace_root=workspace)
        print(json.dumps({"state": "PASS", "aggregate": str(output_dir / 'aggregate.json'), **hashes}, sort_keys=True))
        return 0
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"H3 public-runtime aggregate: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
