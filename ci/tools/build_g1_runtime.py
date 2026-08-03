#!/usr/bin/env python3
"""Build and stage one trusted-local G1 runtime artifact.

This builder is deliberately narrower than the H3 compile-only path.  It
accepts one complete candidate identity and one canonical G1 row, builds only
the dedicated Rust evidence binary, and leaves a small, private artifact
directory behind.  It never starts the resulting binary (or any other
artifact), and it never turns the host stub into GPU evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, exact_sha, read_json  # noqa: E402
import validate_g1_contracts as g1_contracts  # noqa: E402


EXPECTED_ROWS = ("g1-gfx1030", "g1-gfx1201")
EXPECTED_TARGETS = {row: row.removeprefix("g1-") for row in EXPECTED_ROWS}
EXPECTED_TOOLCHAIN_ID = "rocm-7.14.0"
EXPECTED_ROCM_ROOT = Path("/opt/rocm")
EXPECTED_RUST_TOOLCHAIN = "1.97.1"
EXPECTED_CODEGEN_FEATURES = (
    "co_v6,wave32,xnack=unsupported,sramecc=unsupported,"
    "generic_processor_version=0"
)
BINARY_NAME = "sllm-hip-evidence"
METADATA_NAME = "g1-runtime-artifact.json"
SIDECAR_SUFFIX = ".sha256"
OUTPUT_ROOT_PREFIX = "sllm-g1-"
PRIVATE_TMP = Path("/tmp")
MAX_BUILD_TIMEOUT_SECONDS = 900.0
DEFAULT_BUILD_TIMEOUT_SECONDS = 900.0
SHA40 = re.compile(r"^[0-9a-f]{40}$")
LLVM23 = re.compile(r"(?:AMD\s+)?(?:clang|LLVM).*?\b23\.", re.IGNORECASE)


class G1BuilderError(ContractError):
    """A fail-closed builder or input contract error."""


@dataclass(frozen=True)
class CommandOutput:
    argv: tuple[str, ...]
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""


@dataclass(frozen=True)
class BuildResult:
    """Paths and metadata from a successful build."""

    row_id: str
    target: str
    output_dir: Path
    artifact_path: Path
    metadata_path: Path
    artifact_sha256: str
    metadata_sha256: str
    command: tuple[str, ...]


CommandRunner = Callable[..., CommandOutput]


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise G1BuilderError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _decode_output(value: bytes, label: str) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise G1BuilderError(f"{label} returned non-UTF-8 output") from exc


def run_argv(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float,
) -> CommandOutput:
    """Run one argv-only command through the bounded process-group helper.

    The delegated helper uses ``shell=False`` and bounded nonblocking pipes;
    this wrapper preserves the builder-specific error and result types.
    """

    command = tuple(str(value) for value in argv)
    if not command or any("\x00" in value for value in command):
        raise G1BuilderError("subprocess argv is empty or contains NUL")
    if timeout <= 0 or timeout > MAX_BUILD_TIMEOUT_SECONDS:
        raise G1BuilderError(f"subprocess timeout is outside the bounded range: {timeout}")
    try:
        bounded = g1_contracts.run_bounded_argv(
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            max_stdout_bytes=g1_contracts.MAX_SUBPROCESS_STDOUT_BYTES,
            max_stderr_bytes=g1_contracts.MAX_SUBPROCESS_STDERR_BYTES,
        )
    except ContractError as exc:
        raise G1BuilderError(str(exc)) from exc
    result = CommandOutput(command, bounded.returncode, bounded.stdout, bounded.stderr)
    if result.returncode != 0:
        detail = _decode_output(result.stderr, "command stderr").strip()
        raise G1BuilderError(
            f"command failed with exit {result.returncode}: {' '.join(command)}"
            + (f": {detail[:1000]}" if detail else "")
        )
    return result


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject symlinked path components for builder-controlled output paths."""

    if not path.is_absolute() or "\x00" in str(path):
        raise G1BuilderError(f"{label} must be an absolute path without NUL")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            if current.is_symlink():
                raise G1BuilderError(f"{label} contains a symlink: {current}")
        except OSError as exc:
            raise G1BuilderError(f"cannot inspect {label}: {current}: {exc}") from exc


def _require_private_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    if not path.is_dir() or path.is_symlink():
        raise G1BuilderError(f"{label} is not a private directory: {path}")
    try:
        mode = path.stat().st_mode
        owner = path.stat().st_uid
    except OSError as exc:
        raise G1BuilderError(f"cannot stat {label}: {path}: {exc}") from exc
    if owner != os.getuid() or mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise G1BuilderError(f"{label} is not owned privately by the current user: {path}")


def _make_private_directory(path: Path, label: str) -> None:
    _reject_symlink_components(path, label)
    try:
        path.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise G1BuilderError(f"{label} already exists; refusing stale output: {path}") from exc
    except OSError as exc:
        raise G1BuilderError(f"cannot create private {label}: {path}: {exc}") from exc
    _require_private_directory(path, label)


def _repo_path(repo: Path) -> Path:
    if not repo.is_absolute() or repo.is_symlink() or not repo.is_dir():
        raise G1BuilderError(f"repository must be an existing non-symlink directory: {repo}")
    _reject_symlink_components(repo, "repository")
    try:
        return repo.resolve(strict=True)
    except OSError as exc:
        raise G1BuilderError(f"cannot canonicalize repository {repo}: {exc}") from exc


def _validate_expected_hashes(
    reviewed_sha: str | None,
    tested_sha: str | None,
    workflow_sha: str | None,
    tree_oid: str | None,
) -> dict[str, str]:
    values = {
        "reviewed_sha": reviewed_sha,
        "tested_sha": tested_sha,
        "workflow_sha": workflow_sha,
        "git_tree_oid": tree_oid,
    }
    for name, value in values.items():
        try:
            exact_sha(value, name)
        except ContractError as exc:
            raise G1BuilderError(str(exc)) from exc
    result = {name: str(value) for name, value in values.items()}
    if len({result["reviewed_sha"], result["tested_sha"], result["workflow_sha"]}) != 1:
        raise G1BuilderError("reviewed/tested/workflow candidate SHA values differ")
    return result


def git_candidate(repo: Path, *, runner: CommandRunner | None = None) -> dict[str, str]:
    """Read HEAD and its tree with bounded git commands."""

    runner = runner or run_argv
    commit = _decode_output(
        runner(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repo,
            timeout=30.0,
        ).stdout,
        "git commit identity",
    ).strip()
    tree = _decode_output(
        runner(
            ["git", "rev-parse", "--verify", "HEAD^{tree}"],
            cwd=repo,
            timeout=30.0,
        ).stdout,
        "git tree identity",
    ).strip()
    try:
        exact_sha(commit, "actual commit")
        exact_sha(tree, "actual tree")
    except ContractError as exc:
        raise G1BuilderError(str(exc)) from exc
    return {"commit": commit, "tree": tree}


def ensure_clean_worktree(repo: Path, *, runner: CommandRunner | None = None) -> None:
    runner = runner or run_argv
    result = runner(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        timeout=30.0,
    )
    status = _decode_output(result.stdout, "git worktree status")
    if status:
        preview = ", ".join(status.splitlines()[:8])
        raise G1BuilderError(f"candidate worktree is dirty: {preview[:1000]}")


def verify_candidate(
    repo: Path,
    expected: Mapping[str, str],
    *,
    runner: CommandRunner | None = None,
) -> None:
    runner = runner or run_argv
    ensure_clean_worktree(repo, runner=runner)
    actual = git_candidate(repo, runner=runner)
    if actual["commit"] != expected["reviewed_sha"]:
        raise G1BuilderError("worktree HEAD does not match reviewed candidate SHA")
    if actual["commit"] != expected["tested_sha"] or actual["commit"] != expected["workflow_sha"]:
        raise G1BuilderError("worktree HEAD does not match every candidate SHA")
    if actual["tree"] != expected["git_tree_oid"]:
        raise G1BuilderError("worktree HEAD tree does not match candidate tree OID")


def _validate_rust_toolchain(repo: Path) -> None:
    path = repo / "rust-toolchain.toml"
    try:
        with path.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise G1BuilderError(f"cannot read Rust toolchain pin {path}: {exc}") from exc
    if document.get("toolchain", {}).get("channel") != EXPECTED_RUST_TOOLCHAIN:
        raise G1BuilderError("Rust build is not bound to rust-toolchain.toml channel 1.97.1")


def _validate_toolchain_env(target: str, rocm_root: Path) -> None:
    expected_root = str(EXPECTED_ROCM_ROOT)
    expected_compiler = f"{expected_root}/bin/amdclang++"
    expected_values = {
        "ROCM_PATH": expected_root,
        "HIP_PATH": expected_root,
        "SLLM_HIP_COMPILER": expected_compiler,
        "CMAKE_HIP_ARCHITECTURES": target,
        "SLLM_HIP_CODEGEN_FEATURES": EXPECTED_CODEGEN_FEATURES,
        "SLLM_ENABLE_HIP_RUNTIME": "1",
        "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
    }
    if rocm_root != EXPECTED_ROCM_ROOT:
        raise G1BuilderError(f"G1 requires the canonical ROCm root {EXPECTED_ROCM_ROOT}")
    for name, expected in expected_values.items():
        actual = os.environ.get(name)
        if actual is not None and actual != expected:
            raise G1BuilderError(f"inherited {name} disagrees with the pinned G1 value")
    for name in (
        "CARGO_TARGET_DIR",
        "CARGO_BUILD_TARGET",
        "RUSTC_WRAPPER",
        "RUSTC_WORKSPACE_WRAPPER",
        "CARGO_BUILD_RUSTC",
        "CARGO_BUILD_RUSTC_WRAPPER",
        "RUSTFLAGS",
        "CARGO_ENCODED_RUSTFLAGS",
    ):
        if os.environ.get(name):
            raise G1BuilderError(f"inherited build override is not allowed: {name}")


def _schema_validate(document: Any, schema: dict[str, Any], label: str) -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:  # pragma: no cover - locked CI dependency
        raise G1BuilderError("jsonschema is required for G1 metadata validation") from exc
    try:
        Draft202012Validator.check_schema(schema)
        errors = sorted(
            Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
            key=lambda error: list(error.path),
        )
    except Exception as exc:  # jsonschema has multiple exception classes
        raise G1BuilderError(f"{label} schema could not be checked: {exc}") from exc
    if errors:
        detail = "; ".join(
            f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors[:8]
        )
        raise G1BuilderError(f"{label} schema validation failed: {detail}")


def validate_toolchain(
    repo: Path,
    rocm_root: Path = EXPECTED_ROCM_ROOT,
    *,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Validate the checked-in ROCm 7.14.0 tuple and its live root."""

    runner = runner or run_argv
    manifest_path = repo / "ci/toolchains/rocm-7.14.0.json"
    schema_path = repo / "ci/schema/rocm-toolchain-v1.schema.json"
    manifest = read_json(manifest_path)
    schema = read_json(schema_path)
    if not isinstance(manifest, dict) or not isinstance(schema, dict):
        raise G1BuilderError("ROCm toolchain manifest/schema must be JSON objects")
    _schema_validate(manifest, schema, "ROCm toolchain")
    if manifest.get("toolchain_id") != EXPECTED_TOOLCHAIN_ID:
        raise G1BuilderError("ROCm toolchain is not the pinned 7.14.0 identity")
    if manifest.get("rocm") != {"path": str(EXPECTED_ROCM_ROOT), "version": "7.14.0", "llvm_major": 23}:
        raise G1BuilderError("ROCm root/release/LLVM tuple is not canonical")
    compiler = manifest.get("compiler", {})
    paths = manifest.get("paths", {})
    if compiler.get("path") != f"{EXPECTED_ROCM_ROOT}/bin/amdclang++" or compiler.get("llvm_major") != 23:
        raise G1BuilderError("ROCm compiler path or LLVM major is not canonical")
    if compiler.get("name") != "amdclang++":
        raise G1BuilderError("ROCm compiler is not amdclang++")
    if paths.get("compiler") != compiler.get("path"):
        raise G1BuilderError("ROCm manifest compiler path is inconsistent")

    if rocm_root != EXPECTED_ROCM_ROOT:
        raise G1BuilderError("ROCm root argument is not exactly /opt/rocm")
    _reject_symlink_components(rocm_root, "ROCm root")
    try:
        canonical_root = rocm_root.resolve(strict=True)
    except OSError as exc:
        raise G1BuilderError(f"cannot resolve ROCm root {rocm_root}: {exc}") from exc
    if not canonical_root.is_dir():
        raise G1BuilderError("ROCm root is not a directory")

    required_paths = dict(paths)
    required_paths["rocm_root"] = str(rocm_root)
    resolved: dict[str, Path] = {}
    for name, value in required_paths.items():
        if not isinstance(value, str) or not value.startswith(f"{EXPECTED_ROCM_ROOT}/") and value != str(EXPECTED_ROCM_ROOT):
            raise G1BuilderError(f"ROCm path is outside the canonical root: {name}")
        path = Path(value)
        try:
            real = path.resolve(strict=True)
        except OSError as exc:
            raise G1BuilderError(f"ROCm path is missing: {name}={value}") from exc
        if not _path_is_within(real, canonical_root):
            raise G1BuilderError(f"ROCm path resolves outside the selected root: {name}={value}")
        if name != "rocm_root" and not path.exists():
            raise G1BuilderError(f"ROCm path is missing: {name}={value}")
        resolved[name] = real

    compiler_path = Path(compiler["path"])
    if compiler_path.name != "amdclang++" or not os.access(compiler_path, os.X_OK):
        raise G1BuilderError("ROCm compiler entry point is not executable amdclang++")
    version = _decode_output(
        runner([str(compiler_path), "--version"], timeout=60.0).stdout,
        "amdclang++ version",
    )
    if not LLVM23.search(version):
        raise G1BuilderError("ROCm compiler is not LLVM major 23")
    for name in ("clang_offload_bundler", "llvm_objcopy", "llvm_readobj", "llvm_objdump"):
        path = Path(paths[name])
        if not os.access(path, os.X_OK):
            raise G1BuilderError(f"ROCm LLVM tool is not executable: {name}")
        tool_version = _decode_output(
            runner([str(path), "--version"], timeout=60.0).stdout,
            f"{name} version",
        )
        # llvm-objcopy identifies itself only as a GNU-compatible frontend on
        # this ROCm installation; its canonical absolute path and the LLVM23
        # compiler/tool tuple are the version evidence for that entry point.
        if not LLVM23.search(tool_version) and not (
            name == "llvm_objcopy" and "objcopy" in tool_version.lower()
        ):
            raise G1BuilderError(f"ROCm LLVM tool is not major 23: {name}")

    version_files = (canonical_root / ".info/version", canonical_root / "core-7.14/.info/version")
    observed_versions = []
    for version_file in version_files:
        if version_file.is_file():
            try:
                observed_versions.append(version_file.read_text(encoding="utf-8").strip())
            except OSError as exc:
                raise G1BuilderError(f"cannot read ROCm release marker {version_file}: {exc}") from exc
    if "7.14.0" not in observed_versions:
        raise G1BuilderError("ROCm release marker does not prove version 7.14.0")
    return {
        "toolchain_id": manifest["toolchain_id"],
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(manifest)),
        "rocm_root": str(rocm_root),
        "compiler": str(compiler_path),
        "llvm_major": 23,
    }


def _output_paths(
    repo: Path,
    row_id: str,
    output_dir: Path | None,
) -> tuple[Path, Path, bool]:
    if row_id not in EXPECTED_ROWS:
        raise G1BuilderError(f"unknown G1 row: {row_id}")
    repo_real = repo.resolve(strict=True)
    created_root = False
    if output_dir is None:
        try:
            root = Path(tempfile.mkdtemp(prefix=OUTPUT_ROOT_PREFIX, dir=str(PRIVATE_TMP)))
        except OSError as exc:
            raise G1BuilderError(f"cannot create private G1 output root: {exc}") from exc
        created_root = True
        row_dir = root / row_id
    else:
        row_dir = Path(output_dir)
        if not row_dir.is_absolute() or row_dir.name != row_id:
            raise G1BuilderError("G1 output must be an absolute exact row directory")
        root = row_dir.parent
        if root.parent != PRIVATE_TMP or not root.name.startswith(OUTPUT_ROOT_PREFIX):
            raise G1BuilderError("G1 output must be below a private /tmp/sllm-g1-* root")
        _reject_symlink_components(root, "G1 output root")
        if root.exists():
            _require_private_directory(root, "G1 output root")
        else:
            _make_private_directory(root, "G1 output root")
        if row_dir.exists() or row_dir.is_symlink():
            raise G1BuilderError("G1 row output already exists; refusing stale output")
    _reject_symlink_components(root, "G1 output root")
    _require_private_directory(root, "G1 output root")
    if _path_is_within(root, repo_real) or _path_is_within(row_dir, repo_real):
        raise G1BuilderError("G1 output may not be inside the repository")
    _make_private_directory(row_dir, "G1 row output")
    return root, row_dir, created_root


def _build_environment(target: str, target_dir: Path) -> dict[str, str]:
    _validate_toolchain_env(target, EXPECTED_ROCM_ROOT)
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith("CARGO_FEATURE_"):
            environment.pop(name, None)
    environment.update(
        {
            "ROCM_PATH": str(EXPECTED_ROCM_ROOT),
            "HIP_PATH": str(EXPECTED_ROCM_ROOT),
            "SLLM_HIP_COMPILER": str(EXPECTED_ROCM_ROOT / "bin/amdclang++"),
            "CMAKE_HIP_ARCHITECTURES": target,
            "SLLM_HIP_CODEGEN_FEATURES": EXPECTED_CODEGEN_FEATURES,
            "SLLM_ENABLE_HIP_RUNTIME": "1",
            "SLLM_ENABLE_HIP_COMPILE_PROBE": "0",
            "CARGO_TARGET_DIR": str(target_dir),
        }
    )
    return environment


def _runtime_binary_is_real(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise G1BuilderError("Cargo did not produce a regular dedicated evidence executable")
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise G1BuilderError(f"cannot inspect dedicated evidence executable: {exc}") from exc
    if not mode & stat.S_IXUSR or path.stat().st_size == 0:
        raise G1BuilderError("dedicated evidence output is not a non-empty executable")


def _write_regular(path: Path, content: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise G1BuilderError(f"refusing to overwrite existing {label}: {path}")
    try:
        path.write_bytes(content)
        os.chmod(path, 0o600)
    except OSError as exc:
        raise G1BuilderError(f"cannot write {label} {path}: {exc}") from exc
    if path.is_symlink() or not path.is_file():
        raise G1BuilderError(f"written {label} is not a regular file: {path}")


def _write_sidecar(path: Path, target: Path, label: str) -> str:
    digest = _sha256_file(target)
    content = f"{digest}  {target.name}\n".encode("ascii")
    _write_regular(path, content, label)
    return _sha256_file(path)


def _metadata(
    repo: Path,
    row: Mapping[str, Any],
    candidate: Mapping[str, str],
    build_binary: Path,
    staged_binary: Path,
    observed: Mapping[str, Any],
    device_code_sha256: str,
) -> dict[str, Any]:
    manifest_hashes = g1_contracts._manifest_hashes(Path(repo))
    return {
        "schema_version": "g1-runtime-artifact-v1",
        "metadata_id": f"g1-runtime-artifact-{row['target']}",
        "row_id": row["row_id"],
        "target": row["target"],
        "candidate": {
            "reviewed_sha": candidate["reviewed_sha"],
            "tested_sha": candidate["tested_sha"],
            "workflow_sha": candidate["workflow_sha"],
            "git_tree_oid": candidate["git_tree_oid"],
            "worktree_clean": True,
            "revision_input": "full-sha",
        },
        "toolchain_id": EXPECTED_TOOLCHAIN_ID,
        "toolchain_manifest_sha256": manifest_hashes["toolchain_manifest_sha256"],
        "matrix_manifest_sha256": manifest_hashes["matrix_manifest_sha256"],
        "artifact_schema_sha256": manifest_hashes["artifact_schema_sha256"],
        "gpu": {"bdf": row["bdf"], "uuid": row["uuid"], "target": row["target"]},
        "artifact": {
            "path": str(build_binary),
            "size_bytes": build_binary.stat().st_size,
            "sha256": _sha256_file(build_binary),
            "sidecar_sha256": _sha256_file(build_binary.with_name(build_binary.name + SIDECAR_SUFFIX)),
            "kind": "dedicated-rust-evidence-binary",
        },
        "observed": dict(observed),
        "device_code_sha256": device_code_sha256,
        "scope": {
            "model_used": False,
            "cpu_fallback_allowed": False,
            "cpu_fallback_used": False,
            "binary_command": ["target/release/sllm-hip-evidence", "--timeout-ms", "1000"],
        },
    }


def build_runtime_artifact(
    *,
    repo: Path = ROOT,
    row_id: str,
    reviewed_sha: str | None = None,
    tested_sha: str | None = None,
    workflow_sha: str | None = None,
    tree_oid: str | None = None,
    output_dir: Path | None = None,
    rocm_root: Path = EXPECTED_ROCM_ROOT,
    timeout_seconds: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
    run_id: str = "g1-builder",
    run_attempt: int = 1,
    runner: CommandRunner | None = None,
) -> BuildResult:
    """Build one exact G1 row and return its private staged artifact paths."""

    runner = runner or run_argv
    if timeout_seconds <= 0 or timeout_seconds > MAX_BUILD_TIMEOUT_SECONDS:
        raise G1BuilderError("G1 build timeout must be positive and at most 900 seconds")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", run_id):
        raise G1BuilderError("G1 builder run_id is invalid")
    if isinstance(run_attempt, bool) or not isinstance(run_attempt, int) or run_attempt < 1:
        raise G1BuilderError("G1 builder run_attempt is invalid")
    repo = _repo_path(Path(repo))
    candidate = _validate_expected_hashes(reviewed_sha, tested_sha, workflow_sha, tree_oid)
    target = EXPECTED_TARGETS.get(row_id)
    if target is None:
        raise G1BuilderError(f"unknown G1 row: {row_id}")
    if Path(rocm_root) != EXPECTED_ROCM_ROOT:
        raise G1BuilderError("G1 requires ROCm root /opt/rocm")

    # These are static, checked-in contracts.  This validates the target/BDF/
    # UUID row before any compiler is started.
    matrix = g1_contracts.validate_g1_matrix(repo)
    row = g1_contracts.row_by_id(matrix, row_id)
    if row["target"] != target:
        raise G1BuilderError("G1 row target identity is inconsistent")
    _validate_rust_toolchain(repo)
    verify_candidate(repo, candidate, runner=runner)
    validate_toolchain(repo, Path(rocm_root), runner=runner)

    root, row_dir, created_root = _output_paths(repo, row_id, output_dir)
    target_dir = root / "target"
    try:
        _make_private_directory(target_dir, "Cargo target directory")
        build_binary = target_dir / "release" / BINARY_NAME
        if build_binary.parts[-3:] != ("target", "release", BINARY_NAME):
            raise G1BuilderError("Cargo output is not target/release/sllm-hip-evidence")
        environment = _build_environment(target, target_dir)
        command = (
            "cargo",
            f"+{EXPECTED_RUST_TOOLCHAIN}",
            "build",
            "--locked",
            "--offline",
            "--release",
            "--package",
            "sllm-hip",
            "--bin",
            BINARY_NAME,
        )
        # This is intentionally the only build invocation.  In particular,
        # it has no all-features or feature selector that could change the
        # runtime selection, and it does not consume any H3 output.
        runner(command, cwd=repo, env=environment, timeout=timeout_seconds)
        _runtime_binary_is_real(build_binary)

        # Recheck the immutable checkout after Cargo returns.  A concurrent
        # mutation must never be turned into a candidate-bound artifact.
        verify_candidate(repo, candidate, runner=runner)
        staged_binary = row_dir / BINARY_NAME
        try:
            shutil.copyfile(build_binary, staged_binary, follow_symlinks=False)
            os.chmod(staged_binary, 0o700)
        except OSError as exc:
            raise G1BuilderError(f"cannot stage the dedicated runtime executable: {exc}") from exc
        _runtime_binary_is_real(staged_binary)
        try:
            inspection = g1_contracts.inspect_g1_runtime_artifact(
                staged_binary,
                target,
                tool_runner=runner,
            )
        except (ContractError, OSError, TypeError, ValueError) as exc:
            raise G1BuilderError(f"G1 embedded HIP code-object inspection failed: {exc}") from exc
        # Bind metadata to the source executable and its own sidecar before
        # the source is copied into the row directory.  Both paths remain in
        # the private staging root and are independently hash-checked.
        _write_sidecar(
            build_binary.with_name(build_binary.name + SIDECAR_SUFFIX),
            build_binary,
            "G1 source artifact sidecar",
        )
        artifact_sidecar = staged_binary.with_name(staged_binary.name + SIDECAR_SUFFIX)
        _write_sidecar(artifact_sidecar, staged_binary, "G1 artifact sidecar")

        metadata_path = row_dir / METADATA_NAME
        metadata = _metadata(
            repo,
            row,
            candidate,
            build_binary,
            staged_binary,
            inspection["observed"],
            inspection["device_code_sha256"],
        )
        metadata_bytes = _canonical_json_bytes(metadata)
        _write_regular(metadata_path, metadata_bytes, "G1 runtime metadata")
        metadata_sidecar = metadata_path.with_name(metadata_path.name + SIDECAR_SUFFIX)
        _write_sidecar(metadata_sidecar, metadata_path, "G1 metadata sidecar")

        validation_identity = {
            "run_id": run_id,
            "run_attempt": run_attempt,
            "reviewed_sha": candidate["reviewed_sha"],
            "tested_sha": candidate["tested_sha"],
            "workflow_sha": candidate["workflow_sha"],
            "git_tree_oid": candidate["git_tree_oid"],
        }
        try:
            g1_contracts.validate_artifact_metadata(
                metadata,
                artifact_path=staged_binary,
                metadata_path=metadata_path,
                expected=row,
                identity=validation_identity,
                repo=repo,
                tool_runner=runner,
            )
        except (ContractError, OSError, TypeError, ValueError) as exc:
            raise G1BuilderError(f"G1 runtime metadata validation failed: {exc}") from exc
        expected_files = {
            BINARY_NAME,
            BINARY_NAME + SIDECAR_SUFFIX,
            METADATA_NAME,
            METADATA_NAME + SIDECAR_SUFFIX,
        }
        if {path.name for path in row_dir.iterdir()} != expected_files:
            raise G1BuilderError("G1 row directory contains an unexpected staged file")
        return BuildResult(
            row_id=row_id,
            target=target,
            output_dir=row_dir,
            artifact_path=staged_binary,
            metadata_path=metadata_path,
            artifact_sha256=metadata["artifact"]["sha256"],
            metadata_sha256=_sha256_file(metadata_path),
            command=command,
        )
    except Exception:
        # The output is newly owned by this invocation.  Remove only its
        # private row/target paths; never touch a repository or a caller's
        # unrelated directory.
        shutil.rmtree(row_dir, ignore_errors=True)
        shutil.rmtree(target_dir, ignore_errors=True)
        if created_root:
            try:
                root.rmdir()
            except OSError:
                pass
        raise


# Descriptive aliases keep the callable useful to small local harnesses while
# retaining one implementation and one security boundary.
build_g1_runtime = build_runtime_artifact
build_artifact = build_runtime_artifact


def _candidate_args(args: argparse.Namespace) -> tuple[str, str, str, str]:
    separate = [args.reviewed_sha, args.tested_sha, args.workflow_sha]
    if args.candidate_sha is not None:
        if any(value is not None and value != args.candidate_sha for value in separate):
            raise G1BuilderError("--candidate-sha disagrees with a separate candidate SHA")
        separate = [args.candidate_sha] * 3
    if any(value is None for value in separate) or args.tree_oid is None:
        raise G1BuilderError("a full candidate SHA and tree OID are required")
    return separate[0], separate[1], separate[2], args.tree_oid


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--repo", type=Path, default=ROOT)
    result.add_argument("--row", choices=EXPECTED_ROWS, required=True)
    result.add_argument("--candidate-sha")
    result.add_argument("--reviewed-sha", "--expected-reviewed-sha", dest="reviewed_sha")
    result.add_argument("--tested-sha", "--expected-tested-sha", dest="tested_sha")
    result.add_argument("--workflow-sha", "--expected-workflow-sha", dest="workflow_sha")
    result.add_argument(
        "--tree-oid", "--git-tree-oid", "--expected-tree-oid", dest="tree_oid"
    )
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--rocm-root", type=Path, default=EXPECTED_ROCM_ROOT)
    result.add_argument("--timeout-seconds", type=float, default=DEFAULT_BUILD_TIMEOUT_SECONDS)
    result.add_argument("--run-id", default="g1-builder")
    result.add_argument("--run-attempt", type=int, default=1)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        reviewed_sha, tested_sha, workflow_sha, tree_oid = _candidate_args(args)
        result = build_runtime_artifact(
            repo=args.repo,
            row_id=args.row,
            reviewed_sha=reviewed_sha,
            tested_sha=tested_sha,
            workflow_sha=workflow_sha,
            tree_oid=tree_oid,
            output_dir=args.output_dir,
            rocm_root=args.rocm_root,
            timeout_seconds=args.timeout_seconds,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
    except (G1BuilderError, ContractError, OSError, TypeError, ValueError) as exc:
        print(f"G1 runtime artifact: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "G1 runtime artifact: PASS "
        f"row={result.row_id} target={result.target} output={result.output_dir} "
        f"artifact_sha256={result.artifact_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
