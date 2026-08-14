#!/usr/bin/env python3
"""Create the immutable build identity for the current dirty candidate.

The semantic tree is written from a temporary Git index populated from the
selected immutable base revision plus every nonignored candidate path.  The
repository's real index is never staged or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
from engine_performance_common import (  # noqa: E402
    BUILD_CONFIGURATION_KEYS,
    validate_build_configuration,
)


VERSION = "sllm-build-identity-v2"
SHA40 = set("0123456789abcdef")


def _fail(message: str) -> None:
    raise ContractError(message)


def _git(repo: Path, args: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, check=False,
            text=True, env=env, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"git command failed: {exc}")
    if check and completed.returncode != 0:
        _fail(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _commit(repo: Path, value: str | None) -> str:
    revision = _git(repo, ["rev-parse", "HEAD"] if value is None else ["rev-parse", f"{value}^{{commit}}"])
    if len(revision) != 40 or any(char not in SHA40 for char in revision):
        _fail("source base revision is not an immutable commit")
    if _git(repo, ["cat-file", "-t", revision]) != "commit":
        _fail("source base revision is not a commit object")
    head = _git(repo, ["rev-parse", "HEAD"])
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, head],
            capture_output=True, check=False, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"cannot validate source base ancestry: {exc}")
    if completed.returncode != 0:
        _fail("source base revision is not an ancestor/base identity of HEAD")
    return revision


def _semantic_tree(repo: Path, base_revision: str) -> str:
    fd, index_name = tempfile.mkstemp(prefix="sllm-build-index-")
    os.close(fd)
    index_path = Path(index_name)
    try:
        index_path.unlink()
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = str(index_path)
        _git(repo, ["read-tree", base_revision], env=env)
        _git(repo, ["add", "-A", "--", "."], env=env)
        tree = _git(repo, ["write-tree"], env=env)
    finally:
        for path in (index_path, Path(str(index_path) + ".lock")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if len(tree) != 40 or any(char not in SHA40 for char in tree) or _git(repo, ["cat-file", "-t", tree]) != "tree":
        _fail("temporary-index semantic tree is not a Git tree object")
    return tree


def _candidate_paths(repo: Path, explicit: Iterable[str], excluded: set[str]) -> list[Path]:
    if explicit:
        values = list(explicit)
    else:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                capture_output=True, check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            _fail(f"cannot enumerate candidate source files: {exc}")
        if completed.returncode != 0:
            _fail("cannot enumerate candidate source files")
        values = [item for item in completed.stdout.decode("utf-8").split("\0") if item]
    paths: list[Path] = []
    seen: set[str] = set()
    root = repo.resolve()
    for value in values:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            _fail(f"build input path is unsafe: {value}")
        relative = path.as_posix()
        if relative in excluded or relative in seen:
            continue
        candidate = (repo / path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            _fail(f"build input is not a regular file below source root: {value}")
        seen.add(relative)
        paths.append(candidate)
    paths.sort(key=lambda item: item.relative_to(repo).as_posix())
    return paths


def _parse_build_configuration(records: Iterable[str], target: str) -> dict[str, str]:
    configuration: dict[str, str] = {}
    allowed = set(BUILD_CONFIGURATION_KEYS)
    for record in records:
        if "=" not in record:
            _fail(f"build configuration record must be KEY=VALUE: {record}")
        key, value = record.split("=", 1)
        if key not in allowed:
            _fail(f"unknown build configuration key: {key}")
        if key in configuration:
            _fail(f"duplicate build configuration key: {key}")
        if not value:
            _fail(f"build configuration value must be nonempty: {key}")
        if value != value.strip() or any(ord(char) < 0x20 or ord(char) == 0x7f for char in value):
            _fail(f"build configuration value is unsafe: {key}")
        configuration[key] = value
    missing = [key for key in BUILD_CONFIGURATION_KEYS if key not in configuration]
    if missing:
        _fail("missing build configuration keys: " + ", ".join(missing))
    return validate_build_configuration(configuration, target)


def _build_inputs_digest(
    repo: Path,
    paths: Iterable[Path],
    build_configuration: dict[str, str],
) -> str:
    records = []
    for path in paths:
        data = path.read_bytes()
        records.append({
            "path": path.relative_to(repo).as_posix(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    if not records:
        _fail("build identity has no build input files")
    payload = {
        "schema_version": VERSION,
        "source_files": records,
        "build_configuration": build_configuration,
    }
    return "sha256:" + hashlib.sha256(canonical_bytes(payload)).hexdigest()


def create_identity(
    source_root: Path,
    output: Path,
    binary: Path,
    target: str,
    backend: str,
    rocm_release: str,
    rocm_root: str,
    source_base_revision: str | None,
    build_inputs: list[str],
    build_configs: list[str],
) -> dict[str, object]:
    build_configuration = _parse_build_configuration(build_configs, target)
    repo = source_root.resolve()
    if _git(repo, ["rev-parse", "--show-toplevel"]) != str(repo):
        _fail("source root is not the Git worktree root")
    revision = _commit(repo, source_base_revision)
    tree = _semantic_tree(repo, revision)
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        _fail("build binary must be an executable regular file")
    binary = binary.resolve(strict=True)
    excluded = {output.resolve().relative_to(repo).as_posix()} if output.resolve().is_relative_to(repo) else set()
    if binary.is_relative_to(repo):
        excluded.add(binary.relative_to(repo).as_posix())
    inputs = _candidate_paths(repo, build_inputs, excluded)
    document = {
        "schema_version": VERSION,
        "source_root": str(repo),
        "source_base_revision": revision,
        "semantic_tree": tree,
        "build_inputs_digest": _build_inputs_digest(repo, inputs, build_configuration),
        "build_configuration": build_configuration,
        "target": target,
        "backend": backend,
        "rocm_release": rocm_release,
        "rocm_root": rocm_root,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }
    output = output.resolve()
    if output.exists() or output.is_symlink():
        _fail(f"refusing to overwrite build identity manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(document))
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--target", choices=("gfx1030", "gfx1201"), required=True)
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--rocm-release", default="7.14.0")
    parser.add_argument("--rocm-root", default="/opt/rocm/core-7.14")
    parser.add_argument("--source-base-revision")
    parser.add_argument("--build-input", action="append", default=[])
    parser.add_argument(
        "--build-config", action="append", required=True, metavar="KEY=VALUE",
        help="repeatable; every exact Phase 5 build configuration key is required",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        create_identity(
            args.source_root, args.output, args.binary, args.target, args.backend,
            args.rocm_release, args.rocm_root, args.source_base_revision, args.build_input,
            args.build_config,
        )
    except (ContractError, OSError, ValueError) as exc:
        print(f"engine build identity: FAIL: {exc}", file=__import__("sys").stderr)
        return 1
    print(f"engine build identity: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
