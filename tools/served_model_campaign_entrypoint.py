#!/usr/bin/env python3
"""Protected entrypoint boundary for cross-model campaign control wrappers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

import served_model_campaign_source_seal as source_seal


PRODUCTION_PYTHON = Path("/usr/bin/python3.12")
PRODUCTION_WRAPPERS = frozenset(
    {
        "claim-served-model-v2-cross-model-campaign-authorization.py",
        "issue-served-model-v2-cross-model-campaign-authorization.py",
        "recover-served-model-v2-cross-model-campaign.py",
        "run-served-model-v2-cross-model-campaign.py",
    }
)
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_GIT_OUTPUT_BYTES = 4 * 1024 * 1024


class ProductionEntrypointError(RuntimeError):
    """The campaign wrapper was not entered through its protected boundary."""


def fail(message: str) -> NoReturn:
    raise ProductionEntrypointError(message)


def _git(root: Path, arguments: tuple[str, ...], label: str) -> bytes:
    try:
        completed = subprocess.run(
            source_seal.git_argv(list(arguments)),
            cwd=root,
            env=source_seal.git_environment(),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProductionEntrypointError(f"{label} failed") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(completed.stdout) > MAX_GIT_OUTPUT_BYTES
    ):
        fail(f"{label} differs")
    return completed.stdout


def _git_identity(root: Path) -> tuple[str, str]:
    expected_top = os.fsencode(root) + b"\n"
    if (
        _git(root, ("rev-parse", "--show-toplevel"), "source Git top-level")
        != expected_top
    ):
        fail("source Git top-level differs")
    commit_raw = _git(
        root,
        ("rev-parse", "--verify", "HEAD^{commit}"),
        "source Git commit",
    )
    tree_raw = _git(
        root,
        ("rev-parse", "--verify", "HEAD^{tree}"),
        "source Git tree",
    )
    if _git(
        root,
        (
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=all",
            "--no-renames",
        ),
        "source Git status",
    ):
        fail("campaign wrapper source worktree is not clean")
    try:
        commit = commit_raw.decode("ascii").strip()
        tree = tree_raw.decode("ascii").strip()
    except UnicodeError as error:
        raise ProductionEntrypointError(
            "campaign wrapper source identity is invalid"
        ) from error
    if (
        GIT_OBJECT_RE.fullmatch(commit) is None
        or GIT_OBJECT_RE.fullmatch(tree) is None
    ):
        fail("campaign wrapper source identity differs")
    return commit, tree


def require_production_entrypoint(wrapper_path: Path) -> None:
    """Require the exact interpreter invocation and a clean root-owned seal."""

    root = Path(__file__).resolve().parents[1]
    wrapper = Path(wrapper_path)
    expected_wrapper = root / "tools" / wrapper.name
    original_argv = getattr(sys, "orig_argv", None)
    expected_prefix = [
        os.fspath(PRODUCTION_PYTHON),
        "-I",
        "-S",
        "-B",
        os.fspath(expected_wrapper),
    ]
    if (
        os.geteuid() != 0
        or not wrapper.is_absolute()
        or Path(os.path.abspath(wrapper)) != wrapper
        or wrapper.resolve(strict=True) != wrapper
        or wrapper.name not in PRODUCTION_WRAPPERS
        or wrapper != expected_wrapper
        or not isinstance(original_argv, list)
        or original_argv[:5] != expected_prefix
        or not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode
        or not sys.flags.safe_path
    ):
        fail(
            "campaign wrapper requires root and exact "
            "/usr/bin/python3.12 -I -S -B absolute invocation"
        )
    try:
        initial = source_seal.capture_source_seal(root, required_uid=0)
    except source_seal.SourceSealError as error:
        raise ProductionEntrypointError(
            "campaign wrapper source is not protected"
        ) from error
    first_identity = _git_identity(root)
    try:
        source_seal.require_source_seal(initial, required_uid=0)
    except source_seal.SourceSealError as error:
        raise ProductionEntrypointError(
            "campaign wrapper source changed during entry"
        ) from error
    if _git_identity(root) != first_identity:
        fail("campaign wrapper source identity changed during entry")
    try:
        source_seal.require_source_seal(initial, required_uid=0)
    except source_seal.SourceSealError as error:
        raise ProductionEntrypointError(
            "campaign wrapper source changed across Git repin"
        ) from error


__all__ = [
    "PRODUCTION_PYTHON",
    "PRODUCTION_WRAPPERS",
    "ProductionEntrypointError",
    "require_production_entrypoint",
]
