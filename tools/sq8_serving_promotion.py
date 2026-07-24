#!/usr/bin/env python3
"""Strict SQ8_0 served-model promotion evidence and receipt primitives.

This module deliberately concerns the independently served Qwen3-14B-FP8
``SQ8_0`` worker.  It does not implement, import, or accept the historical
AQ4_0 SQ8-overlay authorization schemas.
"""

from __future__ import annotations

import ctypes
import errno
import base64
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, NoReturn, Sequence


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_VALIDATOR_PATH = ROOT / "tools/validate-sq8-product-promotion.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
GENERATOR_PATH = ROOT / "tools/generate-served-model.py"

EVIDENCE_SCHEMA = "ullm.sq8_serving_promotion_evidence.v1"
RECEIPT_SCHEMA = "ullm.sq8_serving_promotion.v1"
EPHEMERAL_RECEIPT_SCHEMA = (
    "ullm.sq8_serving_promotion_ephemeral_receipt.v1"
)
BUILD_RECEIPT_SCHEMA_V1 = "ullm.sq8_worker_build_receipt.v1"
BUILD_RECEIPT_SCHEMA = "ullm.sq8_worker_build_receipt.v2"
BUILD_PROVENANCE_SCHEMA_V1 = "ullm.sq8_worker_build_provenance.v1"
BUILD_PROVENANCE_SCHEMA = "ullm.sq8_worker_build_provenance.v2"
BUILD_RELEASE_SEAL_SCHEMA_V1 = "ullm.sq8_worker_release_seal.v1"
BUILD_RELEASE_SEAL_SCHEMA = "ullm.sq8_worker_release_seal.v2"
BUILD_WORKER_RELATIVE_PATH = "ullm-sq8-worker"
BUILD_RELEASE_MEMBERS = frozenset(
    {
        "README.md",
        "SHA256SUMS",
        "SEALED.json",
        "build-provenance.json",
        "build-receipt.json",
        BUILD_WORKER_RELATIVE_PATH,
    }
)
BUILD_SUMMED_MEMBERS = (
    "README.md",
    "build-provenance.json",
    "build-receipt.json",
    BUILD_WORKER_RELATIVE_PATH,
)
CPU_CASES_SCHEMA = "ullm.sq8_serving_promotion_cpu_cases.v1"
PRODUCT_SCHEMA = "ullm.sq8_product_promotion.v1"
WORKER_PROTOCOL = "ullm.worker.v2"
SERVED_MODEL_SCHEMA = "ullm.served_model.v2"
FORMAT_ID = "SQ8_0"
IMPLEMENTATION_ID = "qwen3_sq8_rdna4_v1"
MODEL_ID = "ullm-qwen3-14b-sq8"
UPSTREAM_MODEL_ID = "Qwen/Qwen3-14B-FP8"
UPSTREAM_MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
REASONING_DIALECT = "qwen3-thinking-v1"
REASONING_CONTRACT = {
    "enabled_by_default": False,
    "dialect_id": REASONING_DIALECT,
    "start_token_ids": [151667],
    "end_token_ids": [151668],
    "forced_end_token_ids": [151668],
    "initial_phase": "reasoning",
    "eos_policy": "close",
    "effort_budgets": {"low": 32, "medium": 128, "high": 256},
    "max_budget_tokens": 256,
    "reserved_answer_tokens": 1,
    "history_reasoning_policy": "omit",
}

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_HASH_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TEST_OUTPUT_BYTES = 4 * 1024 * 1024
COPY_CHUNK_BYTES = 1024 * 1024
RENAME_NOREPLACE = 1
AT_FDCWD = -100

HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_OBJECT_RE = re.compile(r"[0-9a-f]{40}\Z")
SOURCE_DATE_EPOCH_RE = re.compile(r"[0-9]{1,20}\Z")

REQUIRED_BUILD_INPUTS = frozenset(
    {
        ".cargo/config.toml",
        "Cargo.lock",
        "crates/ullm-engine/Cargo.toml",
        "crates/ullm-engine/src/bin/ullm-sq8-worker.rs",
        "crates/ullm-engine/src/served_model.rs",
        "crates/ullm-engine/src/sq8_serving_runtime.rs",
        "crates/ullm-engine/src/sq8_worker_backend.rs",
        "crates/ullm-engine/src/sq8_worker_protocol.rs",
        "crates/ullm-engine/src/sq8_worker_runtime.rs",
        "crates/ullm-runtime-sys/build.rs",
    }
)
BUILD_INPUTS_V2 = (
    ".cargo/config.toml",
    "Cargo.lock",
    "Cargo.toml",
    "crates/ullm-engine/Cargo.toml",
    "crates/ullm-engine/src/bin/ullm-sq8-worker.rs",
    "crates/ullm-engine/src/reasoning.rs",
    "crates/ullm-engine/src/served_model.rs",
    "crates/ullm-engine/src/sq8_sampling.rs",
    "crates/ullm-engine/src/sq8_serving_runtime.rs",
    "crates/ullm-engine/src/sq8_worker_backend.rs",
    "crates/ullm-engine/src/sq8_worker_protocol.rs",
    "crates/ullm-engine/src/sq8_worker_runtime.rs",
    "crates/ullm-runtime-sys/Cargo.toml",
    "crates/ullm-runtime-sys/build.rs",
)
BUILD_REJECTED_ENVIRONMENT_V2 = (
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
    "CARGO_PROFILE_RELEASE_DEBUG",
    "CARGO_PROFILE_RELEASE_LTO",
    "CARGO_PROFILE_RELEASE_OPT_LEVEL",
    "CARGO_PROFILE_RELEASE_PANIC",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTDOCFLAGS",
    "RUSTFLAGS",
)

EVIDENCE_SOURCE_PATHS = tuple(
    sorted(
        {
            "services/openai-gateway/pyproject.toml",
            "services/openai-gateway/src/ullm_openai_gateway/reasoning.py",
            "services/openai-gateway/src/ullm_openai_gateway/app.py",
            "services/openai-gateway/src/ullm_openai_gateway/served_model.py",
            "services/openai-gateway/tests/test_app.py",
            "services/openai-gateway/uv.lock",
            "tools/build-sq8-worker-release.py",
            "tools/generate-served-model.py",
            "tools/prepare-sq8-serving-promotion-ephemeral.py",
            "tools/run-sq8-serving-promotion-cpu-cases.py",
            "tools/run-sq8-serving-promotion-evidence.py",
            "tools/sq8_canonical_artifact.py",
            "tools/sq8_serving_promotion.py",
            "tools/validate-served-model.py",
            "tools/validate-sq8-product-promotion.py",
            "tools/validate-sq8-serving-promotion-evidence.py",
            "tools/write-sq8-serving-promotion-receipt.py",
        },
        key=lambda value: value.encode("utf-8"),
    )
)

CPU_CASE_IDS = (
    "protocol-v2-generate",
    "protocol-v2-cancel",
    "protocol-v2-shutdown",
    "reject-v1-generate",
    "reject-v1-cancel",
    "reject-v1-shutdown",
    "reasoning-disabled",
    "reasoning-budget-zero",
    "reasoning-low-32",
    "reasoning-medium-128",
    "reasoning-high-256",
    "reasoning-unbounded-natural-close",
    "reasoning-budget-forced-close",
    "reasoning-eos-forced-close",
    "reasoning-answer-reservation",
    "reasoning-natural-accounting",
    "forced-token-rng-unconsumed",
    "cancel-rollback",
    "publish-failure-rollback",
    "reset-accounting",
    "release-usage-reconcile",
)

CPU_RUST_TESTS = (
    (
        "rust-command-schema",
        "sq8_worker_protocol::tests::"
        "loaded_profile_requires_exact_command_schema_for_every_command_kind",
    ),
    (
        "rust-profile-bijection",
        "sq8_worker_protocol::tests::"
        "worker_profile_schema_and_reasoning_presence_are_bijective",
    ),
    (
        "rust-explicit-v2-reasoning",
        "sq8_worker_protocol::tests::"
        "v2_generate_requires_explicit_reasoning_even_when_disabled",
    ),
    (
        "rust-pre-busy-schema",
        "sq8_worker_runtime::tests::"
        "wrong_schema_is_rejected_before_busy_or_control_dispatch",
    ),
    (
        "rust-reasoning-contract",
        "sq8_serving_runtime::tests::"
        "qwen3_reasoning_contract_fixes_effort_budgets_and_token_ids",
    ),
    (
        "rust-request-dialect-binding",
        "sq8_serving_runtime::tests::"
        "reasoning_request_and_loaded_dialect_must_be_present_together",
    ),
    (
        "rust-disabled-accounting",
        "sq8_serving_runtime::tests::"
        "reasoning_disabled_retains_v2_zero_usage_accounting",
    ),
    (
        "rust-bounded-budgets",
        "sq8_serving_runtime::tests::"
        "bounded_reasoning_budgets_close_exactly_at_zero_low_medium_and_high",
    ),
    (
        "rust-unbounded-natural-close",
        "sq8_serving_runtime::tests::"
        "unbounded_reasoning_natural_close_is_sampled_and_not_forced",
    ),
    (
        "rust-answer-reservation",
        "sq8_serving_runtime::tests::"
        "unbounded_reasoning_reserves_forced_close_and_one_answer_token",
    ),
    (
        "rust-eos-rng",
        "sq8_serving_runtime::tests::"
        "reasoning_eos_is_replaced_by_forced_close_without_consuming_rng",
    ),
    (
        "rust-transaction-rollback",
        "sq8_serving_runtime::tests::"
        "reasoning_cancel_and_publication_failure_leave_committed_accounting_unchanged",
    ),
    (
        "rust-release-usage",
        "sq8_serving_runtime::tests::"
        "reasoning_release_summary_keeps_committed_usage_after_active_state_is_cleared",
    ),
    (
        "rust-reset-accounting",
        "sq8_serving_runtime::tests::"
        "reasoning_reuse_starts_with_zero_accounting_after_prior_release",
    ),
)

CPU_PYTEST_TESTS = (
    (
        "gateway-eos-reconcile",
        "services/openai-gateway/tests/test_app.py::"
        "test_stop_reasoning_eos_replacement_is_reconciled_without_counting_sampled_eos",
    ),
    (
        "gateway-length-reconcile",
        "services/openai-gateway/tests/test_app.py::"
        "test_length_reasoning_forced_end_token_is_reconciled",
    ),
)


class PromotionError(RuntimeError):
    """An SQ8 serving-promotion input or publication is invalid."""


def fail(message: str) -> NoReturn:
    raise PromotionError(message)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} fields differ")
    return value


def _hash(value: Any, label: str) -> str:
    if type(value) is not str or HASH_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def _git_object(value: Any, label: str) -> str:
    if type(value) is not str or GIT_OBJECT_RE.fullmatch(value) is None:
        fail(f"{label} is not a full lowercase Git object ID")
    return value


def _text(value: Any, label: str, maximum: int = 4096) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        fail(f"{label} is invalid")
    return value


def _integer(
    value: Any, label: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        fail(f"{label} is invalid")
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise PromotionError("document is not canonicalizable JSON") from error


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            fail("JSON contains duplicate fields")
        value[key] = child
    return value


def _reject_constant(_value: str) -> None:
    fail("JSON contains a non-finite number")


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except PromotionError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"{label} is not strict JSON") from error
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _reject_symlink_components(
    path: Path, label: str, *, leaf_may_absent: bool = False
) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if leaf_may_absent and index == len(absolute.parts[1:]) - 1:
                return
            fail(f"{label} has an absent path component")
        except OSError as error:
            raise PromotionError(f"failed to inspect {label}") from error
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"{label} traverses a symlink")


def _canonical_absolute(path: Path, label: str, *, may_be_absent: bool = False) -> Path:
    if not path.is_absolute():
        fail(f"{label} is not absolute")
    _reject_symlink_components(path, label, leaf_may_absent=may_be_absent)
    if may_be_absent:
        resolved = path.parent.resolve(strict=True) / path.name
    else:
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise PromotionError(f"{label} is unavailable") from error
    if os.fspath(path) != os.fspath(resolved):
        fail(f"{label} is not canonical")
    return resolved


def stable_read(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_JSON_BYTES,
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> bytes:
    path = _canonical_absolute(path, label)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for serving-promotion validation")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PromotionError(f"failed to open {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 1
            or before.st_size > maximum
            or (
                required_mode is not None
                and stat.S_IMODE(before.st_mode) != required_mode
            )
            or (required_nlink is not None and before.st_nlink != required_nlink)
        ):
            fail(f"{label} file identity differs")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(COPY_CHUNK_BYTES, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as error:
            raise PromotionError(f"{label} disappeared while being read") from error
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(named)
        ):
            fail(f"{label} changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def stable_hash(
    path: Path,
    label: str,
    *,
    maximum: int = MAX_HASH_FILE_BYTES,
    required_mode: int | None = None,
    required_nlink: int | None = None,
) -> tuple[int, str]:
    raw = stable_read(
        path,
        label,
        maximum=maximum,
        required_mode=required_mode,
        required_nlink=required_nlink,
    )
    return len(raw), hashlib.sha256(raw).hexdigest()


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    text = _text(value, label, 1024)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or text.startswith("./")
        or "\\" in text
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        fail(f"{label} is not a safe relative path")
    return relative


def _safe_repo_file(root: Path, relative_value: Any, label: str) -> Path:
    relative = _safe_relative(relative_value, label)
    candidate = root.joinpath(*relative.parts)
    _reject_symlink_components(candidate, label)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        fail(f"{label} escapes the source repository")
    return resolved


def _load_json_file(
    path: Path,
    label: str,
    *,
    canonical: bool = False,
    required_mode: int | None = None,
    required_nlink: int | None = None,
    maximum: int = MAX_JSON_BYTES,
) -> tuple[dict[str, Any], bytes]:
    raw = stable_read(
        path,
        label,
        maximum=maximum,
        required_mode=required_mode,
        required_nlink=required_nlink,
    )
    value = _strict_json(raw, label)
    if canonical and raw != _canonical_json(value):
        fail(f"{label} is not canonical JSON")
    return value, raw


def _git(
    root: Path, arguments: Sequence[str], label: str, *, allow_nonzero: bool = False
) -> tuple[int, bytes]:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *arguments],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PromotionError(f"failed to inspect {label}") from error
    if result.returncode != 0 and not allow_nonzero:
        fail(f"{label} inspection failed")
    return result.returncode, result.stdout


def _audit_absolute_path(value: Any, label: str) -> Path:
    """Validate an absolute lexical audit path without dereferencing it."""

    text = _text(value, label, 4096)
    selected = PurePosixPath(text)
    if (
        not selected.is_absolute()
        or text.startswith("//")
        or selected.as_posix() != text
        or any(part in {"", ".", ".."} for part in selected.parts[1:])
    ):
        fail(f"{label} is not a canonical absolute audit path")
    return Path(text)


def _validate_clean_detached_source(
    source: dict[str, Any],
    *,
    verify_live: bool,
    live_root: Path | None = None,
    relocatable: bool = False,
) -> Path:
    source = _exact(
        source,
        {
            "repository_root",
            "commit",
            "tree",
            "detached",
            "worktree_clean",
            "status_sha256",
        },
        "worker build source",
    )
    audit_root = (
        _audit_absolute_path(
            source["repository_root"], "worker build repository root"
        )
        if relocatable
        else _canonical_absolute(
            Path(_text(source["repository_root"], "worker build repository root")),
            "worker build repository root",
        )
    )
    commit = _git_object(source["commit"], "worker build source commit")
    tree = _git_object(source["tree"], "worker build source tree")
    if (
        source["detached"] is not True
        or source["worktree_clean"] is not True
        or source["status_sha256"] != hashlib.sha256(b"").hexdigest()
    ):
        fail("worker build source is not a clean detached checkout")
    if relocatable:
        if live_root is None:
            if verify_live:
                fail("v2 worker build validation requires an explicit source root")
            root = audit_root
        else:
            root = _canonical_absolute(live_root, "current worker build source root")
    else:
        if live_root is not None:
            requested = _canonical_absolute(
                live_root, "requested v1 worker build source root"
            )
            if requested != audit_root:
                fail("v1 worker build source root override differs")
        root = audit_root
    if verify_live:
        if relocatable:
            _, observed_top = _git(
                root, ["rev-parse", "--show-toplevel"], "source repository root"
            )
            if (
                observed_top.strip().decode("utf-8", errors="strict")
                != os.fspath(root)
            ):
                fail("current worker build source is not its Git top-level")
        _, observed_commit = _git(root, ["rev-parse", "HEAD"], "source commit")
        _, observed_tree = _git(root, ["rev-parse", "HEAD^{tree}"], "source tree")
        branch_status, _ = _git(
            root,
            ["symbolic-ref", "-q", "HEAD"],
            "detached HEAD",
            allow_nonzero=True,
        )
        _, status = _git(
            root,
            ["status", "--porcelain=v1", "--untracked-files=all"],
            "source status",
        )
        if (
            observed_commit.strip().decode("ascii", errors="strict") != commit
            or observed_tree.strip().decode("ascii", errors="strict") != tree
            or branch_status == 0
            or status != b""
        ):
            fail("worker build source live checkout differs")
    return root


def _validate_build_environment(value: Any) -> dict[str, Any]:
    environment = _exact(
        value,
        {
            "CARGO_BUILD_JOBS",
            "CARGO_INCREMENTAL",
            "CARGO_TARGET_DIR",
            "CUDA_VISIBLE_DEVICES",
            "GPU_ARCH",
            "HIP_VISIBLE_DEVICES",
            "ROCM_PATH",
            "ROCR_VISIBLE_DEVICES",
            "RUSTC_WRAPPER",
            "SOURCE_DATE_EPOCH",
            "ULLM_HIP_VISIBLE_DEVICES",
        },
        "worker build environment",
    )
    jobs = _text(environment["CARGO_BUILD_JOBS"], "CARGO_BUILD_JOBS", 16)
    if not jobs.isdecimal() or int(jobs) < 1:
        fail("CARGO_BUILD_JOBS is invalid")
    if (
        environment["CARGO_INCREMENTAL"] != "0"
        or environment["GPU_ARCH"] != "gfx1201"
        or environment["CUDA_VISIBLE_DEVICES"] != "-1"
        or environment["HIP_VISIBLE_DEVICES"] != "-1"
        or environment["ROCR_VISIBLE_DEVICES"] != "-1"
        or environment["ULLM_HIP_VISIBLE_DEVICES"] != "-1"
        or environment["RUSTC_WRAPPER"] is not None
    ):
        fail("worker build isolation environment differs")
    for name in ("CARGO_TARGET_DIR", "ROCM_PATH"):
        path = Path(_text(environment[name], name))
        if not path.is_absolute():
            fail(f"{name} is not absolute")
    epoch = _text(environment["SOURCE_DATE_EPOCH"], "SOURCE_DATE_EPOCH", 20)
    if SOURCE_DATE_EPOCH_RE.fullmatch(epoch) is None:
        fail("SOURCE_DATE_EPOCH is invalid")
    return environment


def resolve_build_worker_path(
    path: Path, document: dict[str, Any]
) -> Path:
    """Resolve the live worker while preserving the v1/v2 version boundary."""

    receipt_path = _canonical_absolute(path, "SQ8 worker build receipt")
    schema = document.get("schema_version")
    if schema == BUILD_RECEIPT_SCHEMA_V1:
        worker = _exact(
            document.get("worker"),
            {"path", "bytes", "sha256", "mode", "nlink"},
            "built SQ8 worker",
        )
        worker_path = _canonical_absolute(
            Path(_text(worker["path"], "built SQ8 worker path")),
            "built SQ8 worker path",
        )
    elif schema == BUILD_RECEIPT_SCHEMA:
        worker = _exact(
            document.get("worker"),
            {"relative_path", "bytes", "sha256", "mode", "nlink"},
            "built SQ8 worker",
        )
        if worker["relative_path"] != BUILD_WORKER_RELATIVE_PATH:
            fail("built SQ8 worker relative locator differs")
        if receipt_path.name != "build-receipt.json":
            fail("v2 worker build receipt filename differs")
        worker_path = _canonical_absolute(
            receipt_path.parent / BUILD_WORKER_RELATIVE_PATH,
            "built SQ8 worker",
        )
        if worker_path.parent != receipt_path.parent:
            fail("built SQ8 worker escapes its release root")
    else:
        fail("SQ8 worker build receipt schema differs")
    if worker_path.name != BUILD_WORKER_RELATIVE_PATH:
        fail("built SQ8 worker basename differs")
    return worker_path


def validate_build_receipt(
    path: Path,
    *,
    verify_live_source: bool = True,
    source_root: Path | None = None,
) -> dict[str, Any]:
    receipt_path = _canonical_absolute(path, "SQ8 worker build receipt")
    document, _ = _load_json_file(
        receipt_path,
        "SQ8 worker build receipt",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    document = _exact(
        document,
        {"schema_version", "source", "build", "inputs", "worker"},
        "SQ8 worker build receipt",
    )
    schema = document["schema_version"]
    if schema not in {BUILD_RECEIPT_SCHEMA_V1, BUILD_RECEIPT_SCHEMA}:
        fail("SQ8 worker build receipt schema differs")
    source_root = _validate_clean_detached_source(
        document["source"],
        verify_live=verify_live_source,
        live_root=source_root,
        relocatable=schema == BUILD_RECEIPT_SCHEMA,
    )
    build = _exact(document["build"], {"argv", "environment", "result"}, "worker build")
    argv = build["argv"]
    expected_tail = [
        "build",
        "--locked",
        "--release",
        "-p",
        "ullm-engine",
        "--bin",
        "ullm-sq8-worker",
        "--features",
        "rocm-ck-gfx1201",
    ]
    if (
        type(argv) is not list
        or len(argv) != len(expected_tail) + 1
        or Path(str(argv[0])).name != "cargo"
        or argv[1:] != expected_tail
        or build["result"] != "success"
    ):
        fail("SQ8 worker build command differs")
    build_environment = _validate_build_environment(build["environment"])
    if schema == BUILD_RECEIPT_SCHEMA and verify_live_source:
        _, source_epoch_raw = _git(
            source_root,
            [
                "show",
                "-s",
                "--format=%ct",
                document["source"]["commit"],
            ],
            "worker build source commit timestamp",
        )
        try:
            source_epoch = source_epoch_raw.strip().decode(
                "ascii",
                errors="strict",
            )
        except UnicodeError as error:
            raise PromotionError(
                "worker build source commit timestamp is invalid"
            ) from error
        if (
            SOURCE_DATE_EPOCH_RE.fullmatch(source_epoch) is None
            or build_environment["SOURCE_DATE_EPOCH"] != source_epoch
        ):
            fail("SOURCE_DATE_EPOCH differs from the source commit")
    inputs = document["inputs"]
    if type(inputs) is not list or not inputs:
        fail("SQ8 worker build inputs are empty")
    observed_paths: list[str] = []
    for index, raw in enumerate(inputs):
        entry = _exact(raw, {"path", "sha256"}, f"worker build input {index}")
        relative = _safe_relative(entry["path"], f"worker build input {index} path")
        text = relative.as_posix()
        if text in observed_paths:
            fail("SQ8 worker build inputs contain duplicate paths")
        observed_paths.append(text)
        expected = _hash(entry["sha256"], f"worker build input {text} SHA-256")
        if verify_live_source:
            input_path = _safe_repo_file(
                source_root, text, f"worker build input {text}"
            )
            if stable_hash(input_path, f"worker build input {text}")[1] != expected:
                fail(f"worker build input {text} SHA-256 differs")
    if observed_paths != sorted(
        observed_paths, key=lambda value: value.encode("utf-8")
    ):
        fail("SQ8 worker build inputs are not bytewise sorted")
    if schema == BUILD_RECEIPT_SCHEMA and observed_paths != list(BUILD_INPUTS_V2):
        fail("v2 SQ8 worker build input set differs")
    if schema == BUILD_RECEIPT_SCHEMA_V1 and not REQUIRED_BUILD_INPUTS.issubset(
        observed_paths
    ):
        fail("SQ8 worker build inputs are incomplete")

    worker_fields = (
        {"path", "bytes", "sha256", "mode", "nlink"}
        if schema == BUILD_RECEIPT_SCHEMA_V1
        else {"relative_path", "bytes", "sha256", "mode", "nlink"}
    )
    worker = _exact(document["worker"], worker_fields, "built SQ8 worker")
    worker_path = resolve_build_worker_path(receipt_path, document)
    if worker["mode"] != "0555" or worker["nlink"] != 1:
        fail("built SQ8 worker immutable identity differs")
    size, digest = stable_hash(
        worker_path,
        "built SQ8 worker",
        required_mode=0o555,
        required_nlink=1,
    )
    if (
        _integer(worker["bytes"], "built SQ8 worker bytes", minimum=1) != size
        or _hash(worker["sha256"], "built SQ8 worker SHA-256") != digest
    ):
        fail("built SQ8 worker byte identity differs")
    return document


def _validate_build_provenance(
    path: Path,
    *,
    receipt: dict[str, Any],
    verify_live_source: bool,
    source_root: Path | None,
) -> dict[str, Any]:
    provenance, _ = _load_json_file(
        path,
        "SQ8 worker build provenance",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    provenance = _exact(
        provenance,
        {"schema_version", "source", "build", "worker"},
        "SQ8 worker build provenance",
    )
    receipt_schema = receipt["schema_version"]
    expected_schema = (
        BUILD_PROVENANCE_SCHEMA_V1
        if receipt_schema == BUILD_RECEIPT_SCHEMA_V1
        else BUILD_PROVENANCE_SCHEMA
    )
    if provenance["schema_version"] != expected_schema:
        fail("SQ8 worker build provenance schema differs")
    source = _exact(
        provenance["source"],
        {
            "repository_root",
            "commit",
            "tree",
            "detached",
            "tracked_clean",
            "untracked_clean",
            "inputs",
        },
        "SQ8 worker build provenance source",
    )
    receipt_source = receipt["source"]
    if (
        source["repository_root"] != receipt_source["repository_root"]
        or source["commit"] != receipt_source["commit"]
        or source["tree"] != receipt_source["tree"]
        or source["detached"] is not True
        or source["tracked_clean"] is not True
        or source["untracked_clean"] is not True
    ):
        fail("SQ8 worker build provenance source identity differs")
    provenance_inputs = source["inputs"]
    if type(provenance_inputs) is not dict:
        fail("SQ8 worker build provenance inputs differ")
    for raw in receipt["inputs"]:
        relative = raw["path"]
        entry = _exact(
            provenance_inputs.get(relative),
            {"bytes", "sha256"},
            f"SQ8 worker build provenance input {relative}",
        )
        recorded_bytes = _integer(
            entry["bytes"],
            f"SQ8 worker build provenance input {relative} bytes",
        )
        if (
            recorded_bytes < 0
            or entry["sha256"] != raw["sha256"]
        ):
            fail("SQ8 worker build provenance input identity differs")
        if verify_live_source:
            if source_root is None:
                fail("live build provenance validation requires a source root")
            input_path = _safe_repo_file(
                source_root,
                relative,
                f"SQ8 worker build provenance input {relative}",
            )
            live_bytes, live_sha256 = stable_hash(
                input_path,
                f"SQ8 worker build provenance input {relative}",
            )
            if (
                recorded_bytes != live_bytes
                or entry["sha256"] != live_sha256
            ):
                fail("SQ8 worker build provenance input identity differs")
    if set(provenance_inputs) != {entry["path"] for entry in receipt["inputs"]}:
        fail("SQ8 worker build provenance input set differs")
    build = _exact(
        provenance["build"],
        {
            "argv",
            "working_directory",
            "target_directory",
            "environment",
            "ambient_environment_hermetic",
            "ambient_compile_overrides_rejected",
            "started_unix_ns",
            "finished_unix_ns",
            "toolchain",
            "result",
        },
        "SQ8 worker build provenance build",
    )
    if (
        build["argv"] != receipt["build"]["argv"]
        or build["environment"] != receipt["build"]["environment"]
        or build["result"] != "success"
        or build["working_directory"] != source["repository_root"]
        or build["target_directory"]
        != receipt["build"]["environment"]["CARGO_TARGET_DIR"]
        or build["ambient_environment_hermetic"] is not False
        or build["ambient_compile_overrides_rejected"]
        != list(BUILD_REJECTED_ENVIRONMENT_V2)
        or _integer(build["started_unix_ns"], "build start", minimum=1)
        > _integer(build["finished_unix_ns"], "build finish", minimum=1)
    ):
        fail("SQ8 worker build provenance build identity differs")
    toolchain = _exact(
        build["toolchain"],
        {"cargo", "rustc", "cxx", "hipcc"},
        "SQ8 worker build provenance toolchain",
    )
    for name, value in toolchain.items():
        identity = _exact(
            value,
            {"path", "sha256", "version"},
            f"SQ8 worker build provenance {name}",
        )
        _audit_absolute_path(
            identity["path"],
            f"SQ8 worker build provenance {name} path",
        )
        _hash(
            identity["sha256"],
            f"SQ8 worker build provenance {name} SHA-256",
        )
        _text(
            identity["version"],
            f"SQ8 worker build provenance {name} version",
            4096,
        )
    worker_fields = (
        {
            "path",
            "bytes",
            "sha256",
            "mode",
            "nlink",
            "protocol",
            "format_id",
            "model_id",
        }
        if receipt_schema == BUILD_RECEIPT_SCHEMA_V1
        else {
            "relative_path",
            "bytes",
            "sha256",
            "mode",
            "nlink",
            "protocol",
            "format_id",
            "model_id",
        }
    )
    worker = _exact(provenance["worker"], worker_fields, "build provenance worker")
    if (
        {key: worker[key] for key in receipt["worker"]} != receipt["worker"]
        or worker["protocol"] != WORKER_PROTOCOL
        or worker["format_id"] != FORMAT_ID
        or worker["model_id"] != MODEL_ID
    ):
        fail("SQ8 worker build provenance worker identity differs")
    return provenance


def validate_build_release(
    root: Path,
    *,
    verify_live_source: bool = True,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Validate one exact, relocatable SQ8 worker release directory."""

    release_root = _canonical_absolute(root, "SQ8 worker build release")
    metadata = release_root.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o555
    ):
        fail("SQ8 worker build release directory identity differs")
    try:
        observed_members = {entry.name for entry in os.scandir(release_root)}
    except OSError as error:
        raise PromotionError("SQ8 worker build release cannot be inventoried") from error
    if observed_members != BUILD_RELEASE_MEMBERS:
        fail("SQ8 worker build release member set differs")

    receipt_path = release_root / "build-receipt.json"
    receipt = validate_build_receipt(
        receipt_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    if receipt["schema_version"] != BUILD_RECEIPT_SCHEMA:
        fail("complete SQ8 worker build release requires receipt v2")
    worker_path = resolve_build_worker_path(receipt_path, receipt)
    if (
        receipt["schema_version"] == BUILD_RECEIPT_SCHEMA
        and worker_path != release_root / BUILD_WORKER_RELATIVE_PATH
    ):
        fail("SQ8 worker build release locator differs")
    provenance_path = release_root / "build-provenance.json"
    _validate_build_provenance(
        provenance_path,
        receipt=receipt,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )

    member_hashes: dict[str, str] = {}
    for name in BUILD_SUMMED_MEMBERS:
        member_hashes[name] = stable_hash(
            release_root / name,
            f"SQ8 worker release member {name}",
            required_mode=0o555 if name == BUILD_WORKER_RELATIVE_PATH else 0o444,
            required_nlink=1,
        )[1]
    sums_path = release_root / "SHA256SUMS"
    sums_raw = stable_read(
        sums_path,
        "SQ8 worker release SHA256SUMS",
        required_mode=0o444,
        required_nlink=1,
    )
    expected_sums = "".join(
        f"{member_hashes[name]}  {name}\n" for name in BUILD_SUMMED_MEMBERS
    ).encode("ascii")
    if sums_raw != expected_sums:
        fail("SQ8 worker release SHA256SUMS differs")

    seal_path = release_root / "SEALED.json"
    seal, _ = _load_json_file(
        seal_path,
        "SQ8 worker release seal",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    seal = _exact(
        seal,
        {
            "schema_version",
            "source_commit",
            "source_tree",
            "worker_sha256",
            "build_receipt_sha256",
            "build_provenance_sha256",
            "sha256sums_sha256",
            "complete",
        },
        "SQ8 worker release seal",
    )
    if (
        seal["schema_version"] != BUILD_RELEASE_SEAL_SCHEMA
        or seal["source_commit"] != receipt["source"]["commit"]
        or seal["source_tree"] != receipt["source"]["tree"]
        or seal["worker_sha256"] != receipt["worker"]["sha256"]
        or seal["build_receipt_sha256"] != member_hashes["build-receipt.json"]
        or seal["build_provenance_sha256"]
        != member_hashes["build-provenance.json"]
        or seal["sha256sums_sha256"] != hashlib.sha256(sums_raw).hexdigest()
        or seal["complete"] is not True
    ):
        fail("SQ8 worker release seal identity differs")
    return {
        "schema_version": seal["schema_version"],
        "release_root": os.fspath(release_root),
        "worker_path": os.fspath(worker_path),
        "worker_sha256": receipt["worker"]["sha256"],
        "build_receipt_sha256": member_hashes["build-receipt.json"],
        "seal_sha256": stable_hash(
            seal_path,
            "SQ8 worker release seal",
            required_mode=0o444,
            required_nlink=1,
        )[1],
        "receipt": receipt,
        "seal": seal,
    }


def resolve_build_source_root(
    build: dict[str, Any], source_root: Path | None
) -> Path:
    schema = build.get("schema_version")
    if schema == BUILD_RECEIPT_SCHEMA:
        if source_root is None:
            fail("v2 worker build validation requires an explicit source root")
        return _canonical_absolute(source_root, "current worker build source root")
    if schema != BUILD_RECEIPT_SCHEMA_V1:
        fail("SQ8 worker build receipt schema differs")
    recorded = _canonical_absolute(
        Path(
            _text(
                build["source"]["repository_root"],
                "v1 worker build repository root",
            )
        ),
        "v1 worker build repository root",
    )
    if source_root is None:
        return recorded
    selected = _canonical_absolute(source_root, "requested v1 worker build source root")
    if selected != recorded:
        fail("v1 worker build source root override differs")
    return recorded


def _validate_promotion_build(
    build_receipt_path: Path,
    *,
    verify_live_source: bool,
    source_root: Path | None,
) -> dict[str, Any]:
    """Require the complete v2 release while preserving v1 receipt semantics."""

    build = validate_build_receipt(
        build_receipt_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    if build["schema_version"] == BUILD_RECEIPT_SCHEMA:
        release = validate_build_release(
            _canonical_absolute(
                build_receipt_path, "SQ8 worker build receipt"
            ).parent,
            verify_live_source=verify_live_source,
            source_root=source_root,
        )
        if release["receipt"] != build:
            fail("SQ8 worker build release receipt changed during validation")
    return build


def _cpu_tool_identity(path: Path, label: str) -> dict[str, Any]:
    invocation = path.absolute()
    if (
        not path.is_absolute()
        or os.fspath(path) != os.fspath(invocation)
        or ".." in path.parts
    ):
        fail(f"{label} invocation path is not canonical absolute")
    try:
        invoked_metadata = invocation.lstat()
    except OSError as error:
        raise PromotionError(f"{label} invocation path is unavailable") from error
    if not (
        stat.S_ISREG(invoked_metadata.st_mode) or stat.S_ISLNK(invoked_metadata.st_mode)
    ):
        fail(f"{label} invocation path is not a file or symlink")
    resolved = invocation.resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o111 == 0:
        fail(f"{label} resolved executable identity differs")
    size, digest = stable_hash(resolved, f"{label} resolved executable")
    return {
        "invocation_path": os.fspath(invocation),
        "resolved_path": os.fspath(resolved),
        "bytes": size,
        "sha256": digest,
    }


def _validate_cpu_tool(value: Any, label: str) -> dict[str, Any]:
    tool = _exact(
        value,
        {"invocation_path", "resolved_path", "bytes", "sha256"},
        label,
    )
    observed = _cpu_tool_identity(
        Path(_text(tool["invocation_path"], f"{label} invocation path")),
        label,
    )
    if tool != observed:
        fail(f"{label} live executable identity differs")
    return tool


def _cpu_environment(target_dir: Path, source_root: Path) -> dict[str, str]:
    target = target_dir.absolute()
    if (
        not target_dir.is_absolute()
        or os.fspath(target_dir) != os.fspath(target)
        or ".." in target_dir.parts
    ):
        fail("SQ8 CPU-case Cargo target directory is not canonical absolute")
    try:
        target.mkdir(mode=0o755, parents=False, exist_ok=True)
    except OSError as error:
        raise PromotionError(
            "SQ8 CPU-case Cargo target directory is unavailable"
        ) from error
    metadata = target.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail("SQ8 CPU-case Cargo target directory is unsafe")
    python_path = _safe_repo_file(
        source_root,
        "services/openai-gateway/src",
        "SQ8 CPU-case gateway source",
    )
    if not python_path.is_dir():
        fail("SQ8 CPU-case gateway source is not a directory")
    return {
        "CARGO_INCREMENTAL": "0",
        "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": os.fspath(target),
        "CARGO_TERM_COLOR": "never",
        "CUDA_VISIBLE_DEVICES": "-1",
        "GPU_DEVICE_ORDINAL": "-1",
        "HSA_VISIBLE_DEVICES": "-1",
        "HIP_VISIBLE_DEVICES": "-1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.fspath(python_path),
        "PY_COLORS": "0",
        "ROCR_VISIBLE_DEVICES": "-1",
        "ULLM_HIP_VISIBLE_DEVICES": "-1",
    }


def _validate_cpu_environment(value: Any, *, source_root: Path) -> dict[str, str]:
    environment = _exact(
        value,
        {
            "CARGO_INCREMENTAL",
            "CARGO_NET_OFFLINE",
            "CARGO_TARGET_DIR",
            "CARGO_TERM_COLOR",
            "CUDA_VISIBLE_DEVICES",
            "GPU_DEVICE_ORDINAL",
            "HSA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONNOUSERSITE",
            "PYTHONPATH",
            "PY_COLORS",
            "ROCR_VISIBLE_DEVICES",
            "ULLM_HIP_VISIBLE_DEVICES",
        },
        "SQ8 CPU-case environment",
    )
    target = Path(
        _text(environment["CARGO_TARGET_DIR"], "SQ8 CPU-case Cargo target directory")
    )
    python_path = _safe_repo_file(
        source_root,
        "services/openai-gateway/src",
        "SQ8 CPU-case gateway source",
    )
    if (
        not target.is_absolute()
        or os.fspath(target) != os.fspath(target.absolute())
        or ".." in target.parts
        or environment
        != {
            "CARGO_INCREMENTAL": "0",
            "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": os.fspath(target),
            "CARGO_TERM_COLOR": "never",
            "CUDA_VISIBLE_DEVICES": "-1",
            "GPU_DEVICE_ORDINAL": "-1",
            "HSA_VISIBLE_DEVICES": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": os.fspath(python_path),
            "PY_COLORS": "0",
            "ROCR_VISIBLE_DEVICES": "-1",
            "ULLM_HIP_VISIBLE_DEVICES": "-1",
        }
    ):
        fail("SQ8 CPU-case environment differs")
    return environment


def _cpu_test_argv(
    *,
    framework: str,
    selector: str,
    cargo_path: str,
    python_path: str,
) -> list[str]:
    if framework == "cargo-test":
        return [
            cargo_path,
            "test",
            "--locked",
            "--offline",
            "-p",
            "ullm-engine",
            "--features",
            "rocm-ck-gfx1201",
            "--lib",
            selector,
            "--",
            "--exact",
            "--test-threads=1",
        ]
    if framework == "pytest":
        return [
            python_path,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            selector,
        ]
    fail("SQ8 CPU-case test framework differs")


def _run_cpu_test(
    argv: Sequence[str],
    *,
    source_root: Path,
    environment: dict[str, str],
) -> tuple[int, bytes, bytes]:
    process_environment = os.environ.copy()
    for name in tuple(process_environment):
        if name.startswith(
            (
                "CUDA_",
                "GPU_DEVICE_",
                "HIP_",
                "HSA_",
                "ROCM_",
                "ROCR_",
                "ULLM_HIP_",
                "ULLM_REQUIRE_HIP_",
            )
        ):
            process_environment.pop(name, None)
    process_environment.update(environment)
    try:
        result = subprocess.run(
            list(argv),
            check=False,
            cwd=source_root,
            env=process_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PromotionError("SQ8 CPU-case command execution failed") from error
    if (
        len(result.stdout) > MAX_TEST_OUTPUT_BYTES
        or len(result.stderr) > MAX_TEST_OUTPUT_BYTES
    ):
        fail("SQ8 CPU-case command output exceeds its bound")
    return result.returncode, result.stdout, result.stderr


def _cpu_output(raw: bytes) -> dict[str, Any]:
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "base64": base64.b64encode(raw).decode("ascii"),
    }


def _validate_cpu_output(value: Any, label: str) -> bytes:
    output = _exact(value, {"bytes", "sha256", "base64"}, label)
    encoded = output["base64"]
    if (
        type(encoded) is not str
        or "\x00" in encoded
        or len(encoded.encode("ascii", errors="ignore")) != len(encoded)
        or len(encoded) > MAX_TEST_OUTPUT_BYTES * 2
    ):
        fail(f"{label} base64 is invalid")
    try:
        raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    except (UnicodeError, ValueError) as error:
        raise PromotionError(f"{label} base64 is invalid") from error
    if (
        len(raw) > MAX_TEST_OUTPUT_BYTES
        or _integer(output["bytes"], f"{label} bytes") != len(raw)
        or _hash(output["sha256"], f"{label} SHA-256")
        != hashlib.sha256(raw).hexdigest()
    ):
        fail(f"{label} byte identity differs")
    return raw


def _cpu_test_specs() -> tuple[tuple[str, str, str], ...]:
    return tuple(
        ("cargo-test", run_id, selector) for run_id, selector in CPU_RUST_TESTS
    ) + tuple(("pytest", run_id, selector) for run_id, selector in CPU_PYTEST_TESTS)


def _cpu_case_run_map() -> dict[str, list[str]]:
    return {
        "protocol-v2-generate": [
            "rust-command-schema",
            "rust-profile-bijection",
            "rust-explicit-v2-reasoning",
            "rust-pre-busy-schema",
        ],
        "protocol-v2-cancel": ["rust-command-schema", "rust-pre-busy-schema"],
        "protocol-v2-shutdown": ["rust-command-schema", "rust-pre-busy-schema"],
        "reject-v1-generate": ["rust-command-schema", "rust-pre-busy-schema"],
        "reject-v1-cancel": ["rust-command-schema", "rust-pre-busy-schema"],
        "reject-v1-shutdown": ["rust-command-schema", "rust-pre-busy-schema"],
        "reasoning-disabled": [
            "rust-profile-bijection",
            "rust-explicit-v2-reasoning",
            "rust-request-dialect-binding",
            "rust-disabled-accounting",
        ],
        "reasoning-budget-zero": [
            "rust-reasoning-contract",
            "rust-request-dialect-binding",
            "rust-bounded-budgets",
        ],
        "reasoning-low-32": [
            "rust-reasoning-contract",
            "rust-bounded-budgets",
        ],
        "reasoning-medium-128": [
            "rust-reasoning-contract",
            "rust-bounded-budgets",
        ],
        "reasoning-high-256": [
            "rust-reasoning-contract",
            "rust-bounded-budgets",
        ],
        "reasoning-unbounded-natural-close": [
            "rust-reasoning-contract",
            "rust-unbounded-natural-close",
        ],
        "reasoning-budget-forced-close": [
            "rust-reasoning-contract",
            "rust-bounded-budgets",
        ],
        "reasoning-eos-forced-close": [
            "rust-reasoning-contract",
            "rust-eos-rng",
            "gateway-eos-reconcile",
        ],
        "reasoning-answer-reservation": [
            "rust-reasoning-contract",
            "rust-answer-reservation",
        ],
        "reasoning-natural-accounting": [
            "rust-unbounded-natural-close",
            "gateway-eos-reconcile",
        ],
        "forced-token-rng-unconsumed": ["rust-eos-rng"],
        "cancel-rollback": ["rust-transaction-rollback"],
        "publish-failure-rollback": ["rust-transaction-rollback"],
        "reset-accounting": ["rust-reset-accounting"],
        "release-usage-reconcile": [
            "rust-release-usage",
            "gateway-eos-reconcile",
            "gateway-length-reconcile",
        ],
    }


def build_cpu_cases_report(
    *,
    build_receipt_path: Path,
    source_root: Path | None = None,
    ephemeral_manifest_path: Path,
    cargo_path: Path,
    python_path: Path,
    target_dir: Path,
    verify_live_source: bool = True,
) -> dict[str, Any]:
    """Execute the exact GPU-hidden CPU admission tests and build their report."""

    build = _validate_promotion_build(
        build_receipt_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    source_root = resolve_build_source_root(build, source_root)
    worker_path = resolve_build_worker_path(build_receipt_path, build)
    sources = _evidence_source_entries(source_root)
    _validate_evidence_sources(
        sources,
        source_root=source_root,
        verify_live=verify_live_source,
    )
    manifest, manifest_raw = _load_json_file(
        ephemeral_manifest_path,
        "SQ8 ephemeral served-model manifest",
        required_mode=0o444,
        required_nlink=1,
        maximum=1024 * 1024,
    )
    if (
        manifest.get("schema_version") != SERVED_MODEL_SCHEMA
        or manifest.get("format", {}).get("format_id") != FORMAT_ID
        or manifest.get("worker", {}).get("protocol") != WORKER_PROTOCOL
        or manifest.get("worker", {}).get("binary") != os.fspath(worker_path)
        or manifest.get("worker", {}).get("binary_sha256") != build["worker"]["sha256"]
        or manifest.get("promotion", {}).get("source_commit")
        != build["source"]["commit"]
        or manifest.get("reasoning") != REASONING_CONTRACT
    ):
        fail("SQ8 CPU-case manifest identity differs")
    try:
        summary = _load_module(
            "_ullm_sq8_cpu_manifest_validator", SERVED_MODEL_VALIDATOR_PATH
        ).validation_summary(ephemeral_manifest_path)
    except Exception as error:
        raise PromotionError(
            "SQ8 CPU-case manifest failed strict validation"
        ) from error
    if (
        summary.get("model_id") != MODEL_ID
        or summary.get("format_id") != FORMAT_ID
        or summary.get("worker", {}).get("binary_sha256") != build["worker"]["sha256"]
    ):
        fail("SQ8 CPU-case strict manifest identity differs")

    tools = {
        "cargo": _cpu_tool_identity(cargo_path, "SQ8 CPU-case Cargo"),
        "python": _cpu_tool_identity(python_path, "SQ8 CPU-case Python"),
    }
    environment = _cpu_environment(target_dir, source_root)
    try:
        Path(environment["CARGO_TARGET_DIR"]).relative_to(source_root)
    except ValueError:
        pass
    else:
        fail("SQ8 CPU-case Cargo target directory is inside the source checkout")
    runs: list[dict[str, Any]] = []
    for framework, run_id, selector in _cpu_test_specs():
        argv = _cpu_test_argv(
            framework=framework,
            selector=selector,
            cargo_path=tools["cargo"]["invocation_path"],
            python_path=tools["python"]["invocation_path"],
        )
        returncode, stdout, stderr = _run_cpu_test(
            argv, source_root=source_root, environment=environment
        )
        if framework == "cargo-test":
            passed = (
                returncode == 0 and f"test {selector} ... ok".encode("utf-8") in stdout
            )
        else:
            passed = (
                returncode == 0
                and re.search(rb"\b[1-9][0-9]* passed\b", stdout) is not None
                and b" failed" not in stdout
            )
        if not passed:
            fail(f"SQ8 CPU-case test failed: {run_id}")
        runs.append(
            {
                "id": run_id,
                "framework": framework,
                "selector": selector,
                "argv": argv,
                "exit_code": returncode,
                "stdout": _cpu_output(stdout),
                "stderr": _cpu_output(stderr),
                "result": "pass",
            }
        )
    _validate_clean_detached_source(
        build["source"],
        verify_live=verify_live_source,
        live_root=source_root,
        relocatable=build["schema_version"] == BUILD_RECEIPT_SCHEMA,
    )
    _validate_evidence_sources(
        sources,
        source_root=source_root,
        verify_live=verify_live_source,
    )
    case_run_map = _cpu_case_run_map()
    report = {
        "schema_version": CPU_CASES_SCHEMA,
        "source_root": os.fspath(source_root),
        "source_commit": build["source"]["commit"],
        "source_tree": build["source"]["tree"],
        "served_model_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "worker_binary_sha256": build["worker"]["sha256"],
        "identity": {
            "format_id": FORMAT_ID,
            "worker_protocol": WORKER_PROTOCOL,
            "reasoning_dialect": REASONING_DIALECT,
        },
        "tools": tools,
        "environment": environment,
        "test_runs": runs,
        "cases": [
            {
                "id": case_id,
                "result": "pass",
                "details": {"test_run_ids": case_run_map[case_id]},
            }
            for case_id in CPU_CASE_IDS
        ],
        "summary": {
            "required_case_ids": list(CPU_CASE_IDS),
            "test_run_count": len(runs),
            "pass_count": len(CPU_CASE_IDS),
            "fail_count": 0,
            "all_pass": True,
        },
    }
    validate_cpu_cases_document(
        report,
        source_root=source_root,
        source_commit=build["source"]["commit"],
        source_tree=build["source"]["tree"],
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        worker_sha256=build["worker"]["sha256"],
        reasoning=manifest["reasoning"],
    )
    return report


def validate_cpu_cases_document(
    document: Any,
    *,
    source_root: Path,
    source_commit: str,
    source_tree: str,
    manifest_sha256: str,
    worker_sha256: str,
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    document = _exact(
        document,
        {
            "schema_version",
            "source_root",
            "source_commit",
            "source_tree",
            "served_model_manifest_sha256",
            "worker_binary_sha256",
            "identity",
            "tools",
            "environment",
            "test_runs",
            "cases",
            "summary",
        },
        "SQ8 serving CPU cases",
    )
    if document["schema_version"] != CPU_CASES_SCHEMA:
        fail("SQ8 serving CPU-case schema differs")
    expected_source_root = _canonical_absolute(
        source_root, "expected SQ8 CPU-case source root"
    )
    if (
        document["source_root"] != os.fspath(expected_source_root)
        or document["source_commit"] != source_commit
        or document["source_tree"] != source_tree
        or document["served_model_manifest_sha256"] != manifest_sha256
        or document["worker_binary_sha256"] != worker_sha256
    ):
        fail("SQ8 serving CPU-case source or binary identity differs")
    _git_object(document["source_commit"], "SQ8 serving CPU-case source commit")
    _git_object(document["source_tree"], "SQ8 serving CPU-case source tree")
    _hash(
        document["served_model_manifest_sha256"],
        "SQ8 serving CPU-case manifest SHA-256",
    )
    _hash(
        document["worker_binary_sha256"],
        "SQ8 serving CPU-case worker SHA-256",
    )
    identity = _exact(
        document["identity"],
        {"format_id", "worker_protocol", "reasoning_dialect"},
        "SQ8 serving CPU-case identity",
    )
    if (
        identity
        != {
            "format_id": FORMAT_ID,
            "worker_protocol": WORKER_PROTOCOL,
            "reasoning_dialect": REASONING_DIALECT,
        }
        or reasoning != REASONING_CONTRACT
    ):
        fail("SQ8 serving CPU-case protocol or reasoning identity differs")

    tools = _exact(document["tools"], {"cargo", "python"}, "SQ8 CPU-case tools")
    cargo = _validate_cpu_tool(tools["cargo"], "SQ8 CPU-case Cargo")
    python = _validate_cpu_tool(tools["python"], "SQ8 CPU-case Python")
    environment = _validate_cpu_environment(
        document["environment"], source_root=expected_source_root
    )
    del environment

    expected_specs = _cpu_test_specs()
    test_runs = document["test_runs"]
    if type(test_runs) is not list or len(test_runs) != len(expected_specs):
        fail("SQ8 CPU-case test-run set differs")
    observed_run_ids: list[str] = []
    for (framework, expected_id, selector), raw in zip(
        expected_specs, test_runs, strict=True
    ):
        run = _exact(
            raw,
            {
                "id",
                "framework",
                "selector",
                "argv",
                "exit_code",
                "stdout",
                "stderr",
                "result",
            },
            f"SQ8 CPU test run {expected_id}",
        )
        expected_argv = _cpu_test_argv(
            framework=framework,
            selector=selector,
            cargo_path=cargo["invocation_path"],
            python_path=python["invocation_path"],
        )
        if (
            run["id"] != expected_id
            or run["framework"] != framework
            or run["selector"] != selector
            or run["argv"] != expected_argv
            or type(run["exit_code"]) is not int
            or run["exit_code"] != 0
            or run["result"] != "pass"
        ):
            fail(f"SQ8 CPU test run {expected_id} identity differs")
        stdout = _validate_cpu_output(
            run["stdout"], f"SQ8 CPU test run {expected_id} stdout"
        )
        _validate_cpu_output(run["stderr"], f"SQ8 CPU test run {expected_id} stderr")
        if framework == "cargo-test":
            if f"test {selector} ... ok".encode("utf-8") not in stdout:
                fail(f"SQ8 CPU test run {expected_id} has no exact pass record")
        elif (
            re.search(rb"\b[1-9][0-9]* passed\b", stdout) is None
            or b" failed" in stdout
        ):
            fail(f"SQ8 CPU test run {expected_id} has no pytest pass record")
        observed_run_ids.append(expected_id)

    expected_run_ids = [run_id for _, run_id, _ in expected_specs]
    if observed_run_ids != expected_run_ids:
        fail("SQ8 CPU-case test-run ordering differs")
    cases = document["cases"]
    if type(cases) is not list or len(cases) != len(CPU_CASE_IDS):
        fail("SQ8 serving CPU-case set differs")
    case_run_map = _cpu_case_run_map()
    for expected_id, raw in zip(CPU_CASE_IDS, cases, strict=True):
        case = _exact(raw, {"id", "result", "details"}, f"CPU case {expected_id}")
        if case["id"] != expected_id or case["result"] != "pass":
            fail(f"CPU case {expected_id} did not pass")
        details = _exact(
            case["details"],
            {"test_run_ids"},
            f"CPU case {expected_id} details",
        )
        if details["test_run_ids"] != case_run_map[expected_id] or any(
            run_id not in expected_run_ids for run_id in details["test_run_ids"]
        ):
            fail(f"CPU case {expected_id} test bindings differ")
    summary = _exact(
        document["summary"],
        {
            "required_case_ids",
            "test_run_count",
            "pass_count",
            "fail_count",
            "all_pass",
        },
        "SQ8 serving CPU-case summary",
    )
    if (
        summary["required_case_ids"] != list(CPU_CASE_IDS)
        or summary["test_run_count"] != len(expected_specs)
        or summary["pass_count"] != len(CPU_CASE_IDS)
        or summary["fail_count"] != 0
        or summary["all_pass"] is not True
    ):
        fail("SQ8 serving CPU-case summary differs")
    return document


def validate_cpu_cases(
    path: Path,
    *,
    source_root: Path,
    source_commit: str,
    source_tree: str,
    manifest_sha256: str,
    worker_sha256: str,
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    document, _ = _load_json_file(
        path,
        "SQ8 serving CPU cases",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    return validate_cpu_cases_document(
        document,
        source_root=source_root,
        source_commit=source_commit,
        source_tree=source_tree,
        manifest_sha256=manifest_sha256,
        worker_sha256=worker_sha256,
        reasoning=reasoning,
    )


def _load_module(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"required validator is unavailable: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def run_full_product_validation(product_root: Path) -> dict[str, Any]:
    """Recompute the existing SQ8 product validator over every payload."""

    module = _load_module("_ullm_sq8_serving_product_validator", PRODUCT_VALIDATOR_PATH)
    root = _canonical_absolute(product_root, "SQ8 product root")
    try:
        promotion = module.validate_promotion(root)
        artifact = module.validate_artifact(root / "artifact", full_payloads=True)
        package = module.validate_package(root / "package", full_payloads=True)
    except Exception as error:
        raise PromotionError("full SQ8 product validation failed") from error
    return {
        "schema_version": PRODUCT_SCHEMA,
        "product_root": os.fspath(root),
        "created_at": promotion["created_at"],
        "model_revision": module.EXPECTED_MODEL_REVISION,
        "artifact": artifact,
        "package": package,
        "read_only": True,
        "full_payloads": True,
        "verified": True,
    }


def _validate_product_result(value: Any, *, product_root: Path) -> dict[str, Any]:
    result = _exact(
        value,
        {
            "schema_version",
            "product_root",
            "created_at",
            "model_revision",
            "artifact",
            "package",
            "read_only",
            "full_payloads",
            "verified",
        },
        "SQ8 full product validation",
    )
    if (
        result["schema_version"] != PRODUCT_SCHEMA
        or result["product_root"] != os.fspath(product_root)
        or result["read_only"] is not True
        or result["full_payloads"] is not True
        or result["verified"] is not True
    ):
        fail("SQ8 full product validation state differs")
    _text(result["created_at"], "SQ8 product creation time")
    _git_object(result["model_revision"], "SQ8 model revision")
    artifact = _exact(
        result["artifact"],
        {
            "manifest_sha256",
            "content_sha256",
            "selected_pair_count",
            "payloads_hashed",
        },
        "SQ8 validated artifact",
    )
    package = _exact(
        result["package"],
        {
            "manifest_sha256",
            "payload_count",
            "payload_bytes",
            "payloads_hashed",
        },
        "SQ8 validated package",
    )
    if (
        artifact["payloads_hashed"] is not True
        or package["payloads_hashed"] is not True
    ):
        fail("SQ8 product payload validation is incomplete")
    _hash(artifact["manifest_sha256"], "SQ8 artifact manifest SHA-256")
    _hash(artifact["content_sha256"], "SQ8 artifact content SHA-256")
    _integer(artifact["selected_pair_count"], "SQ8 artifact pair count", minimum=1)
    _hash(package["manifest_sha256"], "SQ8 package manifest SHA-256")
    _integer(package["payload_count"], "SQ8 package payload count", minimum=1)
    _integer(package["payload_bytes"], "SQ8 package payload bytes", minimum=1)
    return result


def _profile_contract(profile: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    if profile.get("schema_version") != "ullm.served_model.profile.v1":
        fail("SQ8 served-model profile schema differs")
    expected_root = {
        "schema_version",
        "public",
        "generation",
        "format",
        "tokenizer",
        "worker",
        "reasoning",
        "product",
        "promotion",
    }
    if set(profile) != expected_root:
        fail("SQ8 served-model profile root fields differ")
    public = _exact(
        profile["public"],
        {
            "id",
            "name",
            "description",
            "upstream_id",
            "revision",
            "context_length",
        },
        "SQ8 profile public identity",
    )
    format_value = _exact(
        profile["format"], {"format_id", "implementation_id"}, "SQ8 profile format"
    )
    worker = _exact(
        profile["worker"],
        {
            "protocol",
            "binary",
            "arguments",
            "required_environment",
            "identity",
        },
        "SQ8 profile worker",
    )
    promotion = _exact(
        profile["promotion"],
        {
            "receipt",
            "source_commit_from_receipt",
            "required_schema_version",
            "evidence_from_receipt",
            "evidence_sha256_from_receipt",
        },
        "SQ8 profile promotion",
    )
    product = _exact(
        profile["product"], {"root", "artifact", "package"}, "SQ8 profile product"
    )
    artifact = _exact(
        product["artifact"],
        {"manifest_path", "content_sha256_from_receipt"},
        "SQ8 profile artifact",
    )
    if (
        public["id"] != MODEL_ID
        or public["upstream_id"] != UPSTREAM_MODEL_ID
        or public["revision"] != UPSTREAM_MODEL_REVISION
        or format_value
        != {"format_id": FORMAT_ID, "implementation_id": IMPLEMENTATION_ID}
        or worker["protocol"] != WORKER_PROTOCOL
        or Path(str(worker["binary"])).name != "ullm-sq8-worker"
        or promotion["source_commit_from_receipt"] != ["source_commit"]
        or promotion["required_schema_version"] != RECEIPT_SCHEMA
        or promotion["evidence_from_receipt"] != ["evidence", "path"]
        or promotion["evidence_sha256_from_receipt"] != ["evidence", "sha256"]
        or artifact["content_sha256_from_receipt"]
        != ["product", "artifact_content_sha256"]
    ):
        fail("SQ8 served-model profile promotion contract differs")
    reasoning = _exact(
        profile["reasoning"],
        {
            "enabled_by_default",
            "dialect_id",
            "start_token_ids",
            "end_token_ids",
            "forced_end_token_ids",
            "initial_phase",
            "eos_policy",
            "effort_budgets",
            "max_budget_tokens",
            "reserved_answer_tokens",
            "history_reasoning_policy",
        },
        "SQ8 profile reasoning",
    )
    if reasoning != REASONING_CONTRACT:
        fail("SQ8 profile reasoning contract differs")
    receipt_path = _canonical_absolute(
        Path(_text(promotion["receipt"], "SQ8 serving receipt path")),
        "SQ8 serving receipt path",
        may_be_absent=True,
    )
    product_root = _canonical_absolute(
        Path(_text(product["root"], "SQ8 profile product root")),
        "SQ8 profile product root",
    )
    if receipt_path.parent != product_root:
        fail("SQ8 serving receipt is outside the immutable product root")
    return reasoning, receipt_path


def _manifest_semantics(
    profile: dict[str, Any],
    manifest: dict[str, Any],
    *,
    worker_path: Path,
    worker_sha256: str,
    product_root: Path,
    product_result: dict[str, Any],
) -> dict[str, Any]:
    if (
        set(manifest)
        != {
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "promotion",
            "reasoning",
        }
        or manifest.get("schema_version") != SERVED_MODEL_SCHEMA
    ):
        fail("SQ8 ephemeral served-model manifest root differs")
    if any(
        manifest.get(field) != profile.get(field)
        for field in ("public", "generation", "format", "reasoning")
    ):
        fail("SQ8 ephemeral manifest profile semantics differ")
    observed_worker = _exact(
        manifest["worker"],
        {
            "protocol",
            "binary",
            "binary_sha256",
            "arguments",
            "required_environment",
            "identity",
        },
        "SQ8 ephemeral manifest worker",
    )
    expected_worker_profile = profile["worker"]
    if (
        observed_worker["protocol"] != WORKER_PROTOCOL
        or observed_worker["binary"] != os.fspath(worker_path)
        or observed_worker["binary_sha256"] != worker_sha256
        or any(
            observed_worker[field] != expected_worker_profile[field]
            for field in ("arguments", "required_environment", "identity")
        )
    ):
        fail("SQ8 ephemeral manifest worker binding differs")
    tokenizer_profile = profile["tokenizer"]
    tokenizer = _exact(
        manifest["tokenizer"],
        {
            "root",
            "transformers_version",
            "class",
            "chat_template_sha256",
            "files",
            "template_options",
        },
        "SQ8 ephemeral manifest tokenizer",
    )
    tokenizer_root = _canonical_absolute(
        Path(_text(tokenizer["root"], "SQ8 tokenizer root")), "SQ8 tokenizer root"
    )
    if (
        tokenizer["root"]
        != os.fspath(
            _canonical_absolute(
                Path(_text(tokenizer_profile["root"], "SQ8 profile tokenizer root")),
                "SQ8 profile tokenizer root",
            )
        )
        or tokenizer["transformers_version"]
        != tokenizer_profile["transformers_version"]
        or tokenizer["class"] != tokenizer_profile["class"]
        or tokenizer["template_options"] != tokenizer_profile["template_options"]
    ):
        fail("SQ8 ephemeral manifest tokenizer profile differs")
    raw_files = tokenizer_profile["files"]
    if type(raw_files) is not list or set(tokenizer["files"]) != set(raw_files):
        fail("SQ8 ephemeral manifest tokenizer file set differs")
    for relative in raw_files:
        expected_hash = _hash(
            tokenizer["files"].get(relative), f"SQ8 tokenizer {relative} SHA-256"
        )
        actual = _safe_repo_file(tokenizer_root, relative, f"SQ8 tokenizer {relative}")
        if stable_hash(actual, f"SQ8 tokenizer {relative}")[1] != expected_hash:
            fail(f"SQ8 tokenizer {relative} SHA-256 differs")
    tokenizer_config, tokenizer_config_raw = _load_json_file(
        tokenizer_root / "tokenizer_config.json", "SQ8 tokenizer config"
    )
    chat_template = tokenizer_config.get("chat_template")
    if (
        type(chat_template) is not str
        or hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        != tokenizer["chat_template_sha256"]
    ):
        fail("SQ8 tokenizer chat-template identity differs")
    del tokenizer_config_raw

    product_profile = profile["product"]
    product = _exact(
        manifest["product"], {"root", "artifact", "package"}, "SQ8 manifest product"
    )
    artifact_profile = product_profile["artifact"]
    package_profile = product_profile["package"]
    artifact = _exact(
        product["artifact"],
        {"manifest_path", "manifest_sha256", "content_sha256"},
        "SQ8 manifest artifact",
    )
    package = _exact(
        product["package"],
        {"manifest_path", "manifest_sha256"},
        "SQ8 manifest package",
    )
    if (
        product["root"] != os.fspath(product_root)
        or artifact["manifest_path"] != artifact_profile["manifest_path"]
        or package["manifest_path"] != package_profile["manifest_path"]
        or artifact["manifest_sha256"] != product_result["artifact"]["manifest_sha256"]
        or artifact["content_sha256"] != product_result["artifact"]["content_sha256"]
        or package["manifest_sha256"] != product_result["package"]["manifest_sha256"]
    ):
        fail("SQ8 ephemeral manifest product binding differs")
    if (
        stable_hash(
            product_root / artifact["manifest_path"],
            "SQ8 artifact manifest",
            required_mode=0o444,
            required_nlink=1,
        )[1]
        != artifact["manifest_sha256"]
        or stable_hash(
            product_root / package["manifest_path"],
            "SQ8 package manifest",
            required_mode=0o444,
            required_nlink=1,
        )[1]
        != package["manifest_sha256"]
    ):
        fail("SQ8 live product manifest binding differs")
    promotion = _exact(
        manifest["promotion"],
        {"source_commit", "receipt", "receipt_sha256"},
        "SQ8 ephemeral manifest promotion",
    )
    _git_object(promotion["source_commit"], "SQ8 ephemeral promotion source commit")
    receipt_path = _canonical_absolute(
        Path(_text(promotion["receipt"], "SQ8 ephemeral receipt path")),
        "SQ8 ephemeral receipt path",
    )
    receipt, receipt_raw = _load_json_file(
        receipt_path,
        "SQ8 ephemeral receipt",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    receipt_product = _exact(
        receipt.get("product"),
        {"artifact_content_sha256"},
        "SQ8 ephemeral receipt product",
    )
    if (
        set(receipt) != {"schema_version", "source_commit", "product"}
        or receipt.get("schema_version") != EPHEMERAL_RECEIPT_SCHEMA
        or receipt.get("source_commit") != promotion["source_commit"]
        or _hash(
            receipt_product["artifact_content_sha256"],
            "SQ8 ephemeral receipt artifact content SHA-256",
        )
        != artifact["content_sha256"]
        or hashlib.sha256(receipt_raw).hexdigest()
        != _hash(
            promotion["receipt_sha256"],
            "SQ8 ephemeral receipt SHA-256",
        )
    ):
        fail("SQ8 ephemeral promotion receipt binding differs")
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "reasoning",
        )
    }


def _ephemeral_product_metadata(product_root: Path) -> dict[str, Any]:
    module = _load_module(
        "_ullm_sq8_ephemeral_product_validator", PRODUCT_VALIDATOR_PATH
    )
    try:
        promotion = module.validate_promotion(product_root)
        artifact = module.validate_artifact(
            product_root / "artifact", full_payloads=False
        )
        package = module.validate_package(product_root / "package", full_payloads=False)
    except Exception as error:
        raise PromotionError(
            "SQ8 ephemeral product metadata validation failed"
        ) from error
    return {
        "model_revision": module.EXPECTED_MODEL_REVISION,
        "created_at": promotion["created_at"],
        "artifact": artifact,
        "package": package,
    }


def prepare_ephemeral_manifest(
    *,
    build_receipt_path: Path,
    source_root: Path | None = None,
    profile_path: Path,
    receipt_output_path: Path,
    manifest_output_path: Path,
    verify_live_source: bool = True,
) -> dict[str, Any]:
    """Publish the pre-receipt scaffold and its strict SQ8 v2 manifest."""

    build = _validate_promotion_build(
        build_receipt_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    source_root = resolve_build_source_root(build, source_root)
    sources = _evidence_source_entries(source_root)
    _validate_evidence_sources(
        sources,
        source_root=source_root,
        verify_live=verify_live_source,
    )
    profile, _ = _load_json_file(profile_path, "SQ8 served-model profile")
    reasoning, final_receipt = _profile_contract(profile)
    if os.path.lexists(final_receipt):
        fail("SQ8 production serving receipt already exists")
    worker_path = _canonical_absolute(
        Path(_text(profile["worker"]["binary"], "SQ8 profile worker")),
        "SQ8 profile worker",
    )
    build_worker_path = resolve_build_worker_path(build_receipt_path, build)
    if (
        worker_path != build_worker_path
        or build["worker"]["sha256"]
        != stable_hash(
            worker_path,
            "SQ8 profile worker",
            required_mode=0o555,
            required_nlink=1,
        )[1]
    ):
        fail("SQ8 ephemeral profile and build worker identities differ")

    receipt_output = _canonical_absolute(
        receipt_output_path, "SQ8 ephemeral receipt output", may_be_absent=True
    )
    manifest_output = _canonical_absolute(
        manifest_output_path, "SQ8 ephemeral manifest output", may_be_absent=True
    )
    if (
        receipt_output == final_receipt
        or receipt_output == manifest_output
        or receipt_output.parent != manifest_output.parent
        or os.path.lexists(receipt_output)
        or os.path.lexists(manifest_output)
    ):
        fail("SQ8 ephemeral output identity differs or already exists")

    product_root = _canonical_absolute(
        Path(_text(profile["product"]["root"], "SQ8 product root")),
        "SQ8 product root",
    )
    product = _ephemeral_product_metadata(product_root)
    if product["model_revision"] != profile["public"]["revision"]:
        fail("SQ8 ephemeral product and profile revisions differ")
    scaffold = {
        "schema_version": EPHEMERAL_RECEIPT_SCHEMA,
        "source_commit": build["source"]["commit"],
        "product": {"artifact_content_sha256": product["artifact"]["content_sha256"]},
    }
    scaffold_sha256 = publish_immutable_json(receipt_output, scaffold)

    profile_copy = json.loads(json.dumps(profile, allow_nan=False))
    profile_copy["promotion"] = {
        "receipt": os.fspath(receipt_output),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": EPHEMERAL_RECEIPT_SCHEMA,
    }
    generator = _load_module("_ullm_sq8_ephemeral_generator", GENERATOR_PATH)
    with tempfile.TemporaryDirectory(
        prefix=".sq8-ephemeral.", dir=manifest_output.parent
    ) as temporary_raw:
        temporary = Path(temporary_raw)
        temporary_profile = temporary / "profile.json"
        temporary_manifest = temporary / "manifest.json"
        temporary_profile.write_bytes(_canonical_json(profile_copy))
        try:
            generator.generate_sq8_promotion_ephemeral(
                temporary_profile,
                temporary_manifest,
                source_root=source_root,
            )
        except Exception as error:
            raise PromotionError("SQ8 ephemeral manifest generation failed") from error
        generated, _ = _load_json_file(
            temporary_manifest,
            "generated SQ8 ephemeral manifest",
            maximum=1024 * 1024,
        )
        try:
            summary = _load_module(
                "_ullm_sq8_ephemeral_manifest_validator",
                SERVED_MODEL_VALIDATOR_PATH,
            ).validation_summary(temporary_manifest)
        except Exception as error:
            raise PromotionError(
                "generated SQ8 ephemeral manifest failed strict validation"
            ) from error
    if (
        summary.get("model_id") != MODEL_ID
        or summary.get("format_id") != FORMAT_ID
        or summary.get("worker", {}).get("protocol") != WORKER_PROTOCOL
        or summary.get("worker", {}).get("binary_sha256") != build["worker"]["sha256"]
        or generated.get("reasoning") != reasoning
        or generated.get("promotion")
        != {
            "source_commit": build["source"]["commit"],
            "receipt": os.fspath(receipt_output),
            "receipt_sha256": scaffold_sha256,
        }
    ):
        fail("generated SQ8 ephemeral manifest identity differs")
    manifest_sha256 = publish_immutable_json(manifest_output, generated)
    try:
        final_summary = _load_module(
            "_ullm_sq8_ephemeral_manifest_validator", SERVED_MODEL_VALIDATOR_PATH
        ).validation_summary(manifest_output)
    except Exception as error:
        raise PromotionError(
            "published SQ8 ephemeral manifest failed strict validation"
        ) from error
    if final_summary.get("manifest_sha256") != manifest_sha256:
        fail("published SQ8 ephemeral manifest SHA-256 differs")
    _validate_clean_detached_source(
        build["source"],
        verify_live=verify_live_source,
        live_root=source_root,
        relocatable=build["schema_version"] == BUILD_RECEIPT_SCHEMA,
    )
    return {
        "receipt": scaffold,
        "receipt_path": os.fspath(receipt_output),
        "receipt_sha256": scaffold_sha256,
        "manifest": generated,
        "manifest_path": os.fspath(manifest_output),
        "manifest_sha256": manifest_sha256,
    }


def _validate_evidence_sources(
    value: Any, *, source_root: Path, verify_live: bool
) -> list[dict[str, str]]:
    if type(value) is not list or len(value) != len(EVIDENCE_SOURCE_PATHS):
        fail("SQ8 serving evidence source set differs")
    result: list[dict[str, str]] = []
    for expected_path, raw in zip(EVIDENCE_SOURCE_PATHS, value, strict=True):
        entry = _exact(raw, {"path", "sha256"}, f"evidence source {expected_path}")
        if entry["path"] != expected_path:
            fail("SQ8 serving evidence source order or path differs")
        digest = _hash(entry["sha256"], f"evidence source {expected_path} SHA-256")
        if verify_live:
            source = _safe_repo_file(
                source_root, expected_path, f"evidence source {expected_path}"
            )
            if stable_hash(source, f"evidence source {expected_path}")[1] != digest:
                fail(f"evidence source {expected_path} SHA-256 differs")
            running_source = _safe_repo_file(
                ROOT, expected_path, f"running evidence source {expected_path}"
            )
            if (
                stable_hash(running_source, f"running evidence source {expected_path}")[
                    1
                ]
                != digest
            ):
                fail(f"running evidence source {expected_path} SHA-256 differs")
        result.append({"path": expected_path, "sha256": digest})
    return result


def _evidence_source_entries(source_root: Path) -> list[dict[str, str]]:
    return [
        {
            "path": relative,
            "sha256": stable_hash(
                _safe_repo_file(source_root, relative, f"evidence source {relative}"),
                f"evidence source {relative}",
            )[1],
        }
        for relative in EVIDENCE_SOURCE_PATHS
    ]


def _product_evidence(
    profile: dict[str, Any],
    product_result: dict[str, Any],
) -> dict[str, Any]:
    product_root = _canonical_absolute(
        Path(_text(profile["product"]["root"], "SQ8 product root")),
        "SQ8 product root",
    )
    result = _validate_product_result(product_result, product_root=product_root)
    if result["model_revision"] != profile["public"]["revision"]:
        fail("SQ8 product and profile model revisions differ")
    product_receipt = product_root / "promotion.json"
    artifact_manifest = product_root / profile["product"]["artifact"]["manifest_path"]
    package_manifest = product_root / profile["product"]["package"]["manifest_path"]
    receipt_hash = stable_hash(
        product_receipt,
        "SQ8 product receipt",
        required_mode=0o444,
        required_nlink=1,
    )[1]
    artifact_hash = stable_hash(
        artifact_manifest,
        "SQ8 artifact manifest",
        required_mode=0o444,
        required_nlink=1,
    )[1]
    package_hash = stable_hash(
        package_manifest,
        "SQ8 package manifest",
        required_mode=0o444,
        required_nlink=1,
    )[1]
    if (
        artifact_hash != result["artifact"]["manifest_sha256"]
        or package_hash != result["package"]["manifest_sha256"]
    ):
        fail("SQ8 validated product manifest hashes differ")
    return {
        "root": os.fspath(product_root),
        "receipt": {
            "path": os.fspath(product_receipt),
            "sha256": receipt_hash,
        },
        "artifact": {
            "manifest_path": profile["product"]["artifact"]["manifest_path"],
            "manifest_sha256": artifact_hash,
            "content_sha256": result["artifact"]["content_sha256"],
        },
        "package": {
            "manifest_path": profile["product"]["package"]["manifest_path"],
            "manifest_sha256": package_hash,
        },
        "validation": result,
        "validation_sha256": hashlib.sha256(_canonical_json(result)).hexdigest(),
    }


def build_evidence(
    *,
    build_receipt_path: Path,
    source_root: Path | None = None,
    profile_path: Path,
    ephemeral_manifest_path: Path,
    cpu_cases_path: Path,
    product_validation: dict[str, Any] | None = None,
    verify_live_source: bool = True,
) -> dict[str, Any]:
    """Construct and fully validate one pre-receipt SQ8 promotion document."""

    build = _validate_promotion_build(
        build_receipt_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    source_root = resolve_build_source_root(build, source_root)
    profile, profile_raw = _load_json_file(profile_path, "SQ8 served-model profile")
    reasoning, final_receipt_path = _profile_contract(profile)
    if final_receipt_path.exists() or final_receipt_path.is_symlink():
        fail("SQ8 production serving receipt already exists")
    worker_path = resolve_build_worker_path(build_receipt_path, build)
    worker_sha256 = build["worker"]["sha256"]
    if (
        _canonical_absolute(
            Path(_text(profile["worker"]["binary"], "SQ8 profile worker")),
            "SQ8 profile worker",
        )
        != worker_path
    ):
        fail("SQ8 profile and build receipt worker paths differ")
    manifest, manifest_raw = _load_json_file(
        ephemeral_manifest_path,
        "SQ8 ephemeral served-model manifest",
        canonical=False,
        required_mode=0o444,
        required_nlink=1,
        maximum=1024 * 1024,
    )
    try:
        summary = _load_module(
            "_ullm_sq8_serving_manifest_validator", SERVED_MODEL_VALIDATOR_PATH
        ).validation_summary(ephemeral_manifest_path)
    except Exception as error:
        raise PromotionError(
            "SQ8 ephemeral served-model manifest failed validation"
        ) from error
    if (
        summary.get("model_id") != MODEL_ID
        or summary.get("format_id") != FORMAT_ID
        or summary.get("worker", {}).get("protocol") != WORKER_PROTOCOL
        or summary.get("worker", {}).get("binary_sha256") != worker_sha256
    ):
        fail("SQ8 ephemeral served-model validator identity differs")
    product_root = _canonical_absolute(
        Path(_text(profile["product"]["root"], "SQ8 product root")),
        "SQ8 product root",
    )
    validation = (
        run_full_product_validation(product_root)
        if product_validation is None
        else product_validation
    )
    product = _product_evidence(profile, validation)
    semantics = _manifest_semantics(
        profile,
        manifest,
        worker_path=worker_path,
        worker_sha256=worker_sha256,
        product_root=product_root,
        product_result=product["validation"],
    )
    if manifest["promotion"]["source_commit"] != build["source"]["commit"]:
        fail("SQ8 ephemeral manifest source commit differs from build")
    cpu = validate_cpu_cases(
        cpu_cases_path,
        source_root=source_root,
        source_commit=build["source"]["commit"],
        source_tree=build["source"]["tree"],
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        worker_sha256=worker_sha256,
        reasoning=reasoning,
    )
    evidence_source = dict(build["source"])
    if build["schema_version"] == BUILD_RECEIPT_SCHEMA:
        # The build-time pathname is audit provenance only.  Promotion
        # evidence names the current sealed checkout used for live validation.
        evidence_source["repository_root"] = os.fspath(source_root)
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "verified": True,
        "production_receipt_written": False,
        "source": {
            **evidence_source,
            "evidence_files": _evidence_source_entries(source_root),
        },
        "worker_build_receipt": {
            "path": os.fspath(_canonical_absolute(build_receipt_path, "build receipt")),
            "sha256": stable_hash(
                build_receipt_path,
                "SQ8 worker build receipt",
                required_mode=0o444,
                required_nlink=1,
            )[1],
            "schema_version": build["schema_version"],
        },
        "worker": {
            "binary": os.fspath(worker_path),
            "bytes": build["worker"]["bytes"],
            "sha256": worker_sha256,
            "protocol": WORKER_PROTOCOL,
            "mode": "0555",
            "nlink": 1,
        },
        "profile": {
            "path": os.fspath(_canonical_absolute(profile_path, "SQ8 profile")),
            "sha256": hashlib.sha256(profile_raw).hexdigest(),
        },
        "ephemeral_manifest": {
            "path": os.fspath(
                _canonical_absolute(
                    ephemeral_manifest_path, "SQ8 ephemeral served-model manifest"
                )
            ),
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "semantic_sha256": hashlib.sha256(_canonical_json(semantics)).hexdigest(),
            "semantics": semantics,
        },
        "product": product,
        "reasoning": reasoning,
        "cpu_cases": {
            "path": os.fspath(
                _canonical_absolute(cpu_cases_path, "SQ8 serving CPU cases")
            ),
            "sha256": stable_hash(
                cpu_cases_path,
                "SQ8 serving CPU cases",
                required_mode=0o444,
                required_nlink=1,
            )[1],
            "schema_version": CPU_CASES_SCHEMA,
            "case_ids": list(CPU_CASE_IDS),
            "case_count": len(CPU_CASE_IDS),
            "report": cpu,
        },
    }
    validate_evidence_document(
        evidence,
        expected_profile_path=profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
        require_receipt_absent=True,
    )
    return evidence


def _validate_reference(
    value: Any,
    label: str,
    *,
    required_mode: int | None = None,
    required_nlink: int | None = None,
    maximum: int = MAX_JSON_BYTES,
) -> tuple[Path, bytes]:
    reference = _exact(value, {"path", "sha256"}, label)
    path = _canonical_absolute(
        Path(_text(reference["path"], f"{label} path")), f"{label} path"
    )
    raw = stable_read(
        path,
        label,
        maximum=maximum,
        required_mode=required_mode,
        required_nlink=required_nlink,
    )
    if hashlib.sha256(raw).hexdigest() != _hash(
        reference["sha256"], f"{label} SHA-256"
    ):
        fail(f"{label} SHA-256 differs")
    return path, raw


def validate_evidence_document(
    document: Any,
    *,
    expected_profile_path: Path | None = None,
    source_root: Path | None = None,
    verify_live_source: bool = True,
    require_receipt_absent: bool = False,
) -> dict[str, Any]:
    evidence = _exact(
        document,
        {
            "schema_version",
            "verified",
            "production_receipt_written",
            "source",
            "worker_build_receipt",
            "worker",
            "profile",
            "ephemeral_manifest",
            "product",
            "reasoning",
            "cpu_cases",
        },
        "SQ8 serving promotion evidence",
    )
    if (
        evidence["schema_version"] != EVIDENCE_SCHEMA
        or evidence["verified"] is not True
        or evidence["production_receipt_written"] is not False
    ):
        fail("SQ8 serving promotion evidence state differs")
    source = _exact(
        evidence["source"],
        {
            "repository_root",
            "commit",
            "tree",
            "detached",
            "worktree_clean",
            "status_sha256",
            "evidence_files",
        },
        "SQ8 serving promotion source",
    )
    build_source = {
        key: source[key]
        for key in (
            "repository_root",
            "commit",
            "tree",
            "detached",
            "worktree_clean",
            "status_sha256",
        )
    }
    build_reference = _exact(
        evidence["worker_build_receipt"],
        {"path", "sha256", "schema_version"},
        "SQ8 worker build-receipt reference",
    )
    if build_reference["schema_version"] not in {
        BUILD_RECEIPT_SCHEMA_V1,
        BUILD_RECEIPT_SCHEMA,
    }:
        fail("SQ8 worker build-receipt reference schema differs")
    build_path = _canonical_absolute(
        Path(_text(build_reference["path"], "SQ8 worker build-receipt path")),
        "SQ8 worker build-receipt path",
    )
    build_raw = stable_read(
        build_path,
        "SQ8 worker build receipt",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(build_raw).hexdigest() != _hash(
        build_reference["sha256"], "SQ8 worker build-receipt SHA-256"
    ):
        fail("SQ8 worker build-receipt SHA-256 differs")
    build = _validate_promotion_build(
        build_path,
        verify_live_source=verify_live_source,
        source_root=source_root,
    )
    source_identity_fields = {
        "commit",
        "tree",
        "detached",
        "worktree_clean",
        "status_sha256",
    }
    if (
        build["schema_version"] != build_reference["schema_version"]
        or any(
            build["source"][field] != build_source[field]
            for field in source_identity_fields
        )
        or (
            build["schema_version"] == BUILD_RECEIPT_SCHEMA_V1
            and build["source"]["repository_root"]
            != build_source["repository_root"]
        )
    ):
        fail("SQ8 evidence and build-receipt source identities differ")
    current_source_root = resolve_build_source_root(build, source_root)
    if (
        build["schema_version"] == BUILD_RECEIPT_SCHEMA
        and build_source["repository_root"] != os.fspath(current_source_root)
    ):
        fail("SQ8 evidence current source root differs")
    _validate_evidence_sources(
        source["evidence_files"],
        source_root=current_source_root,
        verify_live=verify_live_source,
    )
    build_worker_path = resolve_build_worker_path(build_path, build)

    worker = _exact(
        evidence["worker"],
        {"binary", "bytes", "sha256", "protocol", "mode", "nlink"},
        "SQ8 serving worker",
    )
    if (
        worker["binary"] != os.fspath(build_worker_path)
        or worker["bytes"] != build["worker"]["bytes"]
        or worker["sha256"] != build["worker"]["sha256"]
        or worker["protocol"] != WORKER_PROTOCOL
        or worker["mode"] != "0555"
        or worker["nlink"] != 1
    ):
        fail("SQ8 evidence and build-receipt worker identities differ")
    stable_hash(
        Path(worker["binary"]),
        "SQ8 serving worker",
        required_mode=0o555,
        required_nlink=1,
    )

    profile_path, profile_raw = _validate_reference(
        evidence["profile"], "SQ8 served-model profile"
    )
    if expected_profile_path is not None and profile_path != _canonical_absolute(
        expected_profile_path, "expected SQ8 served-model profile"
    ):
        fail("SQ8 served-model profile path differs")
    profile = _strict_json(profile_raw, "SQ8 served-model profile")
    reasoning, final_receipt_path = _profile_contract(profile)
    if require_receipt_absent and (
        final_receipt_path.exists() or final_receipt_path.is_symlink()
    ):
        fail("SQ8 production serving receipt existed during pre-receipt validation")
    if evidence["reasoning"] != reasoning:
        fail("SQ8 promotion reasoning binding differs")

    ephemeral = _exact(
        evidence["ephemeral_manifest"],
        {"path", "sha256", "semantic_sha256", "semantics"},
        "SQ8 ephemeral manifest evidence",
    )
    ephemeral_path = _canonical_absolute(
        Path(_text(ephemeral["path"], "SQ8 ephemeral manifest path")),
        "SQ8 ephemeral manifest path",
    )
    manifest_raw = stable_read(
        ephemeral_path,
        "SQ8 ephemeral served-model manifest",
        maximum=1024 * 1024,
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(manifest_raw).hexdigest() != _hash(
        ephemeral["sha256"], "SQ8 ephemeral manifest SHA-256"
    ):
        fail("SQ8 ephemeral manifest SHA-256 differs")
    manifest = _strict_json(manifest_raw, "SQ8 ephemeral served-model manifest")

    product = _exact(
        evidence["product"],
        {
            "root",
            "receipt",
            "artifact",
            "package",
            "validation",
            "validation_sha256",
        },
        "SQ8 serving product evidence",
    )
    product_root = _canonical_absolute(
        Path(_text(product["root"], "SQ8 product root")), "SQ8 product root"
    )
    if product_root != _canonical_absolute(
        Path(_text(profile["product"]["root"], "SQ8 profile product root")),
        "SQ8 profile product root",
    ):
        fail("SQ8 product root differs from profile")
    product_result = _validate_product_result(
        product["validation"], product_root=product_root
    )
    if hashlib.sha256(_canonical_json(product_result)).hexdigest() != _hash(
        product["validation_sha256"], "SQ8 product validation SHA-256"
    ):
        fail("SQ8 product validation SHA-256 differs")
    receipt_path, _ = _validate_reference(
        product["receipt"],
        "SQ8 product receipt",
        required_mode=0o444,
        required_nlink=1,
    )
    if receipt_path != product_root / "promotion.json":
        fail("SQ8 product receipt path differs")
    artifact = _exact(
        product["artifact"],
        {"manifest_path", "manifest_sha256", "content_sha256"},
        "SQ8 product artifact evidence",
    )
    package = _exact(
        product["package"],
        {"manifest_path", "manifest_sha256"},
        "SQ8 product package evidence",
    )
    if (
        product_result["model_revision"] != profile["public"]["revision"]
        or artifact["manifest_sha256"] != product_result["artifact"]["manifest_sha256"]
        or artifact["content_sha256"] != product_result["artifact"]["content_sha256"]
        or package["manifest_sha256"] != product_result["package"]["manifest_sha256"]
        or artifact["manifest_path"] != profile["product"]["artifact"]["manifest_path"]
        or package["manifest_path"] != profile["product"]["package"]["manifest_path"]
        or stable_hash(
            product_root / artifact["manifest_path"],
            "SQ8 artifact manifest",
            required_mode=0o444,
            required_nlink=1,
        )[1]
        != artifact["manifest_sha256"]
        or stable_hash(
            product_root / package["manifest_path"],
            "SQ8 package manifest",
            required_mode=0o444,
            required_nlink=1,
        )[1]
        != package["manifest_sha256"]
    ):
        fail("SQ8 product evidence identity differs")

    semantics = _manifest_semantics(
        profile,
        manifest,
        worker_path=Path(worker["binary"]),
        worker_sha256=worker["sha256"],
        product_root=product_root,
        product_result=product_result,
    )
    if ephemeral["semantics"] != semantics or hashlib.sha256(
        _canonical_json(semantics)
    ).hexdigest() != _hash(
        ephemeral["semantic_sha256"], "SQ8 manifest semantic SHA-256"
    ):
        fail("SQ8 ephemeral manifest semantic identity differs")
    if manifest["promotion"]["source_commit"] != source["commit"]:
        fail("SQ8 ephemeral manifest source commit differs")

    cpu = _exact(
        evidence["cpu_cases"],
        {
            "path",
            "sha256",
            "schema_version",
            "case_ids",
            "case_count",
            "report",
        },
        "SQ8 serving CPU-case evidence",
    )
    if (
        cpu["schema_version"] != CPU_CASES_SCHEMA
        or cpu["case_ids"] != list(CPU_CASE_IDS)
        or cpu["case_count"] != len(CPU_CASE_IDS)
    ):
        fail("SQ8 serving CPU-case evidence identity differs")
    cpu_path = _canonical_absolute(
        Path(_text(cpu["path"], "SQ8 serving CPU-case path")),
        "SQ8 serving CPU-case path",
    )
    cpu_raw = stable_read(
        cpu_path,
        "SQ8 serving CPU cases",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(cpu_raw).hexdigest() != _hash(
        cpu["sha256"], "SQ8 serving CPU-case SHA-256"
    ):
        fail("SQ8 serving CPU-case SHA-256 differs")
    observed_cpu = validate_cpu_cases(
        cpu_path,
        source_root=current_source_root,
        source_commit=source["commit"],
        source_tree=source["tree"],
        manifest_sha256=ephemeral["sha256"],
        worker_sha256=worker["sha256"],
        reasoning=reasoning,
    )
    if cpu["report"] != observed_cpu:
        fail("SQ8 serving CPU-case embedded report differs")
    return evidence


def validate_evidence(
    path: Path,
    *,
    expected_profile_path: Path | None = None,
    source_root: Path | None = None,
    verify_live_source: bool = True,
    require_receipt_absent: bool = False,
) -> dict[str, Any]:
    document, _ = _load_json_file(
        path,
        "SQ8 serving promotion evidence",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    return validate_evidence_document(
        document,
        expected_profile_path=expected_profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
        require_receipt_absent=require_receipt_absent,
    )


def _receipt_product_from_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    product = evidence["product"]
    product_receipt = product["receipt"]
    if Path(product_receipt["path"]).name != "promotion.json":
        fail("SQ8 product receipt filename differs")
    return {
        "receipt": {
            "path": "promotion.json",
            "sha256": product_receipt["sha256"],
        },
        "artifact_manifest_sha256": product["artifact"]["manifest_sha256"],
        "artifact_content_sha256": product["artifact"]["content_sha256"],
        "package_manifest_sha256": product["package"]["manifest_sha256"],
    }


def validate_receipt_document(
    document: Any,
    *,
    receipt_path: Path,
    expected_evidence_path: Path | None = None,
    expected_profile_path: Path | None = None,
    source_root: Path | None = None,
    verify_live_source: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _exact(
        document,
        {"schema_version", "source_commit", "evidence", "product"},
        "SQ8 serving promotion receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA:
        fail("SQ8 serving promotion receipt schema differs")
    source_commit = _git_object(
        receipt["source_commit"], "SQ8 serving promotion receipt source commit"
    )
    reference = _exact(
        receipt["evidence"],
        {"path", "sha256"},
        "SQ8 serving receipt evidence reference",
    )
    relative = _safe_relative(reference["path"], "SQ8 serving receipt evidence path")
    candidate = receipt_path.parent.joinpath(*relative.parts)
    evidence_path = _canonical_absolute(candidate, "SQ8 serving receipt evidence path")
    try:
        evidence_path.relative_to(receipt_path.parent)
    except ValueError:
        fail("SQ8 serving receipt evidence escapes its directory")
    if expected_evidence_path is not None and evidence_path != _canonical_absolute(
        expected_evidence_path, "expected SQ8 promotion evidence"
    ):
        fail("SQ8 serving receipt evidence path differs")
    evidence_raw = stable_read(
        evidence_path,
        "SQ8 serving promotion evidence",
        required_mode=0o444,
        required_nlink=1,
    )
    if hashlib.sha256(evidence_raw).hexdigest() != _hash(
        reference["sha256"], "SQ8 serving receipt evidence SHA-256"
    ):
        fail("SQ8 serving receipt evidence SHA-256 differs")
    evidence = validate_evidence(
        evidence_path,
        expected_profile_path=expected_profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
    )
    if source_commit != evidence["source"]["commit"]:
        fail("SQ8 serving receipt and evidence source commits differ")
    expected_product = _receipt_product_from_evidence(evidence)
    if receipt["product"] != expected_product:
        fail("SQ8 serving receipt product identity differs")
    product_receipt = (
        evidence["product"]["root"] + "/" + expected_product["receipt"]["path"]
    )
    if Path(product_receipt) != Path(evidence["product"]["receipt"]["path"]):
        fail("SQ8 serving receipt product-receipt path differs")
    return receipt, evidence


def validate_receipt(
    path: Path,
    *,
    expected_evidence_path: Path | None = None,
    expected_profile_path: Path | None = None,
    source_root: Path | None = None,
    verify_live_source: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt_path = _canonical_absolute(path, "SQ8 serving promotion receipt")
    document, _ = _load_json_file(
        receipt_path,
        "SQ8 serving promotion receipt",
        canonical=True,
        required_mode=0o444,
        required_nlink=1,
    )
    return validate_receipt_document(
        document,
        receipt_path=receipt_path,
        expected_evidence_path=expected_evidence_path,
        expected_profile_path=expected_profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    operation = getattr(libc, "renameat2", None)
    if operation is None:
        fail("renameat2 is required for immutable publication")
    operation.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    operation.restype = ctypes.c_int
    if (
        operation(
            AT_FDCWD,
            os.fsencode(source),
            AT_FDCWD,
            os.fsencode(destination),
            RENAME_NOREPLACE,
        )
        == 0
    ):
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        fail("immutable publication destination already exists")
    raise PromotionError("immutable no-replace publication failed") from OSError(
        error, os.strerror(error)
    )


def publish_immutable_json(path: Path, document: dict[str, Any]) -> str:
    raw = _canonical_json(document)
    destination = _canonical_absolute(
        path, "immutable publication destination", may_be_absent=True
    )
    if destination.exists() or destination.is_symlink():
        fail("immutable publication destination already exists")
    parent = destination.parent
    metadata = parent.stat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & stat.S_IWOTH:
        fail("immutable publication directory is unsafe")
    temporary: Path | None = None
    descriptor = -1
    published = False
    try:
        descriptor, temporary_raw = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=parent
        )
        temporary = Path(temporary_raw)
        os.fchmod(descriptor, 0o444)
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                fail("immutable publication write made no progress")
            offset += written
        os.fsync(descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or before.st_size != len(raw)
        ):
            fail("immutable publication temporary identity differs")
        os.close(descriptor)
        descriptor = -1
        _rename_noreplace(temporary, destination)
        published = True
        temporary = None
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        observed = stable_read(
            destination,
            "published immutable JSON",
            required_mode=0o444,
            required_nlink=1,
        )
        if observed != raw:
            fail("published immutable JSON bytes differ")
        return hashlib.sha256(observed).hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        # A successfully renamed file is never removed here.  A post-publication
        # verification failure must remain visible and cannot authorize a retry.
        del published


def write_receipt(
    *,
    profile_path: Path,
    evidence_path: Path,
    output_path: Path,
    source_root: Path | None = None,
    verify_live_source: bool = True,
) -> dict[str, Any]:
    profile, _ = _load_json_file(profile_path, "SQ8 served-model profile")
    _, expected_receipt = _profile_contract(profile)
    output = _canonical_absolute(
        output_path, "SQ8 serving receipt output", may_be_absent=True
    )
    if output != expected_receipt:
        fail("SQ8 serving receipt output differs from the profile")
    if os.path.lexists(output):
        fail("SQ8 production serving receipt already exists")
    evidence = validate_evidence(
        evidence_path,
        expected_profile_path=profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
        require_receipt_absent=True,
    )
    evidence = dict(evidence)
    try:
        relative = _canonical_absolute(
            evidence_path, "SQ8 serving promotion evidence"
        ).relative_to(output.parent)
    except ValueError:
        fail("SQ8 serving promotion evidence is outside the receipt directory")
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        fail("SQ8 serving promotion evidence relative path is unsafe")
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "source_commit": evidence["source"]["commit"],
        "evidence": {
            "path": PurePosixPath(*relative.parts).as_posix(),
            "sha256": stable_hash(
                evidence_path,
                "SQ8 serving promotion evidence",
                required_mode=0o444,
                required_nlink=1,
            )[1],
        },
        "product": _receipt_product_from_evidence(evidence),
    }
    # Validate the semantic pair before the destination exists.  The validator
    # accepts the intended absolute path because only the evidence is dereferenced.
    validate_receipt_document(
        receipt,
        receipt_path=output,
        expected_evidence_path=evidence_path,
        expected_profile_path=profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
    )
    publish_immutable_json(output, receipt)
    observed, _ = validate_receipt(
        output,
        expected_evidence_path=evidence_path,
        expected_profile_path=profile_path,
        source_root=source_root,
        verify_live_source=verify_live_source,
    )
    return observed


def validate_generator_binding(
    *,
    evidence_path: Path,
    receipt: dict[str, Any],
    receipt_path: Path,
    profile_path: Path,
    source_commit: str,
    source_root: Path | None = None,
    worker_binary: Path,
    worker_sha256: str,
    manifest: dict[str, Any],
) -> None:
    """Validate the SQ8 receipt/evidence pair during manifest generation."""

    canonical_receipt_path = _canonical_absolute(
        receipt_path, "SQ8 serving promotion receipt"
    )
    receipt_raw = stable_read(
        canonical_receipt_path,
        "SQ8 serving promotion receipt",
        required_mode=0o444,
        required_nlink=1,
    )
    live_receipt = _strict_json(receipt_raw, "SQ8 serving promotion receipt")
    if (
        receipt_raw != _canonical_json(live_receipt)
        or live_receipt != receipt
        or manifest.get("promotion")
        != {
            "source_commit": source_commit,
            "receipt": os.fspath(canonical_receipt_path),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
        }
    ):
        fail("SQ8 generator live promotion receipt binding differs")
    receipt_document, evidence = validate_receipt_document(
        live_receipt,
        receipt_path=canonical_receipt_path,
        expected_evidence_path=evidence_path,
        expected_profile_path=profile_path,
        source_root=source_root,
        verify_live_source=True,
    )
    if (
        receipt_document["source_commit"] != source_commit
        or evidence["worker"]["binary"] != os.fspath(worker_binary)
        or evidence["worker"]["sha256"] != worker_sha256
    ):
        fail("SQ8 generator promotion source or worker identity differs")
    semantics = {
        key: manifest[key]
        for key in (
            "schema_version",
            "public",
            "generation",
            "format",
            "tokenizer",
            "worker",
            "product",
            "reasoning",
        )
    }
    if (
        evidence["ephemeral_manifest"]["semantics"] != semantics
        or evidence["reasoning"] != manifest["reasoning"]
    ):
        fail("SQ8 generator manifest semantics differ from promotion evidence")
    product = evidence["product"]
    receipt_product = receipt_document["product"]
    if (
        manifest["product"]["artifact"]["content_sha256"]
        != receipt_product["artifact_content_sha256"]
        or manifest["product"]["artifact"]["manifest_sha256"]
        != receipt_product["artifact_manifest_sha256"]
        or manifest["product"]["package"]["manifest_sha256"]
        != receipt_product["package_manifest_sha256"]
        or receipt_product != _receipt_product_from_evidence(evidence)
        or product["root"] != manifest["product"]["root"]
    ):
        fail("SQ8 generator product identity differs from promotion evidence")


__all__ = [
    "BUILD_INPUTS_V2",
    "BUILD_REJECTED_ENVIRONMENT_V2",
    "BUILD_RECEIPT_SCHEMA",
    "BUILD_RECEIPT_SCHEMA_V1",
    "BUILD_RELEASE_SEAL_SCHEMA",
    "BUILD_WORKER_RELATIVE_PATH",
    "CPU_CASES_SCHEMA",
    "CPU_CASE_IDS",
    "CPU_PYTEST_TESTS",
    "CPU_RUST_TESTS",
    "EVIDENCE_SCHEMA",
    "PromotionError",
    "REASONING_CONTRACT",
    "RECEIPT_SCHEMA",
    "build_cpu_cases_report",
    "build_evidence",
    "prepare_ephemeral_manifest",
    "publish_immutable_json",
    "run_full_product_validation",
    "stable_hash",
    "stable_read",
    "resolve_build_source_root",
    "resolve_build_worker_path",
    "validate_build_release",
    "validate_build_receipt",
    "validate_cpu_cases",
    "validate_cpu_cases_document",
    "validate_evidence",
    "validate_evidence_document",
    "validate_generator_binding",
    "validate_receipt",
    "validate_receipt_document",
    "write_receipt",
]
