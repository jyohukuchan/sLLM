#!/usr/bin/env python3
"""Assemble hash-only generic reasoning release evidence from measured cases."""

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
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools/validate-generic-reasoning-release.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
MAX_CASES_BYTES = 16 * 1024 * 1024
MAX_CAMPAIGN_FILE_BYTES = 16 * 1024 * 1024
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z")
FORBIDDEN_KEYS = {
    "prompt",
    "response",
    "request_body",
    "response_body",
    "authorization",
    "api_key",
    "token",
    "conversation",
}
EVIDENCE_SCHEMA_V1 = "ullm.generic_reasoning_release_evidence.v1"
EVIDENCE_SCHEMA_V2 = "ullm.generic_reasoning_release_evidence.v2"
CAMPAIGN_LINEAGE_SCHEMA_V2 = "ullm.served_model.campaign_lineage.v2"
REASONING_CAMPAIGN_SCHEMA_V2 = "ullm.generic_reasoning_release_campaign.v2"
ACTIVE_BINDING_SCHEMA = "ullm.served_model.active_binding.v1"
ACTIVE_OBSERVATION_SCHEMA = "ullm.served_model.active_manifest_observation.v1"
REASONING_CAMPAIGN_STAGES = (
    "preflight",
    *(
        stage
        for mode in ("disabled", "budget-32", "budget-128", "budget-256", "unbounded")
        for stage in (f"{mode}:stream", f"{mode}:nonstream")
    ),
    "final",
)
REASONING_CAMPAIGN_FILES = frozenset(
    {
        "cases.json",
        "lifecycle.json",
        "resource-samples.jsonl",
        "summary.json",
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)


class EvidenceError(RuntimeError):
    """Raised when measured cases cannot be safely assembled."""


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("input cases contain duplicate fields")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    raise EvidenceError("input cases contain a non-finite number")


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_read(
    path: Path,
    label: str,
    maximum: int,
    *,
    require_immutable: bool = False,
) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceError("O_NOFOLLOW is required for evidence preparation")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise EvidenceError(f"{label} must be a regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
            or (
                require_immutable
                and (
                    stat.S_IMODE(before.st_mode) != 0o444
                    or before.st_nlink != 1
                )
            )
        ):
            raise EvidenceError(f"{label} file identity differs")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, maximum + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        try:
            named = path.lstat()
        except OSError as error:
            raise EvidenceError(f"{label} disappeared while being read") from error
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(named)
        ):
            raise EvidenceError(f"{label} changed while being read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"{label} is not strict JSON") from error


def _read_json(path: Path) -> Any:
    return _decode_json(
        _stable_read(path, "input cases", MAX_CASES_BYTES),
        "input cases",
    )


def _scan_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_KEYS:
                raise EvidenceError(f"input cases contain forbidden field: {key}")
            _scan_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _scan_forbidden(child)


def _hash_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise EvidenceError(f"file is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError(f"failed to hash file: {path}") from error
    return digest.hexdigest()


def _validate_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise EvidenceError(f"{label} is not a lowercase Git commit")
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError("failed to resolve Git HEAD") from error
    if result.returncode != 0:
        raise EvidenceError("failed to resolve Git HEAD")
    return _validate_commit(result.stdout.strip(), "source_commit")


def _git_status() -> bytes:
    command = [
        "git",
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude).rocprofv3",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError("failed to inspect Git worktree") from error
    if result.returncode != 0:
        raise EvidenceError("failed to inspect Git worktree")
    return bytes(result.stdout)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise EvidenceError("served-model manifest is not an object")
    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("root"), str):
        raise EvidenceError("served-model manifest has no tokenizer root")
    files = tokenizer.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceError("served-model manifest has no tokenizer file map")
    root = Path(tokenizer["root"])
    if not root.is_absolute():
        root = path.parent / root
    return value, os.fspath(root)


def _validate_served_model_manifest(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "_ullm_generic_reasoning_served_model_validator",
        SERVED_MODEL_VALIDATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise EvidenceError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        module.validation_summary(path)
    except Exception as error:
        raise EvidenceError("served-model manifest failed validation") from error


def _tokenizer_identity(manifest: dict[str, Any], root: Path) -> str:
    files = manifest["tokenizer"]["files"]
    digest = hashlib.sha256()
    root = root.resolve()
    for name in sorted(files):
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            raise EvidenceError("tokenizer file name is unsafe")
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise EvidenceError("tokenizer file escapes its root") from error
        observed = _hash_file(path)
        _validate_hash(files[name], f"manifest tokenizer file {name}")
        if observed != files[name]:
            raise EvidenceError(f"tokenizer file hash differs: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(observed))
    return digest.hexdigest()


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_ullm_generic_reasoning_release_preparer_validator", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise EvidenceError("generic release validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact_reference(raw: bytes) -> dict[str, Any]:
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _inventory_sha256(artifacts: dict[str, dict[str, Any]]) -> str:
    canonical = json.dumps(
        artifacts,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(
        CAMPAIGN_LINEAGE_SCHEMA_V2.encode("ascii") + b"\0" + canonical
    ).hexdigest()


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise EvidenceError(f"{label} fields differ")
    return value


def _campaign_lineage(
    campaign_output_dir: Path,
    *,
    cases_path: Path,
    lifecycle_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Recompute the immutable v2 campaign/claim/active-byte lineage."""

    absolute = campaign_output_dir.absolute()
    try:
        root = campaign_output_dir.resolve(strict=True)
        root_metadata = root.lstat()
        observed_names = {entry.name for entry in os.scandir(root)}
    except OSError as error:
        raise EvidenceError("v2 campaign output is unavailable") from error
    if (
        root != absolute
        or stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o555
        or observed_names != REASONING_CAMPAIGN_FILES
    ):
        raise EvidenceError("v2 campaign output layout or mode differs")
    expected_paths = {
        "cases.json": cases_path,
        "lifecycle.json": lifecycle_path,
    }
    for name, supplied in expected_paths.items():
        try:
            if supplied.resolve(strict=True) != root / name:
                raise EvidenceError(f"v2 campaign {name} input path differs")
        except OSError as error:
            raise EvidenceError(f"v2 campaign {name} input is unavailable") from error

    raws = {
        name: _stable_read(
            root / name,
            f"v2 campaign {name}",
            MAX_CAMPAIGN_FILE_BYTES,
            require_immutable=True,
        )
        for name in sorted(REASONING_CAMPAIGN_FILES)
    }
    manifest_raw = _stable_read(
        manifest_path,
        "served-model manifest",
        MAX_CAMPAIGN_FILE_BYTES,
    )
    if manifest_raw != raws["candidate-served-model.json"]:
        raise EvidenceError("served-model manifest differs from campaign candidate")

    summary = _decode_json(raws["summary.json"], "v2 campaign summary")
    binding = _decode_json(
        raws["active-manifest-binding.json"],
        "v2 campaign active binding",
    )
    _exact_object(
        binding,
        {
            "schema_version",
            "status",
            "candidate",
            "actual_active_path",
            "expected_stages",
            "observation_count",
            "observations",
            "claim",
            "campaign",
        },
        "v2 campaign active binding",
    )
    if (
        binding["schema_version"] != ACTIVE_BINDING_SCHEMA
        or binding["status"] != "complete"
        or binding["expected_stages"] != list(REASONING_CAMPAIGN_STAGES)
        or binding["observation_count"] != len(REASONING_CAMPAIGN_STAGES)
    ):
        raise EvidenceError("v2 campaign active binding contract differs")
    candidate = _exact_object(
        binding["candidate"],
        {"artifact", "source_path", "sha256", "bytes"},
        "v2 campaign binding candidate",
    )
    observations_reference = _exact_object(
        binding["observations"],
        {"artifact", "sha256", "bytes"},
        "v2 campaign binding observations",
    )
    claim = _exact_object(
        binding["claim"],
        {
            "path",
            "sha256",
            "bytes",
            "authorization_path",
            "authorization_sha256",
        },
        "v2 campaign claim",
    )
    campaign = _exact_object(
        binding["campaign"],
        {"name", "run_id", "final_path"},
        "v2 campaign identity",
    )
    candidate_raw = raws["candidate-served-model.json"]
    observations_raw = raws["active-manifest-observations.jsonl"]
    if (
        candidate["artifact"] != "candidate-served-model.json"
        or candidate["sha256"] != hashlib.sha256(candidate_raw).hexdigest()
        or candidate["bytes"] != len(candidate_raw)
        or observations_reference["artifact"]
        != "active-manifest-observations.jsonl"
        or observations_reference["sha256"]
        != hashlib.sha256(observations_raw).hexdigest()
        or observations_reference["bytes"] != len(observations_raw)
        or campaign["name"] != "reasoning_release"
        or campaign["final_path"] != os.fspath(root)
        or not isinstance(campaign["run_id"], str)
        or not campaign["run_id"]
    ):
        raise EvidenceError("v2 campaign candidate, observations, or output differs")
    for field in ("sha256", "authorization_sha256"):
        _validate_hash(claim[field], f"v2 campaign claim.{field}")
    for field in ("path", "authorization_path"):
        if not isinstance(claim[field], str) or not Path(claim[field]).is_absolute():
            raise EvidenceError(f"v2 campaign claim.{field} is invalid")
    if type(claim["bytes"]) is not int or claim["bytes"] < 1:
        raise EvidenceError("v2 campaign claim.bytes is invalid")
    claim_raw = _stable_read(
        Path(claim["path"]),
        "v2 campaign authorization claim",
        1_048_576,
        require_immutable=True,
    )
    authorization_raw = _stable_read(
        Path(claim["authorization_path"]),
        "v2 campaign authorization",
        1_048_576,
        require_immutable=True,
    )
    if (
        len(claim_raw) != claim["bytes"]
        or hashlib.sha256(claim_raw).hexdigest() != claim["sha256"]
        or hashlib.sha256(authorization_raw).hexdigest()
        != claim["authorization_sha256"]
    ):
        raise EvidenceError("v2 campaign authorization bytes differ")

    observation_lines = observations_raw.splitlines(keepends=True)
    if (
        len(observation_lines) != len(REASONING_CAMPAIGN_STAGES)
        or any(not line.endswith(b"\n") for line in observation_lines)
    ):
        raise EvidenceError("v2 campaign observation line count differs")
    stage_digests: list[dict[str, Any]] = []
    for sequence, (raw_line, expected_stage) in enumerate(
        zip(observation_lines, REASONING_CAMPAIGN_STAGES, strict=True)
    ):
        row = _decode_json(raw_line, f"v2 campaign observation {sequence}")
        _exact_object(
            row,
            {
                "schema_version",
                "sequence",
                "stage",
                "observed_unix_ns",
                "observed_monotonic_ns",
                "candidate",
                "active",
                "bytes_equal",
                "claim",
            },
            f"v2 campaign observation {sequence}",
        )
        candidate_row = _exact_object(
            row["candidate"],
            {"path", "sha256", "identity"},
            f"v2 campaign observation {sequence} candidate",
        )
        active_row = _exact_object(
            row["active"],
            {"path", "sha256", "identity"},
            f"v2 campaign observation {sequence} active",
        )
        if (
            row["schema_version"] != ACTIVE_OBSERVATION_SCHEMA
            or row["sequence"] != sequence
            or row["stage"] != expected_stage
            or row["bytes_equal"] is not True
            or row["claim"] != claim
            or candidate_row["path"] != candidate["source_path"]
            or candidate_row["sha256"] != candidate["sha256"]
            or active_row["path"] != binding["actual_active_path"]
            or active_row["sha256"] != candidate["sha256"]
        ):
            raise EvidenceError(
                f"v2 campaign observation {sequence} binding differs"
            )
        for label, file_row in (("candidate", candidate_row), ("active", active_row)):
            identity = _exact_object(
                file_row["identity"],
                {
                    "device",
                    "inode",
                    "mode",
                    "links",
                    "uid",
                    "gid",
                    "bytes",
                    "mtime_ns",
                    "ctime_ns",
                },
                f"v2 campaign observation {sequence} {label} identity",
            )
            if (
                any(type(value) is not int or value < 0 for value in identity.values())
                or identity["bytes"] != candidate["bytes"]
            ):
                raise EvidenceError(
                    f"v2 campaign observation {sequence} file identity differs"
                )
        stage_digests.append(
            {
                "sequence": sequence,
                "stage": expected_stage,
                "sha256": hashlib.sha256(raw_line).hexdigest(),
            }
        )

    _exact_object(
        summary,
        {
            "schema_version",
            "status",
            "raw_bodies_stored",
            "case_count",
            "stream_case_count",
            "nonstream_case_count",
            "modes",
            "manifest_sha256",
            "model_id",
            "worker_binary_sha256",
            "gpu_exclusive_preflight",
            "active_manifest_binding",
            "run_id",
        },
        "v2 campaign summary",
    )
    if (
        summary["schema_version"] != REASONING_CAMPAIGN_SCHEMA_V2
        or summary["run_id"] != campaign["run_id"]
        or summary["active_manifest_binding"] != binding
        or summary["manifest_sha256"] != candidate["sha256"]
        or summary["raw_bodies_stored"] is not False
    ):
        raise EvidenceError("v2 campaign summary lineage differs")

    artifacts = {
        name: _artifact_reference(raw)
        for name, raw in sorted(raws.items())
    }
    return {
        "schema_version": CAMPAIGN_LINEAGE_SCHEMA_V2,
        "campaign": {
            "name": "reasoning_release",
            "run_id": campaign["run_id"],
            "final_path": os.fspath(root),
            "final_kind": "directory",
            "files": sorted(REASONING_CAMPAIGN_FILES),
        },
        "claim": dict(claim),
        "artifacts": artifacts,
        "artifact_inventory_sha256": _inventory_sha256(artifacts),
        "observations": {
            "count": len(stage_digests),
            "stages": stage_digests,
        },
    }


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as error:
        raise EvidenceError("output evidence parent is unavailable") from error
    if parent != path.parent.absolute():
        raise EvidenceError("output evidence parent is not canonical")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise EvidenceError("O_NOFOLLOW is required for evidence publication")
    flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, flags)
    temporary_name = f".{path.name}.incomplete-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        raw = (
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
            ).encode("ascii")
            + b"\n"
        )
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvidenceError("output evidence write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise EvidenceError("output evidence already exists") from error
        os.unlink(temporary_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        observed = _stable_read(
            path,
            "published output evidence",
            MAX_CASES_BYTES,
            require_immutable=True,
        )
        if observed != raw:
            raise EvidenceError("published output evidence bytes differ")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def prepare(
    cases_path: Path,
    manifest_path: Path,
    worker_path: Path,
    openwebui_image: str,
    active_promotion_source_commit: str,
    output_path: Path,
    *,
    lifecycle_path: Path | None = None,
    campaign_output_dir: Path | None = None,
    status: str = "incomplete",
) -> dict[str, Any]:
    if status not in {"incomplete", "complete"}:
        raise EvidenceError("evidence status is invalid")
    _validate_commit(active_promotion_source_commit, "active_promotion_source_commit")
    if IMAGE_RE.fullmatch(openwebui_image) is None:
        raise EvidenceError("OpenWebUI image is not content-addressed")
    cases = _read_json(cases_path)
    _scan_forbidden(cases)
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("measured cases must be a nonempty array")
    if len(cases) > 4096:
        raise EvidenceError("measured cases exceed their bound")
    lifecycle = (
        _read_json(lifecycle_path)
        if lifecycle_path is not None
        else {
            "schema_version": "ullm.generic_reasoning_lifecycle_evidence.v1",
            "events": [],
        }
    )
    _scan_forbidden(lifecycle)
    if campaign_output_dir is not None and lifecycle_path is None:
        raise EvidenceError("v2 evidence requires the measured campaign lifecycle")
    _validate_served_model_manifest(manifest_path)
    manifest, tokenizer_root = _load_manifest(manifest_path)
    source_commit = _git_commit()
    status_raw = _git_status()
    worktree_clean = status_raw == b""
    if status == "complete" and not worktree_clean:
        raise EvidenceError("complete evidence requires a clean Git worktree")
    identity = {
        "manifest_sha256": _hash_file(manifest_path),
        "worker_binary_sha256": _hash_file(worker_path),
        "tokenizer_sha256": _tokenizer_identity(manifest, Path(tokenizer_root)),
        "openwebui_image": openwebui_image,
    }
    document: dict[str, Any] = {
        "schema_version": (
            EVIDENCE_SCHEMA_V2
            if campaign_output_dir is not None
            else EVIDENCE_SCHEMA_V1
        ),
        "status": status,
        "production_activation_performed": False,
        "source_commit": source_commit,
        "active_promotion_source_commit": active_promotion_source_commit,
        "source_commit_aligned": source_commit == active_promotion_source_commit,
        "git_worktree_clean": worktree_clean,
        "git_worktree_status_sha256": hashlib.sha256(status_raw).hexdigest(),
        "identity": identity,
        "cases": cases,
        "lifecycle": lifecycle,
    }
    if campaign_output_dir is not None:
        assert lifecycle_path is not None
        document["campaign_lineage"] = _campaign_lineage(
            campaign_output_dir,
            cases_path=cases_path,
            lifecycle_path=lifecycle_path,
            manifest_path=manifest_path,
        )
    validator = _load_validator()
    temporary = output_path.parent / (
        f".{output_path.name}.validate-{uuid.uuid4().hex}"
    )
    try:
        _atomic_write(temporary, document)
        report = validator.validate(temporary)
        if status == "complete" and report["gate_eligible"] is not True:
            raise EvidenceError("complete evidence is not production-gate eligible")
        temporary.unlink()
        _atomic_write(output_path, document)
        observed = validator.validate(output_path)
        if observed != report:
            raise EvidenceError("published evidence validation differs")
        return document
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker-binary", required=True, type=Path)
    parser.add_argument("--openwebui-image", required=True)
    parser.add_argument("--active-promotion-source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lifecycle", type=Path)
    parser.add_argument(
        "--campaign-output-dir",
        type=Path,
        help="selects strict authorization/active-binding evidence v2",
    )
    parser.add_argument("--status", choices=("incomplete", "complete"), default="incomplete")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        document = prepare(
            args.cases,
            args.manifest,
            args.worker_binary,
            args.openwebui_image,
            args.active_promotion_source_commit,
            args.output,
            lifecycle_path=args.lifecycle,
            campaign_output_dir=args.campaign_output_dir,
            status=args.status,
        )
    except Exception as error:
        print(f"Generic reasoning release evidence preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "output": os.fspath(args.output.resolve()),
                "case_count": len(document["cases"]),
                "lifecycle_event_count": len(document["lifecycle"]["events"]),
                "git_worktree_clean": document["git_worktree_clean"],
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
