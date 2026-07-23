#!/usr/bin/env python3
"""Run the hash-only OpenWebUI reasoning browser smoke safely.

The browser script writes only its bounded JSON evidence to stdout.  This
runner executes it in an immutable Playwright container, binds the expected
model identities explicitly, validates the result, and publishes it atomically.
Legacy runs preserve the historical v2 evidence; active-manifest-bound runs
publish one authorized directory containing lineage-bearing v4 evidence.
Prompt, response, token, and
credential contents are never written by this runner.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import hashlib
import importlib.util
import json
import os
import re
import select
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import socket
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from served_model_active_binding import (  # noqa: E402
    ActiveBindingArtifacts,
    ActiveManifestBinding,
    optional_v2_binding,
)


DEFAULT_SCRIPT = ROOT / "deploy/openwebui/browser-reasoning-smoke.cjs"
VALIDATOR_PATH = ROOT / "tools/validate-openwebui-reasoning-browser-smoke.py"
SERVED_MODEL_VALIDATOR_PATH = ROOT / "tools/validate-served-model.py"
MAX_TOKEN_FILE_BYTES = 65_536
MAX_SCRIPT_BYTES = 1 * 1024 * 1024
MAX_EVIDENCE_BYTES = 1 * 1024 * 1024
MAX_TOKENIZER_FILE_BYTES = 512 * 1024 * 1024
MAX_TOKENIZER_TOTAL_BYTES = 1024 * 1024 * 1024
IMAGE_RE = re.compile(
    r"(?:[A-Za-z0-9][A-Za-z0-9._/:+-]*@)?sha256:[0-9a-f]{64}\Z"
)
CONTENT_ADDRESSED_IMAGE_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._/:+-]*@sha256:[0-9a-f]{64}\Z"
)
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
CONTAINER_USER_RE = re.compile(r"[0-9]{1,5}:[0-9]{1,5}\Z")
JWT_SEGMENT_RE = re.compile(r"[A-Za-z0-9_-]{1,16384}\Z")
_VALIDATOR_MODULE_NAME = "_ullm_openwebui_reasoning_browser_validator"
_SERVED_MODEL_MODULE_NAME = "_ullm_reasoning_browser_served_model_validator"
TARGET_GPU_INDEX = "1"
WORKER_PROCESS_BASENAME_BY_FORMAT = {
    "AQ4_0": "ullm-aq4-worker",
    "SQ8_0": "ullm-sq8-worker",
}
ALLOWED_WORKER_PROCESS_BASENAMES = frozenset(
    WORKER_PROCESS_BASENAME_BY_FORMAT.values()
)
BROWSER_EVIDENCE_SCHEMA_V2 = "ullm.openwebui.reasoning_browser_smoke.v2"
BROWSER_EVIDENCE_SCHEMA_V3 = "ullm.openwebui.reasoning_browser_smoke.v3"
BROWSER_EVIDENCE_SCHEMA_V4 = "ullm.openwebui.reasoning_browser_smoke.v4"
CAMPAIGN_LINEAGE_SCHEMA_V2 = "ullm.served_model.campaign_lineage.v2"
BROWSER_EVIDENCE_FILE = "browser-evidence.json"
BROWSER_OUTPUT_FILES_V2 = frozenset(
    {
        BROWSER_EVIDENCE_FILE,
        "candidate-served-model.json",
        "active-manifest-observations.jsonl",
        "active-manifest-binding.json",
    }
)
SWITCH_EVIDENCE_FIELDS = {
    "provider_switch_performed",
    "provider_switch_model_id_sha256",
    "provider_switch_answer",
    "provider_return_performed",
    "provider_return_model_id_sha256",
    "provider_return_answer",
}
ACTIVE_BINDING_STAGES = (
    "preflight",
    "browser-launch",
    "browser-complete",
    "validation",
    "publication",
)


class SmokeError(RuntimeError):
    """Raised when browser smoke evidence cannot be safely published."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_regular(path: Path, label: str, maximum: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise SmokeError("O_NOFOLLOW is required for browser evidence reads")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SmokeError(f"{label} is unavailable") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise SmokeError(f"{label} exceeds its size bound")
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
        named = path.lstat()
        def identity(value: os.stat_result) -> tuple[int, ...]:
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
        if (
            len(raw) != before.st_size
            or len(raw) > maximum
            or identity(before) != identity(after)
            or identity(after) != identity(named)
        ):
            raise SmokeError(f"{label} changed while being read")
        return bytes(raw)
    except OSError as error:
        raise SmokeError(f"{label} cannot be read") from error
    finally:
        os.close(descriptor)


def _validate_image(value: str) -> str:
    if IMAGE_RE.fullmatch(value) is None:
        raise SmokeError("browser image must be an immutable Docker SHA-256 identity")
    return value


def _validate_v3_image(value: str) -> str:
    if CONTENT_ADDRESSED_IMAGE_RE.fullmatch(value) is None:
        raise SmokeError(
            "v3 browser evidence requires a content-addressed Docker image"
        )
    return value


def _stable_hash_regular(
    path: Path,
    label: str,
    *,
    maximum: int,
) -> tuple[str, int]:
    try:
        before = path.lstat()
    except OSError as error:
        raise SmokeError(f"{label} is unavailable") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size <= 0
        or before.st_size > maximum
    ):
        raise SmokeError(f"{label} is not a bounded regular non-symlink file")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise SmokeError(f"{label} cannot be opened safely") from error
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_uid,
            before.st_gid,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_gid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if opened_identity != identity:
            raise SmokeError(f"{label} changed before hashing")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise SmokeError(f"{label} exceeds its size bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_uid,
            after.st_gid,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if after_identity != identity or size != before.st_size:
            raise SmokeError(f"{label} changed while being hashed")
    except OSError as error:
        raise SmokeError(f"{label} cannot be hashed") from error
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _tokenizer_identity(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> str:
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("root"), str):
        raise SmokeError("served-model manifest tokenizer root is missing")
    files = tokenizer.get("files")
    if not isinstance(files, dict) or not files:
        raise SmokeError("served-model manifest tokenizer files are missing")
    root_path = Path(tokenizer["root"])
    if not root_path.is_absolute():
        root_path = manifest_path.parent / root_path
    try:
        root = root_path.resolve(strict=True)
        root_metadata = root.lstat()
    except OSError as error:
        raise SmokeError("served-model tokenizer root is unavailable") from error
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise SmokeError("served-model tokenizer root is not a directory")
    aggregate = hashlib.sha256()
    total_size = 0
    for name in sorted(files):
        expected = files[name]
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or Path(name).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(name).parts)
            or not isinstance(expected, str)
            or HASH_RE.fullmatch(expected) is None
        ):
            raise SmokeError("served-model tokenizer file identity is invalid")
        unresolved = root / name
        try:
            resolved = unresolved.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as error:
            raise SmokeError("served-model tokenizer file escapes its root") from error
        observed, size = _stable_hash_regular(
            resolved,
            f"served-model tokenizer file {name}",
            maximum=MAX_TOKENIZER_FILE_BYTES,
        )
        total_size += size
        if total_size > MAX_TOKENIZER_TOTAL_BYTES:
            raise SmokeError("served-model tokenizer files exceed their total size bound")
        if observed != expected:
            raise SmokeError(f"served-model tokenizer file hash differs: {name}")
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(observed))
    return aggregate.hexdigest()


def _validate_container_user(value: str) -> tuple[int, int]:
    if CONTAINER_USER_RE.fullmatch(value) is None:
        raise SmokeError("browser container user must be UID:GID")
    uid_text, gid_text = value.split(":")
    uid, gid = int(uid_text), int(gid_text)
    if uid > 65535 or gid > 65535:
        raise SmokeError("browser container UID/GID is out of range")
    return uid, gid


def _validate_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise SmokeError("OpenWebUI URL is invalid") from error
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise SmokeError("OpenWebUI URL must be a credential-free HTTP origin")
    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))


def _validate_public_text(value: str, label: str) -> str:
    if not value or len(value.encode("utf-8")) > 65_536:
        raise SmokeError(f"{label} is empty or exceeds its size bound")
    if any(character in value for character in "\r\n\0"):
        raise SmokeError(f"{label} contains a forbidden control character")
    return value


def _decode_session_jwt_object(segment: str, label: str) -> dict[str, Any]:
    if JWT_SEGMENT_RE.fullmatch(segment) is None:
        raise SmokeError(f"OpenWebUI session token {label} segment is invalid")
    padded = segment + ("=" * (-len(segment) % 4))

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON value")

    try:
        raw = base64.b64decode(
            padded.encode("ascii"), altchars=b"-_", validate=True
        )
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError, RecursionError) as error:
        raise SmokeError(
            f"OpenWebUI session token {label} segment is not strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise SmokeError(f"OpenWebUI session token {label} segment is not an object")
    return value


def _validate_openwebui_session_token(
    token: bytes,
    *,
    minimum_validity_seconds: int,
    now_seconds: int | None = None,
) -> None:
    try:
        value = token.decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise SmokeError("OpenWebUI session token is not UTF-8") from error
    if value.endswith("\n"):
        value = value[:-1]
    if (
        not value
        or len(value.encode("utf-8")) > MAX_TOKEN_FILE_BYTES
        or any(character in value for character in "\r\n\0")
        or value.strip() != value
    ):
        raise SmokeError("OpenWebUI session token is not one strict line")
    if not isinstance(minimum_validity_seconds, int) or minimum_validity_seconds < 0:
        raise SmokeError("OpenWebUI session token validity requirement is invalid")
    parts = value.split(".")
    if len(parts) != 3:
        raise SmokeError("OpenWebUI session token is not a JWT")
    header = _decode_session_jwt_object(parts[0], "header")
    payload = _decode_session_jwt_object(parts[1], "payload")
    if not isinstance(header.get("alg"), str) or not header["alg"]:
        raise SmokeError("OpenWebUI session token header lacks an algorithm")
    expiration = payload.get("exp")
    if isinstance(expiration, bool) or not isinstance(expiration, int):
        raise SmokeError("OpenWebUI session token lacks an integer expiration")
    now = int(time.time()) if now_seconds is None else now_seconds
    if not isinstance(now, int) or expiration <= now + minimum_validity_seconds:
        raise SmokeError("OpenWebUI session token expires before the browser gate can finish")


def _load_validator() -> ModuleType:
    existing = sys.modules.get(_VALIDATOR_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _VALIDATOR_MODULE_NAME, VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise SmokeError("browser evidence validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VALIDATOR_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_VALIDATOR_MODULE_NAME, None)
        raise
    return module


def _load_served_model_validator() -> ModuleType:
    existing = sys.modules.get(_SERVED_MODEL_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        _SERVED_MODEL_MODULE_NAME, SERVED_MODEL_VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise SmokeError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_SERVED_MODEL_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_SERVED_MODEL_MODULE_NAME, None)
        raise
    return module


def _strict_json(raw: bytes) -> dict[str, Any]:
    def without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SmokeError("browser evidence contains duplicate fields")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=without_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError("browser evidence is not strict JSON") from error
    if not isinstance(value, dict):
        raise SmokeError("browser evidence root is not an object")
    return value


def _provider_hashes(document: dict[str, Any]) -> list[str]:
    requests = document.get("provider_requests")
    if not isinstance(requests, list) or not 2 <= len(requests) <= 4:
        raise SmokeError("v2 browser evidence has an invalid provider request count")
    hashes: list[str] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or not isinstance(
            request.get("model_id_sha256"), str
        ):
            raise SmokeError(f"provider request {index} has no model identity")
        hashes.append(request["model_id_sha256"])
    return hashes


def _validate_manifest_identity(manifest: Path, model_id: str) -> dict[str, Any]:
    raw = _read_regular(manifest, "served-model manifest", 1_048_576)
    document = _strict_json(raw)
    if document.get("schema_version") != "ullm.served_model.v2":
        raise SmokeError("served-model manifest is not v2")
    public = document.get("public")
    worker = document.get("worker")
    if not isinstance(public, dict) or public.get("id") != model_id:
        raise SmokeError("served-model manifest model ID differs")
    if not isinstance(worker, dict) or worker.get("protocol") != "ullm.worker.v2":
        raise SmokeError("served-model manifest worker protocol is not v2")
    if "reasoning" not in document:
        raise SmokeError("served-model manifest has no reasoning dialect")
    try:
        summary = _load_served_model_validator().validation_summary(manifest)
    except Exception as error:
        raise SmokeError("served-model manifest failed validation") from error
    if summary.get("model_id") != model_id or summary.get("worker", {}).get(
        "protocol"
    ) != "ullm.worker.v2":
        raise SmokeError("served-model validator identity differs")
    return summary


def _worker_process_basename(manifest_summary: dict[str, Any]) -> str:
    worker = manifest_summary.get("worker")
    binary = worker.get("binary") if isinstance(worker, dict) else None
    if not isinstance(binary, str) or not binary or "\x00" in binary:
        raise SmokeError("validated manifest has no worker executable identity")
    binary_path = Path(binary)
    basename = binary_path.name
    expected = WORKER_PROCESS_BASENAME_BY_FORMAT.get(
        manifest_summary.get("format_id")
    )
    if (
        not binary_path.is_absolute()
        or expected is None
        or basename != expected
    ):
        raise SmokeError(
            "validated manifest worker executable is not an allowed release worker"
        )
    return basename


def _v3_release_identity(
    manifest_path: Path,
    manifest_summary: dict[str, Any],
    openwebui_image: str,
) -> tuple[str, dict[str, str]]:
    raw = _read_regular(
        manifest_path,
        "served-model manifest",
        1_048_576,
    )
    manifest_sha256 = _sha256(raw)
    if manifest_summary.get("manifest_sha256") != manifest_sha256:
        raise SmokeError("served-model manifest changed after validation")
    document = _strict_json(raw)
    promotion = document.get("promotion")
    source_commit = (
        promotion.get("source_commit") if isinstance(promotion, dict) else None
    )
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise SmokeError("served-model manifest has no full promotion source commit")
    worker = manifest_summary.get("worker")
    worker_binary_sha256 = (
        worker.get("binary_sha256") if isinstance(worker, dict) else None
    )
    if (
        not isinstance(worker_binary_sha256, str)
        or HASH_RE.fullmatch(worker_binary_sha256) is None
    ):
        raise SmokeError("validated manifest has no worker binary SHA-256")
    return source_commit, {
        "manifest_sha256": manifest_sha256,
        "worker_binary_sha256": worker_binary_sha256,
        "tokenizer_sha256": _tokenizer_identity(document, manifest_path),
        "openwebui_image": _validate_v3_image(openwebui_image),
    }


def _upgrade_browser_evidence_v3(
    document: dict[str, Any],
    *,
    source_commit: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    if document.get("schema_version") != BROWSER_EVIDENCE_SCHEMA_V2:
        raise SmokeError("browser evidence is not the v2 transport schema")
    if "source_commit" in document or "identity" in document:
        raise SmokeError("v2 browser evidence contains reserved v3 identity fields")
    if not isinstance(source_commit, str) or COMMIT_RE.fullmatch(source_commit) is None:
        raise SmokeError("v3 browser evidence source commit is invalid")
    if set(identity) != {
        "manifest_sha256",
        "worker_binary_sha256",
        "tokenizer_sha256",
        "openwebui_image",
    }:
        raise SmokeError("v3 browser evidence identity fields differ")
    if any(
        not isinstance(identity[field], str)
        or HASH_RE.fullmatch(identity[field]) is None
        for field in (
            "manifest_sha256",
            "worker_binary_sha256",
            "tokenizer_sha256",
        )
    ):
        raise SmokeError("v3 browser evidence identity SHA-256 is invalid")
    if not isinstance(identity["openwebui_image"], str):
        raise SmokeError("v3 browser evidence image identity is invalid")
    _validate_v3_image(identity["openwebui_image"])
    return {
        **document,
        "schema_version": BROWSER_EVIDENCE_SCHEMA_V3,
        "source_commit": source_commit,
        "identity": dict(identity),
    }


def _lineage_artifact(raw: bytes) -> dict[str, Any]:
    if not raw:
        raise SmokeError("browser lineage artifact is empty")
    return {"bytes": len(raw), "sha256": _sha256(raw)}


def _lineage_inventory(artifacts: dict[str, dict[str, Any]]) -> str:
    canonical = json.dumps(
        artifacts,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256(
        CAMPAIGN_LINEAGE_SCHEMA_V2.encode("ascii") + b"\0" + canonical
    )


def _upgrade_browser_evidence_v4(
    document: dict[str, Any],
    *,
    output: Path,
    artifacts: ActiveBindingArtifacts,
) -> dict[str, Any]:
    """Bind the v3 payload to its claimed run and every active-byte observation."""

    if document.get("schema_version") != BROWSER_EVIDENCE_SCHEMA_V3:
        raise SmokeError("browser evidence is not the v3 identity schema")
    if "campaign_lineage" in document:
        raise SmokeError("v3 browser evidence contains reserved lineage")
    try:
        binding = _strict_json(artifacts.binding_json)
    except SmokeError:
        raise
    expected_binding_fields = {
        "schema_version",
        "status",
        "candidate",
        "actual_active_path",
        "expected_stages",
        "observation_count",
        "observations",
        "claim",
        "campaign",
    }
    if not isinstance(binding, dict) or set(binding) != expected_binding_fields:
        raise SmokeError("browser active binding fields differ")
    candidate = binding.get("candidate")
    observations_reference = binding.get("observations")
    claim = binding.get("claim")
    campaign = binding.get("campaign")
    if (
        binding.get("schema_version") != "ullm.served_model.active_binding.v1"
        or binding.get("status") != "complete"
        or binding.get("expected_stages") != list(ACTIVE_BINDING_STAGES)
        or binding.get("observation_count") != len(ACTIVE_BINDING_STAGES)
        or not isinstance(candidate, dict)
        or set(candidate) != {"artifact", "source_path", "sha256", "bytes"}
        or not isinstance(observations_reference, dict)
        or set(observations_reference) != {"artifact", "sha256", "bytes"}
        or not isinstance(claim, dict)
        or set(claim)
        != {
            "path",
            "sha256",
            "bytes",
            "authorization_path",
            "authorization_sha256",
        }
        or not isinstance(campaign, dict)
        or set(campaign) != {"name", "run_id", "final_path"}
    ):
        raise SmokeError("browser active binding contract differs")
    output_absolute = output.absolute()
    if (
        not output.is_absolute()
        or output != output_absolute
        or campaign["name"] != "reasoning_browser"
        or campaign["final_path"] != os.fspath(output)
        or not isinstance(campaign["run_id"], str)
        or not campaign["run_id"]
        or candidate["artifact"] != "candidate-served-model.json"
        or candidate["sha256"] != _sha256(artifacts.candidate_manifest)
        or candidate["bytes"] != len(artifacts.candidate_manifest)
        or observations_reference
        != {
            "artifact": "active-manifest-observations.jsonl",
            "sha256": _sha256(artifacts.observations_jsonl),
            "bytes": len(artifacts.observations_jsonl),
        }
    ):
        raise SmokeError("browser candidate, run, or final output binding differs")
    for field in ("sha256", "authorization_sha256"):
        if not isinstance(claim[field], str) or HASH_RE.fullmatch(claim[field]) is None:
            raise SmokeError(f"browser campaign claim {field} differs")
    for field in ("path", "authorization_path"):
        if not isinstance(claim[field], str) or not Path(claim[field]).is_absolute():
            raise SmokeError(f"browser campaign claim {field} differs")
    if type(claim["bytes"]) is not int or claim["bytes"] < 1:
        raise SmokeError("browser campaign claim bytes differ")

    lines = artifacts.observations_jsonl.splitlines(keepends=True)
    if len(lines) != len(ACTIVE_BINDING_STAGES) or any(
        not line.endswith(b"\n") for line in lines
    ):
        raise SmokeError("browser active observation count differs")
    stages: list[dict[str, Any]] = []
    for sequence, (line, expected_stage) in enumerate(
        zip(lines, ACTIVE_BINDING_STAGES, strict=True)
    ):
        row = _strict_json(line)
        if (
            not isinstance(row, dict)
            or row.get("schema_version")
            != "ullm.served_model.active_manifest_observation.v1"
            or row.get("sequence") != sequence
            or row.get("stage") != expected_stage
            or row.get("bytes_equal") is not True
            or row.get("claim") != claim
        ):
            raise SmokeError(f"browser active observation {sequence} differs")
        stages.append(
            {
                "sequence": sequence,
                "stage": expected_stage,
                "sha256": _sha256(line),
            }
        )
    lineage_artifacts = {
        "active-manifest-binding.json": _lineage_artifact(
            artifacts.binding_json
        ),
        "active-manifest-observations.jsonl": _lineage_artifact(
            artifacts.observations_jsonl
        ),
        "candidate-served-model.json": _lineage_artifact(
            artifacts.candidate_manifest
        ),
    }
    lineage = {
        "schema_version": CAMPAIGN_LINEAGE_SCHEMA_V2,
        "campaign": {
            "name": "reasoning_browser",
            "run_id": campaign["run_id"],
            "final_path": os.fspath(output),
            "final_kind": "directory",
            "files": sorted(BROWSER_OUTPUT_FILES_V2),
        },
        "claim": dict(claim),
        "artifacts": lineage_artifacts,
        "artifact_inventory_sha256": _lineage_inventory(lineage_artifacts),
        "observations": {
            "count": len(stages),
            "stages": stages,
        },
    }
    return {
        **document,
        "schema_version": BROWSER_EVIDENCE_SCHEMA_V4,
        "campaign_lineage": lineage,
    }


def _bind_model_identity(
    document: dict[str, Any],
    *,
    model_id: str,
    switch_model_id: str | None,
) -> None:
    expected = _sha256(model_id.encode("utf-8"))
    if document.get("schema_version") != BROWSER_EVIDENCE_SCHEMA_V2:
        raise SmokeError("browser evidence is not the v2 schema")
    if document.get("model_id_sha256") != expected:
        raise SmokeError("browser evidence primary model identity differs")
    hashes = _provider_hashes(document)
    if switch_model_id is None:
        if set(document) & SWITCH_EVIDENCE_FIELDS:
            raise SmokeError("browser evidence unexpectedly contains a provider switch")
        if hashes != [expected, expected]:
            raise SmokeError("browser provider request model sequence differs")
        return
    switch_expected = _sha256(switch_model_id.encode("utf-8"))
    if document.get("provider_switch_model_id_sha256") != switch_expected:
        raise SmokeError("browser evidence switch model identity differs")
    if document.get("provider_return_model_id_sha256") != expected:
        raise SmokeError("browser evidence return model identity differs")
    if hashes[:2] != [expected, expected] or hashes[2:] != [switch_expected, expected]:
        raise SmokeError("browser provider request model sequence differs")


def _atomic_publish(output: Path, raw: bytes) -> None:
    """Publish one immutable legacy evidence file with no-replace semantics."""

    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise SmokeError("browser evidence output exceeds its size bound")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = output.parent.resolve(strict=True)
        metadata = parent.lstat()
    except OSError as error:
        raise SmokeError("browser evidence output directory is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SmokeError("browser evidence output parent is not a real directory")
    if parent != output.parent.absolute():
        raise SmokeError("browser evidence output parent is not canonical")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise SmokeError("O_NOFOLLOW is required for browser publication")
    parent_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, parent_flags)
    incomplete_name = f".{output.name}.incomplete-{uuid.uuid4().hex}"
    descriptor = -1
    try:
        descriptor = os.open(
            incomplete_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            stat.S_IRUSR | stat.S_IWUSR,
            dir_fd=parent_descriptor,
        )
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SmokeError("browser evidence write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(
                incomplete_name,
                output.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise SmokeError("browser evidence output already exists") from error
        os.unlink(incomplete_name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        if _read_regular(output, "published browser evidence", MAX_EVIDENCE_BYTES) != raw:
            raise SmokeError("published browser evidence bytes differ")
        published = output.lstat()
        if stat.S_IMODE(published.st_mode) != 0o444 or published.st_nlink != 1:
            raise SmokeError("published browser evidence identity differs")
    except OSError as error:
        raise SmokeError("browser evidence could not be published atomically") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(incomplete_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _rename_directory_noreplace(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise SmokeError("renameat2 is required for browser publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            1,
        )
        != 0
    ):
        observed_errno = ctypes.get_errno()
        if observed_errno in {errno.EEXIST, errno.ENOTEMPTY}:
            raise SmokeError("browser v2 output already exists")
        raise SmokeError(
            f"browser v2 output publication failed: errno {observed_errno}"
        )


def _write_immutable_stage_file(path: Path, raw: bytes) -> None:
    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise SmokeError("browser v2 artifact exceeds its size bound")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o600,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SmokeError("browser v2 artifact write failed")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _publish_v2_output_directory(
    output: Path,
    artifacts: ActiveBindingArtifacts,
    evidence_raw: bytes,
) -> Path:
    """Publish the one authorized, immutable browser v2 output directory."""

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise SmokeError("browser v2 output parent is unavailable") from error
    if parent != output.parent.absolute():
        raise SmokeError("browser v2 output parent is not canonical")
    parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if not hasattr(os, "O_NOFOLLOW"):
        raise SmokeError("O_NOFOLLOW is required for browser v2 publication")
    parent_flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(parent, parent_flags)
    parent_identity = os.fstat(parent_descriptor)
    stage = parent / f".{output.name}.incomplete-{uuid.uuid4().hex}"
    os.mkdir(stage.name, mode=0o700, dir_fd=parent_descriptor)
    published = False
    try:
        values = {
            **artifacts.by_name(),
            BROWSER_EVIDENCE_FILE: evidence_raw,
        }
        if set(values) != BROWSER_OUTPUT_FILES_V2:
            raise SmokeError("browser v2 output file set differs")
        for name, raw in values.items():
            _write_immutable_stage_file(stage / name, raw)
        stage.chmod(0o555)
        stage_descriptor = os.open(
            stage,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            os.fsync(stage_descriptor)
        finally:
            os.close(stage_descriptor)
        validator = _load_validator()
        report = validator.validate(
            stage / BROWSER_EVIDENCE_FILE,
            lineage_root_override=stage,
        )
        if report.get("gate_eligible") is not True:
            raise SmokeError("browser v4 staged evidence is not gate eligible")
        _rename_directory_noreplace(
            parent_descriptor,
            stage.name,
            output.name,
        )
        published = True
        os.fsync(parent_descriptor)
        verification_descriptor = os.open(parent, parent_flags)
        try:
            before = parent_identity
            after = os.fstat(verification_descriptor)
            if (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_uid,
                before.st_gid,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_uid,
                after.st_gid,
            ):
                raise SmokeError("browser v2 output parent changed")
        finally:
            os.close(verification_descriptor)
        observed = {entry.name for entry in os.scandir(output)}
        metadata = output.lstat()
        if (
            observed != BROWSER_OUTPUT_FILES_V2
            or stat.S_IMODE(metadata.st_mode) != 0o555
        ):
            raise SmokeError("published browser v2 directory identity differs")
        for name, raw in values.items():
            path = output / name
            if _read_regular(path, f"published browser {name}", MAX_EVIDENCE_BYTES) != raw:
                raise SmokeError(f"published browser {name} bytes differ")
            file_metadata = path.lstat()
            if (
                stat.S_IMODE(file_metadata.st_mode) != 0o444
                or file_metadata.st_nlink != 1
            ):
                raise SmokeError(f"published browser {name} identity differs")
        published_report = validator.validate(output / BROWSER_EVIDENCE_FILE)
        if published_report != report:
            raise SmokeError("published browser v4 validation differs")
        return output
    except (OSError, SmokeError) as error:
        if isinstance(error, SmokeError):
            raise
        raise SmokeError("browser v2 output could not be published") from error
    finally:
        os.close(parent_descriptor)
        if not published:
            shutil.rmtree(stage, ignore_errors=True)


def _select_manifest_and_binding(
    *,
    active_binding_mode: str,
    manifest: Path | None,
    candidate_served_model_manifest: Path | None,
    active_served_model_manifest: Path | None,
    expected_served_model_manifest_sha256: str | None,
    campaign_authorization: Path | None,
    run_id: str | None,
    output: Path,
) -> tuple[Path, ActiveManifestBinding | None]:
    if active_binding_mode == "legacy":
        if manifest is None:
            raise SmokeError("legacy browser smoke requires --manifest")
        binding = optional_v2_binding(
            mode="legacy",
            candidate_path=candidate_served_model_manifest,
            active_path=active_served_model_manifest,
            expected_sha256=expected_served_model_manifest_sha256,
            expected_stages=ACTIVE_BINDING_STAGES,
            authorization_path=campaign_authorization,
            campaign_name=None,
            run_id=None,
            final_path=None,
        )
        return manifest, binding
    if manifest is not None:
        raise SmokeError("v2 active binding forbids the legacy --manifest input")
    binding = optional_v2_binding(
        mode=active_binding_mode,
        candidate_path=candidate_served_model_manifest,
        active_path=active_served_model_manifest,
        expected_sha256=expected_served_model_manifest_sha256,
        expected_stages=ACTIVE_BINDING_STAGES,
        authorization_path=campaign_authorization,
        campaign_name="reasoning_browser",
        run_id=run_id,
        final_path=output,
    )
    assert binding is not None
    return binding.candidate.path, binding


def _stop_container(docker: str, name: str) -> None:
    try:
        subprocess.run(
            [docker, "rm", "--force", name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _service_state(service: str, systemctl: str) -> tuple[str, str]:
    try:
        result = subprocess.run(
            [systemctl, "show", service, "--property=LoadState", "--property=ActiveState", "--value"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("service state lookup failed") from error
    states = result.stdout.splitlines()
    if result.returncode != 0 or len(states) != 2:
        raise SmokeError("service state lookup failed")
    return states[0], states[1]


def _service_command(systemctl: str, action: str, service: str) -> None:
    try:
        result = subprocess.run(
            [systemctl, action, service],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("service transition failed") from error
    if result.returncode != 0:
        raise SmokeError("service transition failed")


def _target_gpu_processes(rocm_smi: str) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [rocm_smi, "--showpids", "--json"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError("GPU ownership preflight failed") from error
    if result.returncode != 0:
        raise SmokeError("GPU ownership preflight failed")
    if not result.stdout.strip():
        return []
    value = _strict_json(result.stdout.encode("utf-8"))
    system = value.get("system")
    if not isinstance(system, dict):
        raise SmokeError("GPU ownership data is missing")
    processes: list[dict[str, Any]] = []
    for pid, description in system.items():
        if not isinstance(pid, str) or not pid.startswith("PID") or not isinstance(description, str):
            raise SmokeError("GPU process identity is malformed")
        fields = [field.strip() for field in description.split(",")]
        if len(fields) < 3:
            raise SmokeError("GPU process description is incomplete")
        try:
            vram = int(fields[2])
        except ValueError as error:
            raise SmokeError("GPU process VRAM is invalid") from error
        visible = {item.strip() for item in fields[1].split(",") if item.strip()}
        if TARGET_GPU_INDEX in visible and vram > 0:
            processes.append({"pid": pid[3:], "process": fields[0], "vram_bytes": vram})
    return processes


def _wait_for_tcp_port(
    docker: str,
    host: str,
    port: int,
    timeout_seconds: float = 120.0,
    readiness_path: str = "/readyz",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    consecutive_successes = 0
    while True:
        try:
            result = subprocess.run(
                [
                    docker,
                    "exec",
                    "open-webui",
                    "python",
                    "-c",
                    (
                        "import urllib.error, urllib.request; "
                        f"url='http://{host}:{port}{readiness_path}'; "
                        "\ntry: "
                        " response=urllib.request.urlopen(url, timeout=2.0); "
                        " raise SystemExit(0 if response.status == 200 else 1)"
                        "\nexcept (urllib.error.HTTPError, urllib.error.URLError): "
                        " raise SystemExit(1)"
                    ),
                ],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
            )
            if result.returncode == 0:
                consecutive_successes += 1
                if consecutive_successes >= 3:
                    return
                time.sleep(0.5)
                continue
            consecutive_successes = 0
            if time.monotonic() >= deadline:
                raise SmokeError("service HTTP port did not become ready")
            time.sleep(0.25)
        except (OSError, subprocess.TimeoutExpired):
            consecutive_successes = 0
            if time.monotonic() >= deadline:
                raise SmokeError("service HTTP port did not become ready")
            time.sleep(0.25)


def _wait_for_gpu_owner(rocm_smi: str, expected: set[str], timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        processes = _target_gpu_processes(rocm_smi)
        if {process["process"] for process in processes} == expected:
            return
        if time.monotonic() >= deadline:
            raise SmokeError("target R9700 ownership did not reach the expected state")
        time.sleep(0.25)


class _AlternatingServiceCoordinator:
    def __init__(
        self,
        systemctl: str,
        rocm_smi: str,
        ullm_service: str,
        llama_service: str,
        ullm_worker_process_basename: str,
        docker: str = "docker",
        ullm_port: int = 8000,
        llama_port: int = 8001,
    ) -> None:
        self.systemctl = systemctl
        self.rocm_smi = rocm_smi
        self.ullm_service = ullm_service
        self.llama_service = llama_service
        if ullm_worker_process_basename not in ALLOWED_WORKER_PROCESS_BASENAMES:
            raise SmokeError(
                "uLLM worker process identity is not bound to a validated manifest"
            )
        self.ullm_worker_process_basename = ullm_worker_process_basename
        self.docker = docker
        self.ullm_port = ullm_port
        self.llama_port = llama_port
        self.owner = "ullm"

    def preflight(self, worker_binary_sha256: str) -> None:
        if _service_state(self.ullm_service, self.systemctl)[1] != "active":
            raise SmokeError("uLLM service must be active before alternating browser smoke")
        if _service_state(self.llama_service, self.systemctl)[1] not in {"inactive", "failed"}:
            raise SmokeError("llama.cpp service must be inactive before alternating browser smoke")
        processes = _target_gpu_processes(self.rocm_smi)
        if {process["process"] for process in processes} != {
            self.ullm_worker_process_basename
        }:
            raise SmokeError("target R9700 is not exclusively owned by uLLM")
        for process in processes:
            try:
                target = Path(os.readlink(f"/proc/{process['pid']}/exe"))
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
            except OSError as error:
                raise SmokeError("uLLM worker executable cannot be inspected") from error
            if digest != worker_binary_sha256:
                raise SmokeError("uLLM worker executable differs from the v2 manifest")

    def transition(self, phase: str) -> None:
        if phase == "before-switch":
            self.owner = "transition"
            _service_command(self.systemctl, "stop", self.ullm_service)
            _wait_for_gpu_owner(self.rocm_smi, set())
            _service_command(self.systemctl, "start", self.llama_service)
            _wait_for_gpu_owner(self.rocm_smi, {"llama-server"})
            _wait_for_tcp_port(
                self.docker,
                "172.20.0.1",
                self.llama_port,
                readiness_path="/health",
            )
            self.owner = "llama"
            return
        if phase == "before-return":
            self.owner = "transition"
            _service_command(self.systemctl, "stop", self.llama_service)
            _wait_for_gpu_owner(self.rocm_smi, set())
            _service_command(self.systemctl, "start", self.ullm_service)
            _wait_for_gpu_owner(
                self.rocm_smi,
                {self.ullm_worker_process_basename},
            )
            _wait_for_tcp_port(
                self.docker,
                "172.20.0.1",
                self.ullm_port,
                readiness_path="/readyz",
            )
            self.owner = "ullm"
            return
        raise SmokeError("unknown browser service transition phase")

    def recover(self) -> None:
        if self.owner in {"llama", "transition"}:
            try:
                _service_command(self.systemctl, "stop", self.llama_service)
                _wait_for_gpu_owner(self.rocm_smi, set(), timeout_seconds=30.0)
                _service_command(self.systemctl, "start", self.ullm_service)
            except SmokeError:
                pass
            self.owner = "ullm"


def _wait_for_browser_with_transitions(
    process: subprocess.Popen[bytes],
    server: socket.socket,
    coordinator: _AlternatingServiceCoordinator,
    deadline: float,
) -> int:
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, 0)
        readable, _, _ = select.select([server], [], [], min(remaining, 0.5))
        if not readable:
            continue
        connection, _ = server.accept()
        try:
            phase = connection.recv(128).decode("ascii").strip()
            try:
                coordinator.transition(phase)
            except SmokeError:
                connection.sendall(b"abort\n")
                raise
            connection.sendall(b"continue\n")
        finally:
            connection.close()
    return process.wait(timeout=max(0.1, deadline - time.monotonic()))


def execute(
    *,
    output: Path,
    manifest: Path | None,
    openwebui_session_token_file: Path,
    browser_image: str,
    openwebui_url: str,
    model_id: str,
    model_name: str,
    switch_model_id: str | None = None,
    switch_model_name: str | None = None,
    browser_script: Path = DEFAULT_SCRIPT,
    docker: str = "docker",
    timeout_seconds: float = 900.0,
    alternate_r9700_services: bool = False,
    ullm_service: str = "ullm-openai.service",
    llama_service: str = "llama-qwen35-udq4.service",
    systemctl: str = "systemctl",
    rocm_smi: str = "rocm-smi",
    browser_user: str | None = None,
    active_binding_mode: str = "legacy",
    candidate_served_model_manifest: Path | None = None,
    active_served_model_manifest: Path | None = None,
    expected_served_model_manifest_sha256: str | None = None,
    campaign_authorization: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise SmokeError("browser evidence output already exists or is a symlink")
    selected_manifest, active_binding = _select_manifest_and_binding(
        active_binding_mode=active_binding_mode,
        manifest=manifest,
        candidate_served_model_manifest=candidate_served_model_manifest,
        active_served_model_manifest=active_served_model_manifest,
        expected_served_model_manifest_sha256=(
            expected_served_model_manifest_sha256
        ),
        campaign_authorization=campaign_authorization,
        run_id=run_id,
        output=output,
    )
    if active_binding is not None:
        active_binding.observe("preflight")
    manifest_summary = _validate_manifest_identity(selected_manifest, model_id)
    worker_process_basename = _worker_process_basename(manifest_summary)
    token = _read_regular(
        openwebui_session_token_file,
        "OpenWebUI session token file",
        MAX_TOKEN_FILE_BYTES,
    )
    script = _read_regular(browser_script, "browser smoke script", MAX_SCRIPT_BYTES)
    _validate_openwebui_session_token(
        token,
        minimum_validity_seconds=int(timeout_seconds) + 30,
    )
    url = _validate_url(openwebui_url)
    image = _validate_image(browser_image)
    v3_source_commit: str | None = None
    v3_identity: dict[str, str] | None = None
    if active_binding is not None:
        v3_source_commit, v3_identity = _v3_release_identity(
            selected_manifest,
            manifest_summary,
            image,
        )
    model_id = _validate_public_text(model_id, "model ID")
    model_name = _validate_public_text(model_name, "model name")
    if (switch_model_id is None) != (switch_model_name is None):
        raise SmokeError("switch model ID and name must be supplied together")
    switch_requested = switch_model_id is not None
    if switch_requested:
        switch_model_id = _validate_public_text(switch_model_id, "switch model ID")
        switch_model_name = _validate_public_text(switch_model_name, "switch model name")
        if model_id == switch_model_id:
            raise SmokeError("switch model ID must differ from the candidate model ID")
    if alternate_r9700_services and not switch_requested:
        raise SmokeError("alternate R9700 services require a switch model")
    container_user = browser_user or f"{os.geteuid()}:{os.getegid()}"
    container_uid, container_gid = _validate_container_user(container_user)
    del token, script

    coordinator = (
        _AlternatingServiceCoordinator(
            systemctl,
            rocm_smi,
            ullm_service,
            llama_service,
            worker_process_basename,
            docker=docker,
        )
        if alternate_r9700_services
        else None
    )
    control_directory: tempfile.TemporaryDirectory[str] | None = None
    control_server: socket.socket | None = None
    if coordinator is not None:
        coordinator.preflight(manifest_summary["worker"]["binary_sha256"])
        control_directory = tempfile.TemporaryDirectory(prefix="ullm-browser-transition-")
        control_path = Path(control_directory.name) / "transition.sock"
        control_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        control_server.bind(os.fspath(control_path))
        os.chmod(control_path, 0o600)
        if os.geteuid() == 0:
            os.chown(control_directory.name, container_uid, container_gid)
            os.chown(control_path, container_uid, container_gid)
        control_server.listen(1)

    container_name = f"ullm-reasoning-browser-{uuid.uuid4().hex[:16]}"
    script_path = browser_script.resolve(strict=True)
    token_path = openwebui_session_token_file.resolve(strict=True)
    command = [
        docker,
        "run",
        "--rm",
        "--network=host",
        f"--name={container_name}",
        f"--user={container_user}",
        "--pids-limit=256",
        "--security-opt=no-new-privileges",
        "--mount",
        f"type=bind,src={script_path},dst=/run/ullm-browser-reasoning-smoke.cjs,readonly",
        "--mount",
        f"type=bind,src={token_path},dst=/run/secrets/openwebui-session-token,readonly",
        "--env",
        f"OPENWEBUI_URL={url}",
        "--env",
        "OPENWEBUI_SESSION_TOKEN_FILE=/run/secrets/openwebui-session-token",
        "--env",
        "NODE_PATH=/usr/src/app/node_modules",
        "--env",
        f"ULLM_MODEL_ID={model_id}",
        "--env",
        f"ULLM_MODEL_NAME={model_name}",
    ]
    if switch_requested:
        command.extend(
            [
                "--env",
                f"OPENWEBUI_SWITCH_MODEL_ID={switch_model_id}",
                "--env",
                f"OPENWEBUI_SWITCH_MODEL_NAME={switch_model_name}",
            ]
        )
    if control_server is not None:
        command.extend(
            [
                "--mount",
                f"type=bind,src={control_directory.name},dst=/run/ullm-transition,readonly",
                "--env",
                "OPENWEBUI_TRANSITION_SOCKET=/run/ullm-transition/transition.sock",
            ]
        )
    command.extend(
        [
        "--entrypoint",
        "node",
        image,
        "/run/ullm-browser-reasoning-smoke.cjs",
        ]
    )
    deadline = time.monotonic() + timeout_seconds
    try:
        if active_binding is not None:
            active_binding.observe("browser-launch")
        with tempfile.TemporaryFile() as stdout:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                if control_server is None:
                    return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
                else:
                    return_code = _wait_for_browser_with_transitions(
                        process,
                        control_server,
                        coordinator,
                        deadline,
                    )
            except subprocess.TimeoutExpired as error:
                process.kill()
                process.wait(timeout=15.0)
                _stop_container(docker, container_name)
                raise SmokeError("browser smoke timed out") from error
            except BaseException:
                process.kill()
                process.wait(timeout=15.0)
                _stop_container(docker, container_name)
                raise
            if return_code != 0:
                error_output = process.stderr.read(4097) if process.stderr is not None else b""
                detail = error_output.decode("utf-8", errors="replace").strip()
                if len(detail) > 4096:
                    detail = detail[:4096]
                raise SmokeError(
                    "browser smoke container failed"
                    + (f": {detail}" if detail else "")
                )
            stdout.seek(0)
            raw = stdout.read(MAX_EVIDENCE_BYTES + 1)
    except Exception:
        if coordinator is not None:
            coordinator.recover()
        raise
    finally:
        if control_server is not None:
            control_server.close()
        if control_directory is not None:
            control_directory.cleanup()
    if not raw or len(raw) > MAX_EVIDENCE_BYTES:
        raise SmokeError("browser smoke output exceeds its size bound")
    if active_binding is not None:
        active_binding.observe("browser-complete")
    document = _strict_json(raw.strip())
    _bind_model_identity(
        document,
        model_id=model_id,
        switch_model_id=switch_model_id,
    )
    if active_binding is None:
        published_raw = raw.strip() + b"\n"
        temporary = output.parent / f".{output.name}.validate-{uuid.uuid4().hex}"
        try:
            _atomic_publish(temporary, published_raw)
            report = _load_validator().validate(temporary)
            if report.get("gate_eligible") is not True:
                raise SmokeError("browser evidence is not gate eligible")
            temporary.unlink()
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        _atomic_publish(output, published_raw)
        return {
            "schema_version": document["schema_version"],
            "output": os.fspath(output.resolve()),
            "model_id_sha256": document["model_id_sha256"],
            "provider_request_count": document["provider_request_count"],
            "manifest_sha256": manifest_summary["manifest_sha256"],
        }

    assert v3_source_commit is not None and v3_identity is not None
    document = _upgrade_browser_evidence_v3(
        document,
        source_commit=v3_source_commit,
        identity=v3_identity,
    )
    active_binding.observe("validation")
    active_binding.observe("publication")
    active_artifacts = active_binding.artifacts()
    document = _upgrade_browser_evidence_v4(
        document,
        output=output,
        artifacts=active_artifacts,
    )
    published_raw = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    published_output = _publish_v2_output_directory(
        output,
        active_artifacts,
        published_raw,
    )
    return {
        "schema_version": document["schema_version"],
        "output": os.fspath(published_output.resolve()),
        "evidence": os.fspath(
            (published_output / BROWSER_EVIDENCE_FILE).resolve()
        ),
        "model_id_sha256": document["model_id_sha256"],
        "provider_request_count": document["provider_request_count"],
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "run_id": run_id,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--active-binding-mode",
        choices=("legacy", "v2"),
        default="legacy",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--candidate-served-model-manifest", type=Path)
    parser.add_argument("--active-served-model-manifest", type=Path)
    parser.add_argument("--expected-served-model-manifest-sha256")
    parser.add_argument("--campaign-authorization", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--openwebui-session-token-file",
        type=Path,
        required=True,
        help="private OpenWebUI frontend session JWT; not the gateway API key",
    )
    parser.add_argument("--browser-image", required=True)
    parser.add_argument("--openwebui-url", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--switch-model-id")
    parser.add_argument("--switch-model-name")
    parser.add_argument("--browser-script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--alternate-r9700-services", action="store_true")
    parser.add_argument("--ullm-service", default="ullm-openai.service")
    parser.add_argument("--llama-service", default="llama-qwen35-udq4.service")
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument("--rocm-smi", default="rocm-smi")
    parser.add_argument("--browser-user")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = execute(
            output=args.output,
            manifest=args.manifest,
            openwebui_session_token_file=args.openwebui_session_token_file,
            browser_image=args.browser_image,
            openwebui_url=args.openwebui_url,
            model_id=args.model_id,
            model_name=args.model_name,
            switch_model_id=args.switch_model_id,
            switch_model_name=args.switch_model_name,
            browser_script=args.browser_script,
            docker=args.docker,
            timeout_seconds=args.timeout_seconds,
            alternate_r9700_services=args.alternate_r9700_services,
            ullm_service=args.ullm_service,
            llama_service=args.llama_service,
            systemctl=args.systemctl,
            rocm_smi=args.rocm_smi,
            browser_user=args.browser_user,
            active_binding_mode=args.active_binding_mode,
            candidate_served_model_manifest=args.candidate_served_model_manifest,
            active_served_model_manifest=args.active_served_model_manifest,
            expected_served_model_manifest_sha256=(
                args.expected_served_model_manifest_sha256
            ),
            campaign_authorization=args.campaign_authorization,
            run_id=args.run_id,
        )
    except Exception as error:
        print(f"OpenWebUI reasoning browser smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
