#!/usr/bin/env python3
"""Bounded, streaming helpers for the Qwen3.5-9B AQ4 P2 oracle contract.

The source and path oracles intentionally store only bounded hidden/logit
samples and top-k summaries.  A complete vocabulary or hidden-state matrix is
never required by this interface.  The module is imported by the two CLI
tools; it has no dependency on the engine or on a model runtime.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


SOURCE_SCHEMA = "ullm.qwen35_aq4_source_oracle.v1"
PATH_SCHEMA = "ullm.qwen35_aq4_path_oracle.v1"
LINK_SCHEMA = "ullm.qwen35_aq4_oracle_link.v1"
SCHEMAS = {"source": SOURCE_SCHEMA, "path": PATH_SCHEMA}
ORACLE_KINDS = {"independent_source": SOURCE_SCHEMA, "same_artifact_all_m1": PATH_SCHEMA}
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024
MAX_CASES = 128
MAX_STEPS = 128
MAX_TOP_K = 32
MAX_SAMPLE_VALUES = 256
MAX_JSON_BYTES = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOKENIZER_FILES = (
    "chat_template.jinja",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
RANKING_CONTRACT = {
    "greedy": "maximum_logit_then_smallest_token_id",
    "scope": "entire_vocabulary",
    "topk": "logit_descending_then_token_id_ascending",
}


class OracleError(ValueError):
    """A fail-closed oracle contract error."""


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OracleError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise OracleError(f"non-finite JSON number: {value}")


def _file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mode,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return _file_identity(left) == _file_identity(right)


def _reject_symlink_components(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except OSError as error:
            raise OracleError(f"cannot inspect {label} path component: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise OracleError(f"{label} path contains a symlink component: {current}")


def _open_pinned_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    _reject_symlink_components(path, label)
    try:
        before = path.lstat()
    except OSError as error:
        raise OracleError(f"cannot stat {label}: {error}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OracleError(f"{label} must be a regular non-symlink file")
    if before.st_nlink != 1:
        raise OracleError(f"{label} must have exactly one hard link")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise OracleError(f"cannot open pinned {label}: {error}") from error
    opened = os.fstat(descriptor)
    if opened.st_nlink != 1 or not stat.S_ISREG(opened.st_mode) or not _same_file_identity(before, opened):
        os.close(descriptor)
        raise OracleError(f"{label} path/fd identity changed while opening")
    return descriptor, opened


def _verify_pinned_regular(path: Path, descriptor: int, before: os.stat_result, label: str) -> None:
    try:
        descriptor_after = os.fstat(descriptor)
        path_after = path.lstat()
    except OSError as error:
        raise OracleError(f"cannot restat pinned {label}: {error}") from error
    if (
        descriptor_after.st_nlink != 1
        or path_after.st_nlink != 1
        or not _same_file_identity(before, descriptor_after)
        or not _same_file_identity(before, path_after)
    ):
        raise OracleError(f"{label} fd/path identity changed while reading")


def read_regular_bytes(path: Path, label: str, maximum: int) -> bytes:
    descriptor, before = _open_pinned_regular(path, label)
    try:
        if before.st_size > maximum:
            raise OracleError(f"{label} exceeds {maximum} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > maximum:
                raise OracleError(f"{label} exceeds {maximum} bytes")
            chunks.append(chunk)
        _verify_pinned_regular(path, descriptor, before, label)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            read_regular_bytes(path, f"JSON {path}", MAX_JSON_BYTES).decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleError(f"invalid JSON {path}: {error}") from error


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    descriptor, before = _open_pinned_regular(path, f"hashed file {path}")
    digest = hashlib.sha256()
    try:
        while chunk := os.read(descriptor, chunk_bytes):
            digest.update(chunk)
        _verify_pinned_regular(path, descriptor, before, f"hashed file {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def canonical_sha256(value: Any) -> str:
    raw = (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def safe_relative(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise OracleError(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise OracleError(f"{label} is unsafe: {raw!r}")
    path = root.joinpath(*pure.parts)
    try:
        info = path.lstat()
    except OSError as error:
        raise OracleError(f"missing {label}: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise OracleError(f"{label} must be a regular non-symlink file")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise OracleError(f"{label} escapes oracle root") from error
    return path


def ensure_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise OracleError(f"{label} must be a lowercase SHA-256 digest")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OracleError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OracleError(f"{label} must be finite")
    return result


def integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OracleError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise OracleError(f"{label} must be >= {minimum}")
    return value


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def validate_utc(value: Any, label: str = "created_utc") -> None:
    if not isinstance(value, str):
        raise OracleError(f"{label} must be an ISO-8601 UTC string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise OracleError(f"{label} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise OracleError(f"{label} must include UTC")


def _exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise OracleError(f"{label} keys differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    return value


def _validate_sample(value: Any, label: str) -> dict[str, Any]:
    sample = _exact_keys(value, {"dtype", "indices", "shape", "values"}, label)
    if sample["dtype"] != "f32":
        raise OracleError(f"{label}.dtype must be f32")
    shape = sample["shape"]
    if not isinstance(shape, list) or len(shape) != 1:
        raise OracleError(f"{label}.shape must be one-dimensional")
    shape_size = integer(shape[0], f"{label}.shape[0]", minimum=1)
    indices = sample["indices"]
    values = sample["values"]
    if not isinstance(indices, list) or not isinstance(values, list) or len(indices) != len(values):
        raise OracleError(f"{label} indices and values lengths differ")
    if len(values) == 0 or len(values) > MAX_SAMPLE_VALUES or shape_size < len(values):
        raise OracleError(f"{label} exceeds bounded sample limits")
    previous = -1
    for index, number in zip(indices, values):
        index = integer(index, f"{label}.indices", minimum=0)
        if index <= previous or index >= shape_size:
            raise OracleError(f"{label}.indices must be strictly increasing and in shape")
        previous = index
        finite(number, f"{label}.values")
    return sample


def _validate_topk(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > MAX_TOP_K:
        raise OracleError(f"{label} must contain 1..{MAX_TOP_K} entries")
    result: list[dict[str, Any]] = []
    previous: tuple[float, int] | None = None
    seen: set[int] = set()
    for index, raw in enumerate(value):
        entry = _exact_keys(raw, {"logit", "token_id"}, f"{label}[{index}]")
        token_id = integer(entry["token_id"], f"{label}[{index}].token_id", minimum=0)
        logit = finite(entry["logit"], f"{label}[{index}].logit")
        if token_id in seen:
            raise OracleError(f"{label} contains duplicate token id")
        seen.add(token_id)
        key = (-logit, token_id)
        if previous is not None and key < previous:
            raise OracleError(f"{label} is not ordered by descending logit/token id")
        previous = key
        result.append({"token_id": token_id, "logit": logit})
    return result


def validate_payload_record(raw: Any, label: str) -> dict[str, Any]:
    record = _exact_keys(
        raw,
        {"case_id", "greedy_token_id", "hidden_sample", "logit_sample", "step", "topk"},
        label,
    )
    case_id = record["case_id"]
    if not isinstance(case_id, str) or not case_id or len(case_id) > 128:
        raise OracleError(f"{label}.case_id must be a bounded non-empty string")
    step = integer(record["step"], f"{label}.step", minimum=0)
    if step >= MAX_STEPS:
        raise OracleError(f"{label}.step exceeds bound")
    greedy = integer(record["greedy_token_id"], f"{label}.greedy_token_id", minimum=0)
    hidden = _validate_sample(record["hidden_sample"], f"{label}.hidden_sample")
    logits = _validate_sample(record["logit_sample"], f"{label}.logit_sample")
    topk = _validate_topk(record["topk"], f"{label}.topk")
    if topk[0]["token_id"] != greedy:
        raise OracleError(f"{label}.greedy_token_id differs from topk[0]")
    return {
        "case_id": case_id,
        "step": step,
        "greedy_token_id": greedy,
        "hidden_sample": hidden,
        "logit_sample": logits,
        "topk": topk,
    }


def _iter_payload_bytes(raw_bytes: bytes) -> Iterator[dict[str, Any]]:
    size = len(raw_bytes)
    if size <= 0 or size > MAX_PAYLOAD_BYTES:
        raise OracleError(f"payload bytes must be between 1 and {MAX_PAYLOAD_BYTES}")
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeError as error:
        raise OracleError(f"invalid payload UTF-8: {error}") from error
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise OracleError(f"payload line {line_number} is empty")
        try:
            raw = json.loads(line, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
        except json.JSONDecodeError as error:
            raise OracleError(f"invalid payload JSON line {line_number}: {error}") from error
        yield validate_payload_record(raw, f"payload[{line_number}]")


def iter_payload(path: Path) -> Iterator[dict[str, Any]]:
    yield from _iter_payload_bytes(read_regular_bytes(path, "payload", MAX_PAYLOAD_BYTES))


def digest_payload(path: Path) -> tuple[str, int, int]:
    raw = read_regular_bytes(path, "payload", MAX_PAYLOAD_BYTES)
    records = 0
    for _ in _iter_payload_bytes(raw):
        records += 1
        if records > MAX_CASES * MAX_STEPS:
            raise OracleError("payload record count exceeds bounded limit")
    return hashlib.sha256(raw).hexdigest(), len(raw), records


def metadata_file(root: Path, name: str) -> dict[str, Any]:
    path = safe_relative(root, name, "metadata file")
    return {"file": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def inspect_source_model(root: Path) -> dict[str, Any]:
    """Inspect a real BF16 Qwen3.5 source tree without loading model weights."""
    if root.is_symlink() or not root.is_dir():
        raise OracleError(f"source model root is unavailable: {root}")
    config_path = root / "config.json"
    config = load_json(config_path)
    if not isinstance(config, dict) or config.get("model_type") != "qwen3_5":
        raise OracleError("source config is not Qwen3.5")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("dtype") not in {"bfloat16", "float32", "bf16", "f32"}:
        raise OracleError("source config does not declare BF16/F32 text weights")
    required = ["config.json", "model.safetensors.index.json"]
    for name in required:
        metadata_file(root, name)
    index = load_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise OracleError("source safetensors index has no weight_map")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or not name for name in shards):
        raise OracleError("source safetensors index contains an invalid shard name")
    for name in shards:
        metadata_file(root, name)
    revision = None
    metadata_dir = root / ".cache" / "huggingface" / "download"
    if metadata_dir.is_dir():
        revisions = []
        for metadata in sorted(metadata_dir.glob("*.metadata")):
            try:
                first = metadata.read_text(encoding="utf-8").splitlines()[0]
            except (OSError, UnicodeError, IndexError):
                continue
            if first:
                revisions.append(first)
        if revisions and len(set(revisions)) == 1:
            revision = revisions[0]
    return {
        "model_id": "Qwen/Qwen3.5-9B",
        "revision": revision,
        "dtype": text_config["dtype"],
        "root": str(root.resolve(strict=True)),
        "config": metadata_file(root, "config.json"),
        "weight_index": metadata_file(root, "model.safetensors.index.json"),
        "tokenizer_files": [metadata_file(root, name) for name in TOKENIZER_FILES],
        "weight_shards": [metadata_file(root, name) for name in shards],
    }


def source_checkpoint_identity(inspected: dict[str, Any]) -> dict[str, Any]:
    files = sorted(
        [inspected["config"], inspected["weight_index"], *inspected["weight_shards"]],
        key=lambda item: item["file"],
    )
    return {
        "aggregate_sha256": canonical_sha256(files),
        "dtype": inspected["dtype"],
        "files": files,
        "root": inspected["root"],
    }


def _validate_file_identity(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise OracleError(f"{label} must be nonempty")
    previous: str | None = None
    result = []
    for index, raw in enumerate(value):
        entry = _exact_keys(raw, {"bytes", "file", "sha256"}, f"{label}[{index}]")
        name = entry["file"]
        if not isinstance(name, str) or not name or (previous is not None and name <= previous):
            raise OracleError(f"{label} must have unique file-sorted names")
        previous = name
        integer(entry["bytes"], f"{label}[{index}].bytes", minimum=1)
        ensure_sha256(entry["sha256"], f"{label}[{index}].sha256")
        result.append(entry)
    return result


def validate_manifest(root: Path, *, expected_kind: str | None = None) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise OracleError("manifest must be an object")
    kind = manifest.get("oracle_kind")
    if kind not in ORACLE_KINDS:
        raise OracleError("manifest oracle_kind is invalid")
    expected_oracle_kind = {"source": "independent_source", "path": "same_artifact_all_m1"}.get(expected_kind, expected_kind)
    if expected_oracle_kind is not None and kind != expected_oracle_kind:
        raise OracleError(f"manifest kind is {kind}, expected {expected_oracle_kind}")
    if manifest.get("schema_version") != ORACLE_KINDS[kind]:
        raise OracleError("manifest schema_version does not match oracle_kind")
    if manifest.get("status") not in {"available", "fixture", "blocked"}:
        raise OracleError("manifest status is invalid")
    if manifest.get("evidence_class") not in {"production", "synthetic_fixture", "blocked"}:
        raise OracleError("manifest evidence_class is invalid")
    if "promotion_eligible" in manifest:
        raise OracleError("oracle manifest must not make a candidate promotion decision")
    eligibility_key = "usable_as_source_evidence" if kind == "independent_source" else "usable_as_path_evidence"
    if not isinstance(manifest.get(eligibility_key), bool):
        raise OracleError(f"manifest {eligibility_key} must be boolean")
    if manifest["evidence_class"] != "production" and manifest[eligibility_key]:
        raise OracleError("non-production oracle cannot be usable as production evidence")
    validate_utc(manifest.get("created_utc"))
    if manifest.get("ranking") != RANKING_CONTRACT:
        raise OracleError("manifest ranking contract differs")
    identity = _exact_keys(manifest.get("identity"), {"artifact", "model_id", "model_revision", "source_checkpoint", "tokenizer"}, "identity")
    if not isinstance(identity["model_id"], str) or not identity["model_id"]:
        raise OracleError("identity.model_id is invalid")
    if identity["model_revision"] is not None and not isinstance(identity["model_revision"], str):
        raise OracleError("identity.model_revision is invalid")
    artifact_keys = set(identity["artifact"]) if isinstance(identity["artifact"], dict) else set()
    base_artifact_keys = {"package_manifest_sha256", "artifact_manifest_sha256"}
    package_bound_keys = base_artifact_keys | {
        "artifact_binding_kind",
        "served_model_manifest_path",
        "served_model_manifest_sha256",
        "served_package_manifest_sha256",
    }
    if artifact_keys == base_artifact_keys:
        artifact = _exact_keys(identity["artifact"], base_artifact_keys, "identity.artifact")
        # Legacy records remain readable, but package-only promotion must use the
        # explicit served-model binding fields below.
    elif artifact_keys == package_bound_keys:
        artifact = _exact_keys(identity["artifact"], package_bound_keys, "identity.artifact")
        if artifact["artifact_binding_kind"] not in {"artifact_manifest", "package_manifest"}:
            raise OracleError("identity.artifact.artifact_binding_kind is invalid")
        if artifact["artifact_binding_kind"] == "package_manifest" and artifact["artifact_manifest_sha256"] is not None:
            raise OracleError("package_manifest binding must not carry an artifact manifest hash")
        if artifact["artifact_binding_kind"] == "artifact_manifest" and artifact["artifact_manifest_sha256"] is None:
            raise OracleError("artifact_manifest binding requires an artifact manifest hash")
        if not isinstance(artifact["served_model_manifest_path"], str) or not artifact["served_model_manifest_path"]:
            raise OracleError("identity.artifact.served_model_manifest_path is invalid")
        ensure_sha256(artifact["served_model_manifest_sha256"], "identity.artifact.served_model_manifest_sha256")
        ensure_sha256(artifact["served_package_manifest_sha256"], "identity.artifact.served_package_manifest_sha256")
    else:
        raise OracleError("identity.artifact keys differ")
    for key in ("package_manifest_sha256", "artifact_manifest_sha256"):
        if artifact[key] is not None:
            ensure_sha256(artifact[key], f"identity.artifact.{key}")
    tokenizer = _exact_keys(identity["tokenizer"], {"aggregate_sha256", "files", "root"}, "identity.tokenizer")
    if not isinstance(tokenizer["root"], str) or not Path(tokenizer["root"]).is_absolute():
        raise OracleError("identity.tokenizer.root must be absolute")
    ensure_sha256(tokenizer["aggregate_sha256"], "identity.tokenizer.aggregate_sha256")
    tokenizer_files = _validate_file_identity(tokenizer["files"], "identity.tokenizer.files")
    if tokenizer["aggregate_sha256"] != canonical_sha256(tokenizer_files):
        raise OracleError("identity.tokenizer aggregate differs")
    source_checkpoint = identity["source_checkpoint"]
    if kind == "independent_source":
        source_checkpoint = _exact_keys(source_checkpoint, {"aggregate_sha256", "dtype", "files", "root"}, "identity.source_checkpoint")
        if not isinstance(source_checkpoint["root"], str) or not Path(source_checkpoint["root"]).is_absolute():
            raise OracleError("identity.source_checkpoint.root must be absolute")
        if source_checkpoint["dtype"] not in {"bfloat16", "float32", "bf16", "f32"}:
            raise OracleError("identity.source_checkpoint.dtype must be BF16/F32")
        checkpoint_files = _validate_file_identity(source_checkpoint["files"], "identity.source_checkpoint.files")
        ensure_sha256(source_checkpoint["aggregate_sha256"], "identity.source_checkpoint.aggregate_sha256")
        if source_checkpoint["aggregate_sha256"] != canonical_sha256(checkpoint_files):
            raise OracleError("identity.source_checkpoint aggregate differs")
    elif source_checkpoint is not None:
        raise OracleError("same-artifact path oracle must not claim a source checkpoint")
    limits = _exact_keys(manifest.get("limits"), {"max_cases", "max_payload_bytes", "max_sample_values", "max_steps", "max_top_k"}, "limits")
    expected_limits = {"max_cases": MAX_CASES, "max_payload_bytes": MAX_PAYLOAD_BYTES, "max_sample_values": MAX_SAMPLE_VALUES, "max_steps": MAX_STEPS, "max_top_k": MAX_TOP_K}
    if limits != expected_limits:
        raise OracleError("manifest limits differ from the fixed bounded contract")
    payload = _exact_keys(manifest.get("payload"), {"bytes", "file", "record_count", "sha256"}, "payload")
    payload_path = safe_relative(root, payload["file"], "payload.file")
    ensure_sha256(payload["sha256"], "payload.sha256")
    payload_bytes = integer(payload["bytes"], "payload.bytes", minimum=1)
    records = integer(payload["record_count"], "payload.record_count", minimum=1)
    actual_sha, actual_bytes, actual_records = digest_payload(payload_path)
    if (actual_sha, actual_bytes, actual_records) != (payload["sha256"], payload_bytes, records):
        raise OracleError("payload hash, byte count, or record count differs")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise OracleError("manifest cases exceed bounded contract")
    expected_pairs: set[tuple[str, int]] = set()
    for index, raw in enumerate(cases):
        case = _exact_keys(raw, {"case_id", "prompt_token_count", "prompt_token_ids_sha256", "step_count"}, f"cases[{index}]")
        if not isinstance(case["case_id"], str) or not case["case_id"]:
            raise OracleError("case_id is invalid")
        integer(case["prompt_token_count"], f"cases[{index}].prompt_token_count", minimum=1)
        ensure_sha256(case["prompt_token_ids_sha256"], f"cases[{index}].prompt_token_ids_sha256")
        step_count = integer(case["step_count"], f"cases[{index}].step_count", minimum=1)
        if step_count > MAX_STEPS:
            raise OracleError("case step_count exceeds bounded contract")
        for step in range(step_count):
            if (case["case_id"], step) in expected_pairs:
                raise OracleError("duplicate case and step")
            expected_pairs.add((case["case_id"], step))
    seen_pairs: set[tuple[str, int]] = set()
    for record in iter_payload(payload_path):
        key = (record["case_id"], record["step"])
        if key in seen_pairs:
            raise OracleError("duplicate payload case and step")
        seen_pairs.add(key)
    if seen_pairs != expected_pairs:
        raise OracleError("payload case/step coverage differs from manifest")
    return manifest


def payload_records(root: Path, manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    path = safe_relative(root, manifest["payload"]["file"], "payload.file")
    yield from iter_payload(path)


def canonical_token_ids_hash(token_ids: Iterable[int]) -> str:
    values = [integer(item, "token_id", minimum=0) for item in token_ids]
    return hashlib.sha256((json.dumps(values, separators=(",", ":")) + "\n").encode()).hexdigest()


def compare_payloads(source_root: Path, source: dict[str, Any], path_root: Path, path: dict[str, Any], *, logit_atol: float = 1e-5, hidden_atol: float = 1e-5) -> dict[str, Any]:
    source_iter = iter(payload_records(source_root, source))
    path_iter = iter(payload_records(path_root, path))
    count = 0
    greedy_exact = True
    topk_exact = True
    hidden_shape_exact = True
    logit_shape_exact = True
    hidden_max = 0.0
    logit_max = 0.0
    hidden_relative_l2_max = 0.0
    hidden_cosine_min = 1.0
    logit_relative_l2_max = 0.0
    logit_cosine_min = 1.0
    topk_overlap_min = 1.0
    topk_overlap_sum = 0.0
    while True:
        try:
            left = next(source_iter)
        except StopIteration:
            left = None
        try:
            right = next(path_iter)
        except StopIteration:
            right = None
        if left is None or right is None:
            if left is not None or right is not None:
                raise OracleError("source/path payload record counts differ")
            break
        if (left["case_id"], left["step"]) != (right["case_id"], right["step"]):
            raise OracleError("source/path payload ordering differs")
        count += 1
        greedy_exact &= left["greedy_token_id"] == right["greedy_token_id"]
        topk_exact &= left["topk"] == right["topk"]
        left_topk = {entry["token_id"] for entry in left["topk"]}
        right_topk = {entry["token_id"] for entry in right["topk"]}
        topk_overlap = len(left_topk & right_topk) / max(len(left_topk), len(right_topk))
        topk_overlap_min = min(topk_overlap_min, topk_overlap)
        topk_overlap_sum += topk_overlap
        for field, target, tolerance in (("hidden_sample", "hidden_sample", hidden_atol), ("logit_sample", "logit_sample", logit_atol)):
            lsample, rsample = left[field], right[target]
            if lsample["shape"] != rsample["shape"]:
                if field == "hidden_sample":
                    hidden_shape_exact = False
                else:
                    logit_shape_exact = False
            left_values = dict(zip(lsample["indices"], lsample["values"]))
            right_values = dict(zip(rsample["indices"], rsample["values"]))
            common = sorted(set(left_values) & set(right_values))
            if not common:
                continue
            pairs = [(float(left_values[index]), float(right_values[index])) for index in common]
            delta = max(abs(a - b) for a, b in pairs)
            left_norm = math.sqrt(sum(a * a for a, _ in pairs))
            right_norm = math.sqrt(sum(b * b for _, b in pairs))
            diff_norm = math.sqrt(sum((a - b) * (a - b) for a, b in pairs))
            relative_l2 = diff_norm / max(left_norm, 1e-12)
            cosine = sum(a * b for a, b in pairs) / max(left_norm * right_norm, 1e-12)
            if field == "hidden_sample":
                hidden_max = max(hidden_max, delta)
                hidden_relative_l2_max = max(hidden_relative_l2_max, relative_l2)
                hidden_cosine_min = min(hidden_cosine_min, cosine)
            else:
                logit_max = max(logit_max, delta)
                logit_relative_l2_max = max(logit_relative_l2_max, relative_l2)
                logit_cosine_min = min(logit_cosine_min, cosine)
            if delta > tolerance:
                # Keep collecting bounded metrics; caller decides promotion.
                pass
    return {
        "record_count": count,
        "greedy_token_exact": greedy_exact,
        "topk_exact": topk_exact,
        "topk_overlap_min": topk_overlap_min,
        "topk_overlap_mean": topk_overlap_sum / count if count else 0.0,
        "hidden_sample_shape_exact": hidden_shape_exact,
        "logit_sample_shape_exact": logit_shape_exact,
        "hidden_sample_max_abs_diff": hidden_max,
        "hidden_sample_bounded_relative_l2_max": hidden_relative_l2_max,
        "hidden_sample_bounded_cosine_min": hidden_cosine_min,
        "logit_sample_max_abs_diff": logit_max,
        "logit_sample_bounded_relative_l2_max": logit_relative_l2_max,
        "logit_sample_bounded_cosine_min": logit_cosine_min,
        "bounded_metric_scope": "intersection_of_stored_indices",
        "hidden_sample_within_atol": hidden_shape_exact and hidden_max <= hidden_atol,
        "logit_sample_within_atol": logit_shape_exact and logit_max <= logit_atol,
    }
