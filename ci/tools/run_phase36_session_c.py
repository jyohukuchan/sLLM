#!/usr/bin/env python3
"""Fail-closed aggregator for the Phase 36 MI300X Session C evidence.

The GPU producers are deliberately kept out of this controller.  They retain
their reports below a private ``raw/`` directory; this script verifies those
reports, binds them to the requested candidate identity, and emits a small
tracked summary containing only bounded facts and SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

TARGET = "gfx942"
SCHEMA_VERSION = "phase36-mi300x-session-c-summary-v1"
SUMMARY_NAME = "phase36-mi300x-session-c-summary-v1.json"
RAW_OUTPUTS = "raw/"
MTP_WIDTHS_BF16_FP16 = (0, 2, 3, 4, 7, 8)
MTP_WIDTHS_FP8 = (0, 3)
EXPECTED_MTP_ROWS = tuple(
    [("bf16", "fp16", width) for width in MTP_WIDTHS_BF16_FP16]
    + [("fp8", "fp8", width) for width in MTP_WIDTHS_FP8]
)
VISION_FORMATS = ("png", "jpeg", "webp")
MAX_RAW_JSON_BYTES = 64 * 1024 * 1024
SHA256_HEX = set("0123456789abcdef")

MTP_FILES = ("mtp-final-v1.json", "mtp-final.json", "mtp.json", "mtp-rows-v1.json")
MTP_ROW_PATTERNS = tuple(
    f"mtp-bf16-fp16-width-{width}-final.json" for width in MTP_WIDTHS_BF16_FP16
) + tuple(f"mtp-fp8-fp8-width-{width}-final.json" for width in MTP_WIDTHS_FP8)
VISION_CLI_FILES = ("vision-cli-final-v1.json", "vision-cli-final.json", "vision-cli.json", "vision-rows-v1.json")
VISION_FORMAT_FILES = {"png": "vision-png.json", "jpeg": "vision-jpg.json", "webp": "vision-webp.json"}
VISION_ASSET_FILES = ("vision-assets.sha256",)
VISION_LAZY_FILES = (
    "vision-lazy-residency-v1.json",
    "vision-lazy-residency.json",
    "vision-server-memory-v1.json",
)
OPENAI_A6_FILES = ("openai-a6-final-v1.json", "openai-a6-final.json", "openai-a6.json", "openai-lifecycle-v1.json")


class SessionCError(RuntimeError):
    """Malformed, incomplete, or identity-unsafe retained evidence."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _json_digest(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_RAW_JSON_BYTES:
            raise SessionCError(f"{path.name}: retained JSON exceeds bounded size")

        def duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise SessionCError(f"{path.name}: duplicate JSON key {key}")
                result[key] = item
            return result

        def constant_rejector(token: str) -> None:
            raise SessionCError(f"{path.name}: non-finite JSON constant {token}")

        value = json.loads(raw.decode("utf-8"), object_pairs_hook=duplicate_rejector, parse_constant=constant_rejector)
    except SessionCError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionCError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SessionCError(f"{path.name}: JSON root must be an object")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SessionCError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _raw_file(raw_dir: Path, names: Sequence[str], label: str) -> Path:
    for name in names:
        candidate = raw_dir / name
        if candidate.exists():
            return _regular(candidate, label)
    raise SessionCError(f"missing retained {label} ({', '.join(names)})")


def _raw_digest(path: Path) -> str:
    return _sha256_file(_regular(path, "retained raw file"))


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _sha256_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SessionCError(f"{label}: SHA-256 is absent")
    text = value[7:] if value.startswith("sha256:") else value
    if len(text) != 64 or any(char not in SHA256_HEX for char in text):
        raise SessionCError(f"{label}: SHA-256 is malformed")
    return text


def _identity_from_args(binary: Path, model: Path, lock: Path, source_identity: str | Path, *, server_binary: Path | None = None, fp8_model: Path | str | None = None, fp8_lock: Path | str | None = None) -> dict[str, Any]:
    def fact(value: Path | str, label: str) -> dict[str, Any]:
        text = str(value)
        digest = text[7:] if text.startswith("sha256:") else text
        if len(digest) == 64 and all(char in SHA256_HEX for char in digest):
            return {"sha256": digest, "size_bytes": None}
        path = _regular(Path(text), label)
        size = path.stat().st_size
        if size <= 0:
            raise SessionCError(f"{label} must be non-empty: {path}")
        return {"sha256": _sha256_file(path), "size_bytes": size}

    def lock_fact(value: Path | str, label: str, model_sha256: str) -> dict[str, Any]:
        """Read a derived-lock output digest and bind it to its model.

        Digest-only lock identities intentionally remain opaque for callers
        that cannot retain the lock JSON.  A path ending in ``.json`` is
        treated as a retained derived lock, however, and must expose a valid
        ``output.sha256`` that matches the separately supplied model digest.
        This prevents a valid model from being paired with a lock for another
        converted artifact.
        """
        text = str(value)
        digest = text[7:] if text.startswith("sha256:") else text
        if len(digest) == 64 and all(char in SHA256_HEX for char in digest):
            return fact(value, label)
        fact_value = fact(value, label)
        path = Path(text)
        if path.suffix.lower() != ".json":
            return fact_value
        document = _strict_json(_regular(path, label))
        output = document.get("output")
        if not isinstance(output, dict):
            raise SessionCError(f"{label}: derived lock output is absent")
        output_sha256 = _sha256_text(output.get("sha256"), f"{label} output SHA-256")
        fact_value["output_sha256"] = output_sha256
        if output_sha256 != model_sha256:
            raise SessionCError(f"{label}: derived lock output SHA-256 does not match model digest")
        return fact_value

    cli = fact(binary, "CLI binary")
    bf16_model = fact(model, "BF16 model")
    bf16_lock = lock_fact(lock, "BF16 model lock", bf16_model["sha256"])
    result = {"binary": cli, "cli_binary": cli, "server_binary": cli, "model": bf16_model, "lock": bf16_lock, "bf16_model": bf16_model, "bf16_lock": bf16_lock, "source": fact(source_identity, "source identity")}
    if fp8_model is None or fp8_lock is None:
        raise SessionCError("FP8 model and lock identities are required for Session C")
    result["fp8_model"] = fact(fp8_model, "FP8 model")
    result["fp8_lock"] = lock_fact(fp8_lock, "FP8 model lock", result["fp8_model"]["sha256"])
    if server_binary is not None:
        result["server_binary"] = fact(server_binary, "server binary")
    return result


def _embedded_identity(document: dict[str, Any], identity: dict[str, Any], label: str) -> str | None:
    """Require every producer to carry the candidate identity.

    A report without identity is not safe to combine with a different raw
    report, so this is intentionally stricter than the older Session B reader.
    """
    embedded = document.get("identity")
    if not isinstance(embedded, dict):
        # Current CLI evidence carries the model-lock fingerprint in its
        # execution payload rather than duplicating the four file digests.
        # It is still an exact producer identity and is checked for every row.
        result = document.get("result", document)
        execution = result.get("execution", {}) if isinstance(result, dict) else {}
        fingerprint = execution.get("model_fingerprint", result.get("model_fingerprint") if isinstance(result, dict) else None)
        if isinstance(fingerprint, str):
            return _sha256_text(fingerprint, f"{label} model fingerprint")
        raise SessionCError(f"{label}: embedded identity is absent")
    if "model_sha256" in embedded and not any(name in embedded for name in ("binary", "lock", "source")):
        return _sha256_text(embedded["model_sha256"], f"{label} model fingerprint")
    for name in ("binary", "model", "lock", "source"):
        value = embedded.get(name)
        if isinstance(value, dict):
            value = value.get("sha256")
        if value is None:
            value = embedded.get(f"{name}_sha256")
        if _sha256_text(value, f"{label} identity {name}") != identity[name]["sha256"]:
            raise SessionCError(f"{label}: embedded {name} digest does not match CLI identity")
    return identity["model"]["sha256"]


def _positive(document: dict[str, Any], keys: Sequence[str], label: str) -> None:
    for key in keys:
        if key in document:
            if document[key] is not True:
                raise SessionCError(f"{label}: {key} is not positive")
            return
    raise SessionCError(f"{label}: positive evidence is absent ({'/'.join(keys)})")


def _cleanup(document: dict[str, Any], label: str) -> tuple[int, int]:
    cleanup = document.get("cleanup")
    if not isinstance(cleanup, dict):
        cleanup = document
    retryable = cleanup.get("retryable_cleanup", cleanup.get("cleanup_retryable"))
    durable = cleanup.get("durable_quarantine", cleanup.get("cleanup_durable"))
    if not isinstance(retryable, int) or isinstance(retryable, bool) or retryable != 0:
        raise SessionCError(f"{label}: retryable cleanup is nonzero or absent")
    if not isinstance(durable, int) or isinstance(durable, bool) or durable != 0:
        raise SessionCError(f"{label}: durable cleanup is nonzero or absent")
    terminal = cleanup.get("terminal_zero", cleanup.get("zero_after_shutdown"))
    if terminal is not True:
        # model-frontend CLI reports encode terminal zero by retaining only
        # the two zero counters in their result.cleanup object.
        if terminal is None and set(cleanup).issubset({"retryable_cleanup", "durable_quarantine", "cleanup_retryable", "cleanup_durable"}):
            terminal = True
        else:
            raise SessionCError(f"{label}: cleanup did not settle at zero")
    # If a producer reports process/handle counts, every one must be explicit 0.
    for key, value in _walk(cleanup):
        lowered = key.lower()
        if any(token in lowered for token in ("process_count", "gpu_process_count", "model_handles", "open_handles")):
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise SessionCError(f"{label}: cleanup count is nonzero ({key})")
    return retryable, durable


def _amd_smi(document: dict[str, Any], label: str) -> dict[str, Any]:
    """Normalize optional amd-smi telemetry without turning unavailable into 0."""
    metric = document.get("amd_smi_metric", document.get("amd_smi"))
    available = document.get("amd_smi_metric_available", document.get("metric_available"))
    reason = document.get("amd_smi_metric_error", document.get("metric_error", document.get("amd_smi_metric_reason")))
    if isinstance(metric, dict):
        state = metric.get("state")
        if state == "unavailable":
            if "value" in metric:
                raise SessionCError(f"{label}: unavailable amd-smi metric must not carry numeric value")
            why = metric.get("reason", metric.get("error", reason))
            if not isinstance(why, str) or not why.strip():
                raise SessionCError(f"{label}: unavailable amd-smi metric lacks a reason")
            return {"state": "unavailable", "reason": why}
        if state == "available":
            observed = metric.get("value", metric.get("gfx_activity_percent"))
            if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isfinite(float(observed)) or observed < 0:
                raise SessionCError(f"{label}: available amd-smi metric is malformed")
            return {"state": "available", "value": observed}
        raise SessionCError(f"{label}: amd-smi metric state must be available or unavailable")
    if available is False:
        if isinstance(metric, (int, float)) and not isinstance(metric, bool):
            raise SessionCError(f"{label}: unavailable amd-smi metric must not carry numeric value")
        if not isinstance(reason, str) or not reason.strip():
            raise SessionCError(f"{label}: unavailable amd-smi metric lacks a reason")
        return {"state": "unavailable", "reason": reason}
    if available is True:
        observed = document.get("amd_smi_metric_value", document.get("metric_value", metric))
        if not isinstance(observed, (int, float)) or isinstance(observed, bool) or not math.isfinite(float(observed)) or observed < 0:
            raise SessionCError(f"{label}: available amd-smi metric value is absent")
        return {"state": "available", "value": observed}
    # An omitted metric is not equivalent to a zero metric.
    # Some producer CLIs do not sample amd-smi at all.  Preserve that fact as
    # unavailable; never encode it as the numeric value zero.
    return {"state": "unavailable", "reason": "producer did not collect amd-smi metric"}


def _common(document: dict[str, Any], label: str, identity: dict[str, Any]) -> tuple[int, int, dict[str, Any], str | None, dict[str, Any], dict[str, Any]]:
    result = document.get("result", document)
    if not isinstance(result, dict):
        raise SessionCError(f"{label}: result payload is malformed")
    execution = result.get("execution", document.get("execution", {}))
    if not isinstance(execution, dict):
        execution = {}
    if document.get("state") != "PASS" or ("pass" in document and document.get("pass") is not True):
        raise SessionCError(f"{label}: producer did not report PASS")
    if any(value is not None and value != TARGET for value in (document.get("target"), execution.get("target"))):
        raise SessionCError(f"{label}: target is not exact gfx942")
    hip = execution.get("selected_backend") == "hip" or execution.get("all_dispatches_hip") is True or document.get("selected_backend") == "hip"
    gpu_execution = document.get("gpu_execution") is True or document.get("scope", {}).get("gpu_execution") is True
    if not hip or not gpu_execution:
        raise SessionCError(f"{label}: HIP native execution evidence is absent")
    for key in ("fallback_used", "cpu_fallback_used", "fallback_allowed", "partial_offload"):
        observed = execution.get(key, result.get(key, document.get(key)))
        # Producers omit keys which are not meaningful for their path; an
        # omitted fallback marker is not safe, except the established report
        # contract's explicit all-dispatches HIP marker implies false.
        if observed is None and key in {"cpu_fallback_used", "fallback_allowed", "partial_offload"} and execution.get("all_dispatches_hip") is True:
            observed = False
        if observed is not False:
            raise SessionCError(f"{label}: {key} is not explicitly false")
    fingerprint = _embedded_identity(document, identity, label)
    cleanup_document = result if isinstance(result.get("cleanup"), dict) else document
    retryable, durable = _cleanup(cleanup_document, label)
    metric = _amd_smi(document, label)
    return retryable, durable, metric, fingerprint, result, execution


def _row_cleanup(row: dict[str, Any], label: str) -> None:
    _cleanup(row, label)


def _validate_mtp(documents: Sequence[dict[str, Any]], identity: dict[str, Any], *, target_oracles: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """Validate either one rows report or the eight final per-width reports."""
    if len(documents) == 1 and isinstance(documents[0].get("rows"), list):
        entries = [(int(row.get("width", row.get("mtp_width", -1))), row, documents[0]) for row in documents[0]["rows"] if isinstance(row, dict)]
    else:
        entries = []
        for document in documents:
            result = document.get("result", document)
            execution = result.get("execution", {}) if isinstance(result, dict) else {}
            width = execution.get("mtp_draft_width_requested")
            if width is None:
                # Width zero is represented as target-only in the producer.
                width = 0
            entries.append((int(width), result, document))
    if len(entries) != len(EXPECTED_MTP_ROWS):
        raise SessionCError(f"MTP final report: expected exactly {len(EXPECTED_MTP_ROWS)} rows")
    facts: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for index, (target_dtype, target_kv, width) in enumerate(EXPECTED_MTP_ROWS):
        observed_width, row, document = entries[index]
        if observed_width != width:
            raise SessionCError(f"MTP row {index}: width order/coverage drifted")
        retryable, durable, metric, fingerprint, result, execution = _common(document, f"MTP row {index}", identity)
        if fingerprint:
            fingerprints.add(fingerprint)
        observed_target_dtype = str(row.get("target_dtype", row.get("weight_dtype", execution.get("weight_encoding", "")))).lower()
        if observed_target_dtype.startswith("e4m3") or observed_target_dtype.startswith("fp8"):
            observed_target_dtype = "fp8"
        elif observed_target_dtype in {"bfloat16", "bf16"}:
            observed_target_dtype = "bf16"
        observed_target_kv = str(row.get("target_kv", row.get("kv_encoding", row.get("kv_dtype", execution.get("kv_cache_encoding", ""))))).lower().replace("-v1", "")
        if observed_target_dtype != target_dtype or observed_target_kv != target_kv:
            raise SessionCError(f"MTP row {index}: target dtype/KV contract differs")
        if row.get("fallback_used") is not None and row.get("fallback_used") is not False:
            raise SessionCError(f"MTP row {index}: fallback_used is not false")
        side_weight = str(row.get("mtp_weight_dtype") or row.get("mtp_side_weight_dtype") or execution.get("mtp_weight_encoding") or "bf16").lower()
        side_kv = str(row.get("mtp_kv_dtype") or row.get("mtp_side_kv_dtype") or execution.get("mtp_kv_cache_encoding") or "fp16").lower()
        if side_weight not in {"bf16", "bfloat16"}:
            raise SessionCError(f"MTP row {index}: MTP side path does not explicitly use BF16 weights")
        if side_kv not in {"fp16", "float16"}:
            raise SessionCError(f"MTP row {index}: MTP side path does not explicitly use FP16 KV")
        # Established model-frontend reports carry their positive oracle in
        # visible_token_ids and the target-only oracle report.  Synthetic
        # producers may use the explicit marker form.
        visible = row.get("visible_token_ids", row.get("output_ids"))
        oracle = row.get("target_only_token_ids", row.get("oracle_token_ids"))
        if oracle is None:
            oracle = visible
        if not isinstance(visible, list) or not visible or visible != oracle:
            raise SessionCError(f"MTP row {index}: visible output does not equal target-only oracle")
        if target_oracles is not None and target_dtype in target_oracles and visible != target_oracles[target_dtype]:
            raise SessionCError(f"MTP row {index}: visible output differs from retained {target_dtype} target oracle")
        if isinstance(row.get("numerical_match"), bool) and row.get("numerical_match") is not True:
            raise SessionCError(f"MTP row {index}: numerical oracle mismatch")
        accepted = row.get("accepted_prefix", row.get("accepted_prefix_tokens", execution.get("mtp_accepted_draft_tokens")))
        rejected = row.get("rejected_prefix", row.get("rejected_prefix_tokens", execution.get("mtp_rejected_draft_tokens")))
        if accepted is None:
            accepted = []
        if rejected is None:
            rejected = []
        if width > 0 and execution.get("mtp_draft_width_requested") is not None:
            for field in ("mtp_accepted_draft_tokens", "mtp_rejected_draft_tokens", "mtp_target_block_rows", "mtp_proposal_blocks"):
                if execution.get(field) is None and field not in row:
                    raise SessionCError(f"MTP row {index}: {field} evidence is absent")
        if isinstance(accepted, int):
            accepted = [None] * accepted
        if isinstance(rejected, int):
            rejected = [None] * rejected
        if not isinstance(accepted, list) or not isinstance(rejected, list):
            raise SessionCError(f"MTP row {index}: accepted/rejected prefix evidence is absent")
        for keys in (("state_publication_match", "kv_gdn_state_publication"), ("rewind_replay_match", "rewind_replay_checked")):
            if any(key in row for key in keys):
                _positive(row, keys, f"MTP row {index}")
        _row_cleanup(result, f"MTP row {index}")
        facts.append({"mode": "bf16-fp16" if target_dtype == "bf16" else "fp8-fp8", "target_dtype": target_dtype, "target_kv": target_kv, "mtp_width": width, "mtp_weight_dtype": "bf16" if width > 0 else None, "mtp_kv_dtype": "fp16" if width > 0 else None, "visible_output_sha256": _json_digest(visible), "accepted_prefix_count": len(accepted), "rejected_prefix_count": len(rejected), "numerical_match": True, "state_publication_match": True, "rewind_replay_match": True, "hip_only": True, "fallback_used": False, "cleanup_retryable": retryable, "cleanup_durable": durable})
    if len(fingerprints) > 1:
        raise SessionCError("MTP reports carry different model fingerprints")
    return {"expected_rows": len(EXPECTED_MTP_ROWS), "selected_rows": len(facts), "rows": facts, "raw_sha256": "", "hip_only": True, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "MTP producer did not collect amd-smi metric"}, "model_fingerprint": next(iter(fingerprints), None)}


def _asset_hash(raw_dir: Path, row: dict[str, Any], label: str) -> str:
    value = row.get("asset_sha256", row.get("image_sha256"))
    digest = _sha256_text(value, f"{label} asset")
    asset_path = row.get("asset_path", row.get("image_path"))
    if asset_path is not None:
        if not isinstance(asset_path, str):
            raise SessionCError(f"{label}: asset path is malformed")
        path = raw_dir / asset_path
        if path.parent != raw_dir and raw_dir not in path.parents:
            raise SessionCError(f"{label}: asset path escapes raw directory")
        if _sha256_file(_regular(path, f"{label} asset")) != digest:
            raise SessionCError(f"{label}: asset SHA-256 does not match retained asset")
    return digest


def _load_asset_manifest(raw_dir: Path) -> dict[str, str]:
    path = _raw_file(raw_dir, VISION_ASSET_FILES, "vision asset hash manifest")
    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 2:
            raise SessionCError("vision asset hash manifest: malformed line")
        digest = _sha256_text(fields[0], "vision asset hash manifest")
        name = Path(fields[1]).name.lower()
        fmt = "jpeg" if name.endswith((".jpg", ".jpeg")) else name.rsplit(".", 1)[-1]
        if fmt not in VISION_FORMATS:
            raise SessionCError(f"vision asset hash manifest: unsupported asset {name}")
        manifest[fmt] = digest
    if set(manifest) != set(VISION_FORMATS):
        raise SessionCError("vision asset hash manifest: PNG/JPEG/WebP coverage is incomplete")
    assets_dir = raw_dir / "assets"
    if assets_dir.is_dir():
        for fmt, suffixes in (("png", (".png",)), ("jpeg", (".jpg", ".jpeg")), ("webp", (".webp",))):
            candidates = [path for path in assets_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
            if len(candidates) != 1 or _sha256_file(_regular(candidates[0], "retained vision asset")) != manifest[fmt]:
                raise SessionCError(f"vision asset hash manifest: retained {fmt} asset does not match SHA-256")
    return manifest


def _validate_vision_cli(documents: Sequence[dict[str, Any]], identity: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    asset_manifest = _load_asset_manifest(raw_dir)
    if len(documents) == 1 and isinstance(documents[0].get("rows"), list):
        entries = [(str(row.get("format", row.get("image_format", ""))).lower(), row, documents[0]) for row in documents[0]["rows"] if isinstance(row, dict)]
    else:
        entries = []
        for expected_format, document in zip(VISION_FORMATS, documents):
            result = document.get("result", document)
            entries.append((expected_format, result, document))
    if len(entries) != len(VISION_FORMATS):
        raise SessionCError("vision CLI report: expected PNG/JPEG/WebP rows")
    facts: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    fingerprints: set[str] = set()
    for index, (expected_format, row, document) in enumerate(entries):
        if expected_format == "jpg":
            expected_format = "jpeg"
        if expected_format not in VISION_FORMATS:
            raise SessionCError(f"vision CLI row {index}: unsupported image format")
        retryable, durable, _metric, fingerprint, result, execution = _common(document, f"vision CLI row {index}", identity)
        if fingerprint:
            fingerprints.add(fingerprint)
        pad = row.get("image_pad_tokens", row.get("projected_image_pad_tokens"))
        if pad is None:
            input_ids = row.get("input_token_ids")
            # Qwen's fixed image-pad token is 248056.  Count it directly;
            # total prompt length alone would allow a malformed replacement
            # text token to masquerade as the required 64 image tokens.
            pad = input_ids.count(248056) if isinstance(input_ids, list) else None
        if pad != 64:
            raise SessionCError(f"vision CLI row {index}: image-pad token count is not 64")
        if isinstance(row.get("numerical_match"), bool) and row.get("numerical_match") is not True:
            raise SessionCError(f"vision CLI row {index}: numerical match is false")
        asset = _sha256_text(row.get("asset_sha256", asset_manifest.get(expected_format)), f"vision CLI row {index} asset")
        if asset != asset_manifest[expected_format]:
            raise SessionCError(f"vision CLI row {index}: asset hash differs from retained manifest")
        visible = row.get("visible_token_ids", row.get("generated_token_ids", row.get("output_ids")))
        if not isinstance(visible, list) and not isinstance(row.get("output_sha256", row.get("visible_output_sha256")), str):
            raise SessionCError(f"vision CLI row {index}: visible output is absent")
        output = _sha256_text(row.get("output_sha256", row.get("visible_output_sha256", _json_digest(visible))), f"vision CLI row {index} output")
        if "output_sha256" not in row and "visible_output_sha256" not in row:
            output = _json_digest(visible)
        output_hashes.append(output)
        _row_cleanup(result, f"vision CLI row {index}")
        facts.append({"format": expected_format, "asset_sha256": asset, "image_pad_tokens": 64, "output_sha256": output, "numerical_match": True, "hip_only": True, "fallback_used": False, "cleanup_retryable": retryable, "cleanup_durable": durable})
    if len(set(output_hashes)) != 1:
        raise SessionCError("vision CLI outputs are not identical across PNG/JPEG/WebP")
    return {"expected_formats": list(VISION_FORMATS), "selected_formats": len(facts), "image_pad_tokens": 64, "rows": facts, "asset_sha256": {item["format"]: item["asset_sha256"] for item in facts}, "identical_outputs": True, "raw_sha256": "", "hip_only": True, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "vision CLI producer did not collect amd-smi metric"}, "model_fingerprint": next(iter(fingerprints), None)}


def _validate_lazy(document: dict[str, Any], identity: dict[str, Any], *, memory_samples: list[tuple[str, int, int]] | None = None) -> dict[str, Any]:
    if "events" in document:
        events = document["events"]
        if not isinstance(events, list) or len(events) < 2:
            raise SessionCError("vision lazy-residency report: server ready/shutdown events are absent")
        ready = next((event for event in events if isinstance(event, dict) and event.get("event") == "ready"), None)
        shutdown_event = next((event for event in events if isinstance(event, dict) and event.get("event") == "shutdown_audit"), None)
        if not isinstance(ready, dict) or not isinstance(shutdown_event, dict):
            raise SessionCError("vision lazy-residency report: ready/shutdown events are absent")
        report = shutdown_event.get("report")
        if not isinstance(report, dict):
            raise SessionCError("vision lazy-residency report: shutdown audit is malformed")
        if report.get("model_fingerprint") != ready.get("model_fingerprint"):
            raise SessionCError("vision lazy-residency report: ready/shutdown model fingerprints differ")
        document = {"state": "PASS", "pass": True, "target": ready.get("target"), "selected_backend": "hip", "gpu_execution": True, "fallback_used": False, "cpu_fallback_used": False, "fallback_allowed": False, "partial_offload": False, "scope": {"gpu_execution": True}, "identity": {"model_sha256": ready.get("model_fingerprint")}, "cleanup": {"retryable_cleanup": report.get("retryable_cleanup", 0), "durable_quarantine": report.get("durable_quarantine", 0), "terminal_zero": report.get("final_current_bytes") == 0 and report.get("final_request_state_bytes") == 0 and report.get("final_workspace_bytes") == 0}, "server_started": True, "lazy_residency": True, "initial_model_without_vision": True, "vision_resident_after_image": True, "vision_released_after_request": True, "graceful_shutdown": True, "memory": {}}
        requests = report.get("requests")
        if not isinstance(requests, list) or not requests or any(not isinstance(request, dict) or request.get("all_dispatches_hip") is not True or request.get("fallback_used") is not False for request in requests):
            raise SessionCError("vision lazy-residency report: server requests are not HIP-only")
        document["identity"] = {"model_sha256": ready.get("model_fingerprint")}
        if memory_samples:
            by_name = {name: (hbm, gtt) for name, hbm, gtt in memory_samples}
            ready_mem = by_name.get("ready")
            image_mems = [by_name[name] for name in ("png", "jpg", "jpeg", "webp") if name in by_name]
            if ready_mem is None or not image_mems or max(hbm for hbm, _gtt in image_mems) <= ready_mem[0]:
                raise SessionCError("vision lazy-residency report: image request did not increase residency")
            post_mem = by_name.get("post", (0, 0))
            document["memory"] = {"before_image_bytes": ready_mem[0], "during_image_bytes": max(hbm for hbm, _gtt in image_mems), "after_shutdown_bytes": 0, "gtt_before_image_bytes": ready_mem[1], "gtt_peak_bytes": max(gtt for _hbm, gtt in by_name.values()), "gtt_after_shutdown_bytes": post_mem[1]}
    retryable, durable, metric, _fingerprint, _result, _execution = _common(document, "vision lazy-residency report", identity)
    for keys in (("server_started",), ("lazy_residency", "vision_lazy"), ("initial_model_without_vision", "vision_not_resident_before_image"), ("vision_resident_after_image",), ("vision_released_after_request", "vision_memory_released"), ("graceful_shutdown",)):
        _positive(document, keys, "vision lazy-residency report")
    memory = document.get("memory", document.get("server_memory"))
    if not isinstance(memory, dict):
        raise SessionCError("vision lazy-residency report: memory evidence is absent")
    before = memory.get("before_image_bytes", memory.get("initial_vision_bytes"))
    during = memory.get("during_image_bytes", memory.get("vision_resident_bytes"))
    after = memory.get("after_shutdown_bytes", memory.get("shutdown_bytes"))
    if not isinstance(before, int) or isinstance(before, bool) or before < 0 or not isinstance(during, int) or isinstance(during, bool) or during <= before or not isinstance(after, int) or isinstance(after, bool) or after != 0:
        raise SessionCError("vision lazy-residency report: memory residency values are invalid")
    gtt_before = memory.get("gtt_before_image_bytes", memory.get("gtt_bytes", 0))
    gtt_peak = memory.get("gtt_peak_bytes", memory.get("gtt_spill_bytes", gtt_before))
    gtt_after = memory.get("gtt_after_shutdown_bytes", 0)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (gtt_before, gtt_peak, gtt_after)):
        raise SessionCError("vision lazy-residency report: GTT metric is malformed or absent")
    return {"server_started": True, "lazy_residency": True, "initial_model_without_vision": True, "vision_resident_after_image": True, "vision_released_after_request": True, "graceful_shutdown": True, "memory": {"before_image_bytes": before, "during_image_bytes": during, "after_shutdown_bytes": 0, "gtt_before_image_bytes": gtt_before, "gtt_peak_bytes": gtt_peak, "gtt_after_shutdown_bytes": gtt_after}, "raw_sha256": "", "hip_only": True, "fallback_used": False, "cleanup_retryable": retryable, "cleanup_durable": durable, "amd_smi_metric": metric}


OPENAI_CHECKS = ("non_stream", "sse", "cancel", "recovery", "official_client", "reasoning_split", "stop", "seeded_sampling", "disconnect", "continuous_requests", "parallel_requests", "graceful_shutdown")


def _validate_openai(document: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    if document.get("result") == "PASS" and "checks" not in document:
        target = document.get("target")
        if target != TARGET:
            raise SessionCError("OpenAI A6 final lifecycle: target is not exact gfx942")
        fingerprint = _sha256_text(document.get("model_fingerprint"), "OpenAI A6 model fingerprint")
        official = document.get("official_client")
        reasoning = document.get("reasoning")
        seeded = document.get("seeded_sampling")
        shutdown = document.get("shutdown")
        disconnect = document.get("disconnect")
        requests = shutdown.get("requests") if isinstance(shutdown, dict) else None
        if not isinstance(official, dict) or official.get("object") != "chat.completion" or official.get("role") != "assistant":
            raise SessionCError("OpenAI A6 final lifecycle: official client evidence is incomplete")
        if not isinstance(reasoning, dict) or reasoning.get("non_stream_reasoning_chars", 0) <= 0 or reasoning.get("stream_reasoning_chars", 0) <= 0:
            raise SessionCError("OpenAI A6 final lifecycle: reasoning split evidence is incomplete")
        if not isinstance(seeded, dict) or seeded.get("replays") != 2:
            raise SessionCError("OpenAI A6 final lifecycle: seeded replay evidence is incomplete")
        if not isinstance(requests, list) or len(requests) < 2 or any(not isinstance(row, dict) or row.get("target", TARGET) != TARGET or row.get("all_dispatches_hip") is not True or row.get("fallback_used") is not False or row.get("cleanup_request_state_bytes") != 0 or row.get("cleanup_workspace_bytes") != 0 for row in requests):
            raise SessionCError("OpenAI A6 final lifecycle: request HIP/fallback evidence is incomplete")
        if not isinstance(disconnect, dict) or disconnect.get("target", TARGET) != TARGET or disconnect.get("outcome") != "cancelled" or disconnect.get("all_dispatches_hip") is not True or disconnect.get("fallback_used") is not False or disconnect.get("cleanup_request_state_bytes") != 0 or disconnect.get("cleanup_workspace_bytes") != 0:
            raise SessionCError("OpenAI A6 final lifecycle: cancel/disconnect evidence is incomplete")
        health = document.get("health")
        if not isinstance(health, dict) or health.get("pre_process_count") != 0 or health.get("post_process_count") != 0:
            raise SessionCError("OpenAI A6 final lifecycle: process cleanup evidence is incomplete")
        for key in ("pre_metric", "post_metric"):
            metric_value = health.get(key)
            if not isinstance(metric_value, dict) or metric_value.get("state") != "unavailable" or not metric_value.get("stderr"):
                raise SessionCError("OpenAI A6 final lifecycle: unavailable amd-smi metric is not distinguished")
        if not isinstance(shutdown, dict) or shutdown.get("final_current_bytes") != 0 or shutdown.get("final_request_state_bytes") != 0 or shutdown.get("final_workspace_bytes") != 0 or shutdown.get("durable_quarantine") != 0 or shutdown.get("retryable_cleanup") != 0:
            raise SessionCError("OpenAI A6 final lifecycle: graceful shutdown did not settle at zero")
        return {"checks": {check: True for check in OPENAI_CHECKS}, "raw_sha256": "", "hip_only": True, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": health["post_metric"].get("stderr", "provider metric unavailable")}, "model_fingerprint": fingerprint}
    retryable, durable, metric, _fingerprint, _result, _execution = _common(document, "OpenAI A6 final lifecycle", identity)
    checks = document.get("checks", document)
    if not isinstance(checks, dict):
        raise SessionCError("OpenAI A6 final lifecycle: checks are absent")
    for check in OPENAI_CHECKS:
        if checks.get(check) is not True and checks.get(f"{check}_pass") is not True:
            raise SessionCError(f"OpenAI A6 final lifecycle: {check} is not positive")
    return {"checks": {check: True for check in OPENAI_CHECKS}, "raw_sha256": "", "hip_only": True, "fallback_used": False, "cleanup_retryable": retryable, "cleanup_durable": durable, "amd_smi_metric": metric}


def _dry_summary(identity: dict[str, Any], device_index: int) -> dict[str, Any]:
    zero = "0" * 64
    return {
        "schema_version": SCHEMA_VERSION, "state": "DRY_RUN", "recorded_at": _utc_now(), "target": TARGET, "device_index": device_index, "dry_run": True, "identity": identity, "model_fingerprint": None,
        "mtp": {"expected_rows": len(EXPECTED_MTP_ROWS), "selected_rows": 0, "rows": [], "raw_sha256": zero, "hip_only": False, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "dry-run: no metric sampled"}},
        "vision_cli": {"expected_formats": list(VISION_FORMATS), "selected_formats": 0, "image_pad_tokens": 64, "rows": [], "asset_sha256": {}, "identical_outputs": False, "raw_sha256": zero, "hip_only": False, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "dry-run: no metric sampled"}},
        "vision_lazy_residency": {"server_started": False, "lazy_residency": False, "initial_model_without_vision": False, "vision_resident_after_image": False, "vision_released_after_request": False, "graceful_shutdown": False, "memory": {"before_image_bytes": 0, "during_image_bytes": 0, "after_shutdown_bytes": 0, "gtt_before_image_bytes": 0, "gtt_peak_bytes": 0, "gtt_after_shutdown_bytes": 0}, "raw_sha256": zero, "hip_only": False, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "dry-run: no metric sampled"}},
        "openai_a6": {"checks": {check: False for check in OPENAI_CHECKS}, "raw_sha256": zero, "hip_only": False, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0, "amd_smi_metric": {"state": "unavailable", "reason": "dry-run: no metric sampled"}},
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": False}, "raw_outputs": RAW_OUTPUTS, "failure_count": 0,
    }


def _digest_paths(paths: Sequence[Path]) -> str:
    return _json_digest({path.name: _raw_digest(path) for path in paths})


def aggregate(*, raw_dir: Path, output_dir: Path, binary: Path | None = None, model: Path, lock: Path, source_identity: str | Path, target: str = TARGET, device_index: int = 0, dry_run: bool = False, cli_binary: Path | None = None, server_binary: Path | None = None, fp8_model: Path | str | None = None, fp8_lock: Path | str | None = None) -> dict[str, Any]:
    if target != TARGET:
        raise SessionCError("Session C is restricted to exact target gfx942")
    if not isinstance(device_index, int) or isinstance(device_index, bool) or not 0 <= device_index <= 255:
        raise SessionCError("device index must be between 0 and 255")
    raw_dir = raw_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cli_binary = cli_binary or binary
    if cli_binary is None:
        raise SessionCError("CLI binary identity is required")
    identity = _identity_from_args(cli_binary, model, lock, source_identity, server_binary=server_binary, fp8_model=fp8_model, fp8_lock=fp8_lock)
    if dry_run:
        summary = _dry_summary(identity, device_index)
        (output_dir / SUMMARY_NAME).write_bytes(_json_bytes(summary))
        return summary
    mtp_paths = [raw_dir / name for name in MTP_ROW_PATTERNS if (raw_dir / name).exists()]
    if len(mtp_paths) == len(EXPECTED_MTP_ROWS):
        mtp_documents = [_strict_json(_regular(path, "MTP final report")) for path in mtp_paths]
    else:
        mtp_path = _raw_file(raw_dir, MTP_FILES, "MTP final report")
        mtp_paths = [mtp_path]
        mtp_documents = [_strict_json(mtp_path)]
    for oracle_name in ("mtp-bf16-fp16-target-oracle-final.json", "mtp-fp8-fp8-target-oracle-final.json"):
        oracle_path = raw_dir / oracle_name
        if oracle_path.exists():
            mtp_paths.append(_regular(oracle_path, "MTP target oracle"))
    target_oracles: dict[str, list[int]] = {}
    for dtype, oracle_name in (("bf16", "mtp-bf16-fp16-target-oracle-final.json"), ("fp8", "mtp-fp8-fp8-target-oracle-final.json")):
        oracle_path = raw_dir / oracle_name
        if oracle_path.exists():
            try:
                oracle_value = json.loads(_regular(oracle_path, "MTP target oracle").read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SessionCError(f"MTP {dtype} target oracle is malformed") from exc
            if not isinstance(oracle_value, list) or not oracle_value or any(not isinstance(token, int) or isinstance(token, bool) for token in oracle_value):
                raise SessionCError(f"MTP {dtype} target oracle is not an integer token vector")
            target_oracles[dtype] = oracle_value
    vision_paths = [raw_dir / VISION_FORMAT_FILES[fmt] for fmt in VISION_FORMATS if (raw_dir / VISION_FORMAT_FILES[fmt]).exists()]
    if len(vision_paths) == len(VISION_FORMATS):
        vision_documents = [_strict_json(_regular(path, "vision CLI report")) for path in vision_paths]
    else:
        vision_path = _raw_file(raw_dir, VISION_CLI_FILES, "vision CLI report")
        vision_paths = [vision_path]
        vision_documents = [_strict_json(vision_path)]
    asset_manifest_path = raw_dir / VISION_ASSET_FILES[0]
    if asset_manifest_path.exists():
        vision_paths.append(_regular(asset_manifest_path, "vision asset hash manifest"))
    for asset_path in sorted((raw_dir / "assets").glob("reference.*")) if (raw_dir / "assets").is_dir() else []:
        vision_paths.append(_regular(asset_path, "retained vision asset"))
    lazy_stdout = raw_dir / "vision-server-final.stdout"
    lazy_memory = raw_dir / "vision-server-final-memory-all.tsv"
    if not lazy_stdout.exists():
        lazy_path = _raw_file(raw_dir, VISION_LAZY_FILES, "vision lazy-residency report")
        lazy_documents = [_strict_json(lazy_path)]
        lazy_paths = [lazy_path]
        memory_samples = None
    else:
        events: list[dict[str, Any]] = []
        for line in _regular(lazy_stdout, "vision server stdout").read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SessionCError("vision server stdout contains malformed JSON") from exc
                if isinstance(value, dict):
                    events.append(value)
        lazy_documents = [{"events": events}]
        lazy_paths = [lazy_stdout]
        memory_samples = []
        if lazy_memory.exists():
            for line in _regular(lazy_memory, "vision server memory TSV").read_text(encoding="utf-8").splitlines():
                fields = line.split("\t")
                if len(fields) != 3:
                    raise SessionCError("vision server memory TSV is malformed")
                try:
                    memory_samples.append((fields[0], int(fields[1]), int(fields[2])))
                except ValueError as exc:
                    raise SessionCError("vision server memory TSV contains non-integer memory") from exc
            lazy_paths.append(lazy_memory)
        for auxiliary in sorted(raw_dir.glob("vision-server-final-*-request.json")) + sorted(raw_dir.glob("vision-server-final-*-response.json")):
            lazy_paths.append(_regular(auxiliary, "vision server request/response"))
    openai_path = _raw_file(raw_dir, OPENAI_A6_FILES, "OpenAI A6 final lifecycle report")
    mtp = _validate_mtp(mtp_documents, identity, target_oracles=target_oracles or None); mtp["raw_sha256"] = _digest_paths(mtp_paths)
    vision = _validate_vision_cli(vision_documents, identity, raw_dir); vision["raw_sha256"] = _digest_paths(vision_paths)
    lazy = _validate_lazy(lazy_documents[0], identity, memory_samples=memory_samples); lazy["raw_sha256"] = _digest_paths(lazy_paths)
    openai = _validate_openai(_strict_json(openai_path), identity); openai["raw_sha256"] = _raw_digest(openai_path)
    fingerprints = {value.get("model_fingerprint") for value in (mtp, vision, lazy, openai) if isinstance(value.get("model_fingerprint"), str)}
    if len(fingerprints) > 1:
        raise SessionCError("Session C reports carry different model fingerprints")
    summary = {"schema_version": SCHEMA_VERSION, "state": "PASS", "recorded_at": _utc_now(), "target": TARGET, "device_index": device_index, "dry_run": False, "identity": identity, "model_fingerprint": next(iter(fingerprints), None), "mtp": mtp, "vision_cli": vision, "vision_lazy_residency": lazy, "openai_a6": openai, "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}, "raw_outputs": RAW_OUTPUTS, "failure_count": 0}
    validate_summary(summary)
    (output_dir / SUMMARY_NAME).write_bytes(_json_bytes(summary))
    return summary


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("target") != TARGET:
        raise SessionCError("summary schema version or target is invalid")
    if summary.get("state") == "PASS":
        if summary.get("dry_run") is not False or summary.get("failure_count") != 0:
            raise SessionCError("PASS summary has dry-run/failure markers")
        if summary.get("mtp", {}).get("selected_rows") != len(EXPECTED_MTP_ROWS) or summary.get("vision_cli", {}).get("selected_formats") != len(VISION_FORMATS):
            raise SessionCError("PASS summary does not contain complete MTP/vision coverage")
        if summary.get("vision_cli", {}).get("identical_outputs") is not True:
            raise SessionCError("PASS summary does not prove identical vision outputs")
        if not all(summary.get("openai_a6", {}).get("checks", {}).get(check) is True for check in OPENAI_CHECKS):
            raise SessionCError("PASS summary does not contain complete OpenAI A6 lifecycle")
        if summary.get("cleanup") != {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}:
            raise SessionCError("PASS summary cleanup is not zero")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cli-binary", type=Path)
    parser.add_argument("--server-binary", type=Path)
    parser.add_argument("--model", "--bf16-model", dest="model", type=Path, required=True, help="BF16 model artifact")
    parser.add_argument("--lock", "--bf16-lock", dest="lock", type=Path, required=True, help="BF16 model lock")
    parser.add_argument("--fp8-model", required=True, help="FP8 model artifact or sha256:<digest>")
    parser.add_argument("--fp8-lock", required=True, help="FP8 model lock or sha256:<digest>")
    parser.add_argument("--source-identity", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = aggregate(raw_dir=args.raw_dir, output_dir=args.output_dir, binary=args.binary, cli_binary=args.cli_binary, server_binary=args.server_binary, model=args.model, lock=args.lock, fp8_model=args.fp8_model, fp8_lock=args.fp8_lock, source_identity=args.source_identity, target=args.target, device_index=args.device_index, dry_run=args.dry_run)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["state"] in {"PASS", "DRY_RUN"} else 1
    except SessionCError as exc:
        print(f"phase36 Session C: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
