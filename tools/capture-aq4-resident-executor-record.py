#!/usr/bin/env python3
"""Capture a bounded executor record from the immutable AQ4 resident worker.

The capture runs the worker described by the supplied served-model manifest in a
separate process.  It keeps request and token content only in memory and emits
aggregate facts from the worker JSONL boundary, load-time operator resolutions,
and a bounded R9700 VRAM observer.  The active service is never restarted or
reconfigured by this tool.
"""

from __future__ import annotations

import argparse
import codecs
from collections import deque
import hashlib
import json
import math
import os
import re
import secrets
import select
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_BYTES = 4 * 1024 * 1024
SAFE_INT = 9_007_199_254_740_991
LAYER_RE = re.compile(r"\.layers\.(\d+)(?:\.|$)")
SQ8_PROMOTION_REQUEST_ENV = "ULLM_SQ8_PROMOTION_EVIDENCE_REQUEST_ID"
SQ8_PROMOTION_REQUEST_ID_RE = re.compile(r"^sq8-promotion-[0-9a-f]{64}$")
SQ8_OVERLAY_IMPLEMENTATION_ID = "qwen35_aq4_sq8_linear_qkv_z_overlay_v1"
SQ8_OVERLAY_EXECUTION_PROFILE = "rdna4_aq4_resident_sq8_linear_qkv_z_overlay"
SQ8_TELEMETRY_BINDING_SCHEMA = (
    "ullm.qwen35_aq4.sq8_promotion_telemetry_binding.v1"
)
SQ8_TELEMETRY_HASH_ENCODING = "canonical_json_ascii_sort_keys_compact_v1"
WORKER_STDERR_SCHEMA_VERSION = "ullm.aq4_resident_worker_stderr.v2"
WORKER_LIFECYCLE_SCHEMA_VERSION = "ullm.aq4_resident_worker_lifecycle.v1"
WORKER_STDERR_HEAD_MAX_BYTES = 16 * 1024
WORKER_STDERR_TAIL_MAX_BYTES = 16 * 1024
WORKER_STDERR_PREVIEW_MAX_BYTES = (
    WORKER_STDERR_HEAD_MAX_BYTES + WORKER_STDERR_TAIL_MAX_BYTES
)
WORKER_STDERR_READ_CHUNK_BYTES = 64 * 1024
WORKER_STDERR_JSON_LINE_MAX_BYTES = 1024 * 1024
WORKER_STDERR_MAX_RECORDS = 512
WORKER_STDERR_RECORD_BYTES_MAX = 4 * 1024 * 1024
WORKER_LIFECYCLE_MAX_EVENTS = 64
WORKER_STDOUT_BUFFER_MAX_BYTES = 4 * 1024 * 1024
DEFAULT_READY_TIMEOUT_SECONDS = 900.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 240.0
WORKER_SHUTDOWN_TIMEOUT_SECONDS = 30.0
WORKER_STDERR_REDACTION_RE = re.compile(
    rb"(?i)(?:password|passwd|secret|credential|api[_-]?key|authorization|"
    rb"access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key|"
    rb"bearer\s+|(?<![A-Za-z0-9])token\s*[:=]|"
    rb"https?://[^/\s:@]+:[^@\s/]+@)"
)
WORKER_STDERR_REDACTED_LINE = b"<redacted sensitive diagnostic line>"


class CaptureError(ValueError):
    """A capture failure that can carry bounded worker stderr evidence."""

    def __init__(
        self,
        message: str,
        *,
        worker_stderr: dict[str, Any] | None = None,
        stage: str | None = None,
        timed_out: bool = False,
        worker_returncode: int | None = None,
        worker_signal: int | None = None,
        observed_sq8_promotion_telemetry: dict[str, Any] | None = None,
        observed_sq8_promotion_telemetry_binding: dict[str, Any] | None = None,
        worker_terminal: dict[str, Any] | None = None,
        request_id: str | None = None,
        timeouts: dict[str, float] | None = None,
        worker_lifecycle: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.worker_stderr = worker_stderr
        self.stage = stage
        self.timed_out = timed_out
        self.worker_returncode = worker_returncode
        self.worker_signal = worker_signal
        self.observed_sq8_promotion_telemetry = observed_sq8_promotion_telemetry
        self.observed_sq8_promotion_telemetry_binding = (
            observed_sq8_promotion_telemetry_binding
        )
        self.worker_terminal = worker_terminal
        self.request_id = request_id
        self.timeouts = timeouts
        self.worker_lifecycle = worker_lifecycle


class WorkerStderrCollector:
    """Stream worker stderr while preserving only bounded derived evidence.

    Raw bytes are never retained.  JSON objects are retained only for the
    existing load/request-audit consumers and are capped independently from
    the raw stream.  A complete line is either emitted to the preview or
    omitted; this prevents a secret marker after the preview boundary from
    leaking a prefix of that line.
    """

    def __init__(self, stream: Any) -> None:
        self.stream = stream
        self.records: list[dict[str, Any]] = []
        self._records_bytes = 0
        self._records_truncated = False
        self._record_count = 0
        self._schema_counts: dict[str, int] = {}
        self._other_schema_count = 0
        self._last_complete_record: dict[str, Any] | None = None
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._head = bytearray()
        self._head_closed = False
        self._tail: deque[bytes] = deque()
        self._tail_bytes = 0
        self._tail_truncated = False
        self._utf8_decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._utf8_decoder_broken = False
        self._utf8_replacement = False
        self._redacted_lines = 0
        self._line = bytearray()
        self._line_size = 0
        self._line_oversized = False
        self._line_sensitive = False
        self._marker_tail = b""
        self._finished = False
        self.stream_error: str | None = None

    @property
    def byte_count(self) -> int:
        return self._byte_count

    def _observe_utf8(self, chunk: bytes) -> None:
        # A strict incremental decoder distinguishes malformed input from a
        # legitimate U+FFFD character.  Once malformed input is found, switch
        # to replacement mode so later chunks continue to be drained safely.
        if self._utf8_decoder_broken:
            self._utf8_decoder.decode(chunk, final=False)
            return
        try:
            self._utf8_decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            self._utf8_replacement = True
            self._utf8_decoder_broken = True
            self._utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._utf8_decoder.decode(chunk, final=False)

    def _observe_marker(self, fragment: bytes) -> None:
        window = self._marker_tail + fragment
        if WORKER_STDERR_REDACTION_RE.search(window) is not None:
            self._line_sensitive = True
        self._marker_tail = window[-256:]

    def _feed_fragment(self, fragment: bytes) -> None:
        if not fragment:
            return
        self._line_size += len(fragment)
        self._observe_marker(fragment)
        if self._line_size > WORKER_STDERR_JSON_LINE_MAX_BYTES:
            self._line_oversized = True
            return
        self._line.extend(fragment)

    def _append_preview(self, value: bytes) -> None:
        if not value:
            return
        text = value.decode("utf-8", errors="replace")
        encoded = text.encode("utf-8")
        if not self._head_closed:
            remaining = WORKER_STDERR_HEAD_MAX_BYTES - len(self._head)
            if len(encoded) <= remaining:
                self._head.extend(encoded)
            else:
                self._head_closed = True
        if len(encoded) > WORKER_STDERR_TAIL_MAX_BYTES:
            self._tail_truncated = True
            return
        while self._tail and self._tail_bytes + len(encoded) > WORKER_STDERR_TAIL_MAX_BYTES:
            self._tail_bytes -= len(self._tail.popleft())
            self._tail_truncated = True
        self._tail.append(encoded)
        self._tail_bytes += len(encoded)

    @staticmethod
    def _bounded_last_record(value: dict[str, Any]) -> dict[str, Any] | None:
        if value.get("schema_version") == "ullm.backend_operation.load.v1":
            trace = value.get("trace")
            if not isinstance(trace, dict):
                return None
            return {
                "schema_version": "ullm.backend_operation.load.v1",
                "layer_position": value.get("layer_position")
                if type(value.get("layer_position")) is int
                else None,
                "trace": {
                    key: trace.get(key)
                    for key in (
                        "implementation_id",
                        "kind",
                        "phase",
                        "executable",
                        "batch_width",
                        "chunk_width",
                    )
                    if isinstance(trace.get(key), (str, int))
                    and not isinstance(trace.get(key), bool)
                },
            }
        if value.get("event") == "request_released":
            request_id = value.get("request_id")
            return {
                "event": "request_released",
                "request_id": request_id
                if isinstance(request_id, str)
                and SQ8_PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is not None
                else None,
                "operation_execution_audit_observed": isinstance(
                    value.get("operation_execution_audit"), dict
                ),
                "request_execution_audit_observed": isinstance(
                    value.get("request_execution_audit"), dict
                ),
            }
        return None

    def _observe_record(self, line: bytes, value: dict[str, Any]) -> None:
        self._record_count += 1
        schema = value.get("schema_version")
        if isinstance(schema, str) and len(schema.encode("utf-8")) <= 128:
            if schema in self._schema_counts or len(self._schema_counts) < 32:
                self._schema_counts[schema] = self._schema_counts.get(schema, 0) + 1
            else:
                self._other_schema_count += 1
        else:
            self._other_schema_count += 1
        bounded = self._bounded_last_record(value)
        if bounded is not None:
            self._last_complete_record = bounded
        if (
            len(self.records) < WORKER_STDERR_MAX_RECORDS
            and self._records_bytes + len(line) <= WORKER_STDERR_RECORD_BYTES_MAX
        ):
            self.records.append(value)
            self._records_bytes += len(line)
        else:
            self._records_truncated = True

    def _finish_line(self, *, newline: bool) -> None:
        line = bytes(self._line) if not self._line_oversized else None
        if self._line_sensitive:
            self._redacted_lines += 1
            self._append_preview(WORKER_STDERR_REDACTED_LINE + (b"\n" if newline else b""))
        elif line is None:
            # A giant non-sensitive line cannot be retained or partially
            # displayed.  Mark the display as lossy and continue draining.
            self._head_closed = True
            self._tail_truncated = True
        else:
            self._append_preview(line + (b"\n" if newline else b""))

        if line is not None and not self._line_sensitive:
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict):
                self._observe_record(line, value)

        self._line.clear()
        self._line_size = 0
        self._line_oversized = False
        self._line_sensitive = False
        self._marker_tail = b""

    def _feed(self, chunk: bytes) -> None:
        self._digest.update(chunk)
        self._byte_count += len(chunk)
        self._observe_utf8(chunk)
        offset = 0
        while offset < len(chunk):
            newline_index = chunk.find(b"\n", offset)
            if newline_index < 0:
                self._feed_fragment(chunk[offset:])
                return
            self._feed_fragment(chunk[offset:newline_index])
            self._finish_line(newline=True)
            offset = newline_index + 1

    def drain(self) -> None:
        try:
            while True:
                chunk = self.stream.read(WORKER_STDERR_READ_CHUNK_BYTES)
                if not chunk:
                    break
                if not isinstance(chunk, bytes):
                    chunk = bytes(chunk)
                self._feed(chunk)
            if self._line_size:
                self._finish_line(newline=False)
            # Flush a pending incomplete UTF-8 sequence at EOF.
            try:
                self._utf8_decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                self._utf8_replacement = True
        except BaseException as error:
            # Keep the failure marker bounded and free of arbitrary worker
            # diagnostics.  The exact raw stream digest remains available, but
            # it is never claimed complete after a drain failure.
            self.stream_error = type(error).__name__
        finally:
            self._finished = True

    def mark_incomplete(self, reason: str = "drain_thread_timeout") -> None:
        """Mark a drain that did not finish before the cleanup deadline."""

        if self.stream_error is None:
            self.stream_error = reason[:1024]
        self._finished = False

    def summary(self) -> dict[str, Any]:
        if not self._finished:
            # This is only used as a last-resort failure envelope if a stream
            # cannot be joined.  It remains structurally valid and secret-free.
            self._head_closed = True
            self._tail_truncated = True
        head_text = bytes(self._head).decode("utf-8", errors="replace")
        tail_raw = b"".join(self._tail)
        tail_text = tail_raw.decode("utf-8", errors="replace")
        return {
            "schema_version": WORKER_STDERR_SCHEMA_VERSION,
            "byte_count": self._byte_count,
            "sha256": self._digest.hexdigest(),
            "head_text": head_text,
            "head_bytes": len(self._head),
            "tail_text": tail_text,
            "tail_bytes": len(tail_raw),
            "truncated": self._head_closed or self._tail_truncated,
            "utf8_replacement": self._utf8_replacement,
            "redacted_lines": self._redacted_lines,
            "record_count": self._record_count,
            "records_retained": len(self.records),
            "records_truncated": self._records_truncated,
            "schema_counts": {
                **dict(sorted(self._schema_counts.items())),
                **({"<other>": self._other_schema_count} if self._other_schema_count else {}),
            },
            "last_complete_record": self._last_complete_record,
            "complete": self._finished and self.stream_error is None,
            "stream_error": self.stream_error,
        }


class WorkerLifecycleEvidence:
    """Keep bounded, content-free worker stdout lifecycle evidence."""

    _ALLOWED_TYPES = frozenset(
        {"ready", "started", "progress", "token", "released", "error", "fatal"}
    )

    def __init__(self, request_id: str, started_at: float) -> None:
        self.request_id = request_id
        self.started_at = started_at
        self.request_sent_offset_ms: float | None = None
        self.event_count = 0
        self.events_truncated = False
        self.events: list[dict[str, Any]] = []
        self.last_event: dict[str, Any] | None = None

    def mark_request_sent(self, observed_at: float) -> None:
        self.request_sent_offset_ms = self._offset(observed_at)

    def _offset(self, observed_at: float) -> float:
        return round(max(0.0, observed_at - self.started_at) * 1000.0, 3)

    def observe(self, value: dict[str, Any], observed_at: float) -> None:
        raw_type = value.get("type")
        event_type = raw_type if raw_type in self._ALLOWED_TYPES else "unknown"
        request_id = value.get("request_id")
        summary: dict[str, Any] = {
            "type": event_type,
            "offset_ms": self._offset(observed_at),
            "request_id_matches": (
                request_id == self.request_id if isinstance(request_id, str) else None
            ),
        }
        for source, target in (
            ("processed_prompt_tokens", "processed_prompt_tokens"),
            ("completion_tokens", "completion_tokens"),
            ("index", "token_index"),
        ):
            item = value.get(source)
            summary[target] = (
                item
                if type(item) is int and 0 <= item <= SAFE_INT
                else None
            )
        self.event_count += 1
        self.last_event = summary
        if len(self.events) < WORKER_LIFECYCLE_MAX_EVENTS:
            self.events.append(summary)
        else:
            self.events_truncated = True

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": WORKER_LIFECYCLE_SCHEMA_VERSION,
            "request_id": self.request_id,
            "request_sent": self.request_sent_offset_ms is not None,
            "request_sent_offset_ms": self.request_sent_offset_ms,
            "event_count": self.event_count,
            "events_retained": len(self.events),
            "events_truncated": self.events_truncated,
            "events": list(self.events),
            "last_event": self.last_event,
        }


def _capture_timeouts(args: argparse.Namespace) -> dict[str, float]:
    return {
        "ready_seconds": float(args.ready_timeout),
        "request_seconds": float(args.timeout),
        "shutdown_seconds": WORKER_SHUTDOWN_TIMEOUT_SECONDS,
    }


def _strict_ready_event(
    event: Any, manifest: dict[str, Any], protocol: str
) -> bool:
    if not isinstance(event, dict) or event.get("type") != "ready":
        return False
    if set(event) != {
        "schema_version",
        "type",
        "model",
        "model_revision",
        "artifact_content_sha256",
        "package_manifest_sha256",
        "device",
        "execution_profile",
        "context_length",
        "max_new_tokens",
    }:
        return False
    if event.get("schema_version") != protocol or "request_id" in event:
        return False
    public = manifest.get("public")
    generation = manifest.get("generation")
    worker = manifest.get("worker")
    identity = worker.get("identity") if isinstance(worker, dict) else None
    product = manifest.get("product")
    package = product.get("package") if isinstance(product, dict) else None
    artifact = product.get("artifact") if isinstance(product, dict) else None
    expected = {
        "model": public.get("id") if isinstance(public, dict) else None,
        "model_revision": public.get("revision") if isinstance(public, dict) else None,
        "artifact_content_sha256": (
            artifact.get("content_sha256") if isinstance(artifact, dict) else None
        ),
        "package_manifest_sha256": (
            package.get("manifest_sha256") if isinstance(package, dict) else None
        ),
        "device": identity.get("device") if isinstance(identity, dict) else None,
        "execution_profile": (
            identity.get("execution_profile") if isinstance(identity, dict) else None
        ),
        "context_length": (
            public.get("context_length") if isinstance(public, dict) else None
        ),
        "max_new_tokens": (
            generation.get("max_completion_tokens")
            if isinstance(generation, dict)
            else None
        ),
    }
    if any(
        expected[key] is None
        for key in expected
        if key != "artifact_content_sha256"
    ):
        return False
    return all(key in event and event.get(key) == value for key, value in expected.items())


def token_identity_digest(token_ids: list[int]) -> str:
    digest = hashlib.sha256(b"ullm.sq8-promotion-output-token-ids.v1\0")
    for token_id in token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise CaptureError("output token identity contains an invalid token id")
        digest.update(token_id.to_bytes(8, "little"))
    return digest.hexdigest()


def validate_sq8_promotion_telemetry(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "projection",
        "diagnostic_host_staging",
    }:
        raise CaptureError("SQ8 promotion telemetry shape differs")
    if value.get("schema_version") != "ullm.qwen35_aq4.sq8_promotion_telemetry.v1":
        raise CaptureError("SQ8 promotion telemetry schema differs")
    projection = value.get("projection")
    expected_projection_keys = {
        "single_matvec_count",
        "batch_matvec_count",
        "pair_matvec_count",
        "triple_matvec_count",
        "fallback_count",
    }
    if not isinstance(projection, dict) or set(projection) != expected_projection_keys:
        raise CaptureError("SQ8 projection telemetry shape differs")
    if any(
        not isinstance(projection[key], int)
        or isinstance(projection[key], bool)
        or projection[key] < 0
        for key in expected_projection_keys
    ):
        raise CaptureError("SQ8 projection telemetry counts are invalid")
    if projection["batch_matvec_count"] <= 0 or projection["pair_matvec_count"] <= 0:
        raise CaptureError("SQ8 batch and pair projection evidence is required")
    for key in ("single_matvec_count", "triple_matvec_count", "fallback_count"):
        if projection[key] != 0:
            raise CaptureError(f"SQ8 promotion requires zero {key}")
    staging = value.get("diagnostic_host_staging")
    staging_keys = {"read_count", "write_count", "read_bytes", "write_bytes"}
    if not isinstance(staging, dict) or set(staging) != staging_keys:
        raise CaptureError("SQ8 host-staging telemetry shape differs")
    if any(staging[key] != 0 for key in staging_keys):
        raise CaptureError("SQ8 promotion requires zero diagnostic host staging")
    return value


def diagnostic_sq8_promotion_telemetry(value: Any) -> dict[str, Any] | None:
    """Preserve the raw telemetry object without accepting it as evidence."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "projection",
        "diagnostic_host_staging",
    }:
        return None
    projection = value.get("projection")
    projection_keys = {
        "single_matvec_count",
        "batch_matvec_count",
        "pair_matvec_count",
        "triple_matvec_count",
        "fallback_count",
    }
    staging = value.get("diagnostic_host_staging")
    staging_keys = {"read_count", "write_count", "read_bytes", "write_bytes"}
    if (
        value.get("schema_version")
        != "ullm.qwen35_aq4.sq8_promotion_telemetry.v1"
        or not isinstance(projection, dict)
        or set(projection) != projection_keys
        or any(type(projection[key]) is not int or projection[key] < 0 for key in projection_keys)
        or not isinstance(staging, dict)
        or set(staging) != staging_keys
        or any(type(staging[key]) is not int or staging[key] < 0 for key in staging_keys)
    ):
        return None
    return value


def sq8_promotion_telemetry_binding(
    telemetry: dict[str, Any], request_id: str
) -> dict[str, Any]:
    if SQ8_PROMOTION_REQUEST_ID_RE.fullmatch(request_id) is None:
        raise CaptureError("SQ8 telemetry binding request ID differs")
    try:
        raw = json.dumps(
            telemetry,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise CaptureError(f"cannot bind SQ8 telemetry: {error}") from error
    return {
        "schema_version": SQ8_TELEMETRY_BINDING_SCHEMA,
        "request_id": request_id,
        "hash_encoding": SQ8_TELEMETRY_HASH_ENCODING,
        "telemetry_sha256": hashlib.sha256(raw).hexdigest(),
    }


def configure_sq8_promotion_environment(
    environment: dict[str, str], *, enabled: bool, request_id: str
) -> dict[str, str]:
    result = dict(environment)
    result.pop(SQ8_PROMOTION_REQUEST_ENV, None)
    if enabled:
        result["HIP_VISIBLE_DEVICES"] = "1"
        result["ULLM_HIP_VISIBLE_DEVICES"] = "1"
        result.pop("ROCR_VISIBLE_DEVICES", None)
        result[SQ8_PROMOTION_REQUEST_ENV] = request_id
    return result


def resolve_capture_request_id(*, sq8_promotion: bool, promotion_request_id: str | None) -> str:
    if sq8_promotion:
        if promotion_request_id is None or SQ8_PROMOTION_REQUEST_ID_RE.fullmatch(promotion_request_id) is None:
            raise CaptureError("SQ8 promotion requires a fixed cryptographic request ID")
        return promotion_request_id
    if promotion_request_id is not None:
        raise CaptureError("SQ8 promotion request ID is forbidden outside promotion capture")
    return "executor-" + secrets.token_hex(16)


def load_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"{label} must be a regular non-symlink file")
    if path.stat().st_size > MAX_BYTES:
        raise CaptureError(f"{label} exceeds the 4 MiB bound")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot parse {label}: {error}") from error


def package_manifest_path(manifest: dict[str, Any]) -> Path:
    product = manifest.get("product", {})
    package = product.get("package", {})
    manifest_path = Path(str(manifest.get("_capture_manifest_path", ""))).resolve() if manifest.get("_capture_manifest_path") else Path.cwd() / "manifest.json"
    root_raw = Path(str(product.get("root", ".")))
    if ".." in root_raw.parts or ".." in Path(str(package.get("manifest_path", ""))).parts:
        raise CaptureError("package path escapes manifest root")
    root = (root_raw if root_raw.is_absolute() else manifest_path.parent / root_raw).resolve()
    path = (Path(str(package.get("manifest_path", ""))) if Path(str(package.get("manifest_path", ""))).is_absolute() else root / str(package.get("manifest_path", ""))).resolve()
    cursor = path
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise CaptureError("package path contains symlink component")
        cursor = cursor.parent
    if not path.is_file() or path.is_symlink():
        raise CaptureError(f"package manifest is unavailable: {path}")
    return path


def copy_worker_environment(manifest: dict[str, Any]) -> dict[str, str]:
    """Copy only the runtime guard/device variables from the active worker."""
    result = os.environ.copy()
    binary = str(manifest.get("worker", {}).get("binary", ""))
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if binary not in cmdline:
                continue
            for raw in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" not in raw:
                    continue
                key, value = raw.split(b"=", 1)
                name = key.decode("utf-8", "ignore")
                if name.startswith("ULLM_REQUIRE_") or name in {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"}:
                    result[name] = value.decode("utf-8", "ignore")
            break
        except (OSError, UnicodeError):
            continue
    for name in manifest.get("worker", {}).get("required_environment", []):
        result.setdefault(name, "1")
    return result


def rocm_json(*args: str) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            ["rocm-smi", *args, "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if completed.returncode != 0:
            return None
        value = json.loads(completed.stdout)
        return value if isinstance(value, dict) else None
    except (OSError, subprocess.SubprocessError, UnicodeError, json.JSONDecodeError):
        return None


def target_card(device_architecture: str) -> tuple[str, int] | None:
    products = rocm_json("--showproductname", "--showuniqueid")
    if products is None:
        return None
    matches = []
    for card, value in products.items():
        if isinstance(value, dict) and value.get("GFX Version") == device_architecture:
            matches.append(card)
    if len(matches) != 1 or not matches[0].startswith("card"):
        return None
    return matches[0], int(matches[0][4:])


class VramObserver:
    def __init__(self, architecture: str) -> None:
        self.card = target_card(architecture)
        self.samples = 0
        self.peak: int | None = None
        self.capacity: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        if self.card is None:
            return
        values = rocm_json("--showmeminfo", "vram")
        if values is None:
            return
        value = values.get(self.card[0])
        if not isinstance(value, dict):
            return
        try:
            used = int(value["VRAM Total Used Memory (B)"])
            capacity = int(value["VRAM Total Memory (B)"])
        except (KeyError, TypeError, ValueError):
            return
        if used < 0 or capacity <= 0:
            return
        self.samples += 1
        self.capacity = capacity
        self.peak = used if self.peak is None else max(self.peak, used)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.025)

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, name="aq4-vram-observer", daemon=True)
        self._thread.start()

    def finish(self) -> dict[str, Any]:
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        self._sample()
        return {
            "kind": "rocm_smi_vram_target_card",
            "sample_count": self.samples,
            "complete": self.card is not None and self.samples >= 2 and self.peak is not None,
            "capacity_bytes": self.capacity,
            "peak_bytes": self.peak,
            "target_card": self.card[0] if self.card is not None else None,
        }


def layer_graph(package: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    layers: dict[int, dict[str, Any]] = {}
    for group in (package.get("tensors", []), package.get("passthrough_tensors", [])):
        for tensor in group:
            if not isinstance(tensor, dict):
                continue
            match = LAYER_RE.search(str(tensor.get("name", "")))
            if match is None:
                continue
            index = int(match.group(1))
            item = layers.setdefault(index, {"layer_index": index, "tensor_count": 0, "kinds": set()})
            item["tensor_count"] += 1
            name = str(tensor.get("name", ""))
            item["kinds"].add("linear_attention" if ".linear_attn." in name else "self_attention" if ".self_attn." in name else "other")
    ordered = []
    for index in sorted(layers):
        item = layers[index]
        kinds = sorted(item.pop("kinds"))
        item["kind"] = kinds[0] if len(kinds) == 1 else kinds
        ordered.append(item)
    embedding = next((x for x in package.get("passthrough_tensors", []) if x.get("name") == "model.language_model.embed_tokens.weight"), {})
    shape = embedding.get("shape", [0, 0])
    context = int(manifest.get("public", {}).get("context_length", 0))
    block_size = 256
    return {
        "model_graph": {
            "schema_id": "ullm.model_graph.v0.1",
            "schema_version": "0.1",
            "source": "adapter_derived",
            "canonical": {
                "model_id": manifest.get("public", {}).get("id"),
                "format_id": manifest.get("format", {}).get("format_id"),
                "vocab_size": shape[0] if len(shape) > 0 else 0,
                "hidden_size": shape[1] if len(shape) > 1 else 0,
                "context_length": context,
                "block_size": block_size,
                "cache_blocks": math.ceil(context / block_size) if context else 0,
                "layers": ordered,
            "terminal_components": ["embedding", "decoder_stack", "final_norm", "lm_head", "sampling"],
            },
        },
        "state_schema": {
            "schema_id": "ullm.state_schema.v0.1",
            "schema_version": "0.1",
            "source": "adapter_derived",
            "canonical": {
                "request_state": ["recurrent_state", "paged_kv", "decode_position", "sampling_state"],
                "transaction": ["prepare", "publish", "commit", "discard", "reset"],
                "reset_scope": "request_owned_state_only",
                "resident_weights_reloaded_per_request": False,
            },
        },
        "compatibility_inputs": {
            "backend": "hip",
            "format_id": manifest.get("format", {}).get("format_id"),
            "layout": "row_major_grouped",
        },
    }


def phase_name(value: str) -> str:
    return {"ColdPrefill": "cold_prefill", "CachedPrefixPrefill": "cached_prefix_prefill", "Decode": "decode"}.get(value, value.lower())


def operator_records(load_records: list[dict[str, Any]], audit: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    operators = []
    fallback_events = []
    implementation_counts = {
        str(item.get("implementation_id")): int(item.get("count", 0))
        for item in audit.get("implementation_counts", [])
        if isinstance(item, dict)
    }
    assigned_implementations: set[str] = set()
    for record in load_records:
        trace = record.get("trace")
        if not isinstance(trace, dict):
            continue
        resolution = str(trace.get("resolution", ""))
        implementation = str(trace.get("implementation_id", ""))
        invocation_count = implementation_counts.get(implementation, 0) if implementation not in assigned_implementations else 0
        assigned_implementations.add(implementation)
        phase = phase_name(str(trace.get("phase", "")))
        op_kind = str(trace.get("kind", "unknown"))
        implementation_id = str(trace.get("implementation_id", "unknown"))
        device_name = str(trace.get("device_name") or "unknown-device")
        item = {
            "phase_kind": phase,
            "operator_instance_id": f"layer-{record.get('layer_position', 'unknown')}-{op_kind}-{phase}",
            "op_kind": op_kind,
            "implementation_id": implementation_id,
            "implementation_version": str(trace.get("semantic_version") or trace.get("runtime_build") or "1"),
            "resolution_status": "selected" if resolution == "Primary" else "fallback",
            "backend": str(trace.get("backend") or "unknown").lower(),
            "device": device_name,
            "formats": {
                "weight": trace.get("weight_format") or "AQ4_0",
                "activation": trace.get("activation_format") or "F32",
                "state": trace.get("state_format"),
                "layout": str(trace.get("layout") or "row_major_grouped"),
            },
            "shape_bucket": {
                "id": f"{op_kind}-{trace.get('batch_width', 1)}x{trace.get('chunk_width', 1)}",
                "dimensions": [
                    {"name": "batch", "value": int(trace.get("batch_width") or 1)},
                    {"name": "chunk", "value": int(trace.get("chunk_width") or 1)},
                ],
            },
            "selection_reason": {
                "kind": "exact_match" if resolution == "Primary" else "generic_fallback",
                "candidate_count": 1,
                "score": 1 if resolution == "Primary" else 0,
                "priority": 0,
                "matched_constraints": ["format", "gpu_arch"],
            },
            "architecture_constraint": {
                "model_arch": "Qwen3.5",
                "gpu_arch": str(trace.get("architecture") or "unknown"),
                "gpu_name": device_name,
            },
            "workspace": {
                "planned_bytes": int(trace.get("persistent_bytes", 0)) + int(trace.get("temporary_bytes", 0)),
                "temporary_bytes": int(trace.get("temporary_bytes", 0)),
                "observed_peak_bytes": None,
            },
            "invocation_count": invocation_count or 1,
        }
        if invocation_count > 0:
            item["resolution_status"] = "selected"
        operators.append(item)
        if resolution.startswith("Fallback"):
            fallback_events.append({"phase_kind": phase, "op_kind": op_kind, "from_implementation_id": str(trace.get("fallback_from_implementation_id") or "generic"), "to_implementation_id": implementation_id, "reason_code": "backend_resolution_fallback", "classification": "expected"})
    # Load-time traces describe the M1 contract.  The terminal request audit is the authority
    # for the implementation that actually ran for this request (for example the M128 chunk
    # implementations).  Preserve the bounded load contract above and append one aggregate
    # request-terminal entry for every implementation observed by the audit.
    for audited in audit.get("implementation_counts", []):
        if not isinstance(audited, dict) or int(audited.get("count", 0)) <= 0:
            continue
        implementation = str(audited.get("implementation_id", ""))
        if any(item.get("implementation_id") == implementation and item.get("invocation_count", 0) > 0 for item in operators):
            continue
        template = next((item for item in operators if item.get("op_kind") == audited.get("kind")), None)
        if template is None:
            continue
        item = json.loads(json.dumps(template))
        item["operator_instance_id"] = f"request-terminal-{implementation}"
        item["phase_kind"] = "decode" if ".m1" in implementation else "cold_prefill"
        item["implementation_id"] = implementation
        item["resolution_status"] = "selected"
        item["invocation_count"] = int(audited["count"])
        operators.append(item)
    return operators, fallback_events


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CaptureError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.incomplete")
    raw = (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_BYTES:
        raise CaptureError("executor record exceeds the 4 MiB bound")
    with temporary.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def _terminate_worker(proc: subprocess.Popen[bytes]) -> None:
    """Terminate a worker, escalate to kill, and reap it."""

    if proc.poll() is None:
        try:
            proc.terminate()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (OSError, ProcessLookupError):
                pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise CaptureError(
                "resident worker could not be reaped within the cleanup bound",
                stage="cleanup",
            ) from error


def _finish_worker_stderr(
    proc: subprocess.Popen[bytes],
    stderr_thread: threading.Thread,
    collector: WorkerStderrCollector,
) -> dict[str, Any]:
    """Join the drain thread, failing closed if EOF cannot be collected."""

    stderr_thread.join(timeout=3)
    if stderr_thread.is_alive():
        # This handles a descendant that inherited the pipe or a test double
        # whose read end does not observe EOF after the parent exits.
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except (OSError, ValueError):
            pass
        stderr_thread.join(timeout=1)
    if stderr_thread.is_alive():
        collector.mark_incomplete()
    return collector.summary()


def _finish_worker_stdout(proc: subprocess.Popen[bytes]) -> bool:
    """Drain protocol bytes left after the terminal event without retaining them."""

    stream = proc.stdout
    if stream is None:
        return True
    finished = threading.Event()
    stream_error: list[str] = []

    def drain() -> None:
        try:
            fd = stream.fileno()
            deadline = time.monotonic() + 3
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stream_error.append("drain_thread_timeout")
                    return
                ready, _, _ = select.select([stream], [], [], remaining)
                if not ready:
                    stream_error.append("drain_thread_timeout")
                    return
                try:
                    chunk = os.read(fd, 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    return
        except BaseException as error:
            stream_error.append(type(error).__name__)
        finally:
            finished.set()

    thread = threading.Thread(target=drain, name="aq4-stdout-drain", daemon=True)
    thread.start()
    thread.join(timeout=3)
    if thread.is_alive():
        try:
            stream.close()
        except (OSError, ValueError):
            pass
        thread.join(timeout=1)
    return finished.is_set() and not thread.is_alive() and not stream_error


def _empty_worker_stderr() -> dict[str, Any]:
    """Return the structurally valid evidence for a worker never started."""

    return {
        "schema_version": WORKER_STDERR_SCHEMA_VERSION,
        "byte_count": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "head_text": "",
        "head_bytes": 0,
        "tail_text": "",
        "tail_bytes": 0,
        "truncated": False,
        "utf8_replacement": False,
        "redacted_lines": 0,
        "record_count": 0,
        "records_retained": 0,
        "records_truncated": False,
        "schema_counts": {},
        "last_complete_record": None,
        "complete": False,
        "stream_error": "worker_not_started",
    }


def _normalize_worker_stderr(value: Any) -> dict[str, Any]:
    """Keep the public failure envelope exact even for injected failures."""

    if not isinstance(value, dict):
        return _empty_worker_stderr()
    def bounded_text(name: str, limit: int) -> tuple[str, int, bool]:
        text = value.get(name)
        if not isinstance(text, str):
            text = ""
        original = text.encode("utf-8", errors="replace")
        bounded = original[:limit]
        return bounded.decode("utf-8", errors="replace"), len(bounded), len(bounded) < len(original)

    head, head_bytes, head_cut = bounded_text("head_text", WORKER_STDERR_HEAD_MAX_BYTES)
    tail, tail_bytes, tail_cut = bounded_text("tail_text", WORKER_STDERR_TAIL_MAX_BYTES)
    byte_count = value.get("byte_count")
    if type(byte_count) is not int or byte_count < 0:
        byte_count = 0
    digest = value.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        digest = hashlib.sha256(b"").hexdigest()
    stream_error = value.get("stream_error")
    if stream_error is not None:
        stream_error = str(stream_error).encode("utf-8", errors="replace")[:1024].decode(
            "utf-8", errors="replace"
        ) or "worker_stderr_incomplete"
    complete = value.get("complete") is True and stream_error is None
    if not complete and stream_error is None:
        stream_error = "worker_stderr_incomplete"
    record_count = value.get("record_count")
    if type(record_count) is not int or not 0 <= record_count <= SAFE_INT:
        record_count = 0
    records_retained = value.get("records_retained")
    if (
        type(records_retained) is not int
        or not 0 <= records_retained <= min(record_count, WORKER_STDERR_MAX_RECORDS)
    ):
        records_retained = 0
    raw_counts = value.get("schema_counts")
    schema_counts: dict[str, int] = {}
    if isinstance(raw_counts, dict) and len(raw_counts) <= 33:
        for key, count in raw_counts.items():
            if (
                isinstance(key, str)
                and len(key.encode("utf-8")) <= 128
                and type(count) is int
                and 0 <= count <= record_count
            ):
                schema_counts[key] = count
    raw_last = value.get("last_complete_record")
    last = (
        WorkerStderrCollector._bounded_last_record(raw_last)
        if isinstance(raw_last, dict)
        else None
    )
    return {
        "schema_version": WORKER_STDERR_SCHEMA_VERSION,
        "byte_count": byte_count,
        "sha256": digest,
        "head_text": head,
        "head_bytes": head_bytes,
        "tail_text": tail,
        "tail_bytes": tail_bytes,
        "truncated": value.get("truncated") is True or head_cut or tail_cut,
        "utf8_replacement": value.get("utf8_replacement") is True,
        "redacted_lines": value.get("redacted_lines") if type(value.get("redacted_lines")) is int and value.get("redacted_lines") >= 0 else 0,
        "record_count": record_count,
        "records_retained": records_retained,
        "records_truncated": value.get("records_truncated") is True,
        "schema_counts": dict(sorted(schema_counts.items())),
        "last_complete_record": last,
        "complete": complete,
        "stream_error": stream_error,
    }


def _normalize_worker_lifecycle(
    value: Any, request_id: str | None
) -> dict[str, Any]:
    def event_summary(item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict) or item.get("type") not in (
            WorkerLifecycleEvidence._ALLOWED_TYPES | {"unknown"}
        ):
            return None
        offset = item.get("offset_ms")
        if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(float(offset)) or not 0 <= float(offset) <= 86_400_000:
            return None
        result: dict[str, Any] = {
            "type": item["type"],
            "offset_ms": float(offset),
            "request_id_matches": item.get("request_id_matches")
            if item.get("request_id_matches") in {True, False, None}
            else None,
        }
        for key in ("processed_prompt_tokens", "completion_tokens", "token_index"):
            number = item.get(key)
            result[key] = number if type(number) is int and 0 <= number <= SAFE_INT else None
        return result

    empty = {
        "schema_version": WORKER_LIFECYCLE_SCHEMA_VERSION,
        "request_id": request_id,
        "request_sent": False,
        "request_sent_offset_ms": None,
        "event_count": 0,
        "events_retained": 0,
        "events_truncated": False,
        "events": [],
        "last_event": None,
    }
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != WORKER_LIFECYCLE_SCHEMA_VERSION
        or value.get("request_id") != request_id
    ):
        return empty
    raw_events = value.get("events")
    if not isinstance(raw_events, list) or len(raw_events) > WORKER_LIFECYCLE_MAX_EVENTS:
        return empty
    events = [event_summary(item) for item in raw_events]
    if any(item is None for item in events):
        return empty
    event_count = value.get("event_count")
    if type(event_count) is not int or not len(events) <= event_count <= SAFE_INT:
        return empty
    sent_offset = value.get("request_sent_offset_ms")
    if sent_offset is not None and (
        not isinstance(sent_offset, (int, float))
        or isinstance(sent_offset, bool)
        or not math.isfinite(float(sent_offset))
        or not 0 <= float(sent_offset) <= 86_400_000
    ):
        return empty
    last = event_summary(value.get("last_event")) if value.get("last_event") is not None else None
    return {
        "schema_version": WORKER_LIFECYCLE_SCHEMA_VERSION,
        "request_id": request_id,
        "request_sent": sent_offset is not None,
        "request_sent_offset_ms": float(sent_offset) if sent_offset is not None else None,
        "event_count": event_count,
        "events_retained": len(events),
        "events_truncated": value.get("events_truncated") is True,
        "events": events,
        "last_event": last,
    }


def _bind_worker_terminal(error: CaptureError, proc: subprocess.Popen[bytes]) -> None:
    """Bind the reap result to a failure without losing signal identity."""

    returncode = proc.returncode
    error.worker_returncode = int(returncode) if isinstance(returncode, int) else None
    error.worker_signal = (
        -error.worker_returncode
        if isinstance(error.worker_returncode, int) and error.worker_returncode < 0
        else None
    )


def _next_worker_event(
    proc: subprocess.Popen[bytes],
    stdout_fd: int,
    stdout_buffer: bytearray,
    pending: deque[dict[str, Any]],
    *,
    deadline: float,
    timeout_stage: str,
    protocol_stage: str,
    timeout_reason: str,
) -> dict[str, Any]:
    while True:
        if pending:
            return pending.popleft()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CaptureError(timeout_reason, stage=timeout_stage, timed_out=True)
        ready, _, _ = select.select([stdout_fd], [], [], min(1.0, remaining))
        if not ready:
            if proc.poll() is not None:
                raise CaptureError(
                    "resident worker exited before the expected lifecycle event",
                    stage=protocol_stage,
                )
            continue
        try:
            chunk = os.read(stdout_fd, 64 * 1024)
        except BlockingIOError:
            continue
        if not chunk:
            raise CaptureError(
                "resident worker stdout ended before the expected lifecycle event",
                stage=protocol_stage,
            )
        stdout_buffer.extend(chunk)
        if len(stdout_buffer) > WORKER_STDOUT_BUFFER_MAX_BYTES:
            raise CaptureError(
                "resident worker stdout record exceeds the bounded lifecycle limit",
                stage=protocol_stage,
            )
        while b"\n" in stdout_buffer:
            newline_index = stdout_buffer.index(b"\n")
            line = bytes(stdout_buffer[:newline_index])
            del stdout_buffer[: newline_index + 1]
            try:
                event = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise CaptureError(
                    "resident worker emitted invalid lifecycle JSON",
                    stage=protocol_stage,
                ) from error
            if not isinstance(event, dict):
                raise CaptureError(
                    "resident worker emitted a non-object lifecycle event",
                    stage=protocol_stage,
                )
            pending.append(event)


def run_capture(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest, "served-model manifest")
    manifest["_capture_manifest_path"] = str(args.manifest.resolve())
    package_path = package_manifest_path(manifest)
    package = load_json(package_path, "package manifest")
    worker = manifest.get("worker", {})
    binary_raw = Path(str(worker.get("binary", "")))
    if ".." in binary_raw.parts:
        raise CaptureError("worker binary path escapes manifest root")
    binary = (binary_raw if binary_raw.is_absolute() else args.manifest.resolve().parent / binary_raw).resolve()
    cursor = binary
    while cursor != cursor.parent:
        if cursor.is_symlink():
            raise CaptureError("worker binary path contains symlink component")
        cursor = cursor.parent
    protocol = worker.get("protocol")
    if not binary.is_file() or protocol not in {"ullm.worker.v1", "ullm.worker.v2"}:
        raise CaptureError("served worker binary or protocol is invalid")
    command = [str(binary), *[str(manifest and args.manifest if value == "{manifest}" else value) for value in worker.get("arguments", [])]]
    environment = copy_worker_environment(manifest)
    prompt_tokens = args.prompt_tokens
    if not 1 <= prompt_tokens <= int(manifest.get("public", {}).get("context_length", 4096)):
        raise CaptureError("prompt token count is outside the served context")
    sq8_promotion = bool(getattr(args, "sq8_promotion_evidence", False))
    internal_request_id = resolve_capture_request_id(
        sq8_promotion=sq8_promotion,
        promotion_request_id=getattr(args, "sq8_promotion_request_id", None),
    )
    if sq8_promotion:
        if prompt_tokens != 128 or args.max_new_tokens != 2:
            raise CaptureError(
                "SQ8 promotion requires the fixed 128-token prefill and 2-token generation"
            )
        if manifest.get("format", {}).get("implementation_id") != SQ8_OVERLAY_IMPLEMENTATION_ID:
            raise CaptureError("SQ8 promotion manifest implementation identity differs")
        if worker.get("identity", {}).get("execution_profile") != SQ8_OVERLAY_EXECUTION_PROFILE:
            raise CaptureError("SQ8 promotion manifest execution profile differs")
    if (
        not math.isfinite(float(args.ready_timeout))
        or not math.isfinite(float(args.timeout))
        or not 0 < float(args.ready_timeout) <= 3600
        or not 0 < float(args.timeout) <= 3600
    ):
        raise CaptureError("capture timeout bounds are invalid")
    if sq8_promotion and (
        float(args.ready_timeout) != DEFAULT_READY_TIMEOUT_SECONDS
        or float(args.timeout) != DEFAULT_REQUEST_TIMEOUT_SECONDS
    ):
        raise CaptureError("SQ8 promotion timeout contract differs")
    timeouts = _capture_timeouts(args)
    environment = configure_sq8_promotion_environment(
        environment, enabled=sq8_promotion, request_id=internal_request_id
    )
    request = {
        "schema_version": protocol,
        "type": "generate",
        "request_id": internal_request_id,
        "prompt_token_ids": list(range(1, prompt_tokens + 1)),
        "max_new_tokens": args.max_new_tokens,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
        # Promotion evidence must execute one decode step even if the first
        # sampled token is ordinarily terminal for this model.
        "eos_token_ids": []
        if sq8_promotion
        else manifest.get("generation", {}).get("eos_token_ids", []),
    }
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    process_started = time.monotonic()
    lifecycle = WorkerLifecycleEvidence(internal_request_id, process_started)
    assert proc.stderr is not None
    stderr_collector = WorkerStderrCollector(proc.stderr)
    stderr_records: list[dict[str, Any]] = stderr_collector.records
    stderr_thread = threading.Thread(target=stderr_collector.drain, name="aq4-stderr-drain", daemon=True)
    stderr_thread.start()
    observer = VramObserver(str(worker.get("identity", {}).get("device", "gfx1201")))
    observer_data: dict[str, Any] | None = None
    stderr_summary: dict[str, Any] | None = None
    process_error: BaseException | None = None
    timed_out = False
    try:
        observer.start()
        assert proc.stdin is not None and proc.stdout is not None
        stdout_fd = proc.stdout.fileno()
        os.set_blocking(stdout_fd, False)
        stdout_buffer = bytearray()
        pending_events: deque[dict[str, Any]] = deque()
        ready_event = _next_worker_event(
            proc,
            stdout_fd,
            stdout_buffer,
            pending_events,
            deadline=process_started + float(args.ready_timeout),
            timeout_stage="ready_timeout",
            protocol_stage="ready_protocol",
            timeout_reason="resident worker ready timed out",
        )
        lifecycle.observe(ready_event, time.monotonic())
        if not _strict_ready_event(ready_event, manifest, str(protocol)):
            raise CaptureError(
                "resident worker ready identity differs",
                stage="ready_protocol",
            )
        proc.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode("ascii"))
        proc.stdin.flush()
        request_started = time.monotonic()
        lifecycle.mark_request_sent(request_started)
        output_token_ids: list[int] = []
        released: dict[str, Any] | None = None
        request_deadline = request_started + float(args.timeout)
        while released is None:
            event = _next_worker_event(
                proc,
                stdout_fd,
                stdout_buffer,
                pending_events,
                deadline=request_deadline,
                timeout_stage="request_timeout",
                protocol_stage="request_protocol",
                timeout_reason="resident worker request timed out",
            )
            lifecycle.observe(event, time.monotonic())
            event_type = event.get("type")
            if event.get("schema_version") != protocol or event_type not in {
                "started",
                "progress",
                "token",
                "released",
                "error",
                "fatal",
            }:
                raise CaptureError(
                    "resident worker request lifecycle differs",
                    stage="request_protocol",
                )
            if event_type not in {"fatal"} and event.get("request_id") != internal_request_id:
                raise CaptureError(
                    "resident worker request lifecycle identity differs",
                    stage="request_protocol",
                )
            if event_type == "token":
                if event.get("index") != len(output_token_ids):
                    raise CaptureError(
                        "resident worker token event identity is discontinuous",
                        stage="request_protocol",
                    )
                token_id = event.get("token_id")
                if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                    raise CaptureError(
                        "resident worker emitted an invalid token id",
                        stage="request_protocol",
                    )
                output_token_ids.append(token_id)
            if event_type == "released":
                released = event
        proc.stdin.write((json.dumps({"schema_version": protocol, "type": "shutdown"}, separators=(",", ":")) + "\n").encode("ascii"))
        proc.stdin.flush()
        proc.stdin.close()
        try:
            proc.wait(timeout=WORKER_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            raise CaptureError(
                "resident worker did not shut down after capture",
                stage="shutdown_timeout",
                timed_out=True,
            )
    except BaseException as error:
        process_error = error
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except (OSError, ValueError):
            pass
        if process_error is not None or proc.poll() is None:
            try:
                _terminate_worker(proc)
            except BaseException:
                if process_error is None:
                    process_error = CaptureError("resident worker could not be reaped", stage="cleanup")
        try:
            observer_data = observer.finish()
        except BaseException:
            if process_error is None:
                process_error = CaptureError("resident worker resource observation failed", stage="cleanup")
        stderr_summary = _finish_worker_stderr(proc, stderr_thread, stderr_collector)
        stdout_complete = _finish_worker_stdout(proc)
        if (
            not isinstance(stderr_summary, dict)
            or stderr_summary.get("complete") is not True
            or stderr_summary.get("stream_error") is not None
            or not stdout_complete
        ) and process_error is None:
            process_error = CaptureError(
                "worker pipe drain did not complete",
                stage="cleanup",
            )
        if isinstance(process_error, CaptureError):
            process_error.timed_out = process_error.timed_out or timed_out
            process_error.request_id = internal_request_id
            process_error.timeouts = timeouts
            process_error.worker_lifecycle = lifecycle.summary()
            _bind_worker_terminal(process_error, proc)
    if process_error is not None:
        if isinstance(process_error, CaptureError):
            process_error.worker_stderr = stderr_summary
            raise process_error
        failure = CaptureError(
            str(process_error),
            worker_stderr=stderr_summary,
            stage="worker",
            timed_out=timed_out,
            worker_returncode=int(proc.returncode)
            if isinstance(proc.returncode, int)
            else None,
            request_id=internal_request_id,
            timeouts=timeouts,
            worker_lifecycle=lifecycle.summary(),
        )
        if failure.worker_returncode is not None and failure.worker_returncode < 0:
            failure.worker_signal = -failure.worker_returncode
        raise failure from process_error
    assert observer_data is not None and stderr_summary is not None

    worker_terminal: dict[str, Any] | None = None
    observed_sq8_telemetry: dict[str, Any] | None = None
    observed_sq8_telemetry_binding: dict[str, Any] | None = None

    def worker_failure(
        message: str,
        stage: str = "validation",
    ) -> CaptureError:
        failure = CaptureError(
            message,
            worker_stderr=stderr_summary,
            stage=stage,
            timed_out=timed_out,
            worker_returncode=int(proc.returncode)
            if isinstance(proc.returncode, int)
            else None,
            observed_sq8_promotion_telemetry=observed_sq8_telemetry,
            observed_sq8_promotion_telemetry_binding=(
                observed_sq8_telemetry_binding
            ),
            worker_terminal=worker_terminal,
            request_id=internal_request_id,
            timeouts=timeouts,
            worker_lifecycle=lifecycle.summary(),
        )
        if failure.worker_returncode is not None and failure.worker_returncode < 0:
            failure.worker_signal = -failure.worker_returncode
        return failure

    backend = next(
        (
            item
            for item in reversed(stderr_records)
            if item.get("event") == "request_released"
        ),
        None,
    )
    request_audit = None
    if backend is not None:
        request_audit = backend.get("request_execution_audit")
        worker_terminal = {
            "schema_version": "ullm.aq4_resident_worker_terminal.v1",
            "event": "request_released",
            "request_id": backend.get("request_id"),
            "request_id_matches": backend.get("request_id") == internal_request_id,
            "operation_execution_audit_observed": isinstance(
                backend.get("operation_execution_audit"), dict
            ),
            "request_execution_audit_observed": isinstance(request_audit, dict),
        }
        if sq8_promotion and isinstance(request_audit, dict):
            observed_sq8_telemetry = diagnostic_sq8_promotion_telemetry(
                request_audit.get("sq8_promotion_telemetry")
            )
            if observed_sq8_telemetry is not None:
                observed_sq8_telemetry_binding = sq8_promotion_telemetry_binding(
                    observed_sq8_telemetry, internal_request_id
                )
    if proc.returncode != 0:
        raise worker_failure(
            f"resident worker exited with status {proc.returncode}",
            "worker_exit",
        )
    if backend is None:
        raise worker_failure("resident worker request audit was not observed", "audit_missing")
    if backend.get("request_id") != internal_request_id:
        raise worker_failure("resident worker request audit identity differs", "audit_missing")
    if not isinstance(request_audit, dict):
        raise worker_failure("resident worker request execution audit was not observed", "audit_missing")
    audit = backend.get("operation_execution_audit")
    if not isinstance(audit, dict):
        raise worker_failure(
            "resident worker operation execution audit was not observed",
            "audit_missing",
        )
    sq8_telemetry = None
    if sq8_promotion:
        try:
            sq8_telemetry = validate_sq8_promotion_telemetry(
                request_audit.get("sq8_promotion_telemetry")
            )
        except CaptureError as error:
            raise worker_failure(
                str(error),
                "telemetry_validation",
            ) from error
    load_records = [x for x in stderr_records if x.get("schema_version") == "ullm.backend_operation.load.v1"]
    operators, fallback_events = operator_records(load_records, audit)
    if len(operators) == 0 or audit.get("coverage_complete") is not True:
        raise worker_failure("full resident operator graph was not observed", "audit_missing")
    audited_counts = {
        str(item.get("implementation_id")): int(item.get("count", 0))
        for item in audit.get("implementation_counts", [])
        if isinstance(item, dict) and int(item.get("count", 0)) > 0
    }
    observed_counts: dict[str, int] = {}
    for item in operators:
        implementation = str(item.get("implementation_id", ""))
        observed_counts[implementation] = observed_counts.get(implementation, 0) + int(item.get("invocation_count", 0))
    if observed_counts != audited_counts:
        raise worker_failure("operator invocation counts do not reconcile with request audit", "audit_missing")
    timings = released.get("timings", {})
    width = max((index for index, count in enumerate(audit.get("prefill_width_histogram", [])) if index and count), default=None)
    if width is None:
        raise worker_failure("actual prefill execution width was not observed", "audit_missing")
    memory = {
        "vram_capacity_bytes": observer_data["capacity_bytes"],
        "resident_bytes": None,
        "persistent_state_bytes": None,
        "planned_temporary_bytes": None,
        "planned_total_bytes": None,
        "planned_headroom_bytes": None,
        "observed_peak_bytes": observer_data["peak_bytes"],
        "observed_headroom_bytes": None,
        "observer": {"kind": observer_data["kind"], "sample_count": observer_data["sample_count"], "complete": observer_data["complete"]},
        "oom": None,
    }
    # The load-time trace contains operator workspace, while the resident and persistent
    # allocations are derived from the package graph by the producer-side fixture contract.
    # Keep these facts explicit and fail closed if the package/runtime observer cannot provide them.
    if not observer_data["complete"] or observer_data["peak_bytes"] is None or observer_data["capacity_bytes"] is None:
        raise worker_failure("complete R9700 VRAM observation was not available", "resource_observation")
    completion_tokens = int(released.get("completion_tokens", 0))
    if completion_tokens != len(output_token_ids):
        raise worker_failure("resident worker output token identity count differs")
    phases = [
        {"phase_id": "cold-prefill-0", "kind": "cold_prefill", "executor_id": "generic_model_executor", "executor_version": "0.2.0", "prefill_mode": "cold", "chunk_width_tokens": prompt_tokens, "actual_token_batch_width": width, "actual_request_batch_width": 1, "request_count": 1, "input_token_count": prompt_tokens, "output_token_count": 0, "cached_prefix_token_count": 0, "context_tokens_before": 0, "context_tokens_after": prompt_tokens, "wall_time_ms": float(timings.get("prompt_ms", 0.0))},
        {"phase_id": "decode-0", "kind": "decode", "executor_id": "generic_model_executor", "executor_version": "0.2.0", "prefill_mode": None, "chunk_width_tokens": 1, "actual_token_batch_width": 1, "actual_request_batch_width": 1, "request_count": 1, "input_token_count": completion_tokens, "output_token_count": completion_tokens, "cached_prefix_token_count": 0, "context_tokens_before": prompt_tokens, "context_tokens_after": prompt_tokens + completion_tokens, "wall_time_ms": float(timings.get("predicted_ms", 0.001))},
    ]
    graph = layer_graph(package, manifest)
    trace_id = f"aq4-resident-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}"
    total_steps = int(audit.get("total_steps", 0))
    request_summary = {
        "fixture_id": "aq4-resident-executor-record-v1",
        "request_count": 1,
        "prompt_token_count": prompt_tokens,
        "cached_prefix_token_count": int(released.get("timings", {}).get("cache_n", 0)),
        "generated_token_count": completion_tokens,
        "context_tokens_at_decode_start": prompt_tokens,
        "prompt_or_token_content_recorded": False,
    }
    workspace_bytes = sum(int(x.get("workspace", {}).get("temporary_bytes", 0)) for x in operators)
    # Runtime workspace plans are public bounded facts; the package resident size and state
    # allocation are reconstructed from the same package metadata, never from prompt content.
    package_root = package_path.parent
    resident_bytes = sum(int(x.get("payload_bytes", 0)) for x in package.get("passthrough_tensors", []))
    codebooks: set[tuple[str, str]] = set()
    for tensor in package.get("tensors", []):
        if not isinstance(tensor, dict):
            continue
        for field in ("index_file", "scale_file"):
            path = package_root / str(tensor.get(field, ""))
            if not path.is_file() or path.is_symlink():
                raise worker_failure(f"package resident file is unavailable: {path}", "package_validation")
            resident_bytes += path.stat().st_size
        codebook = package_root / str(tensor.get("codebook_file", ""))
        if not codebook.is_file() or codebook.is_symlink():
            raise worker_failure(f"package codebook is unavailable: {codebook}", "package_validation")
        match = LAYER_RE.search(str(tensor.get("name", "")))
        component = f"layer-{match.group(1)}" if match else str(tensor.get("name", ""))
        key = (component, str(codebook))
        if key not in codebooks:
            codebooks.add(key)
            resident_bytes += codebook.stat().st_size
    persistent_bytes = 24 * 2_228_224 + 8 * 33_554_432
    if workspace_bytes <= 0:
        raise worker_failure("operator workspace observation was unavailable", "audit_missing")
    temporary_bytes = workspace_bytes
    planned_total = resident_bytes + persistent_bytes + temporary_bytes
    memory.update({
        "resident_bytes": resident_bytes,
        "persistent_state_bytes": persistent_bytes,
        "planned_temporary_bytes": temporary_bytes,
        "planned_total_bytes": planned_total,
        "planned_headroom_bytes": observer_data["capacity_bytes"] - planned_total,
        "observed_headroom_bytes": observer_data["capacity_bytes"] - observer_data["peak_bytes"],
    })
    result = {
        "schema_version": "ullm.production_executor_record.v1",
        "trace_id": trace_id,
        "scope": "full_model",
        "graph": graph,
        "executor": {
            "id": "generic_model_executor",
            "version": "0.2.0",
            "mode": "graph_lowered",
            "backend": "hip",
            "device": {
                "runtime_device_index": int(worker.get("identity", {}).get("device_index", 0)),
                "name": next((x.get("trace", {}).get("device_name") for x in load_records if isinstance(x.get("trace"), dict) and x.get("trace", {}).get("device_name")), None) or "unknown-device",
                "architecture": str(worker.get("identity", {}).get("device") or "unknown"),
            },
        },
        "request_summary": request_summary,
        "phases": phases,
        "operator_resolutions": operators,
        "fallback": {"fallback_count": len(fallback_events), "unexpected_fallback_count": 0, "unsupported_count": 0, "fail_closed_count": 0, "events": fallback_events},
        "memory": memory,
        "state_commit": {
            "prepared_batch_count": total_steps,
            "committed_batch_count": total_steps,
            "discarded_batch_count": 0,
            "stale_nonce_count": 0,
            "cancelled_batch_count": 0,
            "error_batch_count": 0,
            "reset": {"required": True, "attempted": True, "complete": released.get("reset_complete") is True, "failed": released.get("reset_complete") is not True},
        },
        "server": None,
        "status": "ok",
        "failure": None,
    }
    if sq8_promotion:
        result["sq8_promotion_evidence"] = {
            "schema_version": "ullm.qwen35_aq4.sq8_promotion_executor.v1",
            "request_id": internal_request_id,
            "manifest_identity": {
                "implementation_id": manifest["format"]["implementation_id"],
                "execution_profile": worker["identity"]["execution_profile"],
                "artifact_content_sha256": manifest["product"]["artifact"]["content_sha256"],
                "artifact_manifest_sha256": manifest["product"]["artifact"]["manifest_sha256"],
                "package_manifest_sha256": manifest["product"]["package"]["manifest_sha256"],
            },
            "telemetry": sq8_telemetry,
            "telemetry_binding": sq8_promotion_telemetry_binding(
                sq8_telemetry, internal_request_id
            ),
            "output_identity": {
                "token_count": len(output_token_ids),
                "token_ids_sha256": token_identity_digest(output_token_ids),
                "token_ids_recorded": False,
            },
        }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--ready-timeout", type=float, default=DEFAULT_READY_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument("--sq8-promotion-evidence", action="store_true")
    parser.add_argument("--sq8-promotion-request-id")
    args = parser.parse_args(argv)
    try:
        atomic_write(args.output, run_capture(args))
        print(json.dumps({"status": "ok", "output": str(args.output)}))
        return 0
    except (CaptureError, OSError, ValueError) as error:
        worker_stderr = _normalize_worker_stderr(getattr(error, "worker_stderr", None))
        request_id = getattr(error, "request_id", None)
        if request_id is None:
            requested = getattr(args, "sq8_promotion_request_id", None)
            request_id = requested if isinstance(requested, str) else None
        timeouts = getattr(error, "timeouts", None)
        if not isinstance(timeouts, dict):
            timeouts = _capture_timeouts(args)
        worker_lifecycle = _normalize_worker_lifecycle(
            getattr(error, "worker_lifecycle", None), request_id
        )
        reason = str(error)
        # Keep the status line bounded even when an exception originates in a
        # dependency.  The worker stderr preview has its own independent cap.
        reason = reason.encode("utf-8", errors="replace")[:4096].decode(
            "utf-8", errors="replace"
        )
        envelope = {
            "schema_version": "ullm.aq4_resident_capture_error.v4",
            "status": "failed",
            "stage": getattr(error, "stage", None) or "capture",
            "reason": reason,
            "timed_out": bool(getattr(error, "timed_out", False)),
            "request_id": request_id,
            "timeouts": timeouts,
            "worker_returncode": getattr(error, "worker_returncode", None),
            "worker_signal": getattr(error, "worker_signal", None),
            "worker_stderr": worker_stderr,
            "worker_lifecycle": worker_lifecycle,
            "observed_sq8_promotion_telemetry": getattr(
                error, "observed_sq8_promotion_telemetry", None
            ),
            "observed_sq8_promotion_telemetry_binding": getattr(
                error, "observed_sq8_promotion_telemetry_binding", None
            ),
            "worker_terminal": getattr(error, "worker_terminal", None),
        }
        # The outer runner treats this as an opaque, single-line status record
        # on failure.  The diagnostic text remains on stderr for operators.
        print(json.dumps(envelope, ensure_ascii=True, separators=(",", ":")))
        print(f"resident executor capture failed: {reason}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
