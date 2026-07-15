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
WORKER_STDERR_SCHEMA_VERSION = "ullm.aq4_resident_worker_stderr.v1"
WORKER_STDERR_PREVIEW_MAX_BYTES = 32 * 1024
WORKER_STDERR_READ_CHUNK_BYTES = 64 * 1024
WORKER_STDERR_JSON_LINE_MAX_BYTES = 1024 * 1024
WORKER_STDERR_MAX_RECORDS = 8192
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
    ) -> None:
        super().__init__(message)
        self.worker_stderr = worker_stderr
        self.stage = stage


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
        self._digest = hashlib.sha256()
        self._byte_count = 0
        self._preview = bytearray()
        self._preview_truncated = False
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
        self.stream_error: BaseException | None = None

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
        remaining = WORKER_STDERR_PREVIEW_MAX_BYTES - len(self._preview)
        if len(encoded) > remaining:
            self._preview_truncated = True
            return
        self._preview.extend(encoded)

    def _finish_line(self, *, newline: bool) -> None:
        line = bytes(self._line) if not self._line_oversized else None
        if line is None or self._line_size + (1 if newline else 0) > (
            WORKER_STDERR_PREVIEW_MAX_BYTES - len(self._preview)
        ):
            self._preview_truncated = True
        if self._line_sensitive:
            self._redacted_lines += 1
            self._append_preview(WORKER_STDERR_REDACTED_LINE + (b"\n" if newline else b""))
        elif line is None:
            # A giant non-sensitive line cannot be retained or partially
            # displayed.  Mark the display as lossy and continue draining.
            self._preview_truncated = True
        else:
            self._append_preview(line + (b"\n" if newline else b""))

        if line is not None and len(self.records) < WORKER_STDERR_MAX_RECORDS:
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                value = None
            if isinstance(value, dict):
                self.records.append(value)

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
            self.stream_error = error
        finally:
            self._finished = True

    def summary(self) -> dict[str, Any]:
        if not self._finished:
            # This is only used as a last-resort failure envelope if a stream
            # cannot be joined.  It remains structurally valid and secret-free.
            self._preview_truncated = True
        return {
            "schema_version": WORKER_STDERR_SCHEMA_VERSION,
            "byte_count": self._byte_count,
            "sha256": self._digest.hexdigest(),
            "preview_text": bytes(self._preview).decode("utf-8", errors="replace"),
            "captured_bytes": len(self._preview),
            "truncated": self._preview_truncated,
            "utf8_replacement": self._utf8_replacement,
            "redacted_lines": self._redacted_lines,
        }


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
    """Terminate a worker and reap it, tolerating an already-exited process."""

    if proc.poll() is None:
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        # A second wait gives the process a final chance to reap.  Do not leave
        # a child around merely because the first wait raced process teardown.
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
        proc.wait()


def _finish_worker_stderr(
    proc: subprocess.Popen[bytes],
    stderr_thread: threading.Thread,
    collector: WorkerStderrCollector,
) -> dict[str, Any]:
    """Join the drain thread after EOF, closing the pipe as a bounded fallback."""

    stderr_thread.join(timeout=3)
    if stderr_thread.is_alive():
        # This handles a descendant that inherited the pipe or a test double
        # whose read end does not observe EOF after the parent exits.
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except (OSError, ValueError):
            pass
        stderr_thread.join(timeout=3)
    return collector.summary()


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
        if manifest.get("format", {}).get("implementation_id") != SQ8_OVERLAY_IMPLEMENTATION_ID:
            raise CaptureError("SQ8 promotion manifest implementation identity differs")
        if worker.get("identity", {}).get("execution_profile") != SQ8_OVERLAY_EXECUTION_PROFILE:
            raise CaptureError("SQ8 promotion manifest execution profile differs")
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
        "eos_token_ids": manifest.get("generation", {}).get("eos_token_ids", []),
    }
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    assert proc.stderr is not None
    stderr_collector = WorkerStderrCollector(proc.stderr)
    stderr_records: list[dict[str, Any]] = stderr_collector.records
    stderr_thread = threading.Thread(target=stderr_collector.drain, name="aq4-stderr-drain", daemon=True)
    stderr_thread.start()
    observer = VramObserver(str(worker.get("identity", {}).get("device", "gfx1201")))
    observer_data: dict[str, Any] | None = None
    stderr_summary: dict[str, Any] | None = None
    process_error: BaseException | None = None
    try:
        observer.start()
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write((json.dumps(request, separators=(",", ":")) + "\n").encode("ascii"))
        proc.stdin.flush()
        events: list[dict[str, Any]] = []
        output_token_ids: list[int] = []
        released: dict[str, Any] | None = None
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if proc.poll() is not None:
                    break
                continue
            line = proc.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(event, dict):
                continue
            event_type = event.get("type")
            if event_type == "token":
                if event.get("request_id") != internal_request_id or event.get("index") != len(output_token_ids):
                    raise CaptureError("resident worker token event identity is discontinuous")
                token_id = event.get("token_id")
                if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                    raise CaptureError("resident worker emitted an invalid token id")
                output_token_ids.append(token_id)
            events.append({"type": event_type, "processed_prompt_tokens": event.get("processed_prompt_tokens"), "completion_tokens": event.get("completion_tokens")})
            if event_type == "released":
                released = event
                break
        if released is None:
            raise CaptureError("resident worker did not release the capture request", stage="request")
        proc.stdin.write((json.dumps({"schema_version": protocol, "type": "shutdown"}, separators=(",", ":")) + "\n").encode("ascii"))
        proc.stdin.flush()
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise CaptureError("resident worker did not shut down after capture", stage="shutdown")
    except BaseException as error:
        process_error = error
    finally:
        if process_error is not None or proc.poll() is None:
            try:
                _terminate_worker(proc)
            except BaseException as error:
                if process_error is None:
                    process_error = CaptureError("resident worker could not be reaped", stage="cleanup")
        try:
            observer_data = observer.finish()
        except BaseException as error:
            if process_error is None:
                process_error = CaptureError("resident worker resource observation failed", stage="cleanup")
        stderr_summary = _finish_worker_stderr(proc, stderr_thread, stderr_collector)
    if process_error is not None:
        if isinstance(process_error, CaptureError):
            process_error.worker_stderr = stderr_summary
            raise process_error
        raise CaptureError(str(process_error), worker_stderr=stderr_summary, stage="worker") from process_error
    assert observer_data is not None and stderr_summary is not None

    def worker_failure(message: str, stage: str = "validation") -> CaptureError:
        return CaptureError(message, worker_stderr=stderr_summary, stage=stage)

    if proc.returncode != 0:
        raise worker_failure(f"resident worker exited with status {proc.returncode}", "worker_exit")
    backend = next((x for x in reversed(stderr_records) if x.get("event") == "request_released" and isinstance(x.get("operation_execution_audit"), dict)), None)
    if backend is None:
        raise worker_failure("resident worker request audit was not observed", "audit_missing")
    audit = backend["operation_execution_audit"]
    if backend.get("request_id") != internal_request_id:
        raise worker_failure("resident worker request audit identity differs", "audit_missing")
    request_audit = backend.get("request_execution_audit")
    if not isinstance(request_audit, dict):
        raise worker_failure("resident worker request execution audit was not observed", "audit_missing")
    sq8_telemetry = None
    if sq8_promotion:
        sq8_telemetry = validate_sq8_promotion_telemetry(
            request_audit.get("sq8_promotion_telemetry")
        )
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
        "status": "ok",
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
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--sq8-promotion-evidence", action="store_true")
    parser.add_argument("--sq8-promotion-request-id")
    args = parser.parse_args(argv)
    try:
        atomic_write(args.output, run_capture(args))
        print(json.dumps({"status": "ok", "output": str(args.output)}))
        return 0
    except (CaptureError, OSError, ValueError) as error:
        worker_stderr = getattr(error, "worker_stderr", None)
        envelope = {
            "schema_version": "ullm.aq4_resident_capture_error.v1",
            "status": "error",
            "stage": getattr(error, "stage", None) or "capture",
            "reason": str(error),
            "worker_stderr": worker_stderr if isinstance(worker_stderr, dict) else None,
        }
        # The outer runner treats this as an opaque, single-line status record
        # on failure.  The diagnostic text remains on stderr for operators.
        print(json.dumps(envelope, ensure_ascii=True, separators=(",", ":")))
        print(f"resident executor capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
