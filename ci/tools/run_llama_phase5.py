#!/usr/bin/env python3
"""Fail-closed build, run, and aggregate control plane for Phase 5 P3.

The default operations are contract-only or build-only.  Inference is never
implicit: a model path, a target-specific build manifest, an exact matrix row,
and an explicit local artifact directory are required for a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from statistics import median
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover - the pinned host environment supplies it
    Draft202012Validator = None  # type: ignore[assignment,misc]
    FormatChecker = None  # type: ignore[assignment,misc]
    JSONSCHEMA_ERROR = exc
else:
    JSONSCHEMA_ERROR = None

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, canonical_bytes  # noqa: E402
import engine_performance_common as direct_contracts  # noqa: E402
import run_engine_performance as direct_health  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "ci/matrix/llama-phase5-v1.json"
SCHEMA_PATH = ROOT / "ci/schema/llama-phase5-v1.schema.json"
DIRECT_MATRIX_PATH = ROOT / "ci/matrix/engine-performance-direct-v1.json"
DIRECT_AGGREGATE_SCHEMA_PATH = ROOT / "ci/schema/engine-performance-aggregate-v1.schema.json"
REFERENCE_PATH = ROOT / "reference/llama.cpp"
WRAPPER_SOURCE = ROOT / "ci/tools/llama_phase5_wrapper.cpp"
PINNED_COMMIT = "f5919bf458ef190468b5c329bb293f8a54a1e69c"
PINNED_TREE = "e9b6173953477054a4068884aa5fc9aeef6475e8"
MODEL_SHA256 = "636158bd8a217374134cc2455aa40603f7579366fda0f0f5efcbf8bcba37c045"
WRAPPER_SOURCE_SHA256 = "43e7db595d5cc739021af6285b41b5bcf3d26d6bd25e4af70e0bf2732248296e"
SCHEMA_SHA256 = "2fed4ab03759c8c21a10dc4313f07d21ed44eeb0f1e73b59b5647796c58f48fe"
DIRECT_MATRIX_REVISION = 4
DIRECT_MATRIX_SHA256 = "fb0fab31e6c9b21a78e023bf9739a4363470e7a7df0f0898d0d878131e50cb78"
DIRECT_AGGREGATE_SCHEMA_SHA256 = "e2d9802887b6f8773657b3863f93502509b73a18ae678df13659f23ea6c9946f"
OFFICIAL_LLAMA_BENCH_DEVICE = "ROCm0"
ROCM_ROOT = Path("/opt/rocm/core-7.14")
COMPILER = ROCM_ROOT / "bin/amdclang++"
CONVERTER_PATH = REFERENCE_PATH / "convert_hf_to_gguf.py"
CONVERTER_SHA256 = "8f1bed9466221e57e434caa7ee720abe1569deb6bc2fe5a65da950ea66c8e737"
SOURCE_LOCK_PATH = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
SOURCE_LOCK_SHA256 = "4071e1b36901e523a3c5c65559f2cecda7c9cc258185770f049886f52d1fe678"
SOURCE_LOCK_FILE_SET_SHA256 = "ea94e7fe8d4e916590236b9a7f47b8aa0e896ce0a75a7252348b86c9f318d934"
SOURCE_LOCK_FINGERPRINT = "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae"
SOURCE_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_SNAPSHOT_PATH = Path("/home/homelab1/.cache/sllm/models/Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
CONVERSION_OUTPUT_PATH = Path("/home/homelab1/.cache/sllm/benchmarks/phase5/p3-llama-cpp-qwen3.5-4b-851bf6e8/qwen3.5-4b-bf16-no-mtp.gguf")
CONVERSION_SOURCE_MANIFEST_PATH = Path("/home/homelab1/.cache/sllm/benchmarks/phase5/p3-llama-cpp-qwen3.5-4b-851bf6e8/manifest.json")
CONVERSION_SOURCE_MANIFEST_SCHEMA = "phase5-p3-llama-cpp-artifacts-v1"
CONVERSION_SOURCE_MANIFEST_SHA256 = "09fceef231a65ea8793b0749fd1340f9eaffd00562aeffad7beaac74d1991f21"
MODEL_SIZE_BYTES = 8424393568
CONVERSION_DURATION_SECONDS = 27.04
ROCM_VERSION = "7.14.60850-0000000"
ROCM_COMPILER_REALPATH = "/opt/rocm/core-7.14/lib/llvm/bin/amdllvm"
ROCM_COMPILER_VERSION = "AMD clang version 23.0.0git (ROCm/llvm-project 46fcb339fb61119b337f973c7ca9e710a319fdd0+PATCHED:440716f8b87be9d8e20ed910e10e5b6d14d57cf6)"
BUILD_ROOTS = {
    "gfx1030": Path("/tmp/sllm-phase5-p3-llama-cpp-gfx1030"),
    "gfx1201": Path("/tmp/sllm-phase5-p3-llama-cpp-gfx1201"),
}
TARGETS = ("gfx1030", "gfx1201")
CASES = ("minimum", "short-odd", "boundary-255", "boundary-256", "boundary-257", "prefill-long", "decode-long")
STOP_IDS = (248046, 248044)
COMMON_VISIBILITY_NAMES = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")
MAX_JSON_BYTES = 256 * 1024 * 1024
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
DEFAULT_TIMEOUT_SECONDS = 5400
PREFILL_LONG_TIMEOUT_SECONDS = 10800
BUNDLE_PROTOCOL = "sllm-artifact-bundle-v1"
BUNDLE_COMMIT_NAME = "bundle.complete.json"


def fail(message: str) -> None:
    raise ContractError(message)


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_json_bytes(data: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        fail(f"non-finite JSON value in {label}: {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except ContractError:
        raise
    except (UnicodeError, ValueError) as exc:
        fail(f"cannot parse {label}: {exc}")


def read_json(path: Path, label: str, max_bytes: int = MAX_JSON_BYTES) -> tuple[Any, bytes, str]:
    try:
        if path.is_symlink() or not path.is_file():
            fail(f"{label} must be a regular non-symlink file: {path}")
        if path.stat().st_size > max_bytes:
            fail(f"{label} exceeds the bounded size: {path}")
        data = path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {label} {path}: {exc}")
    return parse_json_bytes(data, label), data, hashlib.sha256(data).hexdigest()


def verify_digest_sidecar(path: Path, digest: str, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        sidecar_data = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        fail(f"{label} digest sidecar is unavailable: {exc}")
    if sidecar_data != f"{digest}  {path.name}\n":
        fail(f"{label} digest sidecar is stale or tampered")


def sha256_file(path: Path, label: str) -> str:
    try:
        if path.is_symlink() or not path.is_file():
            fail(f"{label} must be a regular non-symlink file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        fail(f"cannot hash {label} {path}: {exc}")
    return digest.hexdigest()


def write_new(path: Path, data: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        fail(f"refusing to overwrite existing {label}: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        fail(f"cannot write {label} {path}: {exc}")


def _bundle_commit(payloads: Mapping[Path, bytes]) -> bytes:
    return canonical_bytes({
        "protocol": BUNDLE_PROTOCOL,
        "state": "COMMITTED",
        "members": {
            str(path.resolve()): {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for path, payload in sorted(payloads.items(), key=lambda item: str(item[0].resolve()))
        },
    })


def verify_completed_bundle(marker: Path, members: Iterable[Path], label: str) -> dict[str, Any]:
    """Accept a bundle only after its last-published commit record is present."""
    expected = {str(path.resolve()): path.resolve() for path in members}
    commit, _, _ = read_json(marker.resolve(), f"{label} completion record", 1024 * 1024)
    if not isinstance(commit, dict) or set(commit) != {"protocol", "state", "members"}:
        fail(f"{label} completion record is malformed")
    if commit["protocol"] != BUNDLE_PROTOCOL or commit["state"] != "COMMITTED":
        fail(f"{label} completion record is stale or incomplete")
    identities = commit["members"]
    if not isinstance(identities, dict) or set(identities) != set(expected):
        fail(f"{label} completion record has an incomplete member set")
    for name, path in expected.items():
        identity = identities[name]
        if not isinstance(identity, dict) or set(identity) != {"sha256", "bytes"}:
            fail(f"{label} completion member identity is malformed: {path}")
        digest = sha256_file(path, f"{label} member")
        try:
            size = path.stat().st_size
        except OSError as exc:
            fail(f"cannot stat {label} member {path}: {exc}")
        if identity["sha256"] != digest or identity["bytes"] != size:
            fail(f"{label} completion member is stale or tampered: {path}")
    return commit


def publish_completed_bundle(
    payloads: Mapping[Path, bytes], marker: Path, label: str,
) -> None:
    """Publish no-replace members first and the durable completion record last."""
    normalized = {path.parent.resolve() / path.name: payload for path, payload in payloads.items()}
    marker = marker.parent.resolve() / marker.name
    if not normalized or len(normalized) != len(payloads) or marker in normalized:
        fail(f"{label} publication bundle is malformed")
    destinations = [*normalized, marker]
    if any(path.exists() or path.is_symlink() for path in destinations):
        fail(f"refusing to overwrite existing {label} output")
    commit_payload = _bundle_commit(normalized)
    temporary: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    created_parents: set[Path] = set()
    try:
        for parent in sorted({path.parent for path in destinations}, key=lambda path: (len(path.parts), str(path))):
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                fail(f"{label} output directory is not a regular directory: {parent}")
            if not parent.exists():
                parent.mkdir(parents=True, exist_ok=False)
                created_parents.add(parent)
            if parent.is_symlink() or not parent.is_dir():
                fail(f"{label} output directory changed during publication: {parent}")
        if any(path.exists() or path.is_symlink() for path in destinations):
            fail(f"refusing to overwrite existing {label} output")
        staged = dict(normalized)
        staged[marker] = commit_payload
        for destination, payload in staged.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
            )
            temporary_path = Path(temporary_name)
            temporary[destination] = temporary_path
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for destination in normalized:
            source = temporary[destination]
            os.link(source, destination)
            published.append((source, destination))
        for parent in {path.parent for path in normalized}:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        source = temporary[marker]
        os.link(source, marker)
        published.append((source, marker))
        descriptor = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except ContractError:
        for source, destination in reversed(published):
            try:
                if destination.exists() and os.path.samefile(source, destination):
                    destination.unlink()
            except OSError:
                pass
        raise
    except (OSError, ValueError) as exc:
        for source, destination in reversed(published):
            try:
                if destination.exists() and os.path.samefile(source, destination):
                    destination.unlink()
            except OSError:
                pass
        fail(f"cannot publish {label} bundle: {exc}")
    finally:
        for path in temporary.values():
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        for parent in sorted(created_parents, key=lambda path: len(path.parts), reverse=True):
            try:
                parent.rmdir()
            except OSError:
                pass


def publish_aggregate_bundle(output: Path, data: bytes) -> tuple[Path, str]:
    """Publish aggregate, sidecar, and a last-published completion record."""
    output = output.parent.resolve() / output.name
    digest = hashlib.sha256(data).hexdigest()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    payloads = {
        output: data,
        sidecar: f"{digest}  {output.name}\n".encode("ascii"),
    }
    if len(payloads) != 2:
        fail("llama wrapper aggregate output and digest sidecar are not distinct")
    marker = output.with_suffix(output.suffix + ".complete.json")
    publish_completed_bundle(payloads, marker, "llama wrapper aggregate")
    verify_completed_bundle(marker, payloads, "llama wrapper aggregate")
    return output, digest


def schema_validate(value: Any, definition: str, label: str) -> None:
    if Draft202012Validator is None:
        fail(f"jsonschema is required for {label}: {JSONSCHEMA_ERROR}")
    schema, _, schema_digest = read_json(SCHEMA_PATH, "llama Phase 5 schema", 8 * 1024 * 1024)
    if schema_digest != SCHEMA_SHA256:
        fail("llama Phase 5 schema is stale or tampered")
    target = {"$schema": schema["$schema"], "$ref": f"#/$defs/{definition}", "$defs": schema["$defs"]}
    errors = sorted(
        Draft202012Validator(target, format_checker=FormatChecker()).iter_errors(value),
        key=lambda item: list(item.path),
    )
    if errors:
        fail(f"{label} schema validation failed: " + "; ".join(error.message for error in errors[:5]))


def validate_document_schema(value: Any, schema_path: Path, expected_digest: str, label: str) -> None:
    if Draft202012Validator is None:
        fail(f"jsonschema is required for {label}: {JSONSCHEMA_ERROR}")
    schema, _, schema_digest = read_json(schema_path, f"{label} schema", 8 * 1024 * 1024)
    if schema_digest != expected_digest:
        fail(f"{label} schema is stale or tampered")
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
        key=lambda item: list(item.path),
    )
    if errors:
        fail(f"{label} schema validation failed: " + "; ".join(error.message for error in errors[:5]))


def run_checked(command: list[str], label: str, *, cwd: Path | None = None, timeout: int = 30) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        fail(f"{label} failed to start or timed out: {exc}")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
        fail(f"{label} failed with exit {result.returncode}: {detail}")
    return result


def run_wrapper_process(
    command: list[str], environment: Mapping[str, str], timeout: float, *,
    monitor_provider: Callable[[str, int, int], dict[str, Any]] | None = None,
    monitor_target: str | None = None,
) -> dict[str, Any]:
    """Execute one llama command with bounded concurrent output and group cleanup."""
    return direct_health._execute_bounded(
        command, environment, ROOT, timeout,
        monitor_provider=monitor_provider, monitor_target=monitor_target,
    )


def require_successful_llama_output(
    capture: Mapping[str, Any], label: str, *, allow_stderr: bool = False,
) -> tuple[bytes, bytes]:
    overflow = capture.get("output_overflow")
    if not isinstance(overflow, list) or any(name not in ("stdout", "stderr") for name in overflow):
        fail(f"{label} returned malformed output-bound evidence")
    if overflow:
        fail(f"{label} output exceeded the bounded limit on {', '.join(overflow)}")
    stdout = capture.get("stdout")
    stderr = capture.get("stderr")
    if not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
        fail(f"{label} returned malformed output bytes")
    if len(stdout) > MAX_JSON_BYTES or len(stderr) > MAX_JSON_BYTES:
        fail(f"{label} output exceeds the JSON artifact bound")
    if (
        capture.get("exit_code") != 0
        or capture.get("timed_out") is not False
        or (stderr != b"" and not allow_stderr)
        or capture.get("process_group_gone") is not True
    ):
        stderr_tail = stderr.decode("utf-8", errors="replace")[-1000:].replace("\n", "\\n")
        fail(
            f"{label} failed: exit={capture.get('exit_code')!r}, "
            f"timed_out={capture.get('timed_out')!r}, "
            f"process_group_gone={capture.get('process_group_gone')!r}, "
            f"stderr_bytes={len(stderr)}, stderr_tail={stderr_tail!r}"
        )
    return stdout, stderr


def validate_official_startup_stderr(stderr: bytes, device: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only llama.cpp's exact one-device ROCm discovery notice."""
    try:
        text = stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        fail(f"official llama-bench startup stderr is not UTF-8: {exc}")
    pattern = (
        r"ggml_cuda_init: found 1 ROCm devices \(Total VRAM: [1-9][0-9]* MiB\):\n"
        r"  Device 0: " + re.escape(str(device["product"])) + r", "
        + re.escape(str(device["target"]))
        + r" \(0x[0-9a-f]+\), VMM: (?:yes|no), Wave Size: 32, VRAM: [1-9][0-9]* MiB\n"
    )
    if re.fullmatch(pattern, text) is None:
        fail("official llama-bench wrote unexpected stderr")
    return {"sha256": hashlib.sha256(stderr).hexdigest(), "bytes": len(stderr), "kind": "validated_rocm_device_discovery"}


def reference_identity() -> dict[str, str]:
    if REFERENCE_PATH.is_symlink() or not REFERENCE_PATH.is_dir():
        fail(f"pinned llama.cpp reference is missing: {REFERENCE_PATH}")
    head = run_checked(["git", "-C", str(REFERENCE_PATH), "rev-parse", "HEAD"], "llama.cpp commit").stdout.decode().strip()
    tree = run_checked(["git", "-C", str(REFERENCE_PATH), "rev-parse", "HEAD^{tree}"], "llama.cpp tree").stdout.decode().strip()
    status = run_checked(["git", "-C", str(REFERENCE_PATH), "status", "--porcelain=v1", "--untracked-files=all"], "llama.cpp status").stdout
    if head != PINNED_COMMIT or tree != PINNED_TREE or status != b"":
        fail("pinned llama.cpp source is stale, dirty, or tampered")
    return {"commit": head, "tree": tree}


def direct_source_identity() -> tuple[dict[str, Any], str]:
    direct, _, digest = read_json(DIRECT_MATRIX_PATH, "Phase 5 direct matrix")
    if direct.get("schema_version") != "engine-performance-direct-v1" or direct.get("matrix_id") != "engine-performance-direct-v1" or direct.get("revision") != DIRECT_MATRIX_REVISION:
        fail("Phase 5 direct matrix identity is stale")
    if digest != DIRECT_MATRIX_SHA256:
        fail("Phase 5 direct matrix was modified after the llama comparison matrix was fixed")
    if direct.get("protocol", {}).get("stop_token_ids") != list(STOP_IDS) or direct.get("protocol", {}).get("warmup_requests") != 3 or direct.get("protocol", {}).get("measured_requests") != 10:
        fail("Phase 5 direct matrix protocol is not the fixed wrapper protocol")
    sequences = direct.get("token_sequences")
    if not isinstance(sequences, list) or [item.get("sequence_id") for item in sequences] != list(CASES):
        fail("Phase 5 direct matrix token sequence set is incomplete or reordered")
    for item, case_id in zip(sequences, CASES):
        ids = item.get("input_token_ids")
        if item.get("input_tokens") != len(ids) or not isinstance(ids, list) or not ids or any(not is_int(token) or token < 0 for token in ids):
            fail(f"direct matrix token recipe is invalid: {case_id}")
    models = direct.get("models")
    model_4b = next((item for item in models if item.get("model_size") == "4B"), None) if isinstance(models, list) else None
    if not isinstance(model_4b, dict) or [item.get("case_id") for item in model_4b.get("cases", [])] != list(CASES):
        fail("direct matrix does not expose all seven 4B cases")
    return direct, digest


def load_matrix() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    matrix, _, matrix_digest = read_json(MATRIX_PATH, "llama Phase 5 matrix", 8 * 1024 * 1024)
    schema_validate(matrix, "matrix", "llama Phase 5 matrix")
    source = reference_identity()
    direct, direct_digest = direct_source_identity()
    if matrix["llama"]["commit"] != source["commit"] or matrix["llama"]["source_tree"] != source["tree"]:
        fail("llama matrix source identity does not match the pinned checkout")
    if matrix["source_direct_matrix"]["sha256"] != direct_digest:
        fail("llama matrix direct-source digest is stale")
    if matrix["source_direct_matrix"]["revision"] != direct["revision"]:
        fail("llama matrix direct-source revision is stale")
    target_map = {item["target"]: item for item in direct["targets"]}
    if matrix["targets"] != direct["targets"]:
        fail("llama matrix target mapping is not directly inherited from the direct matrix")
    seq_map = {item["sequence_id"]: item["input_token_ids"] for item in direct["token_sequences"]}
    cases = matrix["cases"]
    if [item["case_id"] for item in cases] != list(CASES):
        fail("llama matrix case order is not the seven-case direct order")
    for case in cases:
        ids = seq_map.get(case["direct_sequence_id"])
        if ids is None or case["input_tokens"] != len(ids):
            fail(f"llama matrix case does not directly match its token recipe: {case['case_id']}")
    expected_rows: list[dict[str, Any]] = []
    order = 0
    for target in TARGETS:
        for case in cases:
            expected_rows.append({
                "order": order,
                "row_id": f"llama-phase5-4b-{target}-{case['case_id']}",
                "target": target,
                "case_id": case["case_id"],
                "input_tokens": case["input_tokens"],
                "requested_output_tokens": case["requested_output_tokens"],
            })
            order += 1
    if matrix["rows"] != expected_rows:
        fail("llama matrix rows are missing, duplicated, reordered, or changed")
    direct_rows = {(item.get("target"), item.get("case_id")): item for item in direct.get("rows", []) if item.get("model_size") == "4B"}
    for row in expected_rows:
        direct_row = direct_rows.get((row["target"], row["case_id"]))
        if not isinstance(direct_row, dict) or any(direct_row.get(key) != row[key] for key in ("input_tokens", "requested_output_tokens")):
            fail(f"llama row diverges from direct row: {row['row_id']}")
    if matrix["protocol"] != {"backend": "hip", "dtype": "BF16", "batch_size": 1, "sequences": 1, "warmup_requests": 3, "measured_requests": 10, "stop_token_ids": [248046, 248044], "visible_stop_tokens": False, "n_batch": 2048, "n_ubatch": 512, "n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "greedy": True, "bos_insertion": False}:
        fail("llama matrix protocol drifted")
    if matrix["model"]["gguf_sha256"] != MODEL_SHA256:
        fail("llama matrix model GGUF identity drifted")
    conversion = matrix.get("conversion")
    if not isinstance(conversion, dict) or conversion != expected_conversion_identity(PINNED_TREE):
        fail("llama matrix conversion contract is missing")
    official = matrix["official_llama_bench"]
    isolation = official["uuid_isolation"]
    if isolation != {
        "environment_variable": "ROCR_VISIBLE_DEVICES",
        "value": "exact target gpu_uuid",
        "visible_device_count": 1,
        "llama_bench_device": OFFICIAL_LLAMA_BENCH_DEVICE,
    }:
        fail("official llama-bench UUID isolation metadata drifted")
    common_arguments = official["common_arguments"]
    try:
        device_argument = common_arguments[common_arguments.index("-dev") + 1]
    except (ValueError, IndexError):
        fail("official llama-bench common arguments have no complete -dev selection")
    if device_argument != OFFICIAL_LLAMA_BENCH_DEVICE:
        fail("official llama-bench must address the isolated logical device as ROCm0")
    commands = official["commands"]
    command_text = " ".join([commands["prompt_processing"], commands["decode"], *commands["paired"]])
    if "-dev " + OFFICIAL_LLAMA_BENCH_DEVICE not in command_text or "${GPU_UUID}" in command_text:
        fail("official llama-bench commands do not use ROCm0 with separate UUID isolation")
    return matrix, matrix_digest, direct, direct_digest


def row_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["row_id"]: row for row in matrix["rows"]}


def case_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in matrix["cases"]}


def direct_tokens(direct: Mapping[str, Any], sequence_id: str) -> list[int]:
    for item in direct["token_sequences"]:
        if item["sequence_id"] == sequence_id:
            return list(item["input_token_ids"])
    fail(f"unknown direct token sequence: {sequence_id}")


def _regular_path(path: Path, label: str, *, executable: bool = False) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        fail(f"{label} is not executable: {path}")
    return path


def validate_official_bench(target: str, path: Path) -> Path:
    expected_root = BUILD_ROOTS[target].resolve()
    expected = (expected_root / "bin" / "llama-bench").resolve()
    bench = _regular_path(path, "llama-bench", executable=True)
    if bench != expected:
        fail(f"llama-bench is not the pinned target build output: {bench}")
    cache = _regular_path(expected_root / "CMakeCache.txt", "llama.cpp CMake cache")
    cache_text = cache.read_text(encoding="utf-8")
    required = (
        "CMAKE_BUILD_TYPE:STRING=Release",
        f"CMAKE_HIP_ARCHITECTURES:UNINITIALIZED={target}",
        "GGML_HIP:BOOL=ON",
    )
    if any(value not in cache_text.splitlines() for value in required):
        fail("llama-bench build configuration is not the pinned HIP target configuration")
    return bench


def validate_source_lock(path: Path) -> dict[str, Any]:
    path = _regular_path(path, "Qwen3.5-4B source lock")
    if path != SOURCE_LOCK_PATH.resolve():
        fail("source lock path is not the exact Phase 5 lock")
    document, _, digest = read_json(path, "Qwen3.5-4B source lock", 16 * 1024 * 1024)
    model = document.get("model") if isinstance(document, dict) else None
    if digest != SOURCE_LOCK_SHA256:
        fail("source lock file is stale or tampered")
    if document.get("schema_version") != "model-lock-v1" or not isinstance(model, dict) or document.get("fingerprint") != SOURCE_LOCK_FINGERPRINT:
        fail("source lock fingerprint is stale")
    if model.get("repo_id") != "Qwen/Qwen3.5-4B" or model.get("resolved_revision") != SOURCE_MODEL_REVISION:
        fail("source lock repository/revision identity is stale")
    files = _source_lock_file_projection(document)
    file_set_sha256 = hashlib.sha256(canonical_bytes(files)).hexdigest()
    if file_set_sha256 != SOURCE_LOCK_FILE_SET_SHA256:
        fail("source lock file set is stale or tampered")
    return {"path": "docs/models/locks/qwen3.5-4b-bf16.json", "sha256": digest, "fingerprint": SOURCE_LOCK_FINGERPRINT, "resolved_revision": SOURCE_MODEL_REVISION, "file_set_sha256": file_set_sha256}


def _source_lock_file_projection(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = document.get("model")
    files = model.get("files") if isinstance(model, dict) else None
    if not isinstance(files, list) or len(files) != 13:
        fail("source lock file set is missing or has the wrong size")
    projection: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not is_int(item.get("size_bytes")) or not isinstance(item.get("sha256"), str):
            fail("source lock file set contains an invalid entry")
        projection.append({"path": item["path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    if len({item["path"] for item in projection}) != len(projection):
        fail("source lock file set contains duplicate paths")
    return projection


def expected_conversion_identity(source_tree: str = PINNED_TREE) -> dict[str, Any]:
    return {
        "manifest_schema": CONVERSION_SOURCE_MANIFEST_SCHEMA,
        "source_manifest": {"path": str(CONVERSION_SOURCE_MANIFEST_PATH.resolve()), "sha256": CONVERSION_SOURCE_MANIFEST_SHA256},
        "source_lock": {
            "path": "docs/models/locks/qwen3.5-4b-bf16.json",
            "sha256": SOURCE_LOCK_SHA256,
            "fingerprint": SOURCE_LOCK_FINGERPRINT,
            "resolved_revision": SOURCE_MODEL_REVISION,
            "file_set_sha256": SOURCE_LOCK_FILE_SET_SHA256,
        },
        "source": {
            "repository": "https://github.com/ggml-org/llama.cpp",
            "path": str(REFERENCE_PATH.resolve()),
            "commit": PINNED_COMMIT,
            "tree": source_tree,
            "checkout": "detached",
            "clean": True,
        },
        "tool": {
            "path": str(CONVERTER_PATH.resolve()),
            "sha256": CONVERTER_SHA256,
            "commit": PINNED_COMMIT,
            "tree": source_tree,
        },
        "arguments": [
            "python3", str(CONVERTER_PATH.resolve()), str(MODEL_SNAPSHOT_PATH.resolve()), "--outfile", str(CONVERSION_OUTPUT_PATH.resolve()),
            "--outtype", "bf16", "--no-mtp",
        ],
        "duration_seconds": CONVERSION_DURATION_SECONDS,
        "output": {"path": str(CONVERSION_OUTPUT_PATH.resolve()), "sha256": MODEL_SHA256, "bytes": MODEL_SIZE_BYTES, "format": "GGUF", "dtype": "BF16"},
        "gguf": {"architecture": "qwen35", "name": SOURCE_MODEL_REVISION, "file_type": 32, "quantization_version": 2, "tensor_count": 426, "mtp_tensor_count": 0},
        "toolchain": {"rocm_root": str(ROCM_ROOT), "rocm_version": ROCM_VERSION, "compiler_path": str(COMPILER), "compiler_realpath": ROCM_COMPILER_REALPATH, "compiler_version": ROCM_COMPILER_VERSION},
    }


def validate_conversion_manifest(path: Path, source_lock: Mapping[str, Any], model: Path) -> dict[str, Any]:
    path = _regular_path(path, "GGUF conversion manifest")
    if path != CONVERSION_SOURCE_MANIFEST_PATH.resolve():
        fail("GGUF conversion manifest path is not the immutable Phase 5 cache manifest")
    document, _, digest = read_json(path, "GGUF conversion manifest", 16 * 1024 * 1024)
    if digest != CONVERSION_SOURCE_MANIFEST_SHA256:
        fail("GGUF conversion manifest is stale or tampered")
    schema_validate(document, "conversion_manifest", "GGUF conversion manifest")
    expected_lock = validate_source_lock(SOURCE_LOCK_PATH)
    if dict(source_lock) != expected_lock:
        fail("conversion manifest source lock identity is stale")
    manifest_model = document["model"]
    lock_document, _, lock_digest = read_json(SOURCE_LOCK_PATH, "Qwen3.5-4B source lock", 16 * 1024 * 1024)
    if lock_digest != expected_lock["sha256"] or manifest_model["files"] != _source_lock_file_projection(lock_document):
        fail("conversion manifest source lock file set is stale")
    if manifest_model["snapshot_path"] != str(MODEL_SNAPSHOT_PATH.resolve()) or manifest_model["lock_path"] != str(SOURCE_LOCK_PATH.resolve()):
        fail("conversion manifest cache or source-lock path is stale")
    if document["source"]["commit"] != PINNED_COMMIT or document["source"]["local_path"] != str(REFERENCE_PATH.resolve()) or document["source"]["checkout"] != "detached" or document["source"]["clean_before"] is not True or document["source"]["clean_after"] is not True:
        fail("conversion manifest llama.cpp source identity is stale")
    source = reference_identity()
    if document["repository_after"]["reference_head"] != PINNED_COMMIT or document["repository_after"]["reference_path"] != str(REFERENCE_PATH.resolve()) or document["repository_after"]["reference_status"] != "clean" or document["repository_after"]["reference_check"] != "PASS":
        fail("conversion manifest reference checkout identity is stale")
    if document["source"]["commit"] != source["commit"]:
        fail("conversion manifest source commit does not match the pinned checkout")
    if document["scope"]["gpu_benchmarks_run"] is not False or document["scope"]["generation_run"] is not False or document["scope"]["repository_files_modified"] is not False or document["scope"]["repository_files_created"] is not False:
        fail("conversion manifest scope is not the closed conversion-only evidence")
    conversion = document["conversion"]
    if conversion["converter_path"] != str(CONVERTER_PATH.resolve()) or conversion["converter_sha256"] != CONVERTER_SHA256:
        fail("conversion manifest converter identity is stale")
    if sha256_file(CONVERTER_PATH, "llama.cpp conversion tool") != CONVERTER_SHA256:
        fail("conversion tool is stale or tampered")
    expected_args = expected_conversion_identity(source["tree"])["arguments"]
    expected_dry_run_args = expected_args + ["--dry-run"]
    if conversion["run"]["args"] != expected_args or conversion["dry_run"]["args"] != expected_dry_run_args or conversion["run"]["result"] != "PASS" or conversion["dry_run"]["result"] != "PASS" or conversion["dry_run"]["output_created"] is not False:
        fail("conversion manifest arguments or PASS status are stale")
    if conversion["run"]["duration_seconds"] != CONVERSION_DURATION_SECONDS:
        fail("conversion manifest duration is stale")
    if document["gguf_metadata_validation"] != {
        "method": "Python gguf.GGUFReader metadata-only read; no ROCm/HIP binary or GPU access", "result": "PASS", "field_count": 44,
        "general_architecture": "qwen35", "general_name": SOURCE_MODEL_REVISION, "general_file_type": 32, "general_quantization_version": 2,
        "tensor_count": 426, "mtp_tensor_count": 0,
    }:
        fail("conversion manifest GGUF metadata identity is stale")
    toolchain = document["toolchain"]
    if any(toolchain.get(key) != value for key, value in {
        "rocm_root": str(ROCM_ROOT), "rocm_version": ROCM_VERSION, "compiler_path": str(COMPILER), "compiler_realpath": ROCM_COMPILER_REALPATH, "compiler_version": ROCM_COMPILER_VERSION,
    }.items()):
        fail("conversion manifest ROCm toolchain identity is stale")
    for build in document["builds"]:
        target = build.get("target")
        if target not in TARGETS or build.get("source_build_identity") != {"reported_by_cmake": "ggml commit: f5919bf", "llama_cli_version": "version: 1 (f5919bf)"} or build.get("target_audit", {}).get("offload_arches") != [target] or build.get("target_audit", {}).get("offload_images") != [target] or build.get("target_audit", {}).get("native_flag_command_count") != 0 or build.get("runtime", {}).get("rocm_resolution") != "PASS":
            fail("conversion manifest llama.cpp build identity is stale")
    if [build.get("target") for build in document["builds"]] != list(TARGETS):
        fail("conversion manifest target build set is incomplete or reordered")
    limits = document["validation_and_limits"]
    if limits.get("fallback_or_timeout") != "none observed" or limits.get("gpu_benchmark") != "DEFERRED" or limits.get("generation") != "DEFERRED":
        fail("conversion manifest validation limits are stale")
    output = conversion["run"]
    output_path = Path(output["output_path"]).resolve()
    if model.resolve() != CONVERSION_OUTPUT_PATH.resolve() or output_path != CONVERSION_OUTPUT_PATH.resolve() or output["output_size_bytes"] != MODEL_SIZE_BYTES or output["output_sha256"] != MODEL_SHA256:
        fail("conversion manifest GGUF output identity is stale")
    if output_path.stat().st_size != MODEL_SIZE_BYTES or sha256_file(output_path, "converted GGUF") != MODEL_SHA256:
        fail("converted GGUF is stale or tampered")
    normalized = expected_conversion_identity(source["tree"])
    if normalized["duration_seconds"] != conversion["run"]["duration_seconds"]:
        fail("normalized conversion duration is stale")
    return {"path": str(path), "sha256": digest, "manifest": normalized}


def expected_device(matrix: Mapping[str, Any], target: str) -> dict[str, Any]:
    for item in matrix["targets"]:
        if item["target"] == target:
            return dict(item)
    fail(f"unknown exact target: {target}")


def matrix_case(matrix: Mapping[str, Any], direct: Mapping[str, Any], row_id: str) -> tuple[dict[str, Any], list[int]]:
    rows = row_map(matrix)
    if row_id not in rows:
        fail(f"row is not in the closed llama matrix: {row_id}")
    row = rows[row_id]
    case = case_map(matrix).get(row["case_id"])
    if case is None:
        fail("row references an unknown case")
    tokens = direct_tokens(direct, case["direct_sequence_id"])
    if len(tokens) != row["input_tokens"]:
        fail("row token length does not match the direct recipe")
    return row, tokens


def build_root_for(target: str, build_roots: Mapping[str, Path] | None = None) -> Path:
    roots = build_roots or BUILD_ROOTS
    if target not in roots:
        fail(f"no pinned shared build is registered for {target}")
    root = Path(roots[target]).resolve()
    if root.is_symlink() or not root.is_dir():
        fail(f"llama.cpp shared build directory is missing: {root}")
    return root


def cache_value(cache: Path, key: str) -> str | None:
    try:
        lines = cache.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        fail(f"cannot read CMakeCache: {exc}")
    for line in lines:
        if line.startswith(key + ":") or line.startswith(key + "="):
            return line.split("=", 1)[1]
    return None


def compiler_version() -> str:
    if not COMPILER.is_file() or not os.access(COMPILER, os.X_OK):
        fail(f"pinned ROCm compiler is unavailable: {COMPILER}")
    result = run_checked([str(COMPILER), "--version"], "amdclang++ version")
    version = result.stdout.decode("utf-8", errors="replace").splitlines()[0].strip()
    if "clang" not in version.lower() or "23" not in version:
        fail(f"compiler is not the expected ROCm 7.14 LLVM 23 compiler: {version}")
    return version


def build_audit(target: str, build_root: Path) -> tuple[dict[str, Any], list[str]]:
    cache = build_root / "CMakeCache.txt"
    if cache.is_symlink() or not cache.is_file():
        fail(f"shared build has no regular CMakeCache: {build_root}")
    source = reference_identity()
    cmake_home = cache_value(cache, "CMAKE_HOME_DIRECTORY:INTERNAL")
    cmake_target = cache_value(cache, "CMAKE_HIP_ARCHITECTURES")
    cxx = cache_value(cache, "CMAKE_CXX_COMPILER")
    if cmake_home != str(REFERENCE_PATH) or cmake_target != target or cxx is None:
        fail(f"shared build identity is not exact for {target}")
    compiler_real = Path(cxx).resolve()
    if compiler_real != COMPILER.resolve():
        fail(f"shared build compiler is not the pinned amdclang++: {cxx}")
    if cache_value(cache, "BUILD_SHARED_LIBS") != "ON" or cache_value(cache, "GGML_HIP") != "ON":
        fail("shared build is not a shared HIP build")
    bin_dir = build_root / "bin"
    hip_library = (bin_dir / "libggml-hip.so").resolve()
    llama_library = (bin_dir / "libllama.so").resolve()
    if not hip_library.is_file() or not llama_library.is_file():
        fail("shared build is missing libggml-hip or libllama")
    marker = f"amdgcn-amd-amdhsa--{target}"
    marker_bytes = run_checked(["strings", str(hip_library)], "HIP target marker audit").stdout
    if marker.encode("ascii") not in marker_bytes:
        fail(f"shared HIP library does not contain the exact target audit marker: {target}")
    expected_runpath = f"{ROCM_ROOT}/lib:{ROCM_ROOT}/lib/llvm/lib:{bin_dir}"
    return {"commit": source["commit"], "tree": source["tree"], "bin_dir": bin_dir, "hip_library": hip_library, "llama_library": llama_library, "runpath": expected_runpath, "cmake_cache_sha256": sha256_file(cache, "llama.cpp CMakeCache")}, [str(COMPILER), "-std=c++17", "-O2"]


def library_closure(binary: Path, target: str, build_root: Path, expected_runpath: str) -> list[dict[str, str]]:
    dynamic = run_checked(["readelf", "-d", str(binary)], "wrapper readelf")
    text = dynamic.stdout.decode("utf-8", errors="replace")
    match = re.search(r"RUNPATH.*\[(.*?)\]", text)
    if match is None or match.group(1) != expected_runpath:
        fail("wrapper RUNPATH is not the exact build/ROCm closure")
    ldd = run_checked(["ldd", str(binary)], "wrapper ldd")
    libraries: list[dict[str, str]] = []
    allowed_roots = (build_root / "bin").resolve(), ROCM_ROOT.resolve(), Path("/lib").resolve(), Path("/usr/lib").resolve(), Path("/lib64").resolve(), Path("/usr/lib64").resolve()
    for line in ldd.stdout.decode("utf-8", errors="replace").splitlines():
        if "not found" in line:
            fail(f"wrapper library closure has an unresolved dependency: {line}")
        item = re.match(r"\s*(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-f]+\)", line)
        if item is None:
            continue
        name, path_value = item.groups()
        path = Path(path_value).resolve()
        if not path.is_file() or not any(path == root or root in path.parents for root in allowed_roots):
            fail(f"wrapper dependency escapes the approved closure: {name} -> {path}")
        libraries.append({"name": name, "path": str(path), "sha256": sha256_file(path, f"dependency {name}")})
    if not libraries:
        fail("wrapper dependency closure is empty")
    return libraries


def build_one(target: str, output_dir: Path, build_roots: Mapping[str, Path] | None = None) -> dict[str, Any]:
    matrix, _, _, _ = load_matrix()
    if target not in TARGETS:
        fail(f"target is outside the exact build matrix: {target}")
    build_root = build_root_for(target, build_roots)
    audit, _ = build_audit(target, build_root)
    version = compiler_version()
    output_dir = output_dir.resolve()
    binary = output_dir / f"llama-phase5-{target}"
    manifest_path = output_dir / f"llama-phase5-{target}.build.json"
    if binary.exists() or manifest_path.exists():
        fail(f"refusing to overwrite build output: {binary}")
    command = [
        str(COMPILER), "-std=c++17", "-O2", "-fPIC",
        "-I", str(REFERENCE_PATH / "include"), "-I", str(REFERENCE_PATH / "ggml/include"),
        str(WRAPPER_SOURCE), "-L", str(audit["bin_dir"]), "-Wl,--no-as-needed", "-Wl,-z,now",
        f"-Wl,-rpath,{audit['runpath']}", "-lllama", "-lggml", "-lggml-base", "-o", str(binary),
    ]
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(f"cannot create build output directory: {exc}")
    run_checked(command, f"build wrapper {target}", cwd=ROOT, timeout=180)
    if binary.is_symlink() or not binary.is_file() or not os.access(binary, os.X_OK):
        fail("compiler did not produce an executable regular wrapper")
    libraries = library_closure(binary, target, build_root, audit["runpath"])
    inputs = {
        "cmake_cache_sha256": audit["cmake_cache_sha256"], "compiler": str(COMPILER), "compiler_version": version,
        "rocm_root": str(ROCM_ROOT), "target": target, "source_commit": audit["commit"], "source_tree": audit["tree"],
        "wrapper_sha256": sha256_file(WRAPPER_SOURCE, "wrapper source"),
    }
    manifest = {
        "schema_version": "llama-phase5-v1", "record_kind": "build_manifest", "state": "PASS", "target": target,
        "binary": {"path": str(binary), "sha256": sha256_file(binary, "wrapper binary"), "bytes": binary.stat().st_size},
        "build": {"path": str(build_root), "compiler": str(COMPILER), "compiler_version": version, "rocm_root": str(ROCM_ROOT), "cmake_target": target, "cmake_source": "reference/llama.cpp", "rpath": [str(ROCM_ROOT / "lib"), str(ROCM_ROOT / "lib/llvm/lib"), str(audit["bin_dir"])], "inputs": inputs, "inputs_sha256": hashlib.sha256(canonical_bytes(inputs)).hexdigest()},
        "source": {"repository": "https://github.com/ggml-org/llama.cpp", "commit": audit["commit"], "tree": audit["tree"], "header_sha256": sha256_file(REFERENCE_PATH / "include/llama.h", "pinned llama.h"), "wrapper_sha256": sha256_file(WRAPPER_SOURCE, "wrapper source")},
        "closure": {"readelf_runpath_exact": True, "ldd_clean": True, "target_audit": True, "libraries": libraries},
        "command": command,
    }
    schema_validate(manifest, "build_manifest", "wrapper build manifest")
    encoded = canonical_bytes(manifest)
    write_new(manifest_path, encoded, "wrapper build manifest")
    write_new(manifest_path.with_suffix(manifest_path.suffix + ".sha256"), f"{hashlib.sha256(encoded).hexdigest()}  {manifest_path.name}\n".encode("ascii"), "build manifest digest")
    return {"target": target, "binary": str(binary), "manifest": str(manifest_path), "binary_sha256": manifest["binary"]["sha256"], "command": command}


def validate_build_manifest(path: Path, target: str) -> dict[str, Any]:
    manifest, _, digest = read_json(path, "wrapper build manifest", 8 * 1024 * 1024)
    schema_validate(manifest, "build_manifest", "wrapper build manifest")
    if manifest["target"] != target or manifest["source"]["commit"] != PINNED_COMMIT or manifest["source"]["tree"] != PINNED_TREE or manifest["build"]["cmake_target"] != target:
        fail("wrapper build manifest target/source identity is stale")
    if manifest["source"]["wrapper_sha256"] != WRAPPER_SOURCE_SHA256 or sha256_file(WRAPPER_SOURCE, "wrapper source") != manifest["source"]["wrapper_sha256"]:
        fail("wrapper source is stale or tampered")
    binary = Path(manifest["binary"]["path"]).resolve()
    if binary.is_symlink() or not binary.is_file() or binary.stat().st_size != manifest["binary"]["bytes"] or sha256_file(binary, "wrapper binary") != manifest["binary"]["sha256"]:
        fail("wrapper binary is stale or tampered")
    build_root = Path(manifest["build"]["path"]).resolve()
    if build_root != build_root_for(target):
        fail("wrapper build root is outside the pinned target build")
    audit, _ = build_audit(target, build_root)
    expected_runpath = audit["runpath"]
    if manifest["build"]["rpath"] != [str(ROCM_ROOT / "lib"), str(ROCM_ROOT / "lib/llvm/lib"), str(audit["bin_dir"])]:
        fail("wrapper build RUNPATH metadata is stale")
    if manifest["build"]["compiler_version"] != compiler_version():
        fail("wrapper build compiler identity is stale")
    inputs = manifest["build"]["inputs"]
    expected_inputs = {
        "cmake_cache_sha256": audit["cmake_cache_sha256"], "compiler": str(COMPILER),
        "compiler_version": manifest["build"]["compiler_version"], "rocm_root": str(ROCM_ROOT),
        "target": target, "source_commit": PINNED_COMMIT, "source_tree": PINNED_TREE,
        "wrapper_sha256": WRAPPER_SOURCE_SHA256,
    }
    if inputs != expected_inputs or manifest["build"]["inputs_sha256"] != hashlib.sha256(canonical_bytes(expected_inputs)).hexdigest():
        fail("wrapper build inputs identity is stale")
    if manifest["source"]["header_sha256"] != sha256_file(REFERENCE_PATH / "include/llama.h", "pinned llama.h"):
        fail("pinned llama.h is stale or tampered")
    if library_closure(binary, target, build_root, expected_runpath) != manifest["closure"]["libraries"]:
        fail("wrapper dependency closure changed after build")
    verify_digest_sidecar(path, digest, "wrapper build manifest")
    return manifest


def expected_command(binary: Path, row: Mapping[str, Any], model: Path, tokens: list[int]) -> list[str]:
    return [
        str(binary), "--benchmark-schema-version", "llama-phase5-v1", "--model", str(model), "--model-sha256", MODEL_SHA256,
        "--target", row["target"], "--row-id", row["row_id"], "--case-id", row["case_id"],
        "--input-token-ids", ",".join(str(token) for token in tokens), "--max-new-tokens", str(row["requested_output_tokens"]),
        "--warmup-requests", "3", "--measured-requests", "10", "--batch-size", "1", "--sequences", "1",
        "--n-batch", "2048", "--n-ubatch", "512", "--main-gpu", "0",
    ]


def median(values: Iterable[float | int]) -> float:
    ordered = sorted(values)
    if not ordered:
        fail("median requires a non-empty sample")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (float(ordered[middle - 1]) + float(ordered[middle])) / 2.0


def validate_sample(sample: Mapping[str, Any], row: Mapping[str, Any], tokens: list[int]) -> dict[str, Any]:
    events = sample["events"]
    ordered = [events["request_start_ns"], events["prefill_submit_ns"], events["prefill_complete_ns"], events["first_token_ns"], *events["later_token_publications_ns"], events["stop_ns"], events["cleanup_complete_ns"]]
    if any(not is_int(value) or value < 0 for value in ordered) or any(left >= right for left, right in zip(ordered, ordered[1:])):
        fail("sample event order is missing, non-monotonic, or non-positive")
    token_doc = sample["tokens"]
    if token_doc["input_token_ids"] != tokens or token_doc["bos_inserted"] is not False or token_doc["stop_token_ids_fed_back"] != []:
        fail("sample input/BOS/stop-feedback token evidence is wrong")
    generated = token_doc["generated_token_ids"]
    visible = token_doc["visible_token_ids"]
    if len(events["later_token_publications_ns"]) != len(generated) - 1 or not generated or len(generated) > row["requested_output_tokens"]:
        fail("sample publication count or output budget is invalid")
    stop = sample["stop"]
    if stop["kind"] == "stop_token":
        if generated[-1] != stop["token_id"] or stop["token_id"] not in STOP_IDS or visible != generated[:-1]:
            fail("stop-token sequence is inconsistent")
    elif stop["kind"] == "max_new_tokens":
        if stop["token_id"] is not None or len(generated) != row["requested_output_tokens"] or visible != generated:
            fail("max-token sequence is inconsistent")
    else:
        fail("unknown stop reason")
    if any(token in STOP_IDS for token in visible) or any(token in STOP_IDS for token in generated[:-1]):
        fail("stop token appeared in visible or fed-back token sequence")
    audit = sample["audit"]
    if audit["prefill_logits_index"] != len(tokens) - 1 or audit["prefill_logits_position"] != len(tokens) - 1 or audit["decode_first_position"] != len(tokens):
        fail("logits index/position audit is wrong")
    derived = sample["derived"]
    ttft = events["first_token_ns"] - events["request_start_ns"]
    prefill = events["prefill_complete_ns"] - events["prefill_submit_ns"]
    e2e = events["cleanup_complete_ns"] - events["request_start_ns"]
    tpot = [right - left for left, right in zip([events["first_token_ns"], *events["later_token_publications_ns"]], events["later_token_publications_ns"])]
    decode_rate = None if not events["later_token_publications_ns"] else (len(generated) - 1) * 1_000_000_000 / (events["later_token_publications_ns"][-1] - events["first_token_ns"])
    expected = {"ttft_ns": ttft, "prefill_ns": prefill, "prefill_tokens_per_second": len(tokens) * 1_000_000_000 / prefill, "tpot_ns": tpot, "decode_tokens": len(generated) - 1, "decode_tokens_per_second": decode_rate, "e2e_ns": e2e}
    if derived["ttft_ns"] != expected["ttft_ns"] or derived["prefill_ns"] != expected["prefill_ns"] or derived["tpot_ns"] != expected["tpot_ns"] or derived["decode_tokens"] != expected["decode_tokens"] or derived["e2e_ns"] != expected["e2e_ns"]:
        fail("sample derived integer metrics do not match event arithmetic")
    for key in ("prefill_tokens_per_second", "decode_tokens_per_second"):
        actual = derived[key]
        expected_value = expected[key]
        if expected_value is None:
            if actual is not None:
                fail(f"sample {key} must be null for a one-token decode")
        elif not isinstance(actual, (int, float)) or not math.isclose(float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-9):
            fail(f"sample {key} does not match event arithmetic")
    return dict(derived)


def validate_offload_evidence(offload: Mapping[str, Any]) -> str:
    schema_validate(offload, "offload_evidence", "llama GPU-offload evidence")
    selected = offload["selected_device"]
    requested = offload["requested"]
    observed = offload["observed"]
    device_memory = observed["device_memory"]
    if offload["gpu_offload_supported"] is not True or offload["visible_gpu_device_count"] != 1:
        fail("wrapper did not observe one GPU-offload-capable device")
    if selected["type"] != "GPU" or not selected["name"].startswith("ROCm") or not selected["description"]:
        fail("wrapper selected-device evidence is not an observable ROCm GPU")
    if requested != {"n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "offload_kqv": True, "op_offload": True}:
        fail("wrapper GPU-offload request parameters drifted")
    if observed["offloaded_layers"] <= 0 or observed["offloaded_layers"] != observed["offloadable_layers"] or observed["gpu_model_buffer_mib"] <= 0 or observed["captured_log_bytes"] <= 0:
        fail("wrapper logs did not prove positive full-layer GPU offload")
    expected_decrease = device_memory["free_before_bytes"] - device_memory["free_model_ready_bytes"]
    if device_memory["total_before_bytes"] <= 0 or device_memory["total_model_ready_bytes"] != device_memory["total_before_bytes"] or expected_decrease <= 0 or device_memory["observed_decrease_bytes"] != expected_decrease:
        fail("wrapper device-memory observation does not prove a model-ready GPU allocation")
    return hashlib.sha256(canonical_bytes(offload)).hexdigest()


def validate_result(result: Any, row: Mapping[str, Any], tokens: list[int], model: Path) -> dict[str, Any]:
    schema_validate(result, "result", "llama wrapper result")
    if result["state"] != "PASS" or result["row_id"] != row["row_id"] or result["case_id"] != row["case_id"] or result["llama_commit"] != PINNED_COMMIT:
        fail("wrapper result row or commit identity is stale")
    if result["model"]["path"] != str(model) or result["model"]["sha256"] != MODEL_SHA256:
        fail("wrapper result model identity is stale")
    device = expected_device(_MATRIX_CACHE, row["target"])
    if result["target"] != {"exact": device["target"], "gpu_uuid": device["gpu_uuid"], "main_gpu": 0, "logical_device_index": 0}:
        fail("wrapper result exact target mapping is wrong")
    if result["input_token_ids"] != tokens or result["protocol"]["n_ctx"] != row["input_tokens"] + row["requested_output_tokens"]:
        fail("wrapper result input or context identity is wrong")
    if result["model_lifecycle"]["load_count"] != 1 or result["model_lifecycle"]["context_count"] != 1 or result["model_lifecycle"]["resident_reused"] is not True or result["model_lifecycle"]["model_ready_ns"] <= result["model_lifecycle"]["load_start_ns"]:
        fail("wrapper did not prove model/context reuse")
    offload = result["offload_evidence"]
    offload_digest = validate_offload_evidence(offload)
    audit = result["audit"]
    if any(audit[key] is not True for key in ("sample_equality", "request_memory_reset", "sampler_reset", "stop_tokens_not_fed_back", "model_reused", "context_reused")) or audit["early_error_count"] != 0 or audit["errors"] != []:
        fail("wrapper audit is not fail-closed")
    warmups = result["warmups"]["samples"]
    measured = result["measured"]["samples"]
    all_samples = warmups + measured
    if [sample["sample_index"] for sample in warmups] != [0, 1, 2] or [sample["sample_index"] for sample in measured] != list(range(3, 13)):
        fail("wrapper sample indexes are not deterministic")
    derived = [validate_sample(sample, row, tokens) for sample in all_samples]
    token_signature = [(sample["tokens"]["generated_token_ids"], sample["tokens"]["visible_token_ids"], sample["stop"]) for sample in all_samples]
    if any(signature != token_signature[0] for signature in token_signature[1:]):
        fail("wrapper samples are not equal after request reset")
    return {"measured": derived[3:], "token_signature": token_signature[0], "offload_sha256": offload_digest}


def validate_model(path: Path) -> str:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        fail(f"model must be an explicit regular non-symlink file: {path}")
    if path != CONVERSION_OUTPUT_PATH.resolve():
        fail(f"model must be the exact Phase 5 cache GGUF: {path}")
    if path.stat().st_size != MODEL_SIZE_BYTES:
        fail("model GGUF size is stale or tampered")
    digest = sha256_file(path, "Qwen3.5-4B BF16 GGUF")
    if digest != MODEL_SHA256:
        fail("model GGUF is stale or tampered")
    return digest


def run_row(
    row_id: str, binary_manifest: Path, model: Path, artifact_dir: Path, *,
    conversion_manifest: Path | None = None, source_lock: Path = SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    global _MATRIX_CACHE
    matrix, matrix_digest, direct, direct_digest = load_matrix()
    _MATRIX_CACHE = matrix
    row, tokens = matrix_case(matrix, direct, row_id)
    model = model.resolve()
    validate_model(model)
    lock_identity = validate_source_lock(source_lock)
    if conversion_manifest is None:
        fail("--run-row requires --conversion-manifest")
    conversion_identity = validate_conversion_manifest(conversion_manifest, lock_identity, model)
    build_manifest_path = binary_manifest.resolve()
    build_manifest = validate_build_manifest(build_manifest_path, row["target"])
    binary = Path(build_manifest["binary"]["path"]).resolve()
    row_dir = artifact_dir.resolve() / row_id
    if row_dir.exists() or row_dir.is_symlink():
        fail(f"refusing to reuse row artifact directory: {row_dir}")
    raw_path = row_dir / "raw-result.json"
    stderr_path = row_dir / "stderr.txt"
    manifest_path = row_dir / "manifest.json"
    command = expected_command(binary, row, model, tokens)
    environment = os.environ.copy()
    for name in COMMON_VISIBILITY_NAMES:
        environment.pop(name, None)
    device = expected_device(matrix, row["target"])
    environment["ROCR_VISIBLE_DEVICES"] = device["gpu_uuid"]
    environment["SLLM_LLAMA_PHASE5_TARGET"] = row["target"]
    timeout_seconds = PREFILL_LONG_TIMEOUT_SECONDS if row["case_id"] == "prefill-long" else DEFAULT_TIMEOUT_SECONDS
    pre = direct_health.validate_observation(direct_health._amd_smi_observation(row["target"], "pre"), row["target"], "pre")
    pre_evidence = direct_health._amd_smi_phase_evidence(row["target"], "pre")
    try:
        capture = run_wrapper_process(
            command, environment, timeout_seconds,
            monitor_provider=direct_health._amd_smi_monitor_sample, monitor_target=row["target"],
        )
    finally:
        post = direct_health.validate_observation(direct_health._amd_smi_observation(row["target"], "post"), row["target"], "post")
        post_evidence = direct_health._amd_smi_phase_evidence(row["target"], "post")
    raw_bytes, stderr_bytes = require_successful_llama_output(capture, "wrapper execution")
    raw = parse_json_bytes(raw_bytes, "raw wrapper result")
    raw_digest = hashlib.sha256(raw_bytes).hexdigest()
    validation = validate_result(raw, row, tokens, model)
    evidence = direct_health._build_evidence(pre_evidence, post_evidence, capture, row["target"], {"path": direct_health.AMD_SMI_EXECUTABLE, **direct_health._amd_smi_version()})
    if not direct_health._observations_have_stable_authorization(pre, post) or evidence["checks"]["process_group_cleanup"] is not True:
        fail("wrapper pre/post health or cleanup evidence changed")
    lifecycle = raw["model_lifecycle"]
    model_load_ns = lifecycle["model_ready_ns"] - lifecycle["load_start_ns"]
    available_vram = [evidence["pre"]["vram_auxiliary"]["free_mb"], evidence["post"]["vram_auxiliary"]["free_mb"]]
    if model_load_ns <= 0 or any(not isinstance(value, (int, float)) or value < 0 for value in available_vram):
        fail("model-load or available-VRAM evidence is invalid")
    build_digest = sha256_file(build_manifest_path, "wrapper build manifest")
    manifest = {
        "schema_version": "llama-phase5-v1", "record_kind": "run_manifest", "state": "PASS", "row_id": row_id,
        "matrix": {"path": "ci/matrix/llama-phase5-v1.json", "sha256": matrix_digest},
        "schema": {"path": "ci/schema/llama-phase5-v1.schema.json", "sha256": sha256_file(SCHEMA_PATH, "llama Phase 5 schema")},
        "source": {"commit": PINNED_COMMIT, "tree": PINNED_TREE, "direct_matrix_sha256": direct_digest},
        "model": {"path": str(model), "sha256": MODEL_SHA256}, "source_lock": lock_identity,
        "conversion": conversion_identity,
        "target": {key: device[key] for key in ("target", "gpu_uuid", "gpu_bdf", "product", "logical_device_index")} | {"rocm_device": OFFICIAL_LLAMA_BENCH_DEVICE},
        "binary": {"path": str(binary), "sha256": build_manifest["binary"]["sha256"]},
        "build_manifest": {"path": str(build_manifest_path), "sha256": build_digest},
        "raw_result": {"path": str(raw_path), "sha256": raw_digest, "bytes": len(raw_bytes)},
        "stderr": {"path": str(stderr_path), "sha256": EMPTY_SHA256, "bytes": 0},
        "command": command,
        "execution": {"exit_code": 0, "timed_out": False, "timeout_seconds": timeout_seconds, "process_group_gone": True, "stderr_bytes": 0, "visibility": {"selector": "ROCR_VISIBLE_DEVICES", "uuid": device["gpu_uuid"], "visible_device_count": 1, "cleared": list(COMMON_VISIBILITY_NAMES)}},
        "evidence": evidence,
        "offload_evidence": {"observation": raw["offload_evidence"], "sha256": validation["offload_sha256"]},
        "metrics": {"model_load_ns": model_load_ns, "available_vram_mb": available_vram},
        "cleanup": {"raw_output_preserved": True, "raw_output_sha256": raw_digest, "stderr_empty": True, "process_group_gone": True, "temporary_files_removed": []},
    }
    schema_validate(manifest, "run_manifest", "llama wrapper run manifest")
    encoded = canonical_bytes(manifest)
    manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    row_marker = row_dir / BUNDLE_COMMIT_NAME
    row_payloads = {
        raw_path: raw_bytes,
        stderr_path: stderr_bytes,
        manifest_path: encoded,
        manifest_sidecar: f"{hashlib.sha256(encoded).hexdigest()}  {manifest_path.name}\n".encode("ascii"),
    }
    publish_completed_bundle(row_payloads, row_marker, "llama wrapper row")
    verify_completed_bundle(row_marker, row_payloads, "llama wrapper row")
    return {"state": "PASS", "row_id": row_id, "raw_result": str(raw_path), "raw_sha256": raw_digest, "metrics": validation["measured"]}


def run_all(
    binary_manifests: Mapping[str, Path], model: Path, artifact_dir: Path, *,
    conversion_manifest: Path | None = None, source_lock: Path = SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    matrix, _, _, _ = load_matrix()
    results = []
    for row in matrix["rows"]:
        manifest = binary_manifests.get(row["target"])
        if manifest is None:
            fail(f"missing target build manifest for serial row execution: {row['target']}")
        results.append(run_row(row["row_id"], manifest, model, artifact_dir, conversion_manifest=conversion_manifest, source_lock=source_lock))
    return {"state": "PASS", "rows": results, "count": len(results)}


def distribution_stats(values: Iterable[float | int]) -> dict[str, float | int]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        fail("cannot summarize an empty distribution")
    middle = median(ordered)
    deviations = [abs(value - middle) for value in ordered]
    return {
        "median": middle, "p10": percentile(ordered, 0.10), "p90": percentile(ordered, 0.90),
        "mad": median(deviations), "min": min(ordered), "max": max(ordered), "count": len(ordered),
    }


def percentile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def llama_metric_distributions(measured: list[Mapping[str, Any]]) -> dict[str, list[float | int]]:
    if len(measured) != 10:
        fail("llama metric summary requires exactly ten measured requests")
    distributions = {
        "ttft_ns": [sample["ttft_ns"] for sample in measured],
        "prefill_ns": [sample["prefill_ns"] for sample in measured],
        "prefill_tokens_per_second": [sample["prefill_tokens_per_second"] for sample in measured],
        # Phase 5 defines the row distribution over each request's median TPOT,
        # not over a flattened population of token-to-token intervals.
        "tpot_ns": [median(sample["tpot_ns"]) for sample in measured if sample["tpot_ns"]],
        "decode_tokens_per_second": [sample["decode_tokens_per_second"] for sample in measured if sample["decode_tokens_per_second"] is not None],
        "e2e_ns": [sample["e2e_ns"] for sample in measured],
    }
    for metric in ("tpot_ns", "decode_tokens_per_second"):
        if distributions[metric] and len(distributions[metric]) != 10:
            fail(f"llama {metric} distribution must contain one value per measured request")
    return distributions


def validate_direct_aggregate(path: Path, direct: Mapping[str, Any], direct_digest: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if path.name != "summary.json":
        fail("verified sLLM direct aggregate must use the current summary.json CLI layout")
    direct_contracts.verify_aggregate_bundle(path.parent, "sLLM direct aggregate")
    document, _, digest = read_json(path, "verified sLLM direct aggregate")
    if not isinstance(document, dict):
        fail("verified sLLM direct aggregate must be an object")
    verify_digest_sidecar(path, digest, "verified sLLM direct aggregate")
    validate_document_schema(
        document, DIRECT_AGGREGATE_SCHEMA_PATH, DIRECT_AGGREGATE_SCHEMA_SHA256,
        "sLLM direct aggregate",
    )
    if document["state"] != "PASS" or document["claims"] != {"baseline_only": True, "optimized": False, "faster": False, "hard_gate": False}:
        fail("sLLM direct aggregate is not a baseline-only PASS")
    matrix = document["matrix"]
    if Path(matrix["path"]).resolve() != DIRECT_MATRIX_PATH.resolve() or matrix["matrix_id"] != "engine-performance-direct-v1" or matrix["sha256"] != direct_digest:
        fail("sLLM direct aggregate is not bound to the exact Phase 5 direct matrix")
    expected_ids = [row["row_id"] for row in direct["rows"]]
    if document["expected_rows"] != expected_ids or [row["row_id"] for row in document["rows"]] != expected_ids:
        fail("sLLM direct aggregate rows are missing, duplicated, or reordered")
    graph = document["graph_csv"]
    graph_path = Path(graph["path"]).resolve()
    graph_digest = sha256_file(graph_path, "sLLM direct aggregate graph")
    if graph_digest != graph["sha256"] or graph_path.stat().st_size != graph["bytes"]:
        fail("sLLM direct aggregate graph digest or size is stale")
    verify_digest_sidecar(graph_path, graph_digest, "sLLM direct aggregate graph")
    identity = document["identity"]
    direct_rows = {row["row_id"]: row for row in direct["rows"]}
    for row in document["rows"]:
        expected = direct_rows[row["row_id"]]
        if any(row[key] != expected[key] for key in ("target", "model_size", "case_id", "input_tokens", "requested_output_tokens")):
            fail(f"sLLM direct aggregate row identity drifted: {row['row_id']}")
        if row["binary_sha256"] != identity["binary_sha256_by_target"][row["target"]] or row["model_lock_sha256"] != identity["model_lock_sha256_by_model"][row["model_size"]] or row["model_lock_fingerprint"] != identity["model_lock_fingerprint_by_model"][row["model_size"]]:
            fail(f"sLLM direct aggregate artifact identity drifted: {row['row_id']}")
    model_4b = next((model for model in direct["models"] if model["model_size"] == "4B"), None)
    if not isinstance(model_4b, dict) or model_4b.get("resolved_revision") != SOURCE_MODEL_REVISION or model_4b.get("lock_fingerprint") != SOURCE_LOCK_FINGERPRINT:
        fail("sLLM direct aggregate source model revision is not the exact Phase 5 4B lock")
    return document, digest


def comparison_metric_definitions() -> list[dict[str, Any]]:
    return [
        {"metric": "ttft_ns", "classification": "comparable", "sllm_metric": "ttft_ns", "llama_metric": "ttft_ns", "definition": "request start to first generated-token publication; exact prompt, batch, warmups, and ten requests match", "ratio": "llama_median_divided_by_sllm_median"},
        {"metric": "prefill_ns", "classification": "comparable", "sllm_metric": "prefill_ns", "llama_metric": "prefill_ns", "definition": "prefill submit to synchronized prefill completion; exact input token IDs match", "ratio": "llama_median_divided_by_sllm_median"},
        {"metric": "prefill_tokens_per_second", "classification": "comparable", "sllm_metric": "prefill_token_per_s", "llama_metric": "prefill_tokens_per_second", "definition": "exact input token count divided by the matching prefill interval", "ratio": "llama_median_divided_by_sllm_median"},
        {"metric": "tpot_ns", "classification": "context-only", "sllm_metric": "tpot_ns", "llama_metric": "tpot_ns", "definition": "ten per-request median adjacent-publication intervals when defined; the direct aggregate does not retain cross-engine realized output token identity/count", "ratio": None},
        {"metric": "decode_tokens_per_second", "classification": "context-only", "sllm_metric": "decode_token_per_s", "llama_metric": "decode_tokens_per_second", "definition": "generated tokens after the first divided by first-to-last publication window; realized output identity/count is not available in the direct aggregate", "ratio": None},
        {"metric": "e2e_ns", "classification": "context-only", "sllm_metric": "e2e_ns", "llama_metric": "e2e_ns", "definition": "request start through engine-specific cleanup; cleanup implementations and realized output identity/count are not cross-proven", "ratio": None},
        {"metric": "resident_vram_bytes", "classification": "context-only", "sllm_metric": "resident_vram_bytes", "llama_metric": None, "definition": "sLLM runtime allocator model-resident high-water; llama wrapper has no equivalent allocator measure", "ratio": None},
        {"metric": "peak_vram_bytes", "classification": "context-only", "sllm_metric": "peak_vram_bytes", "llama_metric": None, "definition": "sLLM runtime allocator peak; llama available-device-memory observations are not the same definition", "ratio": None},
        {"metric": "model_load_ns", "classification": "context-only", "sllm_metric": None, "llama_metric": "model_load_ns", "definition": "llama one-time model/context readiness interval; the direct aggregate does not publish a matching model-load distribution", "ratio": None},
        {"metric": "available_vram_mb", "classification": "context-only", "sllm_metric": None, "llama_metric": "available_vram_mb", "definition": "llama pre/post AMD SMI free-memory context; not resident or peak allocator VRAM", "ratio": None},
    ]


def cross_engine_rows(llama_rows: list[Mapping[str, Any]], direct_aggregate: Mapping[str, Any]) -> list[dict[str, Any]]:
    direct_rows = {
        (row["target"], row["case_id"]): row
        for row in direct_aggregate["rows"]
        if row["model_size"] == "4B"
    }
    if len(direct_rows) != 14 or len(llama_rows) != 14:
        fail("cross-engine comparison requires exactly fourteen 4B rows from each engine")
    definitions = comparison_metric_definitions()
    result: list[dict[str, Any]] = []
    for llama_row in llama_rows:
        key = (llama_row["target"], llama_row["case_id"])
        direct_row = direct_rows.get(key)
        if direct_row is None:
            fail(f"sLLM direct aggregate has no exact target/case row: {key}")
        if any(llama_row[field] != direct_row[field] for field in ("input_tokens", "requested_output_tokens")) or llama_row["sample_count"] != direct_row["sample_count"] or direct_row["warmup_count"] != 3:
            fail(f"cross-engine protocol or token-count mismatch: {key}")
        comparisons = []
        for definition in definitions:
            sllm_stats = direct_row["metrics"].get(definition["sllm_metric"]) if definition["sllm_metric"] else None
            llama_stats = llama_row["metrics"].get(definition["llama_metric"]) if definition["llama_metric"] else None
            ratio = None
            if definition["classification"] == "comparable" and sllm_stats is not None and llama_stats is not None:
                if sllm_stats["count"] != 10 or llama_stats["count"] != 10 or sllm_stats["median"] <= 0:
                    fail(f"comparable metric lacks ten positive per-request observations: {key} {definition['metric']}")
                ratio = float(llama_stats["median"]) / float(sllm_stats["median"])
            comparisons.append({
                "metric": definition["metric"], "classification": definition["classification"],
                "sllm_distribution": sllm_stats, "llama_distribution": llama_stats,
                "ratio": ratio,
            })
        result.append({
            "order": llama_row["order"], "target": llama_row["target"], "case_id": llama_row["case_id"],
            "input_tokens": llama_row["input_tokens"], "requested_output_tokens": llama_row["requested_output_tokens"],
            "warmup_count": 3, "sample_count": 10,
            "artifacts": {
                "sllm": {key: direct_row[key] for key in ("row_id", "manifest_sha256", "raw_result_sha256", "binary_sha256", "model_lock_sha256", "model_lock_fingerprint")},
                "llama": {key: llama_row[key] for key in ("row_id", "manifest_sha256", "raw_sha256", "binary_sha256", "build_manifest_sha256", "offload_evidence_sha256")},
            },
            "metrics": comparisons,
        })
    return result


def aggregate(artifact_dir: Path, sllm_aggregate: Path, output: Path) -> dict[str, Any]:
    global _MATRIX_CACHE
    matrix, matrix_digest, direct, direct_digest = load_matrix()
    _MATRIX_CACHE = matrix
    direct_aggregate, direct_aggregate_digest = validate_direct_aggregate(sllm_aggregate, direct, direct_digest)
    expected = row_map(matrix)
    artifact_root = artifact_dir.resolve()
    if artifact_dir.is_symlink() or not artifact_root.is_dir():
        fail("aggregate artifact directory is missing or is a symlink")
    actual_row_dirs = {entry.name for entry in artifact_root.iterdir() if entry.is_dir() and not entry.is_symlink()}
    if actual_row_dirs != set(expected):
        fail("aggregate artifact directory has missing, extra, or symlinked row directories")
    rows = []
    seen: set[str] = set()
    model_hash_cache: dict[str, str] = {}
    llama_binary_by_target: dict[str, str] = {}
    llama_build_manifest_by_target: dict[str, str] = {}
    for order, row in enumerate(matrix["rows"]):
        row_dir = artifact_dir.resolve() / row["row_id"]
        manifest_path = row_dir / "manifest.json"
        verify_completed_bundle(
            row_dir / BUNDLE_COMMIT_NAME,
            (row_dir / "raw-result.json", row_dir / "stderr.txt", manifest_path, manifest_path.with_suffix(".json.sha256")),
            "llama wrapper row",
        )
        manifest, _, manifest_digest = read_json(manifest_path, "row run manifest", 16 * 1024 * 1024)
        verify_digest_sidecar(manifest_path, manifest_digest, "row run manifest")
        schema_validate(manifest, "run_manifest", "row run manifest")
        if manifest["state"] != "PASS" or manifest["row_id"] != row["row_id"] or row["row_id"] in seen:
            fail("aggregate has a missing, duplicate, or failed row manifest")
        seen.add(row["row_id"])
        if manifest["matrix"]["sha256"] != matrix_digest or manifest["schema"]["sha256"] != sha256_file(SCHEMA_PATH, "llama Phase 5 schema") or manifest["source"] != {"commit": PINNED_COMMIT, "tree": PINNED_TREE, "direct_matrix_sha256": direct_digest}:
            fail("row manifest has stale matrix/schema/source identity")
        lock_identity = validate_source_lock(SOURCE_LOCK_PATH)
        if manifest["source_lock"] != lock_identity:
            fail("row manifest source-lock identity is stale")
        model_path = Path(manifest["model"]["path"]).resolve()
        if str(model_path) not in model_hash_cache:
            model_hash_cache[str(model_path)] = validate_model(model_path)
        if model_hash_cache[str(model_path)] != MODEL_SHA256:
            fail("aggregate model identity is stale")
        conversion_identity = validate_conversion_manifest(Path(manifest["conversion"]["path"]), lock_identity, model_path)
        if conversion_identity != manifest["conversion"]:
            fail("row manifest conversion identity is stale")
        build_manifest_path = Path(manifest["build_manifest"]["path"]).resolve()
        build_manifest = validate_build_manifest(build_manifest_path, row["target"])
        build_digest = sha256_file(build_manifest_path, "wrapper build manifest")
        if manifest["build_manifest"]["sha256"] != build_digest:
            fail("row manifest build-manifest digest is stale")
        if manifest["binary"] != {"path": str(Path(build_manifest["binary"]["path"]).resolve()), "sha256": build_manifest["binary"]["sha256"]}:
            fail("row manifest binary identity is stale")
        raw_path = Path(manifest["raw_result"]["path"]).resolve()
        raw, _, raw_digest = read_json(raw_path, "row raw result")
        if raw_digest != manifest["raw_result"]["sha256"] or raw_path.stat().st_size != manifest["raw_result"]["bytes"]:
            fail("row raw result was modified after execution")
        stderr_path = Path(manifest["stderr"]["path"]).resolve()
        if sha256_file(stderr_path, "row stderr") != EMPTY_SHA256 or stderr_path.stat().st_size != 0:
            fail("row stderr is not empty")
        row_tokens = direct_tokens(direct, case_map(matrix)[row["case_id"]]["direct_sequence_id"])
        validated = validate_result(raw, row, row_tokens, model_path)
        if manifest["offload_evidence"] != {"observation": raw["offload_evidence"], "sha256": validated["offload_sha256"]}:
            fail("row manifest GPU-offload evidence digest is stale")
        measured = validated["measured"]
        if manifest["conversion"]["manifest"]["output"]["sha256"] != MODEL_SHA256:
            fail("row manifest conversion output identity is stale")
        raw_distribution = llama_metric_distributions(measured)
        expected_decode_values = 0 if row["requested_output_tokens"] == 1 else 10
        if len(raw_distribution["tpot_ns"]) != expected_decode_values or len(raw_distribution["decode_tokens_per_second"]) != expected_decode_values:
            fail("llama TPOT/decode summary is missing a per-request value for the fixed output budget")
        distribution_digest = hashlib.sha256(canonical_bytes(raw_distribution)).hexdigest()
        model_load_values = [float(manifest["metrics"]["model_load_ns"])]
        vram_values = [float(value) for value in manifest["metrics"]["available_vram_mb"]]
        binary_sha = build_manifest["binary"]["sha256"]
        if row["target"] in llama_binary_by_target and llama_binary_by_target[row["target"]] != binary_sha:
            fail(f"mixed llama wrapper binary identity for {row['target']}")
        if row["target"] in llama_build_manifest_by_target and llama_build_manifest_by_target[row["target"]] != build_digest:
            fail(f"mixed llama wrapper build-manifest identity for {row['target']}")
        llama_binary_by_target[row["target"]] = binary_sha
        llama_build_manifest_by_target[row["target"]] = build_digest
        rows.append({
            "order": order, "row_id": row["row_id"], "target": row["target"], "case_id": row["case_id"], "input_tokens": row["input_tokens"], "requested_output_tokens": row["requested_output_tokens"], "sample_count": 10,
            "metrics": {key: distribution_stats(values) if values else None for key, values in raw_distribution.items()} | {"model_load_ns": distribution_stats(model_load_values), "available_vram_mb": distribution_stats(vram_values)},
            "manifest_sha256": manifest_digest, "raw_sha256": raw_digest,
            "binary_sha256": binary_sha, "build_manifest_sha256": build_digest,
            "offload_evidence_sha256": validated["offload_sha256"],
            "raw_distribution": {"sha256": distribution_digest, "metrics": raw_distribution},
        })
    if seen != set(expected):
        fail("aggregate row set is incomplete")
    comparison_rows = cross_engine_rows(rows, direct_aggregate)
    aggregate_doc = {
        "schema_version": "llama-phase5-v1", "record_kind": "aggregate", "state": "PASS",
        "claims": {"baseline_only": True, "optimized": False, "faster": False, "hard_gate": False},
        "inputs": {
            "sllm_aggregate": {"path": str(sllm_aggregate.resolve()), "sha256": direct_aggregate_digest, "schema_path": "ci/schema/engine-performance-aggregate-v1.schema.json", "schema_sha256": DIRECT_AGGREGATE_SCHEMA_SHA256},
            "llama_artifacts": {"path": str(artifact_root), "matrix_sha256": matrix_digest, "schema_sha256": sha256_file(SCHEMA_PATH, "llama Phase 5 schema")},
        },
        "model_source": {
            "repo_id": "Qwen/Qwen3.5-4B", "resolved_revision": SOURCE_MODEL_REVISION,
            "source_lock_sha256": SOURCE_LOCK_SHA256, "source_lock_fingerprint": SOURCE_LOCK_FINGERPRINT,
            "sllm_model_cache_sha256": direct_aggregate["identity"]["model_cache_sha256_by_model"]["4B"],
            "llama_gguf_sha256": MODEL_SHA256, "conversion_manifest_sha256": CONVERSION_SOURCE_MANIFEST_SHA256,
        },
        "engine_identities": {
            "sllm": {"source": direct_aggregate["identity"]["source"], "build_identity_by_target": direct_aggregate["identity"]["build_identity_by_target"], "matrix_sha256": direct_digest, "aggregate_sha256": direct_aggregate_digest, "binary_sha256_by_target": direct_aggregate["identity"]["binary_sha256_by_target"], "model_lock_sha256": direct_aggregate["identity"]["model_lock_sha256_by_model"]["4B"]},
            "llama": {"source_commit": PINNED_COMMIT, "source_tree": PINNED_TREE, "wrapper_source_sha256": WRAPPER_SOURCE_SHA256, "binary_sha256_by_target": {target: llama_binary_by_target[target] for target in TARGETS}, "build_manifest_sha256_by_target": {target: llama_build_manifest_by_target[target] for target in TARGETS}},
        },
        "gpu_tuples": [{key: target[key] for key in ("target", "backend", "gpu_uuid", "gpu_bdf", "product", "physical_hip_index", "logical_device_index")} | {"rocm_release": ROCM_VERSION} for target in matrix["targets"]],
        "metric_definitions": comparison_metric_definitions(),
        "rows": comparison_rows,
        "context": {
            "llama_rows": rows,
            "official_llama_bench": {"classification": "context-only", "ratio_comparable": False, "reason": "official llama-bench uses one warmup and random/zero-initialized tokens; it does not measure the exact-token wrapper protocol"},
        },
    }
    schema_validate(aggregate_doc, "aggregate", "llama wrapper aggregate")
    encoded = canonical_bytes(aggregate_doc)
    aggregate_path, digest = publish_aggregate_bundle(output, encoded)
    return {"state": "PASS", "aggregate_path": str(aggregate_path), "aggregate_sha256": digest, "row_count": len(rows)}


def official_commands(matrix: Mapping[str, Any], bench: Path, model: Path) -> list[tuple[str, list[str]]]:
    official = matrix["official_llama_bench"]
    templates: list[tuple[str, str]] = [("prompt_processing", official["commands"]["prompt_processing"]), ("decode", official["commands"]["decode"])]
    templates.extend(("paired", command) for command in official["commands"]["paired"])
    result: list[tuple[str, list[str]]] = []
    for kind, template in templates:
        rendered = template.replace("${LLAMA_BENCH}", str(bench)).replace("${MODEL}", str(model))
        command = shlex.split(rendered)
        # llama-bench treats -pg as additive to its default -p 512 / -n 128
        # tests.  Explicitly disable those defaults so each official paired
        # command publishes only its requested context row.
        if "-pg" in command:
            command[1:1] = ["-p", "0", "-n", "0"]
        if not command or Path(command[0]).resolve() != bench.resolve() or "-dev" not in command or command[command.index("-dev") + 1] != OFFICIAL_LLAMA_BENCH_DEVICE:
            fail("official llama-bench command is not bound to the exact ROCm0 device")
        result.append((kind, command))
    return result


def official_timeout_seconds(command: list[str]) -> int:
    """Use the canonical extended bound for commands that include 1024-token prefill."""
    for option in ("-p", "-pg"):
        try:
            value = command[command.index(option) + 1]
        except (ValueError, IndexError):
            continue
        if option == "-p" and "1024" in value.split(","):
            return PREFILL_LONG_TIMEOUT_SECONDS
        if option == "-pg" and value.split(",", 1)[0] == "1024":
            return PREFILL_LONG_TIMEOUT_SECONDS
    return DEFAULT_TIMEOUT_SECONDS


def _command_values(command: list[str], option: str) -> list[int]:
    try:
        raw = command[command.index(option) + 1]
    except (ValueError, IndexError):
        fail(f"official llama-bench command has no complete {option} option")
    try:
        values = [int(value) for value in raw.split(",")]
    except ValueError:
        fail(f"official llama-bench command has invalid {option} values")
    if not values or any(value < 0 for value in values):
        fail(f"official llama-bench command has invalid {option} values")
    return values


def validate_official_json(value: Any, command: list[str]) -> None:
    if "-pg" in command:
        pair = _command_values(command, "-pg")
        if len(pair) != 2:
            fail("official llama-bench -pg command does not contain one prompt/generation pair")
        if _command_values(command, "-p") != [0] or _command_values(command, "-n") != [0]:
            fail("official llama-bench -pg command did not disable default prompt/generation tests")
        expected = [(pair[0], pair[1])]
    else:
        prompts = _command_values(command, "-p")
        generations = _command_values(command, "-n")
        if generations == [0]:
            expected = [(prompt, 0) for prompt in prompts]
        elif prompts == [0]:
            expected = [(0, generation) for generation in generations]
        else:
            fail("official llama-bench command mixes unmatched prompt/generation lists")
    if not isinstance(value, list) or len(value) != len(expected):
        actual_pairs = [
            (record.get("n_prompt"), record.get("n_gen")) if isinstance(record, dict) else None
            for record in value
        ] if isinstance(value, list) else type(value).__name__
        fail(
            "official llama-bench JSON result count does not match the command: "
            f"expected={expected!r}, actual={actual_pairs!r}"
        )
    for index, (record, pair) in enumerate(zip(value, expected)):
        if not isinstance(record, dict) or (record.get("n_prompt"), record.get("n_gen")) != pair:
            fail(f"official llama-bench JSON row {index} does not match the requested token counts")
        if record.get("n_batch") != 2048 or record.get("n_ubatch") != 512 or record.get("main_gpu") != 0 or record.get("split_mode") != "none":
            fail(f"official llama-bench JSON row {index} does not match the fixed execution protocol")
        for field in ("samples_ns", "samples_ts"):
            samples = record.get(field)
            if not isinstance(samples, list) or len(samples) != 10 or any(
                not isinstance(sample, (int, float)) or isinstance(sample, bool) or not math.isfinite(sample) or sample <= 0
                for sample in samples
            ):
                fail(f"official llama-bench JSON row {index} has invalid {field}")
        for field in ("avg_ns", "avg_ts"):
            metric = record.get(field)
            if not isinstance(metric, (int, float)) or isinstance(metric, bool) or not math.isfinite(metric) or metric <= 0:
                fail(f"official llama-bench JSON row {index} has invalid {field}")


def run_official_context(
    target: str, llama_bench: Path, model: Path, conversion_manifest: Path,
    artifact_dir: Path, output: Path, *, source_lock: Path = SOURCE_LOCK_PATH,
) -> dict[str, Any]:
    global _MATRIX_CACHE
    matrix, matrix_digest, _direct, direct_digest = load_matrix()
    _MATRIX_CACHE = matrix
    device = expected_device(matrix, target)
    bench = validate_official_bench(target, llama_bench)
    model = model.resolve()
    validate_model(model)
    lock_identity = validate_source_lock(source_lock)
    conversion_identity = validate_conversion_manifest(conversion_manifest, lock_identity, model)
    source = reference_identity()
    artifact_root = artifact_dir.resolve()
    if artifact_root.exists() or artifact_root.is_symlink():
        fail(f"refusing to reuse official context artifact directory: {artifact_root}")
    context_path = output.resolve()
    context_sidecar = context_path.with_suffix(context_path.suffix + ".sha256")
    context_marker = context_path.with_suffix(context_path.suffix + ".complete.json")
    if any(path.exists() or path.is_symlink() for path in (context_path, context_sidecar, context_marker)):
        fail(f"refusing to overwrite official context artifact: {context_path}")
    tests: list[dict[str, Any]] = []
    evidence_records: list[dict[str, Any]] = []
    pending_files: list[tuple[Path, bytes, str]] = []
    for order, (kind, command) in enumerate(official_commands(matrix, bench, model)):
        raw_path = artifact_root / f"raw-{order:02d}-{kind}.json"
        stderr_path = artifact_root / f"stderr-{order:02d}-{kind}.txt"
        environment = os.environ.copy()
        for name in COMMON_VISIBILITY_NAMES:
            environment.pop(name, None)
        environment["ROCR_VISIBLE_DEVICES"] = device["gpu_uuid"]
        pre = direct_health.validate_observation(direct_health._amd_smi_observation(target, "pre"), target, "pre")
        pre_evidence = direct_health._amd_smi_phase_evidence(target, "pre")
        timeout_seconds = official_timeout_seconds(command)
        try:
            capture = run_wrapper_process(
                command, environment, timeout_seconds,
                monitor_provider=direct_health._amd_smi_monitor_sample, monitor_target=target,
            )
        finally:
            post = direct_health.validate_observation(direct_health._amd_smi_observation(target, "post"), target, "post")
            post_evidence = direct_health._amd_smi_phase_evidence(target, "post")
        raw_bytes, startup_stderr = require_successful_llama_output(
            capture, "official llama-bench", allow_stderr=True,
        )
        startup_stderr_identity = validate_official_startup_stderr(startup_stderr, device)
        stderr_bytes = b""
        if not direct_health._observations_have_stable_authorization(pre, post):
            fail("official llama-bench changed health")
        official_json = parse_json_bytes(raw_bytes, "official llama-bench JSON")
        raw_digest = hashlib.sha256(raw_bytes).hexdigest()
        validate_official_json(official_json, command)
        evidence = direct_health._build_evidence(pre_evidence, post_evidence, capture, target, {"path": direct_health.AMD_SMI_EXECUTABLE, **direct_health._amd_smi_version()})
        evidence_records.append(evidence)
        tests.append({
            "kind": kind, "order": order, "command": command,
            "raw_json": {"path": str(raw_path), "sha256": raw_digest, "bytes": len(raw_bytes)},
            "stderr": {"path": str(stderr_path), "sha256": EMPTY_SHA256, "bytes": 0},
            "execution": {"exit_code": 0, "timed_out": False, "timeout_seconds": timeout_seconds, "process_group_gone": True},
            "health_evidence": {
                "raw_json_type": type(official_json).__name__,
                "validated_startup_stderr": startup_stderr_identity,
                "evidence": evidence,
            },
        })
        pending_files.extend(((raw_path, raw_bytes, "official llama-bench JSON"), (stderr_path, stderr_bytes, "official llama-bench stderr")))
    context = {
        "schema_version": "llama-phase5-v1", "record_kind": "official_context", "state": "PASS",
        "matrix": {"path": "ci/matrix/llama-phase5-v1.json", "sha256": matrix_digest},
        "source": {"commit": PINNED_COMMIT, "tree": source["tree"], "direct_matrix_sha256": direct_digest},
        "target": {key: device[key] for key in ("target", "gpu_uuid", "gpu_bdf", "product", "logical_device_index")} | {"rocm_device": OFFICIAL_LLAMA_BENCH_DEVICE},
        "model": {"path": str(model), "sha256": MODEL_SHA256, "format": "GGUF", "dtype": "BF16"},
        "source_lock": lock_identity,
        "conversion": {"path": conversion_identity["path"], "sha256": conversion_identity["sha256"]},
        "build": {"binary": str(bench), "binary_sha256": sha256_file(bench, "llama-bench"), "source_commit": PINNED_COMMIT, "source_tree": source["tree"], "rocm_root": str(ROCM_ROOT), "target": target},
        "health_evidence": {"per_test": evidence_records}, "tests": tests,
        "metric_definitions": matrix["official_llama_bench"]["metric_definitions"],
        "comparison": {"context_only": True, "ratio_comparable": False, "reason": "official llama-bench uses one warmup and random/zero-initialized tokens; values are context-only and are not mixed into dedicated-wrapper ratios"},
        "cleanup": {"raw_outputs_preserved": True, "process_groups_gone": True, "stderr_empty": True},
    }
    schema_validate(context, "official_context", "official llama-bench context")
    encoded = canonical_bytes(context)
    payloads = {path: data for path, data, _label in pending_files}
    payloads[context_path] = encoded
    payloads[context_sidecar] = f"{hashlib.sha256(encoded).hexdigest()}  {context_path.name}\n".encode("ascii")
    publish_completed_bundle(payloads, context_marker, "official llama-bench context")
    verify_completed_bundle(context_marker, payloads, "official llama-bench context")
    return {"state": "PASS", "context_path": str(context_path), "context_sha256": hashlib.sha256(encoded).hexdigest(), "test_count": len(tests)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--contract-only", "--test-contract", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    modes.add_argument("--build-all", action="store_true")
    modes.add_argument("--run-row")
    modes.add_argument("--run-all", action="store_true")
    modes.add_argument("--aggregate", action="store_true")
    modes.add_argument("--official-context", action="store_true")
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--binary-manifest", type=Path)
    parser.add_argument("--gfx1030-build-manifest", type=Path)
    parser.add_argument("--gfx1201-build-manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK_PATH)
    parser.add_argument("--conversion-manifest", type=Path)
    parser.add_argument("--llama-bench", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--sllm-aggregate", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.contract_only:
            matrix, matrix_digest, direct, direct_digest = load_matrix()
            print(json.dumps({"state": "PASS", "matrix_id": matrix["matrix_id"], "matrix_sha256": matrix_digest, "direct_matrix_sha256": direct_digest, "source_direct_matrix_sha256": {"recorded": matrix["source_direct_matrix"]["sha256"], "actual": direct_digest, "pending": False}, "source_commit": PINNED_COMMIT, "sequence_lengths": [len(item["input_token_ids"]) for item in direct["token_sequences"]]}, sort_keys=True, separators=(",", ":")))
            return 0
        if args.build_only or args.build_all:
            output_dir = args.output_dir or Path("/tmp/sllm-phase5-p3-llama-wrapper")
            targets = TARGETS if args.build_all else ((args.target,) if args.target else ())
            if not targets:
                fail("--build-only requires --target; use --build-all for both targets")
            results = [build_one(target, output_dir) for target in targets]
            print(json.dumps({"state": "PASS", "builds": results}, sort_keys=True, separators=(",", ":")))
            return 0
        if args.run_row:
            if args.binary_manifest is None or args.model is None or args.artifact_dir is None:
                fail("--run-row requires --binary-manifest, --model, and --artifact-dir")
            print(json.dumps(run_row(args.run_row, args.binary_manifest, args.model, args.artifact_dir, conversion_manifest=args.conversion_manifest, source_lock=args.source_lock), sort_keys=True, separators=(",", ":")))
            return 0
        if args.run_all:
            if args.gfx1030_build_manifest is None or args.gfx1201_build_manifest is None or args.model is None or args.artifact_dir is None:
                fail("--run-all requires both target manifests, --model, and --artifact-dir")
            manifests = {"gfx1030": args.gfx1030_build_manifest, "gfx1201": args.gfx1201_build_manifest}
            if args.conversion_manifest is None:
                fail("--run-all requires --conversion-manifest")
            print(json.dumps(run_all(manifests, args.model, args.artifact_dir, conversion_manifest=args.conversion_manifest, source_lock=args.source_lock), sort_keys=True, separators=(",", ":")))
            return 0
        if args.aggregate:
            if args.artifact_dir is None or args.sllm_aggregate is None or args.output is None:
                fail("--aggregate requires --artifact-dir, --sllm-aggregate, and --output")
            print(json.dumps(aggregate(args.artifact_dir, args.sllm_aggregate, args.output), sort_keys=True, separators=(",", ":")))
            return 0
        if args.official_context:
            if args.target is None or args.llama_bench is None or args.model is None or args.conversion_manifest is None or args.artifact_dir is None or args.output is None:
                fail("--official-context requires --target, --llama-bench, --model, --conversion-manifest, --artifact-dir, and --output")
            print(json.dumps(run_official_context(args.target, args.llama_bench, args.model, args.conversion_manifest, args.artifact_dir, args.output, source_lock=args.source_lock), sort_keys=True, separators=(",", ":")))
            return 0
        fail("no operation selected")
    except (ContractError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"llama-phase5: FAIL: {exc}", file=sys.stderr)
        return 1


_MATRIX_CACHE: dict[str, Any] = {}


if __name__ == "__main__":
    raise SystemExit(main())
