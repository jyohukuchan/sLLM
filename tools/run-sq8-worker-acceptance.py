#!/usr/bin/env python3
"""Produce frozen standalone SQ8 worker acceptance evidence on the R9700."""

from __future__ import annotations

import argparse
import dataclasses
import errno
import fractions
import hashlib
import json
import math
import os
import queue
import re
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable, Iterable


RAW_SCHEMA = "ullm.sq8.worker_acceptance.raw.v2"
WORKER_SCHEMA = "ullm.worker.v1"
MAX_LINE_BYTES = 8 * 1024 * 1024
STDOUT_QUEUE_ITEMS = 2
COMMAND_TIMEOUT_SECONDS = 30.0
READY_TIMEOUT_NS = 600_000_000_000
REQUEST_TIMEOUT_NS = 180_000_000_000
PROGRESS_TIMEOUT_NS = 30_000_000_000
CANCEL_TIMEOUT_NS = 5_000_000_000
SHUTDOWN_TIMEOUT_NS = 30_000_000_000
PROCESS_GROUP_GRACE_SECONDS = 2.0
LATENCY_P95_MAX_NS = 2_000_000_000
IDLE_SETTLE_NS = 5_000_000_000
SAMPLE_INTERVAL_NS = 1_000_000_000
KFD_SNAPSHOT_DEADLINE_NS = 1_000_000_000
KFD_SNAPSHOT_RETRY_SLEEP_SECONDS = 0.005
THEIL_SEN_MAX_BYTES_PER_REQUEST = 262_144
FINAL_DELTA_MAX_BYTES = 67_108_864

GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545
EXPECTED_ARTIFACT_MANIFEST_SHA256 = (
    "23977f4e9bed4bac4cc64c177c35d7f83355861426bf32027a69cf7a241552e2"
)
EXPECTED_ARTIFACT_CONTENT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
EXPECTED_MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
REQUIRED_AMD_SMI_VERSION_PARTS = (
    "AMDSMI Tool: 26.2.2+e1a6bc5663",
    "AMDSMI Library version: 26.2.2",
    "ROCm version: 7.2.1",
)
REQUIRED_HIP_GUARDS = (
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
)
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": 20, "seed": 0}
EOS_TOKEN_IDS = [151_645, 151_643]
REQUEST_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
HEX40_RE = re.compile(r"[0-9a-f]{40}\Z")


class AcceptanceError(RuntimeError):
    """A fail-closed acceptance producer error."""


def fail(message: str) -> None:
    raise AcceptanceError(message)


def emit_progress(message: str) -> None:
    print(f"[sq8-acceptance] {message}", file=sys.stderr, flush=True)


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def integer(
    value: Any, label: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        fail(f"{label} is out of range")
    return value


def nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty string")
    return value


def duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_json_constant(value: str) -> None:
    fail(f"non-finite JSON constant: {value}")


def parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail(f"non-finite JSON number: {value}")
    return parsed


def strict_json_bytes(raw: bytes, label: str) -> Any:
    if len(raw) > MAX_LINE_BYTES:
        fail(f"{label} exceeds the 8 MiB limit")
    try:
        text = raw.decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=duplicate_rejecting_object,
            parse_float=parse_finite_float,
            parse_constant=reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        fail(f"{label} is not strict UTF-8 JSON: {error}")


def strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    value = strict_json_bytes(raw, label)
    if not isinstance(value, dict):
        fail(f"{label} root must be an object")
    return value


def json_type_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int/float coercions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            json_type_equal(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            json_type_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return left == right


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label} {path}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        fail(f"{label} is not executable")
    return path.resolve(strict=True)


def regular_directory(path: Path, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label} {path}: {error}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        fail(f"{label} must be a directory, not a symlink")
    return path.resolve(strict=True)


def prospective_output_directory(output_dir: Path, repo_root: Path) -> Path:
    """Resolve a fresh output path while rejecting repo placement and symlinks."""
    candidate = output_dir if output_dir.is_absolute() else Path.cwd() / output_dir
    if ".." in candidate.parts:
        fail("evidence output directory must not contain parent traversal")
    try:
        candidate.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        fail(f"failed to inspect evidence output directory {candidate}: {error}")
    else:
        fail("evidence output directory must be fresh")

    current = Path(candidate.anchor)
    for component in candidate.parent.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except OSError as error:
            fail(f"failed to resolve evidence output parent {current}: {error}")
        if stat.S_ISLNK(metadata.st_mode):
            fail(f"evidence output path contains a symlink: {current}")
        if not stat.S_ISDIR(metadata.st_mode):
            fail(f"evidence output parent is not a directory: {current}")

    try:
        canonical_parent = candidate.parent.resolve(strict=True)
        canonical_repo = repo_root.resolve(strict=True)
    except OSError as error:
        fail(f"failed to resolve repository/output boundary: {error}")
    prospective = canonical_parent / candidate.name
    if prospective == canonical_repo or canonical_repo in prospective.parents:
        fail("evidence output directory must be outside the source repository")
    return prospective


def open_directory_no_symlinks(directory: Path) -> int:
    if not directory.is_absolute():
        fail("secure directory traversal requires an absolute path")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(directory.anchor, flags)
    try:
        for component in directory.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def create_fresh_output_directory(output_dir: Path, repo_root: Path) -> tuple[Path, int]:
    prospective = prospective_output_directory(output_dir, repo_root)
    try:
        parent_descriptor = open_directory_no_symlinks(prospective.parent)
    except OSError as error:
        fail(f"failed to reopen evidence output parent without symlinks: {error}")
    output_descriptor = -1
    try:
        os.mkdir(prospective.name, mode=0o700, dir_fd=parent_descriptor)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        output_descriptor = os.open(
            prospective.name, flags, dir_fd=parent_descriptor
        )
    except OSError as error:
        fail(f"failed to create fresh evidence directory {prospective}: {error}")
    finally:
        os.close(parent_descriptor)
    return prospective, output_descriptor


def read_small_file(path: Path, label: str, maximum: int = MAX_LINE_BYTES) -> bytes:
    path = regular_file(path, label)
    try:
        size = path.stat().st_size
        if size > maximum:
            fail(f"{label} exceeds {maximum} bytes")
        return path.read_bytes()
    except OSError as error:
        fail(f"failed to read {label} {path}: {error}")


def prompt_sha256(prompt_tokens: int) -> str:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_tokens + 1):
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def percentile_linear(values: Iterable[int], probability: fractions.Fraction) -> fractions.Fraction:
    ordered = sorted(int(value) for value in values)
    if not ordered or probability < 0 or probability > 1:
        fail("invalid percentile input")
    rank = fractions.Fraction(len(ordered) - 1) * probability
    lower = rank.numerator // rank.denominator
    upper = math.ceil(rank)
    if lower == upper:
        return fractions.Fraction(ordered[lower])
    return fractions.Fraction(ordered[lower]) + (rank - lower) * (
        ordered[upper] - ordered[lower]
    )


def median(values: Iterable[int | fractions.Fraction]) -> fractions.Fraction:
    ordered = sorted(fractions.Fraction(value) for value in values)
    if not ordered:
        fail("cannot take the median of an empty series")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def theil_sen(values: list[int]) -> fractions.Fraction:
    if len(values) < 2:
        fail("Theil-Sen requires at least two points")
    slopes = [
        fractions.Fraction(values[j] - values[i], j - i)
        for i in range(len(values))
        for j in range(i + 1, len(values))
    ]
    return median(slopes)


def next_strict_timestamp(last: int) -> int:
    while True:
        value = time.monotonic_ns()
        if value > last:
            return value


class EvidenceWriter:
    """Stream raw records to an incomplete file and publish only on success."""

    def __init__(self, output_dir: Path, *, repo_root: Path | None = None):
        self._directory_fd = -1
        if repo_root is None:
            try:
                output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
                self.output_dir = output_dir.resolve(strict=True)
                self._directory_fd = open_directory_no_symlinks(self.output_dir)
            except OSError as error:
                fail(f"failed to create fresh evidence directory {output_dir}: {error}")
        else:
            self.output_dir, self._directory_fd = create_fresh_output_directory(
                output_dir, repo_root
            )
        self.raw_incomplete = self.output_dir / "raw.jsonl.incomplete"
        self.raw_final = self.output_dir / "raw.jsonl"
        self.stderr_incomplete = self.output_dir / "worker-stderr.jsonl.incomplete"
        self.stderr_final = self.output_dir / "worker-stderr.jsonl"
        try:
            descriptor = os.open(
                self.raw_incomplete.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory_fd,
            )
            self._raw = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        except OSError as error:
            os.close(self._directory_fd)
            self._directory_fd = -1
            fail(f"failed to create fresh evidence directory {output_dir}: {error}")
        self._closed = False

    def write(self, record: dict[str, Any]) -> None:
        if self._closed:
            fail("evidence writer is closed")
        if record.get("schema_version") != RAW_SCHEMA or "passed" in record:
            fail("producer attempted to write an invalid raw record")
        try:
            line = json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as error:
            fail(f"failed to encode raw record: {error}")
        if len(line.encode("utf-8")) > MAX_LINE_BYTES:
            fail("raw evidence line exceeds 8 MiB")
        try:
            self._raw.write(line)
            self._raw.write("\n")
            self._raw.flush()
        except OSError as error:
            fail(f"failed to stream raw evidence: {error}")

    def open_stderr(self) -> BinaryIO:
        try:
            descriptor = os.open(
                self.stderr_incomplete.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self._directory_fd,
            )
            return os.fdopen(descriptor, "wb", buffering=0)
        except OSError as error:
            fail(f"failed to create incomplete stderr evidence: {error}")

    def finish_stderr(self) -> str:
        try:
            descriptor = os.open(
                self.stderr_incomplete.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=self._directory_fd,
            )
            digest = hashlib.sha256()
            with os.fdopen(descriptor, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            os.rename(
                self.stderr_incomplete.name,
                self.stderr_final.name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
        except OSError as error:
            fail(f"failed to publish worker stderr: {error}")
        return digest.hexdigest()

    def _verify_output_path_identity(self) -> None:
        try:
            reopened = open_directory_no_symlinks(self.output_dir)
        except OSError as error:
            fail(f"failed to reopen evidence output path: {error}")
        try:
            expected = os.fstat(self._directory_fd)
            actual = os.fstat(reopened)
        finally:
            os.close(reopened)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            fail("evidence output path identity changed during the run")

    def publish(self) -> None:
        if self._closed:
            fail("evidence writer is already closed")
        renamed = False
        try:
            self._raw.flush()
            os.fsync(self._raw.fileno())
            self._raw.close()
            self._closed = True
            self._verify_output_path_identity()
            os.rename(
                self.raw_incomplete.name,
                self.raw_final.name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            renamed = True
            os.fsync(self._directory_fd)
            self._verify_output_path_identity()
            os.close(self._directory_fd)
            self._directory_fd = -1
        except BaseException as error:
            rollback_error: BaseException | None = None
            if renamed:
                try:
                    os.rename(
                        self.raw_final.name,
                        self.raw_incomplete.name,
                        src_dir_fd=self._directory_fd,
                        dst_dir_fd=self._directory_fd,
                    )
                    os.fsync(self._directory_fd)
                except BaseException as caught:
                    rollback_error = caught
            if self._directory_fd >= 0:
                try:
                    os.close(self._directory_fd)
                except OSError as caught:
                    if rollback_error is None:
                        rollback_error = caught
                self._directory_fd = -1
            detail = f"failed to publish raw evidence: {error}"
            if rollback_error is not None:
                detail += f"; rollback failed: {rollback_error}"
            if isinstance(error, (OSError, AcceptanceError)):
                fail(detail)
            if rollback_error is not None:
                error.add_note(detail)
            raise

    def abort(self) -> None:
        if not self._closed:
            try:
                self._raw.flush()
                os.fsync(self._raw.fileno())
                self._raw.close()
            except OSError:
                pass
            self._closed = True
        if self._directory_fd >= 0:
            os.close(self._directory_fd)
            self._directory_fd = -1


@dataclasses.dataclass(frozen=True)
class PumpEvent:
    observed_monotonic_ns: int
    event: dict[str, Any]
    raw_json: str
    raw_sha256: str


@dataclasses.dataclass(frozen=True)
class PumpEof:
    observed_monotonic_ns: int


@dataclasses.dataclass(frozen=True)
class PumpFailure:
    error: BaseException


PumpItem = PumpEvent | PumpEof | PumpFailure


def validate_request_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or REQUEST_ID_RE.fullmatch(value) is None:
        fail(f"{label} violates worker request ID syntax")
    return value


def validate_worker_event_shape(event: dict[str, Any]) -> None:
    event_type = event.get("type")
    if event.get("schema_version") != WORKER_SCHEMA or not isinstance(event_type, str):
        fail("worker event discriminator differs")
    if event_type == "ready":
        exact_keys(
            event,
            {
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
            },
            "ready event",
        )
    elif event_type == "started":
        exact_keys(event, {"schema_version", "type", "request_id", "prompt_tokens"}, "started event")
        validate_request_id(event["request_id"], "started.request_id")
        integer(event["prompt_tokens"], "started.prompt_tokens", minimum=1, maximum=4096)
    elif event_type == "progress":
        exact_keys(
            event,
            {"schema_version", "type", "request_id", "phase", "processed_prompt_tokens"},
            "progress event",
        )
        validate_request_id(event["request_id"], "progress.request_id")
        if event["phase"] != "prefill":
            fail("progress phase must be prefill")
        integer(event["processed_prompt_tokens"], "progress.processed_prompt_tokens", minimum=1, maximum=4096)
    elif event_type == "token":
        exact_keys(event, {"schema_version", "type", "request_id", "index", "token_id"}, "token event")
        validate_request_id(event["request_id"], "token.request_id")
        integer(event["index"], "token.index", maximum=511)
        integer(event["token_id"], "token.token_id", maximum=151_935)
    elif event_type == "released":
        expected = {
            "schema_version",
            "type",
            "request_id",
            "outcome",
            "prompt_tokens",
            "completion_tokens",
            "reset_complete",
        }
        if event.get("outcome") == "cancelled":
            expected.add("cancel_reason")
        exact_keys(event, expected, "released event")
        validate_request_id(event["request_id"], "released.request_id")
        if event["outcome"] not in {"stop", "length", "cancelled"}:
            fail("released outcome differs")
        integer(event["prompt_tokens"], "released.prompt_tokens", minimum=1, maximum=4096)
        integer(event["completion_tokens"], "released.completion_tokens", maximum=512)
        if event["reset_complete"] is not True:
            fail("released reset_complete must be true")
    elif event_type == "error":
        exact_keys(
            event,
            {"schema_version", "type", "request_id", "code", "recoverable", "message"},
            "error event",
        )
        if event["request_id"] is not None:
            validate_request_id(event["request_id"], "error.request_id")
        nonempty_string(event["code"], "error.code")
        if not isinstance(event["recoverable"], bool):
            fail("error.recoverable must be boolean")
        message = nonempty_string(event["message"], "error.message")
        if len(message.encode("utf-8")) > 1024 or any(ord(character) < 32 for character in message):
            fail("error.message violates its bound")
    else:
        fail(f"unknown worker event type: {event_type!r}")


class StdoutPump:
    def __init__(self, stream: BinaryIO):
        self.stream = stream
        self.items: queue.Queue[PumpItem] = queue.Queue(maxsize=STDOUT_QUEUE_ITEMS)
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sq8-acceptance-stdout", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _deliver(self, item: PumpItem) -> None:
        while not self.stop.is_set():
            try:
                self.items.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _run(self) -> None:
        last_observed = -1
        try:
            while not self.stop.is_set():
                raw = self.stream.readline(MAX_LINE_BYTES + 2)
                if raw == b"":
                    self._deliver(PumpEof(time.monotonic_ns()))
                    return
                if not raw.endswith(b"\n"):
                    if len(raw) > MAX_LINE_BYTES:
                        fail("worker stdout line exceeds 8 MiB")
                    fail("worker stdout ended with a partial line")
                payload = raw[:-1]
                if len(payload) > MAX_LINE_BYTES:
                    fail("worker stdout line exceeds 8 MiB")
                event = strict_json_object(payload, "worker stdout line")
                validate_worker_event_shape(event)
                observed = next_strict_timestamp(last_observed)
                last_observed = observed
                self._deliver(
                    PumpEvent(
                        observed,
                        event,
                        payload.decode("utf-8"),
                        sha256_bytes(payload),
                    )
                )
        except BaseException as error:
            self._deliver(PumpFailure(error))

    def receive(self, deadline_ns: int) -> PumpEvent | PumpEof:
        try:
            item = self.items.get_nowait()
        except queue.Empty:
            remaining = deadline_ns - time.monotonic_ns()
            if remaining <= 0:
                fail("worker event deadline expired")
            try:
                item = self.items.get(timeout=remaining / 1_000_000_000)
            except queue.Empty:
                fail("worker event deadline expired")
        if isinstance(item, PumpFailure):
            if isinstance(item.error, AcceptanceError):
                raise item.error
            fail(f"worker stdout pump failed: {item.error}")
        if item.observed_monotonic_ns > deadline_ns:
            fail("worker event deadline expired")
        return item

    def close(self) -> None:
        self.stop.set()
        try:
            self.stream.close()
        except OSError:
            pass
        self.thread.join(timeout=2.0)


class StderrDrain:
    def __init__(self, stream: BinaryIO, output: BinaryIO):
        self.stream = stream
        self.output = output
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="sq8-acceptance-stderr", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while chunk := self.stream.read(64 * 1024):
                self.output.write(chunk)
        except BaseException as error:
            self.error = error
        finally:
            try:
                self.stream.close()
                self.output.flush()
                os.fsync(self.output.fileno())
                self.output.close()
            except BaseException as error:
                if self.error is None:
                    self.error = error

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout=timeout)
        if self.thread.is_alive():
            fail("worker stderr drain did not terminate")
        if self.error is not None:
            fail(f"worker stderr drain failed: {self.error}")


def terminate_process_group(
    process: subprocess.Popen[Any],
    process_group: int,
    *,
    grace_seconds: float = PROCESS_GROUP_GRACE_SECONDS,
) -> None:
    if process_group <= 0 or process_group != process.pid:
        fail("child process group identity differs from its session leader")
    if not math.isfinite(grace_seconds) or grace_seconds <= 0:
        fail("child process group grace period must be positive")

    def group_exists() -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError as error:
            fail(f"cannot inspect child process group: {error}")
        except OSError as error:
            fail(f"failed to inspect child process group: {error}")
        return True

    def signal_group(signum: signal.Signals) -> None:
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            pass
        except OSError as error:
            fail(f"failed to signal child process group with {signum.name}: {error}")

    def group_and_leader_gone(deadline_ns: int) -> bool:
        while True:
            leader_gone = process.poll() is not None
            group_gone = not group_exists()
            if leader_gone and group_gone:
                try:
                    process.wait(timeout=0)
                except subprocess.TimeoutExpired:
                    pass
                else:
                    return True
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return False
            time.sleep(min(0.01, remaining_ns / 1_000_000_000))

    grace_ns = int(grace_seconds * 1_000_000_000)
    signal_group(signal.SIGTERM)
    if group_and_leader_gone(time.monotonic_ns() + grace_ns):
        return
    signal_group(signal.SIGKILL)
    if group_and_leader_gone(time.monotonic_ns() + grace_ns):
        return
    fail("child process group or session leader remained after SIGKILL")


def run_bounded_command(
    arguments: list[str],
    label: str,
    *,
    cwd: Path | None = None,
) -> bytes:
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        try:
            process = subprocess.Popen(
                arguments,
                cwd=cwd,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=COMMAND_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                terminate_process_group(process, process.pid)
                fail(f"{label} timed out")
            stdout_file.seek(0)
            raw = stdout_file.read(MAX_LINE_BYTES + 1)
            stderr_file.seek(0)
            diagnostic = stderr_file.read(64 * 1024).decode("utf-8", errors="replace")
        except OSError as error:
            fail(f"failed to execute {label}: {error}")
    if code != 0:
        fail(f"{label} exited {code}: {diagnostic.strip()}")
    if len(raw) > MAX_LINE_BYTES:
        fail(f"{label} output exceeds 8 MiB")
    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        fail(f"{label} output is not UTF-8: {error}")
    return raw


CommandRunner = Callable[[list[str], str], bytes]


def parse_amd_smi_list(raw: bytes) -> None:
    document = strict_json_bytes(raw, "amd-smi list output")
    if not isinstance(document, list):
        fail("amd-smi list root must be an array")
    index_matches = [
        item
        for item in document
        if isinstance(item, dict) and item.get("gpu") == GPU_INDEX
    ]
    if len(index_matches) != 1 or any(
        index_matches[0].get(key) != expected
        for key, expected in {
            "gpu": GPU_INDEX,
            "bdf": GPU_BDF,
            "uuid": GPU_UUID,
            "kfd_id": KFD_GPU_ID,
        }.items()
    ):
        fail("amd-smi list does not contain one unique matching GPU index 2")


def parse_amd_process(raw: bytes, worker_pid: int) -> int:
    document = strict_json_bytes(raw, "amd-smi process output")
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        fail("amd-smi process output must contain one GPU object")
    gpu = document[0]
    if gpu.get("gpu") != GPU_INDEX:
        fail("amd-smi process GPU index differs")
    processes = gpu.get("process_list")
    if not isinstance(processes, list) or len(processes) != 1 or not isinstance(processes[0], dict):
        fail("amd-smi process must contain exactly one real process")
    info = processes[0].get("process_info")
    if not isinstance(info, dict) or info.get("pid") != worker_pid:
        fail("amd-smi process PID differs from worker")
    memory = info.get("mem_usage")
    if not isinstance(memory, dict) or memory.get("unit") != "B":
        fail("amd-smi process memory unit must be B")
    return integer(memory.get("value"), "amd-smi mem_usage.value", minimum=1)


def capture_gpu_metric(amd_smi: str, runner: CommandRunner, boundary: str) -> dict[str, Any]:
    raw = runner([amd_smi, "metric", "--gpu", str(GPU_INDEX), "--json"], "amd-smi metric")
    document = strict_json_object(raw, "amd-smi metric output")
    gpu_data = document.get("gpu_data")
    if (
        not isinstance(gpu_data, list)
        or len(gpu_data) != 1
        or not isinstance(gpu_data[0], dict)
        or gpu_data[0].get("gpu") != GPU_INDEX
    ):
        fail("amd-smi metric output must contain exactly GPU 2")
    captured = time.monotonic_ns()
    return {
        "schema_version": RAW_SCHEMA,
        "record_type": "gpu_metric",
        "boundary": boundary,
        "captured_monotonic_ns": captured,
        "raw_json": raw.decode("utf-8"),
        "raw_sha256": sha256_bytes(raw),
    }


def parse_proc_stat(raw: str, expected_pid: int) -> tuple[int, int]:
    prefix = f"{expected_pid} ("
    if not raw.startswith(prefix):
        fail("/proc stat PID prefix differs")
    candidates = list(re.finditer(r"\) ([A-Za-z]) ", raw))
    for match in reversed(candidates):
        fields = [match.group(1), *raw[match.end() :].strip().split()]
        if len(fields) >= 20:
            try:
                ppid = int(fields[1], 10)
                starttime = int(fields[19], 10)
            except ValueError:
                continue
            if ppid >= 0 and starttime >= 0:
                return ppid, starttime
    fail("/proc stat lacks a valid rightmost comm delimiter")


def read_proc_raw(path: Path, label: str, maximum_bytes: int = 1024 * 1024) -> bytes:
    try:
        with path.open("rb") as handle:
            raw = handle.read(maximum_bytes + 1)
    except OSError as error:
        fail(f"failed to read {label}: {error}")
    if len(raw) > maximum_bytes:
        fail(f"{label} exceeds {maximum_bytes} bytes")
    return raw


def decode_proc_raw(raw: bytes, label: str, encoding: str = "utf-8") -> str:
    try:
        return raw.decode(encoding, errors="strict")
    except UnicodeError as error:
        fail(f"{label} is not valid {encoding}: {error}")


def read_proc_identity_raw(proc_root: Path, pid: int) -> tuple[bytes, str, int, int]:
    raw_bytes = read_proc_raw(proc_root / str(pid) / "stat", f"/proc/{pid}/stat")
    raw = decode_proc_raw(raw_bytes, f"/proc/{pid}/stat")
    ppid, starttime = parse_proc_stat(raw, pid)
    if ppid <= 0 or starttime <= 0:
        fail(f"/proc/{pid}/stat contains a non-positive identity field")
    return raw_bytes, raw, ppid, starttime


def read_proc_identity(proc_root: Path, pid: int) -> tuple[int, int]:
    _, _, ppid, starttime = read_proc_identity_raw(proc_root, pid)
    return ppid, starttime


def parse_proc_status(raw: str) -> tuple[int, int]:
    vmrss: int | None = None
    threads: int | None = None
    for line in raw.splitlines():
        if line.startswith("VmRSS:"):
            match = re.fullmatch(r"VmRSS:\s+([0-9]+) kB", line)
            if match is None or vmrss is not None:
                fail("/proc status VmRSS is malformed or repeated")
            vmrss = int(match.group(1), 10)
        elif line.startswith("Threads:"):
            match = re.fullmatch(r"Threads:\s+([0-9]+)", line)
            if match is None or threads is not None:
                fail("/proc status Threads is malformed or repeated")
            threads = int(match.group(1), 10)
    if vmrss is None or threads is None or threads < 1:
        fail("/proc status lacks VmRSS or Threads")
    if vmrss > ((1 << 63) - 1) // 1024:
        fail("VmRSS byte conversion overflows")
    return vmrss, threads


def capture_worker_proc_with_probe(
    proc_root: Path,
    pid: int,
    probe: Callable[[], Any],
) -> tuple[dict[str, Any], Any]:
    stat_before_bytes, stat_before, ppid_before, start_before = (
        read_proc_identity_raw(proc_root, pid)
    )
    process_dir = proc_root / str(pid)
    try:
        status_bytes = read_proc_raw(process_dir / "status", f"/proc/{pid}/status")
        status_raw = decode_proc_raw(status_bytes, f"/proc/{pid}/status")
        exe_target = os.readlink(process_dir / "exe")
        with os.scandir(process_dir / "fd") as entries:
            fd_names = [entry.name for entry in entries]
        children_bytes = read_proc_raw(
            process_dir / "task" / str(pid) / "children",
            f"/proc/{pid}/task/{pid}/children",
        )
        children_raw = decode_proc_raw(
            children_bytes, f"/proc/{pid}/task/{pid}/children", "ascii"
        )
    except (OSError, UnicodeError) as error:
        fail(f"failed to collect worker /proc diagnostics: {error}")
    if any(not name.isascii() or not name.isdecimal() for name in fd_names):
        fail("worker FD directory contains a non-decimal entry")
    fd_names.sort(key=lambda name: int(name, 10))
    vmrss_kb, threads = parse_proc_status(status_raw)
    try:
        children = [int(value, 10) for value in children_raw.split()]
    except ValueError:
        fail("worker children list is malformed")
    if children != sorted(set(children)) or any(child <= 0 for child in children):
        fail("worker children must be ascending unique positive PIDs")
    probe_result = probe()
    stat_after_bytes, stat_after, ppid_after, start_after = read_proc_identity_raw(
        proc_root, pid
    )
    if (ppid_before, start_before) != (ppid_after, start_after):
        fail("worker identity changed during /proc sampling")
    return {
        "pid": pid,
        "ppid": ppid_before,
        "exe": exe_target,
        "starttime_ticks_before": start_before,
        "starttime_ticks_after": start_after,
        "vmrss_kb": vmrss_kb,
        "vmrss_bytes": vmrss_kb * 1024,
        "threads": threads,
        "fd_count": len(fd_names),
        "children": children,
        "stat_before_raw": stat_before,
        "stat_before_raw_sha256": sha256_bytes(stat_before_bytes),
        "status_raw": status_raw,
        "status_raw_sha256": sha256_bytes(status_bytes),
        "exe_target": exe_target,
        "fd_names": fd_names,
        "children_raw": children_raw,
        "children_raw_sha256": sha256_bytes(children_bytes),
        "stat_after_raw": stat_after,
        "stat_after_raw_sha256": sha256_bytes(stat_after_bytes),
    }, probe_result


def capture_worker_proc(proc_root: Path, pid: int) -> dict[str, Any]:
    worker, _ = capture_worker_proc_with_probe(proc_root, pid, lambda: None)
    return worker


class KfdSnapshotUnstable(RuntimeError):
    def __init__(self, reason: str, stage: str, pid: int | None):
        super().__init__(f"{reason}:{stage}:{pid if pid is not None else 'null'}")
        self.reason = reason
        self.stage = stage
        self.pid = pid


def _enumerate_kfd_pid_names(root_descriptor: int, stage: str) -> tuple[str, ...]:
    try:
        names = os.listdir(root_descriptor)
    except OSError as error:
        fail(f"failed to enumerate KFD processes: {error}")
    numeric: list[tuple[int, str]] = []
    for name in names:
        if not name.isascii() or not name.isdecimal():
            continue
        pid = int(name, 10)
        if pid <= 0:
            fail(f"KFD process PID must be positive: {name}")
        if name != str(pid):
            fail(f"KFD process PID must use canonical decimal syntax: {name}")
        numeric.append((pid, name))
    numeric.sort()
    return tuple(name for _, name in numeric)


def _require_kfd_worker_pid(
    pid_names: tuple[str, ...], expected_worker_pid: int | None, stage: str
) -> None:
    if expected_worker_pid is not None and str(expected_worker_pid) not in pid_names:
        fail(f"required worker PID {expected_worker_pid} is missing from KFD {stage} set")


def _open_kfd_pid_directories(
    root_descriptor: int,
    pid_names: tuple[str, ...],
    expected_worker_pid: int | None,
    stage: str,
    opened: dict[str, tuple[int, int, int]],
) -> None:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    for pid_name in pid_names:
        pid = int(pid_name, 10)
        try:
            descriptor = os.open(pid_name, directory_flags, dir_fd=root_descriptor)
        except OSError as error:
            if error.errno == errno.ENOENT:
                if pid == expected_worker_pid:
                    fail(f"required worker PID {pid} disappeared from KFD {stage} set")
                raise KfdSnapshotUnstable("entry_disappeared", stage, pid) from error
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                fail(f"KFD process entry is not a real directory: {pid_name}")
            fail(f"failed to open KFD process PID {pid}: {error}")
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            os.close(descriptor)
            fail(f"failed to inspect KFD process PID {pid}: {error}")
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            fail(f"KFD process entry is not a real directory: {pid_name}")
        opened[pid_name] = (descriptor, metadata.st_dev, metadata.st_ino)


def _kfd_identities(
    opened: dict[str, tuple[int, int, int]]
) -> list[dict[str, int]]:
    return [
        {"pid": int(name, 10), "st_dev": item[1], "st_ino": item[2]}
        for name, item in opened.items()
    ]


def _close_kfd_pid_directories(
    opened: dict[str, tuple[int, int, int]]
) -> list[OSError]:
    errors: list[OSError] = []
    while opened:
        _, item = opened.popitem()
        try:
            os.close(item[0])
        except OSError as error:
            errors.append(error)
    return errors


def _read_kfd_vram(
    process_descriptor: int,
    pid: int,
    st_dev: int,
    st_ino: int,
    expected_worker_pid: int | None,
) -> dict[str, Any]:
    file_descriptor = -1
    try:
        try:
            file_descriptor = os.open(
                f"vram_{KFD_GPU_ID}",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=process_descriptor,
            )
        except OSError as error:
            if error.errno == errno.ELOOP:
                fail(f"KFD VRAM for PID {pid} must not be a symbolic link")
            if error.errno == errno.ENOENT:
                if pid == expected_worker_pid:
                    fail(f"required worker KFD VRAM file is missing for PID {pid}")
                raise KfdSnapshotUnstable("entry_disappeared", "read", pid) from error
            fail(f"failed to open KFD VRAM for PID {pid}: {error}")
        try:
            metadata = os.fstat(file_descriptor)
        except OSError as error:
            fail(f"failed to inspect KFD VRAM for PID {pid}: {error}")
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"KFD VRAM for PID {pid} is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(file_descriptor, min(4096, 4097 - total))
            except OSError as error:
                if error.errno == errno.ENOENT:
                    if pid == expected_worker_pid:
                        fail(f"required worker KFD VRAM read disappeared for PID {pid}")
                    if chunks:
                        fail(
                            "KFD VRAM read disappeared after partial data for PID "
                            f"{pid}"
                        )
                    raise KfdSnapshotUnstable(
                        "entry_disappeared", "read", pid
                    ) from error
                fail(f"failed to read KFD VRAM for PID {pid}: {error}")
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 4096:
                fail(f"KFD VRAM for PID {pid} exceeds 4096 bytes")
        raw_bytes = b"".join(chunks)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
    try:
        raw = raw_bytes.decode("ascii", errors="strict").strip()
    except UnicodeError as error:
        fail(f"KFD VRAM for PID {pid} is not ASCII: {error}")
    if not raw.isdecimal():
        fail(f"KFD VRAM for PID {pid} is not a non-negative integer")
    return {
        "pid": pid,
        "st_dev": st_dev,
        "st_ino": st_ino,
        "vram_raw": raw,
        "vram_bytes": int(raw, 10),
    }


def _capture_kfd_attempt(
    root_descriptor: int,
    expected_worker_pid: int | None,
    attempt_index: int,
    started_ns: int,
) -> tuple[dict[str, Any], bool]:
    pids_before: tuple[str, ...] = ()
    pids_after: tuple[str, ...] | None = None
    before_opened: dict[str, tuple[int, int, int]] = {}
    after_opened: dict[str, tuple[int, int, int]] = {}
    processes: list[dict[str, Any]] = []
    unstable: KfdSnapshotUnstable | None = None
    stable = False
    try:
        try:
            pids_before = _enumerate_kfd_pid_names(root_descriptor, "before")
            _require_kfd_worker_pid(pids_before, expected_worker_pid, "before")
            _open_kfd_pid_directories(
                root_descriptor,
                pids_before,
                expected_worker_pid,
                "before",
                before_opened,
            )
            for pid_name in pids_before:
                descriptor, st_dev, st_ino = before_opened[pid_name]
                process = _read_kfd_vram(
                    descriptor,
                    int(pid_name, 10),
                    st_dev,
                    st_ino,
                    expected_worker_pid,
                )
                processes.append(process)
                if (
                    process["vram_bytes"] > 0
                    and process["pid"] != expected_worker_pid
                ):
                    fail(
                        f"unexpected positive KFD owner observed: {process['pid']}"
                    )

            pids_after = _enumerate_kfd_pid_names(root_descriptor, "after")
            _require_kfd_worker_pid(pids_after, expected_worker_pid, "after")
            _open_kfd_pid_directories(
                root_descriptor,
                pids_after,
                expected_worker_pid,
                "after",
                after_opened,
            )
            if pids_before != pids_after:
                raise KfdSnapshotUnstable("pid_set_changed", "after", None)
            stable = True
        except KfdSnapshotUnstable as error:
            unstable = error

        for pid_name in set(before_opened) & set(after_opened):
            if before_opened[pid_name][1:] != after_opened[pid_name][1:]:
                fail(f"KFD PID {pid_name} directory identity changed during snapshot")

        before_identities = _kfd_identities(before_opened)
        after_identities = _kfd_identities(after_opened)
    finally:
        close_errors = _close_kfd_pid_directories(after_opened)
        close_errors.extend(_close_kfd_pid_directories(before_opened))
        if close_errors:
            fail(f"failed to close KFD process directory: {close_errors[0]}")
    completed_ns = next_strict_timestamp(started_ns)
    attempt = {
        "attempt_index": attempt_index,
        "started_monotonic_ns": started_ns,
        "completed_monotonic_ns": completed_ns,
        "outcome": "stable" if stable else "retry",
        "retry_reason": None if unstable is None else unstable.reason,
        "retry_stage": None if unstable is None else unstable.stage,
        "retry_pid": None if unstable is None else unstable.pid,
        "pids_before": [int(name, 10) for name in pids_before],
        "before_identities": before_identities,
        "processes": processes,
        "pids_after": (
            None if pids_after is None else [int(name, 10) for name in pids_after]
        ),
        "after_identities": after_identities,
    }
    return attempt, stable


def capture_kfd_snapshot(
    kfd_proc_root: Path,
    expected_worker_pid: int | None,
    *,
    not_before_ns: int | None = None,
) -> dict[str, Any]:
    if expected_worker_pid is not None:
        integer(expected_worker_pid, "expected KFD worker PID", minimum=1)
    if not_before_ns is not None:
        integer(not_before_ns, "KFD acquisition ordering floor", minimum=0)
    try:
        root_descriptor = os.open(
            kfd_proc_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as error:
        fail(f"failed to open KFD process root: {error}")
    try:
        root_metadata = os.fstat(root_descriptor)
    except OSError as error:
        os.close(root_descriptor)
        fail(f"failed to inspect KFD process root: {error}")
    if not stat.S_ISDIR(root_metadata.st_mode):
        os.close(root_descriptor)
        fail("KFD process root is not a directory")
    acquisition_started_ns = (
        time.monotonic_ns()
        if not_before_ns is None
        else next_strict_timestamp(not_before_ns)
    )
    deadline_ns = acquisition_started_ns + KFD_SNAPSHOT_DEADLINE_NS
    attempts: list[dict[str, Any]] = []
    attempt_started_ns = acquisition_started_ns
    try:
        while True:
            attempt, stable = _capture_kfd_attempt(
                root_descriptor,
                expected_worker_pid,
                len(attempts),
                attempt_started_ns,
            )
            attempts.append(attempt)
            if attempt["completed_monotonic_ns"] > deadline_ns:
                fail("stable KFD snapshot acquisition exceeded one second")
            if stable:
                return {
                    "acquisition_started_monotonic_ns": acquisition_started_ns,
                    "acquisition_completed_monotonic_ns": attempt[
                        "completed_monotonic_ns"
                    ],
                    "deadline_monotonic_ns": deadline_ns,
                    "attempt_count": len(attempts),
                    "retry_reasons": [
                        f"{item['retry_reason']}:{item['retry_stage']}:"
                        f"{item['retry_pid'] if item['retry_pid'] is not None else 'null'}"
                        for item in attempts[:-1]
                    ],
                    "attempts": attempts,
                    "before_identities": attempt["before_identities"],
                    "processes": attempt["processes"],
                    "after_identities": attempt["after_identities"],
                }
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                fail("stable KFD snapshot acquisition exceeded one second")
            time.sleep(
                min(KFD_SNAPSHOT_RETRY_SLEEP_SECONDS, remaining_ns / 1_000_000_000)
            )
            attempt_started_ns = next_strict_timestamp(
                attempt["completed_monotonic_ns"]
            )
            if attempt_started_ns > deadline_ns:
                fail("stable KFD snapshot acquisition exceeded one second")
    finally:
        os.close(root_descriptor)


def positive_kfd_processes(
    processes: list[dict[str, Any]],
) -> list[dict[str, int]]:
    return [
        {"pid": item["pid"], "vram_bytes": item["vram_bytes"]}
        for item in processes
        if item["vram_bytes"] > 0
    ]


def scan_kfd_positive(
    kfd_proc_root: Path, expected_worker_pid: int | None
) -> list[dict[str, int]]:
    return positive_kfd_processes(
        capture_kfd_snapshot(kfd_proc_root, expected_worker_pid)["processes"]
    )


def require_isolated_kfd_processes(
    processes: list[dict[str, Any]], worker_pid: int | None
) -> None:
    positive = positive_kfd_processes(processes)
    expected = [] if worker_pid is None else [worker_pid]
    if [item["pid"] for item in positive] != expected:
        fail("R9700 KFD ownership is not isolated")


def require_isolated_kfd(kfd_proc_root: Path, worker_pid: int | None) -> None:
    snapshot = capture_kfd_snapshot(kfd_proc_root, worker_pid)
    require_isolated_kfd_processes(snapshot["processes"], worker_pid)


@dataclasses.dataclass(frozen=True)
class ProbeContext:
    amd_smi: str
    proc_root: Path = Path("/proc")
    kfd_proc_root: Path = Path("/sys/class/kfd/kfd/proc")
    command_runner: CommandRunner = run_bounded_command


def capture_resource_sample(context: ProbeContext, worker_pid: int) -> tuple[dict[str, Any], int, int]:
    def probe() -> tuple[bytes, int, dict[str, Any]]:
        process_raw = context.command_runner(
            [
                context.amd_smi,
                "process",
                "--gpu",
                str(GPU_INDEX),
                "--general",
                "--json",
            ],
            "amd-smi process",
        )
        vram = parse_amd_process(process_raw, worker_pid)
        return process_raw, vram, capture_kfd_snapshot(
            context.kfd_proc_root, worker_pid
        )

    worker, probe_result = capture_worker_proc_with_probe(
        context.proc_root, worker_pid, probe
    )
    process_raw, vram, kfd_snapshot = probe_result
    kfd_processes = kfd_snapshot["processes"]
    kfd_positive = positive_kfd_processes(kfd_processes)
    own = [item["vram_bytes"] for item in kfd_positive if item["pid"] == worker_pid]
    unrelated = [item["pid"] for item in kfd_positive if item["pid"] != worker_pid]
    if own != [vram] or unrelated:
        fail("AMD SMI and isolated KFD VRAM ownership differ")
    gpu = {
        "index": GPU_INDEX,
        "bdf": GPU_BDF,
        "uuid": GPU_UUID,
        "kfd_gpu_id": KFD_GPU_ID,
        "process_raw_json": process_raw.decode("utf-8"),
        "process_raw_sha256": sha256_bytes(process_raw),
        "worker_pid": worker_pid,
        "mem_usage_value": vram,
        "mem_usage_unit": "B",
        "kfd_snapshot": kfd_snapshot,
    }
    return {"worker": worker, "gpu": gpu}, worker["vmrss_bytes"], vram


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    phase: str
    request_index: int
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    cancel_target: str | None = None


@dataclasses.dataclass(frozen=True)
class ReleaseObservation:
    request_id: str
    outcome: str
    completion_tokens: int
    observed_monotonic_ns: int
    cancel_write_started_monotonic_ns: int | None


def record_isolation_check(
    evidence: EvidenceWriter,
    kfd_proc_root: Path,
    worker_pid: int,
    *,
    phase: str,
    request_index: int | None,
    request_id: str | None,
    release_observed_monotonic_ns: int | None,
    not_before_monotonic_ns: int | None = None,
) -> None:
    ordering_floor = (
        release_observed_monotonic_ns
        if release_observed_monotonic_ns is not None
        else not_before_monotonic_ns
    )
    if ordering_floor is None:
        fail("ready isolation check requires an ordering floor")
    snapshot = capture_kfd_snapshot(
        kfd_proc_root, worker_pid, not_before_ns=ordering_floor
    )
    processes = snapshot["processes"]
    require_isolated_kfd_processes(processes, worker_pid)
    evidence.write(
        {
            "schema_version": RAW_SCHEMA,
            "record_type": "isolation_check",
            "phase": phase,
            "request_index": request_index,
            "request_id": request_id,
            "release_observed_monotonic_ns": release_observed_monotonic_ns,
            "kfd_snapshot": snapshot,
        }
    )


def write_all_before_deadline(
    stream: BinaryIO,
    payload: bytes,
    deadline_ns: int,
    label: str = "worker command write",
) -> None:
    try:
        descriptor = stream.fileno()
    except (AttributeError, OSError):
        stream.write(payload)
        stream.flush()
        if time.monotonic_ns() > deadline_ns:
            fail(f"{label} exceeded its absolute deadline")
        return

    was_blocking = os.get_blocking(descriptor)
    os.set_blocking(descriptor, False)
    try:
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except BlockingIOError:
                timeout = remaining_seconds(deadline_ns, label)
                _, writable, _ = select.select([], [descriptor], [], timeout)
                if not writable:
                    fail(f"{label} absolute deadline expired")
                continue
            if written <= 0:
                fail(f"{label} made no progress")
            offset += written
        if time.monotonic_ns() > deadline_ns:
            fail(f"{label} exceeded its absolute deadline")
    finally:
        os.set_blocking(descriptor, was_blocking)


class WorkerTransport:
    def __init__(self, stdin: BinaryIO, evidence: EvidenceWriter):
        self.stdin = stdin
        self.evidence = evidence
        self.last_command_timestamp = -1

    def _send(
        self,
        worker_command: dict[str, Any],
        raw_record: dict[str, Any],
        *,
        write_timeout_ns: int,
    ) -> tuple[int, int]:
        payload = json.dumps(
            worker_command, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
        encoded = payload + b"\n"
        start = next_strict_timestamp(self.last_command_timestamp)
        try:
            write_all_before_deadline(
                self.stdin,
                encoded,
                start + write_timeout_ns,
                f"worker {worker_command['type']} write",
            )
        except OSError as error:
            fail(f"worker command write failed: {error}")
        completed = next_strict_timestamp(start)
        self.last_command_timestamp = completed
        raw_record.update(
            {
                "schema_version": RAW_SCHEMA,
                "record_type": "command",
                "write_started_monotonic_ns": start,
                "write_completed_monotonic_ns": completed,
                "raw_json": payload.decode("utf-8"),
                "raw_sha256": sha256_bytes(payload),
            }
        )
        self.evidence.write(raw_record)
        return start, completed

    def generate(self, spec: RequestSpec) -> int:
        prompt_ids = list(range(1, spec.prompt_tokens + 1))
        command = {
            "schema_version": WORKER_SCHEMA,
            "type": "generate",
            "request_id": spec.request_id,
            "prompt_token_ids": prompt_ids,
            "max_new_tokens": spec.max_new_tokens,
            "sampling": SAMPLING,
            "eos_token_ids": EOS_TOKEN_IDS,
        }
        raw = {
            "phase": spec.phase,
            "request_index": spec.request_index,
            "request_id": spec.request_id,
            "command_type": "generate",
            "prompt_tokens": spec.prompt_tokens,
            "prompt_token_ids_sha256": prompt_sha256(spec.prompt_tokens),
            "max_new_tokens": spec.max_new_tokens,
            "sampling": SAMPLING,
            "eos_token_ids": EOS_TOKEN_IDS,
        }
        return self._send(
            command,
            raw,
            write_timeout_ns=PROGRESS_TIMEOUT_NS,
        )[0]

    def cancel(self, spec: RequestSpec) -> int:
        if spec.cancel_target not in {"prompt", "decode"}:
            fail("cancel request lacks a target")
        command = {
            "schema_version": WORKER_SCHEMA,
            "type": "cancel",
            "request_id": spec.request_id,
            "reason": "operator",
        }
        raw = {
            "phase": spec.phase,
            "request_index": spec.request_index,
            "request_id": spec.request_id,
            "command_type": "cancel",
            "cancel_reason": "operator",
            "cancel_target": spec.cancel_target,
        }
        return self._send(
            command,
            raw,
            write_timeout_ns=CANCEL_TIMEOUT_NS,
        )[0]

    def shutdown(self) -> int:
        command = {"schema_version": WORKER_SCHEMA, "type": "shutdown"}
        raw = {
            "phase": "shutdown",
            "request_index": None,
            "request_id": None,
            "command_type": "shutdown",
        }
        return self._send(
            command,
            raw,
            write_timeout_ns=SHUTDOWN_TIMEOUT_NS,
        )[0]


class WorkerEvents:
    def __init__(self, pump: StdoutPump, evidence: EvidenceWriter):
        self.pump = pump
        self.evidence = evidence

    def next(self, deadline_ns: int) -> PumpEvent | PumpEof:
        item = self.pump.receive(deadline_ns)
        if isinstance(item, PumpEvent):
            self.evidence.write(
                {
                    "schema_version": RAW_SCHEMA,
                    "record_type": "worker_event",
                    "observed_monotonic_ns": item.observed_monotonic_ns,
                    "raw_json": item.raw_json,
                    "raw_sha256": item.raw_sha256,
                    "event": item.event,
                }
            )
        return item


def validate_ready(event: dict[str, Any]) -> None:
    expected = {
        "schema_version": WORKER_SCHEMA,
        "type": "ready",
        "model": "ullm-qwen3-14b-sq8",
        "model_revision": EXPECTED_MODEL_REVISION,
        "artifact_content_sha256": EXPECTED_ARTIFACT_CONTENT_SHA256,
        "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA256,
        "device": "gfx1201",
        "execution_profile": "rdna4_w8a8_block_ck",
        "context_length": 4096,
        "max_new_tokens": 512,
    }
    if not json_type_equal(event, expected):
        fail("ready event differs from the frozen product identity")


def cancel_after_target(
    spec: RequestSpec,
    transport: WorkerTransport,
    target_observed_monotonic_ns: int,
) -> int:
    cancel_started = transport.cancel(spec)
    trigger_gap = cancel_started - target_observed_monotonic_ns
    if trigger_gap <= 0:
        fail("cancel write start does not follow its observed target event")
    if trigger_gap > PROGRESS_TIMEOUT_NS:
        fail("cancel target-to-write gap exceeds thirty seconds")
    return cancel_started


def run_request(
    spec: RequestSpec,
    transport: WorkerTransport,
    events: WorkerEvents,
) -> ReleaseObservation:
    generate_started = transport.generate(spec)
    request_deadline = generate_started + REQUEST_TIMEOUT_NS
    progress_deadline = generate_started + PROGRESS_TIMEOUT_NS
    cancel_started: int | None = None
    started = False
    transition_seen = False
    last_progress = 0
    token_count = 0
    while True:
        deadline = min(request_deadline, progress_deadline)
        if cancel_started is not None:
            deadline = min(deadline, cancel_started + CANCEL_TIMEOUT_NS)
        item = events.next(deadline)
        if isinstance(item, PumpEof):
            fail("worker stdout reached EOF during an active request")
        event = item.event
        if event["type"] == "error":
            fail(f"worker emitted error during {spec.request_id}: {event['code']}")
        if event.get("request_id") != spec.request_id:
            fail("worker event request ID differs from the active request")
        event_type = event["type"]
        if event_type == "started":
            if started or event["prompt_tokens"] != spec.prompt_tokens:
                fail("started event is repeated or differs")
            started = True
            progress_deadline = item.observed_monotonic_ns + PROGRESS_TIMEOUT_NS
            if spec.cancel_target == "prompt":
                cancel_started = cancel_after_target(
                    spec, transport, item.observed_monotonic_ns
                )
        elif event_type == "progress":
            processed = event["processed_prompt_tokens"]
            if not started or token_count or processed <= last_progress or processed > spec.prompt_tokens:
                fail("progress event ordering differs")
            if processed != spec.prompt_tokens and processed % 128 != 0:
                fail("progress milestone differs from M=128 contract")
            last_progress = processed
            transition_seen = processed == spec.prompt_tokens
            progress_deadline = item.observed_monotonic_ns + PROGRESS_TIMEOUT_NS
        elif event_type == "token":
            if not started or not transition_seen or event["index"] != token_count:
                fail("token event ordering or index differs")
            if spec.cancel_target == "prompt":
                fail("prompt-target cancellation published a token")
            token_count += 1
            progress_deadline = item.observed_monotonic_ns + PROGRESS_TIMEOUT_NS
            if spec.cancel_target == "decode" and token_count == 1:
                cancel_started = cancel_after_target(
                    spec, transport, item.observed_monotonic_ns
                )
        elif event_type == "released":
            if not started or event["prompt_tokens"] != spec.prompt_tokens:
                fail("released event arrived before matching started")
            if event["completion_tokens"] != token_count:
                fail("released completion count differs from token events")
            if spec.cancel_target is None:
                if event["outcome"] != "length" or token_count != 2 or spec.max_new_tokens != 2:
                    fail("normal request did not release with length and two tokens")
            else:
                if cancel_started is None:
                    fail("cancelled release arrived before the scheduled cancel write")
                if event["outcome"] != "cancelled" or event.get("cancel_reason") != "operator":
                    fail("cancelled request release differs")
                if token_count >= 512:
                    fail("cancelled request reached its generation limit")
                if spec.cancel_target == "prompt" and token_count != 0:
                    fail("prompt-target cancellation has completion tokens")
                if spec.cancel_target == "decode" and token_count < 1:
                    fail("decode-target cancellation has no completion token")
                bound = item.observed_monotonic_ns - cancel_started
                if bound < 0 or bound > CANCEL_TIMEOUT_NS:
                    fail("cancel-to-release upper bound exceeds five seconds")
            return ReleaseObservation(
                request_id=spec.request_id,
                outcome=event["outcome"],
                completion_tokens=token_count,
                observed_monotonic_ns=item.observed_monotonic_ns,
                cancel_write_started_monotonic_ns=cancel_started,
            )
        else:
            fail(f"unexpected event {event_type!r} during active request")


def normal_spec(phase: str, index: int, request_id: str) -> RequestSpec:
    return RequestSpec(phase, index, request_id, 8, 2, None)


def cancel_spec(phase: str, index: int, request_id: str, target: str) -> RequestSpec:
    return RequestSpec(phase, index, request_id, 128 if target == "prompt" else 8, 512, target)


def resource_request_spec(phase: str, index: int) -> RequestSpec:
    width = 2 if phase == "resource_warmup" else 3
    request_id = f"p8c-{phase.replace('_', '-')}-{index:0{width}d}"
    if index % 5 != 4:
        return normal_spec(phase, index, request_id)
    ordinal = (index + 1) // 5
    target = "prompt" if ordinal % 2 else "decode"
    return cancel_spec(phase, index, request_id, target)


def wait_until(deadline_ns: int) -> None:
    while True:
        remaining = deadline_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        time.sleep(remaining / 1_000_000_000)


def collect_resource_point(
    evidence: EvidenceWriter,
    context: ProbeContext,
    worker_pid: int,
    *,
    release: ReleaseObservation | None,
    request_index: int | None,
    settle_started_ns: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    rss_values: list[int] = []
    vram_values: list[int] = []
    thread_values: list[int] = []
    fd_values: list[int] = []
    child_values: list[int] = []
    previous_sample_start: int | None = None
    for sample_index in range(5):
        deadline = (
            settle_started_ns + IDLE_SETTLE_NS
            if sample_index == 0
            else int(previous_sample_start) + SAMPLE_INTERVAL_NS
        )
        wait_until(deadline)
        sample_started = time.monotonic_ns()
        captured, rss, vram = capture_resource_sample(context, worker_pid)
        worker = captured["worker"]
        evidence.write(
            {
                "schema_version": RAW_SCHEMA,
                "record_type": "resource_sample",
                "phase": "baseline" if release is None else "post_release",
                "request_index": request_index,
                "request_id": None if release is None else release.request_id,
                "release_outcome": None if release is None else release.outcome,
                "release_observed_monotonic_ns": (
                    None if release is None else release.observed_monotonic_ns
                ),
                "settle_started_monotonic_ns": settle_started_ns,
                "sample_index": sample_index,
                "sample_started_monotonic_ns": sample_started,
                "worker": worker,
                "gpu": captured["gpu"],
            }
        )
        rss_values.append(rss)
        vram_values.append(vram)
        thread_values.append(worker["threads"])
        fd_values.append(worker["fd_count"])
        child_values.append(len(worker["children"]))
        previous_sample_start = sample_started
    return rss_values, vram_values, thread_values, fd_values, child_values


def validate_git_status_raw(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        fail(f"git porcelain status is not UTF-8: {error}")
    for line in text.splitlines():
        if not line.startswith("?? "):
            fail("source tree has tracked changes")
        path = line[3:]
        if (
            not path.startswith(".rocprofv3/")
            or path == ".rocprofv3/.."
            or ".." in Path(path).parts
        ):
            fail(f"source tree has a disallowed untracked path: {path}")
    return text


def require_git_toplevel(repo_root: Path) -> None:
    raw = run_bounded_command(
        ["git", "rev-parse", "--show-toplevel"],
        "git worktree root",
        cwd=repo_root,
    )
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError as error:
        fail(f"git worktree root is not UTF-8: {error}")
    lines = text.splitlines()
    if len(lines) != 1 or not lines[0]:
        fail("git worktree root output differs")
    actual_root = regular_directory(Path(lines[0]), "git worktree root")
    if actual_root != repo_root:
        fail("--repo-root must be the Git worktree top-level directory")


def git_identity(repo_root: Path) -> tuple[str, str]:
    commit_raw = run_bounded_command(
        ["git", "rev-parse", "HEAD"], "git rev-parse", cwd=repo_root
    )
    status_raw = run_bounded_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        "git status",
        cwd=repo_root,
    )
    try:
        commit = commit_raw.decode("ascii", errors="strict").strip()
    except UnicodeError as error:
        fail(f"git commit is not ASCII: {error}")
    if HEX40_RE.fullmatch(commit) is None:
        fail("git HEAD is not a full lowercase commit")
    return commit, validate_git_status_raw(status_raw)


@dataclasses.dataclass(frozen=True)
class Preflight:
    repo_root: Path
    output_dir: Path
    worker: Path
    artifact: Path
    package: Path
    git_commit: str
    git_status_raw: str
    binary_sha256: str
    amd_version_raw: bytes
    amd_list_raw: bytes
    guards: dict[str, str]
    preflight_kfd_snapshot: dict[str, Any]


def preflight(args: argparse.Namespace, runner: CommandRunner = run_bounded_command) -> Preflight:
    repo_root = regular_directory(args.repo_root, "repository root")
    require_git_toplevel(repo_root)
    output_dir = prospective_output_directory(args.output_dir, repo_root)
    worker = regular_file(args.worker, "worker binary", executable=True)
    artifact = regular_directory(args.artifact, "artifact directory")
    package = regular_directory(args.package, "package directory")
    commit, git_status_raw = git_identity(repo_root)
    artifact_manifest = read_small_file(artifact / "sq_manifest.json", "artifact manifest")
    if sha256_bytes(artifact_manifest) != EXPECTED_ARTIFACT_MANIFEST_SHA256:
        fail("artifact manifest SHA-256 differs")
    artifact_value = strict_json_object(artifact_manifest, "artifact manifest")
    integrity = artifact_value.get("integrity")
    if not isinstance(integrity, dict) or integrity.get("content_sha256") != EXPECTED_ARTIFACT_CONTENT_SHA256:
        fail("artifact content SHA-256 differs")
    package_manifest = read_small_file(package / "manifest.json", "package manifest")
    if sha256_bytes(package_manifest) != EXPECTED_PACKAGE_MANIFEST_SHA256:
        fail("package manifest SHA-256 differs")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "1":
        fail("HIP_VISIBLE_DEVICES must be exactly 1")
    guards = {name: os.environ.get(name, "") for name in REQUIRED_HIP_GUARDS}
    if any(value != "1" for value in guards.values()):
        fail("every required HIP guard must equal 1")
    amd_version = runner([args.amd_smi, "version"], "amd-smi version")
    version_text = amd_version.decode("utf-8")
    if any(part not in version_text for part in REQUIRED_AMD_SMI_VERSION_PARTS):
        fail("AMD SMI version differs from the frozen environment")
    amd_list = runner([args.amd_smi, "list", "--json"], "amd-smi list")
    parse_amd_smi_list(amd_list)
    preflight_kfd_snapshot = capture_kfd_snapshot(
        Path("/sys/class/kfd/kfd/proc"), None
    )
    require_isolated_kfd_processes(preflight_kfd_snapshot["processes"], None)
    return Preflight(
        repo_root=repo_root,
        output_dir=output_dir,
        worker=worker,
        artifact=artifact,
        package=package,
        git_commit=commit,
        git_status_raw=git_status_raw,
        binary_sha256=sha256_file(worker),
        amd_version_raw=amd_version,
        amd_list_raw=amd_list,
        guards=guards,
        preflight_kfd_snapshot=preflight_kfd_snapshot,
    )


def final_git_identity(identity: Preflight) -> tuple[str, str]:
    commit, status_raw = git_identity(identity.repo_root)
    if commit != identity.git_commit:
        fail("git HEAD changed during the acceptance run")
    return commit, status_raw


def header_record(
    preflight_result: Preflight,
    worker_pid: int,
    worker_ppid: int,
    worker_starttime: int,
    worker_exe: str,
) -> dict[str, Any]:
    return {
        "schema_version": RAW_SCHEMA,
        "record_type": "header",
        "clock": "python.time.monotonic_ns",
        "build": {
            "git_commit": preflight_result.git_commit,
            "tracked_clean": True,
            "git_status_raw": preflight_result.git_status_raw,
            "git_status_raw_sha256": sha256_bytes(
                preflight_result.git_status_raw.encode("utf-8")
            ),
            "binary_sha256": preflight_result.binary_sha256,
            "artifact_manifest_sha256": EXPECTED_ARTIFACT_MANIFEST_SHA256,
            "artifact_content_sha256": EXPECTED_ARTIFACT_CONTENT_SHA256,
            "package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA256,
        },
        "worker": {
            "pid": worker_pid,
            "ppid": worker_ppid,
            "starttime_ticks": worker_starttime,
            "exe": worker_exe,
        },
        "device": {
            "gpu_index": GPU_INDEX,
            "bdf": GPU_BDF,
            "uuid": GPU_UUID,
            "kfd_gpu_id": KFD_GPU_ID,
            "amd_smi_list_raw_json": preflight_result.amd_list_raw.decode("utf-8"),
            "amd_smi_list_raw_sha256": sha256_bytes(preflight_result.amd_list_raw),
        },
        "environment": {
            "hip_visible_devices": "1",
            "required_hip_guards": preflight_result.guards,
            "amd_smi_version_raw": preflight_result.amd_version_raw.decode("utf-8"),
            "amd_smi_version_raw_sha256": sha256_bytes(preflight_result.amd_version_raw),
            "preflight_kfd_snapshot": preflight_result.preflight_kfd_snapshot,
        },
        "schedule": {
            "latency_warmups": 2,
            "latency_measured": 10,
            "resource_warmups": 10,
            "resource_requests": 100,
            "cancel_block_size": 5,
            "cancel_block_offset": 4,
            "idle_settle_ms": 5000,
            "samples_per_point": 5,
            "sample_interval_ms": 1000,
        },
        "thresholds": {
            "cancel_sample_max_ns": CANCEL_TIMEOUT_NS,
            "cancel_p95_max_ns": LATENCY_P95_MAX_NS,
            "theil_sen_max_bytes_per_request": THEIL_SEN_MAX_BYTES_PER_REQUEST,
            "final_delta_max_bytes": FINAL_DELTA_MAX_BYTES,
            "request_max_ns": REQUEST_TIMEOUT_NS,
            "progress_max_ns": PROGRESS_TIMEOUT_NS,
            "shutdown_max_ns": SHUTDOWN_TIMEOUT_NS,
        },
    }


def terminate_worker(process: subprocess.Popen[bytes], process_group: int) -> None:
    try:
        if process.stdin is not None:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    terminate_process_group(process, process_group)


def remaining_seconds(deadline_ns: int, label: str) -> float:
    remaining_ns = deadline_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        fail(f"{label} absolute deadline expired")
    return remaining_ns / 1_000_000_000


def spawn_worker(
    identity: Preflight, environment: dict[str, str]
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            str(identity.worker),
            "--artifact",
            str(identity.artifact),
            "--package",
            str(identity.package),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env=environment,
        start_new_session=True,
    )


def execute(args: argparse.Namespace) -> None:
    identity = preflight(args)
    evidence = EvidenceWriter(identity.output_dir, repo_root=identity.repo_root)
    process: subprocess.Popen[bytes] | None = None
    process_group: int | None = None
    stdout_pump: StdoutPump | None = None
    stderr_drain: StderrDrain | None = None
    stderr_output: BinaryIO | None = None
    try:
        stderr_output = evidence.open_stderr()
        environment = os.environ.copy()
        worker_spawn_started = time.monotonic_ns()
        process = spawn_worker(identity, environment)
        process_group = process.pid
        if process.stdin is None or process.stdout is None or process.stderr is None:
            fail("worker pipes were not created")
        stdout_pump = StdoutPump(process.stdout)
        stderr_drain = StderrDrain(process.stderr, stderr_output)
        stderr_output = None
        stdout_pump.start()
        stderr_drain.start()
        worker_ppid, worker_starttime = read_proc_identity(Path("/proc"), process.pid)
        worker_exe = os.readlink(Path("/proc") / str(process.pid) / "exe")
        if worker_ppid != os.getpid() or Path(worker_exe).resolve() != identity.worker:
            fail("spawned worker process identity differs")
        evidence.write(
            header_record(identity, process.pid, worker_ppid, worker_starttime, worker_exe)
        )
        events = WorkerEvents(stdout_pump, evidence)
        ready_item = events.next(worker_spawn_started + READY_TIMEOUT_NS)
        if isinstance(ready_item, PumpEof):
            fail("worker exited before ready")
        validate_ready(ready_item.event)
        record_isolation_check(
            evidence,
            Path("/sys/class/kfd/kfd/proc"),
            process.pid,
            phase="ready",
            request_index=None,
            request_id=None,
            release_observed_monotonic_ns=None,
            not_before_monotonic_ns=ready_item.observed_monotonic_ns,
        )
        transport = WorkerTransport(process.stdin, evidence)

        measured_bounds: list[int] = []
        latency_phases = (("latency_warmup", 2), ("latency_measured", 10))
        overall_ordinal = 0
        for phase, count in latency_phases:
            emit_progress(f"phase {phase} started")
            for index in range(1, count + 1):
                overall_ordinal += 1
                target = "prompt" if overall_ordinal % 2 else "decode"
                request_id = f"p8c-{phase.replace('_', '-')}-{index:02d}"
                release = run_request(
                    cancel_spec(phase, index, request_id, target), transport, events
                )
                record_isolation_check(
                    evidence,
                    Path("/sys/class/kfd/kfd/proc"),
                    process.pid,
                    phase=phase,
                    request_index=index,
                    request_id=request_id,
                    release_observed_monotonic_ns=release.observed_monotonic_ns,
                )
                if phase == "latency_measured":
                    assert release.cancel_write_started_monotonic_ns is not None
                    measured_bounds.append(
                        release.observed_monotonic_ns
                        - release.cancel_write_started_monotonic_ns
                    )
                recovery_id = request_id + "-recovery"
                recovery = run_request(
                    normal_spec(phase, index, recovery_id), transport, events
                )
                record_isolation_check(
                    evidence,
                    Path("/sys/class/kfd/kfd/proc"),
                    process.pid,
                    phase=phase,
                    request_index=index,
                    request_id=recovery_id,
                    release_observed_monotonic_ns=recovery.observed_monotonic_ns,
                )
        if percentile_linear(measured_bounds, fractions.Fraction(95, 100)) > LATENCY_P95_MAX_NS:
            fail("measured cancellation upper-bound p95 exceeds two seconds")
        emit_progress("latency complete")

        probe_context = ProbeContext(args.amd_smi)
        evidence.write(capture_gpu_metric(args.amd_smi, run_bounded_command, "before"))

        emit_progress("phase resource_warmup started")
        last_warmup: ReleaseObservation | None = None
        for index in range(1, 11):
            last_warmup = run_request(resource_request_spec("resource_warmup", index), transport, events)
            record_isolation_check(
                evidence,
                probe_context.kfd_proc_root,
                process.pid,
                phase="resource_warmup",
                request_index=index,
                request_id=last_warmup.request_id,
                release_observed_monotonic_ns=last_warmup.observed_monotonic_ns,
            )
        emit_progress("resource warmup complete (10/10)")
        assert last_warmup is not None
        emit_progress("phase resource_baseline started")
        baseline = collect_resource_point(
            evidence,
            probe_context,
            process.pid,
            release=None,
            request_index=None,
            settle_started_ns=last_warmup.observed_monotonic_ns,
        )
        baseline_medians = tuple(median(series) for series in baseline)
        rss_points: list[int] = []
        vram_points: list[int] = []
        emit_progress("phase resource_measured started")
        for index in range(1, 101):
            release = run_request(resource_request_spec("resource_measured", index), transport, events)
            record_isolation_check(
                evidence,
                probe_context.kfd_proc_root,
                process.pid,
                phase="resource_measured",
                request_index=index,
                request_id=release.request_id,
                release_observed_monotonic_ns=release.observed_monotonic_ns,
            )
            point = collect_resource_point(
                evidence,
                probe_context,
                process.pid,
                release=release,
                request_index=index,
                settle_started_ns=release.observed_monotonic_ns,
            )
            point_medians = tuple(median(series) for series in point)
            if point_medians[2:] != baseline_medians[2:]:
                fail("worker thread, FD, or child-count median differs from baseline")
            rss_points.append(int(point_medians[0]))
            vram_points.append(int(point_medians[1]))
            if index % 10 == 0:
                emit_progress(f"resource measured {index}/100 complete")
        evidence.write(capture_gpu_metric(args.amd_smi, run_bounded_command, "after"))
        for label, values, baseline_value in (
            ("RSS", rss_points, baseline_medians[0]),
            ("VRAM", vram_points, baseline_medians[1]),
        ):
            if theil_sen(values) > THEIL_SEN_MAX_BYTES_PER_REQUEST:
                fail(f"worker {label} Theil-Sen slope exceeds the threshold")
            if fractions.Fraction(values[-1]) - baseline_value > FINAL_DELTA_MAX_BYTES:
                fail(f"worker {label} final delta exceeds 64 MiB")

        emit_progress("phase shutdown started")
        shutdown_started = transport.shutdown()
        shutdown_deadline = shutdown_started + SHUTDOWN_TIMEOUT_NS
        eof = events.next(shutdown_deadline)
        if not isinstance(eof, PumpEof):
            fail("worker emitted an event after shutdown")
        try:
            exit_code = process.wait(
                timeout=remaining_seconds(shutdown_deadline, "worker shutdown")
            )
        except subprocess.TimeoutExpired:
            fail("worker did not exit after shutdown")
        exit_observed = time.monotonic_ns()
        if exit_observed > shutdown_deadline:
            fail("worker shutdown exceeded thirty seconds")
        stderr_drain.join(
            timeout=remaining_seconds(shutdown_deadline, "worker shutdown")
        )
        stderr_digest = evidence.finish_stderr()
        if time.monotonic_ns() > shutdown_deadline:
            fail("worker shutdown evidence finalization exceeded thirty seconds")
        if exit_code != 0:
            fail("worker shutdown exit code is nonzero")
        final_git_commit, final_git_status_raw = final_git_identity(identity)
        evidence.write(
            {
                "schema_version": RAW_SCHEMA,
                "record_type": "process_exit",
                "stdout_eof_monotonic_ns": eof.observed_monotonic_ns,
                "exit_observed_monotonic_ns": exit_observed,
                "exit_code": exit_code,
                "stderr_file": "worker-stderr.jsonl",
                "stderr_sha256": stderr_digest,
                "final_git_commit": final_git_commit,
                "final_git_status_raw": final_git_status_raw,
                "final_git_status_raw_sha256": sha256_bytes(
                    final_git_status_raw.encode("utf-8")
                ),
            }
        )
        evidence.publish()
        emit_progress("complete")
    except BaseException:
        if process is not None and process_group is not None:
            terminate_worker(process, process_group)
        if stderr_drain is not None:
            try:
                stderr_drain.join()
            except AcceptanceError:
                pass
        elif stderr_output is not None:
            try:
                stderr_output.close()
            except OSError:
                pass
        evidence.abort()
        raise
    finally:
        if stdout_pump is not None:
            stdout_pump.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/artifact"),
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=Path("/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/package"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--amd-smi", default="amd-smi")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        execute(parse_args(argv))
        return 0
    except (AcceptanceError, OSError, ValueError) as error:
        print(f"SQ8 worker acceptance failed: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("SQ8 worker acceptance interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
