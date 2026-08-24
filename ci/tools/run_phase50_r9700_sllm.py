#!/usr/bin/env python3
"""Run the bounded Phase 50 R9700 direct-performance matrix.

This producer is intentionally self contained so that it can be copied to the
R9700 host together with the release binary and run without model-cache helpers.
The benchmark binary remains the authority for model loading and direct-engine
timing; this script owns the frozen input matrix,
process lifetime, raw-output retention, and fail-closed result checks.

All output is written below an explicitly supplied directory.  Raw stdout,
stderr, and 100 ms sysfs HBM/GTT samples are retained per row.  A completed
row is resumable, while an incomplete row is never silently overwritten.
Execution failures are retained as explicit FAIL rows with bounded reasons;
later independent rows continue and any failed row makes the summary FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence


TARGET = "gfx1201"
GPU_UUID = "GPU-a8e9ddefa2d60f55"
GPU_BDF = "0000:07:00.0"
MODEL_SIZE = "4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SCHEMA_VERSION = "phase50-r9700-sllm-v1"
ROW_SCHEMA_VERSION = "phase50-r9700-sllm-row-v1"
DIRECT_SCHEMA_VERSION = "engine-performance-direct-v2"
STANDARD_WARMUPS = 3
STANDARD_MEASURED = 10
EXTENDED_WARMUPS = 1
EXTENDED_MEASURED = 3
MONITOR_PERIOD_SECONDS = 0.1
DEFAULT_TIMEOUT_SECONDS = 86_400.0
MAX_STDOUT_BYTES = 128 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_MONITOR_SAMPLES = 100_000
DEFAULT_AMD_SMI = "/opt/rocm/bin/amd-smi"
MAX_FAILURE_REASON_CHARS = 2048
STOP_IDS = (248046, 248044)
VISIBILITY_NAMES = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

SHORT_ODD = [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43]
LONG_INPUT = [23066] * 10_001
VERY_LONG_INPUT = [23066] * 100_000


class SessionDError(RuntimeError):
    """Malformed evidence, execution failure, or unsafe output state."""


DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_TOKEN_ID = 248319


def _fail(message: str) -> NoReturn:
    raise SessionDError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _json_load(data: bytes, label: str) -> Any:
    if not data or len(data) > MAX_JSON_BYTES:
        _fail(f"{label}: empty or oversized JSON")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(token: str) -> NoReturn:
        _fail(f"{label}: non-finite JSON constant {token}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=reject_duplicates, parse_constant=reject_constant)
    except SessionDError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: malformed JSON: {exc}")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"cannot stat {label}: {exc}")
    if size <= 0:
        _fail(f"{label} is empty: {path}")
    if executable and not os.access(path, os.X_OK):
        _fail(f"benchmark binary is not executable: {path}")
    return path


def file_identity(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot hash {label}: {exc}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def input_ids_for(case_id: str) -> list[int]:
    if case_id == "short-odd":
        return list(SHORT_ODD)
    if case_id in {"32-32", "prefill-long"}:
        count = 32 if case_id == "32-32" else 1024
        result = list(SHORT_ODD)
        result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), count))
        return result
    if case_id in {"decode-long", "decode-20000"}:
        result = list(SHORT_ODD)
        result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), 32))
        return result
    if case_id == "long-10001":
        return list(LONG_INPUT)
    if case_id == "long-100000":
        return list(VERY_LONG_INPUT)
    _fail(f"unknown performance case: {case_id}")


CASE_SPECS: tuple[tuple[str, int, int], ...] = (
    ("short-odd", 17, 17),
    ("32-32", 32, 32),
    ("prefill-long", 1024, 128),
    ("decode-long", 32, 256),
    ("long-10001", 10_001, 2),
    ("long-100000", 100_000, 2),
    ("decode-20000", 32, 20_000),
)


def matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id, input_count, output_count in CASE_SPECS:
        ids = input_ids_for(case_id)
        if len(ids) != input_count:
            _fail(f"matrix case {case_id} has {len(ids)} inputs, expected {input_count}")
        rows.append(
            {
                    "row_id": f"phase50-r9700-sllm-{case_id}",
                    "weight": "bf16",
                    "model_size": MODEL_SIZE,
                    "case_id": case_id,
                    "input_token_ids": ids,
                    "input_token_count": input_count,
                    "requested_output_tokens": output_count,
                    "target": TARGET,
                    "device_index": 0,
                    "warmups": EXTENDED_WARMUPS if case_id in {"long-100000", "decode-20000"} else STANDARD_WARMUPS,
                    "measured": EXTENDED_MEASURED if case_id in {"long-100000", "decode-20000"} else STANDARD_MEASURED,
                    "context_length": 131_072 if case_id in {"long-100000", "decode-20000"} else input_count + output_count,
                    "ignore_eos": case_id == "decode-20000",
                    "prefill_chunk_tokens": None,
            }
        )
    return rows


def _atomic_write(path: Path, data: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite existing {label}: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        _fail(f"cannot atomically write {label}: {exc}")
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _parse_sysfs_value(path: Path) -> int:
    try:
        text = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read sysfs memory file {path}: {exc}")
    match = re.fullmatch(r"([0-9]+)(?:\s*(?:B|bytes))?", text)
    if match is None:
        _fail(f"sysfs memory value is malformed: {path}: {text!r}")
    return int(match.group(1), 10)


def _sysfs_files(root: Path, names: Sequence[str]) -> list[Path]:
    found: list[Path] = []
    for name in names:
        direct = root / name
        if direct.is_file() and not direct.is_symlink():
            found.append(direct)
        try:
            found.extend(path for path in root.rglob(name) if path.is_file() and not path.is_symlink())
        except OSError as exc:
            _fail(f"cannot enumerate sysfs {root}: {exc}")
    return sorted(set(found))


def read_sysfs_memory(root: Path) -> dict[str, Any]:
    """Read HBM/GTT byte counters from standard AMD sysfs files."""
    vram = _sysfs_files(root, ("mem_info_vram_used", "hbm_used", "vram_used"))
    gtt = _sysfs_files(root, ("mem_info_gtt_used", "gtt_used"))
    if not vram or not gtt:
        _fail("sysfs HBM/GTT counters are unavailable")
    vram_values = {str(path): _parse_sysfs_value(path) for path in vram}
    gtt_values = {str(path): _parse_sysfs_value(path) for path in gtt}
    return {
        "hbm_bytes": sum(vram_values.values()),
        "gtt_bytes": sum(gtt_values.values()),
        "hbm_files": vram_values,
        "gtt_files": gtt_values,
    }


def _process_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def process_snapshot(
    binary: Path,
    owned_pids: Iterable[int] = (),
    *,
    amd_smi: Path | None = None,
    gpu_bdf: str | None = None,
) -> dict[str, Any]:
    """Record an auditable process view without claiming unrelated GPU PIDs."""
    owned = {int(pid) for pid in owned_pids if int(pid) > 0}
    needle = binary.name.lower()
    records: list[dict[str, Any]] = []
    try:
        proc_entries = sorted(Path("/proc").iterdir(), key=lambda path: path.name)
    except OSError:
        proc_entries = []
    for entry in proc_entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = _process_cmdline(pid)
        if pid in owned or needle in command.lower() or "sllm" in command.lower():
            records.append({"pid": pid, "command": command, "owned": pid in owned})
    report: dict[str, Any] = {
        "source": "procfs-command-and-owned-pid",
        "available": True,
        "reliable": True,
        "gpu_processes": records,
        "owned_pids": sorted(owned),
        "timestamp_ns": time.monotonic_ns(),
    }
    if amd_smi is not None and gpu_bdf is not None and amd_smi.is_file() and os.access(amd_smi, os.X_OK):
        command = [str(amd_smi), "process", "--json", "-g", gpu_bdf]
        try:
            completed = subprocess.run(command, capture_output=True, check=False, timeout=8)
            query: dict[str, Any] = {
                "command": command,
                "exit_code": completed.returncode,
                "stderr": completed.stderr.decode("utf-8", errors="replace"),
            }
            if completed.returncode == 0 and completed.stdout:
                query["report"] = _json_load(completed.stdout, "amd-smi process report")
                query["available"] = True
            else:
                query["available"] = False
            report["gpu_process_query"] = query
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["gpu_process_query"] = {"command": command, "available": False, "error": str(exc)}
    else:
        report["gpu_process_query"] = {"available": False, "reason": "amd-smi process observer unavailable"}
    return report


def _safe_process_snapshot(binary: Path, owned_pids: Iterable[int] = (), *, amd_smi: Path | None = None, gpu_bdf: str | None = None) -> dict[str, Any]:
    owned = list(owned_pids)
    try:
        return process_snapshot(binary, owned, amd_smi=amd_smi, gpu_bdf=gpu_bdf)
    except SessionDError as exc:
        return {"source": "process-observer", "available": False, "reliable": False, "gpu_processes": [], "owned_pids": [int(pid) for pid in owned if isinstance(pid, int) and pid > 0], "error": _bounded_reason(exc), "timestamp_ns": time.monotonic_ns()}


class _SysfsMonitor:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.thread: threading.Thread | None = None

    def _capture(self) -> None:
        try:
            sample = read_sysfs_memory(self.root)
            sample["timestamp_ns"] = time.monotonic_ns()
            self.samples.append(sample)
            if len(self.samples) > MAX_MONITOR_SAMPLES:
                self.errors.append("monitor sample bound exceeded")
                self.stop.set()
        except SessionDError as exc:
            self.errors.append(str(exc))
            self.stop.set()

    def _run(self) -> None:
        while not self.stop.is_set():
            self._capture()
            self.stop.wait(MONITOR_PERIOD_SECONDS)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="phase50-r9700-sllm-sysfs", daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            if self.thread.is_alive():
                self.errors.append("sysfs monitor did not terminate")


def _settled_memory(root: Path, *, timeout_seconds: float = 3.0) -> dict[str, Any]:
    previous: dict[str, Any] | None = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        current = read_sysfs_memory(root)
        if previous is not None and current["hbm_bytes"] == previous["hbm_bytes"] and current["gtt_bytes"] == previous["gtt_bytes"]:
            current["settled"] = True
            return current
        previous = current
        time.sleep(MONITOR_PERIOD_SECONDS)
    current = read_sysfs_memory(root)
    current["settled"] = False
    return current


def _terminate_group(process: subprocess.Popen[bytes]) -> dict[str, bool]:
    sent_term = sent_kill = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        sent_term = True
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            sent_kill = True
        except (ProcessLookupError, OSError):
            pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            _fail("benchmark process group did not terminate")
    return {"term_sent": sent_term, "kill_sent": sent_kill}


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _run_command(command: list[str], env: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env),
            start_new_session=True,
        )
    except OSError as exc:
        _fail(f"benchmark process could not start: {exc}")
    assert process.stdout is not None and process.stderr is not None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        termination = {"term_sent": False, "kill_sent": False}
        timed_out = False
    except subprocess.TimeoutExpired:
        termination = _terminate_group(process)
        stdout, stderr = process.communicate()
        timed_out = True
    if len(stdout) > MAX_STDOUT_BYTES or len(stderr) > MAX_STDERR_BYTES:
        _fail("benchmark stdout/stderr exceeded bounded raw size")
    return {
        "pid": process.pid,
        "exit_code": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "termination": termination,
        "process_group_gone": process.poll() is not None and _process_group_gone(process.pid),
    }


def _empty_capture(error: str | None = None) -> dict[str, Any]:
    capture: dict[str, Any] = {
        "pid": None,
        "exit_code": None,
        "stdout": b"",
        "stderr": b"",
        "timed_out": False,
        "termination": {"term_sent": False, "kill_sent": False},
        "process_group_gone": False,
    }
    if error:
        capture["error"] = _bounded_reason(error)
    return capture


def _bounded_reason(value: Any) -> str:
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        text = "unspecified failure"
    if len(text) > MAX_FAILURE_REASON_CHARS:
        text = text[:MAX_FAILURE_REASON_CHARS] + "..."
    return text


def _failure_class(
    capture: Mapping[str, Any],
    execution_error: str | None,
    monitor_errors: Sequence[str],
    baseline: Mapping[str, Any] | None,
    settled: Mapping[str, Any] | None,
    validation_error: str | None = None,
) -> tuple[str, str]:
    if validation_error is not None:
        return "validation", _bounded_reason(validation_error)
    if capture.get("timed_out") is True:
        return "timeout", "benchmark exceeded the configured timeout"
    exit_code = capture.get("exit_code")
    stderr = _bounded_reason(capture.get("stderr", b""))
    if isinstance(exit_code, int) and exit_code < 0:
        return "crash", f"process terminated by signal {-exit_code}; stderr={stderr}"
    if isinstance(exit_code, int) and exit_code != 0:
        lowered = stderr.lower()
        if any(token in lowered for token in ("out of memory", "oom", "physical commitment", "allocation failed", "hip_error_out_of_memory")):
            return "oom", f"nonzero exit {exit_code}; stderr={stderr}"
        return "nonzero", f"nonzero exit {exit_code}; stderr={stderr}"
    if execution_error is not None:
        return "crash", _bounded_reason(execution_error)
    if capture.get("process_group_gone") is not True:
        return "crash", "process group did not terminate cleanly"
    if monitor_errors:
        return "monitor", _bounded_reason("; ".join(monitor_errors))
    if not baseline or not settled:
        return "resource", "HBM/GTT memory snapshot was unavailable"
    if settled.get("settled") is not True or settled.get("hbm_bytes") != baseline.get("hbm_bytes") or settled.get("gtt_bytes") != baseline.get("gtt_bytes"):
        return "resource", "HBM/GTT usage did not return to the baseline"
    return "failure", "row did not produce a valid PASS result"


def _expected_command(
    binary: Path,
    model: Path,
    lock: Path,
    row: Mapping[str, Any],
    input_token_file: Path | None = None,
) -> list[str]:
    command = [
        str(binary),
        "benchmark",
        "--lane", "direct",
        "--gguf", str(model),
        "--derived-lock", str(lock),
        "--row-id", str(row["row_id"]),
        "--model-size", MODEL_SIZE,
        "--case-id", str(row["case_id"]),
        "--max-new-tokens", str(row["requested_output_tokens"]),
        "--device-index", str(row["device_index"]),
        "--target", TARGET,
        "--kv-cache-encoding", "fp16",
        "--greedy",
        "--warmups", str(row["warmups"]),
        "--measured", str(row["measured"]),
        "--context-length", str(row["context_length"]),
        "--completion-timeout-seconds", "21600",
    ]
    if input_token_file is None:
        command[command.index("--max-new-tokens"):command.index("--max-new-tokens")] = [
            "--input-token-ids",
            ",".join(str(value) for value in row["input_token_ids"]),
        ]
    else:
        command[command.index("--max-new-tokens"):command.index("--max-new-tokens")] = [
            "--input-token-ids-file",
            str(input_token_file),
        ]
    if row["ignore_eos"]:
        command.append("--ignore-eos")
    return command


def _execution_environment(
    row_id: str,
    device_index: int,
    base: Mapping[str, str] | None = None,
    gpu_uuid: str = GPU_UUID,
) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    for name in VISIBILITY_NAMES:
        env.pop(name, None)
    # The CLI keeps logical device index 0, while ROCr visibility must bind the
    # exact canonical R9700 UUID rather than relying on ordinal discovery.
    env["ROCR_VISIBLE_DEVICES"] = gpu_uuid
    env["SLLM_PHASE50_R9700_ROW"] = row_id
    # Do not inherit an arbitrary loader search path into the fixed ROCm lane.
    # The source/build identity separately pins this provider to ROCm 7.14.
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib:/opt/rocm/lib/llvm/lib"
    return env


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _validate_no_fallback(value: Any, label: str) -> None:
    for key, item in _walk(value):
        lowered = key.lower()
        if lowered in {"fallback_used", "cpu_fallback_used", "partial_offload"} and item is not False:
            _fail(f"{label}: {key} is not false")
        if lowered.startswith("fallback") and isinstance(item, int) and not isinstance(item, bool) and item != 0:
            _fail(f"{label}: {key} is nonzero")
        if lowered.startswith("no_fallback") and item is not True:
            _fail(f"{label}: {key} is not positive")


def _validate_cleanup(value: Any, label: str) -> None:
    seen = False
    zero_required = {
        "retryable_cleanup",
        "durable_quarantine",
        "cleanup_pending",
        "cleanup_durable",
        "cleanup_accounting_errors",
        "terminal_current",
    }
    for key, item in _walk(value):
        lowered = key.lower()
        if "cleanup" in lowered or lowered in {"terminal_zero", "all_requests_dropped"}:
            seen = True
            if isinstance(item, bool):
                if lowered in {"terminal_zero", "all_requests_dropped"} and item is not True:
                    _fail(f"{label}: {key} is not true")
            elif lowered in zero_required and isinstance(item, int) and not isinstance(item, bool) and item != 0:
                _fail(f"{label}: {key} is nonzero")
    if not seen:
        _fail(f"{label}: cleanup evidence is absent")


def _tokens(value: Mapping[str, Any], key: str, label: str) -> list[int]:
    result = value.get(key)
    if not isinstance(result, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > MAX_TOKEN_ID for item in result):
        _fail(f"{label}: {key} is malformed")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None:
        _fail(f"{label}: digest is malformed")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{label}: positive integer is required")
    return value


def _snapshot(value: Any, label: str) -> None:
    if not isinstance(value, dict) or value.get("poisoned") is not False:
        _fail(f"{label}: allocation snapshot is malformed")
    for key in ("current_bytes", "high_water_bytes"):
        if isinstance(value.get(key), bool) or not isinstance(value.get(key), int) or value[key] < 0:
            _fail(f"{label}.{key}: allocation value is malformed")
    for category in ("model_resident", "request_state", "workspace"):
        section = value.get(category)
        if not isinstance(section, dict):
            _fail(f"{label}.{category}: allocation category is absent")
        for key in ("current_bytes", "high_water_bytes"):
            if isinstance(section.get(key), bool) or not isinstance(section.get(key), int) or section[key] < 0:
                _fail(f"{label}.{category}.{key}: allocation value is malformed")


def _request_memory(value: Any, model_ready_bytes: int, label: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{label}: request memory evidence is absent")
    request_start = value.get("request_start")
    after_cleanup = value.get("after_cleanup")
    _snapshot(request_start, f"{label}.request_start")
    _snapshot(after_cleanup, f"{label}.after_cleanup")
    if request_start["current_bytes"] < model_ready_bytes:
        _fail(f"{label}: request start is below the resident model allocation")
    if (
        after_cleanup["current_bytes"] != model_ready_bytes
        or after_cleanup["model_resident"]["current_bytes"] != model_ready_bytes
        or after_cleanup["request_state"]["current_bytes"] != 0
        or after_cleanup["workspace"]["current_bytes"] != 0
    ):
        _fail(f"{label}: request allocations were not released after cleanup")


def _validate_timing(sample: Mapping[str, Any], expected_output: int, label: str) -> None:
    if sample.get("execution_path") != "timed-production" or sample.get("timing_instrumentation") != "on":
        _fail(f"{label}: timed-production instrumentation identity is invalid")
    events = sample.get("events")
    derived = sample.get("derived")
    if not isinstance(events, dict) or not isinstance(derived, dict):
        _fail(f"{label}: events/derived are absent")
    ordered = [events.get("request_start_ns"), events.get("prefill_submit_ns"), events.get("prefill_complete_ns"), events.get("first_token_ns")]
    later = events.get("later_token_publications_ns")
    if not isinstance(later, list) or len(later) != expected_output - 1 or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in later):
        _fail(f"{label}: token publication count is invalid")
    ordered.extend(later)
    ordered.extend([events.get("stop_ns"), events.get("cleanup_ns")])
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ordered) or any(right <= left for left, right in zip(ordered, ordered[1:])):
        _fail(f"{label}: event timestamps are not strictly ordered")
    for key in ("ttft_ns", "prefill_ns", "e2e_ns"):
        if isinstance(derived.get(key), bool) or not isinstance(derived.get(key), int) or derived[key] <= 0:
            _fail(f"{label}: derived {key} is invalid")
    if derived["ttft_ns"] != events["first_token_ns"] - events["request_start_ns"] or derived["prefill_ns"] != events["prefill_complete_ns"] - events["prefill_submit_ns"] or derived["e2e_ns"] != events["cleanup_ns"] - events["request_start_ns"]:
        _fail(f"{label}: event/derived timing mismatch")
    prefill_rate = derived.get("prefill_tokens_per_second")
    if isinstance(prefill_rate, bool) or not isinstance(prefill_rate, (int, float)) or not math.isfinite(float(prefill_rate)) or prefill_rate <= 0:
        _fail(f"{label}: prefill rate is invalid")
    tpot = derived.get("tpot_ns")
    if not isinstance(tpot, list) or len(tpot) != expected_output - 1 or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in tpot):
        _fail(f"{label}: TPOT count/value is invalid")
    previous = events["first_token_ns"]
    for item, publication in zip(tpot, later):
        if item != publication - previous:
            _fail(f"{label}: TPOT/event timing mismatch")
        previous = publication
    if derived.get("decode_tokens") != expected_output - 1:
        _fail(f"{label}: decode token count is invalid")
    decode_rate = derived.get("decode_tokens_per_second")
    if isinstance(decode_rate, bool) or not isinstance(decode_rate, (int, float)) or not math.isfinite(float(decode_rate)) or decode_rate <= 0:
        _fail(f"{label}: decode rate is invalid")


def _validate_direct_shape(result: Mapping[str, Any], row: Mapping[str, Any], label: str) -> None:
    required = ("lane_definition", "model_load", "session_cleanup", "memory")
    if any(key not in result for key in required) or result.get("lane_definition") != "pretokenized direct engine: request start excludes render/tokenize":
        _fail(f"{label}: direct-v2 required lifecycle fields are absent")
    model_load = result.get("model_load")
    if not isinstance(model_load, dict) or model_load.get("event") != "model_load" or model_load.get("load_count") != 1:
        _fail(f"{label}: model-load identity is invalid")
    for key in ("start_ns", "model_ready_ns", "duration_ns"):
        if isinstance(model_load.get(key), bool) or not isinstance(model_load.get(key), int) or model_load[key] < 0:
            _fail(f"{label}: model-load timing is invalid")
    if model_load["duration_ns"] <= 0 or model_load["model_ready_ns"] - model_load["start_ns"] != model_load["duration_ns"]:
        _fail(f"{label}: model-load timing is inconsistent")
    session_cleanup = result.get("session_cleanup")
    if not isinstance(session_cleanup, dict) or session_cleanup.get("retryable_cleanup") != 0 or session_cleanup.get("durable_quarantine") != 0:
        _fail(f"{label}: session cleanup is not empty")
    memory = result.get("memory")
    if not isinstance(memory, dict):
        _fail(f"{label}: top-level memory evidence is absent")
    for key in ("placement_total_memory_bytes", "placement_available_memory_bytes", "placement_required_bytes", "placement_model_resident_bytes", "placement_request_state_bytes", "placement_safety_reserve_bytes", "workspace_separate_allocation_bytes", "workspace_arena_bytes", "model_resident_high_water_bytes", "resident_vram_bytes", "peak_vram_bytes"):
        if isinstance(memory.get(key), bool) or not isinstance(memory.get(key), int) or memory[key] < 0:
            _fail(f"{label}: top-level memory field {key} is invalid")
    for key in ("model_ready", "after_model_drop"):
        _snapshot(memory.get(key), f"{label}.memory.{key}")
    if memory.get("resident_vram_source") != "model_resident_allocator_high_water" or memory.get("peak_source") != "runtime_allocator":
        _fail(f"{label}: top-level memory source identity is invalid")
    identities = result.get("identities")
    if not isinstance(identities, dict) or identities.get("engine") != "sllm" or identities.get("backend") != "hip" or isinstance(identities.get("session_id"), bool) or not isinstance(identities.get("session_id"), int) or identities.get("session_id", 0) <= 0 or identities.get("device_index") != 0 or identities.get("target") != TARGET:
        _fail(f"{label}: direct identity binding is invalid")
    model_identity = identities.get("model")
    binding = identities.get("binding")
    if not isinstance(model_identity, dict) or model_identity.get("model_size") != MODEL_SIZE or model_identity.get("repo_id") != "Qwen/Qwen3.5-4B" or model_identity.get("resolved_revision") != MODEL_REVISION:
        _fail(f"{label}: direct model identity is invalid")
    lock_fingerprint = _digest(model_identity.get("lock_fingerprint"), f"{label}.model.lock_fingerprint")
    if not isinstance(binding, dict) or _digest(binding.get("model_fingerprint"), f"{label}.binding.model_fingerprint") != lock_fingerprint:
        _fail(f"{label}: model fingerprint binding is invalid")
    _digest(binding.get("plan_digest"), f"{label}.binding.plan_digest")
    audit = result.get("audit")
    if not isinstance(audit, dict) or audit.get("device_index") != 0 or audit.get("model_load_count") != 1 or audit.get("request_model_load_count") != 0 or audit.get("model_reused") is not True:
        _fail(f"{label}: aggregate audit lifecycle is invalid")
    for key in ("submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"):
        _positive_int(audit.get(key), f"{label}.audit.{key}")
    if audit.get("model_fingerprint") != lock_fingerprint or _digest(audit.get("plan_digest"), f"{label}.audit.plan_digest") != binding["plan_digest"]:
        _fail(f"{label}: aggregate audit binding differs from identity")
    config = result.get("config")
    if not isinstance(config, dict) or config.get("tokenizer") is not False or config.get("render") is not False or config.get("lane") != "direct" or config.get("kv_cache_encoding") != "fp16" or isinstance(config.get("effective_context_length"), bool) or not isinstance(config.get("effective_context_length"), int) or config.get("effective_context_length") <= 0 or isinstance(config.get("completion_timeout_seconds"), bool) or not isinstance(config.get("completion_timeout_seconds"), int) or config.get("completion_timeout_seconds") <= 0:
        _fail(f"{label}: direct lane/config identity is invalid")
    stop_policy = config.get("stop_policy")
    if not isinstance(stop_policy, dict) or stop_policy.get("visible_stop_tokens") is not False or stop_policy.get("ignore_eos") is not row["ignore_eos"] or stop_policy.get("stop_token_ids") != ([] if row["ignore_eos"] else list(STOP_IDS)):
        _fail(f"{label}: stop policy is invalid")


def _sample_contract(sample: Mapping[str, Any], expected_input: list[int], expected_output: int, control: Mapping[str, Any], model_ready_bytes: int, sample_index: int, label: str) -> list[int]:
    tokens = sample.get("tokens")
    if not isinstance(tokens, dict):
        _fail(f"{label}: tokens are absent")
    input_ids = _tokens(tokens, "input_token_ids", label)
    generated = _tokens(tokens, "generated_token_ids", label)
    visible = _tokens(tokens, "visible_token_ids", label)
    decode_input = _tokens(tokens, "decode_input_token_ids", label)
    if input_ids != expected_input:
        _fail(f"{label}: fixed input token IDs changed")
    if len(generated) != expected_output or len(visible) != expected_output or visible != generated or decode_input != generated[:-1]:
        _fail(f"{label}: output/decode token shape is invalid")
    audit = sample.get("audit")
    if not isinstance(audit, dict) or audit.get("selected_backend") != "hip" or audit.get("target") != TARGET or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        _fail(f"{label}: HIP dispatch evidence is absent")
    control_audit = control.get("audit")
    dispatch_fields = (
        "selected_backend",
        "target",
        "device_index",
        "model_fingerprint",
        "plan_digest",
        "fallback_used",
        "all_dispatches_hip",
        "submission_count",
        "kernel_dispatch_count",
        "segment_count",
        "boundary_count",
    )
    if not isinstance(control_audit, dict) or any(audit.get(key) != control_audit.get(key) for key in dispatch_fields):
        _fail(f"{label}: correctness reference dispatch identity/counts differ")
    _request_memory(sample.get("memory"), model_ready_bytes, f"{label} memory")
    cleanup = sample.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("sample_index") != sample_index or cleanup.get("request_dropped") is not True or cleanup.get("allocator_cleanup_validated") is not True or cleanup.get("retryable_cleanup") != 0 or cleanup.get("durable_quarantine") != 0:
        _fail(f"{label}: per-sample cleanup evidence is invalid")
    _validate_no_fallback(sample, label)
    _validate_cleanup(sample, label)
    control_tokens = control.get("tokens")
    if not isinstance(control_tokens, dict):
        _fail(f"{label}: control tokens are absent")
    for key in ("generated_token_ids", "visible_token_ids", "decode_input_token_ids"):
        if _tokens(tokens, key, label) != _tokens(control_tokens, key, f"{label} control"):
            _fail(f"{label}: correctness control mismatch in {key}")
    if sample.get("stop") != control.get("stop"):
        _fail(f"{label}: correctness control stop mismatch")
    stop = sample.get("stop")
    if not isinstance(stop, dict) or stop.get("version") != 1 or stop.get("reason_version") != 1 or stop.get("kind") != "max_new_tokens" or stop.get("token_id") is not None:
        _fail(f"{label}: fixed max-token stop identity is invalid")
    return generated


def validate_result(document: Any, row: Mapping[str, Any], weight: str) -> dict[str, Any]:
    if not isinstance(document, dict):
        _fail(f"{row['row_id']}: benchmark report is not an object")
    # `sllm benchmark` emits the direct result itself (unlike generate, whose
    # public frontend report wraps it in an outer `result` member).  Accept
    # both forms, but never accept a wrapper whose state is not PASS.
    if isinstance(document.get("result"), dict):
        if document.get("state") != "PASS":
            _fail(f"{row['row_id']}: outer report/result is not PASS")
        result = document["result"]
    else:
        result = document
    if result.get("benchmark_schema_version") != DIRECT_SCHEMA_VERSION or result.get("state") != "PASS" or result.get("lane") != "direct":
        _fail(f"{row['row_id']}: direct schema/lane is invalid")
    _validate_direct_shape(result, row, str(row["row_id"]))
    config = result.get("config")
    if not isinstance(config, dict) or config.get("input_token_ids") != row["input_token_ids"] or config.get("input_token_count") != row["input_token_count"] or config.get("max_new_tokens") != row["requested_output_tokens"] or config.get("greedy") is not True or config.get("warmups") != row["warmups"] or config.get("measured") != row["measured"] or config.get("context_length") != row["context_length"] or config.get("ignore_eos") is not row["ignore_eos"] or config.get("prefill_chunk_tokens") is not None:
        _fail(f"{row['row_id']}: fixed direct configuration is invalid")
    effective_chunk = config.get("effective_prefill_chunk_tokens")
    allowed_chunks: list[int] = []
    for candidate in (16_384, 8_192, 4_096, 2_048, 512):
        bounded = min(row["input_token_count"], candidate)
        if bounded not in allowed_chunks:
            allowed_chunks.append(bounded)
    if effective_chunk not in allowed_chunks:
        _fail(f"{row['row_id']}: effective prefill chunk is outside the deterministic placement candidates")
    row_identity = result.get("row")
    if not isinstance(row_identity, dict) or row_identity.get("row_id") != row["row_id"] or row_identity.get("case_id") != row["case_id"] or row_identity.get("model_size") != MODEL_SIZE or row_identity.get("input_token_ids") != row["input_token_ids"] or row_identity.get("input_token_count") != row["input_token_count"] or row_identity.get("requested_output_tokens") != row["requested_output_tokens"]:
        _fail(f"{row['row_id']}: direct row identity is invalid")
    identities = result.get("identities")
    audit = result.get("audit")
    if not isinstance(identities, dict) or identities.get("target") != TARGET or not isinstance(audit, dict) or audit.get("selected_backend") != "hip" or audit.get("target") != TARGET or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        _fail(f"{row['row_id']}: exact gfx1201/HIP identity is invalid")
    model_identity = identities.get("model") if isinstance(identities, dict) else None
    if not isinstance(model_identity, dict) or model_identity.get("model_size") != MODEL_SIZE or model_identity.get("repo_id") != "Qwen/Qwen3.5-4B" or model_identity.get("resolved_revision") != MODEL_REVISION:
        _fail(f"{row['row_id']}: model identity is not the fixed Qwen3.5-4B")
    if weight == "bf16" and audit.get("weight_encoding") != "bf16":
        _fail(f"{row['row_id']}: BF16 weight encoding is not selected")
    if weight == "fp8" and (audit.get("fp8_provider") != "native-fnuz" or audit.get("weight_encoding") != "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32"):
        _fail(f"{row['row_id']}: embedded FNUZ FP8 provider/encoding is not selected")
    _validate_no_fallback(result, str(row["row_id"]))
    _validate_cleanup(result, str(row["row_id"]))
    cleanup = result.get("cleanup")
    expected_cleanup_counts = {
        "correctness_control_request_count": 0,
        "warmup_request_count": row["warmups"],
        "measured_request_count": row["measured"],
        "request_cleanup_count": row["warmups"] + row["measured"],
        "performance_sample_count": row["warmups"] + row["measured"],
    }
    if not isinstance(cleanup, dict) or any(cleanup.get(key) != expected for key, expected in expected_cleanup_counts.items()) or cleanup.get("correctness_control_source") != "first-warmup-sample" or cleanup.get("correctness_control_reference_sample_index") != 0 or cleanup.get("all_requests_dropped") is not True or cleanup.get("retryable_cleanup") != 0 or cleanup.get("durable_quarantine") != 0:
        _fail(f"{row['row_id']}: cleanup request counts or terminal drop evidence is invalid")
    control = result.get("correctness_control")
    if not isinstance(control, dict) or control.get("label") != "correctness-reference" or control.get("execution_path") != "first-warmup-sample" or control.get("timing_instrumentation") != "on" or control.get("included_in_performance_statistics") is not False or control.get("source") != {"kind": "warmup-sample", "sample_index": 0, "request_count": 0}:
        _fail(f"{row['row_id']}: correctness control is absent")
    control_tokens = control.get("tokens")
    control_audit = control.get("audit")
    if not isinstance(control_tokens, dict) or _tokens(control_tokens, "input_token_ids", str(row["row_id"]) + " control") != row["input_token_ids"] or not isinstance(control_audit, dict) or control_audit.get("selected_backend") != "hip" or control_audit.get("target") != TARGET or control_audit.get("all_dispatches_hip") is not True or control_audit.get("fallback_used") is not False:
        _fail(f"{row['row_id']}: correctness control fixed input mismatch")
    model_ready_bytes = result["memory"]["model_ready"]["current_bytes"]
    _request_memory(control.get("memory"), model_ready_bytes, f"{row['row_id']} control memory")
    if control_audit.get("model_fingerprint") != result["identities"]["model"]["lock_fingerprint"] or control_audit.get("plan_digest") != result["identities"]["binding"]["plan_digest"]:
        _fail(f"{row['row_id']}: correctness control identity binding differs")
    _validate_no_fallback(control, str(row["row_id"]) + " control")
    _validate_cleanup(control, str(row["row_id"]) + " control")
    control_cleanup = control.get("cleanup")
    comparison = control.get("comparison")
    if not isinstance(control_cleanup, dict) or control_cleanup.get("reference_sample") is not True or control_cleanup.get("request_dropped") is not True or control_cleanup.get("allocator_cleanup_validated") is not True or control_cleanup.get("retryable_cleanup") != 0 or control_cleanup.get("durable_quarantine") != 0:
        _fail(f"{row['row_id']}: correctness reference cleanup is invalid")
    if not isinstance(comparison, dict) or comparison.get("mode") != "exact" or comparison.get("scope") != "first_warmup_reference_against_every_remaining_warmup_and_measured_sample" or comparison.get("reference_source") != "warmups.samples[0]" or comparison.get("dispatch_count_rule") != "exact_when_token_and_stop_fields_match":
        _fail(f"{row['row_id']}: correctness reference comparison contract is invalid")
    warmups = result.get("warmups")
    measured = result.get("measured")
    if not isinstance(warmups, dict) or warmups.get("count") != row["warmups"] or not isinstance(warmups.get("samples"), list) or len(warmups["samples"]) != row["warmups"]:
        _fail(f"{row['row_id']}: warmup count differs")
    if not isinstance(measured, dict) or measured.get("count") != row["measured"] or not isinstance(measured.get("samples"), list) or len(measured["samples"]) != row["measured"]:
        _fail(f"{row['row_id']}: measured count differs")
    first_warmup = warmups["samples"][0]
    if not isinstance(first_warmup, dict) or any(control.get(key) != first_warmup.get(key) for key in ("tokens", "stop", "audit", "memory")):
        _fail(f"{row['row_id']}: correctness reference is not the first warmup sample")
    generated: list[int] | None = None
    all_samples: list[Mapping[str, Any]] = []
    for section, samples in (("warmup", warmups["samples"]), ("measured", measured["samples"])):
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                _fail(f"{row['row_id']}: {section} sample {index} is malformed")
            _validate_timing(sample, row["requested_output_tokens"], f"{row['row_id']} {section} sample {index}")
            current = _sample_contract(sample, row["input_token_ids"], row["requested_output_tokens"], control, model_ready_bytes, index, f"{row['row_id']} {section} sample {index}")
            all_samples.append(sample)
            if generated is None:
                generated = current
    expected_audit_counts = {
        key: sum(int(sample["audit"][key]) for sample in all_samples)
        for key in ("submission_count", "kernel_dispatch_count", "segment_count", "boundary_count")
    }
    expected_request_count = row["warmups"] + row["measured"]
    if any(audit.get(key) != value for key, value in expected_audit_counts.items()) or audit.get("sample_count") != expected_request_count or audit.get("correctness_control_request_count") != 0 or audit.get("correctness_control_source") != "first-warmup-sample" or audit.get("correctness_control_reference_sample_index") != 0 or audit.get("total_request_count") != expected_request_count:
        _fail(f"{row['row_id']}: aggregate audit counts or request accounting are invalid")
    control_generated = _tokens(control_tokens, "generated_token_ids", str(row["row_id"]) + " control")
    if generated is None or generated != control_generated:
        _fail(f"{row['row_id']}: sample/control output mismatch")
    if row["case_id"] in {"long-10001", "long-100000"} and generated != [23066, 23066]:
        _fail(f"{row['row_id']}: BF16 long output is not [23066,23066]")
    return result


def _write_monitor_tsv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    lines = ["timestamp_ns\thbm_bytes\tgtt_bytes\n"]
    for sample in samples:
        lines.append(f"{sample['timestamp_ns']}\t{sample['hbm_bytes']}\t{sample['gtt_bytes']}\n")
    _atomic_write(path, "".join(lines).encode("ascii"), "monitor TSV")


def _failure_report(
    row_dir: Path,
    row: Mapping[str, Any],
    weight: str,
    binary_identity: Mapping[str, Any],
    model: Path,
    lock: Path,
    gpu_uuid: str,
    gpu_bdf: str | None,
    command: Sequence[str],
    environment: Mapping[str, Any],
    before_process: Mapping[str, Any],
    after_process: Mapping[str, Any],
    capture: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    settled: Mapping[str, Any] | None,
    monitor: _SysfsMonitor,
    raw_paths: Mapping[str, Path],
    kind: str,
    reason: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": ROW_SCHEMA_VERSION,
        "state": "FAIL",
        "row": dict(row),
        "weight": weight,
        "binary": dict(binary_identity),
        "model": file_identity(model, f"{weight} GGUF"),
        "lock": file_identity(lock, f"{weight} derived lock"),
        "target": TARGET,
        "gpu_uuid": gpu_uuid,
        "gpu_bdf": gpu_bdf,
        "device_index": row["device_index"],
        "command": list(command),
        "environment": {"ROCR_VISIBLE_DEVICES": environment.get("ROCR_VISIBLE_DEVICES"), "row": environment.get("SLLM_PHASE50_R9700_ROW")},
        "process": {"pre": dict(before_process), "post": dict(after_process), "capture": {key: value for key, value in capture.items() if key not in {"stdout", "stderr"}}},
        "memory": {"baseline": baseline, "settled": settled},
        "monitor": {"cadence_ms": 100, "samples": len(monitor.samples), "errors": list(monitor.errors), "tsv": str(raw_paths["monitor_tsv"].resolve())},
        "raw": {
            "stdout": {"path": str(raw_paths["stdout"].resolve()), "sha256": _sha256_bytes(raw_paths["stdout"].read_bytes())},
            "stderr": {"path": str(raw_paths["stderr"].resolve()), "sha256": _sha256_bytes(raw_paths["stderr"].read_bytes())},
            "monitor_tsv": {"path": str(raw_paths["monitor_tsv"].resolve()), "sha256": _sha256_bytes(raw_paths["monitor_tsv"].read_bytes())},
        },
        "failure": {"kind": kind, "reason": _bounded_reason(reason)},
    }
    input_token_path = raw_paths.get("input_token_ids")
    if input_token_path is not None:
        report["raw"]["input_token_ids"] = {"path": str(input_token_path.resolve()), "sha256": _sha256_bytes(input_token_path.read_bytes())}
    _atomic_write(row_dir / "row.json", canonical_bytes(report), "failed row report")
    return report


def _validate_external_resources(report: Mapping[str, Any], label: str) -> None:
    process = report.get("process")
    if not isinstance(process, dict) or not isinstance(process.get("capture"), dict) or process["capture"].get("process_group_gone") is not True:
        _fail(f"{label}: process group cleanup evidence is absent")
    memory = report.get("memory")
    if not isinstance(memory, dict) or not isinstance(memory.get("baseline"), dict) or not isinstance(memory.get("settled"), dict):
        _fail(f"{label}: HBM/GTT resource evidence is absent")
    baseline = memory["baseline"]
    settled = memory["settled"]
    if settled.get("settled") is not True or settled.get("hbm_bytes") != baseline.get("hbm_bytes") or settled.get("gtt_bytes") != baseline.get("gtt_bytes"):
        _fail(f"{label}: HBM/GTT resources did not return to baseline")
    monitor = report.get("monitor")
    if not isinstance(monitor, dict) or monitor.get("errors") not in ([], None) or not isinstance(monitor.get("samples"), int) or monitor.get("samples") <= 0:
        _fail(f"{label}: external resource monitor evidence is invalid")


def _verify_raw_item(item: Mapping[str, Any], label: str) -> None:
    path_value = item.get("path")
    digest = item.get("sha256")
    if not isinstance(path_value, str) or not isinstance(digest, str) or not DIGEST_RE.fullmatch(f"sha256:{digest}"):
        _fail(f"{label}: raw manifest is malformed")
    path = Path(path_value)
    if path.is_symlink() or not path.is_file() or _sha256_bytes(path.read_bytes()) != digest:
        _fail(f"{label}: raw artifact is missing or changed")


def _validate_failure_row(report: Mapping[str, Any], row: Mapping[str, Any], binary_identity: Mapping[str, Any], model: Path, lock: Path) -> None:
    label = str(row["row_id"])
    if report.get("schema_version") != ROW_SCHEMA_VERSION or report.get("state") != "FAIL" or report.get("row") != dict(row):
        _fail(f"{label}: existing failed row identity differs")
    if report.get("target") != TARGET or report.get("gpu_uuid") != GPU_UUID or report.get("gpu_bdf") != GPU_BDF:
        _fail(f"{label}: existing failed row GPU identity differs")
    if report.get("binary", {}).get("sha256") != binary_identity.get("sha256") or report.get("model", {}).get("sha256") != file_identity(model, "bf16 GGUF")["sha256"] or report.get("lock", {}).get("sha256") != file_identity(lock, "bf16 derived lock")["sha256"]:
        _fail(f"{label}: existing failed row source identity differs")
    failure = report.get("failure")
    if not isinstance(failure, dict) or not isinstance(failure.get("kind"), str) or not isinstance(failure.get("reason"), str) or not failure["kind"] or not failure["reason"] or len(failure["reason"]) > MAX_FAILURE_REASON_CHARS:
        _fail(f"{label}: existing failed row failure evidence is malformed")
    process = report.get("process")
    memory = report.get("memory")
    monitor = report.get("monitor")
    if not isinstance(report.get("command"), list) or not report["command"] or not isinstance(process, dict) or not isinstance(memory, dict) or not isinstance(monitor, dict):
        _fail(f"{label}: existing failed row resource evidence is absent")
    raw = report.get("raw")
    if not isinstance(raw, dict):
        _fail(f"{label}: existing failed row raw manifest is absent")
    for key in ("stdout", "stderr", "monitor_tsv"):
        item = raw.get(key)
        if not isinstance(item, dict):
            _fail(f"{label}: existing failed row raw {key} manifest is absent")
        _verify_raw_item(item, f"{label} {key}")


def _row_dir_state(
    row_dir: Path,
    row: Mapping[str, Any],
    weight: str,
    binary_identity: Mapping[str, Any],
    model: Path,
    lock: Path,
) -> dict[str, Any] | None:
    report_path = row_dir / "row.json"
    entries = list(row_dir.iterdir()) if row_dir.exists() else []
    if report_path.exists() and report_path.is_file():
        try:
            report = _json_load(report_path.read_bytes(), f"{row['row_id']} row report")
        except OSError as exc:
            _fail(f"cannot read existing row report: {exc}")
        if isinstance(report, dict) and report.get("state") == "FAIL":
            _validate_failure_row(report, row, binary_identity, model, lock)
            return report
        if not isinstance(report, dict) or report.get("state") != "PASS":
            _fail(f"existing row is incomplete and will not be overwritten: {row['row_id']}")
        if report.get("row", {}).get("row_id") != row["row_id"]:
            _fail(f"existing row identity differs: {row['row_id']}")
        if report.get("row") != dict(row):
            _fail(f"existing row matrix identity differs: {row['row_id']}")
        if report.get("binary", {}).get("sha256") != binary_identity.get("sha256"):
            _fail(f"existing row binary identity differs: {row['row_id']}")
        if report.get("model", {}).get("sha256") != file_identity(model, f"{weight} GGUF")["sha256"] or report.get("lock", {}).get("sha256") != file_identity(lock, f"{weight} derived lock")["sha256"]:
            _fail(f"existing row model/lock identity differs: {row['row_id']}")
        raw = report.get("raw")
        if not isinstance(raw, dict):
            _fail(f"existing row raw manifest is absent: {row['row_id']}")
        for key in ("stdout", "stderr", "monitor_tsv"):
            item = raw.get(key)
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                _fail(f"existing row raw manifest is malformed: {row['row_id']} {key}")
            path = Path(item["path"])
            if path.is_symlink() or not path.is_file() or _sha256_bytes(path.read_bytes()) != item["sha256"]:
                _fail(f"existing row raw artifact is missing or changed: {row['row_id']} {key}")
        _validate_external_resources(report, str(row["row_id"]))
        if len(row["input_token_ids"]) > 10_001:
            item = raw.get("input_token_ids")
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                _fail(f"existing row input-token artifact is malformed: {row['row_id']}")
            path = Path(item["path"])
            if path.is_symlink() or not path.is_file() or _sha256_bytes(path.read_bytes()) != item["sha256"]:
                _fail(f"existing row input-token artifact is missing or changed: {row['row_id']}")
        validate_result({"state": "PASS", "result": report.get("result")}, row, weight)
        return report
    if entries:
        _fail(f"partial raw row exists and will not be overwritten: {row['row_id']}")
    return None


def run_row(
    binary: Path,
    binary_identity: Mapping[str, Any],
    model: Path,
    lock: Path,
    row: Mapping[str, Any],
    weight: str,
    output_dir: Path,
    sysfs_root: Path,
    timeout_seconds: float,
    amd_smi: Path | None = None,
    gpu_bdf: str | None = None,
    gpu_uuid: str = GPU_UUID,
) -> dict[str, Any]:
    row_dir = output_dir / "raw" / str(row["row_id"])
    existing = _row_dir_state(row_dir, row, weight, binary_identity, model, lock)
    if existing is not None:
        return existing
    row_dir.mkdir(parents=True, exist_ok=False)
    input_token_path: Path | None = None
    if len(row["input_token_ids"]) > 10_001:
        input_token_path = row_dir / "input-token-ids.csv"
        input_bytes = (",".join(str(value) for value in row["input_token_ids"]) + "\n").encode("ascii")
        _atomic_write(input_token_path, input_bytes, "input token IDs")
    baseline: Mapping[str, Any] | None = None
    settled: Mapping[str, Any] | None = None
    baseline_error: str | None = None
    try:
        baseline = read_sysfs_memory(sysfs_root)
    except SessionDError as exc:
        baseline_error = str(exc)
    before_process = _safe_process_snapshot(binary, amd_smi=amd_smi, gpu_bdf=gpu_bdf)
    command = _expected_command(binary, model, lock, row, input_token_path)
    monitor = _SysfsMonitor(sysfs_root)
    monitor.start()
    env = _execution_environment(str(row["row_id"]), int(row["device_index"]), gpu_uuid=gpu_uuid)
    capture: dict[str, Any] = _empty_capture()
    execution_error: str | None = None
    try:
        if baseline_error is not None:
            execution_error = baseline_error
        else:
            capture = _run_command(command, env, timeout_seconds)
    except SessionDError as exc:
        execution_error = str(exc)
        capture["error"] = _bounded_reason(execution_error)
    finally:
        monitor.finish()
    try:
        settled = _settled_memory(sysfs_root)
    except SessionDError as exc:
        settled = {"settled": False, "error": _bounded_reason(exc)}
    after_process = _safe_process_snapshot(binary, amd_smi=amd_smi, gpu_bdf=gpu_bdf)
    stdout_path = row_dir / "stdout.json"
    stderr_path = row_dir / "stderr.log"
    tsv_path = row_dir / "hbm-gtt.tsv"
    _atomic_write(stdout_path, capture["stdout"], "benchmark stdout")
    _atomic_write(stderr_path, capture["stderr"], "benchmark stderr")
    _write_monitor_tsv(tsv_path, monitor.samples)
    raw_paths: dict[str, Path] = {"stdout": stdout_path, "stderr": stderr_path, "monitor_tsv": tsv_path}
    if input_token_path is not None:
        raw_paths["input_token_ids"] = input_token_path
    validation_error: str | None = None
    result: dict[str, Any] | None = None
    if execution_error is None:
        try:
            if capture["timed_out"] or capture["exit_code"] != 0 or capture["process_group_gone"] is not True:
                pass
            elif monitor.errors or not monitor.samples:
                pass
            elif not baseline or not settled or settled.get("settled") is not True or settled.get("hbm_bytes") != baseline.get("hbm_bytes") or settled.get("gtt_bytes") != baseline.get("gtt_bytes"):
                pass
            else:
                document = _json_load(capture["stdout"], f"{row['row_id']} benchmark stdout")
                result = validate_result(document, row, weight)
        except SessionDError as exc:
            validation_error = str(exc)
    if result is None:
        kind, reason = _failure_class(capture, execution_error, monitor.errors, baseline, settled, validation_error)
        return _failure_report(row_dir, row, weight, binary_identity, model, lock, gpu_uuid, gpu_bdf, command, env, before_process, after_process, capture, baseline, settled, monitor, raw_paths, kind, reason)
    report = {
        "schema_version": ROW_SCHEMA_VERSION,
        "state": "PASS",
        "row": dict(row),
        "weight": weight,
        "binary": dict(binary_identity),
        "model": file_identity(model, f"{weight} GGUF"),
        "lock": file_identity(lock, f"{weight} derived lock"),
        "target": TARGET,
        "gpu_uuid": gpu_uuid,
        "gpu_bdf": gpu_bdf,
        "device_index": row["device_index"],
        "command": command,
        "environment": {"ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"), "row": env.get("SLLM_PHASE50_R9700_ROW")},
        "process": {"pre": before_process, "post": after_process, "capture": {key: value for key, value in capture.items() if key not in {"stdout", "stderr"}}},
        "memory": {"baseline": baseline, "settled": settled},
        "monitor": {"cadence_ms": 100, "samples": len(monitor.samples), "errors": monitor.errors, "tsv": str(tsv_path.resolve())},
        "raw": {
            "stdout": {"path": str(stdout_path.resolve()), "sha256": _sha256_bytes(capture["stdout"])},
            "stderr": {"path": str(stderr_path.resolve()), "sha256": _sha256_bytes(capture["stderr"])},
            "monitor_tsv": {"path": str(tsv_path.resolve()), "sha256": _sha256_bytes(tsv_path.read_bytes())},
        },
        "result": result,
    }
    if input_token_path is not None:
        report["raw"]["input_token_ids"] = {
            "path": str(input_token_path.resolve()),
            "sha256": _sha256_bytes(input_token_path.read_bytes()),
        }
    _atomic_write(row_dir / "row.json", canonical_bytes(report), "row report")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target != TARGET:
        _fail("Phase 50 is restricted to exact target gfx1201")
    if args.gpu_uuid != GPU_UUID:
        _fail(f"Phase 50 is restricted to canonical R9700 UUID {GPU_UUID}")
    if args.gpu_bdf != GPU_BDF:
        _fail(f"Phase 50 is restricted to canonical R9700 BDF {GPU_BDF}")
    if args.device_index < 0:
        _fail("device index must be nonnegative")
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    if output_dir == project_root or project_root in output_dir.parents:
        _fail("Phase 50 raw output must be outside the repository")
    binary = _regular_file(Path(args.binary), "benchmark binary", executable=True)
    binary_identity = file_identity(binary, "benchmark binary")
    bf16_model = _regular_file(Path(args.bf16_gguf), "BF16 GGUF")
    bf16_lock = _regular_file(Path(args.bf16_lock), "BF16 derived lock")
    sysfs_root = Path(args.sysfs_root)
    if not sysfs_root.is_dir():
        _fail(f"sysfs root is not a directory: {sysfs_root}")
    amd_smi = Path(args.amd_smi) if args.amd_smi else None
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = matrix()
    reports: list[dict[str, Any]] = []
    for row in rows:
        reports.append(run_row(binary, binary_identity, bf16_model, bf16_lock, {**row, "device_index": args.device_index}, "bf16", output_dir, sysfs_root, args.timeout_seconds, amd_smi, args.gpu_bdf, args.gpu_uuid))
    summary_state = "FAIL" if any(report.get("state") == "FAIL" for report in reports) else "PASS"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": summary_state,
        "target": TARGET,
        "gpu_uuid": GPU_UUID,
        "gpu_bdf": GPU_BDF,
        "device_index": args.device_index,
        "model_size": MODEL_SIZE,
        "binary": binary_identity,
        "protocol": {"normal": {"warmups": 3, "measured": 10}, "extended": {"warmups": 1, "measured": 3, "context_length": 131072}, "kv_cache_encoding": "fp16", "greedy": True, "batch_size": 1, "sequences": 1},
        "matrix": {"weights": ["bf16"], "cases": [case for case, _, _ in CASE_SPECS], "row_count": len(reports)},
        "models": {"bf16": file_identity(bf16_model, "BF16 GGUF"), "bf16_lock": file_identity(bf16_lock, "BF16 derived lock")},
        "rows": reports,
    }
    failures = [
        {"case_id": report["row"]["case_id"], "row_id": report["row"]["row_id"], "kind": report["failure"]["kind"], "reason": report["failure"]["reason"]}
        for report in reports if report.get("state") == "FAIL" and isinstance(report.get("failure"), dict)
    ]
    if failures:
        summary["failure_count"] = len(failures)
        summary["failures"] = failures
    summary_path = output_dir / "phase50-r9700-sllm-v1.json"
    if summary_path.exists():
        existing = _json_load(summary_path.read_bytes(), "existing Phase 50 summary")
        if existing != summary:
            _fail("existing Phase 50 sLLM summary differs; refusing overwrite")
    else:
        _atomic_write(summary_path, canonical_bytes(summary), "Phase 50 sLLM summary")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="exact sLLM benchmark executable")
    parser.add_argument("--bf16-gguf", "--bf16-model", dest="bf16_gguf", required=True)
    parser.add_argument("--bf16-lock", "--bf16-derived-lock", dest="bf16_lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sysfs-root", default="/sys")
    parser.add_argument("--amd-smi", default=DEFAULT_AMD_SMI, help="optional amd-smi process observer")
    parser.add_argument("--gpu-bdf", default=GPU_BDF, help="exact R9700 GPU BDF for amd-smi process observation")
    parser.add_argument("--gpu-uuid", default=GPU_UUID, help="canonical R9700 UUID used for ROCr visibility")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--resume", action="store_true", default=True, help="resume complete rows; incomplete rows are never overwritten")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        print("FAIL: --timeout-seconds must be positive", file=sys.stderr)
        return 2
    try:
        summary = run(args)
    except SessionDError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("state") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
