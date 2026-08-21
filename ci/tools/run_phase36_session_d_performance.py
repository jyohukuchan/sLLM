#!/usr/bin/env python3
"""Run the bounded Phase 36 Session D direct-performance matrix.

This producer is intentionally self contained so that it can be copied to the
MI300X VM together with the release binary and run without the repository's
model cache helpers.  The benchmark binary remains the authority for model
loading and direct-engine timing; this script owns the frozen input matrix,
process lifetime, raw-output retention, and fail-closed result checks.

All output is written below an explicitly supplied directory.  Raw stdout,
stderr, and 100 ms sysfs HBM/GTT samples are retained per row.  A completed
row is resumable, while an incomplete row is never silently overwritten.
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
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence


TARGET = "gfx942"
GPU_UUID = "GPU-1228c84fe776f2f4"
GPU_BDF = "0000:ff:00.0"
MODEL_SIZE = "4B"
SCHEMA_VERSION = "phase36-session-d-performance-v1"
ROW_SCHEMA_VERSION = "phase36-session-d-performance-row-v1"
DIRECT_SCHEMA_VERSION = "engine-performance-direct-v1"
WARMUPS = 3
MEASURED = 10
SAMPLE_COUNT = WARMUPS + MEASURED
MONITOR_PERIOD_SECONDS = 0.1
DEFAULT_TIMEOUT_SECONDS = 3600.0
MAX_STDOUT_BYTES = 128 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_MONITOR_SAMPLES = 100_000
DEFAULT_AMD_SMI = "/opt/rocm/bin/amd-smi"
STOP_IDS = (248046, 248044)
VISIBILITY_NAMES = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

SHORT_ODD = [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43]
LONG_INPUT = [23066] * 10_001


class SessionDError(RuntimeError):
    """Malformed evidence, execution failure, or unsafe output state."""


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
    if case_id == "decode-long":
        result = list(SHORT_ODD)
        result.extend(((index * 7919 + 41) % 248000) for index in range(len(result), 32))
        return result
    if case_id == "long-10001":
        return list(LONG_INPUT)
    _fail(f"unknown performance case: {case_id}")


CASE_SPECS: tuple[tuple[str, int, int], ...] = (
    ("short-odd", 17, 17),
    ("32-32", 32, 32),
    ("prefill-long", 1024, 128),
    ("decode-long", 32, 256),
    ("long-10001", 10_001, 2),
)


def matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in ("bf16", "fp8"):
        for case_id, input_count, output_count in CASE_SPECS:
            ids = input_ids_for(case_id)
            if len(ids) != input_count:
                _fail(f"matrix case {case_id} has {len(ids)} inputs, expected {input_count}")
            rows.append(
                {
                    "row_id": f"phase36-d-{weight}-{case_id}",
                    "weight": weight,
                    "model_size": MODEL_SIZE,
                    "case_id": case_id,
                    "input_token_ids": ids,
                    "input_token_count": input_count,
                    "requested_output_tokens": output_count,
                    "target": TARGET,
                    "device_index": 0,
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
        self.thread = threading.Thread(target=self._run, name="phase36-session-d-sysfs", daemon=True)
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


def _expected_command(binary: Path, model: Path, lock: Path, row: Mapping[str, Any]) -> list[str]:
    ids = ",".join(str(value) for value in row["input_token_ids"])
    return [
        str(binary),
        "benchmark",
        "--lane", "direct",
        "--gguf", str(model),
        "--derived-lock", str(lock),
        "--row-id", str(row["row_id"]),
        "--model-size", MODEL_SIZE,
        "--case-id", str(row["case_id"]),
        "--input-token-ids", ids,
        "--max-new-tokens", str(row["requested_output_tokens"]),
        "--device-index", str(row["device_index"]),
        "--target", TARGET,
        "--kv-cache-encoding", "fp16",
        "--greedy",
        "--warmups", str(WARMUPS),
        "--measured", str(MEASURED),
    ]


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
    # exact MI300X UUID rather than relying on ordinal discovery.
    env["ROCR_VISIBLE_DEVICES"] = gpu_uuid
    env["SLLM_PHASE36_SESSION_D_ROW"] = row_id
    # Do not inherit an arbitrary loader search path into the fixed ROCm lane.
    # The source/build identity separately pins this provider to ROCm 7.14.
    env["LD_LIBRARY_PATH"] = "/opt/rocm/lib"
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
        if "cleanup" in lowered or lowered in {"terminal_zero", "all_requests_dropped", "correctness_control_dropped"}:
            seen = True
            if isinstance(item, bool):
                if lowered in {"terminal_zero", "all_requests_dropped", "correctness_control_dropped"} and item is not True:
                    _fail(f"{label}: {key} is not true")
            elif lowered in zero_required and isinstance(item, int) and not isinstance(item, bool) and item != 0:
                _fail(f"{label}: {key} is nonzero")
    if not seen:
        _fail(f"{label}: cleanup evidence is absent")


def _tokens(value: Mapping[str, Any], key: str, label: str) -> list[int]:
    result = value.get(key)
    if not isinstance(result, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in result):
        _fail(f"{label}: {key} is malformed")
    return result


def _sample_contract(sample: Mapping[str, Any], expected_input: list[int], expected_output: int, control: Mapping[str, Any], label: str) -> list[int]:
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
    config = result.get("config")
    if not isinstance(config, dict) or config.get("input_token_ids") != row["input_token_ids"] or config.get("input_token_count") != row["input_token_count"] or config.get("max_new_tokens") != row["requested_output_tokens"] or config.get("greedy") is not True or config.get("warmups") != WARMUPS or config.get("measured") != MEASURED:
        _fail(f"{row['row_id']}: fixed direct configuration is invalid")
    row_identity = result.get("row")
    if not isinstance(row_identity, dict) or row_identity.get("row_id") != row["row_id"] or row_identity.get("case_id") != row["case_id"] or row_identity.get("model_size") != MODEL_SIZE or row_identity.get("input_token_ids") != row["input_token_ids"]:
        _fail(f"{row['row_id']}: direct row identity is invalid")
    identities = result.get("identities")
    audit = result.get("audit")
    if not isinstance(identities, dict) or identities.get("target") != TARGET or not isinstance(audit, dict) or audit.get("selected_backend") != "hip" or audit.get("target") != TARGET or audit.get("all_dispatches_hip") is not True or audit.get("fallback_used") is not False:
        _fail(f"{row['row_id']}: exact gfx942/HIP identity is invalid")
    model_identity = identities.get("model") if isinstance(identities, dict) else None
    if not isinstance(model_identity, dict) or model_identity.get("model_size") != MODEL_SIZE or model_identity.get("repo_id") != "Qwen/Qwen3.5-4B":
        _fail(f"{row['row_id']}: model identity is not the fixed Qwen3.5-4B")
    if weight == "bf16" and audit.get("weight_encoding") != "bf16":
        _fail(f"{row['row_id']}: BF16 weight encoding is not selected")
    if weight == "fp8" and (audit.get("fp8_provider") != "native-fnuz" or audit.get("weight_encoding") != "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32"):
        _fail(f"{row['row_id']}: embedded FNUZ FP8 provider/encoding is not selected")
    _validate_no_fallback(result, str(row["row_id"]))
    _validate_cleanup(result, str(row["row_id"]))
    cleanup = result.get("cleanup")
    expected_cleanup_counts = {
        "correctness_control_request_count": 1,
        "warmup_request_count": WARMUPS,
        "measured_request_count": MEASURED,
        "request_cleanup_count": SAMPLE_COUNT + 1,
        "performance_sample_count": SAMPLE_COUNT,
    }
    if not isinstance(cleanup, dict) or any(cleanup.get(key) != expected for key, expected in expected_cleanup_counts.items()) or cleanup.get("all_requests_dropped") is not True or cleanup.get("correctness_control_dropped") is not True or cleanup.get("retryable_cleanup") != 0 or cleanup.get("durable_quarantine") != 0:
        _fail(f"{row['row_id']}: cleanup request counts or terminal drop evidence is invalid")
    control = result.get("correctness_control")
    if not isinstance(control, dict):
        _fail(f"{row['row_id']}: correctness control is absent")
    control_tokens = control.get("tokens")
    control_audit = control.get("audit")
    if not isinstance(control_tokens, dict) or _tokens(control_tokens, "input_token_ids", str(row["row_id"]) + " control") != row["input_token_ids"] or not isinstance(control_audit, dict) or control_audit.get("selected_backend") != "hip" or control_audit.get("target") != TARGET or control_audit.get("all_dispatches_hip") is not True or control_audit.get("fallback_used") is not False:
        _fail(f"{row['row_id']}: correctness control fixed input mismatch")
    _validate_no_fallback(control, str(row["row_id"]) + " control")
    _validate_cleanup(control, str(row["row_id"]) + " control")
    warmups = result.get("warmups")
    measured = result.get("measured")
    if not isinstance(warmups, dict) or warmups.get("count") != WARMUPS or not isinstance(warmups.get("samples"), list) or len(warmups["samples"]) != WARMUPS:
        _fail(f"{row['row_id']}: warmup count is not 3")
    if not isinstance(measured, dict) or measured.get("count") != MEASURED or not isinstance(measured.get("samples"), list) or len(measured["samples"]) != MEASURED:
        _fail(f"{row['row_id']}: measured count is not 10")
    generated: list[int] | None = None
    for index, sample in enumerate(warmups["samples"] + measured["samples"]):
        if not isinstance(sample, dict):
            _fail(f"{row['row_id']}: sample {index} is malformed")
        current = _sample_contract(sample, row["input_token_ids"], row["requested_output_tokens"], control, f"{row['row_id']} sample {index}")
        if generated is None:
            generated = current
    control_generated = _tokens(control_tokens, "generated_token_ids", str(row["row_id"]) + " control")
    if generated is None or generated != control_generated:
        _fail(f"{row['row_id']}: sample/control output mismatch")
    if row["case_id"] == "long-10001" and weight == "bf16" and generated != [23066, 23066]:
        _fail(f"{row['row_id']}: BF16 long output is not [23066,23066]")
    return result


def _write_monitor_tsv(path: Path, samples: Sequence[Mapping[str, Any]]) -> None:
    lines = ["timestamp_ns\thbm_bytes\tgtt_bytes\n"]
    for sample in samples:
        lines.append(f"{sample['timestamp_ns']}\t{sample['hbm_bytes']}\t{sample['gtt_bytes']}\n")
    _atomic_write(path, "".join(lines).encode("ascii"), "monitor TSV")


def _row_dir_state(
    row_dir: Path,
    row: Mapping[str, Any],
    weight: str,
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
        if not isinstance(report, dict) or report.get("state") != "PASS":
            _fail(f"existing row is incomplete and will not be overwritten: {row['row_id']}")
        if report.get("row", {}).get("row_id") != row["row_id"]:
            _fail(f"existing row identity differs: {row['row_id']}")
        if report.get("row") != dict(row):
            _fail(f"existing row matrix identity differs: {row['row_id']}")
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
        validate_result({"state": "PASS", "result": report.get("result")}, row, weight)
        return report
    if entries:
        _fail(f"partial raw row exists and will not be overwritten: {row['row_id']}")
    return None


def run_row(
    binary: Path,
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
    existing = _row_dir_state(row_dir, row, weight, model, lock)
    if existing is not None:
        return existing
    row_dir.mkdir(parents=True, exist_ok=False)
    baseline = read_sysfs_memory(sysfs_root)
    before_process = process_snapshot(binary, amd_smi=amd_smi, gpu_bdf=gpu_bdf)
    command = _expected_command(binary, model, lock, row)
    monitor = _SysfsMonitor(sysfs_root)
    monitor.start()
    env = _execution_environment(str(row["row_id"]), int(row["device_index"]), gpu_uuid=gpu_uuid)
    capture: dict[str, Any]
    try:
        capture = _run_command(command, env, timeout_seconds)
    finally:
        monitor.finish()
    settled = _settled_memory(sysfs_root)
    after_process = process_snapshot(binary, amd_smi=amd_smi, gpu_bdf=gpu_bdf)
    stdout_path = row_dir / "stdout.json"
    stderr_path = row_dir / "stderr.log"
    tsv_path = row_dir / "hbm-gtt.tsv"
    _atomic_write(stdout_path, capture["stdout"], "benchmark stdout")
    _atomic_write(stderr_path, capture["stderr"], "benchmark stderr")
    _write_monitor_tsv(tsv_path, monitor.samples)
    if capture["timed_out"] or capture["exit_code"] != 0 or not capture["process_group_gone"]:
        _fail(f"{row['row_id']}: benchmark process failed (exit={capture['exit_code']}, timeout={capture['timed_out']})")
    if monitor.errors or not monitor.samples:
        _fail(f"{row['row_id']}: sysfs monitor failed: {monitor.errors or ['zero samples']}")
    document = _json_load(capture["stdout"], f"{row['row_id']} benchmark stdout")
    result = validate_result(document, row, weight)
    report = {
        "schema_version": ROW_SCHEMA_VERSION,
        "state": "PASS",
        "row": dict(row),
        "weight": weight,
        "model": file_identity(model, f"{weight} GGUF"),
        "lock": file_identity(lock, f"{weight} derived lock"),
        "target": TARGET,
        "gpu_uuid": gpu_uuid,
        "device_index": row["device_index"],
        "command": command,
        "environment": {"ROCR_VISIBLE_DEVICES": env.get("ROCR_VISIBLE_DEVICES"), "row": env.get("SLLM_PHASE36_SESSION_D_ROW")},
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
    _atomic_write(row_dir / "row.json", canonical_bytes(report), "row report")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target != TARGET:
        _fail("Session D is restricted to exact target gfx942")
    if args.gpu_uuid != GPU_UUID:
        _fail(f"Session D is restricted to exact MI300X UUID {GPU_UUID}")
    if args.device_index < 0:
        _fail("device index must be nonnegative")
    output_dir = Path(args.output_dir).resolve()
    project_root = Path(__file__).resolve().parents[2]
    if output_dir == project_root or project_root in output_dir.parents:
        _fail("Session D raw output must be outside the repository")
    binary = _regular_file(Path(args.binary), "benchmark binary", executable=True)
    bf16_model = _regular_file(Path(args.bf16_gguf), "BF16 GGUF")
    bf16_lock = _regular_file(Path(args.bf16_lock), "BF16 derived lock")
    fp8_model = _regular_file(Path(args.fp8_gguf), "FP8 GGUF")
    fp8_lock = _regular_file(Path(args.fp8_lock), "FP8 derived lock")
    sysfs_root = Path(args.sysfs_root)
    if not sysfs_root.is_dir():
        _fail(f"sysfs root is not a directory: {sysfs_root}")
    amd_smi = Path(args.amd_smi) if args.amd_smi else None
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = matrix()
    reports: list[dict[str, Any]] = []
    model_args = {"bf16": (bf16_model, bf16_lock), "fp8": (fp8_model, fp8_lock)}
    for row in rows:
        model, lock = model_args[row["weight"]]
        reports.append(run_row(binary, model, lock, {**row, "device_index": args.device_index}, row["weight"], output_dir, sysfs_root, args.timeout_seconds, amd_smi, args.gpu_bdf, args.gpu_uuid))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "target": TARGET,
        "gpu_uuid": GPU_UUID,
        "device_index": args.device_index,
        "model_size": MODEL_SIZE,
        "protocol": {"warmups": WARMUPS, "measured": MEASURED, "kv_cache_encoding": "fp16", "greedy": True},
        "matrix": {"weights": ["bf16", "fp8"], "cases": [case for case, _, _ in CASE_SPECS], "row_count": len(reports)},
        "models": {"bf16": file_identity(bf16_model, "BF16 GGUF"), "bf16_lock": file_identity(bf16_lock, "BF16 derived lock"), "fp8": file_identity(fp8_model, "FP8 GGUF"), "fp8_lock": file_identity(fp8_lock, "FP8 derived lock")},
        "rows": reports,
    }
    summary_path = output_dir / "phase36-session-d-performance-v1.json"
    if summary_path.exists():
        existing = _json_load(summary_path.read_bytes(), "existing Session D summary")
        if existing != summary:
            _fail("existing Session D summary differs; refusing overwrite")
    else:
        _atomic_write(summary_path, canonical_bytes(summary), "Session D summary")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="exact sLLM benchmark executable")
    parser.add_argument("--bf16-gguf", "--bf16-model", dest="bf16_gguf", required=True)
    parser.add_argument("--bf16-lock", "--bf16-derived-lock", dest="bf16_lock", required=True)
    parser.add_argument("--fp8-gguf", "--fp8-model", dest="fp8_gguf", required=True)
    parser.add_argument("--fp8-lock", "--fp8-derived-lock", dest="fp8_lock", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sysfs-root", default="/sys")
    parser.add_argument("--amd-smi", default=DEFAULT_AMD_SMI, help="optional amd-smi process observer")
    parser.add_argument("--gpu-bdf", default=GPU_BDF, help="exact GPU BDF for amd-smi process observation")
    parser.add_argument("--gpu-uuid", default=GPU_UUID, help="exact MI300X UUID used for ROCr visibility")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
