#!/usr/bin/env python3
"""Produce the fixed llama.cpp Phase 49 V620 comparison rows.

The llama wrapper owns model execution and per-request timing.  This producer
owns the immutable token matrix, exact process/visibility contract, 100 ms
HBM/GTT observation, raw retention, and resumable publication boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, NoReturn, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET = "gfx1030"
GPU_UUID = "GPU-76a08c022586fed6"
LLAMA_COMMIT = "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70"
LLAMA_TAG = "b10453"
SCHEMA_VERSION = "phase49-v620-llama-v1"
ROW_SCHEMA_VERSION = "phase49-v620-llama-row-v1"
WRAPPER_SCHEMA_VERSION = "llama-phase49-v620-v1"
MONITOR_PERIOD_SECONDS = 0.1
DEFAULT_TIMEOUT_SECONDS = 86_400.0
MAX_STDOUT_BYTES = 128 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_MONITOR_SAMPLES = 100_000
STOP_IDS = (248046, 248044)
MAX_TOKEN_ID = 248319
VISIBILITY_NAMES = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")

SHORT_ODD = [1, 3, 17, 37, 73, 255, 256, 257, 2, 5, 11, 19, 23, 29, 31, 41, 43]
LONG_INPUT = [23066] * 10_001
VERY_LONG_INPUT = [23066] * 100_000


class SessionDLlamaError(RuntimeError):
    """Malformed evidence, execution failure, or unsafe publication state."""


def _fail(message: str) -> NoReturn:
    raise SessionDLlamaError(message)


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
    except SessionDLlamaError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label}: malformed JSON: {exc}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def regular_file(path: Path, label: str, *, executable: bool = False) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file: {path}")
    if path.stat().st_size <= 0:
        _fail(f"{label} is empty: {path}")
    if executable and not os.access(path, os.X_OK):
        _fail(f"{label} is not executable: {path}")
    return path


def file_identity(path: Path, label: str) -> dict[str, Any]:
    path = regular_file(path, label)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
    _fail(f"unknown case: {case_id}")


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
            _fail(f"{case_id}: input count is {len(ids)}, expected {input_count}")
        rows.append({
            "row_id": f"phase49-v620-llama-{case_id}",
            "case_id": case_id,
            "input_token_ids": ids,
            "input_token_count": input_count,
            "requested_output_tokens": output_count,
            "target": TARGET,
            "gpu_uuid": GPU_UUID,
            "warmups": 1 if case_id in {"long-100000", "decode-20000"} else 3,
            "measured": 3 if case_id in {"long-100000", "decode-20000"} else 10,
            "context_length": 131_072 if case_id in {"long-100000", "decode-20000"} else input_count + output_count,
            "ignore_eos": case_id == "decode-20000",
        })
    return rows


def atomic_write(path: Path, data: bytes, label: str) -> None:
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
        _fail(f"cannot atomically publish {label}: {exc}")
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def _parse_sysfs_value(path: Path) -> int:
    text = path.read_text(encoding="ascii").strip()
    if text.endswith(" bytes"):
        text = text[:-6].strip()
    if not text.isdigit():
        _fail(f"malformed sysfs memory value: {path}")
    return int(text)


def read_sysfs_memory(root: Path) -> dict[str, Any]:
    files: dict[str, list[Path]] = {"hbm": [], "gtt": []}
    for name in ("mem_info_vram_used", "hbm_used", "vram_used"):
        files["hbm"].extend(p for p in root.rglob(name) if p.is_file() and not p.is_symlink())
    for name in ("mem_info_gtt_used", "gtt_used"):
        files["gtt"].extend(p for p in root.rglob(name) if p.is_file() and not p.is_symlink())
    files = {key: sorted(set(value)) for key, value in files.items()}
    if not files["hbm"] or not files["gtt"]:
        _fail("sysfs HBM/GTT counters are unavailable")
    hbm = {str(path): _parse_sysfs_value(path) for path in files["hbm"]}
    gtt = {str(path): _parse_sysfs_value(path) for path in files["gtt"]}
    return {"hbm_bytes": sum(hbm.values()), "gtt_bytes": sum(gtt.values()), "hbm_files": hbm, "gtt_files": gtt}


def _process_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def process_snapshot(binary: Path, owned_pids: Iterable[int] = ()) -> dict[str, Any]:
    owned = {int(pid) for pid in owned_pids if int(pid) > 0}
    needle = binary.name.lower()
    records: list[dict[str, Any]] = []
    try:
        entries = sorted(Path("/proc").iterdir(), key=lambda p: p.name)
    except OSError:
        entries = []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        command = _process_cmdline(pid)
        if pid in owned or needle in command.lower():
            records.append({"pid": pid, "command": command, "owned": pid in owned})
    return {"source": "procfs-command-and-owned-pid", "available": True, "reliable": True, "gpu_processes": records, "owned_pids": sorted(owned), "timestamp_ns": time.monotonic_ns()}


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
        except (OSError, SessionDLlamaError) as exc:
            self.errors.append(str(exc))
            self.stop.set()

    def _run(self) -> None:
        while not self.stop.is_set():
            self._capture()
            self.stop.wait(MONITOR_PERIOD_SECONDS)

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="phase49-v620-llama-sysfs", daemon=True)
        self.thread.start()

    def finish(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.errors.append("sysfs monitor did not terminate")


def settled_memory(root: Path, timeout_seconds: float = 3.0) -> dict[str, Any]:
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


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _terminate_group(process: subprocess.Popen[bytes]) -> dict[str, bool]:
    term = kill = False
    try:
        os.killpg(process.pid, signal.SIGTERM)
        term = True
    except OSError:
        pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            kill = True
        except OSError:
            pass
        process.wait(timeout=2)
    return {"term_sent": term, "kill_sent": kill}


def run_process(command: list[str], env: Mapping[str, str], timeout_seconds: float) -> dict[str, Any]:
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(env), start_new_session=True)
    except OSError as exc:
        _fail(f"wrapper process could not start: {exc}")
    assert process.stdout is not None and process.stderr is not None
    timed_out = False
    termination = {"term_sent": False, "kill_sent": False}
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        termination = _terminate_group(process)
        stdout, stderr = process.communicate()
    if len(stdout) > MAX_STDOUT_BYTES or len(stderr) > MAX_STDERR_BYTES:
        _fail("wrapper stdout/stderr exceeded bounds")
    return {"pid": process.pid, "exit_code": process.returncode, "stdout": stdout, "stderr": stderr, "timed_out": timed_out, "termination": termination, "process_group_gone": process.poll() is not None and _process_group_gone(process.pid)}


def execution_environment(row_id: str, library_dir: Path, base: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base if base is not None else os.environ)
    for name in VISIBILITY_NAMES:
        env.pop(name, None)
    env["ROCR_VISIBLE_DEVICES"] = GPU_UUID
    env["LD_LIBRARY_PATH"] = f"{library_dir}:/opt/rocm/core-7.14/lib"
    env["SLLM_PHASE49_V620_LLAMA_ROW"] = row_id
    return env


def expected_command(binary: Path, model: Path, model_sha256: str, row: Mapping[str, Any], input_token_file: Path | None = None) -> list[str]:
    command = [
        str(binary), "--model", str(model), "--model-sha256", model_sha256,
        "--row-id", str(row["row_id"]), "--case-id", str(row["case_id"]),
        "--max-new-tokens", str(row["requested_output_tokens"]),
        "--warmup-requests", str(row["warmups"]), "--measured-requests", str(row["measured"]),
        "--batch-size", "1", "--sequences", "1", "--n-batch", "2048", "--n-ubatch", "512", "--main-gpu", "0",
        "--context-length", str(row["context_length"]),
        "--benchmark-schema-version", WRAPPER_SCHEMA_VERSION,
    ]
    input_option = ["--input-token-ids", ",".join(str(value) for value in row["input_token_ids"])] if input_token_file is None else ["--input-token-ids-file", str(input_token_file)]
    command[command.index("--max-new-tokens"):command.index("--max-new-tokens")] = input_option
    if row["ignore_eos"]:
        command.append("--ignore-eos")
    return command


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _validate_cleanup(value: Any, label: str) -> None:
    cleanup = value.get("cleanup") if isinstance(value, dict) else None
    if not isinstance(cleanup, dict) or cleanup.get("backend_release_completed") is not True or cleanup.get("cleanup_failures") != 0:
        _fail(f"{label}: backend cleanup evidence is invalid")


def _validate_events(sample: Mapping[str, Any], label: str) -> None:
    events = sample.get("events")
    derived = sample.get("derived")
    if not isinstance(events, dict) or not isinstance(derived, dict):
        _fail(f"{label}: events/derived are absent")
    ordered = [events.get("request_start_ns"), events.get("prefill_submit_ns"), events.get("prefill_complete_ns"), events.get("first_token_ns")]
    pubs = events.get("token_publications_ns")
    if not isinstance(pubs, list) or any(not isinstance(item, int) for item in pubs):
        _fail(f"{label}: token publication events are malformed")
    ordered.extend(pubs)
    ordered.extend([events.get("stop_ns"), events.get("cleanup_complete_ns")])
    if any(not isinstance(item, int) for item in ordered) or any(right <= left for left, right in zip(ordered, ordered[1:])):
        _fail(f"{label}: event timestamps are not strictly ordered")
    if not all(isinstance(derived.get(key), (int, float)) and derived[key] > 0 for key in ("ttft_ns", "prefill_ns", "e2e_ns")):
        _fail(f"{label}: required derived timing is invalid")
    if not isinstance(derived.get("tpot_ns"), list) or any(not isinstance(item, int) or item <= 0 for item in derived["tpot_ns"]):
        _fail(f"{label}: TPOT distribution is invalid")
    if not isinstance(derived.get("decode_ns"), int) or derived["decode_ns"] <= 0:
        _fail(f"{label}: decode timing is invalid")
    if derived["ttft_ns"] != events["first_token_ns"] - events["request_start_ns"] or derived["prefill_ns"] != events["prefill_complete_ns"] - events["prefill_submit_ns"] or derived["decode_ns"] != events["token_publications_ns"][-1] - events["first_token_ns"] or derived["e2e_ns"] != events["cleanup_complete_ns"] - events["request_start_ns"]:
        _fail(f"{label}: derived/event timing mismatch")


def validate_result(document: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict) or document.get("state") != "PASS" or document.get("schema_version") != WRAPPER_SCHEMA_VERSION:
        _fail(f"{row['row_id']}: wrapper result is not PASS/current schema")
    llama = document.get("llama")
    if not isinstance(llama, dict) or llama.get("commit") != LLAMA_COMMIT or llama.get("tag") != LLAMA_TAG:
        _fail(f"{row['row_id']}: llama commit/tag identity is invalid")
    target = document.get("target")
    if not isinstance(target, dict) or target.get("exact") != TARGET or target.get("gpu_uuid") != GPU_UUID or target.get("logical_device_index") != 0:
        _fail(f"{row['row_id']}: target/UUID identity is invalid")
    if document.get("row_id") != row["row_id"] or document.get("case_id") != row["case_id"]:
        _fail(f"{row['row_id']}: wrapper row identity is invalid")
    model = document.get("model")
    if not isinstance(model, dict) or model.get("format") != "GGUF" or model.get("weights") != "BF16" or model.get("kv") != "F16":
        _fail(f"{row['row_id']}: BF16/F16 model contract is invalid")
    if model.get("sha256") != row["model_sha256"]:
        _fail(f"{row['row_id']}: model digest differs from command identity")
    protocol = document.get("protocol")
    expected_protocol = {"batch_size": 1, "sequences": 1, "warmup_requests": row["warmups"], "measured_requests": row["measured"], "max_new_tokens": row["requested_output_tokens"], "n_ctx": row["context_length"], "n_batch": 2048, "n_ubatch": 512, "n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "offload_kqv": True, "op_offload": True, "greedy": True, "ignore_eos": row["ignore_eos"], "stop_token_ids": [] if row["ignore_eos"] else list(STOP_IDS), "bos_inserted": False}
    if not isinstance(protocol, dict) or any(protocol.get(key) != value for key, value in expected_protocol.items()):
        _fail(f"{row['row_id']}: fixed protocol drifted")
    if document.get("input_token_ids") != row["input_token_ids"]:
        _fail(f"{row['row_id']}: exact input IDs changed")
    offload = document.get("offload_evidence")
    if not isinstance(offload, dict) or offload.get("visible_gpu_device_count") != 1 or offload.get("selected_device", {}).get("type") != "GPU" or offload.get("requested", {}).get("n_gpu_layers") != -1 or offload.get("requested", {}).get("split_mode") != "none" or offload.get("requested", {}).get("main_gpu") != 0 or offload.get("requested", {}).get("offload_kqv") is not True or offload.get("requested", {}).get("op_offload") is not True or offload.get("observed", {}).get("offloaded_layers") != offload.get("observed", {}).get("offloadable_layers"):
        _fail(f"{row['row_id']}: full GPU offload evidence is invalid")
    _validate_cleanup(document, str(row["row_id"]))
    all_samples: list[Mapping[str, Any]] = []
    for group, count in (("warmups", row["warmups"]), ("measured", row["measured"])):
        value = document.get(group)
        if not isinstance(value, dict) or value.get("count") != count or not isinstance(value.get("samples"), list) or len(value["samples"]) != count:
            _fail(f"{row['row_id']}: {group} count is invalid")
        all_samples.extend(value["samples"])
    baseline: list[int] | None = None
    baseline_stop: Mapping[str, Any] | None = None
    for index, sample in enumerate(all_samples):
        if not isinstance(sample, dict):
            _fail(f"{row['row_id']}: sample {index} is malformed")
        tokens = sample.get("tokens")
        if not isinstance(tokens, dict) or tokens.get("input_token_ids") != row["input_token_ids"]:
            _fail(f"{row['row_id']}: sample {index} input IDs differ")
        generated = tokens.get("generated_token_ids")
        visible = tokens.get("visible_token_ids")
        if not isinstance(generated, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > MAX_TOKEN_ID for item in generated) or not isinstance(visible, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item > MAX_TOKEN_ID for item in visible) or len(generated) != row["requested_output_tokens"] or visible != generated or tokens.get("stop_token_ids_fed_back") != [] or tokens.get("bos_inserted") is not False:
            _fail(f"{row['row_id']}: sample {index} token output contract is invalid")
        stop = sample.get("stop")
        if not isinstance(stop, dict) or stop.get("version") != 1 or stop.get("reason_version") != 1 or stop.get("kind") != "max_new_tokens" or stop.get("token_id") is not None:
            _fail(f"{row['row_id']}: sample {index} stop contract is invalid")
        if baseline is None:
            baseline = generated
            baseline_stop = stop
        if generated != baseline:
            _fail(f"{row['row_id']}: sample {index} output differs from first sample")
        if stop != baseline_stop:
            _fail(f"{row['row_id']}: sample {index} stop differs from first sample")
        _validate_events(sample, f"{row['row_id']} sample {index}")
        derived = sample["derived"]
        if derived.get("decode_tokens") != row["requested_output_tokens"] - 1 or len(derived.get("tpot_ns", [])) != row["requested_output_tokens"] - 1:
            _fail(f"{row['row_id']}: sample {index} decode metric count is invalid")
    return document


def _verify_raw_item(item: Mapping[str, Any], label: str) -> None:
    path = Path(str(item.get("path", "")))
    digest = item.get("sha256")
    if path.is_symlink() or not path.is_file() or not isinstance(digest, str) or sha256_bytes(path.read_bytes()) != digest:
        _fail(f"{label}: raw digest/path verification failed")


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


def _resume_row(row_dir: Path, row: Mapping[str, Any], model_sha256: str, binary_sha256: str) -> dict[str, Any] | None:
    if not row_dir.exists():
        return None
    entries = list(row_dir.iterdir())
    report_path = row_dir / "row.json"
    if not report_path.is_file():
        _fail(f"{row['row_id']}: partial row exists and will not be overwritten")
    report = _json_load(report_path.read_bytes(), f"{row['row_id']} row report")
    if not isinstance(report, dict) or report.get("state") != "PASS" or report.get("row") != dict(row) or report.get("model", {}).get("sha256") != model_sha256 or report.get("binary", {}).get("sha256") != binary_sha256:
        _fail(f"{row['row_id']}: existing complete row identity differs")
    raw = report.get("raw")
    if not isinstance(raw, dict):
        _fail(f"{row['row_id']}: raw manifest is absent")
    for key in ("stdout", "stderr", "monitor_tsv"):
        item = raw.get(key)
        if not isinstance(item, dict):
            _fail(f"{row['row_id']}: raw {key} manifest is absent")
        _verify_raw_item(item, f"{row['row_id']} {key}")
    _validate_external_resources(report, str(row["row_id"]))
    if len(row["input_token_ids"]) > 10_001:
        item = raw.get("input_token_ids")
        if not isinstance(item, dict):
            _fail(f"{row['row_id']}: raw input-token manifest is absent")
        _verify_raw_item(item, f"{row['row_id']} input token IDs")
    result = validate_result(_json_load(Path(raw["stdout"]["path"]).read_bytes(), f"{row['row_id']} stdout"), row)
    if report.get("result") != result:
        _fail(f"{row['row_id']}: stored result differs from raw stdout")
    return report


def _write_monitor(path: Path, samples: list[Mapping[str, Any]]) -> None:
    lines = ["timestamp_ns\thbm_bytes\tgtt_bytes\n"]
    lines.extend(f"{sample['timestamp_ns']}\t{sample['hbm_bytes']}\t{sample['gtt_bytes']}\n" for sample in samples)
    atomic_write(path, "".join(lines).encode("ascii"), "monitor TSV")


def run_row(binary: Path, binary_identity: Mapping[str, Any], model: Path, model_identity: Mapping[str, Any], row: Mapping[str, Any], output_dir: Path, sysfs_root: Path, timeout_seconds: float, base_env: Mapping[str, str] | None = None) -> dict[str, Any]:
    row = {**row, "model_sha256": model_identity["sha256"]}
    row_dir = output_dir / "raw" / str(row["row_id"])
    existing = _resume_row(row_dir, row, str(model_identity["sha256"]), str(binary_identity["sha256"]))
    if existing is not None:
        return existing
    row_dir.mkdir(parents=True, exist_ok=False)
    input_token_path: Path | None = None
    if len(row["input_token_ids"]) > 10_001:
        input_token_path = row_dir / "input-token-ids.csv"
        atomic_write(input_token_path, (",".join(str(value) for value in row["input_token_ids"]) + "\n").encode("ascii"), "input token IDs")
    baseline = read_sysfs_memory(sysfs_root)
    before_process = process_snapshot(binary)
    command = expected_command(binary, model, str(model_identity["sha256"]), row, input_token_path)
    monitor = _SysfsMonitor(sysfs_root)
    monitor.start()
    env = execution_environment(str(row["row_id"]), binary.parent, base_env)
    try:
        capture = run_process(command, env, timeout_seconds)
    finally:
        monitor.finish()
    settled = settled_memory(sysfs_root)
    after_process = process_snapshot(binary, [capture["pid"]])
    stdout_path = row_dir / "stdout.json"
    stderr_path = row_dir / "stderr.log"
    monitor_path = row_dir / "hbm-gtt.tsv"
    atomic_write(stdout_path, capture["stdout"], "wrapper stdout")
    atomic_write(stderr_path, capture["stderr"], "wrapper stderr")
    _write_monitor(monitor_path, monitor.samples)
    if capture["timed_out"] or capture["exit_code"] != 0 or not capture["process_group_gone"]:
        _fail(f"{row['row_id']}: wrapper process failed")
    if monitor.errors or not monitor.samples:
        _fail(f"{row['row_id']}: sysfs monitor failed: {monitor.errors or ['zero samples']}")
    if settled.get("settled") is not True or settled.get("hbm_bytes") != baseline.get("hbm_bytes") or settled.get("gtt_bytes") != baseline.get("gtt_bytes"):
        _fail(f"{row['row_id']}: post-process HBM/GTT did not return to baseline")
    result = validate_result(_json_load(capture["stdout"], f"{row['row_id']} wrapper stdout"), row)
    report = {
        "schema_version": ROW_SCHEMA_VERSION, "state": "PASS", "row": row,
        "binary": dict(binary_identity), "model": dict(model_identity), "target": TARGET, "gpu_uuid": GPU_UUID,
        "command": command, "environment": {"ROCR_VISIBLE_DEVICES": env["ROCR_VISIBLE_DEVICES"], "LD_LIBRARY_PATH": env["LD_LIBRARY_PATH"]},
        "process": {"pre": before_process, "post": after_process, "capture": {key: value for key, value in capture.items() if key not in {"stdout", "stderr"}}},
        "memory": {"baseline": baseline, "settled": settled}, "monitor": {"cadence_ms": 100, "samples": len(monitor.samples), "errors": monitor.errors},
        "raw": {"stdout": {"path": str(stdout_path.resolve()), "sha256": sha256_bytes(capture["stdout"])}, "stderr": {"path": str(stderr_path.resolve()), "sha256": sha256_bytes(capture["stderr"])}, "monitor_tsv": {"path": str(monitor_path.resolve()), "sha256": sha256_bytes(monitor_path.read_bytes())}},
        "result": result,
    }
    if input_token_path is not None:
        report["raw"]["input_token_ids"] = {"path": str(input_token_path.resolve()), "sha256": sha256_bytes(input_token_path.read_bytes())}
    atomic_write(row_dir / "row.json", canonical_bytes(report), "row report")
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.target != TARGET or args.gpu_uuid != GPU_UUID:
        _fail(f"Phase 49 llama producer is restricted to exact {TARGET}/{GPU_UUID}")
    binary = regular_file(Path(args.binary), "llama wrapper binary", executable=True)
    model = regular_file(Path(args.model), "BF16 GGUF")
    output_dir = Path(args.output_dir)
    sysfs_root = Path(args.sysfs_root)
    if not sysfs_root.is_dir():
        _fail(f"sysfs root is not a directory: {sysfs_root}")
    output_resolved = output_dir.resolve()
    if output_resolved == PROJECT_ROOT or PROJECT_ROOT in output_resolved.parents:
        _fail("raw output directory must be outside the repository")
    model_identity = file_identity(model, "BF16 GGUF")
    binary_identity = file_identity(binary, "llama wrapper binary")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [run_row(binary, binary_identity, model, model_identity, row, output_dir, sysfs_root, args.timeout_seconds) for row in matrix()]
    summary = {"schema_version": SCHEMA_VERSION, "state": "PASS", "target": TARGET, "gpu_uuid": GPU_UUID, "llama": {"commit": LLAMA_COMMIT, "tag": LLAMA_TAG}, "protocol": {"normal": {"warmups": 3, "measured": 10}, "extended": {"warmups": 1, "measured": 3, "context_length": 131072}, "batch_size": 1, "n_batch": 2048, "n_ubatch": 512, "weights": "BF16", "kv": "F16"}, "matrix": {"cases": [item[0] for item in CASE_SPECS], "row_count": len(reports)}, "binary": binary_identity, "model": model_identity, "rows": reports}
    summary_path = output_dir / "phase49-v620-llama-v1.json"
    if summary_path.exists():
        if _json_load(summary_path.read_bytes(), "existing llama summary") != summary:
            _fail("existing llama summary differs; refusing overwrite")
    else:
        atomic_write(summary_path, canonical_bytes(summary), "llama summary")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sysfs-root", default="/sys")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        print("FAIL: timeout must be positive", file=sys.stderr)
        return 2
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
        return 0
    except SessionDLlamaError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
