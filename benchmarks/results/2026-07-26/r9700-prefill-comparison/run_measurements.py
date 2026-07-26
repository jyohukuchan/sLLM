#!/usr/bin/env python3
"""One-window, R9700-only prefill measurement runner.

This is deliberately self-contained so its command logs and raw telemetry can
be committed with the result.  It never starts/stops services; the enclosing
wrapper owns that lifecycle and restores ullm-openai.service through an EXIT
trap.
"""

from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT = Path(__file__).resolve().parent
RAW = RESULT / "raw"
ENVIRONMENT = RESULT / "environment"
AMD_SMI = "amd-smi"
GPU_INDEX = "2"
PROMPT_LENGTHS = (128, 512, 1024, 2048, 4095)
REPEATS = 5
THERMAL_GATE = {
    "edge_c_max": 40.0,
    "hotspot_c_max": 42.0,
    "socket_power_w_max": 35.0,
    "poll_seconds": 5.0,
    "timeout_seconds": 900.0,
}

LLAMA_BENCH = Path("/home/homelab1/llama.cpp-src/build-rdna4/bin/llama-bench")
GGUF = Path(
    "/home/homelab1/datapool/ai_models/gguf/Qwen/"
    "Qwen3-14B-GGUF-530227a7/Qwen3-14B-Q8_0.gguf"
)
ULLM_DRIVER = Path(
    "/tmp/ullm-prefill-clean-0216b131/target/release/"
    "ullm-sq8-r9700-phase0-profile"
)

ULLM_GUARDS = (
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def json_line(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def nested(mapping: Any, *path: str) -> Any:
    current = mapping
    for item in path:
        if not isinstance(current, dict):
            return None
        current = current.get(item)
    return current


def metric_summary(payload: Any) -> dict[str, Any]:
    gpu_data = payload.get("gpu_data", []) if isinstance(payload, dict) else []
    gpu = gpu_data[0] if gpu_data else {}
    paths = {
        "gpu": ("gpu",),
        "edge_c": ("temperature", "edge", "value"),
        "hotspot_c": ("temperature", "hotspot", "value"),
        "mem_c": ("temperature", "mem", "value"),
        "socket_power_w": ("power", "socket_power", "value"),
        "throttle_status": ("power", "throttle_status"),
        "gfx_clock_mhz": ("clock", "gfx_0", "clk", "value"),
        "mem_clock_mhz": ("clock", "mem_0", "clk", "value"),
        "gfx_activity_pct": ("usage", "gfx_activity", "value"),
        "umc_activity_pct": ("usage", "umc_activity", "value"),
    }
    return {name: nested(gpu, *path) for name, path in paths.items()}


def get_metric() -> dict[str, Any]:
    started = utc_now()
    started_ns = time.monotonic_ns()
    completed = subprocess.run(
        [AMD_SMI, "metric", "--gpu", GPU_INDEX, "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    ended = utc_now()
    entry: dict[str, Any] = {
        "utc_started": started,
        "utc_ended": ended,
        "monotonic_started_ns": started_ns,
        "monotonic_ended_ns": time.monotonic_ns(),
        "returncode": completed.returncode,
        "stderr": completed.stderr,
    }
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        entry["stdout_unparsed"] = completed.stdout
        return entry
    entry["raw"] = payload
    entry["summary"] = metric_summary(payload)
    return entry


class MetricPoller:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="amd-smi-poller", daemon=True)

    def sample(self, marker: str | None = None) -> dict[str, Any]:
        entry = get_metric()
        if marker:
            entry["marker"] = marker
        json_line(self.output, entry)
        return entry

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.sample()
            self.stop_event.wait(1.0)

    def start(self) -> None:
        self.sample("poller-start")
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        self.sample("poller-stop")


def shell_command(argv: list[str], env_overrides: dict[str, str], unset: tuple[str, ...]) -> str:
    parts = ["env"]
    parts.extend(f"-u {shlex.quote(name)}" for name in unset)
    parts.extend(f"{name}={shlex.quote(value)}" for name, value in sorted(env_overrides.items()))
    parts.extend(shlex.quote(item) for item in argv)
    return " ".join(parts)


def run_capture(
    path_stem: Path,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        argv,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    write_text(path_stem.with_suffix(".stdout"), completed.stdout)
    write_text(path_stem.with_suffix(".stderr"), completed.stderr)
    write_json(
        path_stem.with_suffix(".json"),
        {
            "utc": utc_now(),
            "argv": argv,
            "returncode": completed.returncode,
        },
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"failed preflight command: {shlex.join(argv)}")
    return completed


def cooldown(condition_id: str) -> dict[str, Any]:
    output = RAW / "cooldown" / f"{condition_id}.jsonl"
    deadline = time.monotonic() + THERMAL_GATE["timeout_seconds"]
    while True:
        entry = get_metric()
        summary = entry.get("summary", {})
        edge = summary.get("edge_c")
        hotspot = summary.get("hotspot_c")
        power = summary.get("socket_power_w")
        passed = (
            isinstance(edge, (int, float))
            and isinstance(hotspot, (int, float))
            and isinstance(power, (int, float))
            and edge <= THERMAL_GATE["edge_c_max"]
            and hotspot <= THERMAL_GATE["hotspot_c_max"]
            and power <= THERMAL_GATE["socket_power_w_max"]
        )
        entry["thermal_gate"] = {"limits": THERMAL_GATE, "passed": passed}
        json_line(output, entry)
        if passed:
            return entry
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"thermal cooldown timed out for {condition_id}: {json.dumps(summary, sort_keys=True)}"
            )
        time.sleep(THERMAL_GATE["poll_seconds"])


def stream_process(
    condition_dir: Path,
    argv: list[str],
    env: dict[str, str],
    env_overrides: dict[str, str],
    unset: tuple[str, ...],
) -> dict[str, Any]:
    stdout_path = condition_dir / "stdout.log"
    stderr_path = condition_dir / "stderr.log"
    events_path = condition_dir / "stream-events.jsonl"
    write_text(
        condition_dir / "command.txt",
        "# exact executable command (credential-free)\n"
        + shell_command(argv, env_overrides, unset)
        + "\n",
    )
    write_json(
        condition_dir / "command.json",
        {
            "argv": argv,
            "environment_overrides": env_overrides,
            "unset_environment": list(unset),
            "cwd": str(RESULT),
        },
    )

    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    process = subprocess.Popen(
        argv,
        cwd=RESULT,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    event_lock = threading.Lock()

    def pump(pipe: Any, output: Path, stream: str) -> None:
        with output.open("w", encoding="utf-8") as handle:
            for line in iter(pipe.readline, ""):
                handle.write(line)
                handle.flush()
                with event_lock:
                    json_line(
                        events_path,
                        {
                            "utc": utc_now(),
                            "monotonic_ns": time.monotonic_ns(),
                            "stream": stream,
                            "line": line.rstrip("\n"),
                        },
                    )
            pipe.close()

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_thread = threading.Thread(target=pump, args=(process.stdout, stdout_path, "stdout"))
    stderr_thread = threading.Thread(target=pump, args=(process.stderr, stderr_path, "stderr"))
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join(timeout=30)
    stderr_thread.join(timeout=30)
    ended_utc = utc_now()
    ended_ns = time.monotonic_ns()
    result = {
        "utc_started": started_utc,
        "utc_ended": ended_utc,
        "monotonic_started_ns": started_ns,
        "monotonic_ended_ns": ended_ns,
        "wall_seconds": (ended_ns - started_ns) / 1e9,
        "returncode": returncode,
        "pid": process.pid,
    }
    write_json(condition_dir / "process.json", result)
    return result


@dataclass(frozen=True)
class Condition:
    id: str
    engine: str
    prompt_tokens: int
    kv_dtype: str
    argv: list[str]
    env_overrides: dict[str, str]
    unset: tuple[str, ...]


def ullm_condition(prompt_tokens: int) -> Condition:
    return Condition(
        id=f"ullm-sq8_0-f32-kv-p{prompt_tokens}",
        engine="uLLM SQ8_0",
        prompt_tokens=prompt_tokens,
        kv_dtype="f32",
        argv=[
            str(ULLM_DRIVER),
            "--phase",
            "prefill",
            "--prompt-tokens",
            str(prompt_tokens),
            "--repeats",
            str(REPEATS),
        ],
        env_overrides={"HIP_VISIBLE_DEVICES": "1", **{name: "1" for name in ULLM_GUARDS}},
        unset=("ROCR_VISIBLE_DEVICES",),
    )


def llama_condition(prompt_tokens: int, kv_dtype: str) -> Condition:
    assert kv_dtype in ("f32", "f16")
    return Condition(
        id=f"llama-cpp-q8_0-{kv_dtype}-kv-p{prompt_tokens}",
        engine="llama.cpp Q8_0",
        prompt_tokens=prompt_tokens,
        kv_dtype=kv_dtype,
        argv=[
            str(LLAMA_BENCH),
            "-m",
            str(GGUF),
            "-o",
            "json",
            "-r",
            str(REPEATS),
            "-p",
            str(prompt_tokens),
            "-n",
            "0",
            "-b",
            str(prompt_tokens),
            "-ub",
            "128",
            "-ctk",
            kv_dtype,
            "-ctv",
            kv_dtype,
            "-ngl",
            "999",
            "-sm",
            "none",
            "-mg",
            "0",
            "-dev",
            "ROCm0",
            "-nkvo",
            "0",
            "-fa",
            "on",
            "-t",
            "1",
            "-mmp",
            "1",
            "--progress",
            "-v",
        ],
        env_overrides={"HIP_VISIBLE_DEVICES": "1"},
        unset=("ROCR_VISIBLE_DEVICES",),
    )


def verified_env(condition: Condition) -> dict[str, str]:
    env = dict(os.environ)
    for name in condition.unset:
        env.pop(name, None)
    env.update(condition.env_overrides)
    return env


def service_preflight() -> None:
    commands = [
        ("llama-qwen35-state", ["systemctl", "show", "llama-qwen35-udq4.service", "--property=ActiveState,UnitFileState,SubState,MainPID"]),
        ("ullm-openai-state", ["systemctl", "show", "ullm-openai.service", "--property=ActiveState,UnitFileState,SubState,MainPID,NRestarts"]),
        ("r9700-static", [AMD_SMI, "static", "--gpu", GPU_INDEX, "--json"]),
        ("r9700-pre-window-metric", [AMD_SMI, "metric", "--gpu", GPU_INDEX, "--json"]),
        ("r9700-pre-window-processes", [AMD_SMI, "process", "--gpu", GPU_INDEX, "--json"]),
    ]
    for name, argv in commands:
        run_capture(ENVIRONMENT / name, argv, check=False)

    llama_state = (ENVIRONMENT / "llama-qwen35-state.stdout").read_text(encoding="utf-8")
    if "ActiveState=inactive" not in llama_state or "UnitFileState=disabled" not in llama_state:
        raise RuntimeError("llama-qwen35-udq4.service is not inactive+disabled; refusing benchmark")

    env = dict(os.environ)
    env.pop("ROCR_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = "1"
    run_capture(ENVIRONMENT / "llama-r9700-device-list", [str(LLAMA_BENCH), "--list-devices"], env=env)
    device_list = (
        (ENVIRONMENT / "llama-r9700-device-list.stdout").read_text(encoding="utf-8")
        + (ENVIRONMENT / "llama-r9700-device-list.stderr").read_text(encoding="utf-8")
    )
    if "gfx1201" not in device_list or "gfx1030" in device_list:
        raise RuntimeError("llama.cpp visibility validation did not expose only gfx1201")

    identity_commands = [
        ("gguf-sha256", ["sha256sum", str(GGUF)]),
        ("gguf-stat", ["stat", "--printf=size=%s bytes\\n", str(GGUF)]),
        ("llama-bench-sha256", ["sha256sum", str(LLAMA_BENCH)]),
        ("ullm-driver-sha256", ["sha256sum", str(ULLM_DRIVER)]),
        ("ullm-clean-head", ["git", "-C", "/tmp/ullm-prefill-clean-0216b131", "rev-parse", "HEAD"]),
        ("ullm-clean-status", ["git", "-C", "/tmp/ullm-prefill-clean-0216b131", "status", "--short"]),
        ("llama-build-head", ["git", "-C", "/home/homelab1/llama.cpp-src", "rev-parse", "HEAD"]),
        ("llama-build-status", ["git", "-C", "/home/homelab1/llama.cpp-src", "status", "--short"]),
    ]
    for name, argv in identity_commands:
        run_capture(ENVIRONMENT / name, argv, check=False)

    source_snippets = {
        "llama-bench-timing-source.cpp": (
            "/home/homelab1/llama.cpp-src/tools/llama-bench/llama-bench.cpp",
            ((2088, 2122), (2324, 2408)),
        ),
        "llama-last-output-source.cpp": (
            "/home/homelab1/llama.cpp-src/src/llama-batch.cpp",
            ((116, 132),),
        ),
    }
    for name, (source, spans) in source_snippets.items():
        content = ""
        for start, end in spans:
            content += f"--- {source}:{start}-{end} ---\n"
            with open(source, encoding="utf-8") as handle:
                content += "".join(handle.readlines()[start - 1 : end])
        write_text(ENVIRONMENT / name, content)


def run_condition(condition: Condition) -> None:
    condition_dir = RAW / condition.id
    condition_dir.mkdir(parents=True, exist_ok=False)
    gate = cooldown(condition.id)
    write_json(condition_dir / "thermal-gate.json", gate)
    poller = MetricPoller(condition_dir / "amd-smi-metric.jsonl")
    poller.start()
    poller.sample("immediately-before-process")
    try:
        result = stream_process(
            condition_dir,
            condition.argv,
            verified_env(condition),
            condition.env_overrides,
            condition.unset,
        )
    finally:
        poller.sample("immediately-after-process")
        poller.stop()
    write_json(
        condition_dir / "condition.json",
        {
            "id": condition.id,
            "engine": condition.engine,
            "prompt_tokens": condition.prompt_tokens,
            "kv_dtype": condition.kv_dtype,
            "repeats": REPEATS,
            "result": result,
        },
    )
    if result["returncode"] != 0:
        raise RuntimeError(f"{condition.id} failed with return code {result['returncode']}")


def main() -> int:
    if not LLAMA_BENCH.is_file() or not GGUF.is_file() or not ULLM_DRIVER.is_file():
        raise RuntimeError("one or more pinned benchmark artifacts are missing")
    if any((RAW / f"{engine}-p{n}").exists() for engine in ("unused",) for n in PROMPT_LENGTHS):
        raise RuntimeError("unreachable guard")
    RAW.mkdir(exist_ok=True)
    ENVIRONMENT.mkdir(exist_ok=True)
    write_json(
        RESULT / "runner-configuration.json",
        {
            "utc_started": utc_now(),
            "prompt_lengths": PROMPT_LENGTHS,
            "repeats": REPEATS,
            "thermal_gate": THERMAL_GATE,
            "gpu": {"amd_smi_index": 2, "expected_bdf": "0000:47:00.0", "expected_gfx": "gfx1201"},
            "uLLM_driver": str(ULLM_DRIVER),
            "llama_bench": str(LLAMA_BENCH),
            "gguf": str(GGUF),
        },
    )
    service_preflight()
    conditions: list[Condition] = []
    for prompt_tokens in PROMPT_LENGTHS:
        conditions.extend(
            (
                ullm_condition(prompt_tokens),
                llama_condition(prompt_tokens, "f32"),
                llama_condition(prompt_tokens, "f16"),
            )
        )
    for condition in conditions:
        run_condition(condition)
    write_json(RESULT / "runner-complete.json", {"utc_completed": utc_now(), "conditions": [item.id for item in conditions]})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        write_text(RESULT / "runner-failure.txt", f"{utc_now()} {type(error).__name__}: {error}\n")
        raise
