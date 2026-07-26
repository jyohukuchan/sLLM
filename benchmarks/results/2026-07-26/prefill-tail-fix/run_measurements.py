#!/usr/bin/env python3
"""One-window R9700-only SQ8_0 prefill remeasurement for the tail fix.

The enclosing shell wrapper owns `ullm-openai.service`; this program never
starts or stops a service.  Timed throughput comes only from the driver's
`Instant` measurements, never from a profiler range duration.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
ENVIRONMENT = ROOT / "environment"
AMD_SMI = "amd-smi"
GPU_INDEX = "2"
PROMPT_LENGTHS = (128, 512, 1024, 2048, 4095)
REPEATS = 5
DRIVER = Path("/tmp/ullm-prefill-tail-fix-target/release/ullm-sq8-r9700-prefill-tail-fix-profile")
THERMAL_GATE = {
    "edge_c_max": 40.0,
    "hotspot_c_max": 42.0,
    "socket_power_w_max": 35.0,
    "poll_seconds": 5.0,
    "timeout_seconds": 900.0,
}
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


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric() -> dict[str, Any]:
    started = time.monotonic_ns()
    result = subprocess.run(
        [AMD_SMI, "metric", "--gpu", GPU_INDEX, "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    entry: dict[str, Any] = {
        "utc": utc_now(),
        "monotonic_started_ns": started,
        "monotonic_ended_ns": time.monotonic_ns(),
        "returncode": result.returncode,
        "stderr": result.stderr,
    }
    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        entry["stdout_unparsed"] = result.stdout
        return entry
    gpu_data = raw.get("gpu_data", []) if isinstance(raw, dict) else []
    gpu = gpu_data[0] if gpu_data else {}
    entry["raw"] = raw
    entry["summary"] = {
        "edge_c": nested(gpu, "temperature", "edge", "value"),
        "hotspot_c": nested(gpu, "temperature", "hotspot", "value"),
        "mem_c": nested(gpu, "temperature", "mem", "value"),
        "socket_power_w": nested(gpu, "power", "socket_power", "value"),
        "throttle_status": nested(gpu, "power", "throttle_status"),
        "gfx_clock_mhz": nested(gpu, "clock", "gfx_0", "clk", "value"),
        "mem_clock_mhz": nested(gpu, "clock", "mem_0", "clk", "value"),
        "gfx_activity_pct": nested(gpu, "usage", "gfx_activity", "value"),
        "umc_activity_pct": nested(gpu, "usage", "umc_activity", "value"),
    }
    return entry


def capture_command(name: str, argv: list[str]) -> None:
    completed = subprocess.run(argv, text=True, capture_output=True, check=False)
    write_text_new(ENVIRONMENT / f"{name}.stdout", completed.stdout)
    write_text_new(ENVIRONMENT / f"{name}.stderr", completed.stderr)
    write_json_new(
        ENVIRONMENT / f"{name}.json",
        {"utc": utc_now(), "argv": argv, "returncode": completed.returncode},
    )


def cooldown(condition_id: str) -> dict[str, Any]:
    path = RAW / "cooldown" / f"{condition_id}.jsonl"
    deadline = time.monotonic() + THERMAL_GATE["timeout_seconds"]
    while True:
        entry = metric()
        summary = entry.get("summary", {})
        edge = summary.get("edge_c")
        hotspot = summary.get("hotspot_c")
        power = summary.get("socket_power_w")
        passed = all(isinstance(value, (int, float)) for value in (edge, hotspot, power)) and (
            edge <= THERMAL_GATE["edge_c_max"]
            and hotspot <= THERMAL_GATE["hotspot_c_max"]
            and power <= THERMAL_GATE["socket_power_w_max"]
        )
        entry["thermal_gate"] = {"limits": THERMAL_GATE, "passed": passed}
        append_json_line(path, entry)
        if passed:
            return entry
        if time.monotonic() >= deadline:
            raise RuntimeError(f"thermal cooldown timed out: {condition_id}: {summary}")
        time.sleep(THERMAL_GATE["poll_seconds"])


class MetricPoller:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def sample(self, marker: str | None = None) -> None:
        entry = metric()
        if marker:
            entry["marker"] = marker
        append_json_line(self.path, entry)

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


def driver_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("ROCR_VISIBLE_DEVICES", None)
    environment["HIP_VISIBLE_DEVICES"] = "1"
    environment.update({name: "1" for name in ULLM_GUARDS})
    return environment


def condition(prompt_tokens: int) -> None:
    condition_id = f"ullm-sq8_0-f32-kv-p{prompt_tokens}"
    directory = RAW / condition_id
    directory.mkdir(exist_ok=False)
    gate = cooldown(condition_id)
    write_json_new(directory / "thermal-gate.json", gate)
    argv = [
        str(DRIVER),
        "--phase",
        "prefill",
        "--prompt-tokens",
        str(prompt_tokens),
        "--repeats",
        str(REPEATS),
    ]
    overrides = {"HIP_VISIBLE_DEVICES": "1", **{name: "1" for name in ULLM_GUARDS}}
    write_text_new(
        directory / "command.txt",
        "# exact executable command (credential-free)\n"
        + "env -u ROCR_VISIBLE_DEVICES "
        + " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(overrides.items()))
        + " "
        + shlex.join(argv)
        + "\n",
    )
    write_json_new(
        directory / "command.json",
        {
            "argv": argv,
            "environment_overrides": overrides,
            "unset_environment": ["ROCR_VISIBLE_DEVICES"],
            "cwd": str(ROOT),
        },
    )
    started_utc = utc_now()
    started_ns = time.monotonic_ns()
    poller = MetricPoller(directory / "amd-smi-metric.jsonl")
    poller.start()
    poller.sample("immediately-before-process")
    try:
        completed = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, env=driver_environment())
    finally:
        poller.sample("immediately-after-process")
        poller.stop()
    ended_ns = time.monotonic_ns()
    write_text_new(directory / "stdout.log", completed.stdout)
    write_text_new(directory / "stderr.log", completed.stderr)
    write_json_new(
        directory / "process.json",
        {
            "utc_started": started_utc,
            "utc_ended": utc_now(),
            "monotonic_started_ns": started_ns,
            "monotonic_ended_ns": ended_ns,
            "wall_seconds": (ended_ns - started_ns) / 1e9,
            "returncode": completed.returncode,
        },
    )
    if completed.returncode:
        raise RuntimeError(f"{condition_id} failed with return code {completed.returncode}")


def main() -> int:
    if not DRIVER.is_file():
        raise RuntimeError(f"candidate driver is missing: {DRIVER}")
    if RAW.exists() and any(RAW.iterdir()):
        raise RuntimeError(f"raw directory is not empty: {RAW}")
    RAW.mkdir(exist_ok=True)
    ENVIRONMENT.mkdir(exist_ok=True)
    write_json_new(
        ROOT / "runner-configuration.json",
        {
            "utc_started": utc_now(),
            "prompt_lengths": PROMPT_LENGTHS,
            "repeats": REPEATS,
            "thermal_gate": THERMAL_GATE,
            "gpu": {"amd_smi_index": 2, "expected_bdf": "0000:47:00.0", "expected_gfx": "gfx1201"},
            "driver": str(DRIVER),
            "timer": "driver std::time::Instant around prefill advances; not a profiler range duration",
            "comparison_condition_reference": "../r9700-prefill-comparison/conditions.md",
        },
    )
    capture_command("r9700-static", [AMD_SMI, "static", "--gpu", GPU_INDEX, "--json"])
    capture_command("r9700-pre-window-metric", [AMD_SMI, "metric", "--gpu", GPU_INDEX, "--json"])
    capture_command("r9700-pre-window-processes", [AMD_SMI, "process", "--gpu", GPU_INDEX, "--json"])
    for prompt_tokens in PROMPT_LENGTHS:
        condition(prompt_tokens)
    write_json_new(
        ROOT / "runner-complete.json",
        {"utc_completed": utc_now(), "conditions": list(PROMPT_LENGTHS)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
