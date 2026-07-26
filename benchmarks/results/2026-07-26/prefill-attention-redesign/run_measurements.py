#!/usr/bin/env python3
"""Run unprofiled SQ8_0 measurements inside an already-owned GPU window.

This program never starts or stops a service.  The enclosing shell wrapper
owns isolation.  Throughput is taken only from the driver's synchronized
`Instant` interval, never from rocprof durations.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RAW = ROOT / os.environ.get("ULLM_PREFILL_RAW_SUBDIR", "raw/throughput")
GPU_INDEX = "2"
DRIVER = Path(
    os.environ.get(
        "ULLM_PREFILL_DRIVER",
        "/tmp/ullm-br-prefill-gqa20-target/release/ullm-sq8-r9700-phase0-profile",
    )
)
CANDIDATE_LABEL = os.environ.get("ULLM_PREFILL_CANDIDATE_LABEL", "gqa_grouped_tile20")
CANDIDATE_ENV = os.environ.get(
    "ULLM_PREFILL_CANDIDATE_ENV", "ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE"
)
BASELINE_ENV = os.environ.get("ULLM_PREFILL_BASELINE_ENV", "")
CANDIDATE_DESCRIPTION = os.environ.get(
    "ULLM_PREFILL_CANDIDATE_DESCRIPTION", f"{CANDIDATE_ENV}=1"
)
CONFIG_NAME = os.environ.get("ULLM_PREFILL_CONFIG_NAME", "throughput-run-configuration.json")
SUMMARY_NAME = os.environ.get("ULLM_PREFILL_SUMMARY_NAME", "throughput-summary.json")
PROMPTS = (128, 512, 1024, 2048, 4095)
REPEATS = 5
THERMAL_GATE = {
    "edge_c_max": 40.0,
    "hotspot_c_max": 42.0,
    "socket_power_w_max": 35.0,
    "poll_seconds": 5.0,
    "timeout_seconds": 900.0,
}
HIP_GUARDS = (
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
EXPERIMENT_ENVIRONMENTS = (
    "ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_PROTOTYPE",
    "ULLM_USE_SQ8_0_FLASH2_GQA_GROUPED_EXACT_TILE64_PROTOTYPE",
    "ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE",
    "ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        destination.write(value)


def append_json_line(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(value, sort_keys=True) + "\n")


def nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def metric() -> dict[str, Any]:
    started = time.monotonic_ns()
    completed = subprocess.run(
        ["amd-smi", "metric", "--gpu", GPU_INDEX, "--json"],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    result: dict[str, Any] = {
        "utc": utc_now(),
        "monotonic_started_ns": started,
        "monotonic_ended_ns": time.monotonic_ns(),
        "returncode": completed.returncode,
        "stderr": completed.stderr,
    }
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError:
        result["stdout_unparsed"] = completed.stdout
        return result
    gpu_data = raw.get("gpu_data", []) if isinstance(raw, dict) else []
    gpu = gpu_data[0] if gpu_data else {}
    result["raw"] = raw
    result["summary"] = {
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
    return result


def cooldown(condition: str) -> dict[str, Any]:
    path = RAW / "cooldown" / f"{condition}.jsonl"
    deadline = time.monotonic() + THERMAL_GATE["timeout_seconds"]
    while True:
        sample = metric()
        summary = sample.get("summary", {})
        edge = summary.get("edge_c")
        hotspot = summary.get("hotspot_c")
        power = summary.get("socket_power_w")
        passed = all(isinstance(value, (int, float)) for value in (edge, hotspot, power)) and (
            edge <= THERMAL_GATE["edge_c_max"]
            and hotspot <= THERMAL_GATE["hotspot_c_max"]
            and power <= THERMAL_GATE["socket_power_w_max"]
        )
        sample["thermal_gate"] = {"limits": THERMAL_GATE, "passed": passed}
        append_json_line(path, sample)
        if passed:
            return sample
        if time.monotonic() >= deadline:
            raise RuntimeError(f"thermal cooldown timed out for {condition}: {summary}")
        time.sleep(THERMAL_GATE["poll_seconds"])


class MetricPoller:
    def __init__(self, output: Path) -> None:
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)

    def sample(self, marker: str | None = None) -> None:
        result = metric()
        if marker:
            result["marker"] = marker
        append_json_line(self.output, result)

    def _poll(self) -> None:
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


def environment(grouped: bool) -> dict[str, str]:
    result = dict(os.environ)
    result.pop("ROCR_VISIBLE_DEVICES", None)
    for name in EXPERIMENT_ENVIRONMENTS:
        result.pop(name, None)
    result["HIP_VISIBLE_DEVICES"] = "1"
    result.update({name: "1" for name in HIP_GUARDS})
    if grouped and CANDIDATE_ENV:
        result[CANDIDATE_ENV] = "1"
    elif not grouped and BASELINE_ENV:
        result[BASELINE_ENV] = "1"
    return result


def parse_summary(stdout: str) -> dict[str, Any]:
    events = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    summaries = [event for event in events if event.get("event") == "summary"]
    if len(summaries) != 1:
        raise ValueError(f"expected exactly one driver summary, found {len(summaries)}")
    return summaries[0]


def condition(label: str, prompt_tokens: int, grouped: bool, phase: str) -> dict[str, Any]:
    condition_id = f"{label}-{phase}-p{prompt_tokens}"
    directory = RAW / condition_id
    directory.mkdir(parents=True, exist_ok=False)
    gate = cooldown(condition_id)
    write_json_new(directory / "thermal-gate.json", gate)
    argv = [
        str(DRIVER),
        "--phase",
        phase,
        "--prompt-tokens",
        str(prompt_tokens),
        "--repeats",
        str(REPEATS),
    ]
    if phase == "decode":
        argv += ["--warmup-steps", "4", "--measured-steps", "16"]
    overrides = {"HIP_VISIBLE_DEVICES": "1", **{name: "1" for name in HIP_GUARDS}}
    if grouped and CANDIDATE_ENV:
        overrides[CANDIDATE_ENV] = "1"
    elif not grouped and BASELINE_ENV:
        overrides[BASELINE_ENV] = "1"
    write_json_new(
        directory / "command.json",
        {
            "argv": argv,
            "environment_overrides": overrides,
            "unset_environment": [
                "ROCR_VISIBLE_DEVICES",
                *EXPERIMENT_ENVIRONMENTS,
            ],
            "timer": "driver synchronized Instant measurement; rocprof is not used by this condition",
        },
    )
    started_ns = time.monotonic_ns()
    poller = MetricPoller(directory / "amd-smi-metric.jsonl")
    poller.start()
    poller.sample("immediately-before-process")
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, env=environment(grouped))
    finally:
        poller.sample("immediately-after-process")
        poller.stop()
    ended_ns = time.monotonic_ns()
    write_text_new(directory / "stdout.log", completed.stdout)
    write_text_new(directory / "stderr.log", completed.stderr)
    write_json_new(
        directory / "process.json",
        {
            "utc_completed": utc_now(),
            "monotonic_started_ns": started_ns,
            "monotonic_ended_ns": ended_ns,
            "wall_seconds_including_load_and_warmup": (ended_ns - started_ns) / 1e9,
            "returncode": completed.returncode,
        },
    )
    if completed.returncode:
        raise RuntimeError(f"{condition_id} failed with exit status {completed.returncode}")
    summary = parse_summary(completed.stdout)
    return {
        "condition": condition_id,
        "variant": label,
        "phase": phase,
        "prompt_tokens": prompt_tokens,
        "driver_summary": summary,
        "result_directory": str(directory),
    }


def main() -> int:
    if not DRIVER.is_file():
        raise RuntimeError(f"missing driver: {DRIVER}")
    if RAW.exists() and any(RAW.iterdir()):
        raise RuntimeError(f"refusing to mix runs in nonempty {RAW}")
    RAW.mkdir(parents=True, exist_ok=True)
    write_json_new(
        ROOT / CONFIG_NAME,
        {
            "utc_started": utc_now(),
            "driver": str(DRIVER),
            "prompt_lengths": PROMPTS,
            "repeats": REPEATS,
            "thermal_gate": THERMAL_GATE,
            "gpu": {"amd_smi_index": 2, "expected_bdf": "0000:47:00.0", "expected_gfx": "gfx1201"},
            "variants": {
                "generic": (
                    f"same candidate executable with {BASELINE_ENV}=1"
                    if BASELINE_ENV
                    else "same candidate executable with grouped environment variable absent"
                ),
                CANDIDATE_LABEL: CANDIDATE_DESCRIPTION,
            },
        },
    )
    results: list[dict[str, Any]] = []
    for label, grouped in (("generic", False), (CANDIDATE_LABEL, True)):
        for prompt in PROMPTS:
            results.append(condition(label, prompt, grouped, "prefill"))
    # Decode's timed region starts after the seed prefill.  The grouped flag is
    # retained deliberately to catch accidental cross-path selection.
    results.append(condition(CANDIDATE_LABEL, 1024, True, "decode"))
    write_json_new(
        ROOT / SUMMARY_NAME,
        {
            "schema_version": "ullm.prefill_attention_redesign.throughput.v1",
            "timer": "driver synchronized Instant; profiler-range durations are excluded",
            "conditions": results,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
