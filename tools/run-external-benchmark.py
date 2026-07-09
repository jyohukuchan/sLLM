#!/usr/bin/env python3
"""Run an external inference benchmark command and write one uLLM JSONL row."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inference-benchmark-result-v0.1"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ullm_format_ids import canonical_or_original


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off", "none", ""}:
        return False
    return None


def load_prompt_guard_bundle(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read prompt guard bundle {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"failed to parse prompt guard bundle {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"prompt guard bundle {path}: expected JSON object")
    return data


def prompt_suite_token_logits_check(bundle: dict[str, Any]) -> dict[str, Any] | None:
    checks = bundle.get("checks")
    if not isinstance(checks, list):
        return None
    for check in checks:
        if isinstance(check, dict) and check.get("name") == "prompt_suite_token_logits":
            return check
    return None


def prompt_suite_regression_status(check: dict[str, Any] | None) -> str:
    if check is None:
        return "not_attached"
    return "passed" if check.get("passed") is True else "failed"


def prompt_suite_token_logits_metrics(check: dict[str, Any] | None) -> dict[str, Any]:
    if check is None:
        return {}
    metrics = check.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def attach_prompt_guard_bundle_fields(
    row: dict[str, Any], bundle_path: Path, bundle: dict[str, Any]
) -> None:
    check = prompt_suite_token_logits_check(bundle)
    check_metrics = prompt_suite_token_logits_metrics(check)
    quality = row.setdefault("quality", {})
    quality["prompt_suite_regression_status"] = prompt_suite_regression_status(check)
    guards = row.setdefault("guards", {})
    guards["prompt_guard_bundle"] = {
        "status": "ok",
        "artifact": str(bundle_path),
        "passed": bundle.get("passed"),
        "acceptance_mode": check_metrics.get("acceptance_mode"),
        "strict_passed": check_metrics.get("strict_passed"),
        "behavioral_passed": check_metrics.get("behavioral_passed"),
        "compared_case_count": check_metrics.get("compared_case_count"),
        "generated_token_match_count": check_metrics.get("generated_token_match_count"),
        "generated_text_match_count": check_metrics.get("generated_text_match_count"),
        "generated_without_stop_text_match_count": check_metrics.get(
            "generated_without_stop_text_match_count"
        ),
        "top_logits_match_count": check_metrics.get("top_logits_match_count"),
        "max_prefill_top_logit_abs_diff": check_metrics.get(
            "max_prefill_top_logit_abs_diff"
        ),
        "max_decode_last_top_logit_abs_diff": check_metrics.get(
            "max_decode_last_top_logit_abs_diff"
        ),
    }
    artifacts = row.setdefault("artifacts", {})
    artifacts["prompt_guard_bundle_json"] = str(bundle_path)


def first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def canonical_with_legacy(value: str) -> tuple[str, str | None]:
    canonical = canonical_or_original(value)
    legacy = value if canonical != value else None
    return canonical, legacy


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def read_rocm_info() -> dict[str, Any]:
    output = run_text(
        ["rocm-smi", "--showproductname", "--showdriverversion", "--showmeminfo", "vram", "--json"]
    )
    if not output:
        return {}
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {}


def read_rocm_vram() -> dict[str, dict[str, Any]]:
    data = read_rocm_info()
    result: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if not key.startswith("card") or not isinstance(value, dict):
            continue
        result[key] = {
            "used_bytes": parse_int(value.get("VRAM Total Used Memory (B)")),
            "total_bytes": parse_int(value.get("VRAM Total Memory (B)")),
            "series": value.get("Card Series"),
            "gfx": value.get("GFX Version"),
        }
    return result


def total_used_bytes(sample: dict[str, dict[str, Any]]) -> int | None:
    values = [info.get("used_bytes") for info in sample.values()]
    numeric = [value for value in values if isinstance(value, int)]
    if not numeric:
        return None
    return sum(numeric)


def used_by_card(sample: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {
        card: value
        for card, info in sample.items()
        if isinstance((value := info.get("used_bytes")), int)
    }


class RocmMemoryMonitor:
    def __init__(self, log_path: Path, interval_seconds: float) -> None:
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline: dict[str, dict[str, Any]] = {}
        self.peak_by_card: dict[str, int] = {}
        self.peak_total_bytes: int | None = None
        self.sample_count = 0

    def _write_sample(self, stage: str, sample: dict[str, dict[str, Any]]) -> None:
        self.sample_count += 1
        total = total_used_bytes(sample)
        if total is not None:
            self.peak_total_bytes = max(total, self.peak_total_bytes or total)
        for card, used in used_by_card(sample).items():
            self.peak_by_card[card] = max(used, self.peak_by_card.get(card, used))
        row = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stage": stage,
            "total_used_bytes": total,
            "cards": sample,
        }
        with self.log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            output.write("\n")

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path.exists():
            self.log_path.unlink()
        self.baseline = read_rocm_vram()
        self._write_sample("baseline", self.baseline)
        self._thread = threading.Thread(target=self._run, name="rocm-memory-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write_sample("sample", read_rocm_vram())

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 2))
        self._write_sample("final", read_rocm_vram())
        return self.summary()

    def summary(self) -> dict[str, Any]:
        baseline_by_card = used_by_card(self.baseline)
        baseline_total = total_used_bytes(self.baseline)
        consumed_by_card = {
            card: max(0, peak - baseline_by_card.get(card, 0))
            for card, peak in self.peak_by_card.items()
        }
        consumed_total = None
        if baseline_total is not None and self.peak_total_bytes is not None:
            consumed_total = max(0, self.peak_total_bytes - baseline_total)
        return {
            "backend": "rocm-smi",
            "sample_interval_seconds": self.interval_seconds,
            "sample_count": self.sample_count,
            "baseline_total_bytes": baseline_total,
            "peak_total_bytes": self.peak_total_bytes,
            "consumed_total_bytes": consumed_total,
            "baseline_by_card_bytes": baseline_by_card,
            "peak_by_card_bytes": self.peak_by_card,
            "consumed_by_card_bytes": consumed_by_card,
            "log": str(self.log_path),
        }


def rocm_driver() -> str | None:
    data = read_rocm_info()
    system = data.get("system")
    if isinstance(system, dict):
        return system.get("Driver version")
    return None


def rocm_runtime() -> str | None:
    for path in (Path("/opt/rocm/.info/version"), Path("/opt/rocm-7.2.1/.info/version")):
        if path.exists():
            return f"ROCm {path.read_text(encoding='utf-8').strip()}"
    return None


def selected_gpus(cards: list[str]) -> list[dict[str, Any]]:
    data = read_rocm_info()
    gpus: list[dict[str, Any]] = []
    for card in cards:
        info = data.get(card)
        if not isinstance(info, dict):
            continue
        gpus.append(
            {
                "name": info.get("Card Series"),
                "gfx": info.get("GFX Version"),
                "vram_bytes": parse_int(info.get("VRAM Total Memory (B)")),
                "rocm_smi_card": card,
            }
        )
    return gpus


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def env_prefix(names: list[str]) -> str:
    parts: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            parts.append(f"{name}={shlex.quote(value)}")
    return " ".join(parts)


def parse_vllm_metrics(stdout: str, output_json: Path | None, memory: dict[str, Any]) -> dict[str, Any]:
    elapsed = None
    total_tokens_per_second = None
    requests_per_second = None
    if output_json and output_json.exists():
        try:
            data = json.loads(output_json.read_text(encoding="utf-8"))
            elapsed = parse_float(data.get("elapsed_time"))
            total_tokens_per_second = parse_float(data.get("tokens_per_second"))
            requests_per_second = parse_float(data.get("requests_per_second"))
        except json.JSONDecodeError:
            pass

    decode_tokens_per_second = None
    match = re.search(
        r"Throughput:\s+([0-9.]+)\s+requests/s,\s+([0-9.]+)\s+total tokens/s,\s+([0-9.]+)\s+output tokens/s",
        stdout,
    )
    if match:
        requests_per_second = parse_float(match.group(1))
        total_tokens_per_second = parse_float(match.group(2))
        decode_tokens_per_second = parse_float(match.group(3))

    total_prompt_tokens = None
    match = re.search(r"Total num prompt tokens:\s+([0-9]+)", stdout)
    if match:
        total_prompt_tokens = parse_int(match.group(1))

    prefill_tokens_per_second = None
    if elapsed and total_prompt_tokens is not None and elapsed > 0:
        prefill_tokens_per_second = total_prompt_tokens / elapsed

    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_tokens_per_second is not None and consumed_gib is not None:
        product = decode_tokens_per_second * consumed_gib

    metrics = {
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "requests_per_second": requests_per_second,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }
    return metrics


def parse_sglang_serving_metrics(output_json: Path | None, memory: dict[str, Any]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if output_json and output_json.exists():
        text = output_json.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            lines = [line for line in text.splitlines() if line.strip()]
            if lines:
                parsed = json.loads(lines[-1])
                if isinstance(parsed, dict):
                    data = parsed

    decode_tokens_per_second = parse_float(data.get("output_throughput"))
    total_tokens_per_second = parse_float(
        data.get("total_throughput", data.get("total_token_throughput"))
    )
    prefill_tokens_per_second = parse_float(data.get("input_throughput"))
    if (
        prefill_tokens_per_second is None
        and total_tokens_per_second is not None
        and decode_tokens_per_second is not None
    ):
        prefill_tokens_per_second = total_tokens_per_second - decode_tokens_per_second
    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_tokens_per_second is not None and consumed_gib is not None:
        product = decode_tokens_per_second * consumed_gib

    return {
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "requests_per_second": parse_float(data.get("request_throughput")),
        "latency_p50_ms": parse_float(
            data.get("median_e2e_latency_ms", data.get("median_e2el_ms"))
        ),
        "latency_p95_ms": parse_float(data.get("p95_e2e_latency_ms", data.get("p95_e2el_ms"))),
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }


def parse_json_object_text(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_scalar_text(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    parsed_int = parse_int(value)
    if parsed_int is not None and str(parsed_int) == value:
        return parsed_int
    parsed_float = parse_float(value)
    if parsed_float is not None:
        return parsed_float
    return value


def is_selected_layer_sq8_diagnostic_command(command: Any) -> bool:
    return (
        isinstance(command, str)
        and command.startswith("sq-fp8-token-ids-model-loop-smoke")
    )


def sq_execution_mode_allows_fallback(report: dict[str, Any], allow_cli: bool) -> bool:
    if allow_cli:
        return True
    diagnostic = parse_bool(report.get("diagnostic"))
    if diagnostic is True:
        return True
    fallback_allowed = parse_bool(report.get("fallback_allowed"))
    if fallback_allowed is True:
        return True
    command = report.get("command")
    if is_selected_layer_sq8_diagnostic_command(command):
        return True
    return False


def is_materialized_sq_fallback(report: dict[str, Any]) -> bool:
    return (
        isinstance(report.get("sq_execution_mode"), str)
        and report.get("sq_execution_mode") == "materialized_f32_fallback"
    )


def parse_int_csv(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            return []
        number = parse_int(item)
        if number is None:
            return []
        parsed.append(number)
    return parsed


def parse_float_csv(value: Any) -> list[float]:
    if isinstance(value, int | float):
        return [float(value)]
    if not isinstance(value, str) or not value.strip():
        return []
    parsed: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            return []
        number = parse_float(item)
        if number is None:
            return []
        parsed.append(number)
    return parsed


def parse_int_matrix_csv(value: Any) -> list[list[int]]:
    if isinstance(value, int):
        return [[value]]
    if not isinstance(value, str) or not value.strip():
        return []
    rows: list[list[int]] = []
    for raw_row in value.split(";"):
        row = parse_int_csv(raw_row.replace(":", ","))
        if not row:
            return []
        rows.append(row)
    return rows


def parse_float_matrix_csv(value: Any) -> list[list[float]]:
    if isinstance(value, int | float):
        return [[float(value)]]
    if not isinstance(value, str) or not value.strip():
        return []
    rows: list[list[float]] = []
    for raw_row in value.split(";"):
        row = parse_float_csv(raw_row.replace(":", ","))
        if not row:
            return []
        rows.append(row)
    return rows


def parse_key_value_stdout(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return {}
    parts = shlex.split(lines[-1])
    if not parts:
        return {}
    report: dict[str, Any] = {"command": parts[0]}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, raw_value = part.split("=", 1)
        if not key:
            continue
        report[key] = parse_scalar_text(raw_value)
    return report


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_ullm_token_ids_report(stdout: str, output_json: Path | None) -> dict[str, Any]:
    report = parse_json_object_text(stdout)
    if report and output_json:
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_ullm_key_value_report(stdout: str, output_json: Path | None) -> dict[str, Any]:
    report = parse_key_value_stdout(stdout)
    if report and output_json:
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def parse_ullm_token_ids_metrics(
    report: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    prefill = report.get("prefill") if isinstance(report.get("prefill"), dict) else {}
    decode = report.get("decode") if isinstance(report.get("decode"), dict) else {}
    throughput = report.get("throughput") if isinstance(report.get("throughput"), dict) else {}
    timing = report.get("timing_ms") if isinstance(report.get("timing_ms"), dict) else {}

    prefill_ms = parse_float(prefill.get("wall_ms", timing.get("prefill")))
    decode_ms = parse_float(decode.get("wall_ms", timing.get("decode")))
    total_ms = parse_float(throughput.get("total_wall_ms", timing.get("total")))
    decode_tokens_per_second = parse_float(decode.get("timed_step_tps"))
    if decode_tokens_per_second is None:
        decode_tokens_per_second = parse_float(decode.get("end_to_end_generated_tps"))
    total_tokens_per_second = parse_float(throughput.get("model_input_tps"))
    if total_tokens_per_second is None:
        total_tokens_per_second = parse_float(throughput.get("full_forward_tps"))
    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_tokens_per_second is not None and consumed_gib is not None:
        product = decode_tokens_per_second * consumed_gib

    step_ms = decode.get("step_wall_ms")
    step_values = [parse_float(value) for value in step_ms] if isinstance(step_ms, list) else []
    step_values = [value for value in step_values if value is not None]

    return {
        "prefill_tokens_per_second": parse_float(prefill.get("tps")),
        "decode_tokens_per_second": decode_tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "prefill_wall_time_seconds": prefill_ms / 1000.0 if prefill_ms is not None else None,
        "decode_wall_time_seconds": decode_ms / 1000.0 if decode_ms is not None else None,
        "total_wall_time_seconds": total_ms / 1000.0 if total_ms is not None else None,
        "time_to_first_token_ms": prefill_ms,
        "time_per_output_token_ms": (1000.0 / decode_tokens_per_second)
        if decode_tokens_per_second
        else None,
        "latency_p50_ms": percentile(step_values, 0.50),
        "latency_p95_ms": percentile(step_values, 0.95),
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }


def parse_ullm_batch_throughput_metrics(
    report: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    prefill_total_tps = parse_float(metrics.get("prefill_total_input_tps"))
    decode_total_tps = parse_float(metrics.get("decode_total_generated_tps"))
    end_to_end_total_tps = parse_float(metrics.get("end_to_end_total_tps"))
    prefill_ms = parse_float(metrics.get("prefill_wall_ms_sum"))
    decode_ms = parse_float(metrics.get("decode_wall_ms_sum"))
    batch_ms = parse_float(metrics.get("batch_wall_ms"))
    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_total_tps is not None and consumed_gib is not None:
        product = decode_total_tps * consumed_gib

    return {
        "prefill_tokens_per_second": prefill_total_tps,
        "decode_tokens_per_second": decode_total_tps,
        "total_tokens_per_second": end_to_end_total_tps,
        "prefill_total_input_tokens": parse_int(metrics.get("prefill_total_input_tokens")),
        "decode_total_generated_tokens": parse_int(metrics.get("decode_total_generated_tokens")),
        "generated_tokens_total": parse_int(metrics.get("generated_tokens_total")),
        "end_to_end_total_tokens": parse_int(metrics.get("end_to_end_total_tokens")),
        "prefill_total_input_tokens_per_second": prefill_total_tps,
        "decode_total_generated_tokens_per_second": decode_total_tps,
        "end_to_end_total_tokens_per_second": end_to_end_total_tps,
        "prefill_wall_time_seconds": prefill_ms / 1000.0 if prefill_ms is not None else None,
        "decode_wall_time_seconds": decode_ms / 1000.0 if decode_ms is not None else None,
        "total_wall_time_seconds": batch_ms / 1000.0 if batch_ms is not None else None,
        "time_to_first_token_ms": parse_float(metrics.get("time_to_first_token_ms_p50")),
        "time_per_output_token_ms": parse_float(metrics.get("time_per_output_token_ms_p50")),
        "latency_p50_ms": parse_float(metrics.get("request_latency_ms_p50")),
        "latency_p95_ms": parse_float(metrics.get("request_latency_ms_p95")),
        "time_to_first_token_ms_p50": parse_float(metrics.get("time_to_first_token_ms_p50")),
        "time_to_first_token_ms_p95": parse_float(metrics.get("time_to_first_token_ms_p95")),
        "request_latency_ms_p50": parse_float(metrics.get("request_latency_ms_p50")),
        "request_latency_ms_p95": parse_float(metrics.get("request_latency_ms_p95")),
        "time_per_output_token_ms_p50": parse_float(metrics.get("time_per_output_token_ms_p50")),
        "time_per_output_token_ms_p95": parse_float(metrics.get("time_per_output_token_ms_p95")),
        "per_request_decode_tps_mean": parse_float(metrics.get("per_request_decode_tps_mean")),
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }


def parse_ullm_component_prefill_metrics(
    report: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    prompt_tokens = first_non_null(
        parse_int(report.get("prompt_tokens_per_request")),
        parse_int(report.get("prompt_tokens")),
        parse_int(report.get("new_prefill_tokens")),
    )
    request_parallelism = first_non_null(
        parse_int(report.get("request_parallelism")),
        parse_int(report.get("concurrent_requests")),
        parse_int(report.get("batch_count")),
        1,
    )
    prefill_tokens = parse_int(report.get("prefill_total_input_tokens"))
    if (
        prefill_tokens is None
        and prompt_tokens is not None
        and request_parallelism is not None
    ):
        prefill_tokens = prompt_tokens * request_parallelism
    prefill_tps = first_non_null(
        parse_float(report.get("prefill_total_input_tps")),
        parse_float(report.get("token_tps_mean")),
    )
    wall_ms = parse_float(report.get("wall_ms_mean"))
    consumed_bytes = memory.get("consumed_total_bytes")
    return {
        "prefill_tokens_per_second": prefill_tps,
        "decode_tokens_per_second": None,
        "total_tokens_per_second": prefill_tps,
        "prefill_total_input_tokens": prefill_tokens,
        "decode_total_generated_tokens": 0,
        "generated_tokens_total": 0,
        "end_to_end_total_tokens": prefill_tokens,
        "prefill_total_input_tokens_per_second": prefill_tps,
        "decode_total_generated_tokens_per_second": None,
        "end_to_end_total_tokens_per_second": prefill_tps,
        "prefill_wall_time_seconds": wall_ms / 1000.0 if wall_ms is not None else None,
        "decode_wall_time_seconds": 0.0,
        "total_wall_time_seconds": wall_ms / 1000.0 if wall_ms is not None else None,
        "time_to_first_token_ms": wall_ms,
        "time_per_output_token_ms": None,
        "latency_p50_ms": wall_ms,
        "latency_p95_ms": wall_ms,
        "time_to_first_token_ms_p50": wall_ms,
        "time_to_first_token_ms_p95": wall_ms,
        "request_latency_ms_p50": wall_ms,
        "request_latency_ms_p95": wall_ms,
        "time_per_output_token_ms_p50": None,
        "time_per_output_token_ms_p95": None,
        "per_request_decode_tps_mean": None,
        "attention_pair_tps_mean": parse_float(report.get("attention_pair_tps_mean")),
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": None,
        "power_watts_avg": None,
    }


def parse_ullm_model_loop_metrics(
    report: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    prefill_total_tps = parse_float(report.get("prefill_total_input_tps"))
    decode_total_tps = parse_float(report.get("decode_total_generated_tps"))
    end_to_end_total_tps = parse_float(report.get("end_to_end_total_tps"))
    prefill_ms = parse_float(report.get("prefill_wall_ms"))
    decode_ms = parse_float(report.get("decode_wall_ms"))
    final_logits_ms = parse_float(report.get("final_logits_wall_ms"))
    layer_load_ms = parse_float(report.get("layer_load_ms"))
    artifact_load_ms = first_non_null(
        parse_float(report.get("artifact_load_ms")),
        layer_load_ms,
    )
    artifact_materialization_ms = first_non_null(
        parse_float(report.get("artifact_materialization_ms")),
        parse_float(report.get("materialization_ms")),
        parse_float(report.get("materialize_ms")),
        parse_float(report.get("sq_materialization_ms")),
    )
    total_ms = parse_float(report.get("total_wall_ms"))
    outer_ms = parse_float(report.get("outer_wall_ms"))
    consumed_bytes = memory.get("consumed_total_bytes")
    consumed_gib = consumed_bytes / 1024**3 if isinstance(consumed_bytes, int) else None
    product = None
    if decode_total_tps is not None and consumed_gib is not None:
        product = decode_total_tps * consumed_gib

    return {
        "prefill_tokens_per_second": prefill_total_tps,
        "decode_tokens_per_second": decode_total_tps,
        "total_tokens_per_second": end_to_end_total_tps,
        "prefill_total_input_tokens": parse_int(report.get("prefill_total_input_tokens")),
        "decode_total_generated_tokens": parse_int(report.get("decode_total_generated_tokens")),
        "generated_tokens_total": parse_int(report.get("decode_total_generated_tokens")),
        "end_to_end_total_tokens": parse_int(report.get("end_to_end_total_tokens")),
        "prefill_total_input_tokens_per_second": prefill_total_tps,
        "decode_total_generated_tokens_per_second": decode_total_tps,
        "end_to_end_total_tokens_per_second": end_to_end_total_tps,
        "prefill_wall_time_seconds": prefill_ms / 1000.0 if prefill_ms is not None else None,
        "decode_wall_time_seconds": decode_ms / 1000.0 if decode_ms is not None else None,
        "final_logits_wall_time_seconds": (
            final_logits_ms / 1000.0 if final_logits_ms is not None else None
        ),
        "artifact_load_wall_time_seconds": (
            artifact_load_ms / 1000.0 if artifact_load_ms is not None else None
        ),
        "artifact_materialization_wall_time_seconds": (
            artifact_materialization_ms / 1000.0
            if artifact_materialization_ms is not None
            else None
        ),
        "layer_load_wall_time_seconds": (
            layer_load_ms / 1000.0 if layer_load_ms is not None else None
        ),
        "total_wall_time_seconds": total_ms / 1000.0 if total_ms is not None else None,
        "load_excluded_total_wall_time_seconds": (
            total_ms / 1000.0 if total_ms is not None else None
        ),
        "load_included_total_wall_time_seconds": (
            outer_ms / 1000.0 if outer_ms is not None else None
        ),
        "outer_wall_time_seconds": outer_ms / 1000.0 if outer_ms is not None else None,
        "time_to_first_token_ms": prefill_ms,
        "time_per_output_token_ms": (
            (1000.0 / decode_total_tps) if decode_total_tps else None
        ),
        "latency_p50_ms": total_ms,
        "latency_p95_ms": total_ms,
        "time_to_first_token_ms_p50": prefill_ms,
        "time_to_first_token_ms_p95": prefill_ms,
        "request_latency_ms_p50": total_ms,
        "request_latency_ms_p95": total_ms,
        "time_per_output_token_ms_p50": (
            (1000.0 / decode_total_tps) if decode_total_tps else None
        ),
        "time_per_output_token_ms_p95": (
            (1000.0 / decode_total_tps) if decode_total_tps else None
        ),
        "per_request_decode_tps_mean": None,
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": consumed_bytes,
        "decode_tokens_per_second_times_vram_consumed_gib": product,
        "power_watts_avg": None,
    }


def parse_ullm_token_ids_correctness(report: dict[str, Any]) -> dict[str, Any] | None:
    if not report:
        return None
    correctness = report.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    nan_or_inf = correctness.get("nan_or_inf_detected")
    verified = report.get("verified")
    if not isinstance(verified, bool):
        verified = correctness.get("verified")
    return {
        "reference": "none",
        "reference_artifact": None,
        "logits_relative_mse": None,
        "logits_max_abs_diff": None,
        "top_k": parse_int(report.get("top_k")),
        "top_k_agreement": None,
        "generated_prefix_matches_reference": verified if isinstance(verified, bool) else None,
        "nan_count": 0 if nan_or_inf is False else None,
        "inf_count": 0 if nan_or_inf is False else None,
        "logit_min": None,
        "logit_max": None,
    }


def parse_ullm_batch_throughput_correctness(report: dict[str, Any]) -> dict[str, Any] | None:
    if not report:
        return None
    correctness = report.get("correctness")
    correctness = correctness if isinstance(correctness, dict) else {}
    verified_all = report.get("verified")
    if not isinstance(verified_all, bool):
        verified_all = correctness.get("verified_all")
    return {
        "reference": "none",
        "reference_artifact": None,
        "logits_relative_mse": None,
        "logits_max_abs_diff": None,
        "top_k": parse_int(report.get("top_k")),
        "top_k_agreement": None,
        "generated_prefix_matches_reference": verified_all
        if isinstance(verified_all, bool)
        else None,
        "nan_count": None,
        "inf_count": None,
        "logit_min": None,
        "logit_max": None,
        "verified_all": verified_all if isinstance(verified_all, bool) else None,
    }


def enrich_ullm_batch_workload(row: dict[str, Any], report: dict[str, Any]) -> None:
    workload = report.get("workload")
    if not isinstance(workload, dict):
        return
    row_workload = row.get("workload")
    if not isinstance(row_workload, dict):
        return
    for key in (
        "prefill_mode",
        "prefill_executor",
        "resolved_prefill_executor",
        "prompt_tokens_per_request",
        "cached_prefix_tokens_per_request",
        "new_prefill_tokens_per_request",
        "total_context_tokens_after_prefill_per_request",
        "generated_tokens_per_request",
        "fixed_decode_steps",
    ):
        if key in workload:
            row_workload[key] = workload.get(key)
    for key in ("batch_size", "concurrent_requests"):
        value = parse_int(workload.get(key))
        if value is not None:
            row_workload[key] = value
    metrics = report.get("metrics")
    if isinstance(metrics, dict):
        for key in (
            "cached_prefix_total_tokens",
            "total_context_tokens_after_prefill",
            "estimated_prefill_attention_work_tokens",
        ):
            value = parse_int(metrics.get(key))
            if value is not None:
                row_workload[key] = value
    batching = report.get("batching")
    if isinstance(batching, dict):
        if row_workload.get("prefill_executor") is None and "prefill_executor" in batching:
            row_workload["prefill_executor"] = batching.get("prefill_executor")
        if (
            row_workload.get("resolved_prefill_executor") is None
            and "resolved_prefill_executor" in batching
        ):
            row_workload["resolved_prefill_executor"] = batching.get(
                "resolved_prefill_executor"
            )


def enrich_ullm_batch_memory(row: dict[str, Any], report: dict[str, Any]) -> None:
    row_memory = row.get("memory")
    raw_memory = report.get("memory")
    if not isinstance(row_memory, dict) or not isinstance(raw_memory, dict):
        return
    row_memory.update(
        {
            "kv_cache_bytes": raw_memory.get("kv_cache_bytes"),
            "kv_cache_allocated_blocks": raw_memory.get("kv_cache_allocated_blocks"),
            "kv_cache_free_blocks": raw_memory.get("kv_cache_free_blocks"),
            "kv_cache_block_size": raw_memory.get("kv_cache_block_size"),
            "kv_cache_bytes_total": raw_memory.get("kv_cache_bytes_total"),
        }
    )


def enrich_ullm_component_prefill_row(row: dict[str, Any], report: dict[str, Any]) -> None:
    row_workload = row.get("workload")
    if not isinstance(row_workload, dict):
        return
    request_parallelism = first_non_null(parse_int(report.get("request_parallelism")), 1)
    reported_batch_count = parse_int(report.get("batch_count"))
    reported_concurrent_requests = parse_int(report.get("concurrent_requests"))
    existing_batch_count = parse_int(row_workload.get("batch_size"))
    existing_concurrent_requests = parse_int(row_workload.get("concurrent_requests"))
    batch_count = first_non_null(
        reported_batch_count,
        reported_concurrent_requests,
        existing_batch_count,
        request_parallelism,
    )
    concurrent_requests = first_non_null(
        reported_concurrent_requests,
        existing_concurrent_requests,
        batch_count,
    )
    reported_prompt_tokens_per_request = parse_int(report.get("prompt_tokens_per_request"))
    component_prompt_tokens = parse_int(report.get("prompt_tokens"))
    cached_prefix_tokens = parse_int(report.get("cached_prefix_tokens"))
    new_prefill_tokens = parse_int(report.get("new_prefill_tokens"))
    total_context_tokens_after_prefill = parse_int(
        report.get("total_context_tokens_after_prefill")
    )
    prompt_tokens = first_non_null(
        reported_prompt_tokens_per_request,
        parse_int(row_workload.get("prompt_tokens")),
        component_prompt_tokens,
        new_prefill_tokens,
    )
    prefill_tokens = parse_int(report.get("prefill_total_input_tokens"))
    if prefill_tokens is None:
        if new_prefill_tokens is not None and concurrent_requests is not None:
            prefill_tokens = new_prefill_tokens * concurrent_requests
        elif component_prompt_tokens is not None:
            prefill_tokens = component_prompt_tokens
        elif prompt_tokens is not None and concurrent_requests is not None:
            prefill_tokens = prompt_tokens * concurrent_requests
    estimated_work = parse_int(report.get("estimated_prefill_attention_work_tokens"))
    executor = report.get("executor")
    resolved_executor = report.get("resolved_executor")
    selected_implementation_id = report.get("selected_implementation_id")
    dispatch_metadata = {
        "executor_selection": report.get("executor_selection"),
        "dispatch_operation": report.get("dispatch_operation"),
        "dispatch_phase": report.get("dispatch_phase"),
        "dispatch_format_id": report.get("dispatch_format_id"),
        "dispatch_gpu_arch": report.get("dispatch_gpu_arch"),
    }
    batching_mode = report.get("batching_mode")
    if not isinstance(batching_mode, str):
        batching_mode = "real" if report.get("real_batch") is True else None
    if batch_count is not None:
        row_workload["batch_size"] = batch_count
    if concurrent_requests is not None:
        row_workload["concurrent_requests"] = concurrent_requests
    row_workload["prefill_mode"] = report.get("prefill_mode") or "cold"
    if isinstance(executor, str):
        row_workload["prefill_executor"] = executor
    if isinstance(resolved_executor, str):
        row_workload["resolved_prefill_executor"] = resolved_executor
    elif isinstance(executor, str):
        row_workload["resolved_prefill_executor"] = executor
    if isinstance(selected_implementation_id, str):
        row_workload["selected_implementation_id"] = selected_implementation_id
        row_workload["dispatch_selected_implementation_id"] = selected_implementation_id
    enrich_ullm_sq_projection_workload(row_workload, report)
    for key, value in dispatch_metadata.items():
        if value is not None:
            row_workload[key] = value
    if prompt_tokens is not None and concurrent_requests is not None:
        per_request = [prompt_tokens for _ in range(concurrent_requests)]
        row_workload["prompt_tokens_per_request"] = per_request
        per_request_cached_prefix = cached_prefix_tokens if cached_prefix_tokens is not None else 0
        per_request_new_prefill = (
            new_prefill_tokens if new_prefill_tokens is not None else prompt_tokens
        )
        per_request_total_context = first_non_null(
            total_context_tokens_after_prefill,
            (
                per_request_cached_prefix + per_request_new_prefill
                if cached_prefix_tokens is not None or new_prefill_tokens is not None
                else None
            ),
            prompt_tokens,
        )
        row_workload["cached_prefix_tokens_per_request"] = [
            per_request_cached_prefix for _ in range(concurrent_requests)
        ]
        row_workload["new_prefill_tokens_per_request"] = [
            per_request_new_prefill for _ in range(concurrent_requests)
        ]
        row_workload["total_context_tokens_after_prefill_per_request"] = [
            per_request_total_context for _ in range(concurrent_requests)
        ]
    if prefill_tokens is not None:
        row_workload["total_context_tokens_after_prefill"] = prefill_tokens
        row_workload["component_total_input_tokens"] = prefill_tokens
    if cached_prefix_tokens is not None and concurrent_requests is not None:
        row_workload["cached_prefix_total_tokens"] = cached_prefix_tokens * concurrent_requests
    if total_context_tokens_after_prefill is not None and concurrent_requests is not None:
        row_workload["total_context_tokens_after_prefill"] = (
            total_context_tokens_after_prefill * concurrent_requests
        )
    if estimated_work is not None:
        row_workload["estimated_prefill_attention_work_tokens"] = estimated_work
    row["batching"] = {
        "mode": batching_mode,
        "prefill_executor": executor,
        "resolved_prefill_executor": row_workload.get("resolved_prefill_executor"),
        "prefill_real_batch": batching_mode == "real",
        "prefill_executor_token_parallelism": parse_int(report.get("token_parallelism")),
        "prefill_executor_request_parallelism": parse_int(report.get("request_parallelism")),
        "decode_executor": None,
        "decode_real_batch": False,
        "decode_executor_request_parallelism": 0,
        "scheduler_policy": "component_fixed_batch",
        "component_command": report.get("command"),
        "component_package": report.get("package"),
    }
    for key in (
        "dispatch_selected_implementation_id",
        "executor_selection",
        "dispatch_operation",
        "dispatch_phase",
        "dispatch_format_id",
        "dispatch_gpu_arch",
    ):
        value = row_workload.get(key)
        if value is not None:
            row["batching"][key] = value


def enrich_ullm_sq_projection_workload(
    row_workload: dict[str, Any], report: dict[str, Any]
) -> None:
    sq_overlay = report.get("sq_overlay") is True
    if sq_overlay:
        row_workload["sq_overlay"] = True
    for key in (
        "format_id",
        "sq_candidate",
        "sq_candidate_legacy",
        "sq_format_id",
        "sq_implementation_id",
        "sq_artifact",
        "sq_schema_version",
        "sq_execution_mode",
        "sq_projection_boundary",
        "sq_projection_implementation_ids",
    ):
        value = report.get(key)
        if isinstance(value, str) and value != "none":
            if key == "sq_candidate":
                canonical, legacy = canonical_with_legacy(value)
                row_workload[key] = canonical
                if legacy is not None:
                    row_workload["sq_candidate_legacy"] = legacy
            elif key in {"format_id", "sq_format_id"}:
                row_workload[key] = canonical_or_original(value)
            else:
                row_workload[key] = value
        elif isinstance(value, str) and sq_overlay and key in {
            "sq_projection_boundary",
            "sq_projection_implementation_ids",
        }:
            row_workload[key] = value
    for key in (
        "sq_fp8_tensor_count",
        "sq_passthrough_tensor_count",
        "sq_row_chunk",
        "sq_fp8_single_matvec_count",
        "sq_fp8_batch_matvec_count",
        "sq_fp8_expected_all_batch_matvec_count",
        "sq_fp8_pair_matvec_count",
        "sq_fp8_triple_matvec_count",
        "sq_diagnostic_host_staging_read_count",
        "sq_diagnostic_host_staging_write_count",
        "sq_diagnostic_host_staging_read_bytes",
        "sq_diagnostic_host_staging_write_bytes",
        "prefill_sq_fp8_batch_matvec_count",
        "decode_sq_fp8_batch_matvec_count",
    ):
        value = parse_int(report.get(key))
        if value is not None and (
            value > 0
            or key.startswith("sq_fp8_")
            or key.startswith("sq_diagnostic_host_staging_")
            or key in {
                "prefill_sq_fp8_batch_matvec_count",
                "decode_sq_fp8_batch_matvec_count",
            }
        ):
            row_workload[key] = value


def enrich_ullm_model_loop_row(row: dict[str, Any], report: dict[str, Any]) -> None:
    row_workload = row.get("workload")
    if not isinstance(row_workload, dict):
        return
    request_count = first_non_null(
        parse_int(report.get("concurrent_requests")),
        parse_int(report.get("request_count")),
        parse_int(row_workload.get("concurrent_requests")),
    )
    prompt_tokens_per_request = parse_int_csv(report.get("prompt_tokens_csv"))
    generated_tokens_per_request = parse_int_csv(report.get("generated_tokens_csv"))
    if not generated_tokens_per_request:
        generated_tokens_per_request = parse_int_csv(report.get("max_new_tokens_csv"))
    total_context_tokens_per_request = parse_int_csv(report.get("total_tokens_csv"))
    layers_csv = report.get("layers_csv")
    if request_count is not None:
        row_workload["batch_size"] = request_count
        row_workload["concurrent_requests"] = request_count
    row_workload["prefill_mode"] = report.get("prefill_mode") or "synthetic_layer_stack"
    if isinstance(report.get("prefill_executor"), str):
        row_workload["prefill_executor"] = report.get("prefill_executor")
        row_workload["resolved_prefill_executor"] = report.get("prefill_executor")
    if prompt_tokens_per_request:
        row_workload["prompt_tokens_per_request"] = prompt_tokens_per_request
        row_workload["cached_prefix_tokens_per_request"] = [
            0 for _ in prompt_tokens_per_request
        ]
        row_workload["new_prefill_tokens_per_request"] = prompt_tokens_per_request
    if total_context_tokens_per_request:
        row_workload[
            "total_context_tokens_after_prefill_per_request"
        ] = total_context_tokens_per_request
    if generated_tokens_per_request:
        row_workload["generated_tokens_per_request"] = generated_tokens_per_request
    for key in (
        "prefill_total_input_tokens",
        "decode_total_generated_tokens",
        "end_to_end_total_tokens",
    ):
        value = parse_int(report.get(key))
        if value is not None:
            row_workload[key] = value
    if isinstance(layers_csv, str):
        row_workload["layers_csv"] = layers_csv
    if isinstance(report.get("input_source"), str):
        row_workload["input_source"] = report.get("input_source")
    executor = report.get("executor")
    real_batch = parse_bool(report.get("real_batch")) is True
    request_parallelism = parse_int(report.get("request_parallelism"))
    batching_mode = report.get("batching_mode")
    if not isinstance(batching_mode, str):
        batching_mode = "real" if real_batch else None
    enrich_ullm_sq_projection_workload(row_workload, report)
    final_top1_tokens = parse_int_csv(report.get("final_top1_tokens_csv"))
    if final_top1_tokens:
        row_workload["final_top1_tokens"] = final_top1_tokens
    final_topk_tokens = parse_int_matrix_csv(report.get("final_topk_tokens_csv"))
    if final_topk_tokens:
        row_workload["final_topk_tokens"] = final_topk_tokens
    final_topk_logits = parse_float_matrix_csv(report.get("final_topk_logits_csv"))
    if final_topk_logits:
        row_workload["final_topk_logits"] = final_topk_logits
    sequence_len = parse_int(report.get("sequence_len"))
    if sequence_len is not None:
        row_workload["sequence_len"] = sequence_len

    if isinstance(executor, str) and isinstance(row_workload.get("prefill_executor"), str) is False:
        row_workload["prefill_executor"] = executor
        row_workload["resolved_prefill_executor"] = executor
    prefill_batch_request_counts = parse_int_csv(
        report.get("prefill_batch_request_counts_csv")
    )
    prefill_request_grouped = parse_bool(report.get("prefill_request_grouped")) is True
    decode_request_grouped = parse_bool(report.get("decode_request_grouped")) is True
    row["batching"] = {
        "mode": batching_mode if isinstance(batching_mode, str) else "hybrid",
        "prefill_executor": row_workload.get("prefill_executor") or executor,
        "resolved_prefill_executor": row_workload.get("resolved_prefill_executor")
        or executor,
        "prefill_real_batch": report.get("prefill_real_batch") is True
        or real_batch,
        "prefill_executor_token_parallelism": parse_int(report.get("token_parallelism")),
        "prefill_executor_request_parallelism": first_non_null(
            parse_int(report.get("prefill_executor_request_parallelism")),
            request_parallelism,
        ),
        "prefill_request_grouped": prefill_request_grouped,
        "prefill_grouped_request_parallelism": parse_int(
            report.get("prefill_grouped_request_parallelism")
        ),
        "prefill_sq_fp8_batch_matvec_count": parse_int(
            report.get("prefill_sq_fp8_batch_matvec_count")
        ),
        "decode_executor": report.get("decode_executor"),
        "decode_real_batch": report.get("decode_real_batch") is True,
        "decode_executor_request_parallelism": parse_int(
            report.get("decode_executor_request_parallelism")
        ),
        "decode_request_grouped": decode_request_grouped,
        "decode_grouped_request_parallelism": parse_int(
            report.get("decode_grouped_request_parallelism")
        ),
        "decode_sq_fp8_batch_matvec_count": parse_int(
            report.get("decode_sq_fp8_batch_matvec_count")
        ),
        "mixed_request_state_real_batch_projection_used": report.get(
            "mixed_request_state_real_batch_projection_used"
        )
        is True,
        "request_batch_executor": report.get("request_batch_executor") is True,
        "fused_request_batch": report.get("fused_request_batch") is True,
        "throughput_row": report.get("throughput_row") is True,
        "load_excluded_from_total": report.get("load_excluded_from_total") is True,
        "final_logits_in_total": report.get("final_logits_in_total") is True,
        "scheduler_policy": "model_loop_ready_batch",
        "component_command": report.get("command"),
        "component_package": report.get("package"),
    }
    if prefill_batch_request_counts:
        row["batching"]["prefill_batch_request_counts"] = prefill_batch_request_counts


def default_metrics(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "prefill_tokens_per_second": None,
        "decode_tokens_per_second": None,
        "total_tokens_per_second": None,
        "latency_p50_ms": None,
        "latency_p95_ms": None,
        "vram_baseline_bytes": memory.get("baseline_total_bytes"),
        "vram_peak_bytes": memory.get("peak_total_bytes"),
        "vram_consumed_bytes": memory.get("consumed_total_bytes"),
        "decode_tokens_per_second_times_vram_consumed_gib": None,
        "power_watts_avg": None,
    }


def classify_benchmark_harness(parse_name: str) -> dict[str, Any]:
    if parse_name == "vllm-throughput":
        return {
            "class": "serving_throughput_benchmark",
            "serving_parity_candidate": True,
            "includes_http_server": False,
            "notes": ["source=vllm_cli_offline_throughput"],
            "harness_type": "vllm_bench_throughput_cli",
        }
    if parse_name == "ullm-model-loop-throughput":
        return {
            "class": "cli_model_loop_diagnostic",
            "serving_parity_candidate": False,
            "includes_http_server": False,
            "notes": ["source=ullm_cli_model_loop"],
            "harness_type": "ullm_cli_model_loop",
        }
    if parse_name == "ullm-package-batch-throughput":
        return {
            "class": "cli_logical_batch_diagnostic",
            "serving_parity_candidate": False,
            "includes_http_server": False,
            "notes": ["source=ullm_package_batch_cli"],
            "harness_type": "ullm_cli_logical_batch",
        }
    return {
        "class": "unknown_or_unclassified",
        "serving_parity_candidate": False,
        "includes_http_server": False,
        "notes": ["source=unclassified_or_missing_parse"],
        "harness_type": "unknown",
    }


def classify_failure(returncode: int, timed_out: bool, text: str) -> tuple[str, dict[str, str]]:
    lowered = text.lower()
    if timed_out:
        return "failed", {"type": "timeout", "message": "Benchmark command timed out."}
    if "out of memory" in lowered or "memoryerror" in lowered or "hip out of memory" in lowered:
        return "oom", {"type": "oom", "message": "Benchmark command reported out-of-memory."}
    unsupported_tokens = (
        "unsupported",
        "not supported",
        "no kernel image is available",
        "hiperrornobinaryforgpu",
        "invalid device function",
        "not compiled for this gpu",
        "no binary for gpu",
    )
    if any(token in lowered for token in unsupported_tokens):
        return "unsupported", {"type": "unsupported_runtime", "message": "Benchmark command reported unsupported runtime or model path."}
    return "failed", {"type": "command_failed", "message": f"Benchmark command exited with code {returncode}."}


def classify_sq_execution_fallback(
    status: str,
    error: dict[str, str] | None,
    parse_name: str,
    report: dict[str, Any],
    allow_cli: bool,
) -> tuple[str, dict[str, str] | None]:
    if status != "ok" or parse_name != "ullm-model-loop-throughput":
        return status, error
    if not is_materialized_sq_fallback(report):
        return status, error
    if sq_execution_mode_allows_fallback(report, allow_cli):
        return status, error
    return "failed", {
        "type": "invalid_fallback",
        "message": (
            "materialized_f32_fallback rows must be explicitly marked fallback "
            "or excluded from comparable throughput."
        ),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        output.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--stdout-log", required=True, type=Path)
    parser.add_argument("--stderr-log", required=True, type=Path)
    parser.add_argument("--memory-log", required=True, type=Path)
    parser.add_argument("--engine-name", required=True)
    parser.add_argument("--engine-version")
    parser.add_argument("--engine-commit")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-source", default="huggingface")
    parser.add_argument("--model-revision")
    parser.add_argument("--model-format", required=True)
    parser.add_argument("--model-quantization", required=True)
    parser.add_argument("--sq-candidate")
    parser.add_argument("--candidate-artifact")
    parser.add_argument("--prefill-executor")
    parser.add_argument("--resolved-prefill-executor")
    parser.add_argument("--gpu-card", action="append", default=[])
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--pipeline-parallel", type=int, default=1)
    parser.add_argument("--data-parallel", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--generated-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrent-requests", type=int, default=1)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument(
        "--parse",
        choices=[
            "none",
            "vllm-throughput",
            "sglang-serving",
            "ullm-token-ids-generate",
            "ullm-package-batch-throughput",
            "ullm-component-prefill",
            "ullm-model-loop-throughput",
        ],
        default="none",
    )
    parser.add_argument(
        "--allow-materialized-fallback",
        action="store_true",
        help="Allow materialized_f32_fallback model-loop rows to stay valid.",
    )
    parser.add_argument("--result-json", type=Path)
    parser.add_argument("--prompt-guard-bundle-json", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--memory-sample-interval", type=float, default=1.0)
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--", dest="separator", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")

    args.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    args.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        if args.result_json.exists():
            args.result_json.unlink()
    prompt_guard_bundle = None
    if args.prompt_guard_bundle_json is not None:
        prompt_guard_bundle = load_prompt_guard_bundle(args.prompt_guard_bundle_json)

    monitor = RocmMemoryMonitor(args.memory_log, args.memory_sample_interval)
    monitor.start()
    started = time.monotonic()
    timed_out = False
    with args.stdout_log.open("w", encoding="utf-8") as stdout, args.stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(args.command, stdout=stdout, stderr=stderr, text=True)
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
    elapsed_seconds = time.monotonic() - started
    memory = monitor.stop()

    stdout_text = args.stdout_log.read_text(encoding="utf-8", errors="replace")
    stderr_text = args.stderr_log.read_text(encoding="utf-8", errors="replace")
    combined = stdout_text + "\n" + stderr_text

    if returncode == 0 and not timed_out:
        status = "ok"
        error = None
    else:
        status, error = classify_failure(returncode, timed_out, combined)

    ullm_report: dict[str, Any] = {}
    if args.parse == "ullm-token-ids-generate" and status == "ok":
        ullm_report = parse_ullm_token_ids_report(stdout_text, args.result_json)
        metrics = parse_ullm_token_ids_metrics(ullm_report, memory)
    elif args.parse == "ullm-package-batch-throughput" and status == "ok":
        ullm_report = parse_ullm_token_ids_report(stdout_text, args.result_json)
        metrics = parse_ullm_batch_throughput_metrics(ullm_report, memory)
    elif args.parse == "ullm-component-prefill" and status == "ok":
        ullm_report = parse_ullm_key_value_report(stdout_text, args.result_json)
        metrics = parse_ullm_component_prefill_metrics(ullm_report, memory)
    elif args.parse == "ullm-model-loop-throughput" and status == "ok":
        ullm_report = parse_ullm_key_value_report(stdout_text, args.result_json)
        metrics = parse_ullm_model_loop_metrics(ullm_report, memory)
    elif args.parse == "vllm-throughput" and status == "ok":
        metrics = parse_vllm_metrics(stdout_text, args.result_json, memory)
    elif args.parse == "sglang-serving" and status == "ok":
        metrics = parse_sglang_serving_metrics(args.result_json, memory)
    else:
        metrics = default_metrics(memory)

    allow_materialized_fallback = sq_execution_mode_allows_fallback(
        ullm_report, args.allow_materialized_fallback
    )
    saw_materialized_fallback = is_materialized_sq_fallback(ullm_report)

    env_names = [
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "VLLM_LOGGING_LEVEL",
        "VLLM_TARGET_DEVICE",
        "SGLANG_USE_AITER",
        "ATOM_USE_UNIFIED_ATTN",
        "ATOM_USE_TRITON_GEMM",
        "AITER_ROPE_NATIVE_BACKEND",
        "AITER_LOG_LEVEL",
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT",
        "ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT",
        "ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION",
        "PYTORCH_HIP_ALLOC_CONF",
        "HSA_OVERRIDE_GFX_VERSION",
        "ULLM_PREFILL_DEVICE_TOKEN_LOOP",
        "ULLM_SYNC_PREFILL_EACH_LAYER_FOR_TIMING",
        "ULLM_SYNC_LINEAR_ATTN_COMPONENTS_FOR_TIMING",
        "ULLM_SYNC_SELF_ATTN_COMPONENTS_FOR_TIMING",
        "ULLM_REQUIRE_HIP_AQ4_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
        "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL",
        "ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL",
        "ULLM_REQUIRE_HIP_ADD_KERNEL",
        "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
        "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL",
        "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
        "ULLM_REQUIRE_HIP_ROPE_KERNEL",
        "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
        "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
        "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
        "ULLM_REQUIRE_HIP_TOP1_KERNEL",
    ]
    prefix = env_prefix(env_names)
    command = command_string(args.command)
    if prefix:
        command = f"{prefix} {command}"

    row = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "case_id": args.case_id,
        "status": status,
        "engine": {
            "name": args.engine_name,
            "version": args.engine_version,
            "commit": args.engine_commit,
        },
        "model": {
            "name": args.model_name,
            "source": args.model_source,
            "revision": args.model_revision,
            "format": args.model_format,
            "quantization": canonical_or_original(args.model_quantization),
        },
        "hardware": {
            "host": socket.gethostname(),
            "gpu_count": len(args.gpu_card),
            "gpus": selected_gpus(args.gpu_card),
            "cpu": None,
            "driver": rocm_driver(),
            "runtime": rocm_runtime(),
        },
        "parallelism": {
            "tensor_parallel": args.tensor_parallel,
            "pipeline_parallel": args.pipeline_parallel,
            "data_parallel": args.data_parallel,
        },
        "workload": {
            "context_length": args.context_length,
            "prompt_tokens": args.prompt_tokens,
            "generated_tokens": args.generated_tokens,
            "batch_size": args.batch_size,
            "concurrent_requests": args.concurrent_requests,
            "kv_cache_dtype": args.kv_cache_dtype,
            "prefill_executor": args.prefill_executor,
            "resolved_prefill_executor": args.resolved_prefill_executor,
        },
        "metrics": metrics,
        "memory": memory,
        "artifacts": {
            "command": command,
            "stdout_log": str(args.stdout_log),
            "stderr_log": str(args.stderr_log),
            "memory_log": str(args.memory_log),
            "result_json": str(args.result_json) if args.result_json else None,
            "elapsed_seconds": elapsed_seconds,
        },
        "error": error,
        "notes": args.note,
        "harness": classify_benchmark_harness(args.parse),
    }
    if args.sq_candidate or args.candidate_artifact:
        candidate_id = canonical_or_original(args.sq_candidate) if args.sq_candidate else None
        row["candidate"] = {
            "id": candidate_id,
            "artifact": args.candidate_artifact,
        }
        if args.sq_candidate and candidate_id != args.sq_candidate:
            row["candidate"]["legacy_id"] = args.sq_candidate
    ullm_correctness = parse_ullm_token_ids_correctness(ullm_report)
    if args.parse == "ullm-package-batch-throughput":
        enrich_ullm_batch_workload(row, ullm_report)
        batching = ullm_report.get("batching")
        if isinstance(batching, dict):
            row["batching"] = batching
        ullm_correctness = parse_ullm_batch_throughput_correctness(ullm_report)
    elif args.parse == "ullm-component-prefill":
        enrich_ullm_component_prefill_row(row, ullm_report)
        verified = ullm_report.get("verified")
        row["correctness"] = {
            "reference": "sampled",
            "reference_artifact": None,
            "logits_relative_mse": None,
            "logits_max_abs_diff": None,
            "top_k": None,
            "top_k_agreement": None,
            "generated_prefix_matches_reference": verified if isinstance(verified, bool) else None,
            "nan_count": None,
            "inf_count": None,
            "verified_all": verified if isinstance(verified, bool) else None,
            "sample_count": parse_int(ullm_report.get("sample_count")),
            "sampled_max_abs_diff": first_non_null(
                parse_float(ullm_report.get("sampled_max_abs_diff")),
                parse_float(ullm_report.get("max_abs_diff")),
            ),
        }
        ullm_correctness = None
    elif args.parse == "ullm-model-loop-throughput":
        enrich_ullm_model_loop_row(row, ullm_report)
        verified = ullm_report.get("verified")
        row["correctness"] = {
            "reference": "sampled",
            "reference_artifact": None,
            "logits_relative_mse": None,
            "logits_max_abs_diff": None,
            "top_k": None,
            "top_k_agreement": None,
            "generated_prefix_matches_reference": verified if isinstance(verified, bool) else None,
            "nan_count": None,
            "inf_count": None,
            "verified_all": verified if isinstance(verified, bool) else None,
            "sample_count": None,
            "sampled_max_abs_diff": first_non_null(
                parse_float(ullm_report.get("layer_max_abs_diff")),
                parse_float(ullm_report.get("block_max_abs_diff")),
            ),
        }
        ullm_correctness = None
    if args.parse == "ullm-model-loop-throughput" and saw_materialized_fallback:
        workload = row.get("workload")
        if isinstance(workload, dict):
            workload["fallback_allowed"] = allow_materialized_fallback
            workload["diagnostic"] = (
                parse_bool(ullm_report.get("diagnostic")) is True
                or is_selected_layer_sq8_diagnostic_command(ullm_report.get("command"))
            )
        status, error = classify_sq_execution_fallback(
            status, error, args.parse, ullm_report, args.allow_materialized_fallback
        )
        row["status"] = status
        row["error"] = error
    if prompt_guard_bundle is not None:
        assert args.prompt_guard_bundle_json is not None
        attach_prompt_guard_bundle_fields(
            row, args.prompt_guard_bundle_json, prompt_guard_bundle
        )
    if ullm_correctness is not None:
        row["correctness"] = ullm_correctness
        if args.parse == "ullm-package-batch-throughput":
            enrich_ullm_batch_memory(row, ullm_report)
        else:
            raw_memory = ullm_report.get("memory")
            if isinstance(raw_memory, dict):
                row["memory"].update(
                    {
                        "kv_cache_bytes": raw_memory.get("kv_cache_bytes"),
                        "kv_cache_allocated_blocks": raw_memory.get("kv_cache_allocated_blocks"),
                        "kv_cache_free_blocks": raw_memory.get("kv_cache_free_blocks"),
                        "kv_cache_block_size": raw_memory.get("kv_cache_block_size"),
                    }
                )
    append_jsonl(args.output_jsonl, row)
    print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
