#!/usr/bin/env python3
"""Run the Stage 0 FORCE_BASELINE inventory for Qwen3.8-27B NVFP4.

Stage 0 is deliberately an inventory, rather than a pass/fail gate for the
baseline kernels.  It runs the default control and each of the eleven
FORCE_BASELINE selectors against both the long, fixed teacher-forcing prefix
and a short prefix.  The long prefix is the existing
``wmma-check/prefix-one.json`` (8,284 prompt tokens).  The short prefix is the
existing 64-token pilot fixture.  The short run is useful when the long run
fails during prefill: it can still show whether the selector reaches decode.
Changing the prefill chunk size would change the experiment, so the runner
keeps the benchmark's fixed chunk setting and changes only the input fixture.

Every run gets its own immutable attempt directory and retains stdout,
stderr, the Stage 0 report, and rocprof's kernel trace.  A configuration
digest binds the target, UUID, binary, selector, prefix, and command.  An
existing run with a different digest is rejected; a completed run is reused
only when its digest is identical.  This makes an interrupted inventory
resumable without silently mixing evidence from different binaries or flags.

The module uses only the Python standard library.  It does not import sLLM or
run GPU work when imported; execution happens only through ``main``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO / ".local-artifacts/force-baseline-oracle"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "stage0"
DEFAULT_LONG_PREFIX = REPO / ".local-artifacts/mtp-bench/wmma-check/prefix-one.json"
# This is the existing Stage 0 pilot generated prefix.  It has a 64-token
# prompt and therefore reaches the M=1 decode path while keeping the same
# stage0-fixed-prefix-teacher-forcing contract.
DEFAULT_SHORT_PREFIX = ARTIFACT_ROOT / "../phase85-a16-mtp/stage0/pilot-gfx1030/generate/prefixes.json"
DEFAULT_MODEL = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4")
DEFAULT_ROCPROF = Path("/usr/bin/rocprofv3")
DEFAULT_UNIT = "sllm-qwen38-r9700.service"
DEFAULT_SERVER_DIR = REPO / ".local-artifacts/qwen38-r9700-server"
DEFAULT_UNIT_FILE = Path("/home/homelab1/.config/systemd/user") / DEFAULT_UNIT

FORCE_FLAGS = (
    "SLLM_MX_WA_PREFILL_FORCE_BASELINE",
    "SLLM_GDN_FORCE_BASELINE",
    "SLLM_FP8_OUTER_PREFILL_FORCE_BASELINE",
    "SLLM_NVFP4_W4A4_FORCE_BASELINE",
    "SLLM_MATMUL_FORCE_BASELINE",
    "SLLM_CAUSAL_ATTENTION_FORCE_BASELINE",
    "SLLM_NVFP4_FORCE_BASELINE",
    "SLLM_FP8_OUTER_DECODE_FORCE_BASELINE",
    "SLLM_MX_WA_M1_FORCE_BASELINE",
    "SLLM_FP8_QUANT_FORCE_BASELINE",
    "SLLM_FP8_OUTER_FORCE_BASELINE",
)

DEFAULT_TARGETS: dict[str, dict[str, Any]] = {
    "gfx1030": {
        "uuid": "GPU-76a08c022586fed6",
        "binary": ARTIFACT_ROOT / "initial-bin/bench-gfx1030",
        "lease": False,
    },
    "gfx1201": {
        "uuid": "GPU-a8e9ddefa2d60f55",
        "binary": ARTIFACT_ROOT / "initial-bin/bench-gfx1201",
        "lease": True,
    },
}

ROOT_SCHEMA = "force-baseline-oracle-stage0-v1"
RUN_SCHEMA = "force-baseline-oracle-stage0-run-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    """Write a small manifest atomically so interruption cannot truncate it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def path_identity(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        result["sha256"] = sha256_file(path)
        result["size_bytes"] = path.stat().st_size
    except OSError as error:
        result["sha256"] = None
        result["size_bytes"] = None
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def flatten_values(values: Iterable[Any] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        nested = value if isinstance(value, (list, tuple)) else (value,)
        for item in nested:
            result.extend(part.strip() for part in str(item).split(",") if part.strip())
    return result


def parse_target_bindings(values: Iterable[str] | None, name: str) -> dict[str, str]:
    """Parse repeated ``target=VALUE`` bindings, accepting ``target:VALUE``."""

    result: dict[str, str] = {}
    for value in values or ():
        if "=" in value:
            target, bound = value.split("=", 1)
        elif ":" in value and not value.startswith("/"):
            target, bound = value.split(":", 1)
        else:
            raise ValueError(f"--{name} requires TARGET=VALUE (got {value!r})")
        target, bound = target.strip(), bound.strip()
        if not target or not bound:
            raise ValueError(f"--{name} requires non-empty TARGET and VALUE")
        if target in result and result[target] != bound:
            raise ValueError(f"duplicate --{name} binding for {target}")
        result[target] = bound
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        "--target",
        dest="targets",
        action="append",
        nargs="+",
        help="target(s), repeated or comma-separated (default: gfx1030,gfx1201)",
    )
    parser.add_argument(
        "--flag",
        "--flags",
        dest="flags",
        action="append",
        nargs="+",
        help="FORCE_BASELINE selector(s); repeat or comma-separate (default: all eleven)",
    )
    parser.add_argument(
        "--no-default-control",
        action="store_true",
        help="omit the no-FORCE_BASELINE default control",
    )
    parser.add_argument(
        "--binary",
        "--binaries",
        action="append",
        help="target binary binding TARGET=PATH (default: initial frozen binaries)",
    )
    parser.add_argument(
        "--uuid",
        "--device-uuid",
        dest="uuids",
        action="append",
        help="exact HIP UUID binding TARGET=GPU-UUID",
    )
    parser.add_argument(
        "--prefix",
        "--prefix-file",
        "--long-prefix",
        dest="long_prefix",
        type=Path,
        default=DEFAULT_LONG_PREFIX,
        help="existing long Stage 0 prefix fixture",
    )
    parser.add_argument(
        "--short-prefix",
        type=Path,
        default=DEFAULT_SHORT_PREFIX,
        help="short prefix fixture used to probe decode after long-prefill failure",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        "--output-root",
        dest="output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="artifact root for resumable Stage 0 evidence",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rocprof", type=Path, default=DEFAULT_ROCPROF)
    parser.add_argument("--rocm-library-path", default="/opt/rocm/lib")
    parser.add_argument("--unit", default=DEFAULT_UNIT)
    parser.add_argument("--unit-file", type=Path, default=DEFAULT_UNIT_FILE)
    parser.add_argument("--server-dir", type=Path, default=DEFAULT_SERVER_DIR)
    parser.add_argument(
        "--lease-target",
        action="append",
        default=None,
        help="target(s) whose resident service is leased (default: gfx1201)",
    )
    parser.add_argument(
        "--no-service-lease",
        action="store_true",
        help="do not stop or start a resident service; useful for an already isolated host",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="optional per-run timeout in seconds (default: none; do not kill a progressing GPU run)",
    )
    parser.add_argument(
        "--prefix-role",
        dest="prefix_roles",
        action="append",
        choices=("long", "short"),
        help="run only this prefix role; default runs long, then short only for failed long runs (and default control)",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="create a new attempt for a matching failed run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the frozen run matrix without invoking rocprof or systemd",
    )
    parser.add_argument(
        "--fail-on-run-failure",
        action="store_true",
        help="return non-zero when an observed run exits non-zero",
    )
    return parser.parse_args(argv)


def selected_targets(args: argparse.Namespace) -> list[str]:
    values = flatten_values(args.targets)
    return values or list(DEFAULT_TARGETS)


def selected_flags(args: argparse.Namespace) -> list[str | None]:
    requested = flatten_values(args.flags)
    if not requested:
        requested = list(FORCE_FLAGS)
    normalized: list[str] = []
    for value in requested:
        if value.lower() in {"default", "control", "none", "off"}:
            continue
        if value not in FORCE_FLAGS:
            raise ValueError(f"unknown FORCE_BASELINE selector {value!r}; expected one of {FORCE_FLAGS}")
        if value not in normalized:
            normalized.append(value)
    controls: list[str | None] = [] if args.no_default_control else [None]
    return controls + normalized


def target_config(args: argparse.Namespace, targets: list[str]) -> dict[str, dict[str, Any]]:
    binaries = parse_target_bindings(args.binary, "binary")
    uuids = parse_target_bindings(args.uuids, "uuid")
    lease_targets = set(flatten_values(args.lease_target)) if args.lease_target else {"gfx1201"}
    result: dict[str, dict[str, Any]] = {}
    for target in targets:
        default = DEFAULT_TARGETS.get(target, {})
        binary_value = binaries.get(target, default.get("binary"))
        if binary_value is None:
            raise ValueError(f"no binary configured for target {target!r}; pass --binary {target}=PATH")
        binary = Path(binary_value)
        uuid = uuids.get(target, default.get("uuid"))
        if not uuid:
            raise ValueError(f"no UUID configured for target {target!r}; pass --uuid {target}=GPU-...")
        result[target] = {
            "target": target,
            "uuid": str(uuid),
            "binary": path_identity(binary),
            "lease_service": (not args.no_service_lease and target in lease_targets),
        }
    return result


def prefix_identity(path: Path, role: str) -> dict[str, Any]:
    identity = path_identity(path)
    identity["role"] = role
    try:
        document = json.loads(path.read_text())
        entries = document.get("entries", []) if isinstance(document, dict) else []
        identity["schema"] = document.get("schema") if isinstance(document, dict) else None
        identity["entries"] = [
            {
                "case_id": entry.get("case_id"),
                "prompt_token_count": len(entry.get("prompt_tokens", [])),
                "output_prefix_token_count": len(entry.get("output_prefix_tokens", [])),
            }
            for entry in entries
            if isinstance(entry, dict)
        ]
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        identity["schema"] = None
        identity["entries"] = []
    return identity


def health(endpoint: str) -> int:
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
    try:
        connection.request("GET", endpoint)
        response = connection.getresponse()
        response.read()
        return response.status
    except OSError:
        return 0
    finally:
        connection.close()


def service_active(unit: str) -> bool:
    try:
        return subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", unit],
            check=False,
        ).returncode == 0
    except OSError:
        return False


def service_hashes(unit_file: Path, server_dir: Path, binary: Path) -> dict[str, Any]:
    return {
        "unit": path_identity(unit_file),
        "run_script": path_identity(server_dir / "run.sh"),
        "server_binary": path_identity(server_dir / "sllm-server"),
        "measurement_binary": path_identity(binary),
    }


def performance_level_path(uuid: str) -> Path | None:
    """Resolve the DPM level node by exact UUID, without changing it."""

    matches: list[Path] = []
    wanted = uuid.removeprefix("GPU-").lower()
    for device in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            if (device / "unique_id").read_text().strip().lower() == wanted:
                matches.append(device)
        except OSError:
            continue
    if len(matches) != 1:
        return None
    level = matches[0] / "power_dpm_force_performance_level"
    return level if level.exists() else None


def level_snapshot(uuid: str) -> dict[str, Any]:
    path = performance_level_path(uuid)
    result: dict[str, Any] = {"uuid": uuid, "path": str(path) if path else None, "value": None}
    if path is None:
        result["status"] = "unavailable"
        result["error"] = "exact UUID performance-level sysfs node was not found"
        return result
    try:
        result["value"] = path.read_text().strip()
        result["status"] = "observed"
    except OSError as error:
        result["status"] = "unavailable"
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def wait_service(unit: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if service_active(unit) and health("/healthz") == 200 and health("/readyz") == 200:
            return {"active": True, "health": 200, "ready": 200}
        time.sleep(1)
    return {"active": service_active(unit), "health": health("/healthz"), "ready": health("/readyz")}


def trace_summary(trace_dir: Path) -> dict[str, Any]:
    files = sorted(path for path in trace_dir.rglob("*kernel_trace*.csv") if path.is_file())
    counts: dict[str, int] = {}
    errors: list[str] = []
    rows = 0
    for path in files:
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    rows += 1
                    lowered = {str(key).lower(): value for key, value in row.items() if key is not None}
                    name = (
                        lowered.get("kernel_name")
                        or lowered.get("kernelname")
                        or lowered.get("name")
                        or "<unknown>"
                    ).strip()
                    counts[name] = counts.get(name, 0) + 1
        except (OSError, csv.Error) as error:
            errors.append(f"{path}: {type(error).__name__}: {error}")
    return {
        "trace_directory": str(trace_dir),
        "trace_files": [str(path) for path in files],
        "trace_files_count": len(files),
        "total_dispatches_traced": rows,
        "distinct_symbols": len(counts),
        "kernel_symbol_counts": dict(sorted(counts.items())),
        # Keep this alias because matrix consumers historically call this map
        # either kernel symbols or dispatch symbols.
        "dispatch_symbol_counts": dict(sorted(counts.items())),
        "symbol_counts": dict(sorted(counts.items())),
        "parse_errors": errors,
    }


def json_documents(paths: Iterable[Path]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for path in paths:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            documents.append(value)
    return documents


def stage0_observation(documents: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Extract structured evidence while retaining missing fields explicitly."""

    entries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for document in documents:
        fixed = document.get("fixed_prefix")
        if isinstance(fixed, dict) and isinstance(fixed.get("entries"), list):
            reports.append(document)
            for entry in fixed["entries"]:
                if not isinstance(entry, dict):
                    continue
                entries.append(
                    {
                        "case_id": entry.get("case_id"),
                        "target_hidden_sha256": entry.get("target_hidden_sha256"),
                        "logits_sha256": entry.get("logits_sha256"),
                        "selected_backend": entry.get("selected_backend"),
                        "target": entry.get("target"),
                        "target_submission_count": entry.get("target_submission_count"),
                        "target_kernel_dispatch_count": entry.get("target_kernel_dispatch_count"),
                        "draft_submission_count": entry.get("draft_submission_count"),
                        "draft_kernel_dispatch_count": entry.get("draft_kernel_dispatch_count"),
                        "fallback_used": entry.get("fallback_used"),
                        "all_dispatches_hip": entry.get("all_dispatches_hip"),
                        "nonfinite_value_count": entry.get("nonfinite_value_count"),
                    }
                )
    cleanup: list[dict[str, Any]] = []
    for document in reports:
        value = document.get("cleanup")
        if isinstance(value, dict):
            cleanup.append(value)
    return {
        "report_count": len(reports),
        "entries": entries,
        "target_hidden_sha256": {
            str(entry.get("case_id")): entry.get("target_hidden_sha256")
            for entry in entries
            if entry.get("case_id") is not None
        },
        "logits_sha256": {
            str(entry.get("case_id")): entry.get("logits_sha256")
            for entry in entries
            if entry.get("case_id") is not None
        },
        "cleanup": cleanup,
        "cleanup_zero": [value.get("zero") for value in cleanup],
        "retryable_cleanup": [value.get("retryable_cleanup") for value in cleanup],
        "durable_quarantine": [value.get("durable_quarantine") for value in cleanup],
        "fallback_used": [entry.get("fallback_used") for entry in entries],
        "all_dispatches_hip": [entry.get("all_dispatches_hip") for entry in entries],
    }


def bounded_error(stdout: bytes, stderr: bytes, documents: Iterable[dict[str, Any]]) -> str | None:
    for document in documents:
        value = document.get("error")
        if isinstance(value, str) and value:
            return value[-4000:]
    text = stderr.decode(errors="replace").strip() or stdout.decode(errors="replace").strip()
    return text[-4000:] if text else None


def scrub_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SLLM_")
        and key not in {"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"}
    }


def run_environment(
    *,
    target: str,
    uuid: str,
    model: Path,
    prefix: Path,
    output_dir: Path,
    force_flag: str | None,
    rocm_library_path: str,
) -> dict[str, str]:
    environment = scrub_environment()
    environment.update(
        {
            "ROCR_VISIBLE_DEVICES": uuid,
            "LD_LIBRARY_PATH": rocm_library_path,
            "SLLM_PHASE78_MODEL_PATH": str(model),
            "SLLM_PHASE78_TARGET": target,
            "SLLM_PHASE78_DEVICE": "0",
            "SLLM_PHASE78_WARMUPS": "0",
            "SLLM_PHASE78_MEASURED": "1",
            "SLLM_PHASE78_CHUNK_CAPACITY": "2048",
            "SLLM_PHASE83_MODE": "coding8192",
            "SLLM_PHASE83_KV": "mxfp8",
            "SLLM_PHASE83_SAMPLING": "gpu-fixed",
            "SLLM_PHASE83_REPLAY": "0",
            "SLLM_PHASE83_MTP": "on",
            "SLLM_PHASE83_MTP_WIDTH": "2",
            "SLLM_PHASE83_STATE_CAPACITY": "10240",
            "SLLM_PHASE85_A16_STAGE0": "1",
            "SLLM_PHASE85_A16_PREFIX_FILE": str(prefix),
            "SLLM_PHASE85_A16_OUTPUT_DIR": str(output_dir),
            "SLLM_PHASE85_A16_SERIES": "bf16",
        }
    )
    if force_flag is not None:
        environment[force_flag] = "1"
    return environment


def make_root_config(args: argparse.Namespace, targets: list[str], flags: list[str | None]) -> dict[str, Any]:
    configured_targets = target_config(args, targets)
    prefixes = {
        "long": prefix_identity(args.long_prefix.resolve(), "long-prefill-and-decode"),
        "short": prefix_identity(args.short_prefix.resolve(), "short-decode-reach-probe"),
    }
    controls = ["default" if flag is None else flag for flag in flags]
    return {
        "schema_version": ROOT_SCHEMA,
        "targets": configured_targets,
        "controls": controls,
        "prefixes": prefixes,
        "model": path_identity(args.model.resolve()),
        "rocprof": path_identity(args.rocprof.resolve()),
        "rocprof_options": ["--kernel-trace", "--output-format", "csv"],
        "rocm_library_path": args.rocm_library_path,
        "unit": args.unit,
        "unit_file": str(args.unit_file),
        "server_dir": str(args.server_dir),
        "timeout_seconds": args.timeout,
        "lease_service": not args.no_service_lease,
        "experiment": "stage0: each control x long/short prefix x exact target",
    }


def config_digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode())


def control_name(force_flag: str | None) -> str:
    return "default" if force_flag is None else force_flag.removeprefix("SLLM_").lower()


def run_descriptor(target: str, force_flag: str | None, role: str) -> dict[str, Any]:
    return {
        "run_id": f"{target}/{control_name(force_flag)}/{role}",
        "target": target,
        "control": "default" if force_flag is None else force_flag,
        "prefix_role": role,
    }


def load_matching_manifest(path: Path, expected_digest: str, kind: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read existing {kind} manifest {path}: {error}") from error
    if not isinstance(document, dict) or document.get("config_sha256") != expected_digest:
        observed = document.get("config_sha256") if isinstance(document, dict) else None
        raise RuntimeError(
            f"existing {kind} manifest has mismatched configuration: {path} "
            f"expected={expected_digest} observed={observed}"
        )
    return document


def next_attempt(run_dir: Path) -> tuple[int, Path]:
    attempts = []
    for path in run_dir.glob("attempt-[0-9][0-9][0-9]"):
        try:
            attempts.append(int(path.name.rsplit("-", 1)[1]))
        except ValueError:
            continue
    number = max(attempts, default=0) + 1
    return number, run_dir / f"attempt-{number:03d}"


def pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def run_one(
    *,
    args: argparse.Namespace,
    root: Path,
    target_spec: dict[str, Any],
    force_flag: str | None,
    prefix_role: str,
    prefix: Path,
    root_digest: str,
) -> dict[str, Any]:
    target = str(target_spec["target"])
    control = "default" if force_flag is None else force_flag
    control_slug = control_name(force_flag)
    run_id = f"{target}/{control_slug}/{prefix_role}"
    run_dir = root / target / control_slug / prefix_role
    run_manifest_path = run_dir / "run.json"
    binary_path = Path(target_spec["binary"]["path"])
    run_config = {
        "run_id": run_id,
        "target": target,
        "device_uuid": target_spec["uuid"],
        "force_flag": force_flag,
        "flags": [] if force_flag is None else [force_flag],
        "control": control,
        "prefix_role": prefix_role,
        "prefix": path_identity(prefix),
        "binary": path_identity(binary_path),
        "model": path_identity(args.model.resolve()),
        "rocprof": path_identity(args.rocprof.resolve()),
        "rocprof_options": ["--kernel-trace", "--output-format", "csv"],
        "rocm_library_path": args.rocm_library_path,
        "environment_selector": "SLLM_* scrubbed, then fixed Stage0 environment plus one selector",
        "root_config_sha256": root_digest,
    }
    digest = config_digest(run_config)
    existing = load_matching_manifest(run_manifest_path, digest, "run")
    if existing and existing.get("state") in {"complete", "failed"} and not (
        args.rerun_failed and existing.get("state") == "failed"
    ):
        existing["resumed"] = True
        return existing
    if existing and existing.get("state") == "running" and pid_alive(existing.get("pid")):
        # A second invocation may be polling an active GPU process.  Leave its
        # files and PID untouched; creating another attempt would duplicate a
        # still-running launch.
        existing["resumed"] = True
        existing["still_running"] = True
        return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    if existing and existing.get("state") == "running":
        attempt, attempt_dir = next_attempt(run_dir)
    else:
        # A directory without its manifest could contain evidence from an
        # unrelated run; reject it instead of overwriting it.
        if any(run_dir.iterdir()):
            raise RuntimeError(f"existing run directory lacks a matching manifest: {run_dir}")
        attempt, attempt_dir = 1, run_dir / "attempt-001"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = attempt_dir / "trace"
    stage0_dir = attempt_dir / "stage0"
    trace_dir.mkdir(parents=True, exist_ok=True)
    stage0_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "state": "running",
        "run_id": run_id,
        "target": target,
        "device_uuid": target_spec["uuid"],
        "binary_sha256": target_spec["binary"].get("sha256"),
        "flags": [] if force_flag is None else [force_flag],
        "prefix_role": prefix_role,
        "config": run_config,
        "config_sha256": digest,
        "attempt": attempt,
        "attempt_directory": str(attempt_dir),
        "started": time.time(),
    }
    write_json(run_manifest_path, manifest)

    command = [
        str(args.rocprof),
        "--kernel-trace",
        "--output-format",
        "csv",
        "--output-directory",
        str(trace_dir),
        "--",
        str(binary_path),
    ]
    environment = run_environment(
        target=target,
        uuid=str(target_spec["uuid"]),
        model=args.model.resolve(),
        prefix=prefix,
        output_dir=stage0_dir,
        force_flag=force_flag,
        rocm_library_path=args.rocm_library_path,
    )
    manifest["command"] = command
    manifest["environment"] = {
        key: value
        for key, value in environment.items()
        if key.startswith("SLLM_") or key in {"ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH"}
    }
    started = time.monotonic()
    stdout = b""
    stderr = b""
    timed_out = False
    stdout_path = attempt_dir / "run.stdout.json"
    stderr_path = attempt_dir / "run.stderr.log"
    try:
        # Keep the pipes as files while the process runs.  This makes a long
        # full-model launch observable and leaves partial output available if
        # the host or terminal is interrupted.
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                env=environment,
                cwd=str(REPO),
            )
            manifest["pid"] = process.pid
            write_json(run_manifest_path, manifest)
            try:
                exit_code: int | None = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                exit_code = process.wait()
    except subprocess.TimeoutExpired as error:
        exit_code = 124
        timed_out = True
        del error
    except OSError as error:
        exit_code = 127
        stderr = f"{type(error).__name__}: {error}".encode()
    try:
        stdout = stdout_path.read_bytes()
        stderr = stderr_path.read_bytes() + stderr
    except OSError as error:
        stderr = stderr + f"\nread run output: {type(error).__name__}: {error}".encode()
    # The benchmark writes stage0/report.json on a successful Rust report.  A
    # failure is emitted on stderr, so parse both channels and that report.
    documents = json_documents(
        [attempt_dir / "run.stdout.json", attempt_dir / "run.stderr.log", stage0_dir / "report.json"]
    )
    traces = trace_summary(trace_dir)
    observation = stage0_observation(documents)
    benchmark = next((doc for doc in documents if doc.get("schema_version") == "phase85-a16-mtp-stage0-v1"), None)
    run_ok = exit_code == 0 and benchmark is not None and benchmark.get("state") == "PASS"
    manifest.update(
        {
            "state": "complete" if run_ok else "failed",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "wall_seconds": round(time.monotonic() - started, 3),
            "error": bounded_error(stdout, stderr, documents),
            "benchmark_report": str(stage0_dir / "report.json"),
            "benchmark_state": benchmark.get("state") if benchmark else None,
            "result_ok": run_ok,
            "result_failure_reason": None if run_ok else "missing PASS Stage0 report",
            "observed_target": benchmark.get("target") if benchmark else None,
            "target_match": benchmark.get("target") == target if benchmark else False,
            "stage0": observation,
            "stage0_hidden_sha256": observation["target_hidden_sha256"],
            "stage0_logits_sha256": observation["logits_sha256"],
            "cleanup": {
                "reports": observation["cleanup"],
                "zero": observation["cleanup_zero"],
                "retryable_cleanup": observation["retryable_cleanup"],
                "durable_quarantine": observation["durable_quarantine"],
            },
            "fallback": {
                "used": observation["fallback_used"],
                "all_dispatches_hip": observation["all_dispatches_hip"],
            },
            "dispatch": traces,
            "ended": time.time(),
        }
    )
    write_json(run_manifest_path, manifest)
    print(
        f"[{run_id}] exit={exit_code} dispatches={traces['total_dispatches_traced']} "
        f"entries={observation['report_count']} trace_files={traces['trace_files_count']}",
        flush=True,
    )
    return manifest


def service_snapshot(
    args: argparse.Namespace,
    target_spec: dict[str, Any],
) -> dict[str, Any]:
    binary = Path(target_spec["binary"]["path"])
    return {
        "target": target_spec["target"],
        "device_uuid": target_spec["uuid"],
        "was_active": service_active(args.unit),
        "health_before": health("/healthz"),
        "ready_before": health("/readyz"),
        "hashes_before": service_hashes(args.unit_file, args.server_dir, binary),
        "performance_level_before": level_snapshot(str(target_spec["uuid"])),
        "stopped_by_runner": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        targets = selected_targets(args)
        flags = selected_flags(args)
        configured_targets = target_config(args, targets)
        for target in targets:
            if target not in configured_targets:
                raise ValueError(f"target configuration missing for {target}")
        if args.timeout is not None and args.timeout <= 0:
            raise ValueError("--timeout must be positive")
        root_config = make_root_config(args, targets, flags)
        root_digest = config_digest(root_config)
        explicit_roles = list(dict.fromkeys(args.prefix_roles or []))
        # Long prefill is the primary observation.  A short run is added only
        # when the long run fails, because that is the decode-reach probe the
        # plan calls for.  The default control keeps both roles as a useful
        # control pair even when its long run succeeds.
        if not explicit_roles:
            initial_runs = [
                run_descriptor(target, flag, role)
                for target in targets
                for flag in flags
                for role in (["long", "short"] if flag is None else ["long"])
            ]
        else:
            initial_runs = [
                run_descriptor(target, flag, role)
                for target in targets
                for flag in flags
                for role in explicit_roles
            ]
        if args.dry_run:
            print(json.dumps({"config": root_config, "config_sha256": root_digest, "runs": initial_runs}, indent=2, ensure_ascii=False))
            return 0

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        root_manifest_path = output / "execution.json"
        existing_root = load_matching_manifest(root_manifest_path, root_digest, "root")
        if existing_root is None and any(output.iterdir()):
            raise RuntimeError(f"output directory contains data without a matching execution manifest: {output}")
        root_manifest: dict[str, Any] = existing_root or {
            "schema_version": ROOT_SCHEMA,
            "state": "running",
            "config": root_config,
            "config_sha256": root_digest,
            "expected_runs": initial_runs,
            "runs": {},
            "started": time.time(),
            "service": {},
        }
        write_json(root_manifest_path, root_manifest)

        prefixes = {"long": args.long_prefix.resolve(), "short": args.short_prefix.resolve()}
        for path in prefixes.values():
            if not path.is_file():
                raise RuntimeError(f"prefix fixture does not exist: {path}")
        service_target_specs = [spec for spec in configured_targets.values() if spec["lease_service"]]
        service_state: dict[str, Any] | None = None
        # The resident service is leased once for all runs on the target that
        # owns it.  It is stopped only after observing it active and is started
        # in finally only when this invocation stopped it.
        all_results: list[dict[str, Any]] = []
        try:
            if service_target_specs:
                if len(service_target_specs) > 1:
                    raise RuntimeError("at most one service lease target is supported per invocation")
                service_state = service_snapshot(args, service_target_specs[0])
                root_manifest["service"] = service_state
                if service_state["was_active"]:
                    if service_state["ready_before"] != 200:
                        raise RuntimeError("resident service is active but /readyz is not 200; refusing lease")
                    clients = subprocess.run(
                        ["ss", "-Htn", "state", "established", "(", "sport", "=", ":8000", ")"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if clients.stdout.strip():
                        raise RuntimeError("active connections on port 8000; refusing service lease")
                    service_state["stopped_by_runner"] = True
                    subprocess.run(["systemctl", "--user", "stop", args.unit], check=True)
                    write_json(root_manifest_path, root_manifest)
                    for _ in range(60):
                        if not service_active(args.unit):
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError("resident service did not stop")
                write_json(root_manifest_path, root_manifest)

            for target in targets:
                target_spec = configured_targets[target]
                for flag in flags:
                    roles = explicit_roles or ["long"]
                    for role in roles:
                        result = run_one(
                            args=args,
                            root=output,
                            target_spec=target_spec,
                            force_flag=flag,
                            prefix_role=role,
                            prefix=prefixes[role],
                            root_digest=root_digest,
                        )
                        all_results.append(result)
                        root_manifest["runs"][result["run_id"]] = {
                            "state": result.get("state"),
                            "exit_code": result.get("exit_code"),
                            "target": result.get("target"),
                            "device_uuid": result.get("device_uuid"),
                            "flags": result.get("flags"),
                            "binary_sha256": result.get("binary_sha256"),
                            "run_manifest": str(output / result["run_id"] / "run.json"),
                            "attempt": result.get("attempt"),
                            "config_sha256": result.get("config_sha256"),
                        }
                        write_json(root_manifest_path, root_manifest)
                        # A non-default long failure is the only reason to add
                        # its short decode-reach probe to the inventory.  Do
                        # not launch the probe for an active long process.
                        if (
                            not explicit_roles
                            and role == "long"
                            and result.get("state") != "running"
                            and (flag is None or result.get("state") != "complete")
                        ):
                            short_descriptor = run_descriptor(target, flag, "short")
                            if short_descriptor not in root_manifest["expected_runs"]:
                                root_manifest["expected_runs"].append(short_descriptor)
                            short_result = run_one(
                                args=args,
                                root=output,
                                target_spec=target_spec,
                                force_flag=flag,
                                prefix_role="short",
                                prefix=prefixes["short"],
                                root_digest=root_digest,
                            )
                            all_results.append(short_result)
                            root_manifest["runs"][short_result["run_id"]] = {
                                "state": short_result.get("state"),
                                "exit_code": short_result.get("exit_code"),
                                "target": short_result.get("target"),
                                "device_uuid": short_result.get("device_uuid"),
                                "flags": short_result.get("flags"),
                                "binary_sha256": short_result.get("binary_sha256"),
                                "run_manifest": str(output / short_result["run_id"] / "run.json"),
                                "attempt": short_result.get("attempt"),
                                "config_sha256": short_result.get("config_sha256"),
                            }
                            write_json(root_manifest_path, root_manifest)
        finally:
            if service_state and service_state.get("stopped_by_runner"):
                subprocess.run(["systemctl", "--user", "start", args.unit], check=False)
                restored = wait_service(args.unit)
                binary = Path(service_target_specs[0]["binary"]["path"])
                service_state["restore"] = restored
                service_state["hashes_after"] = service_hashes(args.unit_file, args.server_dir, binary)
                service_state["health_after"] = health("/healthz")
                service_state["ready_after"] = health("/readyz")
                service_state["performance_level_after"] = level_snapshot(str(service_state["device_uuid"]))
                before_level = service_state["performance_level_before"]
                after_level = service_state["performance_level_after"]
                service_state["performance_level_restored"] = (
                    before_level.get("status") == "observed"
                    and after_level.get("status") == "observed"
                    and before_level.get("value") == after_level.get("value")
                )
                service_state["restored"] = (
                    restored["active"]
                    and restored["health"] == 200
                    and restored["ready"] == 200
                    and service_state["hashes_before"] == service_state["hashes_after"]
                    and service_state["performance_level_restored"]
                )
            root_manifest["service"] = service_state or root_manifest.get("service", {})
            root_manifest["ended"] = time.time()
            expected_ids = {entry["run_id"] for entry in root_manifest.get("expected_runs", [])}
            observed_ids = set(root_manifest.get("runs", {}))
            root_manifest["state"] = "complete" if expected_ids <= observed_ids else "failed"
            failed = [entry for entry in root_manifest.get("runs", {}).values() if entry.get("state") != "complete"]
            root_manifest["run_failures"] = len(failed)
            root_manifest["verdict"] = (
                "complete: Stage 0 inventory recorded; observed failures remain evidence"
                if root_manifest["state"] == "complete"
                else "incomplete: one or more runs did not produce a terminal manifest"
            )
            if service_state and service_state.get("stopped_by_runner") and not service_state.get("restored", False):
                root_manifest["state"] = "failed"
                root_manifest["verdict"] = "failed: resident service restoration contract did not pass"
            write_json(root_manifest_path, root_manifest)
        return 1 if args.fail_on_run_failure and any(result.get("state") != "complete" for result in all_results) else 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
