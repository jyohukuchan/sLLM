#!/usr/bin/env python3
"""Run a Phase86 job list on one exact GPU and restore its service state.

Jobs supply a name and SLLM_* environment overrides. Each invocation uses a
fresh output directory; stdout, stderr, configuration, and hashes are retained.
The process has no elapsed-time kill limit. Poll execution.json and job logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import run_mtp_teacher_forced_r9700 as lease

REPO = Path(__file__).resolve().parents[2]
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": lease.UUID}
RUNTIME_CONTROL_KEYS = (
    "DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING",
    "DEBUG_HIP_FORCE_GRAPH_QUEUES",
    "DEBUG_HIP_GRAPH_BATCH_SIZE",
    "LD_PRELOAD",
    "HSA_ENABLE_SDMA",
    "ROCPROFILER_QUEUE_INTERPOSITION",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=UUIDS, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    jobs = json.loads(args.jobs.read_text())
    if not jobs or len({j["name"] for j in jobs}) != len(jobs):
        raise ValueError("nonempty jobs with unique names required")
    for job in jobs:
        if Path(job["name"]).name != job["name"] or job["name"] in (".", ".."):
            raise ValueError("job names must be directory basenames")
        if any(not k.startswith("SLLM_") for k in job.get("env", {})):
            raise ValueError("only SLLM_* overrides are supported")
    output.mkdir(parents=True, exist_ok=False)
    uuid = UUIDS[args.target]
    devices = [p.parent for p in Path('/sys/class/drm').glob('card[0-9]*/device/unique_id')
               if p.read_text().strip().lower() == uuid[4:].lower()]
    if len(devices) != 1:
        raise RuntimeError("exact device UUID not found uniquely")
    level = devices[0] / "power_dpm_force_performance_level"
    report = {"schema": "phase86-execution-v1", "state": "running",
              "target": args.target, "uuid": uuid,
              "binary": str(binary), "binary_sha256": digest(binary),
              "jobs_sha256": digest(args.jobs), "started": time.time(),
              "original_performance_level": level.read_text().strip(), "jobs": []}

    def save():
        temporary = output / "execution.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "execution.json")

    stopped = False
    save()
    try:
        if args.target == "gfx1030":
            status = subprocess.check_output(
                ["/home/homelab1/.local/bin/qwen38-subagent-server", "status"], text=True)
            if "process:  stopped" not in status:
                raise RuntimeError("Qwen service must be idle and stopped before GPU work")
        else:
            report["service_hashes_before"] = lease.service_hashes()
            report["service_was_active"] = lease.active()
            if report["service_was_active"]:
                clients = subprocess.check_output(
                    ["ss", "-Htn", "state", "established", "( sport = :8000 )"], text=True)
                if clients.strip() or lease.health("/readyz") != 200:
                    raise RuntimeError("resident server busy or unready; cannot lease")
                stopped = True
                subprocess.run(["systemctl", "--user", "stop", lease.UNIT], check=True)
        base = {k: v for k, v in os.environ.items()
                if not k.startswith("SLLM_") and k not in
                ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        base.update({"ROCR_VISIBLE_DEVICES": uuid, "LD_LIBRARY_PATH": "/opt/rocm/lib",
                     "SLLM_PHASE78_TARGET": args.target, "SLLM_PHASE78_DEVICE": "0",
                     "SLLM_PHASE78_MODEL_PATH": lease.MODEL,
                     "SLLM_PHASE78_CHUNK_CAPACITY": "2048",
                     "SLLM_PHASE78_WARMUPS": "0", "SLLM_PHASE78_MEASURED": "1",
                     "SLLM_PHASE83_MODE": "coding8192", "SLLM_PHASE83_KV": "mxfp8",
                     "SLLM_PHASE83_SAMPLING": "gpu-fixed", "SLLM_PHASE83_REPLAY": "0",
                     "SLLM_PHASE83_MTP": "on", "SLLM_PHASE83_MTP_WIDTH": "2",
                     "SLLM_PHASE83_STATE_CAPACITY": "10240"})
        # Avoid the observed HIP 7.14 segmented-graph signal underflow.
        # An explicit process-level override remains available for diagnostics.
        base.setdefault("DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING",
                        "0" if args.target == "gfx1030" else "1")
        for job in jobs:
            directory = output / job["name"]
            directory.mkdir()
            env = dict(base)
            env.update(job.get("env", {}))
            env["SLLM_PHASE85_A16_OUTPUT_DIR"] = str(directory)
            item = {"name": job["name"], "started": time.time(),
                    "env": {k: v for k, v in env.items() if k.startswith("SLLM_")},
                    "runtime_controls": {
                        k: env[k] for k in RUNTIME_CONTROL_KEYS if k in env
                    }}
            report["jobs"].append(item)
            with (directory / "report.json").open("w") as stdout, \
                    (directory / "stderr.log").open("w") as stderr:
                process = subprocess.Popen([str(binary)], cwd=REPO, env=env,
                                           stdout=stdout, stderr=stderr)
                item["pid"] = process.pid
                save()
                print(f"{job['name']} pid={process.pid}", flush=True)
                item["exit_code"] = process.wait()
            item["elapsed_seconds"] = time.time() - item["started"]
            save()
            if item["exit_code"] != 0:
                raise RuntimeError(f"job failed: {job['name']}")
            document = json.loads((directory / "report.json").read_text())
            if document.get("state") != "PASS":
                raise RuntimeError(f"non-PASS report: {job['name']}")
            item["report_sha256"] = digest(directory / "report.json")
            print(f"{job['name']} PASS {item['elapsed_seconds']:.1f}s", flush=True)
        report["state"] = "complete"
    except Exception as error:
        report["state"] = "failed"
        report["error"] = str(error)
    finally:
        if stopped:
            subprocess.run(["systemctl", "--user", "start", lease.UNIT], check=False)
            for _ in range(120):
                if lease.active() and lease.health('/healthz') == lease.health('/readyz') == 200:
                    break
                time.sleep(1)
            report["restored_health"] = lease.health('/healthz')
            report["restored_ready"] = lease.health('/readyz')
        if "service_was_active" in report:
            report["service_hashes_after"] = lease.service_hashes()
            report["service_restored"] = (
                lease.active() == report["service_was_active"]
                and report["service_hashes_before"] == report["service_hashes_after"]
                and (not stopped or report.get("restored_health") == report.get("restored_ready") == 200))
            if not report["service_restored"]:
                report["state"] = "failed"
        report["performance_level_restored"] = level.read_text().strip() == report["original_performance_level"]
        if not report["performance_level_restored"]:
            report["state"] = "failed"
        report["ended"] = time.time()
        save()
    print(report["state"], report.get("error", ""), flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
