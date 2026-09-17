#!/usr/bin/env python3
"""Teacher-forced M1 pass on the R9700 (gfx1201), leasing the resident service.

The forced prefix is the one generated on gfx1030 so both targets are evaluated
at identical positions; the prefix is a forced input, so which device produced
it does not matter for validity, and reusing it makes the two GPUs comparable.

The existing user service is stopped only for the measurement and restored with
its unit, run.sh and binary hashes verified unchanged plus health/ready 200.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path("/home/homelab1/coding-local/sLLM")
UNIT = "sllm-qwen38-r9700.service"
UUID = "GPU-a8e9ddefa2d60f55"
TARGET = "gfx1201"
OUT = REPO / ".local-artifacts/mtp-bench/claimb-gfx1201"
PREFIX = REPO / ".local-artifacts/mtp-bench/claimb/gen2-bf16/prefixes.json"
BINARY = REPO / ".local-artifacts/mtp-bench/claimb/bin/bench-gfx1201"
MODEL = "/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4"
SERIES = [("bf16", None), ("mxfp8", ".local-artifacts/phase84/mxfp8"),
          ("mxfp6", ".local-artifacts/phase84/mxfp6")]


def health(endpoint: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
    try:
        conn.request("GET", endpoint)
        response = conn.getresponse()
        response.read()
        return response.status
    except OSError:
        return 0
    finally:
        conn.close()


def service_hashes() -> dict[str, str]:
    files = [
        pathlib.Path("/home/homelab1/.config/systemd/user") / UNIT,
        REPO / ".local-artifacts/qwen38-r9700-server/run.sh",
        REPO / ".local-artifacts/qwen38-r9700-server/sllm-server",
    ]
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def active() -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0


def performance_level_path() -> pathlib.Path:
    matches = [p.resolve() for p in pathlib.Path("/sys/class/drm").glob("card[0-9]*/device")
               if (p / "unique_id").exists()
               and (p / "unique_id").read_text().strip().lower() == UUID.removeprefix("GPU-").lower()]
    if len(matches) != 1:
        raise RuntimeError("exact GPU UUID is not unique")
    return matches[0] / "power_dpm_force_performance_level"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": "mtp-teacher-forced-lease-v1", "state": "running",
        "target": TARGET, "device_uuid": UUID,
        "prefix_file": str(PREFIX), "prefix_file_sha256": hashlib.sha256(PREFIX.read_bytes()).hexdigest(),
        "prefix_origin": "generated on gfx1030; reused so both targets are forced at identical positions",
        "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        "started": time.time(), "service_stopped": False, "jobs": [],
    }

    def save():
        (OUT / "execution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    level_path = performance_level_path()
    report["performance_level_original"] = level_path.read_text().strip()
    save()

    try:
        clients = subprocess.check_output(
            ["ss", "-Htn", "state", "established", "( sport = :8000 )"], text=True)
        if clients.strip():
            raise RuntimeError("active connections on the existing server; not leasing")
        report["service_was_active"] = active()
        report["service_hashes_before"] = service_hashes()
        if report["service_was_active"]:
            if health("/readyz") != 200:
                raise RuntimeError("existing server is not ready; not leasing")
            report["service_stopped"] = True
            save()
            subprocess.run(["systemctl", "--user", "stop", UNIT], check=True)
            for _ in range(60):
                if not active():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("service did not stop")

        base_env = {k: v for k, v in os.environ.items()
                    if not k.startswith("SLLM_")
                    and k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        base_env.update({
            "ROCR_VISIBLE_DEVICES": UUID,
            "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "SLLM_PHASE78_MODEL_PATH": MODEL,
            "SLLM_PHASE78_TARGET": TARGET,
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
            "SLLM_PHASE85_A16_PREFIX_FILE": str(PREFIX),
        })

        for name, companion in SERIES:
            env = dict(base_env)
            env["SLLM_PHASE85_A16_SERIES"] = name
            job_dir = OUT / f"fixed-{name}"
            job_dir.mkdir(parents=True, exist_ok=True)
            env["SLLM_PHASE85_A16_OUTPUT_DIR"] = str(job_dir)
            if companion:
                env["SLLM_PHASE84_MTP_COMPANION_PATH"] = str(REPO / companion)
            started = time.time()
            proc = subprocess.run([str(BINARY)], capture_output=True, env=env, cwd=str(REPO))
            (job_dir / "report.json").write_bytes(proc.stdout)
            (job_dir / "stderr.log").write_bytes(proc.stderr)
            item = {"series": name, "companion": companion, "exit_code": proc.returncode,
                    "wall_seconds": round(time.time() - started, 1)}
            if proc.returncode != 0:
                item["error"] = proc.stderr.decode(errors="replace")[-400:]
            report["jobs"].append(item)
            save()
            print("%-6s exit=%d %.0fs" % (name, proc.returncode, item["wall_seconds"]), flush=True)

        report["state"] = "complete" if all(j["exit_code"] == 0 for j in report["jobs"]) else "failed"
    except Exception as error:  # noqa: BLE001 - recorded, service still restored below
        report["state"] = "failed"
        report["error"] = str(error)[:600]
        print("ERROR:", error, flush=True)
    finally:
        observed_level = level_path.read_text().strip()
        report["performance_level_restored"] = observed_level == report["performance_level_original"]
        if report["service_stopped"]:
            subprocess.run(["systemctl", "--user", "start", UNIT], check=False)
            for _ in range(120):
                if active() and health("/healthz") == 200 and health("/readyz") == 200:
                    break
                time.sleep(1)
            report["service_hashes_after"] = service_hashes()
            report["restored_health"] = health("/healthz")
            report["restored_ready"] = health("/readyz")
            report["service_restored"] = (
                active() and report["restored_health"] == 200 and report["restored_ready"] == 200
                and report["service_hashes_before"] == report["service_hashes_after"])
            if not report["service_restored"]:
                report["state"] = "failed"
        report["ended"] = time.time()
        save()
        print("service_restored=%s health=%s ready=%s level_restored=%s state=%s" % (
            report.get("service_restored"), report.get("restored_health"),
            report.get("restored_ready"), report["performance_level_restored"], report["state"]),
            flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
