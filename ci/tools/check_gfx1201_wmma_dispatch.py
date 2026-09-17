#!/usr/bin/env python3
"""Does the gfx1201 BF16-series teacher-forced run dispatch any WMMA kernel?

Only two providers can select a WMMA tile without an environment opt-in:
Mxfp8Gfx1201Wmma (m>=128) and Mxfp6Gfx1201WmmaViaE4M3 (m>=17), and both require
an MX weight format. The NVFP4 W4A4 WMMA body is strictly opt-in
(SLLM_NVFP4_W4A4_PREFILL_FORCE_GFX1201_WMMA). This runner checks the claim
directly with a kernel trace instead of reasoning about the graph.

The resident R9700 user service is leased for the measurement and restored.
"""

from __future__ import annotations

import csv
import glob
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
OUT = REPO / ".local-artifacts/mtp-bench/wmma-check"
PREFIX = OUT / "prefix-one.json"
BINARY = REPO / ".local-artifacts/mtp-bench/claimb/bin/bench-gfx1201"
MODEL = "/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4"
TRACE_DIR = OUT / "trace"


def health(endpoint: str) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=3)
    try:
        conn.request("GET", endpoint)
        r = conn.getresponse()
        r.read()
        return r.status
    except OSError:
        return 0
    finally:
        conn.close()


def service_hashes() -> dict[str, str]:
    files = [pathlib.Path("/home/homelab1/.config/systemd/user") / UNIT,
             REPO / ".local-artifacts/qwen38-r9700-server/run.sh",
             REPO / ".local-artifacts/qwen38-r9700-server/sllm-server"]
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def active() -> bool:
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", UNIT]).returncode == 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": "gfx1201-wmma-dispatch-check-v1", "state": "running",
              "question": "does the default gfx1201 BF16-series path dispatch any WMMA kernel?",
              "condition": "code-rust-bugfix-zh (8284 prompt tokens, 256 forced rows)",
              "binary_sha256": hashlib.sha256(BINARY.read_bytes()).hexdigest(),
              "service_stopped": False, "started": time.time()}

    def save():
        (OUT / "execution.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    save()
    try:
        if subprocess.check_output(["ss", "-Htn", "state", "established", "( sport = :8000 )"],
                                   text=True).strip():
            raise RuntimeError("active connections on the existing server; not leasing")
        report["service_was_active"] = active()
        report["service_hashes_before"] = service_hashes()
        if report["service_was_active"]:
            if health("/readyz") != 200:
                raise RuntimeError("existing server not ready")
            report["service_stopped"] = True
            save()
            subprocess.run(["systemctl", "--user", "stop", UNIT], check=True)
            for _ in range(60):
                if not active():
                    break
                time.sleep(1)

        env = {k: v for k, v in os.environ.items()
               if not k.startswith("SLLM_")
               and k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        env.update({
            "ROCR_VISIBLE_DEVICES": UUID, "LD_LIBRARY_PATH": "/opt/rocm/lib",
            "SLLM_PHASE78_MODEL_PATH": MODEL, "SLLM_PHASE78_TARGET": "gfx1201",
            "SLLM_PHASE78_DEVICE": "0", "SLLM_PHASE78_WARMUPS": "0", "SLLM_PHASE78_MEASURED": "1",
            "SLLM_PHASE78_CHUNK_CAPACITY": "2048", "SLLM_PHASE83_MODE": "coding8192",
            "SLLM_PHASE83_KV": "mxfp8", "SLLM_PHASE83_SAMPLING": "gpu-fixed",
            "SLLM_PHASE83_REPLAY": "0", "SLLM_PHASE83_MTP": "on", "SLLM_PHASE83_MTP_WIDTH": "2",
            "SLLM_PHASE83_STATE_CAPACITY": "10240", "SLLM_PHASE85_A16_STAGE0": "1",
            "SLLM_PHASE85_A16_PREFIX_FILE": str(PREFIX),
            "SLLM_PHASE85_A16_OUTPUT_DIR": str(OUT / "stage0"),
            "SLLM_PHASE85_A16_SERIES": "bf16",
        })
        (OUT / "stage0").mkdir(parents=True, exist_ok=True)
        cmd = ["/usr/bin/rocprofv3", "--kernel-trace", "--output-format", "csv",
               "--output-directory", str(TRACE_DIR), "--", str(BINARY)]
        report["command"] = cmd
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, env=env, cwd=str(REPO))
        report["exit_code"] = proc.returncode
        report["wall_seconds"] = round(time.time() - started, 1)
        (OUT / "run.stdout.json").write_bytes(proc.stdout)
        (OUT / "run.stderr.log").write_bytes(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode(errors="replace")[-500:])

        names: dict[str, int] = {}
        rows = 0
        for path in glob.glob(str(TRACE_DIR / "**" / "*kernel_trace*.csv"), recursive=True):
            with open(path, newline="") as handle:
                for row in csv.DictReader(handle):
                    name = row.get("Kernel_Name") or row.get("Name") or ""
                    names[name] = names.get(name, 0) + 1
                    rows += 1
        report["trace_files"] = sorted(glob.glob(str(TRACE_DIR / "**" / "*kernel_trace*.csv"),
                                                 recursive=True))
        report["total_dispatches_traced"] = rows
        report["distinct_kernels"] = len(names)
        wmma = {k: v for k, v in names.items() if "wmma" in k.lower()}
        report["wmma_kernels"] = wmma
        report["wmma_dispatch_count"] = sum(wmma.values())
        report["top_kernels"] = sorted(names.items(), key=lambda kv: -kv[1])[:15]
        report["verdict"] = ("NO WMMA KERNEL DISPATCHED" if not wmma
                             else "WMMA DISPATCHED: %d" % sum(wmma.values()))
        report["state"] = "complete"
        print(report["verdict"], flush=True)
        print("traced %d dispatches, %d distinct kernels" % (rows, len(names)), flush=True)
    except Exception as error:  # noqa: BLE001
        report["state"] = "failed"
        report["error"] = str(error)[:600]
        print("ERROR:", error, flush=True)
    finally:
        if report["service_stopped"]:
            subprocess.run(["systemctl", "--user", "start", UNIT], check=False)
            for _ in range(120):
                if active() and health("/healthz") == 200 and health("/readyz") == 200:
                    break
                time.sleep(1)
            report["service_hashes_after"] = service_hashes()
            report["restored_health"] = health("/healthz")
            report["restored_ready"] = health("/readyz")
            report["service_restored"] = (active() and report["restored_health"] == 200
                                          and report["restored_ready"] == 200
                                          and report["service_hashes_before"] == report["service_hashes_after"])
        report["ended"] = time.time()
        save()
        print("service_restored=%s state=%s" % (report.get("service_restored"), report["state"]), flush=True)
    return 0 if report["state"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
