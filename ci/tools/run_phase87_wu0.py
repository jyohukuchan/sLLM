#!/usr/bin/env python3
"""Measure read/copy modes on the exact Stage 0 payload list with a GPU lease."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import run_mtp_teacher_forced_r9700 as lease

REPO = Path(__file__).resolve().parents[2]
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": lease.UUID}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=UUIDS, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--payloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--probe-arg", action="append", default=[],
                        help="extra probe argument, e.g. --probe-arg=--warmup-ms --probe-arg=300")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    payloads = [json.loads(line) for line in args.payloads.read_text().splitlines() if line]
    keys = [(row["m"], row["k"], row["n"], row["encoding"]) for row in payloads]
    if len(keys) != 73 or len(set(keys)) != len(keys):
        raise ValueError("expected the 73 distinct frozen Stage 0 payloads")
    binary, output = args.binary.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    uuid = UUIDS[args.target]
    devices = [p.parent for p in Path("/sys/class/drm").glob("card[0-9]*/device/unique_id")
               if p.read_text().strip().lower() == uuid[4:].lower()]
    if len(devices) != 1:
        raise RuntimeError("exact device UUID not unique")
    level = devices[0] / "power_dpm_force_performance_level"
    record = {"schema": "phase87-wu0-execution-v1", "state": "running",
              "target": args.target, "uuid": uuid, "binary": str(binary),
              "binary_sha256": sha256(binary), "payloads": str(args.payloads.resolve()),
              "payloads_sha256": sha256(args.payloads), "probe_args": args.probe_arg,
              "started": time.time(),
              "performance_level_before": level.read_text().strip(), "jobs": []}

    def save():
        (output / "execution.json").write_text(json.dumps(record, indent=2) + "\n")

    stopped = False
    save()
    try:
        if args.target == "gfx1030":
            status = subprocess.check_output(
                ["/home/homelab1/.local/bin/qwen38-subagent-server", "status"], text=True)
            if "process:  stopped" not in status:
                raise RuntimeError("local Qwen must be idle and stopped before GPU measurement")
        else:
            record["service_was_active"] = lease.active()
            record["service_hashes_before"] = lease.service_hashes()
            if record["service_was_active"]:
                clients = subprocess.check_output(
                    ["ss", "-Htn", "state", "established", "( sport = :8000 )"], text=True)
                if clients.strip() or lease.health("/readyz") != 200:
                    raise RuntimeError("resident service busy or unready")
                stopped = True
                subprocess.run(["systemctl", "--user", "stop", lease.UNIT], check=True)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("SLLM_") and key not in
               ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
        env.update(ROCR_VISIBLE_DEVICES=uuid, LD_LIBRARY_PATH="/opt/rocm/lib")
        for repeat in range(args.rounds):
            # Alternate mode order to expose drift rather than selecting the
            # faster measurement post hoc.
            for mode in (("copy", "read") if repeat % 2 == 0 else ("read", "copy")):
                name = f"{mode}-{repeat}"
                command = [str(binary), "--mode", mode, "--warmups", "3", "--measured", "9",
                           *args.probe_arg]
                for m, k, n, encoding in keys:
                    command.extend(["--shape", f"{m}x{k}x{n}:{encoding}"])
                job = {"name": name, "mode": mode, "round": repeat,
                       "command": command, "started": time.time()}
                record["jobs"].append(job)
                with (output / f"{name}.jsonl").open("w") as out, (output / f"{name}.stderr").open("w") as err:
                    process = subprocess.Popen(command, cwd=REPO, env=env, stdout=out, stderr=err)
                    job["pid"] = process.pid
                    save()
                    print(f"{args.target} {name} pid={process.pid}", flush=True)
                    job["exit_code"] = process.wait()
                job["ended"] = time.time()
                save()
                if job["exit_code"]:
                    raise RuntimeError(f"probe failed: {name}")
                rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
                if [(r["m"], r["k"], r["n"], r["encoding"]) for r in rows] != keys:
                    raise RuntimeError("payload set/order differs from Stage 0")
                for source, row in zip(payloads, rows):
                    if row.get("status") != "PASS" or row.get("target") != args.target:
                        raise RuntimeError("GPU result not PASS on exact target")
                    if row["payload_bytes"] != source["payload_bytes"]:
                        raise RuntimeError("logical payload byte contract changed")
                    if not math.isfinite(row["median_ms"]) or row["median_ms"] <= 0:
                        raise RuntimeError("invalid GPU duration")
                job["case_count"] = len(rows)
                job["output_sha256"] = sha256(output / f"{name}.jsonl")
                save()
                print(f"{args.target} {name}: {len(rows)} cases PASS", flush=True)
        record["state"] = "complete"
    except Exception as error:
        record.update(state="failed", error=str(error))
    finally:
        if stopped:
            subprocess.run(["systemctl", "--user", "start", lease.UNIT], check=True)
            for _ in range(120):
                if lease.active() and lease.health("/healthz") == lease.health("/readyz") == 200:
                    break
                time.sleep(1)
        if args.target == "gfx1201" and "service_was_active" in record:
            record["service_hashes_after"] = lease.service_hashes()
            record["service_restored"] = (
                lease.active() == record["service_was_active"]
                and record["service_hashes_before"] == record["service_hashes_after"]
                and (not stopped or lease.health("/healthz") == lease.health("/readyz") == 200))
            if not record["service_restored"]:
                record["state"] = "failed"
        record["performance_level_restored"] = level.read_text().strip() == record["performance_level_before"]
        if not record["performance_level_restored"]:
            record["state"] = "failed"
        record["ended"] = time.time()
        save()
    print(record["state"], record.get("error", ""), flush=True)
    return 0 if record["state"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
