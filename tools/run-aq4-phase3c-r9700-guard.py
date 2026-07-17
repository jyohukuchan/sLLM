#!/usr/bin/env python3
"""Run the Phase 3c R9700-only read-only guard through the service user.

This tool is intentionally separate from lock acquisition and tracing.  It
neither allocates device memory nor launches a kernel.  It is run as root so
that rehearsal and the eventual service window use the identical ``runuser``
boundary.  AMD-SMI is always invoked by its absolute ROCm path because the
default ``runuser`` PATH does not contain ``/opt/rocm/bin``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


AMD_SMI_PATH = Path("/opt/rocm/bin/amd-smi")
EXPECTED_BDF = "0000:47:00.0"
EXPECTED_ARCHITECTURE = "gfx1201"
EXPECTED_DEVICE_ID = "0x7551"
GUARD_USER = "homelab1"
GUARD_ENV = {
    "HOME": "/home/homelab1",
    "HIP_VISIBLE_DEVICES": "1",
    "ULLM_HIP_VISIBLE_DEVICES": "1",
}


class GuardError(RuntimeError):
    """The mandatory R9700 identity contract was not satisfied."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GuardError(message)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_runuser_command(runuser: str, command: Sequence[str]) -> list[str]:
    """Return the one service-user boundary used by all guard queries."""

    return [
        runuser,
        "-u",
        GUARD_USER,
        "--",
        "/usr/bin/env",
        *(f"{key}={value}" for key, value in GUARD_ENV.items()),
        *command,
    ]


def run_as_guard_user(runuser: str, command: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        build_runuser_command(runuser, command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def command_record(result: subprocess.CompletedProcess[bytes], command: Sequence[str]) -> dict[str, Any]:
    return {
        "exit_code": result.returncode,
        "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
        "command": list(command),
    }


def parse_json(raw: bytes, description: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as error:
        raise GuardError(f"{description} was not valid JSON: {error}") from error


def validate_hip_payload(payload: Any) -> tuple[dict[str, Any], str]:
    require(isinstance(payload, dict), "HIP guard payload must be an object")
    require(payload.get("schema_version") == "ullm.r9700_hip_device_guard.v1", "unexpected HIP guard schema")
    require(payload.get("status") == "valid", "HIP guard did not report valid")
    required = payload.get("required")
    actual = payload.get("actual")
    require(isinstance(required, dict), "HIP guard required contract is missing")
    require(isinstance(actual, dict), "HIP guard actual identity is missing")
    expected = {
        "hip_visible_devices": "1",
        "ullm_hip_visible_devices": "1",
        "visible_hip_device_count": 1,
        "architecture": EXPECTED_ARCHITECTURE,
    }
    require(all(required.get(key) == value for key, value in expected.items()), "HIP guard required contract changed")
    require(all(actual.get(key) == value for key, value in expected.items()), "filtered HIP identity mismatched")
    require(actual.get("filtered_hip_ordinal") == 0, "filtered HIP ordinal was not zero")
    name = actual.get("name")
    require(isinstance(name, str) and name.strip(), "filtered HIP device name was empty")
    bdf = str(actual.get("pci_bdf", "")).lower()
    require(re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bdf) is not None, "HIP BDF was invalid")
    require(bdf == EXPECTED_BDF, f"HIP BDF {bdf!r} was not fixed R9700 BDF {EXPECTED_BDF!r}")
    return actual, bdf


def asic_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        records = [value] if isinstance(value.get("asic"), dict) and isinstance(value.get("bus"), dict) else []
        for child in value.values():
            records.extend(asic_records(child))
        return records
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for child in value:
            records.extend(asic_records(child))
        return records
    return []


def validate_amd_smi_payload(payload: Any, hip_bdf: str) -> dict[str, str]:
    records = asic_records(payload)
    require(len(records) == 1, f"expected exactly one targeted AMD-SMI record, got {len(records)}")
    record = records[0]
    architecture = str(record["asic"].get("target_graphics_version", "")).split(":", 1)[0].lower()
    device_id = str(record["asic"].get("device_id", "")).lower()
    market_name = str(record["asic"].get("market_name", "")).strip()
    smi_bdf = str(record["bus"].get("bdf", "")).lower()
    require(architecture == EXPECTED_ARCHITECTURE, f"AMD-SMI architecture was {architecture!r}")
    require(device_id == EXPECTED_DEVICE_ID, f"AMD-SMI device ID was {device_id!r}")
    require(market_name and market_name != "N/A", "AMD-SMI market name was empty")
    require(smi_bdf == hip_bdf, f"AMD-SMI BDF {smi_bdf!r} did not match HIP BDF {hip_bdf!r}")
    return {
        "pci_bdf": smi_bdf,
        "architecture": architecture,
        "pci_device_id": device_id,
        "market_name": market_name,
    }


def save_command(output: Path, stem: str, result: subprocess.CompletedProcess[bytes]) -> None:
    (output / f"{stem}.json").write_bytes(result.stdout)
    (output / f"{stem}.stderr").write_bytes(result.stderr)
    (output / f"{stem}.exit-code").write_text(f"{result.returncode}\n", encoding="utf-8")


def capture_health(output: Path, phase: str, bdf: str, runuser: str) -> dict[str, Any]:
    commands = {
        "metrics": [str(AMD_SMI_PATH), "metric", "--gpu", bdf, "--ecc", "--ecc-blocks", "--clock", "--power", "--temperature", "--perf-level", "--json"],
        "bad-pages": [str(AMD_SMI_PATH), "bad-pages", "--gpu", bdf, "--pending", "--retired", "--un-res", "--json"],
        "static": [str(AMD_SMI_PATH), "static", "--gpu", bdf, "--driver", "--ifwi", "--limit", "--json"],
        "firmware": [str(AMD_SMI_PATH), "firmware", "--gpu", bdf, "--ucode-list", "--json"],
    }
    records: dict[str, Any] = {}
    for name, command in commands.items():
        result = run_as_guard_user(runuser, command)
        save_command(output, f"gpu-health-{phase}-{name}", result)
        try:
            json.loads(result.stdout)
            json_status = "parsed"
        except json.JSONDecodeError:
            json_status = "unparsed"
        records[name] = {
            **command_record(result, build_runuser_command(runuser, command)),
            "json_status": json_status,
        }
    summary = {
        "schema_version": "ullm.aq4_phase3c_gpu_health.v1",
        "status": "complete" if all(record["exit_code"] == 0 for record in records.values()) else "partial",
        "phase": phase,
        "target_pci_bdf": bdf,
        "recorded_at_utc": utc_now(),
        "records": records,
    }
    write_json(output / f"gpu-health-{phase}-summary.json", summary)
    return summary


def run_guard(args: argparse.Namespace) -> int:
    output = args.output
    if output.exists():
        raise GuardError(f"refusing to overwrite existing evidence path: {output}")
    output.mkdir(mode=0o700, parents=False)
    summary: dict[str, Any] = {
        "schema_version": "ullm.aq4_phase3c_r9700_guard_rehearsal.v1",
        "status": "invalid",
        "operation": "read_only_identity_and_health_queries",
        "recorded_at_utc": utc_now(),
        "amd_smi_path": str(AMD_SMI_PATH),
        "expected_bdf": EXPECTED_BDF,
    }
    try:
        require(os.geteuid() == 0, "run as root so the runuser boundary is explicit")
        require(AMD_SMI_PATH.is_file() and os.access(AMD_SMI_PATH, os.X_OK), "AMD-SMI absolute path is unavailable")
        guard_bin = args.guard_bin.resolve()
        require(guard_bin.is_file() and os.access(guard_bin, os.X_OK), "HIP guard binary is unavailable")
        runuser = shutil.which("runuser")
        require(runuser is not None, "runuser is unavailable")

        environment = run_as_guard_user(runuser, ["/usr/bin/env"])
        (output / "runuser-environment.stdout").write_bytes(environment.stdout)
        (output / "runuser-environment.stderr").write_bytes(environment.stderr)
        require(environment.returncode == 0, "could not inspect runuser environment")
        path = next(
            (line.split("=", 1)[1] for line in environment.stdout.decode("utf-8", "replace").splitlines() if line.startswith("PATH=")),
            "",
        )
        summary["runuser_environment"] = {
            **command_record(environment, build_runuser_command(runuser, ["/usr/bin/env"])),
            "path": path,
            "path_contains_opt_rocm_bin": "/opt/rocm/bin" in path.split(":"),
        }

        hip_command = [str(guard_bin)]
        hip = run_as_guard_user(runuser, hip_command)
        save_command(output, "r9700-hip-device-guard", hip)
        require(hip.returncode == 0, f"HIP guard returned {hip.returncode}")
        hip_actual, bdf = validate_hip_payload(parse_json(hip.stdout, "HIP guard output"))
        summary["hip"] = {**command_record(hip, build_runuser_command(runuser, hip_command)), "actual": hip_actual}

        identity_command = [str(AMD_SMI_PATH), "static", "--gpu", bdf, "--asic", "--bus", "--json"]
        identity = run_as_guard_user(runuser, identity_command)
        save_command(output, "r9700-amd-smi-identity", identity)
        require(identity.returncode == 0, f"AMD-SMI ASIC cross-check returned {identity.returncode}")
        amd_smi = validate_amd_smi_payload(parse_json(identity.stdout, "AMD-SMI identity output"), bdf)
        architecture_guard = {
            "schema_version": "ullm.aq4_phase3c_r9700_architecture_guard.v1",
            "status": "valid",
            "r9700_identity_basis": {"hip_visible_devices": "1", "filtered_hip_ordinal": 0, "architecture": EXPECTED_ARCHITECTURE, "pci_device_id": EXPECTED_DEVICE_ID},
            "hip": hip_actual,
            "amd_smi": amd_smi,
        }
        write_json(output / "r9700-architecture-guard.json", architecture_guard)
        summary["amd_smi_identity"] = {**command_record(identity, build_runuser_command(runuser, identity_command)), "actual": amd_smi}

        health = capture_health(output, args.health_phase, bdf, runuser)
        summary["health"] = {"phase": args.health_phase, "status": health["status"]}
        summary["status"] = "valid"
    except GuardError as error:
        summary["failure"] = str(error)
    except Exception as error:  # Preserve failure evidence before exposing an implementation error.
        summary["failure"] = f"unexpected {type(error).__name__}: {error}"
    finally:
        summary["finished_at_utc"] = utc_now()
        write_json(output / "r9700-guard-rehearsal-summary.json", summary)
    return 0 if summary["status"] == "valid" else 1


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory; it must not exist")
    parser.add_argument("--guard-bin", type=Path, required=True, help="prebuilt host-only HIP identity guard")
    parser.add_argument("--health-phase", default="rehearsal", help="label for this targeted health snapshot")
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    try:
        return run_guard(parse_args(argv))
    except GuardError as error:
        print(f"aq4-phase3c-r9700-guard: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
