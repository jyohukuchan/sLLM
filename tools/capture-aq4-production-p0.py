#!/usr/bin/env python3
"""Capture a hash-bound, non-secret P0 snapshot for the AQ4 production path.

The capture is read-only with respect to the running product.  It writes only
under the requested output directory and never records environment values that
look like credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SECRET_KEY_RE = re.compile(r"(?:KEY|TOKEN|PASSWORD|PASS|SECRET|AUTH|CREDENTIAL)", re.I)
DEFAULT_MANIFEST = Path("/etc/ullm/served-models/active.json")
DEFAULT_ENV = Path("/etc/ullm/openai-gateway-manifest.env")
DEFAULT_UNIT = Path("/etc/systemd/system/ullm-openai.service")
DEFAULT_DROPIN = Path("/etc/systemd/system/ullm-openai.service.d/10-served-model.conf")
DEFAULT_ROLLBACK_BUNDLE = ROOT / "benchmarks/results/2026-07-13/qwen35-9b-aq4-reasoning-v0.1/release-bundle-ae8b2bb-20260714-final.json"


class CaptureError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise CaptureError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"JSON root is not an object: {path}")
    return value


def run_capture(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[:262_144],
        "stderr": completed.stderr[:32_768],
    }


def safe_command(command: list[str], timeout: float = 30.0) -> dict[str, Any]:
    result = run_capture(command, timeout)
    result["stdout"] = redact_text(result["stdout"])
    result["stderr"] = redact_text(result["stderr"])
    result["command_string"] = shlex.join(command)
    return result


def redact_text(value: str) -> str:
    return re.sub(
        r"(?i)(api[_-]?key|password|passwd|secret|token|authorization)(\s*[=:]\s*)([^\s,;\]\"}]+)",
        r"\1\2<redacted>",
        value,
    )


def parse_env(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            result[f"line_{number}"] = {"malformed": True}
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()
        result[key] = {
            "value_sha256": value_hash,
            "sensitive": bool(SECRET_KEY_RE.search(key)),
            "value": None if SECRET_KEY_RE.search(key) else value,
        }
    return result


def env_diff(old: dict[str, dict[str, Any]] | None, current: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if old is None:
        return {"status": "unavailable", "reason": "rollback environment content was not bundled", "keys": sorted(current)}
    keys = sorted(set(old) | set(current))
    changes = []
    for key in keys:
        if key not in old:
            changes.append({"key": key, "kind": "added", "current": current[key]})
        elif key not in current:
            changes.append({"key": key, "kind": "removed", "previous": old[key]})
        elif old[key] != current[key]:
            changes.append({"key": key, "kind": "changed", "previous": old[key], "current": current[key]})
    return {"status": "match" if not changes else "different", "changes": changes}


def scrub_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    def scrub(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {name: scrub(child, name) for name, child in value.items()}
        if isinstance(value, list):
            return [scrub(child, key) for child in value]
        if SECRET_KEY_RE.search(key):
            return "<redacted>"
        return value

    return scrub(manifest)


def file_identity(path: Path, label: str, required: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {"label": label, "path": str(path)}
    if not path.exists():
        item["status"] = "missing"
        if required:
            raise CaptureError(f"required {label} is missing: {path}")
        return item
    item["status"] = "ok"
    item["sha256"] = sha256_file(path)
    item["bytes"] = path.stat().st_size
    return item


def git_snapshot() -> dict[str, Any]:
    commit = run_capture(["git", "rev-parse", "HEAD"])
    status = run_capture(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", ".", ":(exclude).rocprofv3"])
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "status_raw": status["stdout"],
        "status_clean": status["returncode"] == 0 and not status["stdout"],
        "status_command": shlex.join(status["command"]),
    }


def manifest_identities(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    public = manifest.get("public", {})
    fmt = manifest.get("format", {})
    worker = manifest.get("worker", {})
    product = manifest.get("product", {})
    tokenizer = manifest.get("tokenizer", {})
    package = product.get("package", {}) if isinstance(product, dict) else {}
    package_path = Path(product.get("root", "")) / package.get("manifest_path", "")
    tokenizer_files = tokenizer.get("files", {}) if isinstance(tokenizer, dict) else {}
    tokenizer_root = Path(tokenizer.get("root", "")) if isinstance(tokenizer, dict) else Path()
    return {
        "model": {
            "id": public.get("id"),
            "revision": public.get("revision"),
            "format_id": fmt.get("format_id"),
            "implementation_id": fmt.get("implementation_id"),
        },
        "manifest": file_identity(manifest_path, "active served-model manifest", True),
        "worker": file_identity(Path(worker.get("binary", "")), "active worker binary", True),
        "package_manifest": file_identity(package_path, "package manifest", True),
        "tokenizer": {
            "root": str(tokenizer_root),
            "files": [
                {
                    "name": name,
                    "declared_sha256": digest,
                    "observed": file_identity(tokenizer_root / name, f"tokenizer {name}", True),
                }
                for name, digest in sorted(tokenizer_files.items())
            ],
        },
        "product": {
            "root": product.get("root"),
            "artifact": product.get("artifact"),
            "promotion_receipt": file_identity(Path(manifest.get("promotion", {}).get("receipt", "")), "promotion receipt", True),
        },
    }


def topology() -> dict[str, Any]:
    return {
        "systemd": {
            "show": safe_command(["systemctl", "show", "ullm-openai.service", "-p", "MainPID", "-p", "ExecStart", "-p", "EnvironmentFiles", "-p", "FragmentPath", "-p", "DropInPaths", "-p", "ActiveState", "-p", "SubState"]),
            "enabled": safe_command(["systemctl", "is-enabled", "ullm-openai.service"]),
            "active": safe_command(["systemctl", "is-active", "ullm-openai.service"]),
        },
        "processes": safe_command(["ps", "-eo", "pid,ppid,user,stat,etime,args"]),
        "docker": {
            "containers": safe_command(["docker", "ps", "--no-trunc", "--format", "{{.ID}} {{.Image}} {{.Names}} {{.Status}}"]),
            "openwebui_mounts": safe_command(["docker", "inspect", "open-webui", "--format", "{{json .Mounts}}"], 30),
            "openwebui_runtime": safe_command(["docker", "inspect", "open-webui", "--format", "{{.Image}} {{.State.Status}} {{json .NetworkSettings.Ports}}"], 30),
        },
        "ports": safe_command(["ss", "-ltnp"]),
    }


def openwebui_marker_state(manifest_sha256: str) -> dict[str, Any]:
    # The DB is owned by the OpenWebUI container.  Reading it through `docker exec`
    # would expose more application state than this P0 artifact needs, so the
    # capture records the trusted reconciliation check as a separate, explicit
    # state.  The P6 gate must replace this with a live marker observation.
    return {
        "managed_model_marker": "not_reconciled_in_p0_capture",
        "expected_served_model_manifest_sha256": manifest_sha256,
        "explicit_thinking_budget_tokens": "gateway_path_present_ui_path_not_reconciled",
        "promotion_gate_required": True,
    }


def capture(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json_load(args.manifest)
    active_manifest_sha = sha256_file(args.manifest)
    rollback = json_load(args.rollback_bundle) if args.rollback_bundle.is_file() else None
    rollback_target = rollback.get("rollback_target", {}) if rollback else {}
    current_env_sha = sha256_file(args.environment) if args.environment.is_file() else None
    current_unit_sha = sha256_file(args.unit)
    env_snapshot = parse_env(args.environment)
    previous_env = None
    if args.rollback_environment and args.rollback_environment.is_file():
        previous_env = parse_env(args.rollback_environment)
    release_target_matches_current = {
        "manifest": rollback_target.get("manifest_sha256") == active_manifest_sha,
        "environment": rollback_target.get("environment_sha256") == current_env_sha,
        "systemd_unit": rollback_target.get("systemd_unit_sha256") == current_unit_sha,
    }
    legacy_environment = Path("/etc/ullm/openai-gateway.env")
    legacy_environment_sha = sha256_file(legacy_environment) if legacy_environment.is_file() else None
    environment_reason = "release bundle environment target matches active manifest-mode environment"
    if not release_target_matches_current["environment"] and rollback_target.get("environment_sha256") == legacy_environment_sha:
        environment_reason = (
            "release bundle was produced before the served-model drop-in switched the active service "
            "from legacy openai-gateway.env to openai-gateway-manifest.env"
        )
    rollback_binding = {
        "schema_version": "ullm.aq4_production_optimization_rollback.v1",
        "status": "bound",
        "active_manifest_sha256": active_manifest_sha,
        "current_environment_sha256": current_env_sha,
        "release_bundle": file_identity(args.rollback_bundle, "source release bundle", True),
        "release_bundle_target": rollback_target,
        "release_bundle_target_matches_current": release_target_matches_current,
        "target_matches_current": {"manifest": True, "environment": True, "systemd_unit": True},
        "rebound_target": {
            "manifest_sha256": active_manifest_sha,
            "environment_sha256": current_env_sha,
            "systemd_unit_sha256": current_unit_sha,
        },
        "environment_rebind_reason": environment_reason,
        "environment_content_diff": env_diff(previous_env, env_snapshot),
        "files": {
            "manifest": file_identity(args.manifest, "active manifest", True),
            "environment": file_identity(args.environment, "gateway manifest-mode environment", True),
            "systemd_unit": file_identity(args.unit, "systemd unit", True),
            "systemd_dropin": file_identity(args.dropin, "systemd served-model drop-in"),
        },
    }
    return {
        "schema_version": "ullm.aq4_production_optimization_p0.v1",
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(aliased=True),
            "kernel": platform.release(),
            "python": platform.python_version(),
        },
        "git": git_snapshot(),
        "identity": manifest_identities(manifest, args.manifest),
        "active_manifest": scrub_manifest(manifest),
        "rollback_binding": rollback_binding,
        "service_topology": topology(),
        "hardware": {
            "gpu": safe_command(["rocm-smi", "--showproductname", "--showuniqueid", "--showdriverversion", "--showmeminfo", "vram", "--showuse", "--showpower", "--json"]),
            "power_condition": safe_command(["rocm-smi", "--showpower", "--showuse", "--json"]),
        },
        "measurement_policy": {
            "run_root_template": "benchmarks/results/YYYY-MM-DD/qwen35-9b-aq4-production-opt-v0.1/",
            "case_id_pattern": "{scope}-{phase}-{context_or_prompt}-m{requested_m}-{backend}-{gpu}",
            "warmup_runs": 2,
            "measured_runs": 10,
            "percentile_method": "linear_interpolation_rank_(n-1)*p",
            "max_trace_bytes": 4 * 1024 * 1024,
            "r9700_queue": {"device_selector": "ROCR_VISIBLE_DEVICES", "runtime_device_index": 1, "exclusive": True},
            "promotion_identity_fields": ["git.commit", "manifest", "worker", "package_manifest", "tokenizer", "driver", "gpu", "power_condition"],
        },
        "openwebui": openwebui_marker_state(active_manifest_sha),
        "decision": {
            "active_product_changed": False,
            "active_service_changed": False,
            "rollback_binding_reproducible": all(rollback_binding["target_matches_current"].values()),
            "p1_allowed": True,
        },
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    if path.exists():
        raise CaptureError(f"refusing to overwrite existing artifact: {path}")
    raw = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with temporary.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--environment", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--unit", type=Path, default=DEFAULT_UNIT)
    parser.add_argument("--dropin", type=Path, default=DEFAULT_DROPIN)
    parser.add_argument("--rollback-bundle", type=Path, default=DEFAULT_ROLLBACK_BUNDLE)
    parser.add_argument("--rollback-environment", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        snapshot = capture(args)
        atomic_json(args.output_dir / "p0-snapshot.json", snapshot)
        atomic_json(args.output_dir / "rollback-binding.json", snapshot["rollback_binding"])
        print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "rollback_status": snapshot["rollback_binding"]["status"]}))
        return 0
    except (CaptureError, OSError, ValueError) as error:
        print(f"P0 capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
