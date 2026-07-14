from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
SPEC = importlib.util.spec_from_file_location("aq4_resident_batch_live", ROOT / "tools/run-aq4-p2-resident-batch.py")
assert SPEC and SPEC.loader
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def _args(run_id: str = "live-run") -> argparse.Namespace:
    return argparse.Namespace(run_id=run_id, bundle_root=BUNDLE_ROOT, lock_path=Path("/run/ullm/r9700.lock"))


def _document(run_id: str = "live-run") -> dict:
    bundle = json.loads((BUNDLE_ROOT / "bundle.json").read_text())
    environment = (
        bundle["expected_runtime"]["environment"]
        | bundle["expected_runtime"]["required_guards"]
        | {
            "ULLM_SERVED_MODEL_MANIFEST": "/etc/ullm/served-models/active.json",
            "ULLM_BUILD_GIT_COMMIT": bundle["resident_driver"]["source_commit"],
        }
    )
    commands = {
        "sudo-n": (["/usr/bin/sudo", "-n", "-v"], 0),
        "service-ullm-openai.service": (["/usr/bin/systemctl", "show", "ullm-openai.service", "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], 0),
        "service-llama-qwen35-udq4.service": (["/usr/bin/systemctl", "show", "llama-qwen35-udq4.service", "--property=ActiveState", "--property=SubState", "--property=MainPID", "--no-pager"], 0),
        "old-worker": (["/usr/bin/pgrep", "-f", "-x", f"{ROOT / 'target/reasoning-v2/release/ullm-aq4-worker'}.*"], 1),
        "amd-smi-list": (["/opt/rocm/bin/amd-smi", "list", "--json"], 0),
        "rocminfo": (["/usr/bin/rocminfo"], 0),
        "amd-smi-process": (["/opt/rocm/bin/amd-smi", "process", "--gpu", "2", "--general", "--json"], 0),
        "amd-smi-static-vram": (["/opt/rocm/bin/amd-smi", "static", "--gpu", "2", "--vram", "--json"], 0),
    }
    return {
        "schema_version": "ullm.aq4_p2_resident_live_preflight.v1",
        "status": "passed",
        "run_id": run_id,
        "captured_unix_ns": 1,
        "prepared_preflight": {
            "path": str(BUNDLE_ROOT / "preflight.json"),
            "sha256": BATCH.sha_file(BUNDLE_ROOT / "preflight.json", "prepared"),
            "role": "synthetic_bundle_contract_only",
        },
        "runtime_mapping": {
            "runtime_device_index": 1,
            "visible_token": "1",
            "amd_smi_index": 2,
            "bdf": "0000:47:00.0",
            "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
            "kfd_id": 51545,
            "node_id": 2,
        },
        "services": [
            {"unit": "ullm-openai.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
            {"unit": "llama-qwen35-udq4.service", "active_state": "inactive", "sub_state": "dead", "main_pid": 0},
        ],
        "worker_pids": [],
        "compute_owners": {"amd_smi": [], "kfd": []},
        "lock": {"path": "/run/ullm/r9700.lock", "free": True, "device": 66306, "inode": 1},
        "environment": environment,
        "vram": {"total_bytes": 32_624_000_000, "used_bytes": 0, "free_bytes": 32_624_000_000, "headroom_bytes": 32_624_000_000},
        "commands": [
            {"label": label, "argv": argv, "exit_code": exit_code, "stdout_sha256": "0" * 64, "stderr_sha256": "0" * 64, "captured_unix_ns": index}
            for index, (label, (argv, exit_code)) in enumerate(commands.items())
        ],
    }


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    path.chmod(0o444)
    return path


def test_valid_live_preflight_is_bound_by_hash_and_identity(tmp_path: Path) -> None:
    path = _write(tmp_path / "live.json", _document())
    link = BATCH.validate_live_preflight(path, _args(), json.loads((BUNDLE_ROOT / "bundle.json").read_text()))
    assert link["path"] == str(path)
    assert link["device"] == path.stat().st_dev
    assert link["inode"] == path.stat().st_ino
    BATCH.verify_live_preflight(path, link)


@pytest.mark.parametrize(
    ("variant", "mutate"),
    [
        ("reused-run", lambda value: value.update(run_id="a-prior-run")),
        ("swapped-prepared", lambda value: value["prepared_preflight"].update(sha256="1" * 64)),
        ("identity", lambda value: value["runtime_mapping"].update(uuid="wrong")),
        ("amd-owner", lambda value: value["compute_owners"].update(amd_smi=[123])),
        ("kfd-owner", lambda value: value["compute_owners"].update(kfd=[456])),
    ],
)
def test_live_preflight_rejects_reuse_swap_identity_and_owners(tmp_path: Path, variant: str, mutate) -> None:
    value = _document()
    mutate(value)
    path = _write(tmp_path / f"{variant}.json", value)
    with pytest.raises(BATCH.BatchError):
        BATCH.validate_live_preflight(path, _args(), json.loads((BUNDLE_ROOT / "bundle.json").read_text()))


def test_live_preflight_rejects_missing_symlink_and_nonregular(tmp_path: Path) -> None:
    bundle = json.loads((BUNDLE_ROOT / "bundle.json").read_text())
    with pytest.raises(BATCH.BatchError):
        BATCH.validate_live_preflight(tmp_path / "missing.json", _args(), bundle)
    target = _write(tmp_path / "target.json", _document())
    symlink = tmp_path / "link.json"
    symlink.symlink_to(target)
    with pytest.raises(BATCH.BatchError):
        BATCH.validate_live_preflight(symlink, _args(), bundle)
    directory = tmp_path / "directory.json"
    directory.mkdir()
    with pytest.raises(BATCH.BatchError):
        BATCH.validate_live_preflight(directory, _args(), bundle)


def test_live_preflight_rejects_mutation_after_initial_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path / "live.json", _document())
    original = BATCH.sha_file
    mutated = False

    def mutate_then_hash(candidate: Path, label: str, *args, **kwargs):
        nonlocal mutated
        if label == "live preflight" and not mutated:
            mutated = True
            candidate.chmod(0o644)
            with candidate.open("ab") as handle:
                handle.write(b" ")
            candidate.chmod(0o444)
        return original(candidate, label, *args, **kwargs)

    monkeypatch.setattr(BATCH, "sha_file", mutate_then_hash)
    with pytest.raises(BATCH.BatchError, match="changed during validation"):
        BATCH.validate_live_preflight(path, _args(), json.loads((BUNDLE_ROOT / "bundle.json").read_text()))


def test_live_preflight_rejects_late_replacement(tmp_path: Path) -> None:
    path = _write(tmp_path / "live.json", _document())
    link = BATCH.validate_live_preflight(path, _args(), json.loads((BUNDLE_ROOT / "bundle.json").read_text()))
    replacement = _write(tmp_path / "replacement.json", _document())
    os.replace(replacement, path)
    with pytest.raises(BATCH.BatchError, match="identity changed"):
        BATCH.verify_live_preflight(path, link)


@pytest.mark.parametrize(
    ("variant", "mutate"),
    [
        ("mapping-unknown", lambda value: value["runtime_mapping"].update(unknown=1)),
        ("mapping-node", lambda value: value["runtime_mapping"].update(node_id=3)),
        ("lock-unknown", lambda value: value["lock"].update(unknown=1)),
        ("lock-negative-device", lambda value: value["lock"].update(device=-1)),
        ("lock-negative-inode", lambda value: value["lock"].update(inode=-1)),
        ("vram-unknown", lambda value: value["vram"].update(unknown=1)),
        ("vram-too-small", lambda value: value["vram"].update(total_bytes=1, free_bytes=1, headroom_bytes=1)),
        ("vram-used", lambda value: value["vram"].update(used_bytes=1, free_bytes=value["vram"]["total_bytes"] - 1, headroom_bytes=value["vram"]["total_bytes"] - 1)),
        ("probe-unknown-field", lambda value: value["commands"][0].update(unknown=1)),
        ("probe-unknown-label", lambda value: value["commands"][0].update(label="unknown")),
        ("probe-duplicate-label", lambda value: value["commands"][1].update(label="sudo-n", argv=["/usr/bin/sudo", "-n", "-v"])),
        ("probe-argv", lambda value: value["commands"][0].update(argv=["/usr/bin/sudo", "-v"])),
        ("probe-success-exit", lambda value: value["commands"][0].update(exit_code=1)),
        ("probe-sha", lambda value: value["commands"][0].update(stdout_sha256="A" * 64)),
        ("probe-time", lambda value: value["commands"][0].update(captured_unix_ns=-1)),
    ],
)
def test_live_preflight_rejects_qa_nested_schema_negatives(tmp_path: Path, variant: str, mutate) -> None:
    value = _document()
    mutate(value)
    path = _write(tmp_path / f"{variant}.json", value)
    with pytest.raises(BATCH.BatchError):
        BATCH.validate_live_preflight(path, _args(), json.loads((BUNDLE_ROOT / "bundle.json").read_text()))
