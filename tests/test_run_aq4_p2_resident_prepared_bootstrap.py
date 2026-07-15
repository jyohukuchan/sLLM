from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/run-aq4-p2-resident-prepared-bootstrap.py"
HISTORICAL = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
SPEC = importlib.util.spec_from_file_location("aq4_prepared_bootstrap", SOURCE)
assert SPEC and SPEC.loader
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    if path.exists():
        path.chmod(0o644)
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="ascii",
    )


def _prepared_v4(tmp_path: Path) -> Path:
    root = tmp_path / "prepared-v4"
    root.mkdir()
    for name in BOOTSTRAP.PREPARED_MEMBER_CONTRACT:
        shutil.copy2(HISTORICAL / name, root / name)
    fixture_index_path = root / "fixture-index.json"
    fixture_index = json.loads(fixture_index_path.read_bytes())
    fixture_index["cases"][0]["fixture_path"] = str(root / "fixture.json")
    _write_json(fixture_index_path, fixture_index)
    bundle = json.loads((HISTORICAL / "bundle.json").read_bytes())
    bundle["schema_version"] = BOOTSTRAP.BUNDLE_SCHEMA
    bundle["canonical_root"] = str(root)
    bundle["run_id"] = "prepared-v4-test"
    bundle["files"]["fixture-index.json"]["sha256"] = _sha(fixture_index_path)
    _write_json(root / "bundle.json", bundle)
    return root


def _command(root: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(SOURCE),
        "--expanded",
        str(root / "case-binding.json"),
        "--fixture-index",
        str(root / "fixture-index.json"),
        "--identity",
        str(root / "identity.json"),
        "--preflight",
        str(root / "preflight.json"),
        "--policy",
        str(root / "policy.json"),
        "--output-dir",
        str(output),
        "--run-id",
        "prepared-v4-test",
        "--baseline-kind",
        "active-production",
        "--one-case-smoke",
        "--dry-run",
    ]


def test_prepared_bootstrap_v4_emits_exact_non_promotable_plan(tmp_path: Path) -> None:
    root = _prepared_v4(tmp_path)
    output = tmp_path / "output"
    completed = subprocess.run(_command(root, output), text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert list(output.iterdir()) == [output / "resident-batch.plan.json"]
    plan = json.loads((output / "resident-batch.plan.json").read_bytes())
    assert set(plan) == {
        "schema_version",
        "status",
        "scope",
        "case_count",
        "warmup_runs",
        "measured_runs",
        "transaction_count",
        "prompt_tokens_across_transactions",
        "resident_model_loads",
        "baseline_identity",
        "links",
        "execution_mode",
        "smoke_only",
        "promotion_eligible",
        "validation",
    }
    assert plan["schema_version"] == "ullm.aq4_p2_resident_batch.v1"
    assert plan["status"] == "dry_run"
    assert plan["case_count"] == 1
    assert plan["transaction_count"] == 12
    assert plan["prompt_tokens_across_transactions"] == 1536
    assert plan["execution_mode"] == "one_case_smoke"
    assert plan["smoke_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["validation"]["mode"] == "validate_only"
    assert plan["validation"]["driver_fake_handshake"] == "passed"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema_version", "ullm.aq4_p2_resident_smoke_binding_bundle.v3", "schema differs"),
        ("schema_version", "ullm.aq4_p2_resident_smoke_binding_bundle.v999", "schema differs"),
        ("status", "ready_for_execute", "status differs"),
        ("promotion", True, "promotion differs"),
        ("promotion", 0, "promotion differs"),
    ),
)
def test_prepared_bootstrap_rejects_header_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    root = _prepared_v4(tmp_path)
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_bytes())
    bundle[field] = value
    _write_json(bundle_path, bundle)
    completed = subprocess.run(
        _command(root, tmp_path / "output"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("missing", ("--one-case-smoke", "--dry-run"))
def test_prepared_bootstrap_requires_preparation_only_flags(
    tmp_path: Path,
    missing: str,
) -> None:
    root = _prepared_v4(tmp_path)
    command = _command(root, tmp_path / "output")
    command.remove(missing)
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "requires --one-case-smoke and --dry-run" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_prepared_bootstrap_has_no_actual_execution_cli(tmp_path: Path) -> None:
    root = _prepared_v4(tmp_path)
    command = _command(root, tmp_path / "output")
    command.extend(["--driver-command", "/bin/false"])
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "unrecognized arguments: --driver-command" in completed.stderr
    assert not (tmp_path / "output").exists()


def test_prepared_bootstrap_rejects_member_hash_drift(tmp_path: Path) -> None:
    root = _prepared_v4(tmp_path)
    policy = json.loads((root / "policy.json").read_bytes())
    policy["status"] = "changed"
    _write_json(root / "policy.json", policy)
    completed = subprocess.run(
        _command(root, tmp_path / "output"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "member SHA differs: policy.json" in completed.stderr
    assert not (tmp_path / "output").exists()
