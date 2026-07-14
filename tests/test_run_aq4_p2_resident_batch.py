from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
DRIVER_IDENTITY = {
    "binary_sha256": "d" * 64,
    "build_git_commit": "e" * 40,
    "protocol": "ullm.aq4_p2_resident_driver.v2",
    "worker_binary_sha256": "c" * 64,
    "package_manifest_sha256": "f" * 64,
    "package_content_sha256": "1" * 64,
    "served_model_manifest_sha256": "b" * 64,
    "model_id": "Qwen3.5-9B-AQ4",
    "model_revision": "fixture-revision",
    "format_id": "AQ4_0",
    "implementation_id": "qwen35_aq4_rdna4_v1",
    "runtime_device": {
        "runtime_device_index": 1,
        "device_id": "r9700-rdna4",
        "backend": "hip",
        "name": "AMD Radeon Graphics",
        "architecture": "gfx1201",
    },
    "guard_set_sha256": "a" * 64,
}
SPEC = importlib.util.spec_from_file_location("aq4_resident_batch", ROOT / "tools/run-aq4-p2-resident-batch.py")
assert SPEC and SPEC.loader
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def _identity(driver_sha256: str | None = None) -> dict:
    value = json.loads(json.dumps(DRIVER_IDENTITY))
    if driver_sha256 is not None:
        value["binary_sha256"] = driver_sha256
    return value


def _detached_python(tmp_path: Path) -> tuple[Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = Path(sys.executable).resolve(strict=True)
    detached = tmp_path / "detached-python"
    shutil.copy2(source, detached)
    detached.chmod(source.stat().st_mode)
    return detached, BATCH.sha_file(detached, "detached driver", absolute=True)


def _case(tmp_path: Path, prompt: int, requested_m: int, mode: str, index: int) -> dict:
    case_id = f"p2-representative-full_model-cold_prefill-{mode}-n{prompt}-m{requested_m}-r9700-aq4_0_target-{index}"
    value = {
        "case_id": case_id,
        "fixture_id": case_id,
        "case_sha256": None,
        "stage_id": "representative",
        "stage_order": index + 1,
        "scope": "full_model",
        "phase": "cold_prefill",
        "mode": mode,
        "baseline_mode": mode,
        "prompt_tokens": prompt,
        "cached_prefix_tokens": 0,
        "context_tokens": prompt,
        "decode_start_tokens": 0,
        "prefill_requested_m": requested_m,
        "resolved_m": 1 if mode == "all_m1" else requested_m,
        "request_count": 1,
        "decode_request_count": 0,
        "generated_tokens": 0,
        "control_id": "aq4_0_target",
        "control": {"control_id": "aq4_0_target", "role": "target", "format_id": "AQ4_0", "implementation_id": "qwen35_aq4_rdna4_v1", "promotion_eligible": True},
        "sampling": {"mode": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
        "format_id": "AQ4_0",
        "implementation_id": "qwen35_aq4_rdna4_v1",
        "path_oracle_case_id": None if mode == "all_m1" else f"path-{index}",
        "path_oracle_result_sha256": None,
        "device": {
            "device_id": "r9700-rdna4",
            "runtime_device_index": 1,
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "architecture": "gfx1201",
        },
    }
    value["case_sha256"] = BATCH.case_hash(value)
    fixture = tmp_path / f"{case_id}.fixture.json"
    fixture.write_text(json.dumps({"cases": [{"case_id": case_id, "prompt_token_ids": [1] * prompt, "step_count": 0}]}), encoding="utf-8")
    return value, {"case_id": case_id, "case_sha256": value["case_sha256"], "fixture_path": str(fixture), "fixture_sha256": BATCH.sha_file(fixture, "fixture"), "prompt_tokens": prompt, "context_tokens": prompt, "generated_tokens": 0}


def _bundle(tmp_path: Path, driver_sha256: str | None = None) -> tuple[Path, Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    expanded_path = tmp_path / "expanded.json"
    fixture_index_path = tmp_path / "fixture-index.json"
    identity_path = tmp_path / "identity.json"
    preflight_path = tmp_path / "preflight.json"
    policy_path = tmp_path / "policy.json"
    cases, entries = [], []
    index = 0
    for prompt in (128, 512, 1011, 1024, 1339, 2048, 3584):
        for requested_m in (1, 8, 16, 32, 64, 128):
            for mode in ("all_m1", "cold_batched"):
                case, entry = _case(tmp_path, prompt, requested_m, mode, index)
                cases.append(case); entries.append(entry); index += 1
    expanded_path.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_expanded.v2", "cases": cases}), encoding="utf-8")
    fixture_index_path.write_text(json.dumps({"schema_version": "ullm.aq4_p2_fixture_index.v1", "case_count": len(entries), "cases": entries}), encoding="utf-8")
    expanded_sha = BATCH.sha_file(expanded_path, "expanded")
    resident_identity = _identity(driver_sha256)
    identity_document = {"schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": None, "expanded_manifest_sha256": expanded_sha, "build_git_commit": resident_identity["build_git_commit"], "resident_driver_identity": resident_identity, "hash_binding": {"bound_case_manifest_sha256": expanded_sha, "served_model_manifest_sha256": resident_identity["served_model_manifest_sha256"], "worker_binary_sha256": resident_identity["worker_binary_sha256"], "package_manifest_sha256": resident_identity["package_manifest_sha256"], "package_content_sha256": resident_identity["package_content_sha256"]}}
    identity_document["identity_sha256"] = BATCH.sha_bytes(BATCH.canonical(identity_document))
    identity_path.write_text(json.dumps(identity_document), encoding="utf-8")
    preflight_path.write_text(json.dumps({"weights_bytes": 1, "persistent_state_bytes": 1, "kv_cache_bytes": 1, "workspace_bytes": 1, "temporary_bytes": 1, "vram_headroom_bytes": 1, "gpu_process_snapshot": []}), encoding="utf-8")
    policy_path.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}), encoding="utf-8")
    return expanded_path, fixture_index_path, identity_path, preflight_path, policy_path


def _one_case_bundle(tmp_path: Path, driver_sha256: str | None = None, *, case_count: int = 1) -> tuple[Path, Path, Path, Path, Path]:
    expanded, fixture_index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    expanded_value = json.loads(expanded.read_text())
    selected = expanded_value["cases"][:case_count]
    case_binding = tmp_path / "case-binding.json"
    case_binding_value = {
        "schema_version": "ullm.aq4_production_p2_expanded.v2",
        "status": "bound_one_case_smoke",
        "source_manifest_sha256": "9" * 64,
        "official_case_sha256": "8" * 64,
        "runtime_binding": {"schema_version": "fixture.v1"},
        "case_count": len(selected),
        "canonical_case_sha256": BATCH.sha_bytes(BATCH.canonical(selected)),
        "cases": selected,
    }
    case_binding.write_text(json.dumps(case_binding_value), encoding="utf-8")
    binding_sha = BATCH.sha_file(case_binding, "case binding")
    index_value = json.loads(fixture_index.read_text())
    selected_ids = {case["case_id"] for case in selected}
    index_value.update({"expanded_manifest_sha256": binding_sha, "served_model_manifest_sha256": "b" * 64, "subset": "resident_one_case_smoke", "case_count": len(selected)})
    index_value["cases"] = [entry for entry in index_value["cases"] if entry["case_id"] in selected_ids]
    fixture_index.write_text(json.dumps(index_value), encoding="utf-8")
    identity_value = json.loads(identity.read_text())
    identity_value["expanded_manifest_sha256"] = binding_sha
    identity_value["hash_binding"]["bound_case_manifest_sha256"] = binding_sha
    identity_value["identity_sha256"] = None
    identity_value["identity_sha256"] = BATCH.sha_bytes(BATCH.canonical(identity_value))
    identity.write_text(json.dumps(identity_value), encoding="utf-8")
    case_sha = selected[0]["case_sha256"] if len(selected) == 1 else "0" * 64
    bundle = {
        "schema_version": "ullm.aq4_p2_resident_smoke_binding_bundle.v3",
        "status": "prepared_not_executed",
        "promotion": False,
        "bindings": {"case_binding_sha256": binding_sha, "case_sha256": case_sha},
        "files": {"case-binding.json": {"sha256": binding_sha, "role": "runtime_bound_case"}},
    }
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    resident_identity = identity_value["resident_driver_identity"]
    fake_ready = {"event": "ready", "schema_version": "ullm.aq4_p2_resident_driver.v2", "model_loads": 1, "resident_session_id": "fake-validate-only", "driver_identity": resident_identity}
    (tmp_path / "fake-ready.json").write_text(json.dumps(fake_ready), encoding="utf-8")
    return case_binding, fixture_index, identity, preflight, policy


def _driver(tmp_path: Path, driver_sha256: str, oom: bool = False, reset_bad: bool = False, drift: str | None = None) -> Path:
    suffix = drift or ("oom" if oom else "reset-bad" if reset_bad else "ok")
    path = tmp_path / f"fake-{suffix}-driver.py"
    path.write_text(
        """import json,sys
oom = %r
reset_bad = %r
drift = %r
identity = %r
if drift == 'driver': identity['binary_sha256'] = '0' * 64
if drift == 'package': identity['package_manifest_sha256'] = '0' * 64
if drift == 'device': identity['runtime_device']['architecture'] = 'gfx9999'
session = 'fake-session'
case_id = None
requested = 1
resolved = 1
print(json.dumps({'event':'ready','schema_version':'ullm.aq4_p2_resident_driver.v2','model_loads':1,'resident_session_id':session,'driver_identity':identity}), flush=True)
for line in sys.stdin:
    msg=json.loads(line)
    if msg['command']=='case_begin':
        case_id=msg['case_id']; requested=msg['execution']['requested_m']; resolved=msg['execution']['resolved_m']
        print(json.dumps({'event':'case_ready','schema_version':'ullm.aq4_p2_resident_driver.v2','resident_session_id':session,'case_id':'wrong-case' if drift=='case_swap' else case_id,'requested_m':requested,'resolved_m':resolved,'baseline_clean':True}), flush=True)
    elif msg['command']=='run':
        status='oom' if oom and msg['run_index']==2 else 'ok'
        reset={'attempted':1,'complete':1,'failed':0}
        if reset_bad and msg['run_index']==3: reset={'attempted':1,'complete':0,'failed':1}
        terminal_status = status == 'oom' or reset['failed'] == 1
        out={'event':'run_complete','schema_version':'ullm.aq4_p2_resident_driver.v2','resident_session_id':session,'case_id':case_id,'run_index':msg['run_index'],'run_kind':msg['run_kind'],'status':status if not reset_bad or reset['failed']==0 else 'failed','elapsed_ms':1.0,'requested_m':requested,'resolved_m':resolved,'actual_token_batch_width':resolved if status=='ok' else None,'actual_request_batch_width':1 if status=='ok' else None,'timing':{'prefill_ms':1.0,'decode_ms':0.0,'end_to_end_ms':1.0,'generated_tokens':0} if status=='ok' else None,'audit':{'coverage_complete':True,'deterministic_digest_sha256':'d'*64} if status=='ok' else None,'state':{'baseline_before':True,'baseline_after':reset['complete']==1,'request_state_sha256':'e'*64} if status=='ok' else None,'lifecycle':{'prepare':1,'commit':1 if status=='ok' else 0,'discard':0 if status=='ok' else 1,'error':0 if status=='ok' else 1,'cancel':0,'reset':reset},'reset':reset,'resource':{'samples':[{'monotonic_ms':1}],'peak':{'vram_used_bytes':1}},'terminal':{'reuse_forbidden':terminal_status,'reason_code':'runtime_out_of_memory' if status=='oom' else 'reset_failed' if reset['failed'] else 'none','oom':status=='oom','hip_fault':False}}
        if drift == 'result_order': out['run_index'] += 1
        print(json.dumps(out), flush=True)
        if terminal_status: break
    elif msg['command']=='case_end': print(json.dumps({'event':'case_complete','schema_version':'ullm.aq4_p2_resident_driver.v2','resident_session_id':session,'case_id':case_id,'release':{'commit':1,'discard':0,'reset':1,'baseline_restored':False if drift=='release' else True}}), flush=True)
    elif msg['command']=='shutdown': break
""" % (oom, reset_bad, drift, _identity(driver_sha256)),
        encoding="utf-8",
    )
    return path


def _live_command(tmp_path: Path, output: Path, expanded: Path, index: Path, identity: Path, preflight: Path, policy: Path, python: Path, driver: Path, run_id: str) -> list[str]:
    return [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", run_id, "--baseline-kind", "p3-current-head", "--lock-path", str(tmp_path / "r9700.lock"), "--driver-command", str(python), str(driver)]


def test_dry_run_selects_exact_84_target_cases_and_separates_baseline(tmp_path: Path) -> None:
    expanded, index, identity, preflight, policy = _bundle(tmp_path)
    output = tmp_path / "dry-run"
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", "r-active", "--baseline-kind", "active-production", "--dry-run"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output / "resident-batch.plan.json").read_text())
    assert plan["case_count"] == 84
    assert plan["transaction_count"] == 84 * 12
    assert plan["prompt_tokens_across_transactions"] == 1_389_024
    assert plan["resident_model_loads"] == 1
    assert plan["baseline_identity"]["kind"] == "active-production"


def test_one_case_smoke_dry_run_executes_bundle_v3_fake_handshake(tmp_path: Path) -> None:
    expanded, index, identity, preflight, policy = _one_case_bundle(tmp_path)
    output = tmp_path / "one-case-dry-run"
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", "one-case", "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output / "resident-batch.plan.json").read_text())
    assert plan["case_count"] == 1
    assert plan["transaction_count"] == 12
    assert plan["warmup_runs"] == 2
    assert plan["measured_runs"] == 10
    assert plan["execution_mode"] == "one_case_smoke"
    assert plan["smoke_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["validation"]["mode"] == "validate_only"
    assert plan["validation"]["driver_fake_handshake"] == "passed"
    assert plan["validation"]["resident_session_id"] == "fake-validate-only"


@pytest.mark.parametrize("case_count", (0, 2))
def test_one_case_smoke_rejects_zero_or_two_target_cases(tmp_path: Path, case_count: int) -> None:
    expanded, index, identity, preflight, policy = _one_case_bundle(tmp_path, case_count=case_count)
    output = tmp_path / "invalid-count"
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", "invalid", "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "must contain exactly 1 target cases" in completed.stderr
    assert not output.exists()


def test_one_case_smoke_rejects_bundle_case_swap_and_normal_mode_still_requires_84(tmp_path: Path) -> None:
    expanded, index, identity, preflight, policy = _one_case_bundle(tmp_path)
    bundle_path = tmp_path / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["bindings"]["case_sha256"] = "0" * 64
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    base = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--run-id", "case-swap", "--baseline-kind", "active-production", "--dry-run"]
    swapped = subprocess.run([*base, "--output-dir", str(tmp_path / "swapped"), "--one-case-smoke"], text=True, capture_output=True)
    assert swapped.returncode != 0
    assert "case binding/hash differs" in swapped.stderr
    ordinary = subprocess.run([*base, "--output-dir", str(tmp_path / "ordinary")], text=True, capture_output=True)
    assert ordinary.returncode != 0
    assert "must contain exactly 84 target cases" in ordinary.stderr


def test_one_case_smoke_fake_driver_runs_exact_two_plus_ten_and_stays_nonpromotion(tmp_path: Path) -> None:
    python, driver_sha256 = _detached_python(tmp_path)
    expanded, index, identity, preflight, policy = _one_case_bundle(tmp_path, driver_sha256)
    output = tmp_path / "one-case-run"
    driver = _driver(tmp_path, driver_sha256)
    command = [*_live_command(tmp_path, output, expanded, index, identity, preflight, policy, python, driver, "one-case-live"), "--one-case-smoke"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 1
    raw = json.loads(raws[0].read_text())
    assert raw["schedule"] == {"warmup_runs": 2, "measured_runs": 10, "completed_runs": 12}
    assert raw["execution_mode"] == "one_case_smoke"
    assert raw["smoke_only"] is True
    assert raw["promotion_eligible"] is False
    summary = json.loads((output / "resident-batch.summary.json").read_text())
    assert summary["completed_cases"] == 1
    assert summary["transaction_count"] == 12
    assert summary["smoke_only"] is True
    assert summary["promotion_eligible"] is False


def test_fake_resident_driver_writes_one_atomic_raw_per_case(tmp_path: Path) -> None:
    python, driver_sha256 = _detached_python(tmp_path)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "run"
    driver = _driver(tmp_path, driver_sha256)
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, python, driver, "r-current")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 84
    sample = json.loads(raws[0].read_text())
    assert sample["status"] == "ok"
    assert sample["resident"] == {"session_id": "fake-session", "model_loads": 1, "driver_identity": _identity(driver_sha256), "case_reset_count": 12}
    assert sample["device_lock"]["driver"]["sha256"] == driver_sha256
    assert sample["schedule"] == {"warmup_runs": 2, "measured_runs": 10, "completed_runs": 12}
    assert json.loads((output / "resident-batch.summary.json").read_text())["completed_cases"] == 84


def test_resident_oom_is_immutable_and_aborts_remaining_cases(tmp_path: Path) -> None:
    python, driver_sha256 = _detached_python(tmp_path)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "oom-run"
    driver = _driver(tmp_path, driver_sha256, oom=True)
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, python, driver, "r-current")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 1
    assert json.loads(raws[0].read_text())["status"] == "oom"


def test_incomplete_reset_is_immutable_and_aborts_process_reuse(tmp_path: Path) -> None:
    python, driver_sha256 = _detached_python(tmp_path)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "reset-bad-run"
    driver = _driver(tmp_path, driver_sha256, reset_bad=True)
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, python, driver, "r-current")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 1
    value = json.loads(raws[0].read_text())
    assert value["status"] == "failed"
    assert value["immutable_status"] is True
    assert value["runs"][-1]["terminal"]["reuse_forbidden"] is True


def test_ready_self_sha_and_identity_drift_are_rejected_before_case_begin(tmp_path: Path) -> None:
    for drift in ("driver", "package", "device"):
        python, driver_sha256 = _detached_python(tmp_path / drift)
        expanded, index, identity, preflight, policy = _bundle(tmp_path / drift, driver_sha256)
        output = tmp_path / f"{drift}-drift-run"
        driver = _driver(tmp_path / drift, driver_sha256, drift=drift)
        command = _live_command(tmp_path / drift, output, expanded, index, identity, preflight, policy, python, driver, f"r-{drift}")
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode != 0
        assert list(output.glob("*.raw.json")) == []


def test_case_swap_result_order_and_release_drift_are_rejected(tmp_path: Path) -> None:
    for drift in ("case_swap", "result_order", "release"):
        python, driver_sha256 = _detached_python(tmp_path / drift)
        expanded, index, identity, preflight, policy = _bundle(tmp_path / drift, driver_sha256)
        output = tmp_path / f"{drift}-run"
        driver = _driver(tmp_path / drift, driver_sha256, drift=drift)
        command = _live_command(tmp_path / drift, output, expanded, index, identity, preflight, policy, python, driver, f"r-{drift}")
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode != 0
        assert list(output.glob("*.raw.json")) == []


def test_cargo_style_driver_hardlink_is_rejected_and_detached_copy_is_accepted(tmp_path: Path) -> None:
    detached, digest = _detached_python(tmp_path)
    cargo_link = tmp_path / "cargo-deps-hardlink"
    os.link(detached, cargo_link)
    bound = {"resident_driver_identity": {"binary_sha256": digest}}
    with pytest.raises(BATCH.BatchError, match="single-link"):
        BATCH.validate_driver_command([str(detached)], bound)
    accepted = tmp_path / "accepted-detached-driver"
    shutil.copy2(detached, accepted)
    evidence = BATCH.validate_driver_command([str(accepted)], bound)
    assert evidence["sha256"] == digest
    assert evidence["nlink"] == 1


def test_lock_contention_fails_before_spawn_and_exception_releases_lock(tmp_path: Path) -> None:
    python, driver_sha256 = _detached_python(tmp_path)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    driver = _driver(tmp_path, driver_sha256, oom=True)
    lock_path = tmp_path / "r9700.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    blocked_output = tmp_path / "blocked-output"
    blocked = subprocess.run(_live_command(tmp_path, blocked_output, expanded, index, identity, preflight, policy, python, driver, "blocked"), text=True, capture_output=True)
    assert blocked.returncode != 0
    assert not blocked_output.exists()
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)

    failed_output = tmp_path / "failed-output"
    failed = subprocess.run(_live_command(tmp_path, failed_output, expanded, index, identity, preflight, policy, python, driver, "oom-cleanup"), text=True, capture_output=True)
    assert failed.returncode != 0
    owner = json.loads((failed_output / "resident-batch.lock-owner.json").read_text())
    assert owner["driver"]["sha256"] == driver_sha256
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_fixture_index_rejects_relative_and_symlink_parent_paths(tmp_path: Path) -> None:
    for variant in ("relative", "symlink-parent"):
        root = tmp_path / variant
        expanded, index, identity, preflight, policy = _bundle(root)
        document = json.loads(index.read_text())
        original = Path(document["cases"][0]["fixture_path"])
        if variant == "relative":
            document["cases"][0]["fixture_path"] = original.name
        else:
            alias = tmp_path / f"{variant}-alias"
            alias.symlink_to(root, target_is_directory=True)
            document["cases"][0]["fixture_path"] = str(alias / original.name)
        index.write_text(json.dumps(document), encoding="utf-8")
        output = root / "invalid-fixture-output"
        command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", variant, "--baseline-kind", "p3-current-head", "--dry-run"]
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode != 0
        assert not output.exists()
