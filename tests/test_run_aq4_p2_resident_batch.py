from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_resident_batch", ROOT / "tools/run-aq4-p2-resident-batch.py")
assert SPEC and SPEC.loader
BATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BATCH)


def _case(tmp_path: Path, prompt: int, requested_m: int, mode: str, index: int) -> dict:
    case_id = f"p2-representative-full_model-cold_prefill-{mode}-n{prompt}-m{requested_m}-r9700-aq4_0_target-{index}"
    value = {
        "case_id": case_id,
        "case_sha256": None,
        "stage_id": "representative",
        "scope": "full_model",
        "phase": "cold_prefill",
        "mode": mode,
        "prompt_tokens": prompt,
        "cached_prefix_tokens": 0,
        "context_tokens": prompt,
        "prefill_requested_m": requested_m,
        "resolved_m": 1 if mode == "all_m1" else requested_m,
        "request_count": 1,
        "generated_tokens": 0,
        "control_id": "aq4_0_target",
        "device": {"device_id": "r9700-rdna4"},
    }
    value["case_sha256"] = BATCH.case_hash(value)
    fixture = tmp_path / f"{case_id}.fixture.json"
    fixture.write_text(json.dumps({"cases": [{"case_id": case_id, "prompt_token_ids": [1] * prompt, "step_count": 0}]}), encoding="utf-8")
    return value, {"case_id": case_id, "case_sha256": value["case_sha256"], "fixture_path": str(fixture), "fixture_sha256": BATCH.sha_file(fixture, "fixture"), "prompt_tokens": prompt, "context_tokens": prompt, "generated_tokens": 0}


def _bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    expanded_path = tmp_path / "expanded.json"
    fixture_index_path = tmp_path / "fixture-index.json"
    identity_path = tmp_path / "identity.json"
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
    identity_path.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "build_git_commit": "a" * 40, "hash_binding": {"served_model_manifest_sha256": "b" * 64, "worker_binary_sha256": "c" * 64}}), encoding="utf-8")
    policy_path.write_text(json.dumps({"schema_version": "ullm.aq4_production_p2_threshold_policy.v1", "status": "bound"}), encoding="utf-8")
    return expanded_path, fixture_index_path, identity_path, policy_path


def _driver(tmp_path: Path, oom: bool = False, reset_bad: bool = False) -> Path:
    suffix = "oom" if oom else "reset-bad" if reset_bad else "ok"
    path = tmp_path / f"fake-{suffix}-driver.py"
    path.write_text(
        """import json,sys
oom = %r
reset_bad = %r
session = 'fake-session'
resolved = 1
print(json.dumps({'event':'ready','schema_version':'ullm.aq4_p2_resident_driver.v1','model_loads':1,'resident_session_id':session}), flush=True)
for line in sys.stdin:
    msg=json.loads(line)
    if msg['command']=='case_begin': resolved=msg['resolved_m']; print(json.dumps({'event':'case_ready','resident_session_id':session}), flush=True)
    elif msg['command']=='run':
        status='oom' if oom and msg['run_index']==2 else 'ok'
        reset={'attempted':1,'complete':1,'failed':0}
        if reset_bad and msg['run_index']==3: reset={'attempted':1,'complete':0,'failed':1}
        out={'event':'run_complete','resident_session_id':session,'status':status,'elapsed_ms':1.0,'reset':reset}
        if status=='ok': out.update({'actual_token_batch_width':resolved,'actual_request_batch_width':1,'audit':{'coverage_complete':True,'deterministic_digest_sha256':'d'*64},'resource':{'samples':[{'monotonic_ms':1}],'peak':{'vram_used_bytes':1}}})
        print(json.dumps(out), flush=True)
    elif msg['command']=='case_end': print(json.dumps({'event':'case_complete','resident_session_id':session}), flush=True)
    elif msg['command']=='shutdown': break
""" % (oom, reset_bad),
        encoding="utf-8",
    )
    return path


def test_dry_run_selects_exact_84_target_cases_and_separates_baseline(tmp_path: Path) -> None:
    expanded, index, identity, policy = _bundle(tmp_path)
    output = tmp_path / "dry-run"
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", "r-active", "--baseline-kind", "active-production", "--dry-run"]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((output / "resident-batch.plan.json").read_text())
    assert plan["case_count"] == 84
    assert plan["transaction_count"] == 84 * 12
    assert plan["prompt_tokens_across_transactions"] == 1_389_024
    assert plan["resident_model_loads"] == 1
    assert plan["baseline_identity"]["kind"] == "active-production"


def test_fake_resident_driver_writes_one_atomic_raw_per_case(tmp_path: Path) -> None:
    expanded, index, identity, policy = _bundle(tmp_path)
    output = tmp_path / "run"
    driver = _driver(tmp_path)
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", "r-current", "--baseline-kind", "p3-current-head", "--driver-command", sys.executable, str(driver)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 84
    sample = json.loads(raws[0].read_text())
    assert sample["status"] == "ok"
    assert sample["resident"] == {"session_id": "fake-session", "model_loads": 1, "case_reset_count": 12}
    assert sample["schedule"] == {"warmup_runs": 2, "measured_runs": 10, "completed_runs": 12}
    assert json.loads((output / "resident-batch.summary.json").read_text())["completed_cases"] == 84


def test_resident_oom_is_immutable_and_aborts_remaining_cases(tmp_path: Path) -> None:
    expanded, index, identity, policy = _bundle(tmp_path)
    output = tmp_path / "oom-run"
    driver = _driver(tmp_path, oom=True)
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", "r-current", "--baseline-kind", "p3-current-head", "--driver-command", sys.executable, str(driver)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 1
    assert json.loads(raws[0].read_text())["status"] == "oom"


def test_incomplete_reset_is_rejected_before_raw_publication(tmp_path: Path) -> None:
    expanded, index, identity, policy = _bundle(tmp_path)
    output = tmp_path / "reset-bad-run"
    driver = _driver(tmp_path, reset_bad=True)
    command = [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--policy", str(policy), "--output-dir", str(output), "--run-id", "r-current", "--baseline-kind", "p3-current-head", "--driver-command", sys.executable, str(driver)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert list(output.glob("*.raw.json")) == []
