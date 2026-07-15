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
TRUSTED_ONE_CASE_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
TRUSTED_BUNDLE_VALIDATOR = ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py"
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


def _identity(
    driver_sha256: str | None = None,
) -> dict:
    value = json.loads(json.dumps(DRIVER_IDENTITY))
    if driver_sha256 is not None:
        value["binary_sha256"] = driver_sha256
    return value


def _legacy_served_binding(
    manifest_path: Path | str = "/served-model.json",
    sha256: str = "b" * 64,
) -> dict:
    return {
        "schema_version": BATCH.SERVED_MODEL_BINDING_SCHEMA,
        "mode": "logical_path",
        "logical_path": str(manifest_path),
        "effective_source": "path_loader",
        "descriptor_transport": "none",
        "closure": "control_input",
        "method": "read",
        "identity": {
            "device": 1,
            "inode": 1,
            "mode": 0o100444,
            "nlink": 1,
            "size": 0,
            "mtime_ns": 1,
            "ctime_ns": 1,
        },
        "sha256": sha256,
        "byte_count": 0,
        "single_read": True,
        "logical_path_opened": True,
    }


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


def _one_case_command(output: Path, *, bundle_root: Path = TRUSTED_ONE_CASE_ROOT) -> list[str]:
    expanded, index, identity, preflight, policy = tuple(bundle_root / name for name in ("case-binding.json", "fixture-index.json", "identity.json", "preflight.json", "policy.json"))
    validator_sha256 = BATCH.sha_file(TRUSTED_BUNDLE_VALIDATOR, "trusted bundle validator", absolute=True)
    return [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--bundle-root", str(bundle_root), "--trusted-validator", str(TRUSTED_BUNDLE_VALIDATOR), "--trusted-validator-sha256", validator_sha256, "--output-dir", str(output), "--run-id", "one-case", "--baseline-kind", "active-production", "--one-case-smoke", "--dry-run"]


def _rewrite_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _rebound_bundle(tmp_path: Path, variant: str) -> Path:
    root = tmp_path / variant
    shutil.copytree(TRUSTED_ONE_CASE_ROOT, root)
    for path in root.iterdir():
        path.chmod(0o755 if path.name == "resident-driver" else 0o644)
    case_binding = json.loads((root / "case-binding.json").read_text())
    fixture_index = json.loads((root / "fixture-index.json").read_text())
    identity = json.loads((root / "identity.json").read_text())
    fake_ready = json.loads((root / "fake-ready.json").read_text())
    launch_command = json.loads((root / "launch-command.json").read_text())
    launch_command["resident_driver_argv"][0] = str(root / "resident-driver")
    _rewrite_json(root / "launch-command.json", launch_command)
    fixture_index["cases"][0]["fixture_path"] = str(root / "fixture.json")
    if variant == "case":
        case = case_binding["cases"][0]
        case["case_id"] += "-rebound"
        case["fixture_id"] = case["case_id"]
        case["case_sha256"] = BATCH.case_hash(case)
        case_binding["canonical_case_sha256"] = BATCH.sha_bytes(BATCH.canonical(case_binding["cases"]))
        fixture_index["cases"][0]["case_id"] = case["case_id"]
        fixture_index["cases"][0]["case_sha256"] = case["case_sha256"]
    if variant == "identity":
        identity["resident_driver_identity"]["model_revision"] = "rebound-identity"
        fake_ready["driver_identity"] = identity["resident_driver_identity"]
    if variant == "fake_ready":
        fake_ready["resident_session_id"] = "swapped-session"
    _rewrite_json(root / "case-binding.json", case_binding)
    case_binding_sha = BATCH.sha_file(root / "case-binding.json", "rebound case binding")
    fixture_index["expanded_manifest_sha256"] = case_binding_sha
    _rewrite_json(root / "fixture-index.json", fixture_index)
    identity["expanded_manifest_sha256"] = case_binding_sha
    identity["hash_binding"]["bound_case_manifest_sha256"] = case_binding_sha
    identity["identity_sha256"] = None
    identity["identity_sha256"] = BATCH.sha_bytes(BATCH.canonical(identity))
    _rewrite_json(root / "identity.json", identity)
    _rewrite_json(root / "fake-ready.json", fake_ready)
    bundle = json.loads((root / "bundle.json").read_text())
    bundle["canonical_root"] = str(root)
    bundle["bindings"].update({
        "case_binding_sha256": case_binding_sha,
        "case_sha256": case_binding["cases"][0]["case_sha256"],
        "fixture_sha256": BATCH.sha_file(root / "fixture.json", "rebound fixture"),
        "identity_file_sha256": BATCH.sha_file(root / "identity.json", "rebound identity"),
        "identity_self_sha256": identity["identity_sha256"],
    })
    for name in bundle["files"]:
        bundle["files"][name]["sha256"] = BATCH.sha_file(root / name, f"rebound {name}")
    _rewrite_json(root / "bundle.json", bundle)
    lines = []
    for name in sorted(set(BATCH.ONE_CASE_MEMBER_CONTRACT) | {"bundle.json"}):
        lines.append(f"{BATCH.sha_file(root / name, f'rebound sum {name}')}  {name}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="ascii")
    for path in root.iterdir():
        path.chmod(0o555 if path.name == "resident-driver" else 0o444)
    return root


def _driver(tmp_path: Path, python: Path, oom: bool = False, reset_bad: bool = False, drift: str | None = None) -> tuple[Path, str]:
    suffix = drift or ("oom" if oom else "reset-bad" if reset_bad else "ok")
    path = tmp_path / f"fake-{suffix}-driver.py"
    path.write_text(
        """#!%s
import hashlib,json,os,sys
oom = %r
reset_bad = %r
drift = %r
identity = %r
served_model_binding = %r
identity['binary_sha256'] = hashlib.sha256(open(sys.argv[0], 'rb').read()).hexdigest()
if 'ULLM_AQ4_PINNED_FD_MAP' in os.environ:
    map_fd = int(os.environ['ULLM_AQ4_PINNED_FD_MAP'])
    fd_map = json.loads(os.pread(map_fd, 1024 * 1024, 0))
    served = next(item for item in fd_map['bindings'] if item['role'] == 'served_manifest')
    raw = os.pread(served['descriptor'], served['identity']['size'] + 1, 0)
    assert sys.argv[2] == served['logical_path']
    assert hashlib.sha256(raw).hexdigest() == served['sha256']
    identity['served_model_manifest_sha256'] = served['sha256']
    served_model_binding = {
        'schema_version': 'ullm.aq4_p2_served_model_binding.v2',
        'mode': 'pinned_fd',
        'logical_path': served['logical_path'],
        'effective_source': 'inherited_sealed_fd',
        'descriptor_transport': 'inherited_fd_map',
        'closure': 'control_input',
        'method': 'read',
        'identity': served['identity'],
        'sha256': served['sha256'],
        'byte_count': len(raw),
        'single_read': True,
        'logical_path_opened': False,
    }
def file_sha(path):
    with open(path, 'rb') as handle: return hashlib.sha256(handle.read()).hexdigest()
def exact_link(value):
    assert set(value) == {'path', 'sha256'}
    assert file_sha(value['path']) == value['sha256']
def validate_case_begin(message):
    assert set(message) == {'command', 'schema_version', 'case_id', 'case_sha256', 'case_binding', 'identity', 'preflight', 'policy', 'fixture', 'execution'}
    assert message['command'] == 'case_begin' and message['schema_version'] == 'ullm.aq4_p2_resident_driver.v2'
    for name in ('case_binding', 'identity', 'preflight', 'policy', 'fixture'): exact_link(message[name])
    with open(message['preflight']['path'], encoding='utf-8') as handle: prepared = json.load(handle)
    assert set(prepared) == {'weights_bytes', 'persistent_state_bytes', 'kv_cache_bytes', 'workspace_bytes', 'temporary_bytes', 'vram_headroom_bytes', 'gpu_process_snapshot'}
    assert all(type(prepared[name]) is int and prepared[name] >= 0 for name in set(prepared) - {'gpu_process_snapshot'})
    assert isinstance(prepared['gpu_process_snapshot'], list)
    assert set(message['execution']) == {'scope', 'phase', 'mode', 'prompt_tokens', 'cached_prefix_tokens', 'context_tokens', 'generated_tokens', 'request_count', 'requested_m', 'resolved_m', 'sampling', 'control'}
    assert set(message['execution']['sampling']) == {'mode', 'temperature', 'top_p', 'top_k', 'seed'}
    assert set(message['execution']['control']) == {'control_id', 'role', 'format_id', 'implementation_id', 'promotion_eligible'}
if drift == 'driver': identity['binary_sha256'] = '0' * 64
if drift == 'package': identity['package_manifest_sha256'] = '0' * 64
if drift == 'device': identity['runtime_device']['architecture'] = 'gfx9999'
session = 'fake-session'
case_id = None
requested = 1
resolved = 1
print(json.dumps({'event':'ready','schema_version':'ullm.aq4_p2_resident_driver.v2','model_loads':1,'resident_session_id':session,'driver_identity':identity,'served_model_binding':served_model_binding}), flush=True)
for line in sys.stdin:
    msg=json.loads(line)
    if msg['command']=='case_begin':
        validate_case_begin(msg)
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
""" % (
            python,
            oom,
            reset_bad,
            drift,
            _identity("0" * 64),
            _legacy_served_binding(tmp_path / "served-model.json"),
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path, BATCH.sha_file(path, "fake resident driver", absolute=True)


def _live_command(tmp_path: Path, output: Path, expanded: Path, index: Path, identity: Path, preflight: Path, policy: Path, driver: Path, run_id: str) -> list[str]:
    return [sys.executable, str(ROOT / "tools/run-aq4-p2-resident-batch.py"), "--expanded", str(expanded), "--fixture-index", str(index), "--identity", str(identity), "--preflight", str(preflight), "--policy", str(policy), "--output-dir", str(output), "--run-id", run_id, "--baseline-kind", "p3-current-head", "--lock-path", str(tmp_path / "r9700.lock"), "--driver-command", str(driver), "--served-model-manifest", str(tmp_path / "served-model.json"), "--device-index", "1", "--build-git-commit", "e" * 40]


def _fault_driver(tmp_path: Path, python: Path, fault: str) -> tuple[Path, str]:
    path = tmp_path / f"fake-{fault}-driver.py"
    path.write_text(
        """#!%s
import hashlib,json,os,signal,sys,time
fault = %r
identity = %r
served_model_binding = %r
identity['binary_sha256'] = hashlib.sha256(open(sys.argv[0], 'rb').read()).hexdigest()
if fault == 'early_exit':
    os.write(2, b'early-boom\\n'); raise SystemExit(23)
if fault == 'large_stderr':
    chunk = b'bounded-large-stderr-' * 4096
    remaining = 2 * 1024 * 1024
    while remaining:
        part = chunk[:remaining]; os.write(2, part); remaining -= len(part)
    raise SystemExit(24)
if fault == 'secret_stderr':
    os.write(2, b'Authorization: Bearer must-not-be-retained\\n'); raise SystemExit(27)
if fault == 'signal':
    os.kill(os.getpid(), signal.SIGTERM)
if fault == 'hang':
    time.sleep(60)
if fault == 'descendant_hang':
    if os.fork() == 0:
        time.sleep(60); raise SystemExit(0)
    raise SystemExit(28)
if fault == 'invalid_json':
    os.write(1, b'not-json\\n'); raise SystemExit(26)
session = 'fault-session'
print(json.dumps({'event':'ready','schema_version':'ullm.aq4_p2_resident_driver.v2','model_loads':1,'resident_session_id':session,'driver_identity':identity,'served_model_binding':served_model_binding}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message['command'] == 'case_begin':
        case_id = message['case_id']
        print(json.dumps({'event':'case_ready','schema_version':'ullm.aq4_p2_resident_driver.v2','resident_session_id':session,'case_id':case_id,'requested_m':message['execution']['requested_m'],'resolved_m':message['execution']['resolved_m'],'baseline_clean':True}), flush=True)
    elif message['command'] == 'run':
        os.write(2, b'midrun-boom\\n'); raise SystemExit(25)
    elif message['command'] == 'shutdown':
        break
"""
        % (
            python,
            fault,
            _identity("0" * 64),
            _legacy_served_binding(tmp_path / "served-model.json"),
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path, BATCH.sha_file(path, "fault resident driver", absolute=True)


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


def test_driver_command_remainder_preserves_option_like_argv_and_exact_schema() -> None:
    driver_argv = [
        "/driver",
        "--served-model-manifest",
        "/served-model.json",
        "--device-index",
        "1",
        "--build-git-commit",
        "e" * 40,
    ]
    parsed = BATCH.parse_args([
        "--expanded", "/expanded.json",
        "--fixture-index", "/fixtures.json",
        "--identity", "/identity.json",
        "--preflight", "/preflight.json",
        "--policy", "/policy.json",
        "--output-dir", "/output",
        "--run-id", "run",
        "--baseline-kind", "active-production",
        "--driver-command", *driver_argv,
    ])
    assert parsed.driver_command == driver_argv
    identity = {"resident_driver_identity": _identity("a" * 64)}
    BATCH.validate_driver_argv_schema(driver_argv, identity)
    BATCH.validate_driver_argv_schema(driver_argv, identity, expected_argv=driver_argv)


@pytest.mark.parametrize("variant", ("trailing", "reordered", "missing", "one_case_swap"))
def test_driver_command_exact_schema_rejects_drift(variant: str) -> None:
    command = ["/driver", "--served-model-manifest", "/served-model.json", "--device-index", "1", "--build-git-commit", "e" * 40]
    identity = {"resident_driver_identity": _identity("a" * 64)}
    expected = command.copy()
    if variant == "trailing":
        command.append("--unexpected")
    elif variant == "reordered":
        command[1:5] = ["--device-index", "1", "--served-model-manifest", "/served-model.json"]
    elif variant == "missing":
        command = command[:-2]
    else:
        expected[2] = "/bound-one-case-served-model.json"
    with pytest.raises(BATCH.BatchError, match="exact production schema|one-case launch binding"):
        BATCH.validate_driver_argv_schema(command, identity, expected_argv=expected if variant == "one_case_swap" else None)


def test_one_case_smoke_dry_run_validates_exact_root_and_subprocesses(tmp_path: Path) -> None:
    output = tmp_path / "one-case-dry-run"
    completed = subprocess.run(_one_case_command(output), text=True, capture_output=True)
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
    assert plan["validation"]["root_contract"] == "ullm.aq4_p2_resident_smoke_bundle_root.v4"
    assert set(plan["validation"]["members"]) == BATCH.ONE_CASE_ROOT_MEMBERS
    assert all(item["type"] == "regular_file" and item["nlink"] == 1 for item in plan["validation"]["members"].values())
    assert plan["validation"]["fake_driver_subprocess_count"] == 1
    assert plan["validation"]["driver_fake_handshake"] == "passed"
    assert plan["validation"]["resident_session_id"] == "offline-fake-ready-not-executed"
    validator = plan["validation"]["trusted_bundle_validator"]
    assert validator["subprocess_count"] == 1
    assert len(validator["source"]["sha256"]) == 64
    assert len(validator["report_sha256"]) == 64


def test_one_case_smoke_requires_explicit_bundle_root(tmp_path: Path) -> None:
    command = _one_case_command(tmp_path / "missing-root")
    root_index = command.index("--bundle-root")
    del command[root_index:root_index + 2]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "--bundle-root is required" in completed.stderr


@pytest.mark.parametrize("option", ("--trusted-validator", "--trusted-validator-sha256"))
def test_one_case_smoke_requires_trusted_validator_and_expected_sha(tmp_path: Path, option: str) -> None:
    command = _one_case_command(tmp_path / "missing-validator")
    index = command.index(option)
    del command[index:index + 2]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "are required with --one-case-smoke" in completed.stderr


def test_one_case_smoke_rejects_trusted_validator_sha_swap(tmp_path: Path) -> None:
    command = _one_case_command(tmp_path / "validator-sha-swap")
    index = command.index("--trusted-validator-sha256")
    command[index + 1] = "0" * 64
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "source differs from expected SHA" in completed.stderr


@pytest.mark.parametrize("variant", ("leaf", "ancestor"))
def test_one_case_smoke_rejects_absolute_validator_symlink(tmp_path: Path, variant: str) -> None:
    command = _one_case_command(tmp_path / f"validator-{variant}-symlink")
    validator_index = command.index("--trusted-validator")
    if variant == "leaf":
        linked = tmp_path / "validator.py"
        linked.symlink_to(TRUSTED_BUNDLE_VALIDATOR)
    else:
        linked_parent = tmp_path / "linked-tools"
        linked_parent.symlink_to(TRUSTED_BUNDLE_VALIDATOR.parent, target_is_directory=True)
        linked = linked_parent / TRUSTED_BUNDLE_VALIDATOR.name
    command[validator_index + 1] = str(linked)
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "trusted bundle validator" in completed.stderr
    assert "symlink" in completed.stderr or "single-link regular file" in completed.stderr


def test_one_case_smoke_rejects_handcrafted_partial_root(tmp_path: Path) -> None:
    handcrafted = tmp_path / "handcrafted"
    shutil.copytree(TRUSTED_ONE_CASE_ROOT, handcrafted)
    (handcrafted / "trust-roots.json").unlink()
    completed = subprocess.run(_one_case_command(tmp_path / "partial-out", bundle_root=handcrafted), text=True, capture_output=True)
    assert completed.returncode != 0
    assert "exact member coverage differs" in completed.stderr


@pytest.mark.parametrize("variant", ("case", "identity", "fake_ready"))
def test_one_case_smoke_rejects_rebound_case_identity_and_fake_ready(tmp_path: Path, variant: str) -> None:
    rebound = _rebound_bundle(tmp_path, variant)
    completed = subprocess.run(_one_case_command(tmp_path / f"{variant}-out", bundle_root=rebound), text=True, capture_output=True)
    assert completed.returncode != 0
    expected = "trusted case ID/hash differs" if variant == "case" else "prepared dry-run identity/handshake binding differs"
    assert expected in completed.stderr


def test_one_case_binding_is_not_accepted_by_normal_84_case_mode(tmp_path: Path) -> None:
    command = _one_case_command(tmp_path / "ordinary")
    command.remove("--one-case-smoke")
    for option in ("--bundle-root", "--trusted-validator", "--trusted-validator-sha256"):
        index = command.index(option)
        del command[index:index + 2]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "must contain exactly 84 target cases" in completed.stderr


def test_fake_resident_driver_writes_one_atomic_raw_per_case(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _driver(tmp_path, python)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "run"
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "r-current")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 84
    sample = json.loads(raws[0].read_text())
    assert sample["status"] == "ok"
    assert sample["resident"] == {
        "session_id": "fake-session",
        "model_loads": 1,
        "driver_identity": _identity(driver_sha256),
        "case_reset_count": 12,
    }
    assert sample["device_lock"]["driver"]["sha256"] == driver_sha256
    assert sample["schedule"] == {"warmup_runs": 2, "measured_runs": 10, "completed_runs": 12}
    assert json.loads((output / "resident-batch.summary.json").read_text())["completed_cases"] == 84


def test_resident_oom_is_immutable_and_aborts_remaining_cases(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _driver(tmp_path, python, oom=True)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "oom-run"
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "r-current")
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode != 0
    raws = list(output.glob("*.raw.json"))
    assert len(raws) == 1
    assert json.loads(raws[0].read_text())["status"] == "oom"
    failure = json.loads((output / "resident-batch.failure.json").read_text())
    assert failure["protocol"]["ready_received"] is True
    assert failure["protocol"]["case_begin_count"] == 1
    assert failure["protocol"]["warmup_completed"] == 2
    assert failure["protocol"]["measured_completed"] == 1
    assert failure["cleanup"]["reaped"] is True


@pytest.mark.parametrize(
    ("fault", "failure_kind", "exit_kind", "exit_value"),
    (
        ("early_exit", "eof", "exit", 23),
        ("signal", "eof", "signal", 15),
        ("hang", "timeout", "signal", 15),
        ("invalid_json", "invalid_json", "exit", 26),
        ("descendant_hang", "timeout", "exit", 28),
    ),
)
def test_pre_ready_failures_preserve_bounded_process_evidence(
    tmp_path: Path,
    fault: str,
    failure_kind: str,
    exit_kind: str,
    exit_value: int,
) -> None:
    case_root = tmp_path / fault
    python, _ = _detached_python(case_root)
    driver, driver_sha256 = _fault_driver(case_root, python, fault)
    expanded, index, identity, preflight, policy = _bundle(case_root, driver_sha256)
    output = case_root / "failure-output"
    command = _live_command(case_root, output, expanded, index, identity, preflight, policy, driver, fault)
    command[command.index("--driver-command"):command.index("--driver-command")] = ["--timeout", "0.15"]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=10)
    assert completed.returncode != 0
    evidence_path = output / "resident-batch.failure.json"
    assert evidence_path.stat().st_mode & 0o777 == 0o444
    evidence = json.loads(evidence_path.read_text())
    assert evidence["schema_version"] == "ullm.aq4_p2_resident_driver_process.v2"
    assert evidence["invocation"]["logical_argv"] == command[
        command.index("--driver-command") + 1 :
    ]
    assert evidence["invocation"]["logical_argv_sha256"] == BATCH.sha_bytes(
        BATCH.canonical(evidence["invocation"]["logical_argv"])
    )
    assert evidence["invocation"]["effective_semantic_bindings"] == []
    assert evidence["invocation"]["pinned_fd_map"] is None
    assert evidence["failure"]["kind"] == failure_kind
    assert evidence["failure"]["stage"] == "ready"
    assert evidence["protocol"]["ready_received"] is False
    assert evidence["protocol"]["case_begin_count"] == 0
    assert evidence["protocol"]["warmup_completed"] == 0
    assert evidence["protocol"]["measured_completed"] == 0
    assert evidence["exit"]["kind"] == exit_kind
    key = "signal" if exit_kind == "signal" else "exit_code"
    assert evidence["exit"][key] == exit_value
    assert evidence["cleanup"]["reaped"] is True
    assert evidence["cleanup"]["process_group_alive_final"] is False
    assert evidence["lock"]["after_driver"]["same_inode"] is True
    assert evidence["gpu_owner"]["status"] == "not_probed"
    stdout = evidence["protocol"]["stdout_records"][-1]
    assert stdout["bytes"] <= BATCH.MAX_DRIVER_STDOUT_LINE_BYTES
    assert len(stdout["sha256"]) == 64
    if fault == "invalid_json":
        assert stdout["outcome"] == "invalid_json"
        assert stdout["sha256"] == hashlib.sha256(b"not-json\n").hexdigest()
    if fault == "hang":
        assert evidence["cleanup"]["wait_timed_out"] is True
    if fault == "descendant_hang":
        assert evidence["cleanup"]["signals"]


def test_large_stderr_streams_without_pipe_deadlock_and_retains_only_bounded_tail(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _fault_driver(tmp_path, python, "large_stderr")
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "large-output"
    completed = subprocess.run(
        _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "large"),
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode != 0
    evidence = json.loads((output / "resident-batch.failure.json").read_text())
    stderr = evidence["stderr"]
    assert stderr["bytes"] == 2 * 1024 * 1024
    assert len(stderr["sha256"]) == 64
    assert stderr["retained_kind"] == "bounded_tail"
    tail = Path(stderr["retained_path"])
    assert tail.stat().st_size == BATCH.MAX_DRIVER_TAIL_BYTES
    assert tail.stat().st_mode & 0o777 == 0o444
    assert not (output / ".resident-driver.stderr.incomplete").exists()


def test_secret_stderr_is_hashed_but_never_retained(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _fault_driver(tmp_path, python, "secret_stderr")
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "secret-output"
    completed = subprocess.run(
        _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "secret"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    stderr = json.loads((output / "resident-batch.failure.json").read_text())["stderr"]
    assert stderr["bytes"] > 0 and len(stderr["sha256"]) == 64
    assert stderr["secret_scan"]["detected"] is True
    assert stderr["retained_path"] is None
    assert list(output.glob("resident-driver.stderr*")) == []


def test_midrun_exit_records_ready_and_exact_completion_stage(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _fault_driver(tmp_path, python, "midrun")
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "midrun-output"
    completed = subprocess.run(
        _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "midrun"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    evidence = json.loads((output / "resident-batch.failure.json").read_text())
    assert evidence["failure"]["kind"] == "eof"
    assert evidence["failure"]["stage"].startswith("run:")
    assert evidence["protocol"]["ready_received"] is True
    assert evidence["protocol"]["case_begin_count"] == 1
    assert evidence["protocol"]["warmup_completed"] == 0
    assert evidence["exit"] == {"kind": "exit", "exit_code": 25, "signal": None, "oom_like_unconfirmed": False}


def test_incomplete_reset_is_immutable_and_aborts_process_reuse(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _driver(tmp_path, python, reset_bad=True)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    output = tmp_path / "reset-bad-run"
    command = _live_command(tmp_path, output, expanded, index, identity, preflight, policy, driver, "r-current")
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
        python, _ = _detached_python(tmp_path / drift)
        driver, driver_sha256 = _driver(tmp_path / drift, python, drift=drift)
        expanded, index, identity, preflight, policy = _bundle(tmp_path / drift, driver_sha256)
        output = tmp_path / f"{drift}-drift-run"
        command = _live_command(tmp_path / drift, output, expanded, index, identity, preflight, policy, driver, f"r-{drift}")
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode != 0
        assert list(output.glob("*.raw.json")) == []


def test_case_swap_result_order_and_release_drift_are_rejected(tmp_path: Path) -> None:
    for drift in ("case_swap", "result_order", "release"):
        python, _ = _detached_python(tmp_path / drift)
        driver, driver_sha256 = _driver(tmp_path / drift, python, drift=drift)
        expanded, index, identity, preflight, policy = _bundle(tmp_path / drift, driver_sha256)
        output = tmp_path / f"{drift}-run"
        command = _live_command(tmp_path / drift, output, expanded, index, identity, preflight, policy, driver, f"r-{drift}")
        completed = subprocess.run(command, text=True, capture_output=True)
        assert completed.returncode != 0
        assert list(output.glob("*.raw.json")) == []


def test_cargo_style_driver_hardlink_is_rejected_and_detached_copy_is_accepted(tmp_path: Path) -> None:
    detached, digest = _detached_python(tmp_path)
    cargo_link = tmp_path / "cargo-deps-hardlink"
    os.link(detached, cargo_link)
    bound = {"resident_driver_identity": _identity(digest)}
    command = lambda path: [str(path), "--served-model-manifest", "/served-model.json", "--device-index", "1", "--build-git-commit", "e" * 40]
    with pytest.raises(BATCH.BatchError, match="single-link"):
        BATCH.validate_driver_command(command(detached), bound)
    accepted = tmp_path / "accepted-detached-driver"
    shutil.copy2(detached, accepted)
    evidence = BATCH.validate_driver_command(command(accepted), bound)
    assert evidence["sha256"] == digest
    assert evidence["nlink"] == 1


def test_lock_contention_fails_before_spawn_and_exception_releases_lock(tmp_path: Path) -> None:
    python, _ = _detached_python(tmp_path)
    driver, driver_sha256 = _driver(tmp_path, python, oom=True)
    expanded, index, identity, preflight, policy = _bundle(tmp_path, driver_sha256)
    lock_path = tmp_path / "r9700.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    blocked_output = tmp_path / "blocked-output"
    blocked = subprocess.run(_live_command(tmp_path, blocked_output, expanded, index, identity, preflight, policy, driver, "blocked"), text=True, capture_output=True)
    assert blocked.returncode != 0
    assert not blocked_output.exists()
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)

    failed_output = tmp_path / "failed-output"
    failed = subprocess.run(_live_command(tmp_path, failed_output, expanded, index, identity, preflight, policy, driver, "oom-cleanup"), text=True, capture_output=True)
    assert failed.returncode != 0
    owner = json.loads((failed_output / "resident-batch.lock-owner.json").read_text())
    assert owner["driver"]["sha256"] == driver_sha256
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_bound_device_lock_reuses_exact_live_preflight_inode(tmp_path: Path) -> None:
    lock_path = tmp_path / "bound-r9700.lock"
    lock_path.touch(mode=0o600)
    metadata = lock_path.lstat()
    expected = {"device": metadata.st_dev, "inode": metadata.st_ino}

    with BATCH.acquire_device_lock(
        lock_path,
        "bound-lock",
        {"sha256": "a" * 64},
        expected_identity=expected,
    ) as owner:
        assert owner["device"] == expected["device"]
        assert owner["inode"] == expected["inode"]
        current = lock_path.lstat()
        assert (current.st_dev, current.st_ino) == (expected["device"], expected["inode"])


def test_bound_device_lock_rejects_missing_and_replaced_inode(tmp_path: Path) -> None:
    missing = tmp_path / "missing.lock"
    with pytest.raises(BATCH.BatchError, match="metadata failed"):
        with BATCH.acquire_device_lock(
            missing,
            "missing",
            {},
            expected_identity={"device": tmp_path.stat().st_dev, "inode": 1},
        ):
            pass
    assert not missing.exists()

    lock_path = tmp_path / "replaced.lock"
    lock_path.touch(mode=0o600)
    original = lock_path.lstat()
    lock_path.rename(tmp_path / "original.lock")
    lock_path.touch(mode=0o600)
    with pytest.raises(BATCH.BatchError, match="identity differs"):
        with BATCH.acquire_device_lock(
            lock_path,
            "replaced",
            {},
            expected_identity={"device": original.st_dev, "inode": original.st_ino},
        ):
            pass


def test_bound_device_lock_rejects_swap_between_lstat_and_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = tmp_path / "raced.lock"
    lock_path.touch(mode=0o600)
    original = lock_path.lstat()
    real_open = os.open
    swapped = False

    def racing_open(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if not swapped and Path(path) == lock_path:
            swapped = True
            lock_path.rename(tmp_path / "original-raced.lock")
            lock_path.touch(mode=0o600)
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", racing_open)
    with pytest.raises(BATCH.BatchError, match="changed while opening"):
        with BATCH.acquire_device_lock(
            lock_path,
            "raced",
            {},
            expected_identity={"device": original.st_dev, "inode": original.st_ino},
        ):
            pass


def test_pinned_control_read_and_lock_flock_use_original_fds_after_logical_path_swap(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    control.write_bytes(b'{"trusted":true}\n')
    control_replacement = tmp_path / "control-replacement.json"
    control_replacement.write_bytes(b'{"trusted":false}\n')
    lock = tmp_path / "device.lock"
    lock.touch(mode=0o600)
    lock_replacement = tmp_path / "lock-replacement"
    lock_replacement.touch(mode=0o600)
    control_descriptor = os.open(control, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    lock_descriptor = os.open(lock, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    control_metadata = os.fstat(control_descriptor)
    lock_metadata = os.fstat(lock_descriptor)
    bindings = [
        {
            "role": "control_fixture",
            "logical_path": str(control),
            "resolved_path": None,
            "descriptor": control_descriptor,
            "kind": "regular_file",
            "closure": "control_input",
            "method": "read",
            "identity": BATCH._named_file_identity(control_metadata),
            "sha256": hashlib.sha256(b'{"trusted":true}\n').hexdigest(),
        },
        {
            "role": "device_lock",
            "logical_path": str(lock),
            "resolved_path": None,
            "descriptor": lock_descriptor,
            "kind": "regular_file",
            "closure": "device_lock",
            "method": "flock",
            "identity": BATCH._named_file_identity(lock_metadata),
            "sha256": None,
        },
    ]
    value = {
        "schema_version": BATCH.FD_MAP_SCHEMA,
        "status": "bound",
        "map_sha256": "0" * 64,
        "logical_argv_sha256": "1" * 64,
        "closure_contract": BATCH.FD_CLOSURE_CONTRACT,
        "bindings": bindings,
    }
    pinned = BATCH.PinnedFdMap(-1, value)
    control_backup = tmp_path / "control-backup.json"
    lock_backup = tmp_path / "lock-backup"
    assert BATCH.ACTIVE_FD_MAP is None
    BATCH.ACTIVE_FD_MAP = pinned
    try:
        control.rename(control_backup)
        control_replacement.rename(control)
        lock.rename(lock_backup)
        lock_replacement.rename(lock)
        raw, digest, metadata = BATCH.read_regular(control, "pinned control")
        assert raw == b'{"trusted":true}\n'
        assert digest == bindings[0]["sha256"]
        assert metadata.st_ino == control_metadata.st_ino
        expected_lock = {"device": lock_metadata.st_dev, "inode": lock_metadata.st_ino}
        with BATCH.acquire_device_lock(
            lock,
            "pinned-lock-test",
            {"sha256": "a" * 64},
            expected_identity=expected_lock,
        ) as owner:
            assert owner["inode"] == lock_metadata.st_ino
        assert b"pinned-lock-test" in lock_backup.read_bytes()
        assert lock.read_bytes() == b""
    finally:
        BATCH.ACTIVE_FD_MAP = None
        if control_backup.exists():
            control.unlink(missing_ok=True)
            control_backup.rename(control)
        if lock_backup.exists():
            lock.unlink(missing_ok=True)
            lock_backup.rename(lock)
        os.close(control_descriptor)
        os.close(lock_descriptor)


def test_pinned_fd_map_rejects_null_sha_for_code_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code = tmp_path / "code.py"
    code.write_text("raise SystemExit(0)\n", encoding="utf-8")
    code_descriptor = os.open(code, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    map_descriptor = os.memfd_create(
        "invalid-null-code-sha",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        value = {
            "schema_version": BATCH.FD_MAP_SCHEMA,
            "status": "bound",
            "map_sha256": None,
            "logical_argv_sha256": "1" * 64,
            "closure_contract": BATCH.FD_CLOSURE_CONTRACT,
            "bindings": [
                {
                    "role": "resident_runner",
                    "logical_path": str(code),
                    "resolved_path": None,
                    "descriptor": code_descriptor,
                    "kind": "regular_file",
                    "closure": "code_execution",
                    "method": "exec",
                    "identity": BATCH._named_file_identity(os.fstat(code_descriptor)),
                    "sha256": None,
                }
            ],
        }
        value["map_sha256"] = BATCH.sha_bytes(BATCH.canonical(value))
        raw = BATCH.canonical(value) + b"\n"
        os.write(map_descriptor, raw)
        seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
        fcntl.fcntl(map_descriptor, fcntl.F_ADD_SEALS, seals)
        monkeypatch.setenv(BATCH.FD_MAP_ENV, str(map_descriptor))
        with pytest.raises(BATCH.BatchError, match="binding value differs"):
            BATCH.PinnedFdMap.from_environment(required=True)
    finally:
        os.close(map_descriptor)
        os.close(code_descriptor)


def test_pinned_served_manifest_handoff_keeps_logical_argv_and_reads_original_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = tmp_path / "fd-aware-driver.py"
    driver.write_text(
        f"#!{Path(sys.executable).resolve()}\n"
        "import hashlib, json, os, sys\n"
        "map_fd = int(os.environ['ULLM_AQ4_PINNED_FD_MAP'])\n"
        "fd_map = json.loads(os.pread(map_fd, 1024 * 1024, 0))\n"
        "binding = next(item for item in fd_map['bindings'] if item['role'] == 'served_manifest')\n"
        "raw = os.pread(binding['descriptor'], binding['identity']['size'] + 1, 0)\n"
        "assert hashlib.sha256(raw).hexdigest() == binding['sha256']\n"
        "print(json.dumps({'logical_manifest': sys.argv[2], 'manifest': raw.decode('ascii')}))\n",
        encoding="utf-8",
    )
    driver.chmod(0o555)
    manifest = tmp_path / "active.json"
    trusted_raw = b'{"trusted":true}\n'
    manifest.write_bytes(trusted_raw)
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b'{"trusted":false}\n')
    driver_fd = os.open(driver, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    manifest_fd = os.open(manifest, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    map_fd = os.memfd_create(
        "served-manifest-handoff",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    logical_argv = [
        str(driver),
        "--served-model-manifest",
        str(manifest),
        "--device-index",
        "1",
        "--build-git-commit",
        "e" * 40,
    ]
    bindings = [
        {
            "role": "resident_driver",
            "logical_path": str(driver),
            "resolved_path": None,
            "descriptor": driver_fd,
            "kind": "regular_file",
            "closure": "code_execution",
            "method": "exec",
            "identity": BATCH._named_file_identity(os.fstat(driver_fd)),
            "sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
        },
        {
            "role": "served_manifest",
            "logical_path": str(manifest),
            "resolved_path": None,
            "descriptor": manifest_fd,
            "kind": "regular_file",
            "closure": "control_input",
            "method": "read",
            "identity": BATCH._named_file_identity(os.fstat(manifest_fd)),
            "sha256": hashlib.sha256(trusted_raw).hexdigest(),
        },
    ]
    value = {
        "schema_version": BATCH.FD_MAP_SCHEMA,
        "status": "bound",
        "map_sha256": None,
        "logical_argv_sha256": BATCH.sha_bytes(BATCH.canonical(logical_argv)),
        "closure_contract": BATCH.FD_CLOSURE_CONTRACT,
        "bindings": bindings,
    }
    value["map_sha256"] = BATCH.sha_bytes(BATCH.canonical(value))
    raw_map = BATCH.canonical(value) + b"\n"
    os.write(map_fd, raw_map)
    seals = fcntl.F_SEAL_SEAL | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE
    fcntl.fcntl(map_fd, fcntl.F_ADD_SEALS, seals)
    monkeypatch.setenv(BATCH.FD_MAP_ENV, str(map_fd))
    pinned = BATCH.PinnedFdMap.from_environment(required=True)
    backup = tmp_path / "trusted-backup.json"
    assert BATCH.ACTIVE_FD_MAP is None
    BATCH.ACTIVE_FD_MAP = pinned
    try:
        manifest.rename(backup)
        replacement.rename(manifest)
        invocation = BATCH._driver_invocation_document(logical_argv)
        effective = list(logical_argv)
        effective[0] = BATCH.effective_fd_path(
            driver, method="exec", role="resident_driver"
        )
        completed = subprocess.run(
            effective,
            check=True,
            text=True,
            capture_output=True,
            **BATCH.fd_child_options(),
        )
        observed = json.loads(completed.stdout)
        assert observed == {
            "logical_manifest": str(manifest),
            "manifest": trusted_raw.decode("ascii"),
        }
        assert effective[2] == str(manifest)
        assert invocation["logical_argv"] == logical_argv
        assert invocation["logical_argv_sha256"] == BATCH.sha_bytes(
            BATCH.canonical(logical_argv)
        )
        assert [item["argument_index"] for item in invocation["effective_semantic_bindings"]] == [0, 2]
        assert [item["role"] for item in invocation["effective_semantic_bindings"]] == [
            "resident_driver",
            "served_manifest",
        ]
        assert invocation["pinned_fd_map"] == {
            "schema_version": BATCH.FD_MAP_SCHEMA,
            "map_sha256": value["map_sha256"],
            "closure_contract": BATCH.FD_CLOSURE_CONTRACT,
        }
        assert all(
            "descriptor" not in item
            for item in invocation["effective_semantic_bindings"]
        )
        ready_binding = {
            "schema_version": BATCH.SERVED_MODEL_BINDING_SCHEMA,
            "mode": "pinned_fd",
            "logical_path": str(manifest),
            "effective_source": "inherited_sealed_fd",
            "descriptor_transport": "inherited_fd_map",
            "closure": "control_input",
            "method": "read",
            "identity": bindings[1]["identity"],
            "sha256": bindings[1]["sha256"],
            "byte_count": len(trusted_raw),
            "single_read": True,
            "logical_path_opened": False,
        }
        assert BATCH._validate_served_model_binding(
            ready_binding, bindings[1]["sha256"]
        ) == ready_binding
        for field, replacement_value in (
            ("logical_path", str(backup)),
            ("logical_path_opened", True),
            ("descriptor_transport", "none"),
        ):
            drifted = json.loads(json.dumps(ready_binding))
            drifted[field] = replacement_value
            with pytest.raises(BATCH.BatchError, match="pinned served model binding"):
                BATCH._validate_served_model_binding(
                    drifted, bindings[1]["sha256"]
                )
    finally:
        BATCH.ACTIVE_FD_MAP = None
        manifest.unlink(missing_ok=True)
        backup.rename(manifest)
        os.close(map_fd)
        os.close(manifest_fd)
        os.close(driver_fd)


def test_required_pinned_fd_map_fails_when_missing_or_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BATCH.FD_MAP_ENV, raising=False)
    with pytest.raises(BATCH.BatchError, match="requires the pinned FD map"):
        BATCH.PinnedFdMap.from_environment(required=True)
    descriptor = os.memfd_create("closed-map", getattr(os, "MFD_CLOEXEC", 0))
    os.close(descriptor)
    monkeypatch.setenv(BATCH.FD_MAP_ENV, str(descriptor))
    with pytest.raises(BATCH.BatchError, match="pinned FD map read failed"):
        BATCH.PinnedFdMap.from_environment(required=True)


def test_actual_profile_rc1_is_the_proc_fd_manifest_regression_fixture() -> None:
    root = (
        ROOT
        / "benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2"
        / "resident-one-case-smoke-profile-execute-v3"
    )
    failure = json.loads((root / "resident-batch.failure.json").read_text())
    stderr = (root / "resident-driver.stderr.log").read_text()
    assert failure["schema_version"] == "ullm.aq4_p2_resident_driver_process.v1"
    assert failure["failure"] == {
        "kind": "eof",
        "reason": "resident driver exited before response",
        "stage": "ready",
    }
    assert failure["protocol"]["ready_received"] is False
    assert failure["protocol"]["warmup_completed"] == 0
    assert failure["protocol"]["measured_completed"] == 0
    assert stderr == (
        "ullm-aq4-p2-resident-driver: served model rejected: "
        "served-model manifest traverses a symlink\n"
    )


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
