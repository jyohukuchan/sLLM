from __future__ import annotations

import importlib.util
import hashlib
import json
import mmap
import os
import subprocess
import errno
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_holdout_runner", ROOT / "tools" / "run-aq4-p2-fidelity-holdout.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_actual_verified_receipt_requires_exact_status(tmp_path: Path) -> None:
    receipt = tmp_path / "actual.json"
    receipt.write_text('{"status":"prepared_not_executed"}\n', encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="shape"):
        RUNNER._actual_verified(receipt)


def test_actual_verified_receipt_rejects_status_only_shape(tmp_path: Path) -> None:
    receipt = tmp_path / "actual.json"
    receipt.write_text('{"status":"actual_verified"}\n', encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="shape"):
        RUNNER._actual_verified(receipt)


def test_failure_receipt_is_create_new_and_immutable(tmp_path: Path) -> None:
    plan = {
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {
            "receipt": {"path": str(tmp_path / "actual.json"), "sha256": "f" * 64}
        },
        "identity": {"device_architecture": "gfx1201"},
    }
    output = tmp_path / "failure.json"
    value = RUNNER._failure(output, plan, "1" * 64, "nonzero", "fixture failure", 1)
    assert value["immutable"] is True
    assert value["failure_kind"] == "nonzero"
    with pytest.raises(RUNNER.HoldoutError, match="overwrite"):
        RUNNER._failure(output, plan, "1" * 64, "timeout", "retry")


def test_execute_refuses_existing_attempt_marker(tmp_path: Path) -> None:
    preflight = tmp_path / "preflight.json"
    marker = tmp_path / "attempt.json"
    marker.write_text('{"status":"started"}\n', encoding="ascii")
    plan = {
        "schema_version": RUNNER.PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "subset": "holdout",
        "row_count": 24,
        "paths": {
            "attempt_marker": str(marker),
            "active_output": str(tmp_path / "active"),
            "split_root": str(tmp_path),
            "source_artifact": str(tmp_path),
        },
        "execution_contract": {
            "command": ["false"],
            "command_sha256": "0" * 64,
            "timeout_seconds": 1,
        },
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {
            "path": str(tmp_path / "actual.json"),
            "sha256": "f" * 64,
        },
        "identity": {},
        "freeze_receipt_path": str(tmp_path / "freeze.json"),
    }
    preflight.write_text(json.dumps(plan) + "\n", encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="retry"):
        RUNNER._execute(
            type(
                "Args",
                (),
                {
                    "preflight": preflight,
                    "receipt_output": tmp_path / "result.json",
                    "device_index": 0,
                },
            )()
        )


def test_decision_only_consumes_frozen_bounds() -> None:
    receipt = {
        "derived_bounds": {
            name: {"bound": 0.5, "direction": spec["direction"]}
            for name, spec in RUNNER.PROTOCOL.METRICS.items()
        }
    }
    means = {
        name: (0.75 if spec["direction"] == "higher" else 0.25)
        for name, spec in RUNNER.PROTOCOL.METRICS.items()
    }
    decision, checks = RUNNER._decision(means, receipt)
    assert decision == "go"
    assert all(item["pass"] for item in checks.values())


def test_atomic_receipts_are_single_link_read_only_files(tmp_path: Path) -> None:
    output = tmp_path / "sealed.json"
    RUNNER._atomic_json(output, {"status": "sealed"}, "receipt")
    info = output.stat()
    assert info.st_nlink == 1
    assert info.st_mode & 0o777 == 0o444
    link = tmp_path / "link.json"
    link.symlink_to(output)
    with pytest.raises(RUNNER.HoldoutError, match="overwrite"):
        RUNNER._atomic_json(link, {"status": "tampered"}, "receipt")


def test_runtime_identity_requires_frozen_single_gpu_contract() -> None:
    expected = {
        "served_model_manifest_sha256": "a" * 64,
        "package_manifest_sha256": "b" * 64,
        "package_content_sha256": "0" * 64,
        "worker_binary_sha256": "c" * 64,
        "capture_binary_sha256": "4" * 64,
        "guard_sha256": "5" * 64,
        "selected_cases_sha256": "d" * 64,
        "split_manifest_sha256": "e" * 64,
        "policy_sha256": "f" * 64,
        "holdout_cases_sha256": "d" * 64,
        "quantized_artifact_revision": "rev",
        "device_architecture": "gfx1201",
        "device_index": 1,
        "device_backend": "hip",
        "device_name": "R9700",
        "device_id": 0,
        "build_sha256": "1" * 64,
        "source_identity": {
            "model_revision": "src",
            "tokenizer": {"aggregate_sha256": "2" * 64},
            "source_checkpoint": {"aggregate_sha256": "3" * 64},
        },
    }
    runtime_identity = {
        key: value
        for key, value in expected.items()
        if key
        in {
            "served_model_manifest_sha256",
            "package_manifest_sha256",
            "package_content_sha256",
            "worker_binary_sha256",
            "capture_binary_sha256",
            "guard_sha256",
            "selected_cases_sha256",
            "split_manifest_sha256",
            "policy_sha256",
            "holdout_cases_sha256",
            "quantized_artifact_revision",
            "build_sha256",
        }
    }
    runtime_identity.update(
        {
            "upstream_model_revision": "src",
            "tokenizer_aggregate_sha256": "2" * 64,
            "source_checkpoint_aggregate_sha256": "3" * 64,
            "device": {
                "requested_index": 1,
                "device_id": 0,
                "backend": "hip",
                "name": "R9700",
                "architecture": "gfx1201",
            },
            "one_process": True,
            "one_model_load": True,
            "gpu_parallelism": 2,
        }
    )
    active = {"manifest": {"runtime": {"runtime": runtime_identity, "model_loads": 1}}}
    with pytest.raises(RUNNER.HoldoutError, match="GPU-parallelism"):
        RUNNER._runtime_identity(active, expected)


def test_revalidate_identity_rejects_content_and_mode_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"one")
    identity = RUNNER._identity_file(artifact, "artifact")
    artifact.write_bytes(b"two")
    with pytest.raises(RUNNER.HoldoutError, match="changed|SHA"):
        RUNNER._revalidate_identity(identity, "artifact")


def test_stable_read_rejects_restored_mtime_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"a" * (2 * 1024 * 1024))
    original = artifact.stat()
    original_read = RUNNER.os.read
    raced = False

    def race_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        chunk = original_read(descriptor, size)
        if chunk and not raced:
            raced = True
            with artifact.open("r+b") as stream:
                stream.seek(1024 * 1024 + 7)
                stream.write(b"b")
                stream.flush()
                os.fsync(stream.fileno())
            os.utime(artifact, ns=(original.st_atime_ns, original.st_mtime_ns))
        return chunk

    monkeypatch.setattr(RUNNER.os, "read", race_read)
    with pytest.raises(RUNNER.HoldoutError, match="changed"):
        RUNNER._stable_file(artifact, "artifact")


def test_child_environment_is_allowlist_only_and_hash_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "/frozen/path")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/forbidden.so")
    monkeypatch.setenv("ULLM_UNLISTED", "forbidden")
    device = {name: "1" for name in RUNNER.DEVICE_ENV}
    ambient, child = RUNNER._frozen_child_environment({"ULLM_GUARD": "1"}, device)
    assert ambient["PATH"] == "/frozen/path"
    assert "LD_PRELOAD" not in child
    assert "ULLM_UNLISTED" not in child
    digest = RUNNER._environment_sha(child)
    monkeypatch.setenv("ULLM_UNLISTED", "changed")
    assert (
        RUNNER._environment_sha(
            RUNNER._frozen_child_environment({"ULLM_GUARD": "1"}, device)[1]
        )
        == digest
    )


def test_runtime_identity_accepts_real_rust_shaped_integer_device_fixture() -> None:
    shas = {
        "served_model_manifest_sha256": "1" * 64,
        "package_manifest_sha256": "2" * 64,
        "package_content_sha256": "3" * 64,
        "worker_binary_sha256": "4" * 64,
        "capture_binary_sha256": "5" * 64,
        "guard_sha256": "6" * 64,
        "selected_cases_sha256": "7" * 64,
        "split_manifest_sha256": "8" * 64,
        "policy_sha256": "9" * 64,
        "holdout_cases_sha256": "7" * 64,
        "quantized_artifact_revision": "quantized-revision",
        "build_sha256": "a" * 64,
    }
    expected = {
        **shas,
        "device_index": 1,
        "device_id": 0,
        "device_backend": "hip",
        "device_name": "AMD Radeon AI PRO R9700",
        "device_architecture": "gfx1201",
        "source_identity": {
            "model_revision": "source-revision",
            "tokenizer": {"aggregate_sha256": "b" * 64},
            "source_checkpoint": {"aggregate_sha256": "c" * 64},
        },
    }
    runtime = {
        **shas,
        "upstream_model_revision": "source-revision",
        "tokenizer_aggregate_sha256": "b" * 64,
        "source_checkpoint_aggregate_sha256": "c" * 64,
        "selected_subset": "holdout",
        "one_process": True,
        "one_model_load": True,
        "gpu_parallelism": 1,
        "device": {
            "requested_index": 1,
            "device_id": 0,
            "backend": "hip",
            "name": "AMD Radeon AI PRO R9700",
            "architecture": "gfx1201",
        },
        "state_evidence": {
            "contract": "full_context_step_zero_reset_v1",
            "rows_started": 24,
            "rows_completed": 24,
            "clean_before_each_row": True,
            "generation_states_observed": 24,
            "reset_calls": 24,
            "clean_after_each_reset": True,
            "scheduler_mode": "not_used_direct_capture",
            "scheduler_pending_before_each_row": 0,
            "scheduler_pending_after_each_row": 0,
        },
    }
    active = {"manifest": {"runtime": {"runtime": runtime, "model_loads": 1}}}
    assert RUNNER._runtime_identity(active, expected) == runtime


def test_validate_plan_rejects_command_sha_and_device_env_mismatch() -> None:
    command = ["/bin/false", "--device-index", "1"]
    plan = {
        "schema_version": RUNNER.PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "subset": "holdout",
        "row_count": 24,
        "identity": {"device_index": 1},
        "execution_contract": {
            "one_process": True,
            "one_model_load": True,
            "gpu_parallelism": 1,
            "capture_binary": {"path": "/bin/false"},
            "command": command,
            "command_sha256": "0" * 64,
            "device_environment": {name: "1" for name in RUNNER.DEVICE_ENV},
        },
    }
    with pytest.raises(RUNNER.HoldoutError, match="command SHA"):
        RUNNER._validate_plan(plan)
    plan["execution_contract"]["command_sha256"] = RUNNER._command_sha(command)
    plan["execution_contract"]["device_environment"]["HIP_VISIBLE_DEVICES"] = "0"
    with pytest.raises(RUNNER.HoldoutError, match="environment"):
        RUNNER._validate_plan(plan)


def test_guard_receipt_rejects_required_environment_drift(tmp_path: Path) -> None:
    required = {"ULLM_TEST_GUARD": "1"}
    digest = hashlib.sha256(b"ullm-aq4-p2-resident-guards-v1\0")
    digest.update(b"ULLM_TEST_GUARD=1\n")
    guard_sha = digest.hexdigest()
    receipt = tmp_path / "guard.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": RUNNER.GUARD_RECEIPT_SCHEMA,
                "status": "ready",
                "guard_sha256": guard_sha,
                "required_environment": required,
            }
        )
        + "\n",
        encoding="ascii",
    )
    assert RUNNER._guard_receipt(receipt, guard_sha)["guard_sha256"] == guard_sha
    value = json.loads(receipt.read_text())
    value["required_environment"]["ULLM_SECOND_GUARD"] = "1"
    receipt.write_text(json.dumps(value) + "\n", encoding="ascii")
    with pytest.raises(RUNNER.HoldoutError, match="SHA contract"):
        RUNNER._guard_receipt(receipt, guard_sha)


def test_build_receipt_binds_clean_git_source_lock_log_and_binary(
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "source"
    worktree.mkdir()
    (worktree / "Cargo.lock").write_text("# lock\n", encoding="ascii")
    subprocess.run(["git", "init", "-q", str(worktree)], check=True)
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree), "config", "user.name", "test"], check=True
    )
    subprocess.run(["git", "-C", str(worktree), "add", "Cargo.lock"], check=True)
    subprocess.run(["git", "-C", str(worktree), "commit", "-qm", "fixture"], check=True)
    binary = tmp_path / "ullm-aq4-fidelity-capture"
    binary.write_bytes(b"capture")
    binary.chmod(0o555)
    log = tmp_path / "build.log"
    log.write_text("ok\n", encoding="ascii")
    receipt = tmp_path / "build-receipt.json"
    value = {
        "schema_version": RUNNER.BUILD_RECEIPT_SCHEMA,
        "status": "ready",
        "source": {
            "commit": subprocess.check_output(
                ["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True
            ).strip(),
            "tree_sha256": subprocess.check_output(
                ["git", "-C", str(worktree), "rev-parse", "HEAD^{tree}"], text=True
            ).strip(),
            "tree_clean": True,
            "cargo_lock_sha256": hashlib.sha256(
                (worktree / "Cargo.lock").read_bytes()
            ).hexdigest(),
        },
        "build": {
            "worktree": str(worktree),
            "command": "cargo build --bin ullm-aq4-fidelity-capture",
            "exit_status": 0,
            "log": str(log),
        },
        "binary": {
            "path": str(binary),
            "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "bytes": binary.stat().st_size,
            "nlink": 1,
            "mode": "0555",
        },
    }
    receipt.write_text(json.dumps(value) + "\n", encoding="ascii")
    assert (
        RUNNER._build_receipt(receipt, binary)["binary"]["sha256"]
        == value["binary"]["sha256"]
    )
    binary.chmod(0o755)
    with pytest.raises(RUNNER.HoldoutError, match="binding"):
        RUNNER._build_receipt(receipt, binary)


def test_package_tree_matches_rust_streaming_contract_and_rejects_hardlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "package"
    (root / "dir").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha")
    (root / "dir/b.bin").write_bytes(bytes([0, 1]))
    assert (
        RUNNER._package_tree(root)["content_sha256"]
        == "0440739e282bc7be23704973be9428815c4e05924b3e66dfd5216e6c3e46913f"
    )
    os.link(root / "a.txt", root / "alias.txt")
    with pytest.raises(RUNNER.HoldoutError, match="hard link"):
        RUNNER._package_tree(root)


def test_external_model_file_census_observes_package_mapping(tmp_path: Path) -> None:
    root = tmp_path / "package"
    root.mkdir()
    weights = root / "weights.bin"
    weights.write_bytes(b"weights-page".ljust(4096, b"\0"))
    with (
        weights.open("rb") as stream,
        mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ),
    ):
        assert "weights.bin" in RUNNER._model_file_census(os.getpid(), root)


def test_execute_seals_pre_spawn_oserror_after_consuming_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = "/bin/false"
    identity = {
        "device_index": 1,
        "served_model_manifest_path": str(tmp_path / "served.json"),
        "served_model_manifest_sha256": "1" * 64,
        "package_manifest_sha256": "2" * 64,
        "worker_binary_sha256": "3" * 64,
        "guard_sha256": "4" * 64,
        "device_architecture": "gfx1201",
        "quantized_artifact_revision": "revision",
    }
    source_cases = str(tmp_path / "cases.json")
    paths = {
        "split_root": str(tmp_path / "split"),
        "source_artifact": str(tmp_path / "source"),
        "active_output": str(tmp_path / "active"),
        "attempt_marker": str(tmp_path / "attempt.json"),
        "result_receipt": str(tmp_path / "result.json"),
    }
    command = [
        binary,
        "--served-model-manifest",
        identity["served_model_manifest_path"],
        "--split-root",
        paths["split_root"],
        "--source",
        paths["source_artifact"],
        "--cases-file",
        source_cases,
        "--output",
        paths["active_output"],
        "--subset",
        "holdout",
        "--device-index",
        "1",
        "--chunk-elements",
        "64",
        "--expected-split-manifest-sha256",
        "a" * 64,
        "--expected-policy-sha256",
        "b" * 64,
        "--expected-calibration-cases-sha256",
        "c" * 64,
        "--expected-holdout-cases-sha256",
        "d" * 64,
        "--expected-served-model-manifest-sha256",
        identity["served_model_manifest_sha256"],
        "--expected-package-manifest-sha256",
        identity["package_manifest_sha256"],
        "--expected-worker-binary-sha256",
        identity["worker_binary_sha256"],
        "--expected-guard-sha256",
        identity["guard_sha256"],
        "--expected-device-architecture",
        "gfx1201",
        "--expected-quantized-artifact-revision",
        "revision",
    ]
    plan = {
        "schema_version": RUNNER.PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "subset": "holdout",
        "row_count": 24,
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {},
        "identity": identity,
        "paths": paths,
        "source_artifact": {"holdout_receipt": {"cases": {"path": source_cases}}},
        "execution_contract": {
            "one_process": True,
            "one_model_load": True,
            "gpu_parallelism": 1,
            "timeout_seconds": 1.0,
            "chunk_elements": 64,
            "capture_binary": {"path": binary},
            "device_environment": {name: "1" for name in RUNNER.DEVICE_ENV},
            "guard_environment": {},
            "ambient_environment_allowlist": list(RUNNER.AMBIENT_ENV_ALLOWLIST),
            "ambient_environment": {},
            "child_environment": {name: "1" for name in RUNNER.DEVICE_ENV},
            "child_environment_sha256": RUNNER._environment_sha(
                {name: "1" for name in RUNNER.DEVICE_ENV}
            ),
            "command": command,
            "command_sha256": RUNNER._command_sha(command),
        },
    }
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(plan) + "\n", encoding="ascii")
    monkeypatch.setattr(
        RUNNER,
        "_revalidate_frozen_plan",
        lambda _plan: (_ for _ in ()).throw(OSError(errno.EIO, "fixture I/O")),
    )
    result = RUNNER._execute(
        type(
            "Args",
            (),
            {"preflight": preflight, "receipt_output": tmp_path / "result.json"},
        )()
    )
    assert result["attempt_consumed"] is True
    assert result["retry_permitted"] is False
    assert result["stage"] == "pre_spawn"
    assert result["errno"] == errno.EIO
    assert (tmp_path / "attempt.json").exists()
    assert (tmp_path / "result.json").stat().st_mode & 0o777 == 0o444


def _minimal_execute_plan(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": RUNNER.PREFLIGHT_SCHEMA,
        "status": "ready_for_execute",
        "subset": "holdout",
        "row_count": 24,
        "split_manifest_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "calibration_cases_sha256": "c" * 64,
        "holdout_cases_sha256": "d" * 64,
        "freeze_receipt_sha256": "e" * 64,
        "actual_verified_receipt": {},
        "identity": {},
        "execution_contract": {"command_sha256": "f" * 64},
        "paths": {
            "attempt_marker": str(tmp_path / "attempt.json"),
            "result_receipt": str(tmp_path / "result.json"),
        },
    }


@pytest.mark.parametrize("fault_stage", ["link", "fsync", "close"])
def test_outer_fail_safe_seals_attempt_publication_exceptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault_stage: str
) -> None:
    plan = _minimal_execute_plan(tmp_path)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(plan) + "\n", encoding="ascii")
    monkeypatch.setattr(RUNNER, "_validate_plan", lambda _plan: None)
    original_atomic = RUNNER._atomic_json

    def faulting_atomic(path: Path, value: object, label: str) -> str:
        digest = original_atomic(path, value, label)
        if label == "attempt marker":
            raise OSError(errno.EIO, f"injected {fault_stage} failure")
        return digest

    monkeypatch.setattr(RUNNER, "_atomic_json", faulting_atomic)
    result = RUNNER._execute(
        type(
            "Args",
            (),
            {"preflight": preflight, "receipt_output": tmp_path / "result.json"},
        )()
    )
    rescue = Path(result["receipt"])
    assert result["failure_kind"] == "fail_safe_rescue"
    assert result["attempt_consumed"] is True
    assert result["retry_permitted"] is False
    assert rescue.is_file()
    assert rescue.stat().st_mode & 0o777 == 0o444
    assert hashlib.sha256(rescue.read_bytes()).hexdigest() == result["receipt_sha256"]


def test_outer_fail_safe_survives_failure_receipt_publication_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _minimal_execute_plan(tmp_path)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(plan) + "\n", encoding="ascii")
    monkeypatch.setattr(RUNNER, "_validate_plan", lambda _plan: None)

    def failed_inner(args: object) -> dict[str, object]:
        RUNNER._atomic_json(
            tmp_path / "attempt.json", {"status": "started"}, "attempt marker"
        )
        raise OSError(errno.EIO, "injected failure publication exception")

    monkeypatch.setattr(RUNNER, "_execute_inner", failed_inner)
    result = RUNNER._execute(
        type(
            "Args",
            (),
            {"preflight": preflight, "receipt_output": tmp_path / "result.json"},
        )()
    )
    assert result["failure_kind"] == "fail_safe_rescue"
    assert Path(result["receipt"]).is_file()
