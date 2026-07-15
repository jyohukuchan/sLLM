from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/run-qwen35-aq4-sq8-overlay-gpu-promotion.py"
SPEC = importlib.util.spec_from_file_location("sq8_overlay_gpu_promotion", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Lease:
    def __init__(self) -> None:
        self.released = False

    def evidence(self) -> dict[str, Any]:
        return {"path": "/run/ullm/device-1.lock", "device": 1, "inode": 2, "held": True}

    def release(self) -> None:
        self.released = True


class ReceiptWriter:
    @staticmethod
    def write_actual_receipt(**kwargs: Any) -> None:
        Path(kwargs["output_path"]).write_text(
            '{"status":"actual_verified"}\n', encoding="ascii"
        )

    @staticmethod
    def write_failure_receipt(**kwargs: Any) -> None:
        Path(kwargs["output_path"]).write_text(
            '{"status":"failed"}\n', encoding="ascii"
        )


def readiness() -> dict[str, Any]:
    network_id = "3" * 64
    return {
        "schema": MODULE.READINESS_SCHEMA,
        "container": {
            "name": "open-webui",
            "id": "1" * 64,
            "image_id": "sha256:" + "2" * 64,
            "config_image": "ghcr.io/open-webui/open-webui:v0.6.18",
        },
        "network": {
            "name": "open-webui-network",
            "id": network_id,
            "driver": "bridge",
            "bridge_interface": f"br-{network_id[:12]}",
        },
        "endpoint": {
            "url": MODULE.READY_URL,
            "path": MODULE.READY_PATH,
            "expected_status": 200,
            "expected_body": MODULE.READY_BODY.decode("ascii"),
            "expected_body_sha256": hashlib.sha256(MODULE.READY_BODY).hexdigest(),
            "timeout_seconds": MODULE.READY_TIMEOUT_SECONDS,
        },
    }


def snapshot(tag: str = "same", *, authorized: bool = True) -> dict[str, Any]:
    return {
        "source": {"commit": "a" * 40, "tree": "b" * 40, "archive_sha256": "c" * 64},
        "files": {
            "binding": {"sha256": "d" * 64},
            "package_manifest": {"sha256": "e" * 64},
        },
        "overlay": {"content_sha256": "f" * 64},
        "authorization": {"actual_run_allowed": authorized},
        "readiness": readiness(),
        "tag": tag,
    }


def service(active: bool, *, epoch: int = 100, worker: int = 200) -> dict[str, Any]:
    return {
        "active": active,
        "running": active,
        "main_pid": epoch if active else 0,
        "worker_pid": worker if active else 0,
        "healthy": active,
        "lock_owned": active,
        "control_group": "/system.slice/ullm-openai.service",
    }


def owners(worker: int | None = None) -> dict[str, Any]:
    values = [] if worker is None else [worker]
    return {"worker_pids": values, "amd_pids": values, "kfd_pids": values}


def candidate(tmp_path: Path) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    receipt = root / "promotion-receipt.json"
    receipt.write_text("{}\n", encoding="ascii")
    profile = {
        "worker": {"required_environment": list(MODULE.REQUIRED_OVERLAY_ENV)},
        "promotion": {"receipt": str(receipt)},
    }
    (root / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (root / "gate.json").write_text(
        json.dumps(
            {
                "request": {
                    "actual": {"request_id": "sq8-promotion-" + "a" * 64}
                }
            }
        ),
        encoding="utf-8",
    )
    return root


def dependencies(
    tmp_path: Path,
    *,
    capture_code: int = 0,
    stop_error: bool = False,
    start_error: bool = False,
) -> tuple[Any, dict[str, Any]]:
    service_values = iter(
        [
            service(True),
            service(False),
            service(False),
            service(True, epoch=101, worker=201),
        ]
    )
    owner_values = iter([owners(), owners(), owners(201)])
    calls: dict[str, Any] = {
        "stop": 0,
        "start": 0,
        "capture": [],
        "lease": Lease(),
        "readiness": [],
    }

    def service_probe(bound_readiness: dict[str, Any]) -> dict[str, Any]:
        calls["readiness"].append(bound_readiness)
        return next(service_values)

    def stop() -> None:
        calls["stop"] += 1
        if stop_error:
            raise MODULE.PromotionError("injected stop failure")

    def start() -> None:
        calls["start"] += 1
        if start_error:
            raise MODULE.PromotionError("injected restore failure")

    def capture_run(argv: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        calls["capture"].append({"argv": argv, "environment": environment})
        output = Path(argv[argv.index("--output") + 1])
        if capture_code == 0:
            output.write_text("{}\n", encoding="utf-8")
        stdout = json.dumps({"status": "ok", "output": str(output)}) if capture_code == 0 else ""
        return subprocess.CompletedProcess(argv, capture_code, stdout=stdout, stderr="")

    deps = MODULE.Dependencies(
        service_snapshot=service_probe,
        owner_snapshot=lambda: next(owner_values),
        stop_service=stop,
        start_service=start,
        acquire_lock=lambda: calls["lease"],
        capture=capture_run,
        monotonic=lambda: 0.0,
        sleep=lambda _: None,
    )
    return deps, calls


def prepare(monkeypatch: pytest.MonkeyPatch, values: list[dict[str, Any]] | None = None) -> None:
    snapshots = iter(values or [snapshot(), snapshot()])
    monkeypatch.setattr(MODULE, "candidate_snapshot", lambda _: next(snapshots))
    monkeypatch.setattr(
        MODULE,
        "validate_executor_record",
        lambda path, identity, request_id: {"status": "ok"},
    )
    monkeypatch.setattr(MODULE, "load_receipt_writer", lambda: ReceiptWriter)


def docker_runner(
    contract: dict[str, Any],
    *,
    curl_status: int = 200,
    curl_body: str | None = None,
    curl_returncode: int = 0,
    curl_timeout: bool = False,
    container_overrides: dict[str, Any] | None = None,
    network_overrides: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    container = contract["container"]
    network = contract["network"]
    observed_container = {
        "id": container["id"],
        "name": "/" + container["name"],
        "image_id": container["image_id"],
        "config_image": container["config_image"],
        "networks": {
            network["name"]: {"NetworkID": network["id"]},
        },
    }
    observed_network = {
        "id": network["id"],
        "name": network["name"],
        "driver": network["driver"],
        "options": {
            "com.docker.network.bridge.name": network["bridge_interface"],
        },
        "containers": {
            container["id"]: {"Name": container["name"]},
        },
    }
    if container_overrides:
        observed_container.update(container_overrides)
    if network_overrides:
        observed_network.update(network_overrides)
    calls: list[dict[str, Any]] = []

    def run(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, "timeout": timeout})
        if argv[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(observed_container), stderr=""
            )
        if argv[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(
                argv, 0, stdout=json.dumps(observed_network), stderr=""
            )
        assert argv[:2] == ["docker", "exec"]
        if curl_timeout:
            raise subprocess.TimeoutExpired(argv, timeout)
        body = contract["endpoint"]["expected_body"] if curl_body is None else curl_body
        return subprocess.CompletedProcess(
            argv,
            curl_returncode,
            stdout=f"{body}\n{curl_status}",
            stderr="" if curl_returncode == 0 else "curl failed",
        )

    return run, calls


def test_docker_readiness_is_exact_and_uses_full_gate_bound_identity() -> None:
    contract = readiness()
    runner, calls = docker_runner(contract)

    assert MODULE._ready(contract, runner, lambda _: True) is True
    assert len(calls) == 3
    assert calls[0]["argv"][-1] == contract["container"]["id"]
    assert calls[1]["argv"][-1] == contract["network"]["id"]
    assert calls[2]["argv"][:3] == [
        "docker", "exec", contract["container"]["id"]
    ]
    assert calls[2]["argv"][-1] == contract["endpoint"]["url"]
    assert all(call["timeout"] == MODULE.READY_TIMEOUT_SECONDS for call in calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"curl_status": 503},
        {"curl_body": '{"status":"starting"}'},
        {"curl_returncode": 7},
        {"curl_timeout": True},
    ],
)
def test_docker_readiness_rejects_status_body_nonzero_and_timeout(
    kwargs: dict[str, Any]
) -> None:
    contract = readiness()
    runner, _calls = docker_runner(contract, **kwargs)

    assert MODULE._ready(contract, runner, lambda _: True) is False


def test_docker_readiness_rejects_container_identity_mismatch() -> None:
    contract = readiness()
    runner, calls = docker_runner(contract, container_overrides={"id": "9" * 64})

    with pytest.raises(MODULE.PromotionError, match="container identity differs"):
        MODULE._ready(contract, runner, lambda _: True)
    assert len(calls) == 2


def test_docker_readiness_rejects_network_identity_mismatch() -> None:
    contract = readiness()
    runner, calls = docker_runner(contract, network_overrides={"driver": "overlay"})

    with pytest.raises(MODULE.PromotionError, match="network identity differs"):
        MODULE._ready(contract, runner, lambda _: True)
    assert len(calls) == 2


def test_readiness_contract_rejects_aliases_and_weak_endpoint() -> None:
    contract = readiness()
    contract["container"]["image_digest"] = contract["container"].pop("image_id")
    with pytest.raises(MODULE.PromotionError, match="container identity"):
        MODULE.validate_readiness_contract(contract)

    contract = readiness()
    contract["endpoint"]["expected_body"] = {"status": "ready"}
    with pytest.raises(MODULE.PromotionError, match="endpoint contract"):
        MODULE.validate_readiness_contract(contract)


def test_success_runs_candidate_once_and_restores_new_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch)
    root = candidate(tmp_path)
    output = tmp_path / "evidence"
    deps, calls = dependencies(tmp_path)

    code, evidence = MODULE.execute(root, output, deps)

    assert code == 0
    assert evidence["status"] == "passed"
    assert evidence["actual_run_count"] == 1
    assert evidence["restore"]["passed"] is True
    assert calls["stop"] == calls["start"] == 1
    assert calls["lease"].released is True
    assert len(calls["readiness"]) == 4
    assert all(value == readiness() for value in calls["readiness"])
    assert len(calls["capture"]) == 1
    invocation = calls["capture"][0]
    assert invocation["argv"][-2:] == [
        "--sq8-promotion-request-id",
        "sq8-promotion-" + "a" * 64,
    ]
    assert invocation["environment"]["HIP_VISIBLE_DEVICES"] == "1"
    assert invocation["environment"]["ULLM_HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in invocation["environment"]
    assert {path.name for path in output.iterdir()} == {
        "maintenance-evidence.json",
        "executor-record.json",
        "promotion-actual-receipt.json",
        "SHA256SUMS",
    }
    sums = (output / "SHA256SUMS").read_text(encoding="ascii")
    assert "maintenance-evidence.json" in sums and "executor-record.json" in sums
    assert not (output / "promotion-failure-receipt.json").exists()


def test_capture_failure_still_releases_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot()])
    deps, calls = dependencies(tmp_path, capture_code=9)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "failure", deps)

    assert code == 1
    assert evidence["status"] == "failed"
    assert evidence["actual_run_count"] == 1
    assert evidence["restore"]["passed"] is True
    assert calls["lease"].released is True
    assert calls["start"] == 1
    assert (tmp_path / "failure" / "promotion-failure-receipt.json").is_file()
    assert not (tmp_path / "failure" / "promotion-actual-receipt.json").exists()


def test_stop_failure_attempts_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot()])
    deps, calls = dependencies(tmp_path, stop_error=True)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "stop-failure", deps)

    assert code == 1
    assert evidence["restore"]["attempted"] is True
    assert evidence["restore"]["passed"] is True
    assert calls["start"] == 1
    assert calls["capture"] == []


def test_restore_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch)
    deps, calls = dependencies(tmp_path, start_error=True)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "restore-failure", deps)

    assert code == 1
    assert evidence["status"] == "failed"
    assert evidence["restore"]["attempted"] is True
    assert evidence["restore"]["passed"] is False
    assert calls["lease"].released is True


def test_candidate_identity_change_is_terminal_but_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot(), snapshot("changed")])
    deps, calls = dependencies(tmp_path)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "identity-failure", deps)

    assert code == 1
    assert "identity changed" in evidence["failure"]["reason"]
    assert evidence["restore"]["passed"] is True
    assert calls["start"] == 1


def test_create_new_output_rejects_existing_directory(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(MODULE.PromotionError, match="create-new"):
        MODULE.finalize_directory(output, {"record.json": {"status": "ok"}})


def test_execute_rejects_unauthorized_candidate_before_service_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot(authorized=False)])
    deps, calls = dependencies(tmp_path)

    with pytest.raises(MODULE.PromotionError, match="not authorized"):
        MODULE.execute(candidate(tmp_path), tmp_path / "forbidden", deps)

    assert calls["stop"] == calls["start"] == 0
    assert calls["capture"] == []
