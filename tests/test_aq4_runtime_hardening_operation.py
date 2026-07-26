from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/aq4_runtime_hardening_operation.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


OPERATION = load_module("aq4_runtime_hardening_operation_test", TOOL)


def binding() -> dict[str, str]:
    return {
        "ULLM_AQ4_RUNTIME_HARDENING_STAGE": "candidate_live_proof",
        "ULLM_AQ4_RUNTIME_HARDENING_PLAN_SHA256": "a" * 64,
        "ULLM_AQ4_RUNTIME_HARDENING_EPOCH": "b" * 64,
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST": "/fixture/active.json",
        "ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_SHA256": "c" * 64,
    }


def endpoint_states(*, ready: bool = True) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "ok": ready or name != "gateway_ready",
            "status": 200 if ready or name != "gateway_ready" else None,
            "cause": None if ready or name != "gateway_ready" else "transport",
        }
        for name in OPERATION.ENDPOINTS
    }


def attempt(
    *,
    coherent: bool,
    process: dict[str, Any] | None = None,
    cause: str | None = None,
) -> dict[str, Any]:
    return {
        "systemd": {
            "unit": OPERATION.SERVICE,
            "active_state": "active",
            "sub_state": "running",
        },
        "process": process
        if process is not None
        else {
            "boot_id": "fixture-boot",
            "pid": 42,
            "ppid": 1,
            "starttime": 7,
            "executable_sha256": "d" * 64,
        },
        "manifest": {
            "active_path": "/fixture/active.json",
            "active_manifest_sha256": "c" * 64,
            "file_match": True,
            "worker_environment_match": True,
            "worker_command_match": True,
        },
        "model_id": OPERATION.MODEL_ID,
        "worker_binary_path": "/fixture/worker",
        "worker_binary_sha256": "e" * 64,
        "endpoints": endpoint_states(ready=coherent),
        "coherent": coherent,
        "cause": cause or ("ready" if coherent else "endpoints_incoherent"),
    }


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.value += delay


def test_slow_gateway_retries_then_requires_stable_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    stable = {
        "boot_id": "fixture-boot",
        "pid": 42,
        "ppid": 1,
        "starttime": 7,
        "executable_sha256": "d" * 64,
    }
    observations = iter(
        [
            attempt(coherent=False),
            attempt(coherent=False),
            attempt(coherent=True, process=stable),
            attempt(coherent=True, process=stable),
        ]
    )
    monkeypatch.setattr(OPERATION, "_readiness_attempt", lambda _values, *, deadline: next(observations))

    document = OPERATION.wait_for_readiness(binding(), clock=clock, sleep=clock.sleep)

    assert document["readiness"] == {
        "timeout_seconds": 120,
        "max_attempts": 15,
        "attempts": 4,
        "stable_pid_observations": 2,
        "elapsed_milliseconds": 3500,
    }
    assert clock.sleeps == [0.5, 1.0, 2.0]


def test_partial_endpoint_success_never_counts_as_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(OPERATION, "READINESS_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(
        OPERATION,
        "_readiness_attempt",
        lambda _values, *, deadline: attempt(coherent=False, cause="endpoints_incoherent"),
    )

    with pytest.raises(OPERATION.ReadinessError) as raised:
        OPERATION.wait_for_readiness(binding(), clock=clock, sleep=clock.sleep)

    diagnostic = raised.value.diagnostic
    assert diagnostic["cause"] == "endpoints_incoherent"
    assert diagnostic["endpoints"]["gateway_ready"] == {
        "ok": False,
        "status": None,
        "cause": "transport",
    }
    assert diagnostic["endpoints"]["gateway_health"]["ok"] is True


def test_readiness_attempt_rejects_partial_endpoints_and_model_id_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = binding()
    process = {
        "boot_id": "fixture-boot",
        "pid": 42,
        "ppid": 1,
        "starttime": 7,
        "executable_sha256": "d" * 64,
    }
    contract = {
        "model_id": OPERATION.MODEL_ID,
        "worker_path": "/fixture/worker",
        "worker_hash": "d" * 64,
    }
    monkeypatch.setattr(OPERATION, "_read_active", lambda _values: ({}, b"fixture"))
    monkeypatch.setattr(OPERATION, "_manifest_contract", lambda _manifest: contract)
    monkeypatch.setattr(OPERATION, "_service_identity", lambda: ("active", "running", 7))
    monkeypatch.setattr(OPERATION, "_process_identity", lambda _pid: process)
    monkeypatch.setattr(
        OPERATION,
        "_proc_environment",
        lambda _pid: {"ULLM_SERVED_MODEL_MANIFEST": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]},
    )
    worker_queries: list[int] = []

    def find_worker(service_pid: int, **_kwargs: object) -> int:
        worker_queries.append(service_pid)
        return 42

    monkeypatch.setattr(OPERATION, "_find_worker_pid", find_worker)
    monkeypatch.setattr(OPERATION, "_probe_endpoints", lambda _deadline: endpoint_states(ready=False))

    result = OPERATION._readiness_attempt(values, deadline=100.0)

    assert result["coherent"] is False
    assert result["cause"] == "endpoints_incoherent"
    assert result["endpoints"]["gateway_ready"]["cause"] == "transport"
    assert worker_queries == [7, 7]
    assert OPERATION._probe(
        lambda: (200, b'{"data":[{"id":"a-different-model"}]}'), require_model=True
    ) == {"ok": False, "status": 200, "cause": "model_id_mismatch"}


def test_readiness_attempt_rejects_process_that_changes_during_one_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = binding()
    identities = iter((("active", "running", 42), ("active", "running", 43)))
    monkeypatch.setattr(OPERATION, "_read_active", lambda _values: ({}, b"fixture"))
    monkeypatch.setattr(
        OPERATION,
        "_manifest_contract",
        lambda _manifest: {
            "model_id": OPERATION.MODEL_ID,
            "worker_path": "/fixture/worker",
            "worker_hash": "d" * 64,
        },
    )
    monkeypatch.setattr(OPERATION, "_service_identity", lambda: next(identities))
    monkeypatch.setattr(
        OPERATION,
        "_process_identity",
        lambda pid: {
            "boot_id": "fixture-boot",
            "pid": pid,
            "ppid": 1,
            "starttime": pid,
            "executable_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        OPERATION,
        "_proc_environment",
        lambda _pid: {"ULLM_SERVED_MODEL_MANIFEST": values["ULLM_AQ4_RUNTIME_HARDENING_ACTIVE_MANIFEST"]},
    )
    monkeypatch.setattr(OPERATION, "_find_worker_pid", lambda _pid, **_kwargs: _pid)
    monkeypatch.setattr(OPERATION, "_probe_endpoints", lambda _deadline: endpoint_states(ready=True))

    result = OPERATION._readiness_attempt(values, deadline=100.0)

    assert result["coherent"] is False
    assert result["cause"] == "process_unstable"


def test_isolated_environment_copies_only_the_live_worker_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    live_contract = {
        "model_id": OPERATION.MODEL_ID,
        "worker_path": "/fixture/live-worker",
        "worker_hash": "d" * 64,
    }
    candidate_contract = {"required_environment": ("ULLM_REQUIRE_HIP_FIXTURE",)}
    worker_environment = {
        "HOME": "/fixture/home",
        "XDG_CACHE_HOME": "/fixture/cache",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HIP_VISIBLE_DEVICES": "1",
        "ULLM_HIP_VISIBLE_DEVICES": "1",
        "ULLM_GPU_LOCK_FILE": "/fixture/lock",
        "ULLM_REQUIRE_HIP_FIXTURE": "1",
        "UNRELATED_API_KEY": "must-not-be-copied",
        "ULLM_SERVED_MODEL_MANIFEST": str(OPERATION.LIVE_ACTIVE_MANIFEST),
    }
    monkeypatch.setattr(OPERATION, "_service_identity", lambda: ("active", "running", 7))
    monkeypatch.setattr(OPERATION, "_read_manifest", lambda _path: ({}, b"fixture"))
    monkeypatch.setattr(OPERATION, "_manifest_contract", lambda _manifest: live_contract)
    monkeypatch.setattr(OPERATION, "_find_worker_pid", lambda _pid, **_kwargs: 42)
    monkeypatch.setattr(OPERATION, "_proc_environment", lambda pid: worker_environment if pid == 42 else {})

    environment = OPERATION._isolated_environment(candidate_contract, "/fixture/candidate.json")

    assert environment["ULLM_SERVED_MODEL_MANIFEST"] == "/fixture/candidate.json"
    assert environment["ULLM_REQUIRE_HIP_FIXTURE"] == "1"
    assert environment["HIP_VISIBLE_DEVICES"] == "1"
    assert "UNRELATED_API_KEY" not in environment


def test_endpoint_deadline_preserves_prior_success_state(monkeypatch: pytest.MonkeyPatch) -> None:
    moments = iter((0.0, 0.0, 1.0, 1.0, 1.0, 1.0))
    monkeypatch.setattr(OPERATION.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(OPERATION, "_read_secret", lambda _path: bytearray(b"fixture"))
    monkeypatch.setattr(OPERATION, "_docker_gateway_get", lambda *_args, **_kwargs: (200, b"{}"))
    monkeypatch.setattr(OPERATION, "_openwebui_get", lambda *_args, **_kwargs: (200, b"{}"))

    endpoints = OPERATION._probe_endpoints(1.0)

    assert endpoints["gateway_health"] == {"ok": True, "status": 200, "cause": None}
    for name in OPERATION.ENDPOINTS[1:]:
        assert endpoints[name] == {
            "ok": False,
            "status": None,
            "cause": "deadline_elapsed",
        }


def test_unstable_pid_times_out_even_when_endpoints_are_coherent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = FakeClock()
    monkeypatch.setattr(OPERATION, "READINESS_MAX_ATTEMPTS", 4)
    pid = 100

    def changing_process(_values: dict[str, str], *, deadline: float) -> dict[str, Any]:
        nonlocal pid
        pid += 1
        return attempt(
            coherent=True,
            process={
                "boot_id": "fixture-boot",
                "pid": pid,
                "ppid": 1,
                "starttime": pid,
                "executable_sha256": "d" * 64,
            },
        )

    monkeypatch.setattr(OPERATION, "_readiness_attempt", changing_process)

    with pytest.raises(OPERATION.ReadinessError) as raised:
        OPERATION.wait_for_readiness(binding(), clock=clock, sleep=clock.sleep)

    diagnostic = raised.value.diagnostic
    assert diagnostic["cause"] == "pid_not_stable"
    assert diagnostic["readiness"]["attempts"] == 4
    assert diagnostic["endpoints"]["gateway_ready"]["ok"] is True


def test_deadline_expiry_is_bounded_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    monkeypatch.setattr(OPERATION, "READINESS_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(OPERATION, "READINESS_MAX_ATTEMPTS", 15)
    monkeypatch.setattr(
        OPERATION,
        "_readiness_attempt",
        lambda _values, *, deadline: attempt(coherent=False, cause="service_not_ready"),
    )

    with pytest.raises(OPERATION.ReadinessError) as raised:
        OPERATION.wait_for_readiness(binding(), clock=clock, sleep=clock.sleep)

    diagnostic = raised.value.diagnostic
    assert diagnostic["cause"] == "service_not_ready"
    assert diagnostic["readiness"]["attempts"] < 15
    assert diagnostic["readiness"]["elapsed_milliseconds"] == 1000
    assert clock.sleeps == [0.5, 0.5]


@pytest.mark.parametrize("stage", ("candidate_reconcile", "rollback_reconcile"))
def test_reconcile_uses_the_same_wait_contract_for_candidate_and_rollback(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    values = binding()
    values["ULLM_AQ4_RUNTIME_HARDENING_STAGE"] = stage
    expected = {"schema_version": "fixture-readiness"}
    calls: list[object] = []

    def fake_run(argv: list[str], *, timeout: float, **_kwargs: object) -> object:
        calls.append((argv, timeout))
        return object()

    monkeypatch.setattr(OPERATION, "_run", fake_run)
    monkeypatch.setattr(OPERATION, "wait_for_readiness", lambda actual: expected)

    assert OPERATION.reconcile(values) == expected
    assert calls == [([OPERATION.SYSTEMCTL, "restart", OPERATION.SERVICE], 90)]
