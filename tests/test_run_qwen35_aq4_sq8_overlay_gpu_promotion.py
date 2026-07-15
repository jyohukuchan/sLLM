from __future__ import annotations

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


def snapshot(tag: str = "same", *, authorized: bool = True) -> dict[str, Any]:
    return {
        "source": {"commit": "a" * 40, "tree": "b" * 40, "archive_sha256": "c" * 64},
        "files": {
            "binding": {"sha256": "d" * 64},
            "package_manifest": {"sha256": "e" * 64},
        },
        "overlay": {"content_sha256": "f" * 64},
        "authorization": {"actual_run_allowed": authorized},
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
    calls: dict[str, Any] = {"stop": 0, "start": 0, "capture": [], "lease": Lease()}

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
        service_snapshot=lambda: next(service_values),
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
