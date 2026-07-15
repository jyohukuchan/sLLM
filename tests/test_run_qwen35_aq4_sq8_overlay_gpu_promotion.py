from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import stat
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
    def __init__(self, *, release_error: bool = False) -> None:
        self.released = False
        self.release_error = release_error

    def evidence(self) -> dict[str, Any]:
        return {"path": "/run/ullm/device-1.lock", "device": 1, "inode": 2, "held": True}

    def release(self) -> None:
        if self.release_error:
            raise MODULE.PromotionError("injected cleanup failure")
        self.released = True


class ReceiptWriter:
    @staticmethod
    def write_actual_receipt(**kwargs: Any) -> None:
        Path(kwargs["output_path"]).write_text(
            '{"status":"actual_verified"}\n', encoding="ascii"
        )

    @staticmethod
    def write_failure_receipt(**kwargs: Any) -> None:
        maintenance = Path(kwargs["maintenance_evidence_path"])
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "status": "failed",
                    "actual": {
                        "maintenance_evidence": {
                            "path": maintenance.name,
                            "sha256": hashlib.sha256(maintenance.read_bytes()).hexdigest(),
                        }
                    },
                }
            )
            + "\n",
            encoding="ascii",
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
        "nrestarts": 0,
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


def worker_stderr_envelope(preview: str = "worker failed\n") -> dict[str, Any]:
    raw = preview.encode("utf-8")
    return {
        "schema_version": MODULE.WORKER_STDERR_SCHEMA,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "preview_text": preview,
        "captured_bytes": len(raw),
        "truncated": False,
        "utf8_replacement": "\ufffd" in preview,
        "redacted_lines": preview.count("<redacted sensitive diagnostic line>"),
        "complete": True,
        "stream_error": None,
    }


def capture_error_envelope(
    *, preview: str = "worker failed\n", stage: str = "worker_exit"
) -> dict[str, Any]:
    return {
        "schema_version": MODULE.CAPTURE_ERROR_SCHEMA,
        "status": "failed",
        "stage": stage,
        "reason": "resident worker exited with status 7",
        "timed_out": False,
        "worker_returncode": 7,
        "worker_signal": None,
        "worker_stderr": worker_stderr_envelope(preview),
    }


def capture_stream(raw: bytes, *, parse: bool = True) -> Any:
    collector = MODULE._CaptureStreamCollector(retain_parse_buffer=parse)
    collector.feed(raw)
    collector.finish()
    return collector.result(False)


def dependencies(
    tmp_path: Path,
    *,
    capture_code: int = 0,
    stop_error: bool = False,
    start_error: bool = False,
    acquire_error: bool = False,
    cleanup_error: bool = False,
    capture_stdout: str | bytes | None = None,
    capture_stderr: str | bytes = "",
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
        "lease": Lease(release_error=cleanup_error),
        "acquire": 0,
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
        stdout = (
            json.dumps({"status": "ok", "output": str(output)})
            if capture_code == 0
            else ""
        ) if capture_stdout is None else capture_stdout
        return subprocess.CompletedProcess(
            argv, capture_code, stdout=stdout, stderr=capture_stderr
        )

    def acquire() -> Lease:
        calls["acquire"] += 1
        if acquire_error:
            raise MODULE.PromotionError("injected acquire failure")
        return calls["lease"]

    deps = MODULE.Dependencies(
        service_snapshot=service_probe,
        owner_snapshot=lambda: next(owner_values),
        stop_service=stop,
        start_service=start,
        acquire_lock=acquire,
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
    assert "capture_failure" not in evidence
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
    stderr = (
        b"worker initialization failed: invalid device \xff\n"
        b"API_KEY=do-not-persist\n"
        b"token=also-do-not-persist\n"
        + b"x" * (MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES + 100)
    )
    deps, calls = dependencies(
        tmp_path,
        capture_code=9,
        capture_stdout=b"Authorization: Bearer do-not-persist\n" * 5000,
        capture_stderr=stderr,
    )

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "failure", deps)

    assert code == 1
    assert evidence["status"] == "failed"
    assert evidence["actual_run_count"] == 1
    assert evidence["restore"]["passed"] is True
    assert calls["lease"].released is True
    assert calls["start"] == 1
    output = tmp_path / "failure"
    assert (output / "promotion-failure-receipt.json").is_file()
    assert not (tmp_path / "failure" / "promotion-actual-receipt.json").exists()
    diagnostic = evidence["capture_failure"]
    assert diagnostic["stage"] == "capture_subprocess_completed"
    assert diagnostic["returncode"] == 9
    assert diagnostic["signal"] is None
    stderr_source = diagnostic["stderr"]["source"]
    stderr_display = diagnostic["stderr"]["display"]
    assert stderr_source["byte_count"] == len(stderr)
    assert stderr_source["prefix_truncated"] is True
    assert stderr_source["captured_prefix_bytes"] == MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
    assert stderr_source["sha256"] == hashlib.sha256(stderr).hexdigest()
    assert "API_KEY" not in stderr_display["text"]
    assert "do-not-persist" not in stderr_display["text"]
    assert "also-do-not-persist" not in stderr_display["text"]
    assert "<redacted sensitive diagnostic line>" in stderr_display["text"]
    assert "\ufffd" in stderr_display["text"]
    for stream_name in ("stdout", "stderr"):
        stream = diagnostic[stream_name]
        serialized = json.dumps(
            stream,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        assert len(serialized) <= MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
        assert stream["display"]["serialized_byte_count"] == len(serialized)
        assert "do-not-persist" not in stream["display"]["text"]
    persisted = json.loads((output / "maintenance-evidence.json").read_text())
    assert persisted["capture_failure"] == diagnostic
    failure_receipt = json.loads(
        (output / "promotion-failure-receipt.json").read_text()
    )
    maintenance_ref = failure_receipt["actual"]["maintenance_evidence"]
    assert maintenance_ref["sha256"] == MODULE.sha_file(
        output / "maintenance-evidence.json"
    )
    sums = (output / "SHA256SUMS").read_text(encoding="ascii")
    assert f'{maintenance_ref["sha256"]}  maintenance-evidence.json\n' in sums
    for path in output.iterdir():
        metadata = path.stat(follow_symlinks=False)
        assert not path.is_symlink()
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) in {0o444, 0o555}


def test_capture_signal_and_timeout_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot()])
    deps, _calls = dependencies(
        tmp_path,
        capture_code=-9,
        capture_stderr="worker killed",
    )
    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "signal", deps)
    assert code == 1
    assert evidence["capture_failure"]["stage"] == "capture_subprocess_completed"
    assert evidence["capture_failure"]["returncode"] == -9
    assert evidence["capture_failure"]["signal"] == {
        "number": 9,
        "name": "SIGKILL",
    }

    prepare(monkeypatch, [snapshot()])
    timeout_root = tmp_path / "timeout-case"
    timeout_root.mkdir()
    deps, _calls = dependencies(timeout_root)

    def timeout(argv: list[str], environment: dict[str, str]) -> Any:
        raise subprocess.TimeoutExpired(
            argv,
            300,
            output=b"partial\xff",
            stderr=b"password=hunter2\nstartup timed out",
        )

    deps.capture = timeout
    code, evidence = MODULE.execute(
        candidate(timeout_root), tmp_path / "timeout", deps
    )
    assert code == 1
    diagnostic = evidence["capture_failure"]
    assert diagnostic["stage"] == "capture_subprocess_timeout"
    assert diagnostic["returncode"] is None
    assert diagnostic["signal"] is None
    assert diagnostic["timeout_seconds"] == 300.0
    assert "hunter2" not in diagnostic["stderr"]["display"]["text"]
    assert evidence["actual_run_count"] == 1


@pytest.mark.parametrize(
    ("raw", "prefix_truncated"),
    [
        (b"token=x\n" * 4000, False),
        (b"password=x\n" * 2000, False),
        (b"\xff" * 100000, True),
        (b"x" * 100000, True),
        (b"ordinary diagnostic line\n" * 10000, True),
    ],
)
def test_bounded_diagnostic_recaps_redacted_and_invalid_display(
    raw: bytes, prefix_truncated: bool
) -> None:
    value = MODULE._bounded_diagnostic(raw)
    serialized = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")

    assert len(serialized) <= MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
    assert value["display"]["serialized_byte_count"] == len(serialized)
    assert value["source"]["byte_count"] == len(raw)
    assert value["source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert value["source"]["captured_prefix_bytes"] == min(
        len(raw), MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
    )
    assert value["source"]["prefix_truncated"] is prefix_truncated
    assert value["display"]["truncated_after_redaction"] is True
    assert "token=x" not in value["display"]["text"]
    assert "password=x" not in value["display"]["text"]


def test_exact_capture_error_envelope_preserves_large_worker_structure() -> None:
    preview = "\ufffd" * 10000
    value = capture_error_envelope(preview=preview)
    raw = json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    assert len(raw) > MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES

    parsed = MODULE._capture_error_envelope(capture_stream(raw))

    assert parsed["validation"] == "valid"
    worker = parsed["worker_stderr"]
    assert worker["byte_count"] == len(preview.encode("utf-8"))
    assert worker["sha256"] == hashlib.sha256(preview.encode("utf-8")).hexdigest()
    display = worker["preview"]
    serialized = json.dumps(
        display,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert len(serialized) <= MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
    assert display["display"]["serialized_byte_count"] == len(serialized)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("unknown", "capture_error_envelope_keys_differ"),
        ("unknown-stage", "capture_error_envelope_type_or_value_differs"),
        ("stage", "capture_error_envelope_stage_terminal_mismatch"),
        ("worker-key", "worker_stderr_keys_differ"),
        ("worker-type", "worker_stderr_type_or_value_differs"),
        ("incomplete", "worker_stderr_incomplete"),
    ],
)
def test_capture_error_envelope_fails_closed_on_shape_and_stage(
    mutation: str, reason: str
) -> None:
    value = capture_error_envelope()
    if mutation == "unknown":
        value["unknown"] = True
    elif mutation == "unknown-stage":
        value["stage"] = "invented_stage"
    elif mutation == "stage":
        value["worker_returncode"] = 0
    elif mutation == "worker-key":
        value["worker_stderr"]["unknown"] = True
    elif mutation == "worker-type":
        value["worker_stderr"]["byte_count"] = True
    else:
        value["worker_stderr"]["complete"] = False
        value["worker_stderr"]["stream_error"] = "drain incomplete"
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")

    parsed = MODULE._capture_error_envelope(capture_stream(raw))

    assert parsed["validation"] == "invalid"
    assert parsed.get("validation_reason", parsed.get("reason")) == reason
    if mutation == "incomplete":
        assert parsed["worker_stderr"]["complete"] is False
        assert parsed["worker_stderr"]["stream_error"] == "drain incomplete"


def test_capture_error_envelope_rejects_duplicate_invalid_and_truncated() -> None:
    value = capture_error_envelope()
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    duplicate = raw.replace(b'{"schema_version":', b'{"status":"failed","schema_version":', 1)
    assert MODULE._capture_error_envelope(capture_stream(duplicate))["reason"] == (
        "capture_error_envelope_duplicate_key"
    )
    assert MODULE._capture_error_envelope(capture_stream(b"{\xff"))["reason"] == (
        "capture_error_envelope_invalid_json"
    )
    stream = capture_stream(raw)
    stream.parse_buffer_truncated = True
    assert MODULE._capture_error_envelope(stream)["reason"] == (
        "capture_error_envelope_truncated"
    )


def test_capture_error_envelope_preserves_worker_signal_and_timeout() -> None:
    signaled = capture_error_envelope()
    signaled["worker_returncode"] = -signal.SIGKILL
    signaled["worker_signal"] = signal.SIGKILL
    parsed = MODULE._capture_error_envelope(
        capture_stream(json.dumps(signaled).encode("utf-8"))
    )
    assert parsed["validation"] == "valid"
    assert parsed["worker_returncode"] == -signal.SIGKILL
    assert parsed["worker_signal"] == signal.SIGKILL

    timed_out = capture_error_envelope(stage="request")
    timed_out["timed_out"] = True
    timed_out["worker_returncode"] = -signal.SIGKILL
    timed_out["worker_signal"] = signal.SIGKILL
    parsed = MODULE._capture_error_envelope(
        capture_stream(json.dumps(timed_out).encode("utf-8"))
    )
    assert parsed["validation"] == "valid"
    assert parsed["timed_out"] is True


def test_default_capture_streams_large_fake_tool_and_preserves_raw_identity() -> None:
    preview = (
        "non-JSON worker cause \ufffd\n"
        "<redacted sensitive diagnostic line>\n"
        + "ordinary worker detail\n" * 1400
    )
    preview = preview.encode("utf-8")[: MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES].decode(
        "utf-8", errors="ignore"
    )
    envelope = capture_error_envelope(preview=preview)
    payload = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))
    script = (
        "import os,sys;"
        "sys.stdout.write(sys.argv[1]+'\\n');sys.stdout.flush();"
        "raw=b'non-json\\xff\\nAPI_KEY=do-not-persist\\n'+b'x'*100000;"
        "sys.stderr.buffer.write(raw);sys.stderr.flush();sys.exit(7)"
    )

    result = MODULE.default_capture([sys.executable, "-c", script, payload], dict(os.environ))

    assert result.returncode == 7 and result.timed_out is False
    assert result.stdout.byte_count == len((payload + "\n").encode("utf-8"))
    assert result.stdout.byte_count > MODULE.CAPTURE_DIAGNOSTIC_MAX_BYTES
    expected_stderr = b"non-json\xff\nAPI_KEY=do-not-persist\n" + b"x" * 100000
    assert result.stderr.byte_count == len(expected_stderr)
    assert result.stderr.sha256 == hashlib.sha256(expected_stderr).hexdigest()
    stderr_evidence = MODULE._stream_diagnostic(result.stderr)
    assert "do-not-persist" not in stderr_evidence["display"]["text"]
    parsed = MODULE._capture_error_envelope(result.stdout)
    assert parsed["validation"] == "valid"
    assert parsed["worker_stderr"]["sha256"] == envelope["worker_stderr"]["sha256"]


def test_default_capture_success_status_is_unchanged() -> None:
    status = {"status": "ok", "output": "/tmp/executor-record.json"}
    result = MODULE.default_capture(
        [
            sys.executable,
            "-c",
            "import json,sys; print(json.dumps(json.loads(sys.argv[1])))",
            json.dumps(status),
        ],
        dict(os.environ),
    )
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.stderr.byte_count == 0
    assert result.stdout.complete is True
    assert MODULE._unique_json_object(result.stdout.parse_buffer) == status


def test_default_capture_preserves_timeout_signal_and_bounded_malformed_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(MODULE, "CAPTURE_SUBPROCESS_TIMEOUT_SECONDS", 0.05)
    timeout = MODULE.default_capture(
        [sys.executable, "-c", "import time; print('partial', flush=True); time.sleep(10)"],
        dict(os.environ),
    )
    assert timeout.timed_out is True
    assert timeout.timeout_seconds == 0.05
    assert timeout.returncode == -signal.SIGKILL
    assert timeout.stdout.sha256 == hashlib.sha256(b"partial\n").hexdigest()

    killed = MODULE.default_capture(
        [sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGTERM)"],
        dict(os.environ),
    )
    assert killed.timed_out is False
    assert killed.returncode == -signal.SIGTERM

    malformed_raw = b"not-json\xff\n" + b"z" * 600000
    malformed = MODULE.default_capture(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'not-json\\xff\\n'+b'z'*600000); sys.exit(3)",
        ],
        dict(os.environ),
    )
    assert malformed.stdout.byte_count == len(malformed_raw)
    assert malformed.stdout.sha256 == hashlib.sha256(malformed_raw).hexdigest()
    assert malformed.stdout.parse_buffer_truncated is True
    assert MODULE._capture_error_envelope(malformed.stdout)["reason"] == (
        "capture_error_envelope_truncated"
    )


def test_real_fake_capture_tool_error_binds_worker_stderr_to_final_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot()])
    deps, _calls = dependencies(tmp_path)
    preview = "non-JSON worker cause \ufffd\n<redacted sensitive diagnostic line>\n"
    envelope = capture_error_envelope(preview=preview)
    payload = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))
    script = (
        "import sys;sys.stdout.write(sys.argv[1]+'\\n');sys.stdout.flush();"
        "sys.stderr.buffer.write(b'API_KEY=do-not-persist\\n'+b'q'*100000);"
        "sys.stderr.flush();sys.exit(7)"
    )

    def real_capture(_argv: list[str], _environment: dict[str, str]) -> Any:
        return MODULE.default_capture(
            [sys.executable, "-c", script, payload], dict(os.environ)
        )

    deps.capture = real_capture
    output = tmp_path / "real-fake-failure"
    code, evidence = MODULE.execute(candidate(tmp_path), output, deps)

    assert code == 1
    tool_error = evidence["capture_failure"]["capture_tool_error"]
    assert tool_error["validation"] == "valid"
    assert tool_error["worker_stderr"]["sha256"] == envelope["worker_stderr"]["sha256"]
    assert "do-not-persist" not in evidence["capture_failure"]["stderr"]["display"]["text"]
    persisted = json.loads((output / "maintenance-evidence.json").read_text())
    assert persisted["capture_failure"]["capture_tool_error"] == tool_error
    receipt = json.loads((output / "promotion-failure-receipt.json").read_text())
    maintenance_sha = MODULE.sha_file(output / "maintenance-evidence.json")
    assert receipt["actual"]["maintenance_evidence"]["sha256"] == maintenance_sha
    assert f"{maintenance_sha}  maintenance-evidence.json\n" in (
        output / "SHA256SUMS"
    ).read_text(encoding="ascii")
    for path in output.iterdir():
        metadata = path.stat(follow_symlinks=False)
        assert metadata.st_nlink == 1
        assert stat.S_IMODE(metadata.st_mode) == 0o444


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
    assert calls["acquire"] == 0


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


def test_acquire_failure_restores_without_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot()])
    deps, calls = dependencies(tmp_path, acquire_error=True)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "acquire-failure", deps)

    assert code == 1 and evidence["restore"]["passed"] is True
    assert calls["acquire"] == 1 and calls["capture"] == [] and calls["start"] == 1


def test_cleanup_failure_still_restores_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch)
    deps, calls = dependencies(tmp_path, cleanup_error=True)

    code, evidence = MODULE.execute(candidate(tmp_path), tmp_path / "cleanup-failure", deps)

    assert code == 1 and evidence["restore"]["passed"] is True
    assert "cleanup failure" in evidence["failure"]["reason"]
    assert calls["start"] == 1


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


@pytest.mark.parametrize("kind", ["output-symlink", "staging-directory", "staging-symlink"])
def test_finalize_rejects_preexisting_and_symlink_paths(
    tmp_path: Path, kind: str
) -> None:
    output = tmp_path / "evidence"
    staging = tmp_path / ".evidence.incomplete"
    target = tmp_path / "target"
    target.mkdir()
    if kind == "output-symlink":
        output.symlink_to(target, target_is_directory=True)
    elif kind == "staging-directory":
        staging.mkdir()
    else:
        staging.symlink_to(target, target_is_directory=True)
    with pytest.raises(MODULE.PromotionError):
        MODULE.finalize_directory(output, {"record.json": {"status": "ok"}})


def test_finalize_rejects_hardlinked_receipt_and_unsafe_document_name(
    tmp_path: Path,
) -> None:
    output = tmp_path / "hardlink"
    external = tmp_path / "external.json"
    external.write_text("{}\n", encoding="ascii")

    def linked_receipt(staging: Path) -> str:
        os.link(external, staging / "receipt.json")
        return "receipt.json"

    with pytest.raises(MODULE.PromotionError, match="topology"):
        MODULE.finalize_directory(
            output, {"record.json": {"status": "ok"}}, linked_receipt
        )
    assert external.stat().st_nlink == 1
    assert not output.exists()

    with pytest.raises(MODULE.PromotionError, match="name is unsafe"):
        MODULE.finalize_directory(
            tmp_path / "unsafe", {"../escape.json": {"status": "bad"}}
        )


def test_execute_rejects_unauthorized_candidate_before_service_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare(monkeypatch, [snapshot(authorized=False)])
    deps, calls = dependencies(tmp_path)

    with pytest.raises(MODULE.PromotionError, match="not authorized"):
        MODULE.execute(candidate(tmp_path), tmp_path / "forbidden", deps)

    assert calls["stop"] == calls["start"] == 0
    assert calls["capture"] == []


def _lock_runner(
    lock_path: Path, *, wrong_mode: bool = False, cleanup_failure: bool = False
) -> tuple[Any, list[list[str]]]:
    calls: list[list[str]] = []

    def runner(argv: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[3] == "create":
            lock_path.parent.mkdir(mode=0o750)
            lock_path.parent.chmod(0o750)
            lock_path.write_bytes(b"")
            lock_path.chmod(0o644 if wrong_mode else 0o600)
            lock = lock_path.stat(follow_symlinks=False)
            directory = lock_path.parent.stat(follow_symlinks=False)
            value = {
                "status": "created",
                "runtime_directory_created": True,
                "runtime_directory": {
                    "path": str(lock_path.parent), "device": directory.st_dev,
                    "inode": directory.st_ino, "mode": "0750", "uid": os.getuid(),
                    "gid": os.getgid(), "nlink": directory.st_nlink,
                },
                "lock": {
                    "path": str(lock_path), "device": lock.st_dev, "inode": lock.st_ino,
                    "mode": "0600", "uid": os.getuid(), "gid": os.getgid(), "nlink": 1,
                },
            }
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value), stderr="")
        if cleanup_failure:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="injected")
        device = int(argv[5])
        inode = int(argv[7])
        lock_path.unlink()
        lock_path.parent.rmdir()
        value = {"status": "removed", "device": device, "inode": inode, "runtime_directory_removed": True}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(value), stderr="")

    return runner, calls


def test_lock_helper_exact_create_flock_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "run" / "device-1.lock"
    monkeypatch.setattr(MODULE, "LOCK_PATH", lock_path)
    monkeypatch.setattr(MODULE, "LOCK_UID", os.getuid())
    monkeypatch.setattr(MODULE, "LOCK_GID", os.getgid())
    runner, calls = _lock_runner(lock_path)

    lease = MODULE.acquire_lock(lock_path, runner)
    assert lease.evidence()["held"] is True
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    lease.release()

    assert calls[0] == ["sudo", "-n", str(MODULE.LOCK_HELPER), "create"]
    assert calls[1][:4] == ["sudo", "-n", str(MODULE.LOCK_HELPER), "remove"]
    assert not lock_path.parent.exists()


def test_lock_helper_rejects_non_whitelisted_argv() -> None:
    called = False

    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(MODULE.PromotionError, match="not whitelisted"):
        MODULE._lock_helper_result(["sudo", "-n", str(MODULE.LOCK_HELPER), "shell"], runner, "bad")
    assert called is False


def test_lock_acquire_rejects_eacces_and_wrong_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "run" / "device-1.lock"
    monkeypatch.setattr(MODULE, "LOCK_PATH", lock_path)
    monkeypatch.setattr(MODULE, "LOCK_UID", os.getuid())
    monkeypatch.setattr(MODULE, "LOCK_GID", os.getgid())
    def denied(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="EACCES")
    with pytest.raises(MODULE.PromotionError, match="helper create failed"):
        MODULE.acquire_lock(lock_path, denied)

    runner, _ = _lock_runner(lock_path, wrong_mode=True)
    with pytest.raises(MODULE.PromotionError, match="lock substrate"):
        MODULE.acquire_lock(lock_path, runner)
    assert not lock_path.parent.exists()


def test_lock_cleanup_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "run" / "device-1.lock"
    monkeypatch.setattr(MODULE, "LOCK_PATH", lock_path)
    monkeypatch.setattr(MODULE, "LOCK_UID", os.getuid())
    monkeypatch.setattr(MODULE, "LOCK_GID", os.getgid())
    runner, _ = _lock_runner(lock_path, cleanup_failure=True)
    lease = MODULE.acquire_lock(lock_path, runner)
    with pytest.raises(MODULE.PromotionError, match="helper remove failed"):
        lease.release()


def test_restore_retries_transient_topology_and_reports_attempts() -> None:
    clock = [0.0]
    services: list[Any] = [
        MODULE.TransientRestoreError("active service process topology differs"),
        service(True, epoch=101, worker=201),
    ]

    def service_snapshot(_: dict[str, Any]) -> dict[str, Any]:
        value = services.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    deps = MODULE.Dependencies(
        service_snapshot=service_snapshot,
        owner_snapshot=lambda: owners(201),
        stop_service=lambda: None,
        start_service=lambda: None,
        acquire_lock=lambda: Lease(),
        capture=lambda argv, env: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = MODULE.poll_restored(deps, service(True), readiness())
    assert result["passed"] is True
    assert result["attempts"] == 2
    assert result["elapsed_seconds"] == MODULE.POLL_SECONDS
    assert result["last_failure"] is None
    assert result["observations"][0] == {
        "transient_failure": "active service process topology differs"
    }


def test_restore_timeout_preserves_last_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [0.0]
    monkeypatch.setattr(MODULE, "RESTORE_TIMEOUT_SECONDS", 0.5)
    deps = MODULE.Dependencies(
        service_snapshot=lambda _: service(False),
        owner_snapshot=lambda: owners(),
        stop_service=lambda: None,
        start_service=lambda: None,
        acquire_lock=lambda: Lease(),
        capture=lambda argv, env: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )
    result = MODULE.poll_restored(deps, service(True), readiness())
    assert result["passed"] is False
    assert result["attempts"] == 2
    assert result["elapsed_seconds"] == 0.5
    assert result["last_failure"] == "service is not active/running yet"


@pytest.mark.parametrize(
    "error",
    [
        MODULE.PromotionError("readiness container identity differs from Gate"),
        OSError("owner source unavailable"),
    ],
)
def test_restore_terminal_identity_or_unexpected_error_is_not_retried(
    error: BaseException,
) -> None:
    sleeps: list[float] = []

    def service_snapshot(_: dict[str, Any]) -> dict[str, Any]:
        raise error

    deps = MODULE.Dependencies(
        service_snapshot=service_snapshot,
        owner_snapshot=lambda: owners(),
        stop_service=lambda: None,
        start_service=lambda: None,
        acquire_lock=lambda: Lease(),
        capture=lambda argv, env: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
        monotonic=lambda: 0.0,
        sleep=sleeps.append,
    )

    with pytest.raises(MODULE.TerminalRestoreError) as captured:
        MODULE.poll_restored(deps, service(True), readiness())

    assert captured.value.details is not None
    assert captured.value.details["attempts"] == 1
    assert captured.value.details["elapsed_seconds"] == 0.0
    assert sleeps == []


def test_restore_epoch_regression_and_foreign_owner_are_terminal() -> None:
    for current, observed, reason in (
        (service(True), owners(200), "main PID epoch regressed"),
        (service(True, epoch=101, worker=201), owners(999), "foreign"),
    ):
        deps = MODULE.Dependencies(
            service_snapshot=lambda _, value=current: value,
            owner_snapshot=lambda value=observed: value,
            stop_service=lambda: None,
            start_service=lambda: None,
            acquire_lock=lambda: Lease(),
            capture=lambda argv, env: subprocess.CompletedProcess(argv, 0, stdout="", stderr=""),
            monotonic=lambda: 0.0,
            sleep=lambda _: (_ for _ in ()).throw(AssertionError("terminal restore slept")),
        )
        with pytest.raises(MODULE.TerminalRestoreError, match=reason):
            MODULE.poll_restored(deps, service(True), readiness())
