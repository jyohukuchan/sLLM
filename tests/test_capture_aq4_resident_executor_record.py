from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
import signal
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/capture-aq4-resident-executor-record.py"
SPEC = importlib.util.spec_from_file_location("capture_aq4_resident_executor_record", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def collect(raw: bytes) -> tuple[TOOL.WorkerStderrCollector, dict]:
    collector = TOOL.WorkerStderrCollector(io.BytesIO(raw))
    collector.drain()
    return collector, collector.summary()


def preview_text(summary: dict) -> str:
    return summary["head_text"] + summary["tail_text"]


def test_stderr_json_dict_compatibility_and_raw_digest() -> None:
    raw = b'{"event":"load","count":1}\n[1,2]\nnot-json\n{"event":"release"}\n'
    collector, summary = collect(raw)

    assert collector.records == [{"event": "load", "count": 1}, {"event": "release"}]
    assert summary["byte_count"] == len(raw)
    assert summary["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["head_bytes"] == len(summary["head_text"].encode("utf-8"))
    assert summary["tail_bytes"] == len(summary["tail_text"].encode("utf-8"))
    assert summary["record_count"] == 2
    assert summary["records_retained"] == 2
    assert summary["utf8_replacement"] is False
    assert summary["complete"] is True
    assert summary["stream_error"] is None


def test_stderr_invalid_utf8_is_drained_and_preview_is_valid_utf8() -> None:
    raw = b'{"ok":true}\n\xff\xfe\n'
    collector, summary = collect(raw)

    assert collector.records == [{"ok": True}]
    assert summary["utf8_replacement"] is True
    preview = preview_text(summary)
    assert "\ufffd" in preview
    assert len(preview.encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_secret_marker_after_32k_boundary_redacts_the_whole_long_line() -> None:
    secret = b"password=do-not-publish"
    raw = b"prefix\n" + b"A" * (TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES + 4096) + b" " + secret + b"\n"
    _, summary = collect(raw)

    assert summary["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["byte_count"] == len(raw)
    assert summary["redacted_lines"] == 1
    preview = preview_text(summary)
    assert secret.decode() not in preview
    assert "<redacted sensitive diagnostic line>" in preview
    assert summary["truncated"] is False
    assert len(preview.encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_many_secret_lines_never_expose_secret_and_preview_has_final_cap() -> None:
    raw = b"".join(f"authorization: secret-{index}\n".encode() for index in range(5000))
    _, summary = collect(raw)

    assert summary["redacted_lines"] == 5000
    preview = preview_text(summary)
    assert "secret-" not in preview
    assert "authorization" not in preview
    assert summary["head_bytes"] <= TOOL.WORKER_STDERR_HEAD_MAX_BYTES
    assert summary["tail_bytes"] <= TOOL.WORKER_STDERR_TAIL_MAX_BYTES
    assert len(preview.encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_giant_non_json_line_is_bounded_without_retaining_the_line() -> None:
    raw = b"Z" * (TOOL.WORKER_STDERR_JSON_LINE_MAX_BYTES + 100) + b"\n"
    collector, summary = collect(raw)

    assert collector.records == []
    assert summary["byte_count"] == len(raw)
    assert summary["truncated"] is True


def test_stderr_head_and_tail_preserve_final_bounded_context() -> None:
    raw = b"".join(
        f"diagnostic-{index:05d}-".encode() + b"x" * 96 + b"\n"
        for index in range(1000)
    )
    _, summary = collect(raw)

    assert "diagnostic-00000" in summary["head_text"]
    assert "diagnostic-00999" in summary["tail_text"]
    assert summary["head_bytes"] <= TOOL.WORKER_STDERR_HEAD_MAX_BYTES
    assert summary["tail_bytes"] <= TOOL.WORKER_STDERR_TAIL_MAX_BYTES
    assert summary["truncated"] is True


def test_stderr_drain_failure_is_incomplete_and_structurally_bounded() -> None:
    class BrokenStream:
        def read(self, _: int) -> bytes:
            raise OSError("simulated drain failure")

    collector = TOOL.WorkerStderrCollector(BrokenStream())
    collector.drain()
    summary = collector.summary()
    assert summary["complete"] is False
    assert summary["stream_error"] == "OSError"


def test_worker_lifecycle_producer_retains_64_events_and_binds_65th_terminal() -> None:
    request_id = "sq8-promotion-" + "a" * 64
    lifecycle = TOOL.WorkerLifecycleEvidence(request_id, started_at=0.0)

    for index in range(65):
        lifecycle.observe(
            {
                "type": "ready" if index == 0 else "progress",
                "request_id": request_id,
                "processed_prompt_tokens": index,
                "completion_tokens": index,
                "index": index,
            },
            observed_at=(index + 1) / 1000.0,
        )

    summary = lifecycle.summary()
    assert summary["event_count"] == 65
    assert summary["events_retained"] == TOOL.WORKER_LIFECYCLE_MAX_EVENTS == 64
    assert summary["events_truncated"] is True
    assert len(summary["events"]) == 64
    assert summary["events"][0]["type"] == "ready"
    assert summary["last_event"]["token_index"] == 64


def test_outer_kill_reap_hang_is_bounded_and_typed_cleanup_failure() -> None:
    class HangingProcess:
        returncode = None

        def __init__(self) -> None:
            self.calls: list[object] = []

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.calls.append("terminate")

        def kill(self) -> None:
            self.calls.append("kill")

        def wait(self, *, timeout: float) -> None:
            self.calls.append(timeout)
            raise subprocess.TimeoutExpired(["fake-worker"], timeout)

    process = HangingProcess()
    with pytest.raises(
        TOOL.CaptureError, match="reaped within the cleanup bound"
    ) as raised:
        TOOL._terminate_worker(process)  # type: ignore[arg-type]

    assert raised.value.stage == "cleanup"
    assert process.calls == [
        "terminate",
        TOOL.WORKER_TERMINATE_GRACE_SECONDS,
        "kill",
        TOOL.WORKER_KILL_REAP_TIMEOUT_SECONDS,
        "kill",
        TOOL.WORKER_FINAL_REAP_TIMEOUT_SECONDS,
    ]


def test_pipe_drain_hang_is_bounded_and_emits_typed_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    worker_source = r'''
import json, sys
request = json.loads(sys.stdin.buffer.readline())
print(json.dumps({"schema_version": "ullm.worker.v1", "type": "released", "request_id": request["request_id"], "completion_tokens": 0}), flush=True)
sys.stdin.buffer.readline()
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    empty = TOOL._empty_worker_stderr()
    incomplete = dict(empty, complete=False, stream_error="drain_thread_timeout")

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)
    monkeypatch.setattr(TOOL, "_finish_worker_stderr", lambda *_args: incomplete)
    monkeypatch.setattr(TOOL, "_finish_worker_stdout", lambda *_args: False)

    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "0.5",
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["stage"] == "cleanup"
    assert envelope["reason"] == "worker pipe drain did not complete"
    assert envelope["timeouts"]["shutdown_seconds"] == (
        TOOL.WORKER_SHUTDOWN_TIMEOUT_SECONDS
    )
    assert envelope["worker_stderr"]["complete"] is False
    assert envelope["worker_stderr"]["stream_error"] == "drain_thread_timeout"


@pytest.mark.parametrize("bad", [True, 1.0, TOOL.SAFE_INT + 1])
def test_implementation_audit_counts_reject_bool_float_and_safe_int_overflow(
    bad: object,
) -> None:
    load_records = [
        {
            "schema_version": "ullm.backend_operation.load.v1",
            "layer_position": 0,
            "trace": {
                "resolution": "Primary",
                "implementation_id": "impl",
                "kind": "linear",
            },
        }
    ]
    audit = {
        "implementation_counts": [
            {"implementation_id": "impl", "kind": "linear", "count": bad}
        ]
    }
    with pytest.raises(TOOL.CaptureError, match="implementation.*count"):
        TOOL.operator_records(load_records, audit)


@pytest.mark.parametrize("field", ["prefill_width_histogram", "total_steps"])
@pytest.mark.parametrize("bad", [True, 1.0, TOOL.SAFE_INT + 1])
def test_audit_histogram_and_total_step_counts_reject_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
    bad: object,
) -> None:
    audit = {
        "coverage_complete": True,
        "implementation_counts": [{"implementation_id": "impl", "count": 1}],
        "prefill_width_histogram": [0, 1],
        "total_steps": 1,
    }
    if field == "prefill_width_histogram":
        audit[field] = [0, bad]
    else:
        audit[field] = bad
    worker_source = f'''\
import json, sys
request = json.loads(sys.stdin.buffer.readline())
request_id = request["request_id"]
print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "token", "request_id": request_id, "index": 0, "token_id": 42}}, separators=(",", ":")), flush=True)
print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "released", "request_id": request_id, "completion_tokens": 1, "timings": {{"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}}}}, separators=(",", ":")), flush=True)
json.dump({{"schema_version": "ullm.backend_operation.load.v1", "layer_position": 0, "trace": {{"resolution": "Primary", "implementation_id": "impl", "kind": "linear", "semantic_version": "1", "backend": "hip", "architecture": "gfx1201", "device_name": "fake", "persistent_bytes": 1, "temporary_bytes": 1, "batch_width": 1, "chunk_width": 1}}}}, sys.stderr); sys.stderr.write("\\n")
json.dump({{"event": "request_released", "request_id": request_id, "operation_execution_audit": {audit!r}, "request_execution_audit": {{}}}}, sys.stderr); sys.stderr.write("\\n"); sys.stderr.flush()
json.loads(sys.stdin.buffer.readline())
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "passthrough_tensors": [],
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_proj.weight",
                        "index_file": "weights.idx",
                        "scale_file": "weights.scale",
                        "codebook_file": "codebook.bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("weights.idx", "weights.scale", "codebook.bin"):
        (tmp_path / name).write_bytes(b"x")

    class CompleteObserver:
        def __init__(self, _: str) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> dict[str, object]:
            return {
                "kind": "fake",
                "sample_count": 2,
                "complete": True,
                "capacity_bytes": 1_000_000,
                "peak_bytes": 100,
                "target_card": "fake",
            }

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "VramObserver", CompleteObserver)
    output = tmp_path / "record.json"
    assert TOOL.main(["--manifest", str(manifest), "--output", str(output)]) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["stage"] == "audit_missing"
    assert "audit" in envelope["reason"]


@pytest.mark.parametrize("bad", [True, 1.0, TOOL.SAFE_INT + 1])
def test_normalized_worker_stderr_rejects_invalid_redacted_line_count(
    bad: object,
) -> None:
    value = TOOL._normalize_worker_stderr(
        dict(
            TOOL._empty_worker_stderr(),
            complete=True,
            stream_error=None,
            redacted_lines=bad,
        )
    )
    assert value["redacted_lines"] == 0


def test_main_emits_fixed_error_envelope_with_worker_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    raw = b"bad\n"
    summary = {
        "schema_version": TOOL.WORKER_STDERR_SCHEMA_VERSION,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "head_text": raw.decode(),
        "head_bytes": len(raw),
        "tail_text": raw.decode(),
        "tail_bytes": len(raw),
        "truncated": False,
        "utf8_replacement": False,
        "redacted_lines": 0,
        "record_count": 0,
        "records_retained": 0,
        "records_truncated": False,
        "schema_counts": {},
        "last_complete_record": None,
        "complete": True,
        "stream_error": None,
    }

    def fail(_: object) -> dict:
        raise TOOL.CaptureError(
            "resident worker did not release",
            worker_stderr=summary,
            stage="request",
            worker_returncode=-signal.SIGKILL,
            worker_signal=signal.SIGKILL,
        )

    monkeypatch.setattr(TOOL, "run_capture", fail)
    assert TOOL.main(["--manifest", "manifest.json", "--output", "record.json"]) == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope == {
        "schema_version": "ullm.aq4_resident_capture_error.v5",
        "status": "failed",
        "stage": "request",
        "reason": "resident worker did not release",
        "timed_out": False,
        "request_id": None,
        "timeouts": {
            "ready_seconds": TOOL.DEFAULT_READY_TIMEOUT_SECONDS,
            "request_seconds": TOOL.DEFAULT_REQUEST_TIMEOUT_SECONDS,
            "shutdown_seconds": TOOL.WORKER_SHUTDOWN_TIMEOUT_SECONDS,
        },
        "worker_returncode": -signal.SIGKILL,
        "worker_signal": signal.SIGKILL,
        "worker_stderr": summary,
        "worker_lifecycle": {
            "schema_version": TOOL.WORKER_LIFECYCLE_SCHEMA_VERSION,
            "request_id": None,
            "request_sent": False,
            "request_sent_offset_ms": None,
            "event_count": 0,
            "events_retained": 0,
            "events_truncated": False,
            "events": [],
            "last_event": None,
        },
        "worker_error": None,
        "observed_sq8_promotion_telemetry": None,
        "observed_sq8_promotion_telemetry_binding": None,
        "worker_terminal": None,
    }
    assert "resident worker did not release" in captured.err


def _fake_manifest(
    tmp_path: Path,
    worker_source: str,
    *,
    emit_ready: bool = True,
    ready_delay: float = 0.0,
) -> Path:
    worker = tmp_path / "fake-worker.py"
    ready_source = f'''\
import time as _ready_time
_ready_time.sleep({ready_delay!r})
''' + r'''
import json as _ready_json
from pathlib import Path as _ReadyPath
_ready_manifest = _ready_json.loads(
    _ReadyPath(__file__).with_name("manifest.json").read_text(encoding="utf-8")
)
_ready_product = _ready_manifest["product"]
_ready_identity = _ready_manifest["worker"]["identity"]
_ready_artifact = _ready_product.get("artifact")
print(_ready_json.dumps({
    "schema_version": _ready_manifest["worker"]["protocol"],
    "type": "ready",
    "model": _ready_manifest["public"]["id"],
    "model_revision": _ready_manifest["public"]["revision"],
    "artifact_content_sha256": (
        _ready_artifact.get("content_sha256") if isinstance(_ready_artifact, dict) else None
    ),
    "package_manifest_sha256": _ready_product["package"]["manifest_sha256"],
    "device": _ready_identity["device"],
    "execution_profile": _ready_identity["execution_profile"],
    "context_length": _ready_manifest["public"]["context_length"],
    "max_new_tokens": _ready_manifest["generation"]["max_completion_tokens"],
}, separators=(",", ":")), flush=True)
'''
    worker.write_text(
        "#!/usr/bin/env python3\n"
        + (ready_source if emit_ready else "")
        + worker_source,
        encoding="utf-8",
    )
    worker.chmod(0o755)
    (tmp_path / "package.json").write_text(
        json.dumps({"passthrough_tensors": [], "tensors": []}), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "product": {
                    "root": ".",
                    "package": {
                        "manifest_path": "package.json",
                        "manifest_sha256": "c" * 64,
                    },
                },
                "worker": {
                    "binary": worker.name,
                    "protocol": "ullm.worker.v1",
                    "identity": {
                        "device": "gfx1201",
                        "device_index": 0,
                        "execution_profile": "test-profile",
                    },
                    "arguments": [],
                },
                "public": {
                    "id": "test-model",
                    "revision": "test-revision",
                    "context_length": 4096,
                },
                "generation": {"eos_token_ids": [], "max_completion_tokens": 2},
                "format": {},
            }
        ),
        encoding="utf-8",
    )
    return manifest


@pytest.mark.parametrize(
    ("worker_source", "expected_signal", "expected_timeout"),
    [
        (
            "import os, signal, sys\nsys.stdin.buffer.readline()\nsys.stderr.buffer.write(b'not-json\\xff\\npassword=secret\\n'+b'x'*40000)\nsys.stderr.flush()\nos.kill(os.getpid(), signal.SIGTERM)\n",
            signal.SIGTERM,
            False,
        ),
        (
            "import sys, time\nsys.stdin.buffer.readline()\nsys.stderr.buffer.write(b'not-json\\xff\\npassword=secret\\n'+b'x'*40000)\nsys.stderr.flush()\ntime.sleep(10)\n",
            None,
            True,
        ),
    ],
)
def test_real_fake_worker_failure_envelope_preserves_terminal_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    worker_source: str,
    expected_signal: int | None,
    expected_timeout: bool,
) -> None:
    manifest = _fake_manifest(tmp_path, worker_source)
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "0.1",
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert set(envelope) == {
        "schema_version",
        "status",
        "stage",
        "reason",
        "timed_out",
        "request_id",
        "timeouts",
        "worker_returncode",
        "worker_signal",
        "worker_stderr",
        "worker_lifecycle",
        "worker_error",
        "observed_sq8_promotion_telemetry",
        "observed_sq8_promotion_telemetry_binding",
        "worker_terminal",
    }
    assert envelope["status"] == "failed"
    assert envelope["timed_out"] is expected_timeout
    assert envelope["stage"] == (
        "request_timeout" if expected_timeout else "request_protocol"
    )
    assert envelope["worker_lifecycle"]["request_sent"] is True
    if expected_signal is not None:
        assert envelope["worker_returncode"] == -expected_signal
        assert envelope["worker_signal"] == expected_signal
    else:
        assert envelope["worker_returncode"] is not None
    assert envelope["worker_stderr"]["complete"] is True
    assert envelope["worker_stderr"]["stream_error"] is None
    stderr = envelope["worker_stderr"]
    assert stderr["byte_count"] > 32 * 1024
    assert stderr["utf8_replacement"] is True
    assert stderr["redacted_lines"] == 1
    assert stderr["truncated"] is True
    assert "password=secret" not in preview_text(stderr)


def test_ready_timeout_occurs_before_request_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = _fake_manifest(
        tmp_path,
        "import time\ntime.sleep(10)\n",
        emit_ready=False,
    )
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)

    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--ready-timeout",
            "0.05",
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["stage"] == "ready_timeout"
    assert envelope["timed_out"] is True
    assert envelope["worker_lifecycle"]["request_sent"] is False
    assert envelope["worker_lifecycle"]["event_count"] == 0


def test_partial_token_timeout_records_only_bounded_lifecycle_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_token_id = 987654321
    worker_source = f'''\
import json, sys, time
request = json.loads(sys.stdin.buffer.readline())
print(json.dumps({{
    "schema_version": "ullm.worker.v1",
    "type": "token",
    "request_id": request["request_id"],
    "index": 0,
    "token_id": {secret_token_id},
}}, separators=(",", ":")), flush=True)
time.sleep(10)
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)

    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "0.05",
        ]
    ) == 1
    captured = capsys.readouterr().out
    envelope = json.loads(captured)
    assert envelope["stage"] == "request_timeout"
    assert envelope["worker_lifecycle"]["last_event"]["type"] == "token"
    assert envelope["worker_lifecycle"]["last_event"]["token_index"] == 0
    assert "token_id" not in envelope["worker_lifecycle"]["last_event"]
    assert str(secret_token_id) not in captured


@pytest.mark.parametrize(
    ("recoverable", "code", "expected_stage"),
    [
        (True, "invalid_request", "worker_error"),
        (False, "runtime_failed", "worker_fatal"),
    ],
)
def test_immediate_typed_worker_error_is_private_bounded_and_reaped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    recoverable: bool,
    code: str,
    expected_stage: str,
) -> None:
    pid_path = tmp_path / "worker.pid"
    secret = "password=do-not-publish prompt_token=991 /tmp/private-model"
    worker_source = f'''\
import json, os, pathlib, sys
pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding="ascii")
request = json.loads(sys.stdin.buffer.readline())
event = {{
    "schema_version": "ullm.worker.v1",
    "type": "error",
    "request_id": request["request_id"],
    "code": {code!r},
    "recoverable": {recoverable!r},
    "message": {secret!r},
}}
print(json.dumps(event, separators=(",", ":")), flush=True)
shutdown = json.loads(sys.stdin.buffer.readline())
assert shutdown == {{"schema_version": "ullm.worker.v1", "type": "shutdown"}}
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)

    started = time.monotonic()
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "30",
        ]
    ) == 1
    elapsed = time.monotonic() - started
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert elapsed < 2.0
    assert envelope["stage"] == expected_stage
    assert envelope["timed_out"] is False
    assert envelope["worker_returncode"] == 0
    assert envelope["worker_signal"] is None
    assert envelope["worker_lifecycle"]["last_event"]["type"] == "error"
    assert envelope["worker_lifecycle"]["last_event"]["request_id_matches"] is True
    summary = envelope["worker_error"]
    assert summary["stage"] == expected_stage
    assert summary["code"] == code
    assert summary["recoverable"] is recoverable
    assert summary["request_id"] == envelope["request_id"]
    assert summary["message"] == {
        "byte_count": len(secret.encode("utf-8")),
        "sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
        "prefix_text": None,
        "prefix_bytes": 0,
        "prefix_truncated": True,
        "redaction": "omitted_by_capture_privacy_policy",
    }
    canonical = json.dumps(
        {
            "schema_version": "ullm.worker.v1",
            "type": "error",
            "request_id": envelope["request_id"],
            "code": code,
            "recoverable": recoverable,
            "message": secret,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert summary["canonical_event_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert summary["shutdown"] == {
        "attempted": True,
        "completed": True,
        "error": None,
    }
    assert secret not in captured.out
    assert secret not in captured.err
    pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_typed_worker_error_escalates_shutdown_to_kill_and_reaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pid_path = tmp_path / "stubborn-worker.pid"
    worker_source = f'''\
import json, os, pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()), encoding="ascii")
request = json.loads(sys.stdin.buffer.readline())
print(json.dumps({{
    "schema_version": "ullm.worker.v1",
    "type": "error",
    "request_id": request["request_id"],
    "code": "runtime_failed",
    "recoverable": False,
    "message": "bounded fatal",
}}, separators=(",", ":")), flush=True)
while True:
    time.sleep(1)
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)
    monkeypatch.setattr(TOOL, "WORKER_SHUTDOWN_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(TOOL, "WORKER_TERMINATE_GRACE_SECONDS", 0.1)

    started = time.monotonic()
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "30",
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert time.monotonic() - started < 2.0
    assert envelope["stage"] == "worker_fatal"
    assert envelope["timed_out"] is False
    assert envelope["worker_returncode"] == -signal.SIGKILL
    assert envelope["worker_signal"] == signal.SIGKILL
    assert envelope["worker_error"]["shutdown"] == {
        "attempted": True,
        "completed": False,
        "error": "shutdown_timeout",
    }
    pid = int(pid_path.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize(
    "event_source",
    [
        # Duplicate keys are rejected before an event can be accepted.
        '''print('{"schema_version":"ullm.worker.v1","type":"error","request_id":'+json.dumps(request["request_id"])+',"code":"invalid_request","code":"busy","recoverable":true,"message":"x"}', flush=True)''',
        '''print(json.dumps({"schema_version":"ullm.worker.v1","type":"error","request_id":request["request_id"],"code":"invented_code","recoverable":True,"message":"x"}, separators=(",", ":")), flush=True)''',
        '''print(json.dumps({"schema_version":"ullm.worker.v1","type":"error","request_id":request["request_id"],"code":"invalid_request","recoverable":1,"message":"x"}, separators=(",", ":")), flush=True)''',
        '''print(json.dumps({"schema_version":"ullm.worker.v1","type":"error","request_id":"other-request","code":"invalid_request","recoverable":True,"message":"x"}, separators=(",", ":")), flush=True)''',
        '''print(json.dumps({"schema_version":"ullm.worker.v1","type":"fatal","request_id":request["request_id"],"code":"runtime_failed","recoverable":False,"message":"x"}, separators=(",", ":")), flush=True)''',
    ],
)
def test_malformed_or_unbound_worker_error_is_protocol_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    event_source: str,
) -> None:
    worker_source = f'''\
import json, sys, time
request = json.loads(sys.stdin.buffer.readline())
{event_source}
time.sleep(10)
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "target_card", lambda _: None)

    started = time.monotonic()
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--timeout",
            "30",
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert time.monotonic() - started < 2.0
    assert envelope["stage"] == "request_protocol"
    assert envelope["timed_out"] is False
    assert envelope["worker_error"] is None
    assert envelope["worker_returncode"] < 0
    assert envelope["worker_signal"] == signal.SIGTERM


def test_real_fake_worker_json_success_path_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worker_source = r'''
import json, sys
request = json.loads(sys.stdin.buffer.readline())
request_id = request["request_id"]
print(json.dumps({"schema_version": "ullm.worker.v1", "type": "token", "request_id": request_id, "index": 0, "token_id": 42}, separators=(",", ":")), flush=True)
print(json.dumps({"schema_version": "ullm.worker.v1", "type": "released", "request_id": request_id, "completion_tokens": 1, "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}, "reset_complete": True}, separators=(",", ":")), flush=True)
json.dump({"schema_version": "ullm.backend_operation.load.v1", "layer_position": 0, "trace": {"resolution": "Primary", "implementation_id": "impl", "kind": "linear", "semantic_version": "1", "backend": "hip", "architecture": "gfx1201", "device_name": "fake", "persistent_bytes": 1, "temporary_bytes": 1, "batch_width": 1, "chunk_width": 1}}, sys.stderr); sys.stderr.write("\n")
json.dump({"event": "request_released", "request_id": request_id, "operation_execution_audit": {"coverage_complete": True, "implementation_counts": [{"implementation_id": "impl", "count": 1}], "prefill_width_histogram": [0, 1], "total_steps": 1}, "request_execution_audit": {}}, sys.stderr); sys.stderr.write("\n"); sys.stderr.flush()
json.loads(sys.stdin.buffer.readline())
'''
    manifest = _fake_manifest(tmp_path, worker_source, ready_delay=0.05)
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "passthrough_tensors": [],
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_proj.weight",
                        "index_file": "weights.idx",
                        "scale_file": "weights.scale",
                        "codebook_file": "codebook.bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("weights.idx", "weights.scale", "codebook.bin"):
        (tmp_path / name).write_bytes(b"x")

    class CompleteObserver:
        def __init__(self, _: str) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> dict[str, object]:
            return {
                "kind": "rocm_smi_vram_target_card",
                "sample_count": 2,
                "complete": True,
                "capacity_bytes": 1_000_000,
                "peak_bytes": 100,
                "target_card": "card0",
            }

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "VramObserver", CompleteObserver)
    output = tmp_path / "record.json"
    started = time.monotonic()
    assert TOOL.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    assert time.monotonic() - started >= 0.04
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "ok"
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["request_summary"]["generated_token_count"] == 1


@pytest.mark.parametrize(
    (
        "projection_overrides",
        "worker_exit_code",
        "omit_operation_audit",
        "expected_stage",
    ),
    [
        ({"batch_matvec_count": 0}, 0, False, "telemetry_validation"),
        ({"pair_matvec_count": 0}, 0, False, "telemetry_validation"),
        ({}, 0, False, "audit_missing"),
        ({}, 0, True, "audit_missing"),
        ({}, 7, False, "worker_exit"),
    ],
)
def test_fake_worker_sq8_failure_preserves_terminal_and_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    projection_overrides: dict[str, int],
    worker_exit_code: int,
    omit_operation_audit: bool,
    expected_stage: str,
) -> None:
    projection = {
        "single_matvec_count": 0,
        "batch_matvec_count": 1,
        "pair_matvec_count": 1,
        "triple_matvec_count": 0,
        "fallback_count": 0,
        **projection_overrides,
    }
    telemetry = {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": projection,
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }
    operation_audit = None if omit_operation_audit else {
        "coverage_complete": True,
        "implementation_counts": [],
        "prefill_width_histogram": [0, 1],
        "total_steps": 2,
    }
    worker_source = f'''\
import json, sys
request = json.loads(sys.stdin.buffer.readline())
assert request["max_new_tokens"] == 2
assert request["eos_token_ids"] == [248044, 248046]
request_id = request["request_id"]
for index, token_id in enumerate((42, 43)):
    print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "token", "request_id": request_id, "index": index, "token_id": token_id}}, separators=(",", ":")), flush=True)
print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "released", "request_id": request_id, "completion_tokens": 2, "timings": {{"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}}, "reset_complete": True}}, separators=(",", ":")), flush=True)
json.dump({{"event": "request_released", "request_id": request_id, "operation_execution_audit": {operation_audit!r}, "request_execution_audit": {{"sq8_promotion_telemetry": {telemetry!r}}}}}, sys.stderr); sys.stderr.write("\\n"); sys.stderr.flush()
json.loads(sys.stdin.buffer.readline())
sys.exit({worker_exit_code})
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["format"] = {
        "implementation_id": TOOL.SQ8_OVERLAY_IMPLEMENTATION_ID
    }
    manifest_value["worker"]["identity"]["execution_profile"] = (
        TOOL.SQ8_OVERLAY_EXECUTION_PROFILE
    )
    manifest_value["generation"]["eos_token_ids"] = [248044, 248046]
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

    class CompleteObserver:
        def __init__(self, _: str) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> dict[str, object]:
            return {
                "kind": "fake",
                "sample_count": 1,
                "complete": True,
                "capacity_bytes": 1_000_000,
                "peak_bytes": 100,
                "target_card": "fake",
            }

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "VramObserver", CompleteObserver)
    request_id = "sq8-promotion-" + "a" * 64
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--prompt-tokens",
            "128",
            "--max-new-tokens",
            "2",
            "--sq8-promotion-evidence",
            "--sq8-promotion-request-id",
            request_id,
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["stage"] == expected_stage
    assert envelope["worker_returncode"] == worker_exit_code
    assert envelope["worker_signal"] is None
    assert envelope["worker_stderr"]["stream_error"] is None
    assert envelope["observed_sq8_promotion_telemetry"] == telemetry
    assert envelope["observed_sq8_promotion_telemetry_binding"] == (
        TOOL.sq8_promotion_telemetry_binding(telemetry, request_id)
    )
    assert envelope["worker_terminal"] == {
        "schema_version": "ullm.aq4_resident_worker_terminal.v1",
        "event": "request_released",
        "request_id": request_id,
        "request_id_matches": True,
        "operation_execution_audit_observed": not omit_operation_audit,
        "request_execution_audit_observed": True,
    }


@pytest.mark.parametrize("expected_stage", ["resource_observation", "package_validation"])
def test_post_telemetry_failure_preserves_raw_telemetry_and_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    expected_stage: str,
) -> None:
    telemetry = {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": {
            "single_matvec_count": 0,
            "batch_matvec_count": 1,
            "pair_matvec_count": 1,
            "triple_matvec_count": 0,
            "fallback_count": 0,
        },
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }
    worker_source = f'''\
import json, sys
request = json.loads(sys.stdin.buffer.readline())
request_id = request["request_id"]
for index, token_id in enumerate((42, 43)):
    print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "token", "request_id": request_id, "index": index, "token_id": token_id}}, separators=(",", ":")), flush=True)
print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "released", "request_id": request_id, "completion_tokens": 2, "timings": {{"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}}, "reset_complete": True}}, separators=(",", ":")), flush=True)
json.dump({{"schema_version": "ullm.backend_operation.load.v1", "layer_position": 0, "trace": {{"resolution": "Primary", "implementation_id": "impl", "kind": "linear", "semantic_version": "1", "backend": "hip", "architecture": "gfx1201", "device_name": "fake", "persistent_bytes": 1, "temporary_bytes": 1, "batch_width": 128, "chunk_width": 128}}}}, sys.stderr); sys.stderr.write("\\n")
json.dump({{"event": "request_released", "request_id": request_id, "operation_execution_audit": {{"coverage_complete": True, "implementation_counts": [{{"implementation_id": "impl", "count": 1}}], "prefill_width_histogram": [0] * 128 + [1], "total_steps": 129}}, "request_execution_audit": {{"sq8_promotion_telemetry": {telemetry!r}}}}}, sys.stderr); sys.stderr.write("\\n"); sys.stderr.flush()
json.loads(sys.stdin.buffer.readline())
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["format"] = {
        "implementation_id": TOOL.SQ8_OVERLAY_IMPLEMENTATION_ID
    }
    manifest_value["worker"]["identity"]["execution_profile"] = (
        TOOL.SQ8_OVERLAY_EXECUTION_PROFILE
    )
    manifest_value["generation"]["eos_token_ids"] = [248044, 248046]
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    if expected_stage == "package_validation":
        (tmp_path / "package.json").write_text(
            json.dumps(
                {
                    "passthrough_tensors": [],
                    "tensors": [
                        {
                            "name": "model.layers.0.self_attn.q_proj.weight",
                            "index_file": "missing.idx",
                            "scale_file": "missing.scale",
                            "codebook_file": "missing.codebook",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    class Observer:
        def __init__(self, _: str) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> dict[str, object]:
            return {
                "kind": "fake",
                "sample_count": 1,
                "complete": expected_stage != "resource_observation",
                "capacity_bytes": 1_000_000_000,
                "peak_bytes": 100,
                "target_card": "fake",
            }

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "VramObserver", Observer)
    request_id = "sq8-promotion-" + "a" * 64
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "record.json"),
            "--prompt-tokens",
            "128",
            "--max-new-tokens",
            "2",
            "--sq8-promotion-evidence",
            "--sq8-promotion-request-id",
            request_id,
        ]
    ) == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["stage"] == expected_stage
    assert envelope["observed_sq8_promotion_telemetry"] == telemetry
    assert envelope["observed_sq8_promotion_telemetry_binding"] == (
        TOOL.sq8_promotion_telemetry_binding(telemetry, request_id)
    )
    assert envelope["worker_terminal"]["request_id"] == request_id
    assert envelope["worker_terminal"]["request_id_matches"] is True


def test_fake_worker_sq8_positive_requires_prefill_and_decode_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    telemetry = {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": {
            "single_matvec_count": 0,
            "batch_matvec_count": 24,
            "pair_matvec_count": 24,
            "triple_matvec_count": 0,
            "fallback_count": 0,
        },
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }
    worker_source = f'''\
import json, sys
request = json.loads(sys.stdin.buffer.readline())
assert request["max_new_tokens"] == 2
assert request["eos_token_ids"] == [248044, 248046]
request_id = request["request_id"]
for index, token_id in enumerate((42, 43)):
    print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "token", "request_id": request_id, "index": index, "token_id": token_id}}, separators=(",", ":")), flush=True)
print(json.dumps({{"schema_version": "ullm.worker.v1", "type": "released", "request_id": request_id, "completion_tokens": 2, "timings": {{"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}}, "reset_complete": True}}, separators=(",", ":")), flush=True)
json.dump({{"schema_version": "ullm.backend_operation.load.v1", "layer_position": 0, "trace": {{"resolution": "Primary", "implementation_id": "impl", "kind": "linear", "semantic_version": "1", "backend": "hip", "architecture": "gfx1201", "device_name": "fake", "persistent_bytes": 1, "temporary_bytes": 1, "batch_width": 128, "chunk_width": 128}}}}, sys.stderr); sys.stderr.write("\\n")
json.dump({{"event": "request_released", "request_id": request_id, "operation_execution_audit": {{"coverage_complete": True, "implementation_counts": [{{"implementation_id": "impl", "count": 1}}], "prefill_width_histogram": [0] * 128 + [1], "total_steps": 129}}, "request_execution_audit": {{"sq8_promotion_telemetry": {telemetry!r}}}}}, sys.stderr); sys.stderr.write("\\n"); sys.stderr.flush()
json.loads(sys.stdin.buffer.readline())
'''
    manifest = _fake_manifest(tmp_path, worker_source)
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "passthrough_tensors": [],
                "tensors": [
                    {
                        "name": "model.layers.0.self_attn.q_proj.weight",
                        "index_file": "weights.idx",
                        "scale_file": "weights.scale",
                        "codebook_file": "codebook.bin",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    for name in ("weights.idx", "weights.scale", "codebook.bin"):
        (tmp_path / name).write_bytes(b"x")
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_value["format"] = {
        "implementation_id": TOOL.SQ8_OVERLAY_IMPLEMENTATION_ID
    }
    manifest_value["worker"]["identity"]["execution_profile"] = (
        TOOL.SQ8_OVERLAY_EXECUTION_PROFILE
    )
    manifest_value["generation"]["eos_token_ids"] = [248044, 248046]
    manifest_value["product"]["artifact"] = {
        "content_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    manifest_value["product"]["package"]["manifest_sha256"] = "c" * 64
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")

    class CompleteObserver:
        def __init__(self, _: str) -> None:
            pass

        def start(self) -> None:
            pass

        def finish(self) -> dict[str, object]:
            return {
                "kind": "fake",
                "sample_count": 1,
                "complete": True,
                "capacity_bytes": 1_000_000_000,
                "peak_bytes": 100,
                "target_card": "fake",
            }

    monkeypatch.setattr(TOOL, "copy_worker_environment", lambda _: dict(os.environ))
    monkeypatch.setattr(TOOL, "VramObserver", CompleteObserver)
    request_id = "sq8-promotion-" + "a" * 64
    output = tmp_path / "sq8-record.json"
    assert TOOL.main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--prompt-tokens",
            "128",
            "--max-new-tokens",
            "2",
            "--sq8-promotion-evidence",
            "--sq8-promotion-request-id",
            request_id,
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    evidence = json.loads(output.read_text(encoding="utf-8"))[
        "sq8_promotion_evidence"
    ]
    assert evidence["telemetry"] == telemetry
    assert evidence["telemetry_binding"] == TOOL.sq8_promotion_telemetry_binding(
        telemetry, request_id
    )
    assert evidence["output_identity"]["token_count"] == 2
