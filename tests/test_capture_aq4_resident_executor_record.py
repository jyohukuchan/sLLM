from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal

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


def test_stderr_json_dict_compatibility_and_raw_digest() -> None:
    raw = b'{"event":"load","count":1}\n[1,2]\nnot-json\n{"event":"release"}\n'
    collector, summary = collect(raw)

    assert collector.records == [{"event": "load", "count": 1}, {"event": "release"}]
    assert summary["byte_count"] == len(raw)
    assert summary["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["captured_bytes"] == len(summary["preview_text"].encode("utf-8"))
    assert summary["utf8_replacement"] is False
    assert summary["complete"] is True
    assert summary["stream_error"] is None


def test_stderr_invalid_utf8_is_drained_and_preview_is_valid_utf8() -> None:
    raw = b'{"ok":true}\n\xff\xfe\n'
    collector, summary = collect(raw)

    assert collector.records == [{"ok": True}]
    assert summary["utf8_replacement"] is True
    preview = summary["preview_text"]
    assert "\ufffd" in preview
    assert len(preview.encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_secret_marker_after_32k_boundary_redacts_the_whole_long_line() -> None:
    secret = b"password=do-not-publish"
    raw = b"prefix\n" + b"A" * (TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES + 4096) + b" " + secret + b"\n"
    _, summary = collect(raw)

    assert summary["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["byte_count"] == len(raw)
    assert summary["redacted_lines"] == 1
    assert secret.decode() not in summary["preview_text"]
    assert "<redacted sensitive diagnostic line>" in summary["preview_text"]
    assert summary["truncated"] is True
    assert len(summary["preview_text"].encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_many_secret_lines_never_expose_secret_and_preview_has_final_cap() -> None:
    raw = b"".join(f"authorization: secret-{index}\n".encode() for index in range(5000))
    _, summary = collect(raw)

    assert summary["redacted_lines"] == 5000
    assert "secret-" not in summary["preview_text"]
    assert "authorization" not in summary["preview_text"]
    assert summary["captured_bytes"] <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES
    assert len(summary["preview_text"].encode("utf-8")) <= TOOL.WORKER_STDERR_PREVIEW_MAX_BYTES


def test_giant_non_json_line_is_bounded_without_retaining_the_line() -> None:
    raw = b"Z" * (TOOL.WORKER_STDERR_JSON_LINE_MAX_BYTES + 100) + b"\n"
    collector, summary = collect(raw)

    assert collector.records == []
    assert summary["byte_count"] == len(raw)
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


def test_main_emits_fixed_error_envelope_with_worker_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    raw = b"bad\n"
    summary = {
        "schema_version": TOOL.WORKER_STDERR_SCHEMA_VERSION,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "preview_text": raw.decode(),
        "captured_bytes": len(raw),
        "truncated": False,
        "utf8_replacement": False,
        "redacted_lines": 0,
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
        "schema_version": "ullm.aq4_resident_capture_error.v1",
        "status": "failed",
        "stage": "request",
        "reason": "resident worker did not release",
        "timed_out": False,
        "worker_returncode": -signal.SIGKILL,
        "worker_signal": signal.SIGKILL,
        "worker_stderr": summary,
    }
    assert "resident worker did not release" in captured.err


def _fake_manifest(tmp_path: Path, worker_source: str) -> Path:
    worker = tmp_path / "fake-worker.py"
    worker.write_text("#!/usr/bin/env python3\n" + worker_source, encoding="utf-8")
    worker.chmod(0o755)
    (tmp_path / "package.json").write_text(
        json.dumps({"passthrough_tensors": [], "tensors": []}), encoding="utf-8"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "product": {"root": ".", "package": {"manifest_path": "package.json"}},
                "worker": {
                    "binary": worker.name,
                    "protocol": "ullm.worker.v1",
                    "identity": {"device": "gfx1201", "device_index": 0},
                    "arguments": [],
                },
                "public": {"context_length": 4096},
                "generation": {"eos_token_ids": []},
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
        "worker_returncode",
        "worker_signal",
        "worker_stderr",
    }
    assert envelope["status"] == "failed"
    assert envelope["timed_out"] is expected_timeout
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
    assert "password=secret" not in stderr["preview_text"]


def test_real_fake_worker_json_success_path_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worker_source = r'''
import json, sys
request = json.loads(sys.stdin.buffer.readline())
request_id = request["request_id"]
print(json.dumps({"type": "token", "request_id": request_id, "index": 0, "token_id": 42}, separators=(",", ":")), flush=True)
print(json.dumps({"type": "released", "request_id": request_id, "completion_tokens": 1, "timings": {"prompt_ms": 1.0, "predicted_ms": 2.0, "cache_n": 0}, "reset_complete": True}, separators=(",", ":")), flush=True)
json.dump({"schema_version": "ullm.backend_operation.load.v1", "layer_position": 0, "trace": {"resolution": "Primary", "implementation_id": "impl", "kind": "linear", "semantic_version": "1", "backend": "hip", "architecture": "gfx1201", "device_name": "fake", "persistent_bytes": 1, "temporary_bytes": 1, "batch_width": 1, "chunk_width": 1}}, sys.stderr); sys.stderr.write("\n")
json.dump({"event": "request_released", "request_id": request_id, "operation_execution_audit": {"coverage_complete": True, "implementation_counts": [{"implementation_id": "impl", "count": 1}], "prefill_width_histogram": [0, 1], "total_steps": 1}, "request_execution_audit": {}}, sys.stderr); sys.stderr.write("\n"); sys.stderr.flush()
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
    assert TOOL.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "ok"
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["request_summary"]["generated_token_count"] == 1
