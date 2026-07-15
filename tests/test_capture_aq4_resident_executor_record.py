from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

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


def test_main_emits_fixed_error_envelope_with_worker_stderr(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    summary = {
        "schema_version": TOOL.WORKER_STDERR_SCHEMA_VERSION,
        "byte_count": 3,
        "sha256": hashlib.sha256(b"bad").hexdigest(),
        "preview_text": "bad\n",
        "captured_bytes": 4,
        "truncated": False,
        "utf8_replacement": False,
        "redacted_lines": 0,
    }

    def fail(_: object) -> dict:
        raise TOOL.CaptureError("resident worker did not release", worker_stderr=summary, stage="request")

    monkeypatch.setattr(TOOL, "run_capture", fail)
    assert TOOL.main(["--manifest", "manifest.json", "--output", "record.json"]) == 1
    captured = capsys.readouterr()
    envelope = json.loads(captured.out)
    assert envelope == {
        "schema_version": "ullm.aq4_resident_capture_error.v1",
        "status": "error",
        "stage": "request",
        "reason": "resident worker did not release",
        "worker_stderr": summary,
    }
    assert "resident worker did not release" in captured.err

