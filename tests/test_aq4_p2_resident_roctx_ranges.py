from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_resident_batch_roctx",
    ROOT / "tools/run-aq4-p2-resident-batch.py",
)
assert SPEC and SPEC.loader
BATCH = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BATCH
try:
    SPEC.loader.exec_module(BATCH)
finally:
    sys.modules.pop(SPEC.name, None)


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeRoctx:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.depth = 0
        self.roctxRangePushA = FakeFunction(self.push)
        self.roctxRangePop = FakeFunction(self.pop)

    def push(self, raw: bytes) -> int:
        self.calls.append(("push", raw.decode()))
        self.depth += 1
        return self.depth

    def pop(self) -> int:
        self.calls.append(("pop", None))
        self.depth -= 1
        return self.depth


def fake_library(tmp_path: Path) -> tuple[Path, str, FakeRoctx]:
    real = tmp_path / "real"
    real.mkdir()
    library = real / "libroctx64.so.4"
    library.write_bytes(b"fake-roctx-library")
    linked_root = tmp_path / "linked-rocm"
    linked_root.symlink_to(real, target_is_directory=True)
    invocation = linked_root / "libroctx64.so"
    leaf = real / "libroctx64.so"
    leaf.symlink_to(library.name)
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    return invocation, digest, FakeRoctx()


def marker(index: int) -> tuple[str, str]:
    kind = "warmup" if index < 2 else "measured"
    return (
        BATCH.roctx_marker_name(
            "profile-run",
            "resident-session",
            "case-a",
            "a" * 64,
            index,
            kind,
        ),
        kind,
    )


def test_safe_symlink_chain_load_and_exact_12_ranges(tmp_path: Path) -> None:
    invocation, digest, fake = fake_library(tmp_path)
    loaded_paths: list[str] = []

    def load_fd(path: str, **_kwargs):
        loaded_paths.append(path)
        return fake

    recorder = BATCH.RoctxRangeRecorder.load(
        invocation, digest, cdll_factory=load_fd
    )
    assert loaded_paths[0].startswith("/proc/self/fd/")
    assert recorder.identity.invocation_path == invocation
    assert recorder.identity.resolved_path.name == "libroctx64.so.4"
    assert [item.path for item in recorder.identity.components] == [
        tmp_path / "linked-rocm",
        invocation,
    ]
    for index in range(12):
        name, kind = marker(index)
        with recorder.range(name, index, kind):
            pass
    evidence = recorder.evidence()
    names = [item["name"] for item in evidence["ranges"]]
    assert len(names) == 12
    assert "/run_index=0/run_kind=warmup" in names[0]
    assert "/run_index=1/run_kind=warmup" in names[1]
    assert all("run_kind=measured" in name for name in names[2:])
    assert fake.depth == 0
    assert [call[0] for call in fake.calls] == [item for _ in range(12) for item in ("push", "pop")]
    assert evidence["library"]["sha256"] == digest
    assert evidence["measurement_eligible"] is False
    clone = json.loads(json.dumps(evidence))
    clone["audit_sha256"] = None
    assert evidence["audit_sha256"] == BATCH.sha_bytes(BATCH.canonical(clone))


def test_missing_library_sha_symbol_and_changed_symlink_fail_closed(tmp_path: Path) -> None:
    invocation, digest, fake = fake_library(tmp_path)
    with pytest.raises(BATCH.BatchError, match="unavailable"):
        BATCH.RoctxRangeRecorder.load(
            tmp_path / "missing" / "libroctx64.so",
            digest,
            cdll_factory=lambda *_args, **_kwargs: fake,
        )
    with pytest.raises(BATCH.BatchError, match="SHA-256 differs"):
        BATCH.RoctxRangeRecorder.load(
            invocation, "0" * 64, cdll_factory=lambda *_args, **_kwargs: fake
        )
    with pytest.raises(BATCH.BatchError, match="lacks"):
        BATCH.RoctxRangeRecorder.load(
            invocation, digest, cdll_factory=lambda *_args, **_kwargs: object()
        )
    recorder = BATCH.RoctxRangeRecorder.load(
        invocation, digest, cdll_factory=lambda *_args, **_kwargs: fake
    )
    (tmp_path / "linked-rocm").unlink()
    (tmp_path / "linked-rocm").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(BATCH.BatchError, match="component changed|resolved target changed"):
        recorder.identity.verify()


def test_exception_always_pops_and_incomplete_or_unbalanced_fails(tmp_path: Path) -> None:
    invocation, digest, fake = fake_library(tmp_path)
    recorder = BATCH.RoctxRangeRecorder.load(
        invocation, digest, cdll_factory=lambda *_args, **_kwargs: fake
    )
    name, kind = marker(0)
    with pytest.raises(RuntimeError, match="driver timeout"):
        with recorder.range(name, 0, kind):
            raise RuntimeError("driver timeout")
    assert recorder.active is None
    assert fake.depth == 0
    with pytest.raises(BATCH.BatchError, match="incomplete"):
        recorder.evidence()
    with pytest.raises(BATCH.BatchError, match="begin order"):
        recorder.begin(*marker(0), 0)  # type: ignore[arg-type]


def test_execute_run_marker_surrounds_send_receive_and_validation(monkeypatch) -> None:
    events: list[str] = []

    class Recorder:
        @contextmanager
        def range(self, name: str, index: int, kind: str):
            assert f"run_index={index}" in name and f"run_kind={kind}" in name
            events.append("push")
            try:
                yield
            finally:
                events.append("pop")

    monkeypatch.setattr(BATCH, "_send", lambda *_args: events.append("send"))

    def receive(*_args):
        events.append("receive")
        return {"raw": True}

    def validate(*_args):
        events.append("validate")
        return {"run_index": 2, "run_kind": "measured"}

    monkeypatch.setattr(BATCH, "_recv", receive)
    monkeypatch.setattr(BATCH, "validate_run", validate)
    case = {"case_id": "case-a", "case_sha256": "a" * 64}
    value = BATCH.execute_resident_run(
        object(), case, "resident-session", 2, "measured", 1.0, Recorder(), "profile-run"
    )
    assert value["run_index"] == 2
    assert events == ["push", "send", "receive", "validate", "pop"]

    events.clear()
    monkeypatch.setattr(
        BATCH,
        "validate_run",
        lambda *_args: (_ for _ in ()).throw(BATCH.BatchError("OOM/failure")),
    )
    with pytest.raises(BATCH.BatchError, match="OOM"):
        BATCH.execute_resident_run(
            object(), case, "resident-session", 2, "measured", 1.0, Recorder(), "profile-run"
        )
    assert events == ["push", "send", "receive", "pop"]


def test_profile_flags_are_explicit_and_normal_parse_is_unchanged() -> None:
    base = [
        "--expanded", "/expanded",
        "--fixture-index", "/fixtures",
        "--identity", "/identity",
        "--preflight", "/preflight",
        "--policy", "/policy",
        "--output-dir", "/output",
        "--run-id", "run",
        "--baseline-kind", "p3-current-head",
        "--dry-run",
    ]
    normal = BATCH.parse_args(base)
    assert normal.profile_roctx_ranges is False
    assert normal.roctx_library is None
    profiled = BATCH.parse_args(
        [
            *base[:-1],
            "--one-case-smoke",
            "--profile-roctx-ranges",
            "--roctx-library", "/libroctx64.so",
            "--roctx-library-sha256", "a" * 64,
        ]
    )
    assert profiled.profile_roctx_ranges is True
