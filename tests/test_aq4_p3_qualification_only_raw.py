from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


RAW = load("aq4_p3_qualification_only_raw_test", ROOT / "tools/build-aq4-p3-qualification-only-raw.py")
QFIX = load("aq4_p3_qualification_only_qfixture", ROOT / "tests/test_aq4_p3_upstream_qualification.py")


def fixture(root: Path) -> tuple[dict, Path, Path]:
    package = QFIX.rejection_package(root / "rejection")
    qualification = RAW.QUALIFICATION.build_rejection(package)
    qualification_path = root / "qualification.json"
    QFIX.write_json(qualification_path, qualification)
    archive_path = root / "p3-source.tar"
    archive_path.write_bytes(b"immutable synthetic source archive\n")
    value = RAW.build(qualification_path, "9" * 40, "8" * 40, archive_path)
    return value, qualification_path, archive_path


def test_rejected_qualification_produces_metric_free_canonical_no_eligible(tmp_path: Path) -> None:
    value, qualification_path, _archive = fixture(tmp_path)
    assert set(value) == RAW.SELECTOR.QUALIFICATION_ONLY_ROOT_FIELDS
    assert not ({"measurements", "capabilities", "full_model_pairs"} & set(value))
    path = tmp_path / "raw.json"
    QFIX.write_json(path, value)
    snapshot = RAW.SELECTOR.capture(path)
    selection = RAW.SELECTOR.select([(snapshot, RAW.SELECTOR.parse_json(snapshot))])
    assert selection["status"] == "no_eligible_candidate"
    assert selection["selected_candidate_id"] is None
    assert selection["input_binding"]["upstream_qualification_status"] == "rejected_no_go"
    assert selection["input_binding"]["qualification_only_p3_implementation"] == [value["p3_implementation"]]
    bindings = selection["input_binding"]["upstream_p2_terminal_bindings"]
    assert len(bindings) == 1
    qualification = json.loads(qualification_path.read_text())
    assert bindings[0] == RAW.SELECTOR.rejected_terminal_bindings(qualification)
    assert all(candidate["eligible"] is False for candidate in selection["candidates"])


@pytest.mark.parametrize("field", ["measurements", "capabilities", "full_model_pairs", "paired_full_model_95ci"])
def test_qualification_only_rejects_extra_metrics_and_fake_ci(tmp_path: Path, field: str) -> None:
    value, _qualification, _archive = fixture(tmp_path)
    changed = copy.deepcopy(value)
    changed[field] = [] if field != "paired_full_model_95ci" else {"ci95_lower_ms": 999.0}
    changed["evidence_sha256"] = RAW.SELECTOR.semantic_sha256(changed)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="fields differ"):
        RAW.SELECTOR.validate_raw(changed)


def test_qualification_only_rejects_qualified_go_cross_variant(tmp_path: Path) -> None:
    value, _qualification, _archive = fixture(tmp_path)
    go_root = tmp_path / "go"
    paths = QFIX.success_chain(go_root)
    qualified = RAW.QUALIFICATION.build_qualified(paths)
    qualified_path = tmp_path / "qualified.json"
    QFIX.write_json(qualified_path, qualified)
    changed = copy.deepcopy(value)
    changed["upstream_qualification"] = {
        "path": str(qualified_path), "sha256": RAW.SELECTOR.hashlib.sha256(qualified_path.read_bytes()).hexdigest(),
        "qualification_sha256": qualified["qualification_sha256"], "status": "qualified_go",
        "promotion_eligible": True, "reason": RAW.QUALIFICATION.GO_REASON,
    }
    changed["evidence_sha256"] = RAW.SELECTOR.semantic_sha256(changed)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="requires rejected_no_go"):
        RAW.SELECTOR.validate_raw(changed)


def test_qualification_only_rejects_hash_tamper(tmp_path: Path) -> None:
    value, _qualification, _archive = fixture(tmp_path)
    value["p3_implementation"]["source_archive"]["sha256"] = "0" * 64
    value["evidence_sha256"] = RAW.SELECTOR.semantic_sha256(value)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="source archive differs"):
        RAW.SELECTOR.validate_raw(value)


def test_qualification_only_rejects_archive_size_type_and_value(tmp_path: Path) -> None:
    for replacement in (True, 999999):
        root = tmp_path / str(replacement)
        root.mkdir()
        value, _qualification, _archive = fixture(root)
        value["p3_implementation"]["source_archive"]["size_bytes"] = replacement
        value["evidence_sha256"] = RAW.SELECTOR.semantic_sha256(value)
        with pytest.raises(RAW.SELECTOR.SelectionError, match="source archive differs"):
            RAW.SELECTOR.validate_raw(value)


def test_qualification_only_publish_is_no_overwrite_and_race_safe(tmp_path: Path) -> None:
    value, _qualification, _archive = fixture(tmp_path)
    output = tmp_path / "raw.json"
    barrier = threading.Barrier(8)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            RAW.publish(output, value)
            result = "ok"
        except (RAW.RawError, FileExistsError):
            result = "exists"
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert results.count("ok") == 1
    assert results.count("exists") == 7
    assert output.stat().st_nlink == 1
    before = output.read_bytes()
    with pytest.raises(RAW.RawError, match="refusing to overwrite"):
        RAW.publish(output, value)
    assert output.read_bytes() == before


def test_streaming_archive_snapshot_rejects_symlink_hardlink_and_size_cap(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"archive-bytes")
    symlink = tmp_path / "symlink.tar"
    symlink.symlink_to(archive)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="symlink"):
        RAW.SELECTOR.capture_digest(symlink)
    hardlink = tmp_path / "hardlink.tar"
    hardlink.hardlink_to(archive)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="single-link"):
        RAW.SELECTOR.capture_digest(archive)
    hardlink.unlink()
    with pytest.raises(RAW.SELECTOR.SelectionError, match="size cap"):
        RAW.SELECTOR.capture_digest(archive, True)
    with pytest.raises(RAW.SELECTOR.SelectionError, match="bounded"):
        RAW.SELECTOR.capture_digest(archive, len(b"archive-bytes") - 1)


@pytest.mark.parametrize("mutation", ["replace", "truncate", "grow"])
def test_streaming_archive_snapshot_rejects_mid_read_mutation(tmp_path: Path, mutation: str) -> None:
    archive = tmp_path / "archive.tar"
    archive.write_bytes(b"a" * 1024)

    def mutate(phase: str) -> None:
        if (mutation in {"replace", "truncate"} and phase != "after_open") or (
            mutation == "grow" and phase != "after_read"
        ):
            return
        if mutation == "replace":
            replacement = tmp_path / "replacement.tar"
            replacement.write_bytes(b"a" * 1024)
            os.replace(replacement, archive)
        elif mutation == "truncate":
            archive.write_bytes(b"short")
        else:
            with archive.open("ab") as handle:
                handle.write(b"grown")

    with pytest.raises(RAW.SELECTOR.SelectionError, match="changed while"):
        RAW.SELECTOR.capture_digest(archive, hook=mutate)
