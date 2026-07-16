from __future__ import annotations

import importlib.util
import json
import sys
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


FINALIZER = load("aq4_p3_no_eligible_finalizer_test", ROOT / "tools/finalize-aq4-p3-no-eligible-package.py")
RAW = load("aq4_p3_no_eligible_raw_fixture", ROOT / "tools/build-aq4-p3-qualification-only-raw.py")
QFIX = load("aq4_p3_no_eligible_q_fixture", ROOT / "tests/test_aq4_p3_upstream_qualification.py")


def package(root: Path) -> None:
    root.mkdir()
    rejection = QFIX.rejection_package(root.parent / "p2-rejection")
    qualification = FINALIZER.QUALIFICATION.build_rejection(rejection)
    qualification_path = root / "upstream-qualification.json"
    QFIX.write_json(qualification_path, qualification)
    archive = root / "p3-source.tar"
    archive.write_bytes(b"source archive\n")
    raw = RAW.build(qualification_path, "9" * 40, "8" * 40, archive)
    raw_path = root / "qualification-only-raw.json"
    QFIX.write_json(raw_path, raw)
    snapshot = FINALIZER.SELECTOR.capture(raw_path)
    selection = FINALIZER.SELECTOR.select([(snapshot, FINALIZER.SELECTOR.parse_json(snapshot))])
    QFIX.write_json(root / "selection.json", selection)


def test_finalizer_creates_immutable_exact_inventory(tmp_path: Path) -> None:
    root = tmp_path / "package"
    package(root)
    result = FINALIZER.finalize(root)
    assert result["status"] == "immutable_no_eligible"
    assert root.stat().st_mode & 0o777 == 0o555
    assert {item.name for item in root.iterdir()} == set(FINALIZER.INVENTORY) | {"SHA256SUMS"}
    for item in root.iterdir():
        assert item.stat().st_mode & 0o777 == 0o444
        assert item.stat().st_nlink == 1
    with pytest.raises(FINALIZER.FinalizeError, match="overwrite"):
        FINALIZER.finalize(root)


def test_finalizer_rejects_selection_tamper_without_publishing_sums(tmp_path: Path) -> None:
    root = tmp_path / "package"
    package(root)
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_candidate_id"] = "sequence-output-direct-v1"
    QFIX.write_json(selection_path, selection)
    with pytest.raises(FINALIZER.FinalizeError, match="selection differs"):
        FINALIZER.finalize(root)
    assert not (root / "SHA256SUMS").exists()
