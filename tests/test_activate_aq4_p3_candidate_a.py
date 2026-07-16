from __future__ import annotations

import copy
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


ACTIVATION = load("aq4_p3_activation_test", ROOT / "tools/activate-aq4-p3-candidate-a.py")
FIXTURES = load("aq4_p3_selector_activation_fixtures", ROOT / "tests/test_select_aq4_p3_candidate.py")


def files(root: Path) -> tuple[Path, Path]:
    raw_path = root / "raw.json"
    FIXTURES.write_json(raw_path, FIXTURES.candidate_a_raw_fixture())
    snapshot = ACTIVATION.SELECTOR.capture(raw_path)
    selection = ACTIVATION.SELECTOR.select([(snapshot, ACTIVATION.SELECTOR.parse_json(snapshot))])
    selection_path = root / "selection.json"
    FIXTURES.write_json(selection_path, selection)
    return raw_path, selection_path


def test_verified_candidate_a_selection_builds_production_activation(tmp_path: Path) -> None:
    raw_path, selection_path = files(tmp_path)
    value = ACTIVATION.build(selection_path, [raw_path])
    result = ACTIVATION.validate(value)
    assert result["status"] == "valid_production_activation"
    assert result["candidate_id"] == ACTIVATION.CANDIDATE_ID
    assert value["upstream_qualification"]["status"] == "qualified_go"


@pytest.mark.parametrize("mutation", ["unknown", "missing", "bool", "candidate", "selection_hash", "qualification"])
def test_activation_contract_mutations_fail_closed(tmp_path: Path, mutation: str) -> None:
    raw_path, selection_path = files(tmp_path)
    changed = copy.deepcopy(ACTIVATION.build(selection_path, [raw_path]))
    if mutation == "unknown":
        changed["unknown"] = True
    elif mutation == "missing":
        del changed["profile"]
    elif mutation == "bool":
        changed["build"]["identity_sha256"] = True
    elif mutation == "candidate":
        changed["candidate"]["candidate_id"] = "paged-kv-table-validation-v1"
    elif mutation == "selection_hash":
        changed["selection"]["sha256"] = "0" * 64
    else:
        changed["upstream_qualification"]["status"] = "rejected_no_go"
        changed["upstream_qualification"]["promotion_eligible"] = False
    changed["activation_sha256"] = ACTIVATION.self_hash(changed)
    with pytest.raises((ACTIVATION.ActivationError, ACTIVATION.SELECTOR.SelectionError)):
        ACTIVATION.validate(changed)


def test_raw_or_selection_swap_invalidates_activation(tmp_path: Path) -> None:
    raw_path, selection_path = files(tmp_path)
    value = ACTIVATION.build(selection_path, [raw_path])
    selection = json.loads(selection_path.read_text())
    selection["selected_candidate_id"] = None
    FIXTURES.write_json(selection_path, selection)
    with pytest.raises(ACTIVATION.ActivationError):
        ACTIVATION.validate(value)
