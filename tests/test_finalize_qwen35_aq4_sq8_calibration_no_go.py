from __future__ import annotations

import copy
import importlib.util
import json
import stat
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sq8_no_go_test", ROOT / "tools/finalize-qwen35-aq4-sq8-calibration-no-go.py")
assert SPEC and SPEC.loader
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


def _rows() -> dict[tuple[str, int], dict]:
    rows = {}
    for index in range(24):
        rows[(f"case-{index:02d}", 0)] = {
            "logits": {"relative_l2": 1.01 + index / 100 if index < 20 else 0.9},
            "hidden": {"relative_l2": 0.5 + index / 1000},
            "greedy": {"source": index, "target": index + int(index != 23)},
            "top_k_overlap": index % 10,
        }
    return rows


def test_observed_no_go_is_derived_without_clipping() -> None:
    observed = finalizer._derive_observed(_rows())
    assert observed["row_count"] == 24
    assert observed["logits_relative_l2"]["count_above_ceiling"] == 20
    assert observed["logits_relative_l2"]["maximum"] == pytest.approx(1.20)
    assert observed["hidden_relative_l2"]["count_above_ceiling"] == 0
    assert observed["greedy_mismatch_rows"] == 23
    assert observed["minimum_top_k_overlap"] == 0


@pytest.mark.parametrize(
    "mutate",
    (
        lambda rows: rows.pop(("case-00", 0)),
        lambda rows: rows[("case-00", 0)]["logits"].__setitem__("relative_l2", float("nan")),
        lambda rows: rows[("case-00", 0)]["hidden"].__setitem__("relative_l2", -0.1),
        lambda rows: rows[("case-00", 0)].__setitem__("top_k_overlap", 0.0),
        lambda rows: rows[("case-00", 0)].__setitem__("greedy", {"source": 1}),
        lambda rows: [row["logits"].__setitem__("relative_l2", 0.9) for row in rows.values()],
    ),
)
def test_observed_shape_type_finite_and_rejection_mutations_fail(mutate) -> None:
    rows = _rows()
    mutate(rows)
    with pytest.raises(finalizer.NoGoEvidenceError):
        finalizer._derive_observed(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _capture_fixture(tmp_path: Path) -> tuple[dict, dict]:
    commit = "1" * 40
    tree = "2" * 40
    gate_script = tmp_path / "gate.sh"
    gate_script.write_text(f'[[ "$(git -C "$REPO" rev-parse HEAD)" = {commit} ]]\n[[ "$(git -C "$REPO" rev-parse HEAD^{{tree}})" = {tree} ]]\n')
    gate_log = tmp_path / "gate.log"
    gate_log.write_text("capture_finish=x\nmanifest.json: OK\nrows.jsonl: OK\nvectors/hidden.f32le: OK\nvectors/logits.f32le: OK\nerror: the following arguments are required: --artifact\ngate_finish=x rc=2\n")
    binary = tmp_path / "capture"
    binary.write_bytes(b"capture")
    binary.chmod(0o555)
    staged = tmp_path / "staged.json"
    _write_json(staged, {"schema_version": "ullm.aq4_fidelity_capture_staged_binary.v1", "status": "ready", "execution_contract": {}, "source": {}, "staged": {"path": str(binary), "sha256": finalizer._sha(binary, "binary"), "bytes": binary.stat().st_size, "mode": "0555", "nlink": 1}})
    offline = tmp_path / "offline.json"
    _write_json(offline, {"command": "validate-target", "status": "ok", "result": {"validator_modified_artifact": False, "report": {"status": "valid", "row_count": 24, "nonfinite_rows": 0}}})
    before = tmp_path / "before.json"
    _write_json(before, {"schema_version": "ullm.aq4_fidelity_service_snapshot.v1", "status": "ready", "service": {"active": True, "running": True, "healthy": True, "nrestarts": 0}})
    stopped = tmp_path / "stopped.json"
    observation = {"owners": {"worker_pids": [], "amd_pids": [], "kfd_pids": []}, "service": {"active": False, "running": False}}
    _write_json(stopped, {"schema_version": "ullm.aq4_fidelity_stopped_stable2.v1", "status": "passed", "observations": [observation, copy.deepcopy(observation)]})
    restore = tmp_path / "restore.json"
    _write_json(restore, {"schema_version": "ullm.aq4_fidelity_service_restore.v1", "status": "passed", "service": {"active": True, "running": True, "healthy": True, "nrestarts": 0, "worker_pid": 42}, "owners": {"worker_pids": [42], "amd_pids": [42], "kfd_pids": [42]}})
    refs = {
        "gate_script": finalizer._ref(gate_script, "gate"),
        "gate_log": finalizer._ref(gate_log, "log"),
        "gate_exit_code": 2,
        "capture_completed": True,
        "post_capture_failure": "validator_cli_missing_required_artifact_argument",
        "staged_binary_receipt": finalizer._ref(staged, "staged"),
        "staged_binary": finalizer._ref(binary, "binary"),
        "offline_target_validation": finalizer._ref(offline, "offline"),
        "service_before": finalizer._ref(before, "before"),
        "stopped_stable2": finalizer._ref(stopped, "stopped"),
        "service_restore": finalizer._ref(restore, "restore"),
    }
    return refs, {"capture_commit": commit, "capture_tree": tree}


def test_capture_and_restore_history_is_exact(tmp_path: Path) -> None:
    capture, lineage = _capture_fixture(tmp_path)
    finalizer._validate_capture_history(capture, lineage)
    for name, value in (("unknown", True), ("gate_exit_code", 2.0), ("capture_completed", False)):
        tampered = copy.deepcopy(capture)
        tampered[name] = value
        with pytest.raises(finalizer.NoGoEvidenceError):
            finalizer._validate_capture_history(tampered, lineage)


def test_package_is_create_new_read_only_and_hash_bound(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(finalizer, "_validate_receipt", lambda value: {"status": "valid_no_go"})
    output = tmp_path / "evidence"
    result = finalizer.publish({"fixed": True}, output)
    assert result["status"] == "published_no_go"
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 and path.stat().st_nlink == 1 for path in output.iterdir())
    checked = finalizer.validate_package(output)
    assert checked["status"] == "valid_no_go"
    with pytest.raises(finalizer.NoGoEvidenceError):
        finalizer.publish({"fixed": True}, output)
