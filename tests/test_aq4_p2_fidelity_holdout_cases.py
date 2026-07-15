from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_holdout_cases_adapter_binds_only_holdout_rows(tmp_path: Path) -> None:
    protocol_test = _module(ROOT / "tests/test_aq4_p2_fidelity_holdout_protocol.py", "protocol_test_for_holdout_cases")
    adapter = _module(ROOT / "tools/prepare-aq4-p2-fidelity-holdout-cases.py", "holdout_cases_adapter")
    expanded, index = protocol_test.FidelityProtocolTests().make_inputs(tmp_path)
    split = tmp_path / "split"
    generated = subprocess.run(["python3", str(ROOT / "tools/generate-aq4-p2-fidelity-holdout.py"), "split", "--expanded", str(expanded), "--fixture-index", str(index), "--output", str(split)], text=True, capture_output=True)
    assert generated.returncode == 0, generated.stderr
    expected = {"expected_split_manifest_sha256": hashlib.sha256((split / "split-manifest.json").read_bytes()).hexdigest(), "expected_policy_sha256": hashlib.sha256((split / "policy.json").read_bytes()).hexdigest(), "expected_calibration_cases_sha256": hashlib.sha256((split / "calibration-cases.jsonl").read_bytes()).hexdigest(), "expected_holdout_cases_sha256": hashlib.sha256((split / "holdout-cases.jsonl").read_bytes()).hexdigest()}
    output = tmp_path / "holdout-source-cases.json"
    args = type("Args", (), {"split_root": split, "output": output, **expected})()
    result = adapter.prepare(args)
    assert result["row_count"] == 24
    payload = json.loads(output.read_text(encoding="utf-8"))
    holdout_rows = [json.loads(line) for line in (split / "holdout-cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {case["case_id"] for case in payload["cases"]} == {row["case_id"] for row in holdout_rows}
    assert not any(case["case_id"] in {json.loads(line)["case_id"] for line in (split / "calibration-cases.jsonl").read_text(encoding="utf-8").splitlines()} for case in payload["cases"])
