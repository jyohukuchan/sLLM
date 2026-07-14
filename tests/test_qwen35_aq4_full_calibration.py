from __future__ import annotations

import importlib.util
import json
import shutil
import struct
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/source-oracle-v2"
LEGACY_CASES = ROOT / "tests/fixtures/qwen35-aq4-p2-oracle/cases.json"


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load_tool("full_calibration_validator", "validate-qwen35-aq4-p2-full-calibration.py")
comparator = load_tool("full_calibration_comparator", "compare-qwen35-aq4-p2-calibration.py")
legacy = validator.legacy_oracle


def canonical_sha(value):
    return legacy.canonical_sha256(value)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sha_sums(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{validator.sha256_file(path, path.name)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="ascii")


def make_artifact(root: Path, *, target: bool = False, perturb: bool = False) -> Path:
    root.mkdir()
    (root / "vectors").mkdir()
    old_manifest = legacy.validate_manifest(LEGACY_ROOT, expected_kind="independent_source")
    cases = json.loads(LEGACY_CASES.read_text(encoding="utf-8"))
    cases_value = {"schema_version": "ullm.qwen35_aq4_source_calibration_cases.v1", "cases": cases["cases"]}
    cases_path = root / "cases.json"
    write_json(cases_path, cases_value)
    rows = []
    hidden_out = (root / "vectors/hidden.f32le").open("wb")
    logits_out = (root / "vectors/logits.f32le").open("wb")
    try:
        for old in legacy.payload_records(LEGACY_ROOT, old_manifest):
            hidden_values = [0.0] * validator.HIDDEN_SIZE
            for index, value in zip(old["hidden_sample"]["indices"], old["hidden_sample"]["values"]):
                hidden_values[index] = float(value)
            logits_values = [-100.0] * validator.VOCAB_SIZE
            for index, value in zip(old["logit_sample"]["indices"], old["logit_sample"]["values"]):
                logits_values[index] = float(value)
            for entry in old["topk"]:
                logits_values[entry["token_id"]] = float(entry["logit"])
            if perturb and old["case_id"] == "fixture-prompt-0" and old["step"] == 0:
                hidden_values[0] += 1.0
            hidden_bytes = struct.pack(f"<{len(hidden_values)}f", *hidden_values)
            logits_bytes = struct.pack(f"<{len(logits_values)}f", *logits_values)
            hidden_offset = hidden_out.tell()
            logits_offset = logits_out.tell()
            hidden_out.write(hidden_bytes)
            logits_out.write(logits_bytes)
            rows.append({"case_id": old["case_id"], "step": old["step"], "semantic_input_id": old["case_id"], "observation": "first_token", "input_token_ids_sha256": legacy.canonical_token_ids_hash(next(item["prompt_token_ids"] for item in cases["cases"] if item["case_id"] == old["case_id"])), "hidden": {"offset_bytes": hidden_offset, "bytes": len(hidden_bytes), "elements": validator.HIDDEN_SIZE, "dtype": "f32", "endianness": "little", "sha256": __import__("hashlib").sha256(hidden_bytes).hexdigest(), "nonfinite_count": 0}, "logits": {"offset_bytes": logits_offset, "bytes": len(logits_bytes), "elements": validator.VOCAB_SIZE, "dtype": "f32", "endianness": "little", "sha256": __import__("hashlib").sha256(logits_bytes).hexdigest(), "nonfinite_count": 0}, "greedy_token_id": old["greedy_token_id"], "topk": old["topk"], "finite": True})
    finally:
        hidden_out.close()
        logits_out.close()
    rows_path = root / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    identity = json.loads(json.dumps(old_manifest["identity"]))
    identity.update({"hidden_size": validator.HIDDEN_SIZE, "vocab_size": validator.VOCAB_SIZE})
    if target:
        identity.update({"package_content_sha256": "a" * 64, "package_manifest_sha256": "b" * 64, "worker_binary_sha256": "c" * 64})
    legacy_payload = old_manifest["payload"]["sha256"]
    legacy_manifest_sha = validator.sha256_file(LEGACY_ROOT / "manifest.json", "legacy manifest")
    manifest = {"schema_version": comparator.TARGET_SCHEMA if target else validator.SCHEMA, "oracle_kind": "aq4_target" if target else validator.ORACLE_KIND, "status": "available", "evidence_class": "synthetic_fixture", "usable_as_source_evidence": not target, "promotion_eligible": False, "created_utc": old_manifest["created_utc"], "identity": identity, "parent_sampled_oracle": {"path": str((LEGACY_ROOT / "manifest.json").resolve()), "manifest_sha256": legacy_manifest_sha, "schema_version": legacy.SOURCE_SCHEMA}, "vector_contract": {"hidden_shape": [validator.HIDDEN_SIZE], "logits_shape": [validator.VOCAB_SIZE], "dtype": "f32", "endianness": "little", "layout": "flat", "chunk_elements": 65536, "row_bytes": validator.ROW_BYTES, "semantic_hidden": "final_rmsnorm_hidden_used_by_lm_head", "semantic_logits": "raw_pre_softmax_lm_head_logits"}, "limits": {"max_case_file_bytes": validator.MAX_CASE_FILE_BYTES, "max_cases": validator.MAX_CASES, "max_rows": validator.MAX_ROWS, "max_steps": validator.MAX_STEPS}, "cases": {"path": str(cases_path.resolve()), "sha256": validator.sha256_file(cases_path, "cases"), "case_count": 2, "row_count": 3}, "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"}, "runtime": {"device": "cpu", "dtype": "bfloat16", "max_resident_logit_rows": 1}, "legacy_cross_check": {"status": "passed", "legacy_manifest_sha256": legacy_manifest_sha, "legacy_payload_sha256": legacy_payload, "row_count": 3, "hidden_sample_max_abs_diff": 0.0, "logit_sample_max_abs_diff": 0.0}}
    write_json(root / "manifest.json", manifest)
    write_sha_sums(root)
    return root


def test_full_source_validator_and_comparator(tmp_path: Path):
    source = make_artifact(tmp_path / "source")
    candidate = make_artifact(tmp_path / "candidate", target=True, perturb=True)
    report = validator.validate(source)
    assert report["status"] == "valid"
    assert report["row_count"] == 3
    comparison_dir = tmp_path / "comparison"
    result = comparator.compare(comparator.load_artifact(source), comparator.load_artifact(candidate), "source_gate", comparison_dir)
    assert result["status"] == "valid"
    assert result["summary"]["row_count"] == 3
    assert result["summary"]["max_hidden_max_abs"] == pytest.approx(1.0)
    assert result["summary"]["greedy_mismatch_rows"] == 0
    assert result["observed_values_only"] is True
    assert (comparison_dir / "manifest.json").exists()


def test_path_gate_rejects_source_reference(tmp_path: Path):
    source = make_artifact(tmp_path / "source")
    candidate = make_artifact(tmp_path / "candidate", target=True)
    with pytest.raises(comparator.ComparisonError, match="path_gate reference"):
        comparator.compare(comparator.load_artifact(source), comparator.load_artifact(candidate), "path_gate", tmp_path / "comparison")


def test_duplicate_manifest_key_is_rejected(tmp_path: Path):
    source = make_artifact(tmp_path / "source")
    path = source / "manifest.json"
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace('"status": "available",', '"status": "available",\n  "status": "available",', 1), encoding="utf-8")
    with pytest.raises(validator.ValidationError, match="duplicate JSON key"):
        validator.validate(source)
