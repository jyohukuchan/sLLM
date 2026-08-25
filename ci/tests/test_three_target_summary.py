import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "ci/tools/aggregate_three_target_summary.py"
SCHEMA_PATH = ROOT / "ci/schema/three-target-gpu-summary-v1.schema.json"
spec = importlib.util.spec_from_file_location("aggregate_three_target_summary", TOOL_PATH)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def _stats(base: int) -> dict[str, int]:
    return {"median": base, "mad": 1, "count": 3, "min": base - 1, "max": base + 1}


def _row(target: str, index: int, *, failure: dict | None = None) -> dict:
    case_id, input_count, output_count = tool.CASE_SPECS[index]
    metric = {name: _stats(100 + index) for name in tool.METRICS}
    return {
        "case_id": case_id,
        "input_token_count": input_count,
        "requested_output_tokens": output_count,
        "protocol": {"warmups": 1 if index >= 5 else 3, "measured": 3 if index >= 5 else 10, "context_length": 131072 if index >= 5 else input_count + output_count, "ignore_eos": index == 6},
        "row_ids": {"sllm": f"{target}-sllm-{case_id}", "llama": f"{target}-llama-{case_id}"},
        "measured_sample_count": {"sllm": 0 if failure else (3 if index >= 5 else 10), "llama": 3 if index >= 5 else 10},
        "tokens": {
            "input_sha256": "a" * 64,
            "generated_sha256": {"sllm": None if failure else "b" * 64, "llama": "c" * 64},
            "visible_sha256": {"sllm": None if failure else "b" * 64, "llama": "c" * 64},
            "stop_sha256": {"sllm": None if failure else "d" * 64, "llama": "d" * 64},
            "generated_equal": None if failure else True,
            "visible_equal": None if failure else True,
            "stop_equal": None if failure else True,
        },
        "metrics": {"sllm": None if failure else metric, "llama": metric},
        "gates": {name: None if failure else {"sllm_median": 100 + index, "sllm_mad": 1, "llama_median": 100 + index, "llama_mad": 1, "limit": 101 + index, "pass": True} for name in tool.METRICS},
        "failures": {"sllm": failure, "llama": None} if failure else None,
    }


def _summary(target: str, *, state: str = "PASS", failure: dict | None = None) -> dict:
    rows = [_row(target, index, failure=failure if index == 5 else None) for index in range(7)]
    schema_version = tool.TARGETS[target]["schema_version"]
    return {
        "schema_version": schema_version,
        "state": state,
        "target": target,
        "gpu_uuid": f"GPU-{target}",
        "gpu_bdf": "0000:00:00.0",
        "matrix": {"cases": list(tool.CASE_IDS), "row_count": 7},
        "rows": rows,
    }


class ThreeTargetSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text())
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def test_three_targets_are_bounded_and_schema_valid(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        document = tool.aggregate_target_summaries(summaries)
        self.validator.validate(document)
        self.assertEqual(document["state"], "PASS")
        self.assertEqual(document["matrix"]["target_count"], 3)
        self.assertEqual([len(item["rows"]) for item in document["targets"]], [7, 7, 7])
        self.assertEqual({item["family"] for item in document["gpu_family_breakdown"]}, {"RDNA2", "RDNA4", "CDNA3"})
        self.assertFalse(document["target_selector"]["fallback_allowed"])
        encoded = json.dumps(document)
        self.assertNotIn("input_token_ids", encoded)
        self.assertNotIn("generated_token_ids", encoded)

    def test_resource_failure_and_explicit_constraint_are_fail_closed(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        summaries["gfx942"] = _summary("gfx942", state="FAIL", failure={"kind": "oom", "reason": "long context exhausted HBM"})
        document = tool.aggregate_target_summaries(
            summaries,
            known_constraints=[{"id": "manual-1", "target": "all", "scope": "performance", "reason": "performance parity is informative"}],
        )
        self.validator.validate(document)
        self.assertFalse(document["resources"]["all_pass"])
        self.assertEqual(document["resources"]["by_target"][2]["status"], "FAIL")
        self.assertEqual(document["targets"][2]["rows"][5]["row_state"], "FAIL")
        ids = {item["id"] for item in document["known_constraints"]}
        self.assertIn("manual-1", ids)
        self.assertIn("failure-gfx942-long-100000-sllm", ids)

    def test_explicit_gating_constraints_update_target_status(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        document = tool.aggregate_target_summaries(
            summaries,
            known_constraints=[
                {"id": "manual-oom", "target": "gfx942", "scope": "summary", "reason": "oom: explicit resource limit"},
                {"id": "manual-correctness", "target": "all", "scope": "summary", "reason": "correctness: shared oracle defect"},
            ],
        )
        self.validator.validate(document)
        self.assertFalse(document["correctness"]["all_pass"])
        self.assertTrue(all(item["status"] == "FAIL" for item in document["correctness"]["by_target"]))
        self.assertTrue(all(item["constraint_count"] == 1 for item in document["correctness"]["by_target"]))
        resources = {item["target"]: item["status"] for item in document["resources"]["by_target"]}
        self.assertEqual(resources, {"gfx1030": "PASS", "gfx1201": "PASS", "gfx942": "FAIL"})

    def test_wrong_target_or_matrix_is_rejected(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        summaries["gfx1030"]["target"] = "gfx1201"
        with self.assertRaises(tool.ThreeTargetError):
            tool.aggregate_target_summaries(summaries)

    def test_per_row_identity_is_preserved_and_schema_bounded(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        split = summaries["gfx1030"]
        split["_identity_scope"] = "per-row-mixed"
        split["_row_sources"] = [
            {
                "case_id": case_id,
                "sllm": {"path": f"/evidence/{case_id}/sllm.json", "sha256": "a" * 64},
                "llama": {"path": f"/evidence/{case_id}/llama.json", "sha256": "b" * 64},
                "artifact_sha256": "c" * 64 if index < 5 else None,
                "llama_artifact_sha256": None,
                "identity_scope": "final-adopted" if index < 5 else "binary-sha-unavailable",
                "comparability": "comparable" if index < 5 else "non-comparable",
                "notes": "bounded fixture provenance",
            }
            for index, (case_id, _, _) in enumerate(tool.CASE_SPECS)
        ]
        document = tool.aggregate_target_summaries(summaries)
        self.validator.validate(document)
        gfx1030 = document["targets"][0]
        self.assertEqual(gfx1030["identity_scope"], "per-row-mixed")
        self.assertEqual(gfx1030["source"]["kind"], "per-row-inputs")
        self.assertEqual(gfx1030["row_sources"][5]["identity_scope"], "binary-sha-unavailable")
        self.assertEqual(gfx1030["row_sources"][6]["comparability"], "non-comparable")

    def test_per_row_identity_requires_all_frozen_sources(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        summaries["gfx1030"]["_identity_scope"] = "per-row-mixed"
        summaries["gfx1030"]["_row_sources"] = []
        with self.assertRaises(tool.ThreeTargetError):
            tool.aggregate_target_summaries(summaries)
        summaries["gfx1030"].pop("_row_sources")
        with self.assertRaises(tool.ThreeTargetError):
            tool.aggregate_target_summaries(summaries)
        summaries = {target: _summary(target) for target in tool.TARGETS}
        summaries["gfx1201"]["rows"].pop()
        with self.assertRaises(tool.ThreeTargetError):
            tool.aggregate_target_summaries(summaries)

    def test_cli_refuses_symlink_and_duplicate_existing_output(self):
        summaries = {target: _summary(target) for target in tool.TARGETS}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for target, summary in summaries.items():
                path = root / f"{target}.json"
                path.write_text(json.dumps(summary))
                paths[target] = path
            symlink = root / "symlink.json"
            symlink.symlink_to(paths["gfx1030"])
            with self.assertRaises(tool.ThreeTargetError):
                tool.load_json(symlink)


if __name__ == "__main__":
    unittest.main()
