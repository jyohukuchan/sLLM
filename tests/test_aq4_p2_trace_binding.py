from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/prefill_validation/aq4_p2_trace_binding.py"
SPEC = importlib.util.spec_from_file_location("aq4_p2_trace_binding", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class Aq4P2TraceBindingTests(unittest.TestCase):
    def setup_run(self, root: Path) -> dict[str, Path | dict]:
        case = {"case_id": "case-1", "case_sha256": "a" * 64, "scope": "production_server"}
        identity = {"schema_version": "ullm.aq4_production_p2_identity.v2", "status": "bound", "identity_sha256": "b" * 64}
        policy = {"status": "bound", "hash_binding": {"policy_sha256": "c" * 64}}
        result = {
            "schema_version": "ullm.qwen35_aq4_p2.full_model_driver.v2",
            "status": "ok",
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "audit": {"coverage_complete": True},
            "lifecycle": {},
            "reset": {"attempted": 1, "complete": 1, "failed": 0},
            "outcome": "length",
            "fallback": {"unexpected_count": 0},
        }
        resource = {"schema_version": "ullm.aq4_p2_resource_observation.v1", "case_id": case["case_id"], "case_sha256": case["case_sha256"]}
        paths = {}
        for name, value in (("case.json", case), ("identity.json", identity), ("policy.json", policy), ("result.json", result), ("resource.json", resource)):
            path = root / name
            write(path, value)
            paths[name.split(".")[0]] = path
        paths["case_value"] = case
        paths["result_value"] = result
        return paths

    def test_existing_p1_trace_without_case_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.setup_run(root)
            old = root / "p1-trace.json"
            write(old, {"schema_version": "ullm.production_execution_trace.v1", "status": "ok", "scope": "production_server", "trace_id": "old"})
            with self.assertRaises(MODULE.TraceBindingError):
                MODULE.build_trace_sidecar(root, paths["case"], paths["identity"], paths["policy"], paths["result"], paths["resource"], old, paths["result_value"])

    def test_missing_production_trace_is_explicitly_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.setup_run(root)
            sidecar = MODULE.build_trace_sidecar(root, paths["case"], paths["identity"], paths["policy"], paths["result"], paths["resource"], None, paths["result_value"])
            self.assertEqual(sidecar["status"], "blocked")
            self.assertIn("production_trace_missing", sidecar["binding"]["reasons"])

    def test_case_bound_trace_and_resource_links_can_be_valid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self.setup_run(root)
            trace = root / "p2-trace.json"
            write(trace, {"schema_version": "ullm.production_execution_trace.v1", "status": "ok", "scope": "production_server", "trace_id": "p2-trace", "case_id": "case-1", "case_sha256": "a" * 64})
            sidecar = MODULE.build_trace_sidecar(root, paths["case"], paths["identity"], paths["policy"], paths["result"], paths["resource"], trace, paths["result_value"])
            self.assertEqual(sidecar["status"], "valid")
            self.assertEqual(sidecar["binding"]["reasons"], [])
            for role in ("identity", "policy", "result", "resource_observation", "production_trace"):
                self.assertEqual(Path(sidecar[role]["path"]).parent, root)


if __name__ == "__main__":
    unittest.main()
