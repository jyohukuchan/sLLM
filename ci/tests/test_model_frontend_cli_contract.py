import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/model-frontend-cli-report-v1.schema.json"


def base_report(command: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "model-frontend-cli-report-v1",
        "command": command,
        "state": "PASS",
        "model": {
            "repo_id": "Qwen/Qwen3.5-4B",
            "resolved_revision": "8" * 40,
            "lock_fingerprint": "sha256:" + "3" * 64,
        },
        "scope": {
            "offline": True,
            "gpu_execution": False,
            "model_execution": False,
            "generation": False,
        },
        "result": result,
    }


class ModelFrontendCliSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def assert_valid(self, report: dict[str, object]) -> None:
        self.assertEqual(list(self.validator.iter_errors(report)), [])

    def assert_invalid(self, report: dict[str, object]) -> None:
        self.assertNotEqual(list(self.validator.iter_errors(report)), [])

    def test_all_command_result_shapes_are_valid(self) -> None:
        self.assert_valid(base_report("tokenize", {"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}))
        self.assert_valid(base_report("render", {"kind": "render", "text": "prompt"}))
        self.assert_valid(base_report("decode", {"kind": "decode", "text": "decoded"}))
        self.assert_valid(base_report("verify-model", {
            "kind": "verify-model", "locked_files": 13, "verified_files": 13,
            "tensor_count": 738, "weight_entries": 738, "loadable_entries": 426,
            "known_unconsumed_entries": 312, "total_destination_bytes": 8_000_000_003,
            "plan_digest": "sha256:" + "9" * 64,
        }))

    def test_command_result_mismatch_and_partial_success_are_rejected(self) -> None:
        mismatch = base_report("tokenize", {"kind": "decode", "text": "x"})
        self.assert_invalid(mismatch)
        extra = base_report("decode", {"kind": "decode", "text": "x"})
        extra["partial"] = True
        self.assert_invalid(extra)
        gpu = copy.deepcopy(base_report("render", {"kind": "render", "text": "x"}))
        gpu["scope"]["gpu_execution"] = True
        self.assert_invalid(gpu)


if __name__ == "__main__":
    unittest.main()
