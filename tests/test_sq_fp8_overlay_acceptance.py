from __future__ import annotations

import importlib.util
import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str):
    path = REPO_ROOT / "tools" / filename
    module_name = filename.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool("evaluate-sq-fp8-overlay-acceptance.py")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SqFp8OverlayAcceptanceTests(unittest.TestCase):
    def test_strict_top1_accepts_all_matching_cases(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard.json"
            write_json(
                path,
                {
                    "layers": "3,7",
                    "token_ids": "1,2,3,4",
                    "cases": [
                        {
                            "name": "candidate",
                            "verified": True,
                            "fp8_tensor_count": 12,
                            "baseline_top1": 10,
                            "sq_top1": 10,
                            "top1_match": True,
                            "baseline_top1_rank_in_sq_topk": 1,
                            "topk_common": 6,
                            "sq_top1_minus_baseline_top1_logit": 0.0,
                        }
                    ],
                },
            )

            result = TOOL.evaluate(
                types.SimpleNamespace(
                    input_json=[path],
                    promotion_rule="strict_top1",
                    diagnostic_min_topk_common=5,
                    diagnostic_max_baseline_rank=2,
                    diagnostic_max_top1_gap=0.15,
                )
            )

        self.assertTrue(result["summary"]["accepted_for_t2_promotion"])
        self.assertEqual(result["summary"]["strict_top1_pass_count"], 1)
        self.assertEqual(result["summary"]["diagnostic_topk_pass_count"], 1)
        self.assertTrue(result["cases"][0]["strict_top1_pass"])

    def test_topk_diagnostic_does_not_override_strict_top1_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "guard.json"
            write_json(
                path,
                {
                    "layers": "23",
                    "token_ids": "1,2,3,4",
                    "cases": [
                        {
                            "name": "candidate",
                            "verified": True,
                            "fp8_tensor_count": 6,
                            "baseline_top1": 10,
                            "sq_top1": 11,
                            "top1_match": False,
                            "baseline_top1_rank_in_sq_topk": 2,
                            "topk_common": 7,
                            "sq_top1_minus_baseline_top1_logit": 0.01,
                        }
                    ],
                },
            )

            result = TOOL.evaluate(
                types.SimpleNamespace(
                    input_json=[path],
                    promotion_rule="strict_top1",
                    diagnostic_min_topk_common=5,
                    diagnostic_max_baseline_rank=2,
                    diagnostic_max_top1_gap=0.15,
                )
            )

        case = result["cases"][0]
        self.assertFalse(result["summary"]["accepted_for_t2_promotion"])
        self.assertFalse(case["strict_top1_pass"])
        self.assertIn("top1_mismatch", case["strict_top1_failure_reasons"])
        self.assertTrue(case["diagnostic_topk_pass"])


if __name__ == "__main__":
    unittest.main()
