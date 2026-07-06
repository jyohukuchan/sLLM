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


PROMPT_SUITE_TOOL = load_tool("compare-package-token-prompt-suite.py")
LOGITS_TOOL = load_tool("compare-package-token-logits.py")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prompt_report(
    *,
    generated_token_ids: list[int] | None = None,
    prefill_logit: float = 3.0,
    decode_logit: float = 4.0,
    verified: bool = True,
) -> dict:
    generated_token_ids = generated_token_ids or [30, 40]
    return {
        "prompt_token_ids": [10, 20],
        "generated_token_ids": generated_token_ids,
        "stop": {
            "reason": "stop_token",
            "stopped": True,
            "stopped_on_token_id": 99,
            "stopped_on_token_sequence": None,
        },
        "verified": verified,
        "prefill": {
            "top_logits": [
                {"token_id": 30, "logit": prefill_logit},
                {"token_id": 31, "logit": 2.0},
            ]
        },
        "decode": {
            "last_top_logits": [
                {"token_id": 99, "logit": decode_logit},
                {"token_id": 98, "logit": 1.0},
            ]
        },
    }


def prompt_summary(report_path: Path) -> dict:
    return {
        "schema_version": "package-token-prompt-suite-summary-v0.3",
        "suite": {"suite_id": "unit-suite"},
        "device_index": 0,
        "cases": [
            {
                "id": "case_a",
                "category": "unit",
                "report": str(report_path),
                "output_status": "ok",
            }
        ],
    }


class PromptSuiteGuardTests(unittest.TestCase):
    def test_prompt_suite_guard_passes_matching_tokens_and_logits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_report = root / "reference" / "case_a.json"
            candidate_report = root / "candidate" / "case_a.json"
            write_json(reference_report, prompt_report())
            write_json(candidate_report, prompt_report())
            reference_summary = root / "reference" / "summary.json"
            candidate_summary = root / "candidate" / "summary.json"
            write_json(reference_summary, prompt_summary(reference_report))
            write_json(candidate_summary, prompt_summary(candidate_report))

            report = PROMPT_SUITE_TOOL.build_report(
                types.SimpleNamespace(
                    reference_summary=reference_summary,
                    candidate_summary=candidate_summary,
                    reference_label="ref",
                    candidate_label="cand",
                    logit_atol=1e-6,
                )
            )

        self.assertTrue(report["metrics"]["passed"])
        self.assertEqual(report["metrics"]["generated_token_match_count"], 1)
        self.assertEqual(report["metrics"]["top_logits_match_count"], 1)
        self.assertEqual(report["metrics"]["max_prefill_top_logit_abs_diff"], 0.0)
        self.assertEqual(report["metrics"]["max_decode_last_top_logit_abs_diff"], 0.0)

    def test_prompt_suite_guard_fails_logit_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference_report = root / "reference" / "case_a.json"
            candidate_report = root / "candidate" / "case_a.json"
            write_json(reference_report, prompt_report())
            write_json(candidate_report, prompt_report(decode_logit=4.01))
            reference_summary = root / "reference" / "summary.json"
            candidate_summary = root / "candidate" / "summary.json"
            write_json(reference_summary, prompt_summary(reference_report))
            write_json(candidate_summary, prompt_summary(candidate_report))

            report = PROMPT_SUITE_TOOL.build_report(
                types.SimpleNamespace(
                    reference_summary=reference_summary,
                    candidate_summary=candidate_summary,
                    reference_label="ref",
                    candidate_label="cand",
                    logit_atol=1e-6,
                )
            )

        self.assertFalse(report["metrics"]["passed"])
        self.assertEqual(report["metrics"]["generated_token_match_count"], 1)
        self.assertEqual(report["metrics"]["top_logits_match_count"], 0)
        self.assertGreater(report["metrics"]["max_decode_last_top_logit_abs_diff"], 0.0)


def logits_report(*, token_id: int = 30, logit: float = 3.0, verified: bool = True) -> dict:
    return {
        "token_ids": [10, 20],
        "top_logits": [
            {"token_id": token_id, "logit": logit},
            {"token_id": 31, "logit": 2.0},
        ],
        "verified": verified,
        "device_index": 0,
        "timing_ms": {"total": 1.0},
    }


class LogitsGuardTests(unittest.TestCase):
    def test_logits_guard_passes_matching_top_logits(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            write_json(reference, logits_report())
            write_json(candidate, logits_report())

            report = LOGITS_TOOL.build_report(
                types.SimpleNamespace(
                    reference=reference,
                    candidate=candidate,
                    reference_label="ref",
                    candidate_label="cand",
                    logit_atol=1e-6,
                )
            )

        self.assertTrue(report["metrics"]["passed"])
        self.assertTrue(report["metrics"]["top_token_ids_match"])
        self.assertEqual(report["metrics"]["max_abs_logit_diff"], 0.0)

    def test_logits_guard_fails_top_token_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference = root / "reference.json"
            candidate = root / "candidate.json"
            write_json(reference, logits_report())
            write_json(candidate, logits_report(token_id=32))

            report = LOGITS_TOOL.build_report(
                types.SimpleNamespace(
                    reference=reference,
                    candidate=candidate,
                    reference_label="ref",
                    candidate_label="cand",
                    logit_atol=1e-6,
                )
            )

        self.assertFalse(report["metrics"]["passed"])
        self.assertFalse(report["metrics"]["top_token_ids_match"])


if __name__ == "__main__":
    unittest.main()
