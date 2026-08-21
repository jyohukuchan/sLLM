from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))
sys.path.insert(0, str(ROOT / "ci" / "tests"))

import run_phase36_session_c as runner  # noqa: E402
from test_phase36_session_c_runner import _populate  # noqa: E402


class Phase36SessionCSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads((ROOT / "ci/schema/phase36-mi300x-session-c-summary-v1.schema.json").read_text(encoding="utf-8"))

    def _summary(self) -> dict[str, object]:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        root = Path(holder.name)
        cli, model, lock, fp8_model, fp8_lock, source = _populate(root)
        return runner.aggregate(raw_dir=root / "raw", output_dir=root / "out", binary=cli, model=model, lock=lock, fp8_model=fp8_model, fp8_lock=fp8_lock, source_identity=source)

    def _rejects(self, document: dict[str, object]) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        self.assertTrue(list(Draft202012Validator(self.schema).iter_errors(document)))

    def test_complete_summary_matches_closed_schema(self) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        self.assertEqual(list(Draft202012Validator(self.schema).iter_errors(self._summary())), [])

    def test_schema_rejects_open_properties_and_false_cleanup(self) -> None:
        summary = self._summary()
        changed = copy.deepcopy(summary)
        changed["unexpected"] = True
        self._rejects(changed)
        changed = copy.deepcopy(summary)
        changed["cleanup"]["retryable_cleanup"] = 1
        self._rejects(changed)

    def test_exact_coverage_and_identity_separation_are_retained(self) -> None:
        summary = self._summary()
        self.assertEqual([row["mtp_width"] for row in summary["mtp"]["rows"]], [0, 2, 3, 4, 7, 8, 0, 3])
        self.assertEqual([row["mode"] for row in summary["mtp"]["rows"]], ["bf16-fp16"] * 6 + ["fp8-fp8"] * 2)
        self.assertIsNone(summary["mtp"]["rows"][0]["mtp_weight_dtype"])
        self.assertEqual(summary["mtp"]["rows"][1]["mtp_weight_dtype"], "bf16")
        self.assertEqual(summary["vision_lazy_residency"]["memory"]["gtt_peak_bytes"], 0)
        self.assertEqual(summary["vision_cli"]["image_pad_tokens"], 64)
        self.assertEqual(set(summary["vision_cli"]["asset_sha256"]), {"png", "jpeg", "webp"})
        self.assertTrue(summary["vision_cli"]["identical_outputs"])
        self.assertNotEqual(summary["identity"]["bf16_model"]["sha256"], summary["identity"]["fp8_model"]["sha256"])
        self.assertNotEqual(summary["identity"]["bf16_lock"]["sha256"], summary["identity"]["fp8_lock"]["sha256"])

    def test_unavailable_metric_is_not_encoded_as_numeric_zero(self) -> None:
        summary = self._summary()
        self.assertEqual(summary["openai_a6"]["amd_smi_metric"]["state"], "unavailable")
        self.assertIn("reason", summary["openai_a6"]["amd_smi_metric"])
        changed = copy.deepcopy(summary)
        changed["openai_a6"]["amd_smi_metric"] = {"state": "unavailable", "reason": "provider"}
        self.assertEqual(list(Draft202012Validator(self.schema).iter_errors(changed)) if Draft202012Validator else [], [])


if __name__ == "__main__":
    unittest.main()
