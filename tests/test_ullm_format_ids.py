import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tools" / "ullm_format_ids.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ullm_format_ids", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FORMAT_IDS = load_module()


class UllmFormatIdsTest(unittest.TestCase):
    def test_canonicalizes_aq4_legacy_ids(self):
        self.assertEqual(FORMAT_IDS.canonical_format_id("AQ4_0"), "AQ4_0")
        self.assertEqual(FORMAT_IDS.canonical_format_id("aq4"), "AQ4_0")
        self.assertEqual(
            FORMAT_IDS.canonical_format_id("aq4-prototype-current-runtime"),
            "AQ4_0",
        )
        self.assertEqual(
            FORMAT_IDS.canonical_format_id("aq4_e4m3_g16_ts_flloyd16"),
            "AQ4_0",
        )

    def test_canonicalizes_sq8_legacy_ids(self):
        self.assertEqual(FORMAT_IDS.canonical_format_id("SQ8_0"), "SQ8_0")
        self.assertEqual(FORMAT_IDS.canonical_format_id("sq"), "SQ8_0")
        self.assertEqual(FORMAT_IDS.canonical_format_id("sq-format-v0.1"), "SQ8_0")
        self.assertEqual(
            FORMAT_IDS.canonical_format_id("sq-fp8-w8a16-r9700-v0-qkv-layer23-k16"),
            "SQ8_0",
        )

    def test_unknown_ids_are_not_rewritten(self):
        self.assertIsNone(FORMAT_IDS.canonical_format_id("bf16"))
        self.assertEqual(FORMAT_IDS.canonical_or_original("bf16"), "bf16")


if __name__ == "__main__":
    unittest.main()
