import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tests.test_sq8_serving_session_matrix import common_result, completed_request


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-generation-matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_generation_matrix", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_tokens(module, prompt_tokens: int, generation_tokens: int) -> list[int]:
    path = (
        module.DEFAULT_FIXTURE_ROOT
        / "prompts"
        / f"raw-p{prompt_tokens:04d}"
        / f"greedy-g{generation_tokens}.u32le"
    )
    return [item[0] for item in struct.iter_unpack("<I", path.read_bytes())]


def generation_result(module, generation_tokens: int) -> dict:
    return common_result(
        module.session,
        [
            completed_request(
                module.session,
                prompt_tokens,
                max_new_tokens=generation_tokens,
                generated_tokens=source_tokens(module, prompt_tokens, generation_tokens),
            )
            for prompt_tokens in module.PROMPT_LENGTHS
        ],
        None,
    )


class Sq8ServingGenerationMatrixTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def evidence(self, root: Path):
        g8 = generation_result(self.module, 8)
        g64 = generation_result(self.module, 64)
        g8_path = root / "g8.json"
        g64_path = root / "g64.json"
        g8_path.write_text(json.dumps(g8), encoding="utf-8")
        g64_path.write_text(json.dumps(g64), encoding="utf-8")
        return g8_path, g64_path, g8, g64

    def test_validate_results_requires_exact_g8_and_reports_g64_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g8_path, g64_path, _, g64 = self.evidence(root)
            g64["requests"][1]["generated_token_ids"][28] += 1
            g64_path.write_text(json.dumps(g64), encoding="utf-8")
            result = self.module.validate_results(g8_path, g64_path)
            self.assertTrue(result["passed"])
            self.assertTrue(all(item["source_exact"] for item in result["g8"]))
            self.assertFalse(result["g64"][1]["source_exact"])
            self.assertEqual(result["g64"][1]["source_common_prefix_tokens"], 28)

    def test_validate_results_rejects_g8_token_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g8_path, g64_path, g8, _ = self.evidence(root)
            g8["requests"][2]["generated_token_ids"][3] += 1
            g8_path.write_text(json.dumps(g8), encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "fixed sequence"):
                self.module.validate_results(g8_path, g64_path)

    def test_validate_results_rejects_g64_first_token_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g8_path, g64_path, _, g64 = self.evidence(root)
            g64["requests"][3]["generated_token_ids"][0] += 1
            g64_path.write_text(json.dumps(g64), encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "first token differs"):
                self.module.validate_results(g8_path, g64_path)

    def test_validate_results_rejects_g64_cache_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            g8_path, g64_path, _, g64 = self.evidence(root)
            g64["requests"][0]["terminal_cache_lengths"][39] += 1
            g64_path.write_text(json.dumps(g64), encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "terminal/cache contract"):
                self.module.validate_results(g8_path, g64_path)


if __name__ == "__main__":
    unittest.main()
