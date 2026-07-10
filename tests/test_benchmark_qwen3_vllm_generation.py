from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "benchmark-qwen3-vllm-generation.py"


def load_tool():
    spec = importlib.util.spec_from_file_location(
        "benchmark_qwen3_vllm_generation", TOOL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def fixed_request(*, token_ids=None, finish_reason="length", finished=True, metrics=None):
    output = SimpleNamespace(
        token_ids=(token_ids or TOOL.EXPECTED_GENERATED_TOKEN_IDS),
        finish_reason=finish_reason,
    )
    request = SimpleNamespace(outputs=[output], finished=finished, metrics=metrics)
    return request, output


class Qwen3VllmGenerationBenchmarkTests(unittest.TestCase):
    def test_fixed_generation_semantics_match_ullm_request(self) -> None:
        self.assertEqual(TOOL.GENERATION_STEPS, 8)
        self.assertEqual(TOOL.MIN_GENERATION_STEPS, 0)
        self.assertFalse(TOOL.IGNORE_EOS)
        self.assertFalse(TOOL.DETOKENIZE)
        self.assertEqual(
            TOOL.DEFAULT_OUTPUT,
            Path(
                "/tmp/ullm-qwen3-14b-fp8-vllm-generation-throughput-m8-g8-v0.2.json"
            ),
        )

    def test_linear_percentiles_and_summary(self) -> None:
        values = [float(value) for value in range(1, 11)]
        self.assertEqual(TOOL.percentile_linear(values, 0.0), 1.0)
        self.assertEqual(TOOL.percentile_linear(values, 50.0), 5.5)
        self.assertAlmostEqual(TOOL.percentile_linear(values, 95.0), 9.55)
        self.assertEqual(TOOL.percentile_linear(values, 100.0), 10.0)
        summary = TOOL.timing_summary(values)
        self.assertEqual(summary["count"], 10)
        self.assertEqual(summary["milliseconds"]["p50"], 5500.0)

    def test_generation_contract_accepts_only_fixed_tokens_and_length(self) -> None:
        request, output = fixed_request()
        self.assertEqual(TOOL.validate_generation_output([request]), (request, output))
        wrong, _ = fixed_request(token_ids=[0] * TOOL.GENERATION_STEPS)
        with self.assertRaisesRegex(RuntimeError, "outside the fixed oracle contract"):
            TOOL.validate_generation_output([wrong])
        stopped, _ = fixed_request(finish_reason="stop")
        with self.assertRaisesRegex(RuntimeError, "finish reason"):
            TOOL.validate_generation_output([stopped])

    def test_request_output_metrics_and_unavailable_reason(self) -> None:
        metrics = SimpleNamespace(
            first_token_latency=0.02,
            first_token_ts=100.0,
            last_token_ts=100.07,
            num_generation_tokens=TOOL.GENERATION_STEPS,
        )
        request, _ = fixed_request(metrics=metrics)
        record = TOOL.request_output_metrics(request)
        self.assertTrue(record["available"])
        self.assertAlmostEqual(record["ttft_seconds"], 0.02)
        self.assertAlmostEqual(record["decode_seconds"], 0.07)
        aggregate = TOOL.summarize_request_output_metrics([record] * 10)
        self.assertTrue(aggregate["available"])
        self.assertEqual(aggregate["ttft"]["count"], 10)

        request_without_metrics, _ = fixed_request(metrics=None)
        unavailable = TOOL.request_output_metrics(request_without_metrics)
        self.assertFalse(unavailable["available"])
        self.assertIn("is None", unavailable["unavailable_reason"])
        aggregate = TOOL.summarize_request_output_metrics([unavailable])
        self.assertFalse(aggregate["available"])
        self.assertIsNone(aggregate["decode"])

    def test_json_publication_never_replaces_existing_or_raced_output(self) -> None:
        oracle = TOOL.load_oracle_exporter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing.json"
            existing.write_text("unchanged\n", encoding="ascii")
            with self.assertRaises(FileExistsError):
                TOOL.publish_json_no_clobber(
                    existing, {"changed": True}, oracle.rename_noreplace
                )
            self.assertEqual(existing.read_text(encoding="ascii"), "unchanged\n")
            self.assertEqual(list(root.glob(".existing.json.incomplete-*")), [])

            dangling = root / "dangling.json"
            os.symlink(root / "missing", dangling)
            with self.assertRaises(FileExistsError):
                TOOL.publish_json_no_clobber(
                    dangling, {"changed": True}, oracle.rename_noreplace
                )
            self.assertTrue(dangling.is_symlink())
            self.assertEqual(list(root.glob(".dangling.json.incomplete-*")), [])


if __name__ == "__main__":
    unittest.main()
