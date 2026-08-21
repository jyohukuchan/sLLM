from __future__ import annotations

import copy
import hashlib
import json
import statistics
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "ci/matrix/r9700-sllm-llama-e2e-v1.json"
SCHEMA = ROOT / "ci/schema/r9700-sllm-llama-e2e-v1.schema.json"


class R9700SllmLlamaE2ESummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    def _schema_errors(self, document: dict[str, object]) -> list[object]:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        return list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document))

    def test_schema_and_fixed_scope(self) -> None:
        self.assertEqual(self._schema_errors(self.summary), [])
        self.assertEqual(self.summary["hardware"]["target"], "gfx1201")
        self.assertEqual(self.summary["hardware"]["gpu_uuid"], "GPU-a8e9ddefa2d60f55")
        self.assertEqual(self.summary["hardware"]["bdf"], "0000:07:00.0")
        self.assertEqual(self.summary["protocol"]["input_token_count"], 10_001)
        self.assertEqual(self.summary["protocol"]["requested_output_tokens"], 2)
        self.assertEqual(self.summary["protocol"]["generated_token_ids"], [23066, 23066])
        self.assertEqual(self.summary["classification"], "E1_SYSTEM_EQUIVALENT")
        self.assertFalse(self.summary["strict_identical"])

    def test_schema_rejects_missing_or_invalid_identity_fields(self) -> None:
        mutations = {
            "hardware target missing": lambda d: d["hardware"].pop("target"),
            "hardware BDF invalid": lambda d: d["hardware"].__setitem__("bdf", "0000:ff:00.0"),
            "software ROCm missing": lambda d: d["software"].pop("rocm"),
            "protocol token count invalid": lambda d: d["protocol"].__setitem__("input_token_count", 10_000),
            "protocol generated IDs missing": lambda d: d["protocol"].pop("generated_token_ids"),
            "sLLM source missing": lambda d: d["engines"]["sllm"]["source"].pop("base_commit"),
            "sLLM model digest invalid": lambda d: d["engines"]["sllm"]["model"].__setitem__("sha256", "bad"),
            "sLLM binary digest wrong": lambda d: d["engines"]["sllm"]["binary"].__setitem__("sha256", "0" * 64),
            "sLLM binary identity missing": lambda d: d["engines"]["sllm"]["binary"].pop("build_id"),
            "sLLM metric values missing": lambda d: d["engines"]["sllm"]["metrics"]["e2e_ns"].pop("values"),
            "llama source target invalid": lambda d: d["engines"]["llama_cpp"]["source"].__setitem__("hip_architectures", "gfx942"),
            "llama model digest missing": lambda d: d["engines"]["llama_cpp"]["model"].pop("sha256"),
            "llama binary digest invalid": lambda d: d["engines"]["llama_cpp"]["binary"].__setitem__("sha256", "bad"),
            "comparison ratio missing": lambda d: d["comparison"]["sllm_over_llama_ratio"].pop("e2e_ns"),
            "health cleanup missing": lambda d: d["health"].pop("post_cleanup_pass"),
            "raw digest invalid": lambda d: d["raw_artifacts"].__setitem__("pre_static_sha256", "bad"),
            "raw reference size missing": lambda d: d["engines"]["sllm"]["raw"].pop("size_bytes"),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                document = copy.deepcopy(self.summary)
                mutate(document)
                self.assertTrue(self._schema_errors(document), label)

    def test_statistics_and_ratios_are_recomputable(self) -> None:
        engines = self.summary["engines"]
        for engine in engines.values():
            for metric in engine["metrics"].values():
                values = metric["values"]
                median = statistics.median(values)
                mad = statistics.median(abs(value - median) for value in values)
                self.assertAlmostEqual(metric["median"], median)
                self.assertAlmostEqual(metric["mad"], mad)
                self.assertEqual(metric["min"], min(values))
                self.assertEqual(metric["max"], max(values))
        ratios = self.summary["comparison"]["sllm_over_llama_ratio"]
        for name, ratio in ratios.items():
            expected = engines["sllm"]["metrics"][name]["median"] / engines["llama_cpp"]["metrics"][name]["median"]
            self.assertAlmostEqual(ratio, expected)
        self.assertAlmostEqual(
            self.summary["comparison"]["e2e_percent_slower"],
            (ratios["e2e_ns"] - 1.0) * 100.0,
        )

    def test_execution_health_is_fail_closed(self) -> None:
        self.assertTrue(self.summary["engines"]["sllm"]["audit"]["all_dispatches_hip"])
        self.assertFalse(self.summary["engines"]["sllm"]["audit"]["fallback_used"])
        self.assertEqual(self.summary["engines"]["sllm"]["audit"]["cleanup_failures"], 0)
        self.assertTrue(self.summary["engines"]["llama_cpp"]["audit"]["full_gpu_offload"])
        self.assertEqual(self.summary["engines"]["llama_cpp"]["audit"]["cleanup_failures"], 0)
        self.assertEqual(self.summary["health"]["pre_process_count"], 0)
        self.assertEqual(self.summary["health"]["post_process_count"], 0)
        self.assertTrue(self.summary["health"]["post_cleanup_pass"])

    def test_external_raw_evidence_is_bound_when_available(self) -> None:
        raw_root = Path(self.summary["raw_artifacts"]["root"])
        paths = {
            name: raw_root / self.summary["engines"][name]["raw"]["file"]
            for name in ("sllm", "llama_cpp")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            self.skipTest(f"external raw evidence unavailable: {', '.join(missing)}")

        expected_input = [self.summary["protocol"]["input_token_id"]] * self.summary["protocol"]["input_token_count"]
        expected_output = self.summary["protocol"]["generated_token_ids"]
        metric_names = ("ttft_ns", "prefill_ns", "prefill_tokens_per_second", "tpot_ns", "decode_tokens_per_second", "e2e_ns")

        for name, path in paths.items():
            reference = self.summary["engines"][name]["raw"]
            payload = path.read_bytes()
            self.assertEqual(reference["size_bytes"], len(payload), name)
            self.assertEqual(reference["sha256"], hashlib.sha256(payload).hexdigest(), name)
            raw = json.loads(payload.decode("utf-8"))
            self.assertEqual(raw["state"], "PASS", name)
            self.assertEqual(raw["warmups"]["count"], 3, name)
            self.assertEqual(raw["measured"]["count"], 10, name)
            if name == "sllm":
                self.assertEqual(raw["benchmark_schema_version"], "engine-performance-direct-v1")
                self.assertEqual(raw["lane"], "direct")
                self.assertEqual(raw["audit"]["target"], self.summary["hardware"]["target"])
                self.assertTrue(raw["audit"]["all_dispatches_hip"])
                self.assertFalse(raw["audit"]["fallback_used"])
                self.assertEqual(raw["config"]["input_token_ids"], expected_input)
                self.assertEqual(raw["config"]["input_token_count"], 10_001)
                self.assertEqual(raw["config"]["max_new_tokens"], 2)
                self.assertEqual(raw["identities"]["model"]["repo_id"], self.summary["engines"][name]["model"]["repo_id"])
                self.assertEqual(raw["identities"]["model"]["resolved_revision"], self.summary["engines"][name]["model"]["revision"])
            else:
                self.assertEqual(raw["schema_version"], "llama-r9700-e2e-v1")
                self.assertEqual(raw["target"]["exact"], self.summary["hardware"]["target"])
                self.assertEqual(raw["target"]["gpu_uuid"], self.summary["hardware"]["gpu_uuid"])
                self.assertEqual(raw["model"]["sha256"], self.summary["engines"][name]["model"]["sha256"])
                self.assertEqual(raw["llama"]["commit"], self.summary["engines"][name]["source"]["commit"])
                self.assertTrue(raw["audit"]["full_gpu_offload"])
                self.assertEqual(raw["protocol"]["n_ctx"], 10_003)
                self.assertEqual(raw["protocol"]["n_batch"], 10_001)
                self.assertEqual(raw["input_token_ids"], expected_input)

            samples = raw["measured"]["samples"]
            self.assertEqual(len(samples), 10, name)
            raw_values: dict[str, list[float]] = {metric: [] for metric in metric_names}
            for sample in samples:
                self.assertEqual(sample["tokens"]["input_token_ids"], expected_input, name)
                self.assertEqual(sample["tokens"]["generated_token_ids"], expected_output, name)
                self.assertEqual(sample["tokens"]["visible_token_ids"], expected_output, name)
                derived = sample["derived"]
                for metric in metric_names:
                    value = derived[metric]
                    raw_values[metric].append(value[0] if metric == "tpot_ns" else value)
            for metric in metric_names:
                self.assertEqual(raw_values[metric], self.summary["engines"][name]["metrics"][metric]["values"], f"{name}:{metric}")


if __name__ == "__main__":
    unittest.main()
