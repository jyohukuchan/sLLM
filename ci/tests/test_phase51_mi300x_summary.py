import copy
import json
import importlib.util
import unittest
from pathlib import Path

import jsonschema


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/phase51-mi300x-summary-v1.schema.json"
AGGREGATE_PATH = ROOT / "ci/tools/aggregate_phase51_mi300x.py"
spec = importlib.util.spec_from_file_location("aggregate_phase51_mi300x", AGGREGATE_PATH)
assert spec is not None and spec.loader is not None
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


CASES = (
    ("short-odd", 17, 17, 34, 3, 10, False),
    ("32-32", 32, 32, 64, 3, 10, False),
    ("prefill-long", 1024, 128, 1152, 3, 10, False),
    ("decode-long", 32, 256, 288, 3, 10, False),
    ("long-10001", 10001, 2, 10003, 3, 10, False),
    ("long-100000", 100000, 2, 131072, 1, 3, False),
    ("decode-20000", 32, 20000, 131072, 1, 3, True),
)


def _stats(count: int) -> dict[str, float | int]:
    return {"median": 10.0, "mad": 1.0, "count": count, "min": 9.0, "max": 11.0}


def _summary_fixture() -> dict[str, object]:
    rows = []
    for case_id, input_count, output_count, context_length, warmups, measured, ignore_eos in CASES:
        metric = {name: _stats(measured) for name in ("e2e_ns", "ttft_ns", "tpot_ns")}
        gate = {
            "sllm_median": 10.0,
            "sllm_mad": 1.0,
            "llama_median": 10.0,
            "llama_mad": 1.0,
            "limit": 11.0,
            "pass": True,
        }
        rows.append(
            {
                "case_id": case_id,
                "input_token_count": input_count,
                "requested_output_tokens": output_count,
                "protocol": {
                    "warmups": warmups,
                    "measured": measured,
                    "context_length": context_length,
                    "ignore_eos": ignore_eos,
                },
                "row_ids": {
                    "sllm": f"phase51-mi300x-sllm-{case_id}",
                    "llama": f"phase51-mi300x-llama-{case_id}",
                },
                "measured_sample_count": {"sllm": measured, "llama": measured},
                "tokens": {
                    "input_sha256": "a" * 64,
                    "generated_sha256": {"sllm": "b" * 64, "llama": "b" * 64},
                    "visible_sha256": {"sllm": "c" * 64, "llama": "c" * 64},
                    "stop_sha256": {"sllm": "d" * 64, "llama": "d" * 64},
                    "generated_equal": True,
                    "visible_equal": True,
                    "stop_equal": True,
                },
                "metrics": {"sllm": metric, "llama": metric},
                "gates": {"e2e_ns": gate, "ttft_ns": gate, "tpot_ns": gate},
            }
        )
    return {
        "schema_version": "phase51-mi300x-summary-v1",
        "state": "PASS",
        "target": "gfx942",
        "gpu_uuid": "GPU-6104e2a75685060a",
        "amd_smi_uuid": "61ff74b5-0000-1000-8004-e2a75685060a",
        "gpu_bdf": "0000:ff:00.0",
        "actual_arch": "gfx942:sramecc+:xnack-",
        "wavefront_size": 64,
        "rocm_root": "/opt/rocm",
        "rocm_source_root": "/opt/rocm-7.2.4/core-7.14",
        "rocm_version": "7.14.0",
        "inputs": {
            "sllm": {"path": "/evidence/sllm.json", "sha256": "e" * 64},
            "llama": {"path": "/evidence/llama.json", "sha256": "f" * 64},
        },
        "identities": {
            "sllm": {
                "schema_version": "phase51-mi300x-sllm-v1",
                "engine": "sllm",
                "backend": "hip",
                "target": "gfx942",
                "gpu_uuid": "GPU-6104e2a75685060a",
                "amd_smi_uuid": "61ff74b5-0000-1000-8004-e2a75685060a",
                "gpu_bdf": "0000:ff:00.0",
                "actual_arch": "gfx942:sramecc+:xnack-",
                "wavefront_size": 64,
                "rocm_root": "/opt/rocm",
                "rocm_source_root": "/opt/rocm-7.2.4/core-7.14",
                "rocm_version": "7.14.0",
            },
            "llama": {
                "schema_version": "phase51-mi300x-llama-v1",
                "engine": "llama.cpp",
                "commit": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70",
                "tag": "b10453",
                "target": "gfx942",
                "gpu_uuid": "GPU-6104e2a75685060a",
                "amd_smi_uuid": "61ff74b5-0000-1000-8004-e2a75685060a",
                "gpu_bdf": "0000:ff:00.0",
                "actual_arch": "gfx942:sramecc+:xnack-",
                "wavefront_size": 64,
                "rocm_root": "/opt/rocm",
                "rocm_source_root": "/opt/rocm-7.2.4/core-7.14",
                "rocm_version": "7.14.0",
            },
        },
        "matrix": {"cases": [case[0] for case in CASES], "row_count": 7},
        "rows": rows,
        "gate": {
            "formula": "sLLM median <= llama.cpp median + max(sLLM MAD, llama.cpp MAD)",
            "e2e": True,
            "ttft": True,
            "tpot": True,
            "all_pass": True,
        },
    }


def _phase50_fixture_producer(engine: str) -> dict[str, object]:
    """Reuse the bounded host fixture shape while rebinding it to Phase 51."""
    fixture_path = ROOT / "ci/tests/test_phase50_r9700_summary.py"
    fixture_spec = importlib.util.spec_from_file_location("phase50_summary_fixture", fixture_path)
    assert fixture_spec is not None and fixture_spec.loader is not None
    fixture_module = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture_module)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            for old, new in (
                ("phase50-r9700", "phase51-mi300x"),
                ("gfx1201", "gfx942"),
                ("GPU-a8e9ddefa2d60f55", "GPU-6104e2a75685060a"),
                ("0000:07:00.0", "0000:ff:00.0"),
            ):
                value = value.replace(old, new)
            return value
        return value

    producer = rewrite(fixture_module._producer(engine))
    producer.update(
        {
            "actual_arch": "gfx942:sramecc+:xnack-",
            "wavefront_size": 64,
            "rocm_root": "/opt/rocm",
            "rocm_source_root": "/opt/rocm-7.2.4/core-7.14",
            "rocm_version": "7.14.0",
            "gpu_uuid": "GPU-6104e2a75685060a",
            "amd_smi_uuid": "61ff74b5-0000-1000-8004-e2a75685060a",
        }
    )
    return producer


class Phase51MI300XSummaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_schema_is_well_formed(self):
        jsonschema.Draft202012Validator.check_schema(self.schema)

    def test_frozen_tuple_identity_and_seven_row_protocol_validate(self):
        document = _summary_fixture()
        jsonschema.Draft202012Validator(self.schema).validate(document)
        self.assertEqual(document["matrix"]["row_count"], 7)
        self.assertEqual(document["identities"]["sllm"]["wavefront_size"], 64)
        self.assertEqual(document["rocm_root"], "/opt/rocm")
        self.assertEqual(document["rocm_source_root"], "/opt/rocm-7.2.4/core-7.14")

    def test_schema_rejects_identity_protocol_digest_and_gate_drift(self):
        for path, value in (
            (("gpu_uuid",), "61ff74b5-0000-1000-8004-e2a75685060a"),
            (("rocm_source_root",), "/opt/rocm/core-7.14"),
            (("rows", 0, "protocol", "measured"), 3),
            (("rows", 0, "tokens", "input_sha256"), "not-a-digest"),
        ):
            invalid = copy.deepcopy(_summary_fixture())
            cursor = invalid
            for key in path[:-1]:
                cursor = cursor[key]
            cursor[path[-1]] = value
            with self.subTest(path=path):
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.Draft202012Validator(self.schema).validate(invalid)

    def test_performance_parity_is_reported_but_not_a_hard_state_gate(self):
        document = _summary_fixture()
        document["gate"].update({"e2e": False, "all_pass": False})
        document["rows"][0]["gates"]["e2e_ns"]["pass"] = False
        jsonschema.Draft202012Validator(self.schema).validate(document)
        self.assertEqual(document["state"], "PASS")
        self.assertFalse(document["gate"]["all_pass"])

    def test_aggregate_header_is_fail_closed_for_runtime_uuid(self):
        summary = _summary_fixture()
        summary["schema_version"] = aggregate.SLLM_SCHEMA
        summary["gpu_uuid"] = "61ff74b5-0000-1000-8004-e2a75685060a"
        with self.assertRaises(aggregate.Phase51Error):
            aggregate._validate_summary_header(summary, "sllm")

    def test_aggregate_validates_producer_identities_performance_and_cleanup(self):
        document = aggregate.aggregate_summaries(
            _phase50_fixture_producer("sllm"), _phase50_fixture_producer("llama")
        )
        self.assertEqual(document["state"], "PASS")
        self.assertEqual(len(document["rows"]), 7)
        jsonschema.Draft202012Validator(self.schema).validate(document)
        self.assertNotIn("input_token_ids", json.dumps(document))
        self.assertNotIn("generated_token_ids", json.dumps(document))

    def test_aggregate_fails_closed_on_model_or_cleanup_drift(self):
        sllm = _phase50_fixture_producer("sllm")
        sllm["rows"][0]["result"]["identities"]["model"]["resolved_revision"] = "stale"
        with self.assertRaises(aggregate.Phase51Error):
            aggregate.aggregate_summaries(sllm, _phase50_fixture_producer("llama"))

        sllm = _phase50_fixture_producer("sllm")
        sllm["rows"][0]["result"]["cleanup"]["all_requests_dropped"] = False
        with self.assertRaises(aggregate.Phase51Error):
            aggregate.aggregate_summaries(sllm, _phase50_fixture_producer("llama"))


if __name__ == "__main__":
    unittest.main()
