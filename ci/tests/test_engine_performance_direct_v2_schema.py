from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ci.tests.test_engine_performance_schema import first_row, result_for  # noqa: E402


SCHEMA_PATH = ROOT / "ci/schema/engine-performance-direct-v2.schema.json"


def current_result(*, warmups: int = 3, measured: int = 10) -> dict[str, object]:
    """Adapt the compact historical fixture to the current direct output shape."""
    result = result_for(first_row())
    result["benchmark_schema_version"] = "engine-performance-direct-v2"
    config = result["config"]
    assert isinstance(config, dict)
    config.update(
        {
            "ignore_eos": False,
            "context_length": 34,
            "effective_context_length": 34,
            "prefill_chunk_tokens": None,
            "effective_prefill_chunk_tokens": 17,
            "completion_timeout_seconds": 3600,
            "lane": "direct",
            "kv_cache_encoding": "fp16",
            "warmups": warmups,
            "measured": measured,
        }
    )
    stop_policy = config["stop_policy"]
    assert isinstance(stop_policy, dict)
    stop_policy["ignore_eos"] = False

    memory = result["memory"]
    assert isinstance(memory, dict)
    memory.update(
        {
            "placement_total_memory_bytes": 32 * 1024**3,
            "placement_available_memory_bytes": 30 * 1024**3,
            "placement_required_bytes": 2 * 1024**3,
            "placement_model_resident_bytes": 1 * 1024**3,
            "placement_request_state_bytes": 512 * 1024**2,
            "placement_safety_reserve_bytes": 256 * 1024**2,
            "workspace_separate_allocation_bytes": 0,
            "workspace_arena_bytes": 128 * 1024**2,
        }
    )

    audit = result["audit"]
    assert isinstance(audit, dict)
    audit.update(
        {
            "correctness_control_request_count": 0,
            "correctness_control_source": "first-warmup-sample",
            "correctness_control_reference_sample_index": 0,
            "total_request_count": warmups + measured,
            "sample_count": warmups + measured,
        }
    )
    result["cleanup"] = {
        "correctness_control_request_count": 0,
        "correctness_control_source": "first-warmup-sample",
        "correctness_control_reference_sample_index": 0,
        "warmup_request_count": warmups,
        "measured_request_count": measured,
        "request_cleanup_count": warmups + measured,
        "performance_sample_count": warmups + measured,
        "all_requests_dropped": True,
        "retryable_cleanup": 0,
        "durable_quarantine": 0,
    }

    all_samples = result["warmups"]["samples"] + result["measured"]["samples"]  # type: ignore[index]
    assert isinstance(all_samples, list) and all_samples
    result["warmups"] = {"count": warmups, "samples": all_samples[:warmups]}
    result["measured"] = {"count": measured, "samples": all_samples[warmups : warmups + measured]}
    samples = result["warmups"]["samples"]  # type: ignore[index]
    assert isinstance(samples, list) and samples
    reference_sample = samples[0]
    result["correctness_control"] = {
        "label": "correctness-reference",
        "execution_path": "first-warmup-sample",
        "timing_instrumentation": "on",
        "included_in_performance_statistics": False,
        "source": {"kind": "warmup-sample", "sample_index": 0, "request_count": 0},
        "tokens": copy.deepcopy(reference_sample["tokens"]),
        "stop": copy.deepcopy(reference_sample["stop"]),
        "audit": copy.deepcopy(reference_sample["audit"]),
        "memory": copy.deepcopy(reference_sample["memory"]),
        "cleanup": {
            "reference_sample": True,
            "request_dropped": True,
            "allocator_cleanup_validated": True,
            "retryable_cleanup": 0,
            "durable_quarantine": 0,
        },
        "comparison": {
            "mode": "exact",
            "scope": "first_warmup_reference_against_every_remaining_warmup_and_measured_sample",
            "reference_source": "warmups.samples[0]",
            "token_fields": ["input_token_ids", "generated_token_ids", "visible_token_ids", "decode_input_token_ids"],
            "stop_fields": ["version", "reason_version", "kind", "token_id"],
            "dispatch_fields": ["selected_backend", "target", "device_index", "model_fingerprint", "plan_digest", "fallback_used", "all_dispatches_hip", "submission_count", "kernel_dispatch_count", "segment_count", "boundary_count"],
            "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
        },
    }
    return result


class DirectV2SchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def assert_valid(self, document: dict[str, object]) -> None:
        errors = list(self.validator.iter_errors(document))
        self.assertEqual(errors, [], "\n".join(error.message for error in errors))

    def test_current_standard_protocol_is_valid(self) -> None:
        self.assert_valid(current_result())

    def test_current_extended_protocol_is_valid(self) -> None:
        result = current_result(warmups=1, measured=3)
        result["config"]["ignore_eos"] = True  # type: ignore[index]
        result["config"]["stop_policy"] = {  # type: ignore[index]
            "stop_token_ids": [],
            "ignore_eos": True,
            "visible_stop_tokens": False,
        }
        self.assert_valid(result)

    def test_schema_allows_extended_phase49_token_bounds(self) -> None:
        for definition, value in (
            ("input_tokens", [23066] * 100_000),
            ("output_tokens", [23066] * 20_000),
            ("decode_input_tokens", [23066] * 19_999),
        ):
            target = {"$schema": self.schema["$schema"], "$ref": f"#/$defs/{definition}", "$defs": self.schema["$defs"]}
            Draft202012Validator(target).validate(value)

    def test_schema_allows_phase52_selector_and_kv_physical_metadata(self) -> None:
        result = current_result(warmups=1, measured=3)
        result["config"].update(  # type: ignore[union-attr]
            {
                "prefill_chunk_selection": "automatic",
                "prefill_chunk_candidates": [2048, 512],
                "prefill_chunk_rejections": [],
            }
        )
        sample = result["measured"]["samples"][0]  # type: ignore[index]
        sample["memory"]["kv"] = {
            "kv_layer_count": 1,
            "committed_kv_bytes": 536_870_912,
            "layers": [
                {
                    "layer": 3,
                    "logical_capacity_tokens": 131_072,
                    "observed_length_tokens": 100_001,
                    "memory_kind": "contiguous-resident",
                    "physical_page_bytes": 2_097_152,
                    "tokens_per_page": 1_024,
                    "mapped_token_capacity": 131_072,
                    "committed_bytes_per_plane": 268_435_456,
                }
            ],
        }
        self.assert_valid(result)

        sample["memory"]["kv"]["layers"][0]["memory_kind"] = "silent-fallback"
        with self.assertRaises(Exception):
            self.validator.validate(result)

    def test_v1_shape_and_separate_control_are_rejected(self) -> None:
        stale = current_result()
        stale["benchmark_schema_version"] = "engine-performance-direct-v1"
        with self.assertRaises(Exception):
            self.validator.validate(stale)

        stale = current_result()
        stale["correctness_control"]["label"] = "correctness-only"  # type: ignore[index]
        with self.assertRaises(Exception):
            self.validator.validate(stale)

    def test_wrong_protocol_counts_are_rejected(self) -> None:
        result = current_result()
        result["config"]["warmups"] = 1  # type: ignore[index]
        result["config"]["measured"] = 3  # type: ignore[index]
        with self.assertRaises(Exception):
            self.validator.validate(result)


if __name__ == "__main__":
    unittest.main()
