import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-generation-result.py"
FIXTURE = (
    REPO_ROOT
    / "benchmarks"
    / "results"
    / "2026-07-10"
    / "sq8-generation-v0.1"
    / "generation-run-01.json"
)


class Sq8GenerationResultValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE.is_file():
            raise RuntimeError(f"fixed generation fixture is absent: {FIXTURE}")
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        spec = importlib.util.spec_from_file_location("sq8_generation_validator", VALIDATOR)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load validator: {VALIDATOR}")
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def run_validator(
        self, path: Path, *, contract_only: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR)]
        if contract_only:
            command.append("--contract-only")
        command.append(str(path))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_modified(self, root: Path, mutate) -> Path:
        value = copy.deepcopy(self.fixture)
        mutate(value)
        path = root / "modified.json"
        path.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path

    def assert_contract_rejects(self, mutate, expected_error: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_modified(Path(temporary), mutate)
            result = self.run_validator(path, contract_only=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)

    def test_trusted_fixture_passes_promotion(self) -> None:
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed=true mode=promotion trusted=true", result.stdout)
        self.assertIn("steps=8 prompt_tokens=8 generated_tokens=8 final_kv_len=15", result.stdout)
        self.assertIn("projections=2240 activation_quantizations=1280", result.stdout)
        self.assertIn("first_token=353 last_token=18", result.stdout)

    def test_trusted_fixture_passes_contract_only_as_untrusted(self) -> None:
        result = self.run_validator(FIXTURE, contract_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=contract-only trusted=false", result.stdout)

    def test_self_consistent_timing_rerun_is_contract_only(self) -> None:
        def mutate(value) -> None:
            delta = 1_000_000
            value["steps"][1]["completed_at_ns"] += delta
            value["steps"][1]["latency_ns"] += delta
            for step in value["steps"][2:]:
                step["started_at_ns"] += delta
                step["completed_at_ns"] += delta
            timing = value["timing"]
            timing["request_latency_ns"] += delta
            timing["decode_elapsed_ns"] += delta
            request_ns = timing["request_latency_ns"]
            decode_ns = timing["decode_elapsed_ns"]
            timing["requests_per_second"] = 1e9 / request_ns
            timing["generated_tokens_per_second"] = 8e9 / request_ns
            timing["total_tokens_per_second"] = 16e9 / request_ns
            timing["decode_tokens_per_second"] = 7e9 / decode_ns

        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_modified(Path(temporary), mutate)
            promotion = self.run_validator(path)
            self.assertNotEqual(promotion.returncode, 0)
            self.assertIn("promotion trust anchor", promotion.stderr)
            contract = self.run_validator(path, contract_only=True)
            self.assertEqual(contract.returncode, 0, contract.stderr)
            self.assertIn("trusted=false", contract.stdout)

    def test_gate_threshold_boundaries_are_inclusive(self) -> None:
        metrics = {
            "nonfinite_count": 0,
            "relative_l2": 0.20,
            "cosine_similarity": 0.98,
        }
        self.assertTrue(self.module.gate_verdict(metrics, 0.20, 0.98))
        above = dict(metrics)
        above["relative_l2"] = math.nextafter(0.20, math.inf)
        self.assertFalse(self.module.gate_verdict(above, 0.20, 0.98))
        below = dict(metrics)
        below["cosine_similarity"] = math.nextafter(0.98, -math.inf)
        self.assertFalse(self.module.gate_verdict(below, 0.20, 0.98))

    def test_duplicate_json_key_is_rejected(self) -> None:
        raw = FIXTURE.read_text(encoding="utf-8")
        needle = '  "schema_version": "ullm.sq8.generation.v1",'
        replacement = needle + "\n" + needle
        self.assertIn(needle, raw)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.json"
            path.write_text(raw.replace(needle, replacement, 1), encoding="utf-8")
            result = self.run_validator(path, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key: schema_version", result.stderr)

    def test_json_nan_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][0].__setitem__("output_logit", math.nan),
            "non-finite JSON number is forbidden: NaN",
        )

    def test_json_infinity_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["timing"].__setitem__("requests_per_second", math.inf),
            "non-finite JSON number is forbidden: Infinity",
        )

    def test_extra_top_level_key_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value.__setitem__("untrusted_note", "extra"),
            "result keys differ",
        )

    def test_missing_nested_key_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][0]["final_hidden"].pop("metrics"),
            "steps[0].final_hidden keys differ",
        )

    def test_boolean_cannot_replace_integer(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"].__setitem__("request_id", True),
            "generation.request_id must be an integer",
        )

    def test_integer_cannot_replace_boolean(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"].__setitem__("fallback_used", 0),
            "execution.fallback_used must be boolean",
        )

    def test_source_hash_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["source"].__setitem__("artifact_content_sha256", "0" * 64),
            "source.artifact_content_sha256 differs",
        )

    def test_model_revision_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["source"].__setitem__("model_revision", "0" * 40),
            "source.model_revision differs",
        )

    def test_payload_hash_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["payloads"].__setitem__("lm_head_payload_sha256", "0" * 64),
            "payloads.lm_head_payload_sha256 differs",
        )

    def test_payload_record_extra_key_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["payloads"]["layer_norms"][0].__setitem__("trusted", True),
            "payloads.layer_norms[0] keys differ",
        )

    def test_generated_token_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"]["generated_token_ids"].__setitem__(1, 11),
            "generation token sequence differs",
        )

    def test_expected_token_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"]["expected_generated_token_ids"].__setitem__(7, 19),
            "generation token sequence differs",
        )

    def test_generated_token_hash_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"].__setitem__(
                "generated_token_ids_u32_le_sha256", "0" * 64
            ),
            "generation token SHA-256",
        )

    def test_decode_feedback_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"]["decode_input_token_ids"].__setitem__(0, 354),
            "do not prove token feedback",
        )

    def test_decode_position_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"]["decode_positions"].__setitem__(3, 99),
            "do not prove token feedback",
        )

    def test_final_kv_length_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"].__setitem__("final_kv_len", 16),
            "generation.final_kv_len is inconsistent",
        )

    def test_release_claim_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["generation"].__setitem__("allocation_released", False),
            "generation.allocation_released is inconsistent",
        )

    def test_step_phase_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][1].__setitem__("phase", "prefill"),
            "steps[1].phase is invalid",
        )

    def test_step_feedback_input_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][2].__setitem__("input_token_id", 999),
            "does not feed back the previous output",
        )

    def test_step_output_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][4].__setitem__("output_token_id", 16),
            "output token differs from the fixed generation sequence",
        )

    def test_relative_l2_above_gate_is_rejected(self) -> None:
        def mutate(value) -> None:
            gate = value["steps"][0]["logits"]
            gate["metrics"]["relative_l2"] = math.nextafter(0.20, math.inf)
            gate["passed"] = True

        self.assert_contract_rejects(mutate, "passed does not match its metrics")

    def test_cosine_below_gate_is_rejected(self) -> None:
        def mutate(value) -> None:
            gate = value["steps"][3]["final_hidden"]
            gate["metrics"]["cosine_similarity"] = math.nextafter(0.98, -math.inf)
            gate["passed"] = True

        self.assert_contract_rejects(mutate, "passed does not match its metrics")

    def test_vllm_oracle_tamper_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][0]["vllm_top_10"][9].__setitem__("logit", 8.80),
            "vLLM reference values differ from the fixed trusted oracle",
        )

    def test_vllm_hash_disagreement_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][2].__setitem__("vllm_logits_sha256", "0" * 64),
            "vLLM logits hashes disagree",
        )

    def test_top_1_claim_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][5].__setitem__("top_1_exact", False),
            "steps[5].top_1_exact is inconsistent",
        )

    def test_top_10_overlap_claim_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][6].__setitem__("top_10_overlap", 8),
            "steps[6].top_10_overlap is inconsistent",
        )

    def test_top_10_overlap_below_minimum_is_rejected(self) -> None:
        def mutate(value) -> None:
            step = value["steps"][0]
            oracle_ids = {entry["token_id"] for entry in step["vllm_top_10"]}
            replacements = iter(
                token_id
                for token_id in range(100_000, 100_100)
                if token_id not in oracle_ids
            )
            for entry in step["device_top_10"][1:]:
                entry["token_id"] = next(replacements)
            step["top_10_overlap"] = 1
            step["passed"] = False

        self.assert_contract_rejects(mutate, "misses its independently derived contract")

    def test_device_top_1_logit_health_mismatch_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][0]["device_logits_health"].__setitem__(
                "maximum", value["steps"][0]["device_logits_health"]["maximum"] + 0.01
            ),
            "device top-1 does not match logits maximum",
        )

    def test_counter_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["counters"].__setitem__("projection_calls", 2239),
            "execution.counters.projection_calls differs",
        )

    def test_stack_step_counter_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["stack_steps"][7].__setitem__(
                "paged_attention_calls", 39
            ),
            "execution.stack_steps[7].paged_attention_calls differs",
        )

    def test_all_layer_cache_length_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["final_cache_lengths"].__setitem__(39, 14),
            "must contain 15 for all 40 layers",
        )

    def test_hip_guard_substitution_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["required_hip_kernel_env"].__setitem__(
                0, "ULLM_REQUIRE_HIP_FAKE_KERNEL"
            ),
            "differs from the exact HIP guard set",
        )

    def test_fallback_use_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"].__setitem__("fallback_used", True),
            "execution.fallback_used differs",
        )

    def test_host_staging_use_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["execution"]["stack_steps"][1].__setitem__(
                "host_staging_used", True
            ),
            "execution.stack_steps[1].host_staging_used differs",
        )

    def test_step_latency_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["steps"][2].__setitem__(
                "latency_ns", value["steps"][2]["latency_ns"] + 1
            ),
            "steps[2].latency_ns does not match its endpoints",
        )

    def test_request_timing_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["timing"].__setitem__(
                "request_latency_ns", value["timing"]["request_latency_ns"] + 1
            ),
            "timing.request_latency_ns does not match step timing endpoints",
        )

    def test_rate_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["timing"].__setitem__(
                "generated_tokens_per_second",
                value["timing"]["generated_tokens_per_second"] + 0.01,
            ),
            "timing.generated_tokens_per_second does not match its nanosecond counters",
        )

    def test_allocator_before_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["allocator"]["before"].__setitem__("free_blocks", 0),
            "allocator.before.free_blocks differs",
        )

    def test_allocator_after_release_corruption_is_rejected(self) -> None:
        self.assert_contract_rejects(
            lambda value: value["allocator"]["after_release"].__setitem__(
                "allocated_blocks", 1
            ),
            "allocator.after_release.allocated_blocks differs",
        )


if __name__ == "__main__":
    unittest.main()
