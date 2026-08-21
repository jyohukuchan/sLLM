import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "ci/schema/model-frontend-cli-report-v1.schema.json"


def base_report(command: str, result: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "model-frontend-cli-report-v1",
        "command": command,
        "state": "PASS",
        "model": {
            "repo_id": "Qwen/Qwen3.5-4B",
            "resolved_revision": "8" * 40,
            "lock_fingerprint": "sha256:" + "3" * 64,
        },
        "scope": {
            "offline": True,
            "gpu_execution": False,
            "model_execution": False,
            "generation": False,
        },
        "result": result,
    }


def generate_report() -> dict[str, object]:
    report = base_report(
        "generate",
        {
            "kind": "generate",
            "input_kind": "prompt",
            "input_token_ids": [9419],
            "generated_token_ids": [220, 220],
            "visible_token_ids": [220, 220],
            "decode_input_token_ids": [220],
            "output_text": "Hello",
            "finish_reason": "length",
            "stop_reason": {
                "version": 1,
                "reason_version": 1,
                "kind": "length",
                "token_id": None,
                "matched_string": None,
            },
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            "sampling": {
                "temperature": 0.0,
                "top_p": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
            },
            "execution": {
                "selected_backend": "hip",
                "target": "gfx1030",
                "device_index": 0,
                "model_fingerprint": "sha256:" + "3" * 64,
                "plan_digest": "sha256:" + "9" * 64,
                "prefill_tokens": 1,
                "logical_state_capacity_tokens": 3,
                "allocated_state_capacity_tokens": 3,
                "mtp_state_slack_tokens": 0,
                "decode_steps": 1,
                "fallback_used": False,
                "submission_count": 42,
                "kernel_dispatch_count": 43,
                "segment_count": 17,
                "boundary_count": 18,
                "all_dispatches_hip": True,
                "weight_encoding": "bf16",
                "fp8_provider": None,
                "prefill_chunk_requested_tokens": None,
                "prefill_chunk_selection": "auto",
                "prefill_chunk_capacity_tokens": 1,
                "prefill_chunk_count": 1,
                "placement_total_memory_bytes": 64 * 1024 * 1024 * 1024,
                "placement_available_memory_bytes": 60 * 1024 * 1024 * 1024,
                "placement_required_bytes": 8 * 1024 * 1024 * 1024,
                "placement_model_resident_bytes": 7 * 1024 * 1024 * 1024,
                "placement_request_state_bytes": 256 * 1024 * 1024,
                "placement_safety_reserve_bytes": 1024 * 1024 * 1024,
                "workspace_separate_allocation_bytes": 0,
                "workspace_arena_bytes": 64 * 1024 * 1024,
                "kv_cache_encoding": "fp16",
                "image_count": 0,
                "mtp_selection": "auto",
                "mtp_draft_width_requested": None,
                "mtp_draft_width_effective": None,
                "mtp_target_block_rows": None,
                "mtp_proposal_blocks": None,
                "mtp_proposed_draft_tokens": None,
                "mtp_accepted_draft_tokens": None,
                "mtp_rejected_draft_tokens": None,
                "mtp_weight_encoding": None,
                "mtp_kv_cache_encoding": None,
            },
            "timing_ns": 123,
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
        },
    )
    report["scope"] = {
        "offline": True,
        "gpu_execution": True,
        "model_execution": True,
        "generation": True,
    }
    return report


class ModelFrontendCliSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def assert_valid(self, report: dict[str, object]) -> None:
        self.assertEqual(list(self.validator.iter_errors(report)), [])

    def assert_invalid(self, report: dict[str, object]) -> None:
        self.assertNotEqual(list(self.validator.iter_errors(report)), [])

    def test_all_command_result_shapes_are_valid(self) -> None:
        self.assert_valid(base_report("tokenize", {"kind": "tokenize", "count": 3, "token_ids": [1, 3, 17]}))
        self.assert_valid(base_report("render", {"kind": "render", "text": "prompt"}))
        self.assert_valid(base_report("decode", {"kind": "decode", "text": "decoded"}))
        self.assert_valid(base_report("verify-model", {
            "kind": "verify-model", "locked_files": 13, "verified_files": 13,
            "tensor_count": 738, "weight_entries": 738, "loadable_entries": 426,
            "known_unconsumed_entries": 312, "total_destination_bytes": 8_000_000_003,
            "plan_digest": "sha256:" + "9" * 64,
        }))
        self.assert_valid(generate_report())

    def test_command_result_mismatch_and_partial_success_are_rejected(self) -> None:
        mismatch = base_report("tokenize", {"kind": "decode", "text": "x"})
        self.assert_invalid(mismatch)
        extra = base_report("decode", {"kind": "decode", "text": "x"})
        extra["partial"] = True
        self.assert_invalid(extra)
        gpu = copy.deepcopy(base_report("render", {"kind": "render", "text": "x"}))
        gpu["scope"]["gpu_execution"] = True
        self.assert_invalid(gpu)

    def test_generate_audit_backend_fallback_scope_and_closed_fields_are_rejected(self) -> None:
        for field, value in [
            ("submission_count", 0),
            ("kernel_dispatch_count", 0),
            ("segment_count", 0),
            ("boundary_count", 0),
            ("all_dispatches_hip", False),
        ]:
            mutated = copy.deepcopy(generate_report())
            mutated["result"]["execution"][field] = value
            self.assert_invalid(mutated)

        wrong_backend = copy.deepcopy(generate_report())
        wrong_backend["result"]["execution"]["selected_backend"] = "cpu"
        self.assert_invalid(wrong_backend)
        fallback = copy.deepcopy(generate_report())
        fallback["result"]["execution"]["fallback_used"] = True
        self.assert_invalid(fallback)
        wrong_scope = copy.deepcopy(generate_report())
        wrong_scope["scope"]["generation"] = False
        self.assert_invalid(wrong_scope)

        closed_execution = copy.deepcopy(generate_report())
        closed_execution["result"]["execution"]["unexpected"] = True
        self.assert_invalid(closed_execution)
        closed_stop = copy.deepcopy(generate_report())
        closed_stop["result"]["stop_reason"]["unexpected"] = True
        self.assert_invalid(closed_stop)
        closed_result = copy.deepcopy(generate_report())
        closed_result["result"]["unexpected"] = True
        self.assert_invalid(closed_result)

    def test_current_backend_specific_report_shapes_are_valid(self) -> None:
        gemma_verify = base_report("verify-model", {
            "kind": "verify-model",
            "model_kind": "gemma4-dense",
            "prompt_mode": "raw-text-only",
            "chat_template": False,
            "locked_files": 3,
            "verified_files": 1,
            "tensor_count": 17,
            "weight_entries": 17,
            "loadable_entries": 17,
            "known_unconsumed_entries": 0,
            "total_destination_bytes": 17,
            "plan_digest": "sha256:" + "9" * 64,
            "weight_encoding": "mixed-nvfp4-w4a4-fp8-w8a8",
            "recipe_digest": "sha256:" + "4" * 64,
        })
        moe_verify = base_report("verify-model", {
            "kind": "verify-model",
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "tensor_count": 17,
            "source_kind": "gguf",
            "weight_entries": 17,
            "total_destination_bytes": 17,
            "plan_digest": "sha256:" + "9" * 64,
            "weight_encoding": "ocp-mxfp4-e2m1-block32-e8m0-mixed",
        })
        self.assert_valid(gemma_verify)
        self.assert_valid(moe_verify)

        moe_generate = generate_report()
        moe_generate["result"]["stop_reason"] = None
        moe_generate["result"].pop("stop_reason")
        moe_generate["result"].pop("sampling")
        moe_generate["result"].pop("cleanup")
        execution = moe_generate["result"]["execution"]
        for field in (
            "prefill_tokens", "logical_state_capacity_tokens", "allocated_state_capacity_tokens",
            "mtp_state_slack_tokens", "decode_steps", "segment_count", "boundary_count",
            "prefill_chunk_requested_tokens", "prefill_chunk_selection", "prefill_chunk_capacity_tokens",
            "prefill_chunk_count", "placement_total_memory_bytes", "placement_available_memory_bytes",
            "placement_required_bytes", "placement_model_resident_bytes", "placement_request_state_bytes",
            "placement_safety_reserve_bytes", "workspace_separate_allocation_bytes", "workspace_arena_bytes",
            "kv_cache_encoding", "image_count", "mtp_selection", "mtp_draft_width_requested",
            "mtp_draft_width_effective", "mtp_target_block_rows", "mtp_proposal_blocks",
            "mtp_proposed_draft_tokens", "mtp_accepted_draft_tokens", "mtp_rejected_draft_tokens",
            "mtp_weight_encoding", "mtp_kv_cache_encoding",
        ):
            execution.pop(field)
        execution["weight_encoding"] = "ocp-mxfp4-e2m1-block32-e8m0-mixed"
        self.assert_valid(moe_generate)

    def test_prefill_and_mtp_conditional_shapes_are_enforced(self) -> None:
        gfx942 = copy.deepcopy(generate_report())
        gfx942["result"]["execution"]["target"] = "gfx942"
        self.assert_valid(gfx942)

        missing_qwen_audit = copy.deepcopy(generate_report())
        missing_qwen_audit["result"]["execution"].pop("kv_cache_encoding")
        self.assert_invalid(missing_qwen_audit)

        explicit_chunk = copy.deepcopy(generate_report())
        explicit_chunk["result"]["execution"]["prefill_chunk_requested_tokens"] = 512
        explicit_chunk["result"]["execution"]["prefill_chunk_selection"] = "explicit"
        self.assert_valid(explicit_chunk)
        explicit_chunk["result"]["execution"]["prefill_chunk_selection"] = "auto"
        self.assert_invalid(explicit_chunk)

        forced_mtp = copy.deepcopy(generate_report())
        forced = forced_mtp["result"]["execution"]
        forced.update({
            "mtp_selection": "forced",
            "mtp_draft_width_requested": 2,
            "mtp_draft_width_effective": 2,
            "mtp_target_block_rows": 3,
            "mtp_proposal_blocks": 0,
            "mtp_proposed_draft_tokens": 0,
            "mtp_accepted_draft_tokens": 0,
            "mtp_rejected_draft_tokens": 0,
            "mtp_weight_encoding": "bf16",
            "mtp_kv_cache_encoding": "fp16",
        })
        self.assert_valid(forced_mtp)
        forced["mtp_weight_encoding"] = None
        self.assert_invalid(forced_mtp)

        target_only = copy.deepcopy(generate_report())
        target_only["result"]["execution"].update({
            "mtp_selection": "target-only",
            "mtp_draft_width_requested": 0,
        })
        self.assert_valid(target_only)
        target_only["result"]["execution"]["mtp_draft_width_requested"] = None
        self.assert_invalid(target_only)


if __name__ == "__main__":
    unittest.main()
