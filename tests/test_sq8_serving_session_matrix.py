import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-session-matrix.py"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_session_matrix", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def completed_request(
    module, prompt_tokens: int, *, max_new_tokens: int = 1, generated_tokens=None
) -> dict:
    if generated_tokens is None:
        generated_tokens = [prompt_tokens % module.VOCAB_SIZE]
    cache_length = prompt_tokens + len(generated_tokens) - 1
    return {
        "request_id": f"serving-smoke-p{prompt_tokens:04}",
        "prompt_token_ids": list(range(1, prompt_tokens + 1)),
        "max_new_tokens": max_new_tokens,
        "generated_token_ids": list(generated_tokens),
        "prompt_progress_events": prompt_tokens - 1,
        "execution_units": cache_length,
        "reserved_context_tokens": prompt_tokens + max_new_tokens,
        "terminal_sequence_tokens": prompt_tokens + len(generated_tokens),
        "terminal_status": "finishing",
        "terminal_expected_cache_len": cache_length,
        "terminal_cache_lengths": [cache_length] * module.STACK_LAYERS,
        "terminal_cache_lengths_all_expected": True,
        "terminal_last_cache_position": cache_length - 1,
        "terminal_last_logical_block": (cache_length - 1) // module.BLOCK_TOKENS,
        "terminal_scheduler_active": 1,
        "terminal_scheduler_waiting": 0,
        "terminal_allocated_blocks": module.CACHE_BLOCKS,
        "terminal_reason": "length",
        "release_outcome": "length",
        "request_seconds": 1.0,
        "reset_seconds": 0.003,
        "oracle_capture": None,
    }


def common_result(module, requests: list[dict], cancelled_request) -> dict:
    return {
        "schema_version": module.INPUT_SCHEMA_VERSION,
        "passed": False,
        "requests": requests,
        "cancelled_request": cancelled_request,
        "load_seconds": 20.0,
        "artifact_content_sha256": module.EXPECTED_ARTIFACT_SHA256,
        "package_manifest_sha256": module.EXPECTED_PACKAGE_SHA256,
        "device": {
            "device_id": 0,
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "gcn_arch_name": "gfx1201",
            "compute_major": 12,
            "compute_minor": 0,
            "total_global_mem": 32 * 1024**3,
        },
        "kv_cache_bytes": module.KV_CACHE_BYTES,
        "cache_blocks": module.CACHE_BLOCKS,
        "context_tokens": module.CONTEXT_TOKENS,
        "post_reset_status": "ready",
        "post_reset_active": 0,
        "post_reset_waiting": 0,
        "post_reset_allocated_blocks": 0,
        "post_reset_cache_lengths": [0] * module.STACK_LAYERS,
        "post_reset_cache_lengths_all_zero": True,
    }


def prefill_cancellation(module) -> dict:
    return {
        "request_id": "serving-smoke-prefill-cancel",
        "cancellation_phase": "prefill",
        "prompt_tokens": module.BOUNDARY_PROMPT_LENGTHS[0],
        "prompt_progress_before_cancel": module.PREFILL_CANCEL_PROGRESS,
        "generated_before_cancel": [],
        "execution_units_before_cancel": module.PREFILL_CANCEL_PROGRESS,
        "status_before_cancel": "prefilling",
        "cache_lengths_before_cancel": [module.PREFILL_CANCEL_PROGRESS]
        * module.STACK_LAYERS,
        "scheduler_active_before_cancel": 1,
        "scheduler_waiting_before_cancel": 0,
        "allocated_blocks_before_cancel": module.CACHE_BLOCKS,
        "status_after_observation": "cancelling",
        "prompt_progress_after_observation": module.PREFILL_CANCEL_PROGRESS,
        "generated_tokens_after_observation": 0,
        "cache_lengths_after_observation": [module.PREFILL_CANCEL_PROGRESS]
        * module.STACK_LAYERS,
        "scheduler_active_after_observation": 1,
        "scheduler_waiting_after_observation": 0,
        "allocated_blocks_after_observation": module.CACHE_BLOCKS,
        "release_outcome": "cancelled",
        "reset_seconds": 0.003,
    }


def decode_cancellation(module) -> dict:
    return {
        "request_id": "serving-smoke-cancel",
        "cancellation_phase": "decode",
        "prompt_tokens": module.DECODE_PROMPT_LENGTH,
        "prompt_progress_before_cancel": module.DECODE_PROMPT_LENGTH - 1,
        "generated_before_cancel": [module.EXPECTED_DECODE_G8[0]],
        "execution_units_before_cancel": module.DECODE_PROMPT_LENGTH,
        "status_before_cancel": "decoding",
        "cache_lengths_before_cancel": [module.DECODE_PROMPT_LENGTH]
        * module.STACK_LAYERS,
        "scheduler_active_before_cancel": 1,
        "scheduler_waiting_before_cancel": 0,
        "allocated_blocks_before_cancel": module.CACHE_BLOCKS,
        "status_after_observation": "cancelling",
        "prompt_progress_after_observation": module.DECODE_PROMPT_LENGTH,
        "generated_tokens_after_observation": 1,
        "cache_lengths_after_observation": [module.DECODE_PROMPT_LENGTH]
        * module.STACK_LAYERS,
        "scheduler_active_after_observation": 1,
        "scheduler_waiting_after_observation": 0,
        "allocated_blocks_after_observation": module.CACHE_BLOCKS,
        "release_outcome": "cancelled",
        "reset_seconds": 0.003,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class Sq8ServingSessionMatrixTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def evidence(self, root: Path):
        boundary = common_result(
            self.module,
            [
                completed_request(self.module, length)
                for length in self.module.BOUNDARY_PROMPT_LENGTHS
            ],
            prefill_cancellation(self.module),
        )
        context = common_result(
            self.module,
            [completed_request(self.module, self.module.CONTEXT_PROMPT_LENGTH)],
            None,
        )
        decode = common_result(
            self.module,
            [
                completed_request(
                    self.module,
                    self.module.DECODE_PROMPT_LENGTH,
                    max_new_tokens=len(self.module.EXPECTED_DECODE_G8),
                    generated_tokens=self.module.EXPECTED_DECODE_G8,
                )
            ],
            decode_cancellation(self.module),
        )
        boundary_path = root / "boundary.json"
        context_path = root / "context.json"
        decode_path = root / "decode.json"
        write_json(boundary_path, boundary)
        write_json(context_path, context)
        write_json(decode_path, decode)
        return boundary_path, context_path, decode_path, boundary, context, decode

    def test_validate_results_recomputes_matrix_and_ignores_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path, context_path, decode_path, _, _, _ = self.evidence(root)
            result = self.module.validate_results(boundary_path, context_path, decode_path)
            self.assertTrue(result["passed"])
            self.assertEqual(
                [item["prompt_tokens"] for item in result["boundary"]["requests"]],
                list(self.module.BOUNDARY_PROMPT_LENGTHS),
            )
            self.assertEqual(
                result["exact_context"]["terminal_sequence_tokens"],
                self.module.CONTEXT_TOKENS,
            )
            self.assertEqual(
                result["evidence"][0]["sha256"],
                hashlib.sha256(boundary_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["decode_cancel"]["cancellation"]["phase"], "decode"
            )

    def test_load_json_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "duplicate JSON key"):
                self.module.load_json(path, "duplicate")

    def test_validate_results_rejects_one_layer_cache_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path, context_path, decode_path, boundary, _, _ = self.evidence(root)
            boundary["requests"][2]["terminal_cache_lengths"][17] += 1
            write_json(boundary_path, boundary)
            with self.assertRaisesRegex(self.module.ValidationError, "terminal/cache contract"):
                self.module.validate_results(boundary_path, context_path, decode_path)

    def test_validate_results_rejects_context_that_does_not_reach_4096(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path, context_path, decode_path, _, context, _ = self.evidence(root)
            context["requests"][0]["reserved_context_tokens"] = 4_095
            write_json(context_path, context)
            with self.assertRaisesRegex(self.module.ValidationError, "terminal/cache contract"):
                self.module.validate_results(boundary_path, context_path, decode_path)

    def test_validate_results_rejects_token_published_before_prefill_cancel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path, context_path, decode_path, boundary, _, _ = self.evidence(root)
            boundary["cancelled_request"]["generated_before_cancel"] = [1]
            write_json(boundary_path, boundary)
            with self.assertRaisesRegex(self.module.ValidationError, "cancellation/cache contract"):
                self.module.validate_results(boundary_path, context_path, decode_path)

    def test_validate_results_rejects_decode_cancel_observation_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundary_path, context_path, decode_path, _, _, decode = self.evidence(root)
            decode["cancelled_request"]["cache_lengths_after_observation"][0] += 1
            write_json(decode_path, decode)
            with self.assertRaisesRegex(self.module.ValidationError, "cancellation/cache contract"):
                self.module.validate_results(boundary_path, context_path, decode_path)

    def test_write_json_create_new_refuses_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.json"
            self.module.write_json_create_new(path, {"passed": True})
            with self.assertRaisesRegex(self.module.ValidationError, "failed to create"):
                self.module.write_json_create_new(path, {"passed": False})


if __name__ == "__main__":
    unittest.main()
