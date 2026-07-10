import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-deep-boundary.py"
GIT_COMMIT = "a" * 40
BINARY_SHA256 = "b" * 64


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_deep_boundary", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_document(module) -> dict:
    generated = [(index * 17 + 3) % module.VOCAB_SIZE for index in range(module.GENERATED_TOKENS)]
    generated[23] = module.EOS_TOKEN_IDS[0]
    generated[301] = module.EOS_TOKEN_IDS[1]
    steps = []
    for index, token_id in enumerate(generated):
        cache_len = module.PROMPT_TOKENS + index
        terminal = index == module.GENERATED_TOKENS - 1
        steps.append(
            {
                "generated_index": index,
                "token_id": token_id,
                "cache_len": cache_len,
                "cache_write_position": None if index == 0 else cache_len - 1,
                "status": "finishing" if terminal else "decoding",
                "cache_lengths": [cache_len] * module.STACK_LAYERS,
                "cache_lengths_all_expected": True,
                "scheduler_active": 1,
                "scheduler_waiting": 0,
                "allocated_blocks": module.CACHE_BLOCKS,
                "terminal_reason": "length" if terminal else None,
            }
        )
    prefill_units = []
    for index in range(module.PREFILL_EXECUTION_CALLS):
        start = index * module.PREFILL_CHUNK_TOKENS
        end = start + module.PREFILL_CHUNK_TOKENS
        prefill_units.append(
            {
                "start_position": start,
                "width": module.PREFILL_CHUNK_TOKENS,
                "end_position": end,
                "final_prompt_unit": index == module.PREFILL_EXECUTION_CALLS - 1,
                "cache_lengths": [end] * module.STACK_LAYERS,
                "cache_lengths_all_expected": True,
                "last_cache_position": end - 1,
                "last_logical_block": (end - 1) // module.BLOCK_TOKENS,
            }
        )
    return {
        "schema_version": module.INPUT_SCHEMA_VERSION,
        "runner_git_commit": GIT_COMMIT,
        "runner_worktree_clean": True,
        "runner_binary_sha256": BINARY_SHA256,
        "passed": False,
        "prefill_mode": module.PREFILL_MODE,
        "prefill_chunk_tokens": module.PREFILL_CHUNK_TOKENS,
        "prefill_implementation": module.PREFILL_IMPLEMENTATION,
        "artifact_content_sha256": module.EXPECTED_ARTIFACT_SHA256,
        "package_manifest_sha256": module.EXPECTED_PACKAGE_SHA256,
        "device": {
            "device_id": 0,
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "gcn_arch_name": "gfx1201",
            "compute_major": 12,
            "compute_minor": 0,
            "total_global_mem": 34_208_743_424,
        },
        "kv_cache_bytes": module.KV_CACHE_BYTES,
        "cache_blocks": module.CACHE_BLOCKS,
        "context_tokens": module.CONTEXT_TOKENS,
        "test_only_ignore_eos": True,
        "cancelled_request": None,
        "requests": [
            {
                "request_id": module.REQUEST_ID,
                "prompt_token_ids": list(range(1, module.PROMPT_TOKENS + 1)),
                "max_new_tokens": module.GENERATED_TOKENS,
                "generated_token_ids": generated,
                "test_only_ignore_eos": True,
                "generated_steps": steps,
                "prompt_progress_events": module.PROMPT_PROGRESS_EVENTS,
                "execution_units": module.TOTAL_EXECUTION_CALLS,
                "processed_prompt_tokens": module.PROMPT_TOKENS,
                "execution_calls": module.TOTAL_EXECUTION_CALLS,
                "prefill_execution_units": prefill_units,
                "reserved_context_tokens": module.CONTEXT_TOKENS,
                "terminal_sequence_tokens": module.CONTEXT_TOKENS,
                "terminal_status": "finishing",
                "terminal_expected_cache_len": module.TERMINAL_CACHE_LEN,
                "terminal_cache_lengths": [module.TERMINAL_CACHE_LEN]
                * module.STACK_LAYERS,
                "terminal_cache_lengths_all_expected": True,
                "terminal_last_cache_position": module.TERMINAL_CACHE_POSITION,
                "terminal_last_logical_block": module.TERMINAL_LOGICAL_BLOCK,
                "terminal_scheduler_active": 1,
                "terminal_scheduler_waiting": 0,
                "terminal_allocated_blocks": module.CACHE_BLOCKS,
                "terminal_reason": "length",
                "release_outcome": "length",
            }
        ],
        "post_reset_status": "ready",
        "post_reset_active": 0,
        "post_reset_waiting": 0,
        "post_reset_allocated_blocks": 0,
        "post_reset_cache_lengths": [0] * module.STACK_LAYERS,
        "post_reset_cache_lengths_all_zero": True,
    }


def write_document(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")


class Sq8ServingDeepBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def validate(self, path: Path):
        return self.module.validate_result(path, GIT_COMMIT, BINARY_SHA256)

    def assert_mutation_rejected(self, mutate, pattern: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-boundary.json"
            document = valid_document(self.module)
            mutate(document)
            write_document(path, document)
            with self.assertRaisesRegex(self.module.ValidationError, pattern):
                self.validate(path)

    def test_recomputes_boundary_and_accepts_eos_before_length(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-boundary.json"
            document = valid_document(self.module)
            write_document(path, document)
            result = self.validate(path)
            self.assertTrue(result["passed"])
            self.assertEqual(result["reserved_context_tokens"], self.module.CONTEXT_TOKENS)
            self.assertEqual(result["terminal_cache_len"], self.module.TERMINAL_CACHE_LEN)
            self.assertEqual(result["observed_eos_generated_indices"], [23, 301])
            self.assertEqual(
                result["evidence"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_rejects_prompt_token_tampering(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["prompt_token_ids"].__setitem__(17, 99),
            "ascending 3584-token prompt",
        )

    def test_rejects_generated_step_token_tampering(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["generated_steps"][211].__setitem__(
                "token_id", 42
            ),
            "generated/cache transition",
        )

    def test_rejects_one_layer_cache_drift(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["generated_steps"][400][
                "cache_lengths"
            ].__setitem__(19, 1),
            "generated/cache transition",
        )

    def test_rejects_generated_step_false_cache_summary(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["generated_steps"][77].__setitem__(
                "cache_lengths_all_expected", False
            ),
            "generated/cache transition",
        )

    def test_rejects_prefill_unit_transition_tampering(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["prefill_execution_units"][255].__setitem__(
                "start_position", 0
            ),
            "M8 prefill/cache transition",
        )

    def test_rejects_execution_count_tampering(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0].__setitem__("execution_calls", 958),
            "fixed deep-boundary execution contract",
        )

    def test_rejects_first_token_cache_write(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["generated_steps"][0].__setitem__(
                "cache_write_position", self.module.PROMPT_TOKENS - 1
            ),
            "generated/cache transition",
        )

    def test_rejects_early_length_reason(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["generated_steps"][23].__setitem__(
                "terminal_reason", "length"
            ),
            "generated/cache transition",
        )

    def test_rejects_terminal_cache_drift(self):
        self.assert_mutation_rejected(
            lambda value: value["requests"][0]["terminal_cache_lengths"].__setitem__(7, 4094),
            "terminal 4096-token boundary",
        )

    def test_rejects_reset_cache_drift(self):
        self.assert_mutation_rejected(
            lambda value: value["post_reset_cache_lengths"].__setitem__(3, 1),
            "post-reset baseline",
        )

    def test_rejects_dirty_or_mismatched_build_identity(self):
        self.assert_mutation_rejected(
            lambda value: value.__setitem__("runner_worktree_clean", False),
            "clean runner build identity",
        )
        self.assert_mutation_rejected(
            lambda value: value.__setitem__("runner_binary_sha256", "c" * 64),
            "clean runner build identity",
        )

    def test_rejects_symlink_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            link = root / "evidence.json"
            write_document(target, valid_document(self.module))
            os.symlink(target, link)
            with self.assertRaisesRegex(self.module.ValidationError, "regular file"):
                self.validate(link)

    def test_rejects_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(self.module.ValidationError, "duplicate JSON key"):
                self.validate(path)


if __name__ == "__main__":
    unittest.main()
