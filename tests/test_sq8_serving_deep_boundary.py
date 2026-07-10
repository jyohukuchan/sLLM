import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-deep-boundary.py"
GIT_COMMIT = "a" * 40
BINARY_SHA256 = "b" * 64
M128_PREFILL_MODE = "m128-chunk128"
M128_INPUT_SCHEMA_VERSION = "ullm.sq8.serving_deep_boundary.v2"
M128_PREFILL_CHUNK_TOKENS = 128
M128_PREFILL_IMPLEMENTATION = "sq8.fixed-m128-cached-prefix.v1"


def load_module():
    spec = importlib.util.spec_from_file_location("sq8_serving_deep_boundary", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prefill_config(module, prefill_mode: str):
    if prefill_mode == module.PREFILL_MODE:
        return {
            "schema_version": module.INPUT_SCHEMA_VERSION,
            "prefill_chunk_tokens": module.PREFILL_CHUNK_TOKENS,
            "prefill_implementation": module.PREFILL_IMPLEMENTATION,
        }
    if prefill_mode == M128_PREFILL_MODE:
        return {
            "schema_version": M128_INPUT_SCHEMA_VERSION,
            "prefill_chunk_tokens": M128_PREFILL_CHUNK_TOKENS,
            "prefill_implementation": M128_PREFILL_IMPLEMENTATION,
        }
    raise AssertionError(f"unsupported test prefill mode: {prefill_mode}")


def prefill_widths(prompt_tokens: int, chunk_tokens: int) -> list[int]:
    return [chunk_tokens] * (prompt_tokens // chunk_tokens) + [1] * (
        prompt_tokens % chunk_tokens
    )


def valid_document(module, *, prefill_mode=None) -> dict:
    prefill_mode = prefill_mode or module.PREFILL_MODE
    config = prefill_config(module, prefill_mode)
    chunk_tokens = config["prefill_chunk_tokens"]
    widths = prefill_widths(module.PROMPT_TOKENS, chunk_tokens)
    prefill_calls = len(widths)
    total_calls = prefill_calls + module.DECODE_EXECUTION_CALLS
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
    position = 0
    for index, width in enumerate(widths):
        start = position
        end = start + width
        prefill_units.append(
            {
                "start_position": start,
                "width": width,
                "end_position": end,
                "final_prompt_unit": index == prefill_calls - 1,
                "cache_lengths": [end] * module.STACK_LAYERS,
                "cache_lengths_all_expected": True,
                "last_cache_position": end - 1,
                "last_logical_block": (end - 1) // module.BLOCK_TOKENS,
            }
        )
        position = end
    return {
        "schema_version": config["schema_version"],
        "runner_git_commit": GIT_COMMIT,
        "runner_worktree_clean": True,
        "runner_binary_sha256": BINARY_SHA256,
        "passed": False,
        "prefill_mode": prefill_mode,
        "prefill_chunk_tokens": chunk_tokens,
        "prefill_implementation": config["prefill_implementation"],
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
                "prompt_progress_events": prefill_calls - 1,
                "execution_units": total_calls,
                "processed_prompt_tokens": module.PROMPT_TOKENS,
                "execution_calls": total_calls,
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

    def validate(self, path: Path, prefill_mode=None):
        if prefill_mode is None:
            return self.module.validate_result(path, GIT_COMMIT, BINARY_SHA256)
        return self.module.validate_result(path, GIT_COMMIT, BINARY_SHA256, prefill_mode)

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

    def test_recomputes_m128_raw_v2_boundary_and_full_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-boundary-m128.json"
            document = valid_document(self.module, prefill_mode=M128_PREFILL_MODE)
            write_document(path, document)

            result = self.validate(path, M128_PREFILL_MODE)

            self.assertTrue(result["passed"])
            self.assertEqual(result["prefill_mode"], M128_PREFILL_MODE)
            self.assertEqual(result["prefill_chunk_tokens"], M128_PREFILL_CHUNK_TOKENS)
            self.assertEqual(result["prefill_execution_calls"], 28)
            self.assertEqual(result["decode_execution_calls"], 511)
            self.assertEqual(result["total_execution_calls"], 539)
            request = document["requests"][0]
            self.assertEqual(request["prompt_progress_events"], 27)
            self.assertEqual(len(request["prefill_execution_units"]), 28)
            self.assertTrue(
                all(
                    unit["width"] == M128_PREFILL_CHUNK_TOKENS
                    for unit in request["prefill_execution_units"]
                )
            )
            self.assertEqual(request["prefill_execution_units"][-1]["end_position"], 3584)
            self.assertEqual(len(request["generated_steps"]), 512)
            self.assertEqual(result["terminal_cache_len"], 4095)
            self.assertEqual(result["terminal_last_cache_position"], 4094)

    def test_prefill_mode_strictly_binds_deep_schema_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            m8_path = root / "deep-m8.json"
            m128_path = root / "deep-m128.json"
            write_document(m8_path, valid_document(self.module))
            write_document(
                m128_path,
                valid_document(self.module, prefill_mode=M128_PREFILL_MODE),
            )

            with self.assertRaisesRegex(self.module.ValidationError, "schema/model/runtime contract"):
                self.validate(m128_path)
            with self.assertRaisesRegex(self.module.ValidationError, "schema/model/runtime contract"):
                self.validate(m8_path, M128_PREFILL_MODE)

    def test_rejects_m128_prefill_call_and_unit_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-m128.json"
            document = valid_document(self.module, prefill_mode=M128_PREFILL_MODE)
            document["requests"][0]["prompt_progress_events"] = 447
            write_document(path, document)
            with self.assertRaisesRegex(
                self.module.ValidationError, "fixed deep-boundary execution contract"
            ):
                self.validate(path, M128_PREFILL_MODE)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-m128.json"
            document = valid_document(self.module, prefill_mode=M128_PREFILL_MODE)
            document["requests"][0]["prefill_execution_units"][7]["width"] = 8
            write_document(path, document)
            with self.assertRaisesRegex(self.module.ValidationError, "M128 prefill/cache transition"):
                self.validate(path, M128_PREFILL_MODE)

    def test_cli_selects_m128_raw_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep-m128.json"
            write_document(path, valid_document(self.module, prefill_mode=M128_PREFILL_MODE))
            run = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    str(path),
                    "--expected-runner-git-commit",
                    GIT_COMMIT,
                    "--expected-binary-sha256",
                    BINARY_SHA256,
                    "--prefill-mode",
                    M128_PREFILL_MODE,
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("passed=true", run.stdout)

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
