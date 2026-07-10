import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = REPO_ROOT / "tools" / "export-sq8-serving-vllm-oracles.py"
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-vllm-oracles.py"
SERVING_FIXTURE_VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-fixtures.py"
INPUT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sq8-serving-v0.1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32_file(path: Path, values) -> None:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        payload.tofile(handle)


def u32_file(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for value in values:
            handle.write(struct.pack("<I", value))


def file_record(module, path: Path, relative: str, elements: int) -> dict:
    return {
        "file": relative,
        "dtype": "f32_le",
        "source_dtype": "torch.bfloat16",
        "shape": [elements],
        "bytes": elements * 4,
        "sha256": module.sha256_file(path),
    }


def build_synthetic_oracle(root: Path, exporter, validator) -> None:
    root.mkdir()
    exporter_bytes = EXPORTER.read_bytes()
    (root / "captured-exporter.py").write_bytes(exporter_bytes)
    shutil.copyfile(
        INPUT_FIXTURE / "manifest.json", root / "input-fixture-manifest.json"
    )
    prompts = []
    hidden_hashes = {}
    for prompt_index, prompt_length in enumerate(exporter.PROMPT_LENGTHS):
        prompt_id = f"raw-p{prompt_length:04d}"
        input_relative = f"inputs/{prompt_id}.u32le"
        input_path = root / input_relative
        u32_file(input_path, range(1, prompt_length + 1))

        hidden_relative = f"prompts/{prompt_id}/final-hidden.f32le"
        hidden_path = root / hidden_relative
        f32_file(
            hidden_path,
            (float(prompt_index) + float(index) / 8192 for index in range(exporter.HIDDEN_SIZE)),
        )
        hidden_record = file_record(
            exporter, hidden_path, hidden_relative, exporter.HIDDEN_SIZE
        )
        hidden_hashes[prompt_id] = hidden_record["sha256"]

        logits_relative = f"prompts/{prompt_id}/prefill-logits.f32le"
        logits_path = root / logits_relative
        top1 = 100 + prompt_index
        second = 200 + prompt_index
        logits = array("f", [0.0]) * exporter.VOCAB_SIZE
        logits[top1] = 10.0
        logits[second] = 9.0
        if sys.byteorder != "little":
            logits.byteswap()
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        with logits_path.open("wb") as handle:
            logits.tofile(handle)
        logits_record = file_record(
            exporter, logits_path, logits_relative, exporter.VOCAB_SIZE
        )
        top_10 = validator.scan_f32(
            logits_path,
            elements=exporter.VOCAB_SIZE,
            top_k=exporter.TOP_K,
            label="synthetic logits",
        )

        full_sequence = [top1] + [1000 + index for index in range(1, 512)]
        generation_cases = []
        for case_id, max_new_tokens, ignore_eos in validator.expected_exported_cases(
            prompt_length
        ):
            tokens = full_sequence[:max_new_tokens]
            token_relative = f"prompts/{prompt_id}/{case_id}.u32le"
            token_path = root / token_relative
            u32_file(token_path, tokens)
            digest = exporter.sha256_file(token_path)
            generation_cases.append(
                {
                    "case_id": case_id,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": ignore_eos,
                    "generated_tokens": len(tokens),
                    "finish_reason": "length",
                    "token_file": token_relative,
                    "token_file_bytes": len(tokens) * 4,
                    "token_ids_u32_le_sha256": digest,
                    "first_token_matches_prefill_top1": True,
                }
            )
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt_tokens": prompt_length,
                "input": {
                    "file": input_relative,
                    "dtype": "u32_le",
                    "bytes": prompt_length * 4,
                    "sha256": exporter.sha256_file(input_path),
                    "token_rule": exporter.PRODUCT_CONTRACT["prompt_rule"],
                    "position_start": 0,
                    "position_end_inclusive": prompt_length - 1,
                    "attention": "causal",
                },
                "prefill": {
                    "capture_case_id": "greedy-g1",
                    "forward_token_count": prompt_length,
                    "final_hidden": hidden_record,
                    "logits": logits_record,
                    "top_10": top_10,
                },
                "generation_cases": generation_cases,
            }
        )

    run_records = []
    for prompt_length in exporter.PROMPT_LENGTHS:
        prompt_id = f"raw-p{prompt_length:04d}"
        for case_id, _, _ in validator.expected_exported_cases(prompt_length):
            run_records.append(
                {
                    "run_index": len(run_records),
                    "prompt_id": prompt_id,
                    "case_id": case_id,
                    "prefill_forward_token_count": prompt_length,
                    "captured_final_norm_rows": 1,
                    "prefill_hidden_f32_sha256": hidden_hashes[prompt_id],
                }
            )

    payload = exporter.payload_manifest(root)
    payload_path = root / "payload-manifest.json"
    exporter.write_json(payload_path, payload)
    revision_names = {
        record["file"]
        for record in exporter.SOURCE_IDENTITY["checkpoint_files"]
        + exporter.TOKENIZER_IDENTITY["files"]
    }
    metadata = {
        "schema_version": exporter.SCHEMA_VERSION,
        "status": "captured_real_vllm",
        "created_utc": "2026-07-10T00:00:00+00:00",
        "source_fixture": {
            "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
            "manifest_file": "input-fixture-manifest.json",
            "manifest_sha256": exporter.INPUT_MANIFEST_SHA256,
        },
        "source_model": {
            "identity": exporter.SOURCE_IDENTITY,
            "tokenizer_identity": exporter.TOKENIZER_IDENTITY,
            "revision_metadata": {
                "revision": exporter.SOURCE_IDENTITY["revision"],
                "revision_consistent": True,
                "per_file_revisions": {
                    name: exporter.SOURCE_IDENTITY["revision"]
                    for name in revision_names
                },
            },
        },
        "execution": {
            "identity": exporter.VLLM_IDENTITY,
            "environment": {
                "packages": {
                    "vllm": exporter.VLLM_IDENTITY["package_version"],
                    "torch": exporter.VLLM_IDENTITY["torch_version"],
                    "transformers": exporter.VLLM_IDENTITY["transformers_version"],
                },
                "python_version": exporter.VLLM_IDENTITY["python_version"],
                "torch_git_version": exporter.VLLM_IDENTITY["torch_git_version"],
                "torch_hip_version": exporter.VLLM_IDENTITY["torch_hip_version"],
                "device": exporter.VLLM_IDENTITY["device"],
            },
            "quantization": exporter.QUANTIZATION,
            "model_info": exporter.MODEL_INFO,
            "engine": {
                "max_model_len": exporter.CONTEXT_LENGTH,
                "max_num_batched_tokens": exporter.CONTEXT_LENGTH,
                "kv_cache_memory_bytes": exporter.KV_CACHE_MEMORY_BYTES,
                "v1_multiprocessing": False,
                "seed": 0,
            },
            "sampling": {
                "method": "greedy",
                "temperature": 0.0,
                "seed": 0,
                "top_k_recorded": exporter.TOP_K,
                "topk_tie_breaker": "logit_descending_token_id_ascending",
                "profiles": {
                    "normal_eos_stop": {
                        "case_ids": ["greedy-g1", "greedy-g8", "greedy-g64"],
                        "min_new_tokens": 0,
                        "ignore_eos": False,
                        "stop_token_ids": list(exporter.EOS_TOKEN_IDS),
                    },
                    "ignore_eos_boundary": {
                        "case_ids": [exporter.BOUNDARY_GENERATION_CASE_ID],
                        "min_new_tokens": 512,
                        "ignore_eos": True,
                        "stop_token_ids": [],
                    },
                },
            },
        },
        "capture": {
            "one_model_load": True,
            "runs_sequential": True,
            "maximum_concurrent_requests": 1,
            "run_order": "prompt_length_then_generation_length_ascending",
            "run_count": len(run_records),
            "hook_semantics": "first_forward_final_norm_last_row_only",
            "captured_final_norm_rows_per_run": 1,
            "full_logits_resident_limit": 1,
            "full_logits_capture_case_id_per_prompt": "greedy-g1",
            "runs": run_records,
        },
        "prompts": prompts,
        "exporter": {
            "git_commit": "a" * 40,
            "git_worktree_dirty": False,
            "exporter_git_status": [],
            "exporter_repo_relative_path": "tools/export-sq8-serving-vllm-oracles.py",
            "captured_file": "captured-exporter.py",
            "captured_file_bytes": len(exporter_bytes),
            "captured_file_sha256": hashlib.sha256(exporter_bytes).hexdigest(),
        },
        "payload_manifest": {
            "file": "payload-manifest.json",
            "bytes": payload_path.stat().st_size,
            "sha256": exporter.sha256_file(payload_path),
        },
    }
    exporter.write_json(root / "metadata.json", metadata)
    exporter.write_sums(root)


def rehash_oracle(root: Path, exporter) -> None:
    payload_path = root / "payload-manifest.json"
    payload = json.loads(payload_path.read_text(encoding="ascii"))
    for record in payload["files"]:
        path = root / record["file"]
        record["bytes"] = path.stat().st_size
        record["sha256"] = exporter.sha256_file(path)
    exporter.write_json(payload_path, payload)
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    metadata["payload_manifest"]["bytes"] = payload_path.stat().st_size
    metadata["payload_manifest"]["sha256"] = exporter.sha256_file(payload_path)
    exporter.write_json(metadata_path, metadata)
    exporter.write_sums(root)


class Sq8ServingVllmOracleToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exporter = load_module("sq8_serving_vllm_exporter", EXPORTER)
        cls.validator = load_module("sq8_serving_vllm_validator", VALIDATOR)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.synthetic = Path(cls.temporary.name) / "synthetic"
        build_synthetic_oracle(
            cls.synthetic,
            exporter=cls.exporter,
            validator=cls.validator,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def copied_oracle(self, parent: Path) -> Path:
        destination = parent / "oracle"
        shutil.copytree(self.synthetic, destination, symlinks=True)
        return destination

    def run_validator(
        self,
        oracle: Path,
        *,
        anchor_sha256: str | None = None,
        anchor_file: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR)]
        if anchor_sha256 is not None:
            command.extend(["--anchor-sha256", anchor_sha256])
        if anchor_file is not None:
            command.extend(["--anchor-file", str(anchor_file)])
        command.append(str(oracle))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_exporter(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(EXPORTER),
                "--model-dir",
                "/definitely/missing/model",
                "--output-dir",
                str(output),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def rewrite_metadata(self, root: Path, mutate) -> None:
        path = root / "metadata.json"
        value = json.loads(path.read_text(encoding="ascii"))
        mutate(value)
        self.exporter.write_json(path, value)
        self.exporter.write_sums(root)

    def assert_metadata_rejected(self, mutate, expected_error: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            self.rewrite_metadata(root, mutate)
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)

    def test_unanchored_synthetic_oracle_is_contract_only(self) -> None:
        result = self.run_validator(self.synthetic)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid=true oracle_status=captured_real_vllm", result.stdout)
        self.assertIn("mode=contract-only trusted=false prompts=6 runs=21", result.stdout)

    def test_cli_anchor_promotes_exact_metadata_only(self) -> None:
        digest = self.exporter.sha256_file(self.synthetic / "metadata.json")
        result = self.run_validator(self.synthetic, anchor_sha256=digest)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=promotion trusted=true", result.stdout)

        wrong = self.run_validator(self.synthetic, anchor_sha256="0" * 64)
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("promotion trust anchor", wrong.stderr)

    def test_external_anchor_file_promotes_but_internal_anchor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            digest = self.exporter.sha256_file(root / "metadata.json")
            external = Path(temporary) / "anchor.txt"
            external.write_text(digest + "\n", encoding="ascii")
            promoted = self.run_validator(root, anchor_file=external)
            self.assertEqual(promoted.returncode, 0, promoted.stderr)
            self.assertIn("trusted=true", promoted.stdout)

            internal = root / "anchor.txt"
            internal.write_text(digest + "\n", encoding="ascii")
            rejected = self.run_validator(root, anchor_file=internal)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("outside the producer-controlled oracle tree", rejected.stderr)

    def test_exporter_refuses_existing_output_before_heavyweight_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oracle"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("unchanged", encoding="ascii")
            result = self.run_exporter(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertEqual(marker.read_text(encoding="ascii"), "unchanged")

    def test_exporter_refuses_dangling_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "oracle"
            os.symlink("missing-target", output)
            result = self.run_exporter(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertTrue(output.is_symlink())

    def test_atomic_publish_does_not_replace_raced_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "source-marker").write_text("source", encoding="ascii")
            (destination / "destination-marker").write_text("destination", encoding="ascii")
            with self.assertRaises(FileExistsError):
                self.exporter.rename_noreplace(source, destination)
            self.assertEqual((source / "source-marker").read_text(), "source")
            self.assertEqual(
                (destination / "destination-marker").read_text(), "destination"
            )

    def test_input_contract_and_schedule_include_all_feasible_cases(self) -> None:
        self.assertEqual(self.exporter.DEFAULT_FIXTURE, INPUT_FIXTURE)
        self.assertEqual(
            self.exporter.sha256_file(INPUT_FIXTURE / "manifest.json"),
            "c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af",
        )
        contract = self.exporter.load_input_contract(self.exporter.DEFAULT_FIXTURE)
        schedule = self.exporter.build_schedule(contract)
        self.assertEqual(len(schedule), 21)
        self.assertEqual(
            sum(
                run["case_id"] == self.exporter.BOUNDARY_GENERATION_CASE_ID
                for run in schedule
            ),
            5,
        )
        self.assertEqual(
            [run["case_id"] for run in schedule if run["prompt_id"] == "raw-p4095"],
            ["greedy-g1"],
        )

    def test_current_fixture_passes_contract_only_dry_validation(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                str(SERVING_FIXTURE_VALIDATOR),
                "--contract-only",
                str(INPUT_FIXTURE),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=contract-only trusted=false", result.stdout)

    def test_producer_passed_field_is_rejected(self) -> None:
        self.assert_metadata_rejected(
            lambda value: value.__setitem__("passed", True),
            "metadata keys differ",
        )

    def test_exporter_git_status_and_dirty_flag_are_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            self.rewrite_metadata(
                root,
                lambda value: (
                    value["exporter"].__setitem__("git_worktree_dirty", True),
                    value["exporter"].__setitem__(
                        "exporter_git_status",
                        ["?? tools/export-sq8-serving-vllm-oracles.py"],
                    ),
                ),
            )
            accepted = self.run_validator(root)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            self.rewrite_metadata(
                root,
                lambda value: value["exporter"].__setitem__(
                    "git_worktree_dirty", False
                ),
            )
            rejected = self.run_validator(root)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("inconsistent with git_worktree_dirty=false", rejected.stderr)

    def test_model_tokenizer_and_vllm_identity_drift_is_rejected(self) -> None:
        cases = (
            (
                lambda value: value["source_model"]["identity"].__setitem__(
                    "revision", "0" * 40
                ),
                "source_model.identity.revision differs",
            ),
            (
                lambda value: value["source_model"]["tokenizer_identity"].__setitem__(
                    "chat_template_sha256", "0" * 64
                ),
                "source_model.tokenizer_identity.chat_template_sha256 differs",
            ),
            (
                lambda value: value["execution"]["identity"].__setitem__(
                    "package_version", "0.0.0"
                ),
                "execution.identity.package_version differs",
            ),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                self.assert_metadata_rejected(mutate, error)

    def test_g512_contract_tamper_is_rejected(self) -> None:
        self.assert_metadata_rejected(
            lambda value: value["prompts"][0]["generation_cases"][3].__setitem__(
                "ignore_eos", False
            ),
            "prompts[0].generation_cases[3].ignore_eos differs",
        )
        self.assert_metadata_rejected(
            lambda value: value["prompts"][0]["generation_cases"][3].__setitem__(
                "generated_tokens", 511
            ),
            "ignore-EOS boundary must emit exactly 512 tokens",
        )
        self.assert_metadata_rejected(
            lambda value: value["prompts"][0]["generation_cases"][3].__setitem__(
                "finish_reason", "stop"
            ),
            "ignore-EOS boundary must emit exactly 512 tokens",
        )
        self.assert_metadata_rejected(
            lambda value: value["execution"]["sampling"]["profiles"][
                "ignore_eos_boundary"
            ].__setitem__("stop_token_ids", list(self.exporter.EOS_TOKEN_IDS)),
            "execution.sampling.profiles.ignore_eos_boundary.stop_token_ids length differs",
        )

    def test_run_capture_count_and_hidden_hash_are_recomputed(self) -> None:
        cases = (
            (
                lambda value: value["capture"].__setitem__("run_count", 20),
                "capture.run_count differs",
            ),
            (
                lambda value: value["capture"]["runs"][1].__setitem__(
                    "captured_final_norm_rows", 2
                ),
                "capture.runs[1].captured_final_norm_rows differs",
            ),
            (
                lambda value: value["capture"]["runs"][1].__setitem__(
                    "prefill_hidden_f32_sha256", "0" * 64
                ),
                "capture.runs[1].prefill_hidden_f32_sha256 differs",
            ),
        )
        for mutate, error in cases:
            with self.subTest(error=error):
                self.assert_metadata_rejected(mutate, error)

    def test_self_consistent_logits_ranking_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "prompts/raw-p0001/prefill-logits.f32le"
            with path.open("r+b") as handle:
                handle.seek(500 * 4)
                handle.write(struct.pack("<f", 100.0))
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            metadata["prompts"][0]["prefill"]["logits"]["sha256"] = (
                self.exporter.sha256_file(path)
            )
            self.exporter.write_json(metadata_path, metadata)
            rehash_oracle(root, self.exporter)
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not match recomputed logits ranking", result.stderr)

    def test_self_consistent_nonfinite_logit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "prompts/raw-p0008/prefill-logits.f32le"
            with path.open("r+b") as handle:
                handle.seek(17 * 4)
                handle.write(struct.pack("<f", math.nan))
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            metadata["prompts"][1]["prefill"]["logits"]["sha256"] = (
                self.exporter.sha256_file(path)
            )
            self.exporter.write_json(metadata_path, metadata)
            rehash_oracle(root, self.exporter)
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("contains a non-finite value", result.stderr)

    def test_self_consistent_generation_prefix_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "prompts/raw-p0032/greedy-g8.u32le"
            with path.open("r+b") as handle:
                handle.seek(4)
                handle.write(struct.pack("<I", 7777))
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            metadata["prompts"][2]["generation_cases"][1][
                "token_ids_u32_le_sha256"
            ] = self.exporter.sha256_file(path)
            self.exporter.write_json(metadata_path, metadata)
            rehash_oracle(root, self.exporter)
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("greedy sequences are not prefix-consistent", result.stderr)

    def test_ignore_eos_boundary_may_continue_after_normal_eos_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            metadata_path = root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="ascii"))
            cases = metadata["prompts"][0]["generation_cases"]
            top1 = metadata["prompts"][0]["prefill"]["top_10"][0]["token_id"]
            eos = self.exporter.EOS_TOKEN_IDS[0]

            def replace_case(case_index: int, tokens: list[int], finish: str) -> None:
                case = cases[case_index]
                path = root / case["token_file"]
                u32_file(path, tokens)
                case["generated_tokens"] = len(tokens)
                case["finish_reason"] = finish
                case["token_file_bytes"] = len(tokens) * 4
                case["token_ids_u32_le_sha256"] = self.exporter.sha256_file(path)

            stopped = [top1, eos]
            replace_case(1, stopped, "stop")
            replace_case(2, stopped, "stop")
            boundary_path = root / cases[3]["token_file"]
            boundary_bytes = boundary_path.read_bytes()
            boundary = [
                struct.unpack("<I", boundary_bytes[index : index + 4])[0]
                for index in range(0, len(boundary_bytes), 4)
            ]
            boundary[1] = eos
            replace_case(3, boundary, "length")
            self.exporter.write_json(metadata_path, metadata)
            rehash_oracle(root, self.exporter)

            accepted = self.run_validator(root)
            self.assertEqual(accepted.returncode, 0, accepted.stderr)

            replace_case(2, boundary[:64], "length")
            self.exporter.write_json(metadata_path, metadata)
            rehash_oracle(root, self.exporter)
            rejected = self.run_validator(root)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("continued after greedy-g8 stopped on EOS", rejected.stderr)

    def test_payload_corruption_is_rejected_by_payload_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "prompts/raw-p4095/final-hidden.f32le"
            with path.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sha256 differs from the artifact", result.stderr)

    def test_duplicate_json_key_and_nonfinite_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "metadata.json"
            raw = path.read_text(encoding="ascii")
            needle = '  "schema_version": "ullm.sq8.serving_oracle.v1",'
            path.write_text(raw.replace(needle, needle + "\n" + needle, 1), encoding="ascii")
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key: schema_version", result.stderr)

        self.assert_metadata_rejected(
            lambda value: value["execution"]["sampling"].__setitem__(
                "temperature", float("nan")
            ),
            "non-finite JSON number is forbidden: NaN",
        )

    def test_extra_file_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            (root / "unexpected").write_text("extra", encoding="ascii")
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("oracle file set differs", result.stderr)

        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_oracle(Path(temporary))
            path = root / "inputs/raw-p0001.u32le"
            path.unlink()
            path.symlink_to("raw-p0008.u32le")
            result = self.run_validator(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular non-symlink file", result.stderr)

    def test_topk_tie_breaks_by_ascending_token_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "logits.f32le"
            logits = array("f", [0.0]) * self.validator.VOCAB_SIZE
            logits[11] = 7.0
            logits[3] = 7.0
            if sys.byteorder != "little":
                logits.byteswap()
            with path.open("wb") as handle:
                logits.tofile(handle)
            top_10 = self.validator.scan_f32(
                path,
                elements=self.validator.VOCAB_SIZE,
                top_k=self.validator.TOP_K,
                label="tie logits",
            )
            self.assertEqual([entry["token_id"] for entry in top_10[:2]], [3, 11])

    def test_validator_does_not_import_exporter_or_ml_frameworks(self) -> None:
        source = VALIDATOR.read_text(encoding="ascii")
        imports = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertTrue({"numpy", "torch", "vllm"}.isdisjoint(imports))
        self.assertNotIn("export_sq8_serving_vllm_oracles", imports)


if __name__ == "__main__":
    unittest.main()
