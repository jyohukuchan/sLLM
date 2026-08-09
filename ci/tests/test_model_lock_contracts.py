#!/usr/bin/env python3
"""Host-only contract tests for the immutable Qwen model lock."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError  # noqa: E402
from validate_matrix import command_is_model_lock_contract  # noqa: E402
from validate_model_lock import (  # noqa: E402
    JCSValidationError,
    fingerprint_for_document,
    jcs_dumps,
    read_json,
    validate_cache,
    validate_document,
    validate_lock_file,
)


LOCK_PATH = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
SCHEMA_PATH = ROOT / "ci/schema/model-lock-v1.schema.json"
FIXTURE_LOCK = ROOT / "ci/fixtures/model-lock-v1/lock.json"
FIXTURE_CACHE = ROOT / "ci/fixtures/model-lock-v1/cache"


class ModelLockContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = validate_lock_file(LOCK_PATH, schema_path=SCHEMA_PATH)

    def assert_rejected(self, document: dict[str, object]) -> None:
        with self.assertRaises((ContractError, JCSValidationError, ValueError)):
            validate_document(document, schema_path=SCHEMA_PATH)

    def _reviewed_qwen_config(self) -> dict[str, object]:
        module = __import__("validate_model_lock")
        text = copy.deepcopy(self.lock["model"]["architecture"]["text_config"])
        text["dtype"] = "bfloat16"
        text["rms_norm_eps"] = 0.000001
        text["eos_token_id"] = self.lock["model"]["tokenizer_contract"]["stop_identity"]["config_eos"]["token_id"]
        text.update(copy.deepcopy(module.QWEN_TEXT_OPTIONAL_CONFIG))
        text["rope_parameters"] = copy.deepcopy(module.QWEN_TEXT_ROPE_PARAMETERS)
        return {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "image_token_id": 248056,
            "model_type": "qwen3_5",
            "text_config": text,
            "tie_word_embeddings": True,
            "transformers_version": "4.57.0.dev0",
            "video_token_id": 248057,
            "vision_config": copy.deepcopy(module.QWEN_VISION_CONFIG),
            "vision_end_token_id": 248054,
            "vision_start_token_id": 248053,
        }

    def test_qwen_lock_is_complete_and_fingerprint_bound(self) -> None:
        model = self.lock["model"]
        self.assertEqual(self.lock["schema_version"], "model-lock-v1")
        self.assertEqual(model["repo_id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(model["requested_revision"], "main")
        self.assertEqual(model["resolved_revision"], "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(self.lock["aliases"], ["qwen3.5-4b-bf16"])
        self.assertEqual(self.lock["fingerprint"], fingerprint_for_document(self.lock))
        self.assertEqual(len(model["files"]), 13)
        self.assertEqual(model["tensor_contract"]["indexed_tensor_count"], 738)
        self.assertEqual(model["architecture"]["text_config"]["layer_types"], [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ] * 8)
        self.assertEqual(model["slice_contract"]["absolute_byte_range"], [94432, 99552])
        self.assertEqual(model["slice_contract"]["header_length_field_bytes"], 8)
        self.assertEqual(model["slice_contract"]["header_length_bytes"], 79064)
        self.assertEqual(model["slice_contract"]["data_buffer_start"], 79072)
        self.assertEqual(model["slice_contract"]["data_offset_basis"], "data-buffer-relative")
        self.assertEqual(model["slice_contract"]["normalization"]["scale_mode"], "offset-one")
        self.assertNotIn("runtime_supported", model["tokenizer_contract"])
        self.assertNotIn("runtime_policy", model["tokenizer_contract"]["stop_identity"])
        self.assertEqual(model["tokenizer_contract"]["stop_identity"]["config_eos"]["token_id"], 248044)
        self.assertEqual(model["tokenizer_contract"]["stop_identity"]["tokenizer_eos"]["token_id"], 248046)
        self.assertEqual(
            model["tokenizer_contract"]["generation_stop_policy"],
            {
                "version": 1,
                "stop_token_ids": [248046, 248044],
                "evaluation": "newly_generated_after_argmax",
                "prompt_evaluation": "never_stop",
                "stop_token": {"visible_output": False, "subsequent_decode_input": False},
                "budget_boundary": "stop_token_wins",
                "max_new_tokens_zero": "max_new_tokens_before_decode",
                "reason_version": 1,
            },
        )
        self.assertEqual(model["generation_config"], {"present": False, "path": None})

    def test_generation_stop_policy_is_strict_and_fails_closed(self) -> None:
        policy = self.lock["model"]["tokenizer_contract"]["generation_stop_policy"]
        self.assertEqual(policy["stop_token_ids"], [248046, 248044])

        def rejected(label: str, mutate, *, recompute: bool = True) -> None:
            with self.subTest(label=label):
                changed = copy.deepcopy(self.lock)
                mutate(changed["model"]["tokenizer_contract"]["generation_stop_policy"])
                if recompute:
                    try:
                        changed["fingerprint"] = fingerprint_for_document(changed)
                    except JCSValidationError:
                        pass
                self.assert_rejected(changed)

        missing_policy = copy.deepcopy(self.lock)
        del missing_policy["model"]["tokenizer_contract"]["generation_stop_policy"]
        self.assert_rejected(missing_policy)
        old_runtime_supported = copy.deepcopy(self.lock)
        old_runtime_supported["model"]["tokenizer_contract"]["runtime_supported"] = False
        self.assert_rejected(old_runtime_supported)
        old_runtime_policy = copy.deepcopy(self.lock)
        old_runtime_policy["model"]["tokenizer_contract"]["stop_identity"]["runtime_policy"] = "unresolved"
        self.assert_rejected(old_runtime_policy)

        for field in (
            "version", "stop_token_ids", "evaluation", "prompt_evaluation", "stop_token",
            "budget_boundary", "max_new_tokens_zero", "reason_version",
        ):
            rejected(f"missing-{field}", lambda value, field=field: value.pop(field))
        for field, value in (
            ("version", 2),
            ("reason_version", 2),
            ("stop_token_ids", []),
            ("stop_token_ids", [248046, 248046]),
            ("stop_token_ids", [248044, 248046]),
            ("stop_token_ids", [-1, 248044]),
            ("stop_token_ids", [248046, 4_294_967_296]),
            ("stop_token_ids", [True, 248044]),
            ("stop_token_ids", [248046.0, 248044]),
            ("stop_token_ids", ["248046", 248044]),
            ("evaluation", "argmax"),
            ("prompt_evaluation", "stop"),
            ("budget_boundary", "budget_wins"),
            ("max_new_tokens_zero", "decode_zero"),
            ("stop_token", {"visible_output": True, "subsequent_decode_input": False}),
            ("stop_token", {"visible_output": False, "subsequent_decode_input": 0}),
            ("stop_token", {"visible_output": False, "subsequent_decode_input": False, "extra": 1}),
        ):
            rejected(
                f"invalid-{field}-{value!r}",
                lambda policy, field=field, value=value: policy.__setitem__(field, value),
            )

        unknown = copy.deepcopy(self.lock)
        unknown["model"]["tokenizer_contract"]["generation_stop_policy"]["unknown"] = 1
        self.assert_rejected(unknown)

    def test_qwen_config_semantics_reject_layer_and_reviewed_field_drift(self) -> None:
        module = __import__("validate_model_lock")
        baseline = self._reviewed_qwen_config()
        with mock.patch.object(module, "_read_cache_json", return_value=baseline):
            module._validate_qwen_config({}, self.lock["model"])

        mutations = (
            lambda config: config["text_config"].pop("layer_types"),
            lambda config: config["text_config"].update({"layer_types": config["text_config"]["layer_types"] + ["linear_attention"]}),
            lambda config: config["text_config"]["layer_types"].__setitem__(0, 7),
            lambda config: config["text_config"]["layer_types"].__setitem__(3, "linear_attention"),
            lambda config: config.__setitem__("tie_word_embeddings", False),
            lambda config: config["vision_config"].__setitem__("depth", 23),
            lambda config: config["text_config"].__setitem__("linear_num_value_heads", 31),
            lambda config: config["text_config"].__setitem__("mtp_use_dedicated_embeddings", True),
        )
        for mutate in mutations:
            changed = copy.deepcopy(baseline)
            mutate(changed)
            with mock.patch.object(module, "_read_cache_json", return_value=changed):
                with self.assertRaises(ContractError):
                    module._validate_qwen_config({}, self.lock["model"])

    def test_qwen_config_semantics_reject_json_numeric_type_aliases(self) -> None:
        module = __import__("validate_model_lock")
        baseline = self._reviewed_qwen_config()
        mutations = (
            ("integer-as-float", lambda config: config.__setitem__("image_token_id", 248056.0)),
            ("integer-as-bool", lambda config: config.__setitem__("image_token_id", True)),
            ("float-as-integer", lambda config: config["text_config"].__setitem__("attention_dropout", 0)),
            ("float-as-bool", lambda config: config["text_config"].__setitem__("attention_dropout", False)),
            ("float-as-string", lambda config: config["text_config"].__setitem__("attention_dropout", "0.0")),
            ("float-as-nan", lambda config: config["text_config"].__setitem__("attention_dropout", math.nan)),
            ("float-as-infinity", lambda config: config["text_config"].__setitem__("attention_dropout", math.inf)),
            ("rms-epsilon-as-string", lambda config: config["text_config"].__setitem__("rms_norm_eps", "1e-6")),
            ("nested-bool-as-integer", lambda config: config["text_config"]["rope_parameters"].__setitem__("mrope_interleaved", 1)),
            ("nested-integer-as-float", lambda config: config["text_config"]["rope_parameters"]["mrope_section"].__setitem__(0, 11.0)),
            ("nested-float-as-integer", lambda config: config["vision_config"].__setitem__("initializer_range", 0)),
            ("nested-integer-as-bool", lambda config: config["text_config"].__setitem__("linear_num_value_heads", False)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = copy.deepcopy(baseline)
                mutate(changed)
                with mock.patch.object(module, "_read_cache_json", return_value=changed):
                    with self.assertRaises(ContractError):
                        module._validate_qwen_config({}, self.lock["model"])

    def test_bound_lock_and_schema_reads_reject_links_races_and_oversize_without_fd_leaks(self) -> None:
        module = __import__("validate_model_lock")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            schema = root / "schema.json"
            shutil.copy2(LOCK_PATH, lock)
            shutil.copy2(SCHEMA_PATH, schema)
            baseline = self._proc_fd_count()
            for _ in range(32):
                validate_lock_file(lock, schema_path=schema)
                self.assertEqual(self._proc_fd_count(), baseline)

            lock_link = root / "lock-link.json"
            lock_link.symlink_to(lock)
            with self.assertRaises(ContractError):
                validate_lock_file(lock_link, schema_path=schema)
            schema_link = root / "schema-link.json"
            schema_link.symlink_to(schema)
            with self.assertRaises(ContractError):
                validate_lock_file(lock, schema_path=schema_link)

            for target, label in ((lock, "lock"), (schema, "schema")):
                oversized = root / f"{label}-oversized.json"
                shutil.copy2(target, oversized)
                with oversized.open("r+b") as handle:
                    handle.truncate(1024 * 1024 + 1)
                with self.assertRaises(ContractError):
                    if label == "lock":
                        validate_lock_file(oversized, schema_path=schema)
                    else:
                        validate_lock_file(lock, schema_path=oversized)

            original_read = module._read_exact
            for target, label in ((lock, "lock"), (schema, "schema")):
                replacement = root / f"{label}-replacement.json"
                shutil.copy2(target, replacement)
                original_stat = os.stat(target)
                replaced = False

                def replace_after_open(descriptor, size, *, offset, max_bytes):
                    nonlocal replaced
                    if not replaced and os.fstat(descriptor).st_ino == original_stat.st_ino:
                        replaced = True
                        os.replace(replacement, target)
                    return original_read(descriptor, size, offset=offset, max_bytes=max_bytes)

                with mock.patch.object(module, "_read_exact", side_effect=replace_after_open):
                    with self.assertRaises(ContractError):
                        if label == "lock":
                            validate_lock_file(lock, schema_path=schema)
                        else:
                            validate_lock_file(lock, schema_path=schema)
                self.assertTrue(replaced)

                shutil.copy2(LOCK_PATH if label == "lock" else SCHEMA_PATH, target)
                original_stat = os.stat(target)
                payload = target.read_bytes()
                offset = payload.index(b"\n")
                mutated = False

                def mutate_same_inode_after_open(descriptor, size, *, offset: int, max_bytes: int):
                    nonlocal mutated
                    if not mutated and os.fstat(descriptor).st_ino == original_stat.st_ino:
                        mutated = True
                        write_descriptor = os.open(
                            target,
                            os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                        )
                        try:
                            self.assertEqual(os.pwrite(write_descriptor, b" ", payload.index(b"\n")), 1)
                        finally:
                            os.close(write_descriptor)
                    return original_read(descriptor, size, offset=offset, max_bytes=max_bytes)

                with mock.patch.object(module, "_read_exact", side_effect=mutate_same_inode_after_open):
                    with self.assertRaises(ContractError):
                        validate_lock_file(lock, schema_path=schema)
                self.assertTrue(mutated)
                shutil.copy2(LOCK_PATH if label == "lock" else SCHEMA_PATH, target)

    def test_jcs_known_answers_cover_order_unicode_escape_and_domain(self) -> None:
        self.assertEqual(jcs_dumps({"b": 1, "a": True, "n": None}), '{"a":true,"b":1,"n":null}')
        # UTF-16 code-unit order puts U+10000 (D800 DC00) before U+E000.
        self.assertEqual(
            jcs_dumps({"\U0000e000": 2, "\U00010000": 1}),
            '{"𐀀":1,"":2}',
        )
        self.assertEqual(
            jcs_dumps({"s": "\b\t\n\f\r\"\\\u0001/é"}),
            '{"s":"\\b\\t\\n\\f\\r\\"\\\\\\u0001/é"}',
        )
        self.assertEqual(jcs_dumps([-17, 0, 2**53 - 1, False]), "[-17,0,9007199254740991,false]")
        self.assertEqual(jcs_dumps(-0), "0")
        with self.assertRaises(JCSValidationError):
            jcs_dumps(2**53)
        with self.assertRaises(JCSValidationError):
            jcs_dumps(2**53 + 1)
        with self.assertRaises(JCSValidationError):
            jcs_dumps(-(2**53))
        with self.assertRaises(JCSValidationError):
            jcs_dumps(Decimal("1"))
        with self.assertRaises(JCSValidationError):
            jcs_dumps({"float": 1.0})
        with self.assertRaises(JCSValidationError):
            jcs_dumps({"surrogate": "\ud800"})
        with self.assertRaises(JCSValidationError):
            jcs_dumps({"\udc00": 1})
        nested: object = None
        for _ in range(65):
            nested = [nested]
        with self.assertRaises(JCSValidationError):
            jcs_dumps(nested)

    def test_lock_controls_and_generated_at_require_canonical_real_utc(self) -> None:
        fixture = read_json(FIXTURE_LOCK)
        for field, value in (
            (("model", "requested_revision"), "fixture\nrevision"),
            (("model", "license", "statement"), "MIT\x7f"),
            (("aliases", 0), "fixture\u0085tiny"),
            (("model", "files", 0, "path"), "config\x00.json"),
        ):
            mutated = copy.deepcopy(fixture)
            target: object = mutated
            for key in field[:-1]:
                target = target[key]  # type: ignore[index]
            target[field[-1]] = value  # type: ignore[index]
            mutated["fingerprint"] = fingerprint_for_document(mutated)
            self.assert_rejected(mutated)
        for value in ("2026-02-30T00:00:00Z", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00.0Z"):
            mutated = copy.deepcopy(fixture)
            mutated["generated_at"] = value
            self.assert_rejected(mutated)

    def test_qwen_machine_readable_tensor_catalog_is_exactly_classified(self) -> None:
        module = __import__("validate_model_lock")
        catalog = module._qwen_tensor_catalog(
            module._qwen_shape_inputs(self._reviewed_qwen_config(), self.lock["model"])
        )
        self.assertEqual(len(catalog), 738)
        self.assertEqual(sum(classification == "text" for classification, _, _ in catalog.values()), 426)
        self.assertEqual(sum(classification == "vision" for classification, _, _ in catalog.values()), 297)
        self.assertEqual(sum(classification == "mtp" for classification, _, _ in catalog.values()), 15)
        self.assertEqual(
            catalog["model.language_model.layers.0.linear_attn.A_log"],
            ("text", "F32", (32,)),
        )
        self.assertEqual(
            catalog["model.language_model.layers.0.linear_attn.dt_bias"],
            ("text", "F32", (32,)),
        )
        self.assertEqual(
            catalog["model.language_model.layers.0.linear_attn.norm.weight"],
            ("text", "BF16", (128,)),
        )
        self.assertEqual(
            catalog["model.language_model.layers.3.self_attn.q_proj.weight"],
            ("text", "BF16", (8192, 2560)),
        )
        self.assertEqual(
            catalog["model.visual.merger.linear_fc1.weight"],
            ("vision", "BF16", (4096, 4096)),
        )
        self.assertEqual(
            catalog["mtp.layers.0.self_attn.o_proj.weight"],
            ("mtp", "BF16", (2560, 4096)),
        )
        self.assertNotIn("model.language_model.layers.3.linear_attn.A_log", catalog)

    def test_qwen_shape_inputs_reject_non_positive_and_checked_overflow(self) -> None:
        module = __import__("validate_model_lock")
        baseline = self._reviewed_qwen_config()
        for value in (0, -1, 1.0, True, "1"):
            changed = copy.deepcopy(baseline)
            changed["text_config"]["hidden_size"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                module._qwen_shape_inputs(changed, self.lock["model"])
        for value in (0, -1, 1.0, True, "1"):
            changed = copy.deepcopy(baseline)
            changed["vision_config"]["patch_size"] = value
            with self.subTest(value=value), self.assertRaises(ContractError):
                module._qwen_shape_inputs(changed, self.lock["model"])
        for value in (1, 3, 17, 2**64 - 1):
            self.assertEqual(module._checked_shape_mul(value, 1, field="boundary"), value)
        with self.assertRaises(ContractError):
            module._checked_shape_mul(2**64 - 1, 2, field="overflow")
        with self.assertRaises(ContractError):
            module._checked_shape_add(2**64 - 1, 1, field="overflow")

    def test_fingerprint_excludes_only_root_bookkeeping(self) -> None:
        changed = copy.deepcopy(self.lock)
        changed["aliases"] = ["qwen3.5-4b-bf16"]
        changed["generated_at"] = "2026-08-05T00:00:00Z"
        self.assertEqual(fingerprint_for_document(changed), self.lock["fingerprint"])
        changed["model"]["requested_revision"] = "main-reviewed"
        self.assertNotEqual(fingerprint_for_document(changed), self.lock["fingerprint"])

    def test_negative_lock_mutations_fail_closed(self) -> None:
        missing = copy.deepcopy(self.lock)
        del missing["model"]["files"][0]
        self.assert_rejected(missing)

        duplicate = copy.deepcopy(self.lock)
        duplicate["model"]["files"].append(copy.deepcopy(duplicate["model"]["files"][0]))
        self.assert_rejected(duplicate)

        wrong_lfs = copy.deepcopy(self.lock)
        wrong_lfs["model"]["files"][5]["lfs_oid"] = "sha256:" + "0" * 64
        self.assert_rejected(wrong_lfs)

        floating_locator = copy.deepcopy(self.lock)
        floating_locator["model"]["files"][0]["download_url"] = floating_locator["model"]["files"][0]["download_url"].replace(
            self.lock["model"]["resolved_revision"], "main"
        )
        self.assert_rejected(floating_locator)

        wrong_fingerprint = copy.deepcopy(self.lock)
        wrong_fingerprint["fingerprint"] = "sha256:" + "0" * 64
        self.assert_rejected(wrong_fingerprint)

        wrong_commit = copy.deepcopy(self.lock)
        wrong_commit["model"]["resolved_revision"] = "0123456789abcdef0123456789abcdef01234567"
        for entry in wrong_commit["model"]["files"]:
            entry["source_page_url"] = entry["source_page_url"].replace(
                self.lock["model"]["resolved_revision"], wrong_commit["model"]["resolved_revision"]
            )
            entry["download_url"] = entry["download_url"].replace(
                self.lock["model"]["resolved_revision"], wrong_commit["model"]["resolved_revision"]
            )
        self.assert_rejected(wrong_commit)

        wrong_repository = copy.deepcopy(self.lock)
        wrong_repository["model"]["repo_id"] = "Qwen/other-model"
        self.assert_rejected(wrong_repository)

        unknown = copy.deepcopy(self.lock)
        unknown["unexpected"] = True
        self.assert_rejected(unknown)

        unsafe_path = copy.deepcopy(self.lock)
        unsafe_path["model"]["files"][0]["path"] = "../LICENSE"
        self.assert_rejected(unsafe_path)

        alias_drift = copy.deepcopy(self.lock)
        alias_drift["aliases"] = ["qwen3.5-4b"]
        self.assert_rejected(alias_drift)

        unknown_version = copy.deepcopy(self.lock)
        unknown_version["schema_version"] = "model-lock-v2"
        self.assert_rejected(unknown_version)

        nested_unknown = copy.deepcopy(self.lock)
        nested_unknown["model"]["tokenizer_contract"]["stop_identity"]["bypass"] = True
        self.assert_rejected(nested_unknown)

    def test_tiny_fixture_uses_the_same_offline_cache_contract(self) -> None:
        fixture = validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=FIXTURE_CACHE)
        self.assertEqual(fixture["fingerprint"], fingerprint_for_document(fixture))
        tokenizer = fixture["model"]["tokenizer_contract"]
        self.assertNotIn("runtime_supported", tokenizer)
        self.assertNotIn("runtime_policy", tokenizer["stop_identity"])
        self.assertEqual(tokenizer["generation_stop_policy"]["stop_token_ids"], [0])
        self.assertLess(sum(path.stat().st_size for path in FIXTURE_CACHE.rglob("*") if path.is_file()), 64 * 1024)

    def test_cache_missing_extra_modified_and_symlink_inputs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory) / "cache"
            shutil.copytree(FIXTURE_CACHE, temp)
            (temp / "config.json").unlink()
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

            shutil.copytree(FIXTURE_CACHE, temp := Path(directory) / "extra")
            (temp / "extra.bin").write_bytes(b"unexpected")
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

            shutil.copytree(FIXTURE_CACHE, temp := Path(directory) / "modified")
            (temp / "config.json").write_bytes(b"modified")
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

            shutil.copytree(FIXTURE_CACHE, temp := Path(directory) / "symlink")
            (temp / "config.json").unlink()
            (temp / "config.json").symlink_to(FIXTURE_CACHE / "config.json")
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

            root_link = Path(directory) / "root-link"
            root_link.symlink_to(FIXTURE_CACHE, target_is_directory=True)
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=root_link)

    def _fixture_document_with_current_hashes(self, cache: Path) -> dict[str, object]:
        document = json.loads(FIXTURE_LOCK.read_text(encoding="utf-8"))
        for entry in document["model"]["files"]:
            payload = (cache / entry["path"]).read_bytes()
            entry["size_bytes"] = len(payload)
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
        document["fingerprint"] = fingerprint_for_document(document)
        return document

    def _rewrite_fixture_header(self, cache: Path, header: dict[str, object] | None = None, raw_header: bytes | None = None) -> None:
        path = cache / "model.safetensors"
        original = path.read_bytes()
        original_length = struct.unpack("<Q", original[:8])[0]
        payload = original[8 + original_length:]
        if raw_header is None:
            raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)

    def _fixture_document_with_two_tensors(
        self,
        cache: Path,
        *,
        second_offsets: list[int],
        payload: bytes,
    ) -> dict[str, object]:
        header = {
            "__metadata__": {"format": "pt"},
            "fixture.tensor": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
            "fixture.second": {"dtype": "BF16", "shape": [1], "data_offsets": second_offsets},
        }
        raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        (cache / "model.safetensors").write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)
        (cache / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 4},
                    "weight_map": {
                        "fixture.tensor": "model.safetensors",
                        "fixture.second": "model.safetensors",
                    },
                }
            ),
            encoding="utf-8",
        )
        document = self._fixture_document_with_current_hashes(cache)
        tensor_contract = document["model"]["tensor_contract"]
        tensor_contract["indexed_tensor_count"] = 2
        tensor_contract["classifications"][0]["tensor_count"] = 2
        slice_contract = document["model"]["slice_contract"]
        data_buffer_start = 8 + len(raw_header)
        slice_contract.update(
            {
                "shape": [1],
                "header_length_bytes": len(raw_header),
                "data_buffer_start": data_buffer_start,
                "data_offsets": [0, 2],
                "absolute_byte_range": [data_buffer_start, data_buffer_start + 2],
                "byte_size": 2,
            }
        )
        document["fingerprint"] = fingerprint_for_document(document)
        return document

    def _mutate_hashed_file_in_place(
        self,
        hashed_file_descriptor: int,
        path: Path,
        original: bytes,
        *,
        offset: int,
        replacement: bytes,
    ) -> None:
        self.assertGreaterEqual(offset, 0)
        self.assertEqual(len(replacement), len(original[offset:offset + len(replacement)]))
        write_flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        write_file_descriptor = os.open(path, write_flags)
        try:
            hashed_stat = os.fstat(hashed_file_descriptor)
            write_stat_before = os.fstat(write_file_descriptor)
            self.assertNotEqual(write_file_descriptor, hashed_file_descriptor)
            self.assertEqual(write_stat_before.st_dev, hashed_stat.st_dev)
            self.assertEqual(write_stat_before.st_ino, hashed_stat.st_ino)
            self.assertEqual(write_stat_before.st_size, len(original))
            self.assertEqual(os.pread(hashed_file_descriptor, len(original), 0), original)

            self.assertEqual(
                os.pwrite(write_file_descriptor, replacement, offset),
                len(replacement),
            )

            write_stat_after = os.fstat(write_file_descriptor)
            self.assertEqual(write_stat_after.st_dev, write_stat_before.st_dev)
            self.assertEqual(write_stat_after.st_ino, write_stat_before.st_ino)
            self.assertEqual(write_stat_after.st_size, write_stat_before.st_size)
            mutated = os.pread(hashed_file_descriptor, len(original), 0)
            self.assertEqual(len(mutated), len(original))
            self.assertNotEqual(mutated, original)
            self.assertEqual(mutated[offset:offset + len(replacement)], replacement)
        finally:
            os.close(write_file_descriptor)

    def _proc_fd_count(self) -> int:
        return len(os.listdir("/proc/self/fd"))

    def test_fixture_safetensors_negative_header_index_and_slice_contracts(self) -> None:
        mutations = []

        def wrong_dtype(header: dict[str, object]) -> None:
            header["fixture.tensor"]["dtype"] = "F32"
        mutations.append(wrong_dtype)

        def same_width_dtype(header: dict[str, object]) -> None:
            header["fixture.tensor"]["dtype"] = "F16"
        mutations.append(same_width_dtype)

        def wrong_rank(header: dict[str, object]) -> None:
            header["fixture.tensor"]["shape"] = [1, 1]
        mutations.append(wrong_rank)

        def wrong_shape(header: dict[str, object]) -> None:
            header["fixture.tensor"]["shape"] = [3]
        mutations.append(wrong_shape)

        def wrong_offset(header: dict[str, object]) -> None:
            header["fixture.tensor"]["data_offsets"] = [1, 5]
        mutations.append(wrong_offset)

        def wrong_range(header: dict[str, object]) -> None:
            header["fixture.tensor"]["data_offsets"] = [0, 6]
        mutations.append(wrong_range)

        with tempfile.TemporaryDirectory() as directory:
            for serial, mutate in enumerate(mutations):
                temp = Path(directory) / f"metadata-{serial}"
                shutil.copytree(FIXTURE_CACHE, temp)
                raw = (temp / "model.safetensors").read_bytes()
                length = struct.unpack("<Q", raw[:8])[0]
                header = json.loads(raw[8:8 + length])
                mutate(header)
                self._rewrite_fixture_header(temp, header=header)
                document = self._fixture_document_with_current_hashes(temp)
                with self.assertRaises(ContractError):
                    validate_cache(document, temp)

            temp = Path(directory) / "bad-json"
            shutil.copytree(FIXTURE_CACHE, temp)
            self._rewrite_fixture_header(temp, raw_header=b"{\"fixture.tensor\":")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "duplicate"
            shutil.copytree(FIXTURE_CACHE, temp)
            duplicate = b'{"__metadata__":{"format":"pt"},"fixture.tensor":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"fixture.tensor":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}'
            self._rewrite_fixture_header(temp, raw_header=duplicate)
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "metadata-type"
            shutil.copytree(FIXTURE_CACHE, temp)
            raw = (temp / "model.safetensors").read_bytes()
            length = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + length])
            header["__metadata__"]["format"] = 1
            self._rewrite_fixture_header(temp, header=header)
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "header-too-large"
            shutil.copytree(FIXTURE_CACHE, temp)
            shard = temp / "model.safetensors"
            raw = bytearray(shard.read_bytes())
            raw[:8] = struct.pack("<Q", 100_000_001)
            shard.write_bytes(raw)
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "trailing-payload"
            shutil.copytree(FIXTURE_CACHE, temp)
            shard = temp / "model.safetensors"
            shard.write_bytes(shard.read_bytes() + b"\xff")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "gap-payload"
            shutil.copytree(FIXTURE_CACHE, temp)
            raw = (temp / "model.safetensors").read_bytes()
            length = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + length])
            header["fixture.tensor"]["data_offsets"] = [1, 5]
            self._rewrite_fixture_header(temp, header=header)
            shard = temp / "model.safetensors"
            shard.write_bytes(shard.read_bytes() + b"\xff")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "offset-overflow"
            shutil.copytree(FIXTURE_CACHE, temp)
            raw = (temp / "model.safetensors").read_bytes()
            length = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8:8 + length])
            header["fixture.tensor"]["data_offsets"] = [0, 2**64]
            self._rewrite_fixture_header(temp, header=header)
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "index-mismatch"
            shutil.copytree(FIXTURE_CACHE, temp)
            index = json.loads((temp / "model.safetensors.index.json").read_text(encoding="utf-8"))
            index["weight_map"]["other.tensor"] = index["weight_map"].pop("fixture.tensor")
            (temp / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "index-total-size"
            shutil.copytree(FIXTURE_CACHE, temp)
            index = json.loads((temp / "model.safetensors.index.json").read_text(encoding="utf-8"))
            index["metadata"]["total_size"] = 8
            (temp / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "target-mismatch"
            shutil.copytree(FIXTURE_CACHE, temp)
            document = self._fixture_document_with_current_hashes(temp)
            document["model"]["slice_contract"]["tensor_name"] = "fixture.unknown"
            document["fingerprint"] = fingerprint_for_document(document)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

    def test_fixture_accepts_multiple_contiguous_tensors_in_one_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory) / "multiple-tensors"
            shutil.copytree(FIXTURE_CACHE, temp)
            document = self._fixture_document_with_two_tensors(
                temp,
                second_offsets=[2, 4],
                payload=b"\x00\x01\x02\x03",
            )
            validate_cache(document, temp)

    def test_fixture_rejects_overlapping_tensor_spans(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory) / "overlapping-tensors"
            shutil.copytree(FIXTURE_CACHE, temp)
            document = self._fixture_document_with_two_tensors(
                temp,
                second_offsets=[1, 3],
                payload=b"\x00\x01\x02",
            )
            with self.assertRaisesRegex(ContractError, "overlapping tensor ranges"):
                validate_cache(document, temp)

    def test_fixture_stop_identity_and_trusted_read_only_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory) / "stop-config-eos-missing"
            shutil.copytree(FIXTURE_CACHE, temp)
            tokenizer_config = json.loads((temp / "tokenizer_config.json").read_text(encoding="utf-8"))
            del tokenizer_config["eos_token"]
            (temp / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            for variant in ("same-id-other-content", "same-content-other-id", "duplicate-correct", "missing"):
                temp = Path(directory) / f"stop-{variant}"
                shutil.copytree(FIXTURE_CACHE, temp)
                document = self._fixture_document_with_current_hashes(temp)
                identity = document["model"]["tokenizer_contract"]["stop_identity"]["config_eos"]
                tokenizer = json.loads((temp / "tokenizer.json").read_text(encoding="utf-8"))
                correct = next(token for token in tokenizer["added_tokens"] if token["id"] == identity["token_id"])
                if variant == "same-id-other-content":
                    changed = copy.deepcopy(correct)
                    changed["content"] = "<|other-content|>"
                    tokenizer["added_tokens"].append(changed)
                elif variant == "same-content-other-id":
                    changed = copy.deepcopy(correct)
                    changed["id"] = identity["token_id"] + 1
                    tokenizer["added_tokens"].append(changed)
                elif variant == "duplicate-correct":
                    tokenizer["added_tokens"].append(copy.deepcopy(correct))
                else:
                    tokenizer["added_tokens"] = [
                        token for token in tokenizer["added_tokens"] if token["id"] != identity["token_id"]
                    ]
                (temp / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
                document = self._fixture_document_with_current_hashes(temp)
                with self.assertRaises(ContractError):
                    validate_cache(document, temp)

            temp = Path(directory) / "stop-swapped"
            shutil.copytree(FIXTURE_CACHE, temp)
            tokenizer = json.loads((temp / "tokenizer.json").read_text(encoding="utf-8"))
            tokenizer["added_tokens"][0]["id"] = 1
            (temp / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
            document = self._fixture_document_with_current_hashes(temp)
            with self.assertRaises(ContractError):
                validate_cache(document, temp)

            temp = Path(directory) / "trusted"
            shutil.copytree(FIXTURE_CACHE, temp)
            temp.chmod(0o500)
            for path in temp.iterdir():
                path.chmod(0o400)
            with mock.patch("validate_model_lock._mount_is_read_only", return_value=True):
                validate_lock_file(
                    FIXTURE_LOCK,
                    schema_path=SCHEMA_PATH,
                    cache_dir=temp,
                    require_trusted_read_only=True,
                )

            temp = Path(directory) / "writable-file"
            shutil.copytree(FIXTURE_CACHE, temp)
            temp.chmod(0o500)
            for path in temp.iterdir():
                path.chmod(0o400)
            (temp / "config.json").chmod(0o600)
            with mock.patch("validate_model_lock._mount_is_read_only", return_value=True):
                with self.assertRaises(ContractError):
                    validate_lock_file(
                        FIXTURE_LOCK,
                        schema_path=SCHEMA_PATH,
                        cache_dir=temp,
                        require_trusted_read_only=True,
                    )

            temp = Path(directory) / "nonregular"
            shutil.copytree(FIXTURE_CACHE, temp)
            (temp / "config.json").unlink()
            os.mkfifo(temp / "config.json")
            with self.assertRaises(ContractError):
                validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

            temp = Path(directory) / "hardlink"
            shutil.copytree(FIXTURE_CACHE, temp)
            external = Path(directory) / "external-config.json"
            shutil.copy2(temp / "config.json", external)
            (temp / "config.json").unlink()
            os.link(external, temp / "config.json")
            temp.chmod(0o500)
            for path in temp.iterdir():
                path.chmod(0o400)
            with mock.patch("validate_model_lock._mount_is_read_only", return_value=True):
                with self.assertRaises(ContractError):
                    validate_lock_file(
                        FIXTURE_LOCK,
                        schema_path=SCHEMA_PATH,
                        cache_dir=temp,
                        require_trusted_read_only=True,
                    )

            temp = Path(directory) / "path-replacement"
            shutil.copytree(FIXTURE_CACHE, temp)
            replacement = Path(directory) / "replacement-config.json"
            shutil.copy2(temp / "config.json", replacement)
            original_open = __import__("validate_model_lock")._open_relative_read_only
            replaced = False

            def replace_before_open(cache: Path, relative: str) -> int:
                nonlocal replaced
                if relative == "config.json" and not replaced:
                    replaced = True
                    (cache / relative).unlink()
                    os.replace(replacement, cache / relative)
                return original_open(cache, relative)

            with mock.patch("validate_model_lock._open_relative_read_only", side_effect=replace_before_open):
                with self.assertRaises(ContractError):
                    validate_lock_file(FIXTURE_LOCK, schema_path=SCHEMA_PATH, cache_dir=temp)

    def test_semantic_reads_bind_hashed_descriptors_and_reject_same_size_replacements(self) -> None:
        module = __import__("validate_model_lock")
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ("config.json", "json"),
                ("model.safetensors.index.json", "json"),
                ("model.safetensors", "header"),
            )
            for serial, (relative, kind) in enumerate(cases):
                temp = Path(directory) / f"same-size-replacement-{serial}"
                shutil.copytree(FIXTURE_CACHE, temp)
                document = self._fixture_document_with_current_hashes(temp)
                original = (temp / relative).read_bytes()
                replacement = Path(directory) / f"replacement-{serial}"
                if kind == "header":
                    changed = original.replace(b'"pt"', b'"xx"', 1)
                else:
                    changed = original.replace(b"\n", b" ", 1)
                self.assertEqual(len(changed), len(original))
                replacement.write_bytes(changed)
                replaced = False

                if kind == "json":
                    original_read = module._read_cache_json

                    def replace_json(files, path):
                        nonlocal replaced
                        if path == relative and not replaced:
                            replaced = True
                            hashed = files[path]
                            self.assertEqual(os.pread(hashed.file_descriptor, hashed.size_bytes, 0), original)
                            os.replace(replacement, temp / path)
                        return original_read(files, path)

                    with mock.patch.object(module, "_read_cache_json", side_effect=replace_json):
                        with self.assertRaises(ContractError):
                            validate_cache(document, temp)
                else:
                    original_read = module._read_safetensors_header

                    def replace_header(files, path):
                        nonlocal replaced
                        if path == relative and not replaced:
                            replaced = True
                            hashed = files[path]
                            self.assertEqual(os.pread(hashed.file_descriptor, hashed.size_bytes, 0), original)
                            os.replace(replacement, temp / path)
                        return original_read(files, path)

                    with mock.patch.object(module, "_read_safetensors_header", side_effect=replace_header):
                        with self.assertRaises(ContractError):
                            validate_cache(document, temp)
                self.assertTrue(replaced)

    def test_semantic_reads_reject_same_inode_same_size_in_place_mutations(self) -> None:
        module = __import__("validate_model_lock")
        with tempfile.TemporaryDirectory() as directory:
            cases = (
                ("config.json", "json", b'"FixtureModel"', b'"FixtureModex"'),
                ("model.safetensors.index.json", "json", b'"model.safetensors"', b'"other.safetensors"'),
                ("model.safetensors", "header", b'"shape":[2]', b'"shape":[3]'),
            )
            for serial, (relative, kind, needle, replacement) in enumerate(cases):
                temp = Path(directory) / f"same-inode-in-place-{serial}"
                shutil.copytree(FIXTURE_CACHE, temp)
                document = self._fixture_document_with_current_hashes(temp)
                original = (temp / relative).read_bytes()
                self.assertEqual(original.count(needle), 1)
                offset = original.index(needle)
                mutated = False

                if kind == "json":
                    original_read = module._read_cache_json

                    def mutate_json(files, path):
                        nonlocal mutated
                        if path == relative and not mutated:
                            mutated = True
                            self._mutate_hashed_file_in_place(
                                files[path].file_descriptor,
                                temp / path,
                                original,
                                offset=offset,
                                replacement=replacement,
                            )
                        return original_read(files, path)

                    with mock.patch.object(module, "_read_cache_json", side_effect=mutate_json):
                        with self.assertRaises(ContractError):
                            validate_cache(document, temp)
                else:
                    original_read = module._read_safetensors_header

                    def mutate_header(files, path):
                        nonlocal mutated
                        if path == relative and not mutated:
                            mutated = True
                            self._mutate_hashed_file_in_place(
                                files[path].file_descriptor,
                                temp / path,
                                original,
                                offset=offset,
                                replacement=replacement,
                            )
                        return original_read(files, path)

                    with mock.patch.object(module, "_read_safetensors_header", side_effect=mutate_header):
                        with self.assertRaises(ContractError):
                            validate_cache(document, temp)
                self.assertTrue(mutated)

    def test_repeated_semantic_failures_do_not_leak_file_descriptors(self) -> None:
        module = __import__("validate_model_lock")
        repeats = 32
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory) / "semantic-failure-fd-cleanup"
            shutil.copytree(FIXTURE_CACHE, temp)
            document = self._fixture_document_with_current_hashes(temp)
            baseline = self._proc_fd_count()

            def fail_semantic(*args, **kwargs):
                raise ContractError("forced semantic validation failure")

            with mock.patch.object(module, "_validate_stop_identity", side_effect=fail_semantic) as failure:
                for _ in range(repeats):
                    with self.assertRaisesRegex(ContractError, "forced semantic validation failure"):
                        validate_cache(document, temp)
                    self.assertEqual(self._proc_fd_count(), baseline)
            self.assertEqual(failure.call_count, repeats)

    def test_model_lock_matrix_exception_is_exact_and_python_is_first(self) -> None:
        self.assertTrue(command_is_model_lock_contract(["{python}", "ci/tests/test_model_lock_contracts.py"]))
        self.assertFalse(command_is_model_lock_contract(["{python}", "ci/tests/test_model_lock_contracts.py", "--bypass"]))
        self.assertFalse(command_is_model_lock_contract(["sh", "ci/tests/test_model_lock_contracts.py"]))
        self.assertFalse(command_is_model_lock_contract(["ci/tests/test_model_lock_contracts.py", "{python}"]))

    def test_lock_and_fixture_json_have_no_duplicate_keys(self) -> None:
        self.assertEqual(read_json(LOCK_PATH)["fingerprint"], self.lock["fingerprint"])
        self.assertEqual(read_json(FIXTURE_LOCK)["schema_version"], "model-lock-v1")


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("SLLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "SLLM_UNITTEST_COUNTS="
            + json.dumps(
                {
                    "collected": selected,
                    "selected": selected,
                    "passed": selected - failed - skipped,
                    "failed": failed,
                    "skipped": skipped,
                    "deselected": 0,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
