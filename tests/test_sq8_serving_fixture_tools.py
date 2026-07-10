import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = REPO_ROOT / "tools" / "export-sq8-serving-fixtures.py"
VALIDATOR = REPO_ROOT / "tools" / "validate-sq8-serving-fixtures.py"
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "sq8-serving-v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def rebuild_sums(root: Path) -> None:
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="ascii",
    )


def rehash_artifacts_and_sums(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    for record in manifest["artifact_files_excluding_manifest_and_sums"]:
        path = root / record["file"]
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    write_json(manifest_path, manifest)
    rebuild_sums(root)


class Sq8ServingFixtureToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not FIXTURE.is_dir():
            raise RuntimeError(f"fixed serving fixture is absent: {FIXTURE}")

    def run_export(self, output: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(EXPORTER), "--output-dir", str(output)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def run_validator(
        self, fixture: Path, *, contract_only: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR)]
        if contract_only:
            command.append("--contract-only")
        command.append(str(fixture))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def copied_fixture(self, parent: Path) -> Path:
        destination = parent / "fixture"
        shutil.copytree(FIXTURE, destination, symlinks=True)
        return destination

    def assert_manifest_rejected(self, mutate, expected_error: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            mutate(manifest)
            write_json(manifest_path, manifest)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)

    def assert_artifact_rejected(
        self, relative: str, mutate, expected_error: str
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / relative
            value = json.loads(path.read_text(encoding="ascii"))
            mutate(value)
            write_json(path, value)
            rehash_artifacts_and_sums(root)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected_error, result.stderr)

    def test_trusted_fixture_passes_without_claiming_oracle_completion(self) -> None:
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid=true oracle_status=pending promotion_eligible=false", result.stdout)
        self.assertIn("mode=promotion trusted=true prompts=6", result.stdout)

    def test_contract_only_mode_is_explicitly_untrusted(self) -> None:
        result = self.run_validator(FIXTURE, contract_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=contract-only trusted=false", result.stdout)

    def test_export_is_deterministic_and_matches_checked_in_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = parent / "first"
            second = parent / "second"
            for output in (first, second):
                result = self.run_export(output)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("oracles_pending=6", result.stdout)

            def tree_bytes(root: Path) -> list[tuple[str, bytes]]:
                return [
                    (path.relative_to(root).as_posix(), path.read_bytes())
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                ]

            self.assertEqual(tree_bytes(first), tree_bytes(second))
            self.assertEqual(tree_bytes(first), tree_bytes(FIXTURE))

    def test_export_refuses_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            output.mkdir()
            result = self.run_export(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)

    def test_export_refuses_dangling_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            output.symlink_to("absent-target", target_is_directory=True)
            result = self.run_export(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertFalse((Path(temporary) / "absent-target").exists())

    def test_manifest_anchor_detects_a_self_consistent_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            manifest_path = root / "manifest.json"
            manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
            rebuild_sums(root)

            promotion = self.run_validator(root)
            self.assertNotEqual(promotion.returncode, 0)
            self.assertIn("promotion trust anchor", promotion.stderr)

            contract = self.run_validator(root, contract_only=True)
            self.assertEqual(contract.returncode, 0, contract.stderr)
            self.assertIn("trusted=false", contract.stdout)

    def test_duplicate_manifest_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / "manifest.json"
            raw = path.read_text(encoding="ascii")
            needle = '  "schema_version": "ullm.sq8.serving_fixtures.v1",'
            self.assertIn(needle, raw)
            path.write_text(raw.replace(needle, needle + "\n" + needle, 1), encoding="ascii")
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate JSON key: schema_version", result.stderr)

    def test_nonfinite_manifest_number_is_rejected(self) -> None:
        self.assert_manifest_rejected(
            lambda manifest: manifest["comparison_contract"][
                "vllm_source_model_gate"
            ].__setitem__("max_relative_l2", float("nan")),
            "non-finite JSON number is forbidden: NaN",
        )

    def test_extra_manifest_key_is_rejected(self) -> None:
        self.assert_manifest_rejected(
            lambda manifest: manifest.__setitem__("producer_passed", True),
            "manifest keys differ",
        )

    def test_boolean_cannot_replace_prompt_length(self) -> None:
        self.assert_manifest_rejected(
            lambda manifest: manifest["raw_prompts"][0].__setitem__(
                "prompt_tokens", True
            ),
            "raw_prompts[0].prompt_tokens type differs",
        )

    def test_model_package_and_tokenizer_identity_drift_is_rejected(self) -> None:
        cases = (
            (
                lambda manifest: manifest["source_identity"].__setitem__(
                    "revision", "0" * 40
                ),
                "source_identity.revision differs",
            ),
            (
                lambda manifest: manifest["source_identity"].__setitem__(
                    "package_manifest_sha256", "0" * 64
                ),
                "source_identity.package_manifest_sha256 differs",
            ),
            (
                lambda manifest: manifest["tokenizer_identity"].__setitem__(
                    "chat_template_sha256", "0" * 64
                ),
                "tokenizer_identity.chat_template_sha256 differs",
            ),
        )
        for mutate, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                self.assert_manifest_rejected(mutate, expected_error)

    def test_vllm_environment_drift_is_rejected(self) -> None:
        for key, replacement in (
            ("package_version", "0.0.0"),
            ("dtype", "float16"),
            ("enforce_eager", False),
        ):
            with self.subTest(key=key):
                self.assert_manifest_rejected(
                    lambda manifest, key=key, replacement=replacement: manifest[
                        "vllm_identity"
                    ].__setitem__(key, replacement),
                    f"vllm_identity.{key} differs",
                )

    def test_generation_and_comparison_contract_drift_is_rejected(self) -> None:
        cases = (
            (
                lambda manifest: manifest["generation_cases"][2].__setitem__(
                    "max_new_tokens", 63
                ),
                "generation_cases[2].max_new_tokens differs",
            ),
            (
                lambda manifest: manifest["comparison_contract"][
                    "ullm_path_equivalence_gate"
                ].__setitem__("min_cosine_similarity", 0.99),
                "comparison_contract.ullm_path_equivalence_gate.min_cosine_similarity differs",
            ),
        )
        for mutate, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                self.assert_manifest_rejected(mutate, expected_error)

    def test_raw_prompt_values_are_recomputed_after_hashes_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / "raw/prompt-0008.u32le"
            payload = bytearray(path.read_bytes())
            struct.pack_into("<I", payload, 0, 2)
            path.write_bytes(payload)
            rehash_artifacts_and_sums(root)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from ascending contract", result.stderr)

    def test_fabricated_oracle_payload_is_rejected(self) -> None:
        self.assert_artifact_rejected(
            "oracles/raw-p0032.pending.json",
            lambda value: value["requested_outputs"]["prefill_logits"].__setitem__(
                "file", "oracles/fabricated-logits.f32le"
            ),
            "prefill_logits.file type differs",
        )

    def test_exporter_success_claim_in_placeholder_is_rejected(self) -> None:
        self.assert_artifact_rejected(
            "oracles/raw-p0001.pending.json",
            lambda value: value.__setitem__("passed", True),
            "vllm-raw-p0001 keys differ",
        )

    def test_fabricated_chat_template_case_is_rejected(self) -> None:
        self.assert_artifact_rejected(
            "chat-template.pending.json",
            lambda value: value["cases"].append(
                {"prompt_tokens": 32, "token_ids": [1] * 32}
            ),
            "chat_template_placeholder_payload.cases length differs",
        )

    def test_openwebui_capture_identity_and_requests_are_exact(self) -> None:
        capture = json.loads(
            (FIXTURE / "openwebui/capture.json").read_text(encoding="ascii")
        )
        self.assertEqual(capture["identity"]["version"], "v0.9.4")
        self.assertEqual(
            capture["identity"]["image_digest"],
            "sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff",
        )
        self.assertTrue(capture["trust"]["captured_via_actual_proxy"])
        self.assertFalse(capture["trust"]["sq8_numeric_oracle"])

        for name in ("stream-request.json", "nonstream-request.json"):
            request = json.loads(
                (FIXTURE / "openwebui" / name).read_text(encoding="ascii")
            )
            self.assertEqual(request["model"], "ullm-qwen3-14b-sq8")
            self.assertIn("max_tokens", request)
            self.assertNotIn("max_completion_tokens", request)
            self.assertNotIn("metadata", request)
            self.assertNotIn("authorization", request)
            self.assertNotIn("cookie", request)

    def test_openwebui_request_drift_is_rejected_after_rehash(self) -> None:
        self.assert_artifact_rejected(
            "openwebui/stream-request.json",
            lambda value: value.__setitem__("max_tokens", 65),
            "openwebui_stream_request.max_tokens differs",
        )

    def test_extra_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            (root / "unexpected.json").write_text("{}\n", encoding="ascii")
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("fixture file set differs", result.stderr)

    def test_symlink_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / "raw/prompt-0001.u32le"
            path.unlink()
            path.symlink_to("prompt-0008.u32le")
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("regular non-symlink file", result.stderr)

    def test_sha256sums_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / "SHA256SUMS"
            raw = path.read_text(encoding="ascii")
            path.write_text("0" * 64 + raw[64:], encoding="ascii")
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA256SUMS digest mismatch", result.stderr)

    def test_pending_fixture_contains_no_numerical_oracle_payloads(self) -> None:
        manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="ascii"))
        self.assertFalse(manifest["trust"]["promotion_eligible"])
        self.assertTrue(manifest["trust"]["synthetic_oracle_values_forbidden"])
        for record in manifest["oracle_placeholders"]:
            value = json.loads(
                (FIXTURE / record["placeholder_file"]).read_text(encoding="ascii")
            )
            self.assertEqual(value["status"], "pending_real_vllm_export")
            for tensor_name in ("prefill_final_hidden", "prefill_logits"):
                tensor = value["requested_outputs"][tensor_name]
                self.assertIsNone(tensor["file"])
                self.assertIsNone(tensor["bytes"])
                self.assertIsNone(tensor["sha256"])
            for generation in value["requested_outputs"]["greedy_generation"]:
                self.assertIsNone(generation["token_file"])
                self.assertIsNone(generation["generated_tokens"])
                self.assertIsNone(generation["token_ids_u32_le_sha256"])


if __name__ == "__main__":
    unittest.main()
