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
PENDING_MANIFEST_SHA256 = (
    "c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af"
)
COMPLETED_MANIFEST_SHA256 = (
    "3b6362fd472debbbfb30fb5616325703dd52e90e82319280490a0d84fcd6bf83"
)
PERSISTENT_REAL_ORACLE = Path(
    "/home/homelab1/datapool/ullm/product/qwen3-14b-fp8-sq8-v0.1/"
    "oracles/vllm-source-v0.1"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pending_fixture() -> Path:
    root_manifest = FIXTURE / "manifest.json"
    if root_manifest.is_file() and sha256_file(root_manifest) == PENDING_MANIFEST_SHA256:
        return FIXTURE
    nested = FIXTURE / "provenance/bootstrap-input-v0.1"
    if (
        (nested / "manifest.json").is_file()
        and sha256_file(nested / "manifest.json") == PENDING_MANIFEST_SHA256
    ):
        return nested
    raise RuntimeError("trusted pending serving fixture is absent")


def real_oracle_fixture() -> Path | None:
    nested = FIXTURE / "oracles/vllm-source-v0.1"
    if nested.is_dir():
        return nested
    if PERSISTENT_REAL_ORACLE.is_dir():
        return PERSISTENT_REAL_ORACLE
    return None


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def rebuild_sums(root: Path) -> None:
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() != "SHA256SUMS"
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
        cls.pending_fixture = pending_fixture()
        cls.real_oracle = real_oracle_fixture()

    def run_export(
        self,
        output: Path,
        *,
        bootstrap: Path | None = None,
        real_oracle: Path | None = None,
        chat_template: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(EXPORTER), "--output-dir", str(output)]
        if chat_template is not None:
            command.extend(["--chat-template-dir", str(chat_template)])
        if bootstrap is not None or real_oracle is not None:
            if bootstrap is None or real_oracle is None:
                raise ValueError("completed export requires both sources")
            command.extend(
                [
                    "--bootstrap-fixture-dir",
                    str(bootstrap),
                    "--real-oracle-dir",
                    str(real_oracle),
                ]
            )
        return subprocess.run(
            command,
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
        shutil.copytree(self.pending_fixture, destination, symlinks=True)
        return destination

    def export_completed_or_skip(self, output: Path) -> None:
        if self.real_oracle is None:
            self.skipTest("real vLLM oracle source is unavailable")
        result = self.run_export(
            output,
            bootstrap=self.pending_fixture,
            real_oracle=self.real_oracle,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("oracles_complete=6 run_count=21", result.stdout)

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
        result = self.run_validator(self.pending_fixture)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("valid=true oracle_status=pending promotion_eligible=false", result.stdout)
        self.assertIn("mode=promotion trusted=true prompts=6", result.stdout)

    def test_trusted_completed_fixture_is_promotion_eligible(self) -> None:
        self.assertEqual(sha256_file(FIXTURE / "manifest.json"), COMPLETED_MANIFEST_SHA256)
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("oracle_status=complete promotion_eligible=true", result.stdout)
        self.assertIn("mode=promotion trusted=true prompts=6", result.stdout)

    def test_contract_only_mode_is_explicitly_untrusted(self) -> None:
        result = self.run_validator(self.pending_fixture, contract_only=True)
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
            self.assertEqual(tree_bytes(first), tree_bytes(self.pending_fixture))

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

    def test_export_refuses_symlinked_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            real_parent = parent / "real-parent"
            real_parent.mkdir()
            linked_parent = parent / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            result = self.run_export(linked_parent / "new-parent" / "fixture")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output parent must not contain a symlink", result.stderr)
            self.assertFalse((real_parent / "new-parent").exists())

    def test_pending_export_rejects_chat_leaf_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            chat = parent / "chat-template"
            shutil.copytree(self.pending_fixture / "chat-template", chat)
            leaf = chat / "fixtures/english-user.json"
            leaf.write_bytes(leaf.read_bytes() + b"\n")
            output = parent / "fixture"
            result = self.run_export(output, chat_template=chat)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "pending staged fixture manifest differs from the fixed trust anchor",
                result.stderr,
            )
            self.assertFalse(output.exists())

    def test_pending_export_rejects_output_inside_chat_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            chat = Path(temporary) / "chat-template"
            shutil.copytree(self.pending_fixture / "chat-template", chat)
            output = chat / "new-parent" / "nested-output"
            result = self.run_export(output, chat_template=chat)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(
                "output directory must not overlap chat-template fixture source",
                result.stderr,
            )
            self.assertFalse((chat / "new-parent").exists())

    def test_completed_export_is_deterministic_and_contract_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            first = parent / "first"
            second = parent / "second"
            self.export_completed_or_skip(first)
            self.export_completed_or_skip(second)

            def tree_bytes(root: Path) -> list[tuple[str, bytes]]:
                return [
                    (path.relative_to(root).as_posix(), path.read_bytes())
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                ]

            self.assertEqual(tree_bytes(first), tree_bytes(second))
            self.assertEqual(tree_bytes(first), tree_bytes(FIXTURE))
            validated = self.run_validator(first, contract_only=True)
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn("oracle_status=complete", validated.stdout)
            self.assertIn("promotion_eligible=false", validated.stdout)
            self.assertIn("mode=contract-only trusted=false", validated.stdout)

            manifest = json.loads((first / "manifest.json").read_text(encoding="ascii"))
            self.assertEqual(
                manifest["status"], "input_contract_ready_real_oracles_complete"
            )
            self.assertNotIn("oracle_placeholders", manifest)
            self.assertTrue(manifest["trust"]["promotion_eligible"])
            self.assertEqual(list((first / "oracles").glob("*.pending.json")), [])
            self.assertEqual(
                len(
                    list(
                        (first / "provenance/bootstrap-input-v0.1/oracles").glob(
                            "*.pending.json"
                        )
                    )
                ),
                6,
            )
            for relative in ("raw", "chat-template", "openwebui"):
                active = first / relative
                preserved = first / "provenance/bootstrap-input-v0.1" / relative
                self.assertEqual(tree_bytes(active), tree_bytes(preserved))
            sums = (first / "SHA256SUMS").read_text(encoding="ascii")
            self.assertIn(
                "provenance/bootstrap-input-v0.1/SHA256SUMS", sums
            )
            self.assertIn("oracles/vllm-source-v0.1/SHA256SUMS", sums)

    def test_completed_export_refuses_existing_output_before_source_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("unchanged", encoding="ascii")
            result = self.run_export(
                output,
                bootstrap=Path(temporary) / "missing-bootstrap",
                real_oracle=Path(temporary) / "missing-oracle",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertEqual(marker.read_text(encoding="ascii"), "unchanged")

    def test_invalid_completed_source_does_not_create_output_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            output = parent / "new-parent" / "fixture"
            result = self.run_export(
                output,
                bootstrap=parent / "missing-bootstrap",
                real_oracle=parent / "missing-oracle",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bootstrap fixture source is unavailable", result.stderr)
            self.assertFalse((parent / "new-parent").exists())

    def test_completed_nested_bootstrap_sums_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            self.export_completed_or_skip(root)
            path = root / "provenance/bootstrap-input-v0.1/SHA256SUMS"
            raw = path.read_text(encoding="ascii")
            path.write_text("0" * 64 + raw[64:], encoding="ascii")
            rehash_artifacts_and_sums(root)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA256SUMS digest mismatch", result.stderr)

    def test_completed_real_oracle_payload_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            self.export_completed_or_skip(root)
            path = root / "oracles/vllm-source-v0.1/prompts/raw-p0001/final-hidden.f32le"
            payload = bytearray(path.read_bytes())
            payload[0] ^= 1
            path.write_bytes(payload)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("independent real oracle validation failed", result.stderr)

    def test_completed_producer_passed_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            self.export_completed_or_skip(root)
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
            manifest["producer_passed"] = True
            write_json(manifest_path, manifest)
            rebuild_sums(root)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manifest keys differ", result.stderr)

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

    def test_self_consistent_chat_template_manifest_tamper_is_rejected(self) -> None:
        self.assert_artifact_rejected(
            "chat-template/manifest.json",
            lambda value: value.__setitem__("passed", True),
            "chat-template manifest differs from its independent trust anchor",
        )

    def test_self_consistent_chat_template_leaf_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self.copied_fixture(Path(temporary))
            path = root / "chat-template/fixtures/english-user.json"
            path.write_bytes(path.read_bytes() + b"\n")
            rehash_artifacts_and_sums(root)
            result = self.run_validator(root, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("differs from the fixture", result.stderr)

    def test_openwebui_capture_identity_and_requests_are_exact(self) -> None:
        capture = json.loads(
            (self.pending_fixture / "openwebui/capture.json").read_text(
                encoding="ascii"
            )
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
                (self.pending_fixture / "openwebui" / name).read_text(
                    encoding="ascii"
                )
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
        manifest = json.loads(
            (self.pending_fixture / "manifest.json").read_text(encoding="ascii")
        )
        self.assertFalse(manifest["trust"]["promotion_eligible"])
        self.assertTrue(manifest["trust"]["synthetic_oracle_values_forbidden"])
        for record in manifest["oracle_placeholders"]:
            value = json.loads(
                (self.pending_fixture / record["placeholder_file"]).read_text(
                    encoding="ascii"
                )
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
