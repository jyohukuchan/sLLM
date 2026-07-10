import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER = REPO_ROOT / "tools" / "export-qwen3-vllm-generation-oracle.py"
VALIDATOR = REPO_ROOT / "tools" / "validate-qwen3-vllm-generation-oracle.py"
FIXTURE = Path("/tmp/ullm-qwen3-14b-fp8-vllm-generation-m8-g8-v0.1")


class Qwen3VllmGenerationOracleToolTests(unittest.TestCase):
    @staticmethod
    def load_module(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"failed to load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    def run_validator(
        self, oracle: Path, *, contract_only: bool = False
    ) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(VALIDATOR)]
        if contract_only:
            command.append("--contract-only")
        command.append(str(oracle))
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_exporter_refuses_existing_output(self) -> None:
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
            root = Path(temporary)
            output = root / "oracle"
            os.symlink(root / "missing", output)
            result = self.run_exporter(output)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertTrue(output.is_symlink())

    def test_atomic_publish_does_not_replace_raced_destination(self) -> None:
        module = self.load_module("generation_exporter_publish", EXPORTER)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "source").write_text("source", encoding="ascii")
            (destination / "destination").write_text("destination", encoding="ascii")
            with self.assertRaises(FileExistsError):
                module.rename_noreplace(source, destination)
            self.assertTrue((source / "source").is_file())
            self.assertTrue((destination / "destination").is_file())

    def test_revision_set_and_toctou_helpers_fail_closed(self) -> None:
        module = self.load_module("generation_exporter_contract", EXPORTER)
        revision = {
            "revision": module.EXPECTED_REVISION,
            "revision_consistent": True,
            "per_file_revisions": {
                name: module.EXPECTED_REVISION
                for name in module.EXPECTED_REVISION_FILES
            },
        }
        module.validate_revision_contract(revision)
        revision["per_file_revisions"]["unexpected.json"] = module.EXPECTED_REVISION
        with self.assertRaises(SystemExit):
            module.validate_revision_contract(revision)
        with self.assertRaises(RuntimeError):
            module.require_unchanged("checkpoint", {"sha": "before"}, {"sha": "after"})

    def test_rerun_uses_startup_exporter_bytes(self) -> None:
        module = self.load_module("generation_exporter_capture", EXPORTER)
        captured = b"#!/usr/bin/env python3\n# startup bytes\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module.write_rerun_files(
                root,
                root / "oracle",
                Path("/fixed/model"),
                captured,
                module.sha256_bytes(captured),
            )
            self.assertEqual(
                (root / "export_generation_oracle.py").read_bytes(), captured
            )

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_real_fixture_passes_trusted_validation(self) -> None:
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed=true mode=promotion trusted=true", result.stdout)
        self.assertIn("steps=8 feedback_edges=7 artifacts=18", result.stdout)
        self.assertIn("generated_token_ids=353,10,4999,1725,15,16,17,18", result.stdout)
        self.assertIn(
            "metadata_sha256=5fc03a28cd15409e84a7fd23fd51c0cbd6ec9cf8761a66d1f5ede7ddfe3226a0",
            result.stdout,
        )

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_real_fixture_is_untrusted_in_contract_only_mode(self) -> None:
        result = self.run_validator(FIXTURE, contract_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode=contract-only trusted=false", result.stdout)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_self_consistent_logits_tamper_is_contract_only(self) -> None:
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            logits_path = copied / "steps" / "step-00-logits.f32"
            logits = np.fromfile(logits_path, dtype="<f4")
            changed_token = 100000
            self.assertNotIn(
                changed_token,
                [entry["token_id"] for entry in metadata["steps"][0]["top_10"]],
            )
            logits[changed_token] = np.nextafter(
                logits[changed_token], np.float32(np.inf), dtype=np.float32
            )
            logits.astype("<f4", copy=False).tofile(logits_path)
            digest = hashlib.sha256(logits_path.read_bytes()).hexdigest()
            tensor = torch.from_numpy(logits)
            finite = torch.isfinite(tensor)
            finite_values = tensor[finite]
            health = {
                "elements": int(tensor.numel()),
                "finite_count": int(finite.sum().item()),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std_population": float(finite_values.std(unbiased=False).item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
                "max_abs": float(finite_values.abs().max().item()),
            }
            metadata["steps"][0]["logits"]["sha256"] = digest
            metadata["steps"][0]["logits"]["health"] = health
            for artifact in metadata["artifact_files_excluding_metadata"]:
                if artifact["file"] == "steps/step-00-logits.f32":
                    artifact["sha256"] = digest
                    break
            else:
                self.fail("step-00 logits is absent from the manifest")
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            promoted = self.run_validator(copied)
            self.assertNotEqual(promoted.returncode, 0)
            self.assertIn("promotion trust anchor", promoted.stderr)
            contract_only = self.run_validator(copied, contract_only=True)
            self.assertEqual(contract_only.returncode, 0, contract_only.stderr)
            self.assertIn("mode=contract-only trusted=false", contract_only.stdout)

    def test_topk_tie_breaks_by_ascending_token_id(self) -> None:
        import numpy as np

        module = self.load_module("generation_validator_tie", VALIDATOR)
        with tempfile.TemporaryDirectory() as temporary:
            logits_path = Path(temporary) / "logits.f32"
            logits = np.zeros(module.VOCAB_SIZE, dtype="<f4")
            logits[11] = 7.0
            logits[3] = 7.0
            logits.tofile(logits_path)
            top_10 = module.recompute_top_10(logits_path)
            self.assertEqual(
                [entry["token_id"] for entry in top_10[:2]], [3, 11]
            )

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_feedback_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["steps"][3]["input_token_id"] = 0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("does not implement token feedback", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_step_position_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["steps"][4]["input_position_id"] = 99
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input_position_id is invalid", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_payload_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            payload = copied / "steps" / "step-07-final-hidden.f32"
            with payload.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 1]))
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact SHA-256 mismatch", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_health_metadata_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["steps"][0]["final_hidden"]["health"]["l2"] += 1.0
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("health.l2 does not match the payload", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_source_revision_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["source"]["revision"]["revision"] = "0" * 40
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source revision set differs", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local eight-step generation fixture is absent")
    def test_extra_schema_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["untrusted_note"] = "not in schema"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            result = self.run_validator(copied, contract_only=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("metadata keys differ", result.stderr)


if __name__ == "__main__":
    unittest.main()
