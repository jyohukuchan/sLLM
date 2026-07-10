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
EXPORTER = REPO_ROOT / "tools" / "export-qwen3-vllm-oracle.py"
VALIDATOR = REPO_ROOT / "tools" / "validate-qwen3-vllm-oracle.py"
FIXTURE = Path("/tmp/ullm-qwen3-14b-fp8-vllm-oracle-m8-v0.1")


class Qwen3VllmOracleToolTests(unittest.TestCase):
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

    def test_exporter_refuses_existing_directory_before_heavyweight_preflight(self) -> None:
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
            os.symlink(root / "missing-target", output)

            result = self.run_exporter(output)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite existing output", result.stderr)
            self.assertTrue(output.is_symlink())

    def test_atomic_publish_does_not_replace_raced_destination(self) -> None:
        module = self.load_module("qwen3_oracle_exporter", EXPORTER)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "source-marker").write_text("source", encoding="ascii")
            (destination / "destination-marker").write_text("destination", encoding="ascii")

            with self.assertRaises(FileExistsError):
                module.rename_noreplace(source, destination)

            self.assertEqual(
                (source / "source-marker").read_text(encoding="ascii"), "source"
            )
            self.assertEqual(
                (destination / "destination-marker").read_text(encoding="ascii"),
                "destination",
            )

    def test_exporter_rejects_revision_set_mismatch(self) -> None:
        module = self.load_module("qwen3_oracle_revision", EXPORTER)
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

    def test_exporter_toctou_helper_rejects_changed_snapshot(self) -> None:
        module = self.load_module("qwen3_oracle_toctou", EXPORTER)
        module.require_unchanged("test snapshot", {"sha256": "a"}, {"sha256": "a"})
        with self.assertRaises(RuntimeError):
            module.require_unchanged(
                "test snapshot", {"sha256": "before"}, {"sha256": "after"}
            )

    def test_rerun_artifact_uses_captured_exporter_bytes(self) -> None:
        module = self.load_module("qwen3_oracle_script_capture", EXPORTER)
        captured = b"#!/usr/bin/env python3\n# captured before inference\n"
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            module.write_rerun_files(
                work_dir,
                work_dir / "oracle",
                Path("/fixed/model"),
                captured,
                module.sha256_bytes(captured),
            )
            self.assertEqual((work_dir / "export_oracle.py").read_bytes(), captured)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_accepts_real_fixture(self) -> None:
        result = self.run_validator(FIXTURE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed=true mode=promotion trusted=true", result.stdout)
        self.assertIn("layers=40 positions=8", result.stdout)
        self.assertIn("sample_token_id=353", result.stdout)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_marks_contract_only_fixture_untrusted(self) -> None:
        result = self.run_validator(FIXTURE, contract_only=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed=true mode=contract-only trusted=false", result.stdout)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_rejects_payload_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            payload = copied / "layers" / "layer-00-output.f32"
            with payload.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes([first[0] ^ 0x01]))

            result = self.run_validator(copied)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("artifact SHA-256 mismatch", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_rejects_sampler_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["sampler_cross_check"]["generated_token_ids"] = [0]
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            result = self.run_validator(copied, contract_only=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("sampler output does not match", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_rejects_shape_metadata_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["oracle"]["layers"][0]["shape"] = [8, 5119]
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            result = self.run_validator(copied, contract_only=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("identity, shape, dtype, or semantic is invalid", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_validator_rejects_health_metadata_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["oracle"]["layers"][0]["health"]["l2"] += 1.0
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            result = self.run_validator(copied, contract_only=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("health.l2 does not match the payload", result.stderr)

    @unittest.skipUnless(FIXTURE.is_dir(), "local vLLM M=8 oracle fixture is absent")
    def test_self_consistent_logits_tamper_is_contract_only(self) -> None:
        import hashlib

        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "oracle"
            shutil.copytree(FIXTURE, copied)
            metadata_path = copied / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            logits_path = copied / "logits.f32"
            logits = np.fromfile(logits_path, dtype="<f4")
            matrix = logits.reshape(8, 151936)
            changed_token = 100000
            self.assertNotIn(
                changed_token,
                metadata["oracle"]["topk_by_position"][0]["token_ids"],
            )
            matrix[0, changed_token] = np.nextafter(
                matrix[0, changed_token], np.float32(np.inf), dtype=np.float32
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
            metadata["oracle"]["logits"]["sha256"] = digest
            metadata["oracle"]["logits"]["health"] = health
            for record in metadata["artifact_files_excluding_metadata"]:
                if record["file"] == "logits.f32":
                    record["sha256"] = digest
                    break
            else:
                self.fail("logits.f32 is absent from the artifact manifest")
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            promoted = self.run_validator(copied)
            self.assertNotEqual(promoted.returncode, 0)
            self.assertIn("promotion trust anchor", promoted.stderr)

            contract_only = self.run_validator(copied, contract_only=True)
            self.assertEqual(contract_only.returncode, 0, contract_only.stderr)
            self.assertIn("mode=contract-only trusted=false", contract_only.stdout)

    def test_topk_ties_use_ascending_token_id(self) -> None:
        import numpy as np

        module = self.load_module("qwen3_oracle_validator_ties", VALIDATOR)
        with tempfile.TemporaryDirectory() as temporary:
            logits_path = Path(temporary) / "logits.f32"
            logits = np.zeros((module.SEQUENCE_LEN, module.VOCAB_SIZE), dtype="<f4")
            logits[0, 11] = 7.0
            logits[0, 3] = 7.0
            logits.tofile(logits_path)

            topk = module.recompute_topk(logits_path)

            self.assertEqual(topk[0]["token_ids"][:2], [3, 11])
            self.assertEqual(topk[0]["logits"][:2], [7.0, 7.0])


if __name__ == "__main__":
    unittest.main()
