from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.test_sq8_canonical_artifact import (
    SCALE_NAME,
    SQ8,
    WEIGHT_NAME,
    source_scale,
    source_weight,
    write_model,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class Sq8PythonRustInteropTests(unittest.TestCase):
    def test_python_artifact_is_verified_and_reconstructed_by_rust(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = write_model(root)
            artifact_dir = root / "artifact"
            manifest = SQ8.build_canonical_artifact(
                model_dir,
                artifact_dir,
                tensor_names=[WEIGHT_NAME],
                copy_chunk_bytes=34,
            )

            completed = subprocess.run(
                [
                    "cargo",
                    "run",
                    "--quiet",
                    "-p",
                    "ullm-engine",
                    "--example",
                    "verify_sq8_canonical",
                    "--",
                    str(artifact_dir),
                    WEIGHT_NAME,
                    "2",
                    "2",
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)

            expected_block = (
                source_weight()[256:259, 256:257].float()
                * source_scale()[2, 2].float()
            )
            digest = hashlib.sha256(
                expected_block.contiguous().numpy().astype("<f4").tobytes()
            ).hexdigest()

            self.assertTrue(report["verified"])
            self.assertEqual(report["schema_version"], SQ8.SCHEMA_VERSION)
            self.assertEqual(
                report["content_sha256"],
                manifest["integrity"]["content_sha256"],
            )
            self.assertEqual(report["selected_pair_count"], 1)
            self.assertEqual(
                report["weight_payload_bytes"],
                source_weight().numel(),
            )
            self.assertEqual(
                report["scale_payload_bytes"],
                source_scale().numel() * 2,
            )
            self.assertEqual(report["reconstructed_block"]["rows"], 3)
            self.assertEqual(report["reconstructed_block"]["cols"], 1)
            self.assertEqual(
                report["reconstructed_block"]["values_sha256"],
                digest,
            )
            self.assertEqual(
                manifest["quantized_tensors"][0]["scale"]["name"],
                SCALE_NAME,
            )


if __name__ == "__main__":
    unittest.main()
