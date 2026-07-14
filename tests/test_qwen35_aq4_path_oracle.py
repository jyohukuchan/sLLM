from __future__ import annotations

import importlib.util
import json
import os
import shutil
import stat
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "qwen35-aq4-p2-oracle"


def load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPORTER = load_tool("export_qwen35_aq4_path_oracle", "export-qwen35-aq4-path-oracle.py")
CAPTURE = EXPORTER.CAPTURE
VALIDATE = EXPORTER.VALIDATE
ORACLE = EXPORTER.ORACLE


class Qwen35Aq4PathOracleTests(unittest.TestCase):
    def test_fake_binary_replay_and_package_only_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            path = root / "path"
            link = root / "link"
            package = root / "package"
            package.mkdir()
            package_manifest = package / "manifest.json"
            shutil.copyfile(FIXTURE / "package-manifest.json", package_manifest)
            source_args = type(
                "Args",
                (),
                {
                    "output": source,
                    "cases": FIXTURE / "cases.json",
                    "payload": FIXTURE / "payload.jsonl",
                    "kind": "source",
                    "source_root": FIXTURE / "source-model",
                    "evidence_class": "synthetic_fixture",
                },
            )()
            CAPTURE.capture(source_args)

            payload_copy = root / "fixture-payload.jsonl"
            shutil.copyfile(FIXTURE / "payload.jsonl", payload_copy)
            fake_binary = root / "fake-path-oracle.py"
            fake_binary.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                "print(Path(__file__).with_name('fixture-payload.jsonl').read_text(), end='')\n",
                encoding="utf-8",
            )
            fake_binary.chmod(fake_binary.stat().st_mode | stat.S_IXUSR)
            served_manifest = root / "served-model.json"
            served_manifest.write_text(
                json.dumps({"worker": {"required_environment": list(EXPORTER.REQUIRED_HIP_KERNEL_ENV)}}),
                encoding="utf-8",
            )

            result = EXPORTER.export(
                type(
                    "Args",
                    (),
                    {
                        "package_dir": package,
                        "package_manifest": package_manifest,
                        "artifact_manifest": None,
                        "allow_package_only": True,
                        "cases": FIXTURE / "cases.json",
                        "source_oracle": source,
                        "tokenizer_root": FIXTURE / "source-model",
                        "output": path,
                        "link_output": link,
                        "binary": fake_binary,
                        "served_model_manifest": served_manifest,
                        "model_id": None,
                        "model_revision": None,
                        "evidence_class": "synthetic_fixture",
                        "device_index": 0,
                        "chunk_bytes": 1024,
                        "prefill_m": 1,
                        "rotary_dim": 64,
                        "rope_base": 10_000_000.0,
                        "timeout_seconds": 30.0,
                    },
                )()
            )
            path_manifest = ORACLE.load_json(path / "manifest.json")
            self.assertIsNone(path_manifest["identity"]["artifact"]["artifact_manifest_sha256"])
            self.assertEqual(path_manifest["identity"]["artifact"]["package_manifest_sha256"], ORACLE.sha256_file(package_manifest))
            self.assertEqual(result["path"]["status"], "valid")
            self.assertFalse(result["path"]["usable_as_path_evidence"])
            self.assertFalse(result["link"]["usable_as_p2_oracle_link"])
            self.assertIn("no artifact manifest", " ".join(result["link"]["blockers"]))
            runtime = ORACLE.load_json(path / "runtime.json")
            self.assertTrue(runtime["all_m1"])
            self.assertEqual(runtime["model_loads"], 1)
            self.assertEqual(runtime["run"]["row_count"], 3)
            self.assertEqual(
                json.loads((path / "payload.jsonl").read_text(encoding="utf-8").splitlines()[0])["greedy_token_id"],
                7,
            )
            self.assertEqual(VALIDATE.validate_link(link, source, path)["agreement"]["record_count"], 3)

    def test_package_only_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            package_manifest = package / "manifest.json"
            shutil.copyfile(FIXTURE / "package-manifest.json", package_manifest)
            with self.assertRaises(ORACLE.OracleError):
                EXPORTER.export(
                    type(
                        "Args",
                        (),
                        {
                            "package_dir": package,
                            "package_manifest": package_manifest,
                            "artifact_manifest": None,
                            "allow_package_only": False,
                            "served_model_manifest": None,
                        },
                    )()
                )


if __name__ == "__main__":
    unittest.main()
