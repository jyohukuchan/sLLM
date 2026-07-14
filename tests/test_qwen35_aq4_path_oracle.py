from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_sums(root: Path) -> None:
    lines = []
    for name in ("manifest.json", "payload.jsonl", "runtime.json"):
        lines.append(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")


class Qwen35Aq4PathOracleTests(unittest.TestCase):
    def capture_source(self, root: Path, *, production: bool = False) -> Path:
        source = root / "source"
        CAPTURE.capture(
            type(
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
        )
        if production:
            manifest = ORACLE.load_json(source / "manifest.json")
            manifest["status"] = "available"
            manifest["evidence_class"] = "production"
            manifest["usable_as_source_evidence"] = True
            manifest["identity"]["model_revision"] = "fixture-source-revision"
            checkpoint_bytes = sum(
                entry["bytes"]
                for entry in manifest["identity"]["source_checkpoint"]["files"]
                if entry["file"].endswith(".safetensors")
            )
            runtime = {
                "device": "cpu",
                "dtype": "bfloat16",
                "full_vocab_ranking": True,
                "inference_mode": True,
                "low_cpu_mem_usage": False,
                "low_cpu_mem_usage_blocker": "accelerate package is unavailable in the installed environment",
                "max_resident_logit_rows": 1,
                "model_loads": 1,
                "preflight": {
                    "checkpoint_bytes": checkpoint_bytes,
                    "headroom_factor": 1.5,
                    "mem_available_bytes": checkpoint_bytes * 2,
                    "mem_total_bytes": checkpoint_bytes * 4,
                    "required_headroom_bytes": int(checkpoint_bytes * 1.5),
                    "status": "passed",
                },
                "python": platform.python_version(),
                "run": {"elapsed_seconds": 1.0, "row_count": manifest["payload"]["record_count"]},
                "runtime": "transformers.AutoModelForCausalLM",
                "safetensors": VALIDATE._package_version("safetensors"),
                "torch": VALIDATE._package_version("torch"),
                "torch_num_interop_threads": 1,
                "torch_num_threads": 1,
                "transformers": VALIDATE._package_version("transformers"),
            }
            manifest["runtime"] = runtime
            write_json(source / "manifest.json", manifest)
            write_json(source / "runtime.json", runtime)
            rewrite_sums(source)
            self.assertTrue(VALIDATE.validate_oracle(source, "source")["usable_as_source_evidence"])
        return source

    def fake_binary(self, root: Path, name: str = "fake-path-oracle.py") -> Path:
        payload_copy = root / "fixture-payload.jsonl"
        if not payload_copy.exists():
            shutil.copyfile(FIXTURE / "payload.jsonl", payload_copy)
        binary = root / name
        binary.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "print(Path(__file__).with_name('fixture-payload.jsonl').read_text(), end='')\n",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def production_export(
        self,
        root: Path,
        *,
        with_artifact: bool = False,
        large_binaries: bool = False,
    ) -> tuple[Path, Path, Path]:
        source = self.capture_source(root, production=True)
        product = root / "product"
        package = product / "package"
        package.mkdir(parents=True)
        package_manifest = package / "manifest.json"
        shutil.copyfile(FIXTURE / "package-manifest.json", package_manifest)
        artifact_manifest = None
        if with_artifact:
            artifact_dir = product / "artifact"
            artifact_dir.mkdir()
            artifact_manifest = artifact_dir / "manifest.json"
            write_json(artifact_manifest, {"artifact": "fixture", "version": 1})
        binary = self.fake_binary(root)
        if large_binaries:
            with binary.open("ab") as handle:
                handle.write(b"\n#" + b"x" * (ORACLE.MAX_JSON_BYTES + 1))
        worker = root / "served-worker"
        shutil.copyfile(binary, worker)
        worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
        served = root / "served-model.json"
        write_json(
            served,
            {
                "schema_version": "ullm.served_model.v2",
                "public": {"upstream_id": "Qwen/Qwen3.5-9B", "revision": "served-fixture-r1"},
                "worker": {
                    "binary": str(worker.resolve()),
                    "binary_sha256": ORACLE.sha256_file(worker),
                    "identity": {"device": "gfx1201", "execution_profile": "rdna4_aq4_resident"},
                    "required_environment": list(EXPORTER.REQUIRED_HIP_KERNEL_ENV),
                },
                "product": {
                    "root": str(product.resolve()),
                    "artifact": (
                        {
                            "content_sha256": "a" * 64,
                            "manifest_path": "artifact/manifest.json",
                            "manifest_sha256": ORACLE.sha256_file(artifact_manifest),
                        }
                        if artifact_manifest is not None
                        else None
                    ),
                    "package": {
                        "manifest_path": "package/manifest.json",
                        "manifest_sha256": ORACLE.sha256_file(package_manifest),
                    },
                },
            },
        )
        path = root / "path"
        result = EXPORTER.export(
            type(
                "Args",
                (),
                {
                    "package_dir": package,
                    "package_manifest": package_manifest,
                    "artifact_manifest": artifact_manifest,
                    "allow_package_only": artifact_manifest is None,
                    "cases": FIXTURE / "cases.json",
                    "source_oracle": source,
                    "tokenizer_root": FIXTURE / "source-model",
                    "output": path,
                    "link_output": None,
                    "binary": binary,
                    "served_model_manifest": served,
                    "model_id": None,
                    "model_revision": None,
                    "evidence_class": "production",
                    "device_kind": "gpu",
                    "device_index": 1,
                    "visible_devices": "1",
                    "chunk_bytes": 1024,
                    "prefill_m": 1,
                    "rotary_dim": 64,
                    "rope_base": 10_000_000.0,
                    "timeout_seconds": 30.0,
                },
            )()
        )
        self.assertTrue(result["path"]["usable_as_path_evidence"])
        return source, path, served

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
            source = self.capture_source(root)
            fake_binary = self.fake_binary(root)
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
                        "device_kind": "cpu",
                        "device_index": 0,
                        "visible_devices": None,
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
            self.assertIn("artifact binding kind", " ".join(result["link"]["blockers"]))
            runtime = ORACLE.load_json(path / "runtime.json")
            self.assertTrue(runtime["all_m1"])
            self.assertEqual(runtime["device_kind"], "cpu")
            self.assertEqual(runtime["device_index"], 0)
            self.assertIsNone(runtime["visible_devices"])
            self.assertEqual(runtime["evidence_scope"], "fixture_only")
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

    def test_production_package_only_requires_active_manifest_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            package_manifest = package / "manifest.json"
            shutil.copyfile(FIXTURE / "package-manifest.json", package_manifest)
            args = type(
                "Args",
                (),
                {
                    "tokenizer_root": FIXTURE / "source-model",
                    "tokenizer_file": list(ORACLE.TOKENIZER_FILES),
                    "package_manifest": package_manifest,
                    "artifact_manifest": None,
                    "model_id": "Qwen/Qwen3.5-9B",
                    "model_revision": "fixture",
                    "evidence_class": "production",
                    "served_model_manifest": None,
                },
            )()
            with self.assertRaisesRegex(ORACLE.OracleError, "served-model-manifest"):
                CAPTURE._path_identity(args)

    def test_production_path_runtime_reconstructs_served_worker_package_device_and_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source, path, _ = self.production_export(Path(temporary))
            report = VALIDATE.validate_oracle(path, "path")
            self.assertTrue(report["usable_as_path_evidence"])
            runtime = ORACLE.load_json(path / "runtime.json")
            self.assertEqual(runtime["execution_environment"]["HIP_VISIBLE_DEVICES"], "1")
            self.assertEqual(runtime["served_model_guard"]["worker"]["device_architecture"], "gfx1201")
            self.assertEqual(runtime["source_replay"]["root"], str(source.resolve()))
            self.assertEqual([item["length"] for item in runtime["source_replay"]["cases"]], [2, 1])

    def test_path_runtime_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, _ = self.production_export(root)
            mutations = {
                "unknown": lambda value: value.__setitem__("unknown", True),
                "device": lambda value: value.__setitem__("device_kind", "cpu"),
                "environment": lambda value: value["execution_environment"].__setitem__("HIP_VISIBLE_DEVICES", "0"),
                "replay": lambda value: value["source_replay"]["cases"][0].__setitem__("source_sequence_sha256", "0" * 64),
                "worker": lambda value: value["served_model_guard"]["worker"].__setitem__("binary_sha256", "0" * 64),
                "package": lambda value: value.__setitem__("package_manifest_sha256", "0" * 64),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    candidate = root / f"tampered-{name}"
                    shutil.copytree(original, candidate)
                    runtime = ORACLE.load_json(candidate / "runtime.json")
                    mutate(runtime)
                    write_json(candidate / "runtime.json", runtime)
                    rewrite_sums(candidate)
                    with self.assertRaises(ORACLE.OracleError):
                        VALIDATE.validate_oracle(candidate, "path")

            stale_sha = root / "tampered-stale-sha"
            shutil.copytree(original, stale_sha)
            runtime = ORACLE.load_json(stale_sha / "runtime.json")
            runtime["run"]["elapsed_seconds"] = 2.0
            write_json(stale_sha / "runtime.json", runtime)
            with self.assertRaisesRegex(ORACLE.OracleError, "SHA256SUMS"):
                VALIDATE.validate_oracle(stale_sha, "path")

            duplicate = root / "tampered-duplicate"
            shutil.copytree(original, duplicate)
            runtime_path = duplicate / "runtime.json"
            text = runtime_path.read_text(encoding="utf-8")
            text = text.replace('"device_kind":"gpu"', '"device_kind":"gpu","device_kind":"gpu"', 1)
            self.assertNotEqual(text, runtime_path.read_text(encoding="utf-8"))
            runtime_path.write_text(text, encoding="utf-8")
            rewrite_sums(duplicate)
            with self.assertRaisesRegex(ORACLE.OracleError, "duplicate JSON key"):
                VALIDATE.validate_oracle(duplicate, "path")

    def test_path_runtime_symlink_hardlink_and_toctou_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, _ = self.production_export(root)
            for kind in ("symlink", "hardlink"):
                with self.subTest(kind=kind):
                    candidate = root / f"tampered-{kind}"
                    shutil.copytree(original, candidate)
                    runtime = candidate / "runtime.json"
                    outside = root / f"{kind}-runtime.json"
                    shutil.copyfile(runtime, outside)
                    runtime.unlink()
                    if kind == "symlink":
                        runtime.symlink_to(outside)
                    else:
                        os.link(outside, runtime)
                    rewrite_sums(candidate)
                    with self.assertRaises(ORACLE.OracleError):
                        VALIDATE.validate_oracle(candidate, "path")
            with mock.patch.object(ORACLE, "_same_file_identity", return_value=False):
                with self.assertRaisesRegex(ORACLE.OracleError, "identity changed"):
                    VALIDATE.validate_oracle(original, "path")

    def test_cross_open_snapshot_rejects_post_hash_rename_and_same_size_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, _ = self.production_export(root)
            for mutation in ("rename", "rewrite"):
                with self.subTest(mutation=mutation):
                    candidate = root / f"cross-open-{mutation}"
                    shutil.copytree(original, candidate)
                    changed = False

                    def hook(stage: str, validation_root: Path) -> None:
                        nonlocal changed
                        if changed or stage != "after_sha256s" or validation_root != candidate:
                            return
                        changed = True
                        runtime = candidate / "runtime.json"
                        info = runtime.stat()
                        data = runtime.read_bytes()
                        if mutation == "rename":
                            replacement = candidate / "replacement.json"
                            replacement.write_bytes(data)
                            os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
                            os.replace(replacement, runtime)
                        else:
                            replaced = data.replace(b'"device_kind":"gpu"', b'"device_kind":"cpu"', 1)
                            self.assertEqual(len(replaced), len(data))
                            runtime.write_bytes(replaced)
                            os.utime(runtime, ns=(info.st_atime_ns, info.st_mtime_ns))

                    with mock.patch.object(VALIDATE, "VALIDATION_TEST_HOOK", hook):
                        with self.assertRaisesRegex(ORACLE.OracleError, "snapshot identity changed"):
                            VALIDATE.validate_oracle(candidate, "path")
                    self.assertTrue(changed)

    def test_external_runtime_references_stream_past_json_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, path, _ = self.production_export(root, large_binaries=True)
            runtime = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
            self.assertGreater(
                Path(runtime["binary"]["path"]).stat().st_size,
                ORACLE.MAX_JSON_BYTES,
            )
            self.assertGreater(
                Path(runtime["served_model_guard"]["worker"]["binary_path"]).stat().st_size,
                ORACLE.MAX_JSON_BYTES,
            )
            self.assertTrue(VALIDATE.validate_oracle(path, "path")["usable_as_path_evidence"])

    def test_external_runtime_snapshot_rejects_post_semantic_replacements(self) -> None:
        mutations = (
            ("binary_rewrite", False),
            ("binary_rename", False),
            ("worker_replace", False),
            ("package_replace", False),
            ("artifact_replace", True),
        )
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            for mutation, with_artifact in mutations:
                with self.subTest(mutation=mutation):
                    root = outer / mutation
                    _, path, _ = self.production_export(
                        root, with_artifact=with_artifact
                    )
                    runtime = json.loads((path / "runtime.json").read_text(encoding="utf-8"))
                    artifact_path = runtime["artifact_manifest"]
                    if mutation == "artifact_replace":
                        self.assertIsInstance(artifact_path, str)
                    targets = {
                        "binary_rewrite": Path(runtime["binary"]["path"]),
                        "binary_rename": Path(runtime["binary"]["path"]),
                        "worker_replace": Path(
                            runtime["served_model_guard"]["worker"]["binary_path"]
                        ),
                        "package_replace": Path(runtime["package_manifest"]),
                        "artifact_replace": Path(artifact_path or runtime["package_manifest"]),
                    }
                    target = targets[mutation]
                    changed = False

                    def hook(stage: str, validation_root: Path) -> None:
                        nonlocal changed
                        if changed or stage != "after_path_semantics" or validation_root != path:
                            return
                        changed = True
                        info = target.stat()
                        data = target.read_bytes()
                        if mutation == "binary_rewrite":
                            replacement = bytes([data[0] ^ 1]) + data[1:]
                            self.assertEqual(len(replacement), len(data))
                            target.write_bytes(replacement)
                            os.chmod(target, info.st_mode)
                            os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))
                        else:
                            replacement_path = target.with_name(target.name + ".replacement")
                            replacement_path.write_bytes(data)
                            os.chmod(replacement_path, info.st_mode)
                            os.utime(
                                replacement_path,
                                ns=(info.st_atime_ns, info.st_mtime_ns),
                            )
                            os.replace(replacement_path, target)

                    with mock.patch.object(VALIDATE, "VALIDATION_TEST_HOOK", hook):
                        with self.assertRaisesRegex(
                            ORACLE.OracleError, "snapshot|directory"
                        ):
                            VALIDATE.validate_oracle(path, "path")
                    self.assertTrue(changed)

    def test_cross_open_snapshot_rejects_semantic_then_sha_version_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, _ = self.production_export(root)
            candidate = root / "cross-open-semantic-sha"
            shutil.copytree(original, candidate)
            changed = False

            def hook(stage: str, validation_root: Path) -> None:
                nonlocal changed
                if changed or stage != "after_path_semantics" or validation_root != candidate:
                    return
                changed = True
                runtime = candidate / "runtime.json"
                data = runtime.read_bytes()
                replacement = data.replace(b'"device_kind":"gpu"', b'"device_kind":"cpu"', 1)
                self.assertEqual(len(replacement), len(data))
                runtime.write_bytes(replacement)
                rewrite_sums(candidate)

            with mock.patch.object(VALIDATE, "VALIDATION_TEST_HOOK", hook):
                with self.assertRaises(ORACLE.OracleError):
                    VALIDATE.validate_oracle(candidate, "path")
            self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
