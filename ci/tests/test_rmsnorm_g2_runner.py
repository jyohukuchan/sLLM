from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes, sha256_file  # noqa: E402
import run_rmsnorm_g2_runtime as runner  # noqa: E402
import build_rmsnorm_g2_runtime as builder  # noqa: E402
import validate_rmsnorm_g2_contracts as contracts  # noqa: E402
from ci.tests.test_rmsnorm_g2_slice import slice_record  # noqa: E402


def candidate() -> dict[str, object]:
    return {"reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40, "worktree_clean": True, "revision_input": "full-sha"}


_FRESH_G2_BINARY: Path | None = None


def fresh_g2_binary() -> Path:
    global _FRESH_G2_BINARY
    if _FRESH_G2_BINARY is None:
        _FRESH_G2_BINARY = builder.build_g2_binary(ROOT)
    return _FRESH_G2_BINARY


def forged_identity_script() -> bytes:
    identity_line = canonical_bytes(contracts.expected_build_identity(ROOT)["identity"]).decode("utf-8")
    marker = contracts.G2_IDENTITY_MARKER.decode("ascii")
    return (
        "#!/usr/bin/env python3\n"
        f"# {marker}{identity_line}"
        "import sys\n"
        f"identity = {identity_line!r}\n"
        "if sys.argv[1:] == ['--query-build-identity']:\n"
        "    sys.stdout.write(identity)\n"
        "else:\n"
        "    sys.stderr.write('HIP unavailable\\n')\n"
        "    raise SystemExit(1)\n"
    ).encode("utf-8")


def protocol_document(target: str = "gfx1030") -> dict[str, object]:
    cases = []
    dispatch = {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "dispatch_count": 1, "workgroup_size_x": 256, "fallback_allowed": False, "fallback_used": False}
    for order, (case_id, rows, seed) in enumerate(zip(contracts.CASE_IDS, contracts.CASE_ROWS, contracts.CASE_SEEDS)):
        payload = base64.b64encode(bytes(rows * 2560 * 2)).decode("ascii")
        cases.append({"order": order, "id": case_id, "rows": rows, "n": 2560, "input_seed": seed, "request_b64": payload, "output_b64": payload, "dispatch": dict(dispatch)})
    return {"schema_version": "rmsnorm-g2-runtime-result-v1", "state": "PASS", "target": target, "model_used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "selected_backend": "hip", "dispatch_count": 6, "fallback_used": False, "cases": cases}


def artifact(target: str, value: dict[str, object]) -> dict[str, object]:
    source_set = contracts._source_set(ROOT)
    binary_bytes = fresh_g2_binary().read_bytes()
    binary_sha = hashlib.sha256(binary_bytes).hexdigest()
    build_identity = contracts.expected_build_identity(ROOT)
    prerequisites = [
        {"kind": "g0", "row_id": f"g0-{target}", "state": "bound-not-executed-by-g2", "candidate_sha256": contracts.candidate_sha256(value), "artifact_sha256": "1" * 64, "report_sha256": "2" * 64},
        {"kind": "private_g1", "row_id": f"g1-{target}", "state": "bound-not-executed-by-g2", "candidate_sha256": contracts.candidate_sha256(value), "artifact_sha256": "3" * 64, "report_sha256": "4" * 64},
        {"kind": "semantic_g1", "row_id": f"rmsnorm-semantic-g1-{target}", "state": "bound-not-executed-by-g2", "candidate_sha256": contracts.candidate_sha256(value), "artifact_sha256": "5" * 64, "report_sha256": "6" * 64},
        {"kind": "h3", "row_id": f"h3-rmsnorm-{target}", "state": "bound-not-executed-by-g2", "candidate_sha256": contracts.candidate_sha256(value), "artifact_sha256": "7" * 64, "report_sha256": "8" * 64},
    ]
    return {"schema_version": "rmsnorm-g2-artifact-v1", "artifact_id": f"rmsnorm-g2-{target}-{binary_sha}", "row_id": f"rmsnorm-g2-{target}", "target": target, "artifact_kind": "rmsnorm-g2-dedicated-public-rmsnorm", "candidate": value, "binary": {"role": "dedicated-g2-runtime", "path": "sllm-rmsnorm-g2-evidence", "sidecar_path": "sllm-rmsnorm-g2-evidence.sha256", "size_bytes": len(binary_bytes), "sha256": binary_sha, "sidecar_sha256": hashlib.sha256(f"{binary_sha}  sllm-rmsnorm-g2-evidence\n".encode()).hexdigest(), "source_path": contracts.G2_SOURCE_PATH, "source_sha256": source_set["files"][0]["sha256"], "build_source_set": source_set, "build_identity": {**build_identity["identity"], "identity_sha256": build_identity["identity_sha256"]}, "build_command": list(contracts.G2_BUILD_COMMAND), "build_profile": contracts.G2_BUILD_PROFILE, "builder_output_path": contracts.G2_BUILDER_OUTPUT_PATH, "g2_binary_name": "sllm-rmsnorm-g2-evidence", "g1_substitution_rejected": True, "h3_substitution_rejected": True}, "scope": {"model_used": True, "full_model_used": False, "tokenizer_used": False, "generation_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False}, "backend": "hip", "dispatch_contract": {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "dispatch_count": 1, "workgroup_size_x": 256, "fallback_allowed": False, "fallback_used": False}, "prerequisites": prerequisites}


class G2RunnerTests(unittest.TestCase):
    def _args(self, root: Path, *, binary: str = "sllm-rmsnorm-g2-evidence") -> Namespace:
        target = "gfx1030"
        (root / "artifact.json").write_text(json.dumps(artifact(target, candidate())), encoding="utf-8")
        from ci.tests.test_rmsnorm_g2_slice import fixture
        fixture_path = root / "synthetic.safetensors"
        fixture(fixture_path)
        record = slice_record()
        record["output"] = {"size_bytes": contracts.BYTE_SIZE, "sha256": hashlib.sha256(fixture_path.read_bytes()[contracts.ABSOLUTE_RANGE[0] : contracts.ABSOLUTE_RANGE[1]]).hexdigest()}
        (root / "slice.json").write_text(json.dumps(record), encoding="utf-8")
        binary_path = root / binary
        actual_binary = fresh_g2_binary().read_bytes()
        binary_path.write_bytes(actual_binary if binary == "sllm-rmsnorm-g2-evidence" else b"substituted-g1")
        binary_path.chmod(0o755)
        (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes(f"{hashlib.sha256(actual_binary).hexdigest()}  sllm-rmsnorm-g2-evidence\n".encode())
        return Namespace(repo=ROOT, target=target, slice_record=root / "slice.json", slice_file=root / "synthetic.safetensors", artifact=root / "artifact.json", binary=binary_path, output_dir=root / "out", reviewed_sha="a" * 40, tested_sha="a" * 40, workflow_sha="a" * 40, tree_oid="b" * 40)

    def test_host_runner_refuses_execution_and_emits_fail_not_fake_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            args = self._args(Path(directory))
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLLM_G2_GPU_EXECUTION", None)
                report = runner.run_row(args)
            self.assertEqual(report["state"], "FAIL")
            self.assertEqual(report["execution"]["exit_code"], 1)
            contracts.validate_report(report)

    def test_fresh_actual_g2_query_is_the_positive_identity_probe(self) -> None:
        self.assertEqual(
            runner.query_build_identity(fresh_g2_binary()),
            contracts.expected_build_identity(ROOT)["identity"],
        )

    def test_query_rejects_every_noncanonical_control_plane_shape(self) -> None:
        binary = fresh_g2_binary()
        expected = canonical_bytes(contracts.expected_build_identity(ROOT)["identity"])
        responses = (
            expected[:-1],
            b" " + expected,
            expected + b"\n",
            expected.replace(b"{", b"{\n", 1),
        )
        for stdout in responses:
            with self.subTest(stdout=stdout), patch.object(
                runner.subprocess,
                "run",
                return_value=runner.subprocess.CompletedProcess([], 0, stdout, b""),
            ), self.assertRaises(ContractError):
                runner.query_build_identity(binary)
        for completed in (
            runner.subprocess.CompletedProcess([], 1, expected, b""),
            runner.subprocess.CompletedProcess([], 0, expected, b"diagnostic"),
        ):
            with self.subTest(completed=completed), patch.object(runner.subprocess, "run", return_value=completed), self.assertRaises(ContractError):
                runner.query_build_identity(binary)
        with patch.object(runner.subprocess, "run", side_effect=runner.subprocess.TimeoutExpired([], 5)), self.assertRaises(ContractError):
            runner.query_build_identity(binary)

    def test_runner_rejects_g1_or_h3_binary_substitution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            args = self._args(Path(directory), binary="sllm-rmsnorm-g1-evidence")
            with self.assertRaises(ContractError):
                runner.run_row(args)

    def test_cpu_or_stub_nonzero_is_not_numeric_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            root = Path(directory)
            args = self._args(root)
            query = runner.subprocess.CompletedProcess([], 0, canonical_bytes(contracts.expected_build_identity(ROOT)["identity"]), b"")
            execution = runner.subprocess.CompletedProcess([], 1, b"", b"HIP unavailable")
            healthy = {"available": True, "reliable": True, "state": "OK", "target": "gfx1030", "ras_uncorrectable_count": 0}
            clean = {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []}
            with patch.dict(os.environ, {"SLLM_G2_GPU_EXECUTION": "1"}), patch.object(runner.subprocess, "run", return_value=query) as mocked_query, patch.object(runner, "_load_observation", side_effect=[healthy, healthy, clean, clean]), patch.object(runner, "_run_bounded_binary", return_value=execution):
                report = runner.run_row(args)
            mocked_query.assert_called_once()
            self.assertEqual(report["state"], "FAIL")
            self.assertEqual(report["scope"]["dispatch_count"], 0)
            self.assertIn("CPU/stub", report["execution"]["failure_reason"])

    def test_bounded_protocol_parser_and_independent_oracle_reject_trailing_duplicate_and_unknown(self) -> None:
        document = protocol_document()
        parsed = runner._parse_protocol(runner.canonical_protocol_bytes(document), "gfx1030")
        cases, passed, protocol_sha = runner._oracle_cases(parsed, bytes(contracts.BYTE_SIZE))
        self.assertTrue(passed)
        self.assertEqual(len(cases), 6)
        self.assertNotEqual(protocol_sha, "0" * 64)
        with self.assertRaises(ContractError):
            runner._parse_protocol(runner.canonical_protocol_bytes(document) + b"\n", "gfx1030")
        duplicate = runner.canonical_protocol_bytes(document).replace(b'"state":"PASS"', b'"state":"PASS","state":"PASS"', 1)
        with self.assertRaises(ContractError):
            runner._parse_protocol(duplicate, "gfx1030")
        unknown = copy.deepcopy(document)
        unknown["unexpected"] = True
        with self.assertRaises(ContractError):
            runner._parse_protocol(runner.canonical_protocol_bytes(unknown), "gfx1030")

    def test_public_build_artifact_owns_build_and_copied_output_is_not_authority(self) -> None:
        self.assertNotIn("_build_artifact_from_owned_binary", vars(builder))
        owned = fresh_g2_binary()
        prerequisites = artifact("gfx1030", candidate())["prerequisites"]
        with tempfile.TemporaryDirectory(prefix="sllm-g2-owned-api-") as directory:
            root = Path(directory)
            copied = root / contracts.G2_BINARY
            copied.write_bytes(owned.read_bytes())
            copied.chmod(0o755)
            with patch.object(builder, "build_g2_binary", return_value=owned) as build:
                with self.assertRaises(ContractError):
                    builder.build_artifact(
                        "gfx1030",
                        copied,
                        candidate(),
                        root / "artifact.json",
                        prerequisites=prerequisites,
                    )
            build.assert_called_once_with(ROOT)

        with patch.object(builder, "build_g2_binary", return_value=owned) as build:
            manifest = builder.build_artifact(
                "gfx1030",
                owned,
                candidate(),
                owned.parent / "artifact.json",
                prerequisites=prerequisites,
            )
        build.assert_called_once_with(ROOT)
        self.assertEqual(manifest["binary"]["path"], contracts.G2_BINARY)

    def test_g2_build_neutralizes_ambient_cargo_target_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-cargo-target-") as directory:
            repo = Path(directory)
            binary = repo / contracts.G2_BUILDER_OUTPUT_PATH
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"owned-output")
            binary.chmod(0o755)
            with patch.dict(os.environ, {"CARGO_TARGET_DIR": "/tmp/sllm-g2-stale-target"}, clear=False), patch.object(
                builder, "builder_output_path", return_value=binary
            ), patch.object(builder, "_stable_file_bytes", return_value=b"owned-output"), patch.object(
                builder, "_validate_builder_owned_output"
            ), patch.object(builder, "query_build_identity"), patch.object(
                builder, "sha256_file", return_value="a" * 64
            ), patch.object(
                builder.subprocess,
                "run",
                return_value=builder.subprocess.CompletedProcess([], 0, b"", b""),
            ) as run:
                result = builder.build_g2_binary(repo)

        self.assertEqual(result, binary)
        self.assertEqual(run.call_args.kwargs["env"]["CARGO_TARGET_DIR"], str((repo / "target").resolve()))
        self.assertEqual(run.call_args.args[0], list(contracts.G2_BUILD_COMMAND))

    def test_cli_builds_once_and_rejects_nonowned_binary_override(self) -> None:
        owned = fresh_g2_binary()
        with tempfile.TemporaryDirectory(prefix="sllm-g2-cli-owned-") as directory:
            root = Path(directory)
            output = owned.parent / "sllm-rmsnorm-g2-cli-test-artifact.json"
            prerequisites = artifact("gfx1030", candidate())["prerequisites"]

            def argv_for(binary: Path) -> list[str]:
                return [
                    "build_rmsnorm_g2_runtime.py",
                    "--repo", str(ROOT),
                    "--target", "gfx1030",
                    "--binary", str(binary),
                    "--output", str(output),
                    "--prerequisites", str(root / "prerequisites.json"),
                    "--reviewed-sha", "a" * 40,
                    "--tested-sha", "a" * 40,
                    "--workflow-sha", "a" * 40,
                    "--tree-oid", "b" * 40,
                ]

            with patch.object(sys, "argv", argv_for(owned)), patch.object(builder, "build_g2_binary", return_value=owned) as build, patch.object(
                builder, "read_json", return_value=prerequisites
            ), patch.object(builder, "validate_candidate"), patch.object(builder, "build_artifact", wraps=builder.build_artifact) as public_build:
                self.assertEqual(builder.main(), 0)
            build.assert_called_once_with(ROOT)
            public_build.assert_called_once()

            copied = root / contracts.G2_BINARY
            copied.write_bytes(owned.read_bytes())
            copied.chmod(0o755)
            with patch.object(sys, "argv", argv_for(copied)), patch.object(builder, "build_g2_binary", return_value=owned) as rejected_build, patch.object(
                builder, "read_json", return_value=prerequisites
            ), patch.object(builder, "validate_candidate"), patch.object(builder, "build_artifact", wraps=builder.build_artifact) as rejected_public:
                self.assertEqual(builder.main(), 1)
            rejected_build.assert_called_once_with(ROOT)
            rejected_public.assert_called_once()

    def test_runner_binds_declared_slice_to_extractor_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            root = Path(directory)
            args = self._args(root)
            declared = json.loads(args.slice_record.read_text(encoding="utf-8"))
            declared["output"]["sha256"] = "1" * 64
            args.slice_record.write_text(json.dumps(declared), encoding="utf-8")
            with self.assertRaises(ContractError):
                runner.run_row(args)
            args = self._args(root)
            linked = root / "linked.safetensors"
            linked.symlink_to(args.slice_file)
            args.slice_file = linked
            with self.assertRaises(ContractError):
                runner.run_row(args)

    def test_builder_emits_dedicated_manifest_and_rejects_g1_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-builder-") as directory:
            root = Path(directory)
            binary = root / "sllm-rmsnorm-g2-evidence"
            actual = fresh_g2_binary().read_bytes()
            binary.write_bytes(actual)
            binary.chmod(0o755)
            binary_sha = hashlib.sha256(actual).hexdigest()
            (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes(f"{binary_sha}  sllm-rmsnorm-g2-evidence\n".encode())
            prerequisites = artifact("gfx1030", candidate())["prerequisites"]
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=prerequisites)
            with patch.object(builder, "build_g2_binary", return_value=fresh_g2_binary()) as build:
                manifest = builder.build_artifact("gfx1030", fresh_g2_binary(), candidate(), fresh_g2_binary().parent / "artifact.json", prerequisites=prerequisites)
            build.assert_called_once_with(ROOT)
            self.assertEqual(manifest["binary"]["g2_binary_name"], "sllm-rmsnorm-g2-evidence")
            bad = root / "sllm-rmsnorm-g1-evidence"
            bad.write_bytes(b"wrong")
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", bad, candidate(), root / "bad.json", prerequisites=prerequisites)

    def test_artifact_consumer_rechecks_actual_binary_sidecar_and_source_set(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-builder-") as directory:
            root = Path(directory)
            binary = root / "sllm-rmsnorm-g2-evidence"
            actual = fresh_g2_binary().read_bytes()
            binary.write_bytes(actual)
            binary.chmod(0o755)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes(f"{digest}  sllm-rmsnorm-g2-evidence\n".encode())
            manifest = artifact("gfx1030", candidate())
            contracts.validate_artifact(manifest, binary_path=binary)
            binary.write_bytes(b"substituted")
            with self.assertRaises(ContractError):
                contracts.validate_artifact(manifest, binary_path=binary)
            binary.write_bytes(actual)
            (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes(b"0" * 64 + b"  sllm-rmsnorm-g2-evidence\n")
            with self.assertRaises(ContractError):
                contracts.validate_artifact(manifest, binary_path=binary)
            changed = copy.deepcopy(manifest)
            changed["binary"]["source_sha256"] = "f" * 64
            with self.assertRaises(ContractError):
                contracts.validate_artifact(changed, binary_path=binary)
            real = root / "real-binary"
            real.write_bytes(actual)
            binary.unlink()
            binary.symlink_to(real)
            with self.assertRaises(ContractError):
                contracts.validate_artifact(manifest, binary_path=binary)

    def test_actual_renamed_g1_binary_is_rejected_even_with_canonical_sidecar(self) -> None:
        g1_candidates = (ROOT / "target/debug/sllm-rmsnorm-g1-evidence", ROOT / "target/release/sllm-rmsnorm-g1-evidence")
        g1 = next((path for path in g1_candidates if path.is_file()), None)
        self.assertIsNotNone(g1, "the host workspace must build the actual G1 evidence binary for this regression")
        with tempfile.TemporaryDirectory(prefix="sllm-g2-g1-substitution-") as directory:
            root = Path(directory)
            binary = root / "sllm-rmsnorm-g2-evidence"
            binary.write_bytes(g1.read_bytes())
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            (root / "sllm-rmsnorm-g2-evidence.sha256").write_bytes(f"{digest}  sllm-rmsnorm-g2-evidence\n".encode())
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])

    def test_binary_identity_rejects_arbitrary_symlink_nonregular_malformed_and_mismatch(self) -> None:
        expected = contracts.expected_build_identity(ROOT)["identity"]
        cases = (
            b"arbitrary-binary-without-g2-identity",
            contracts.G2_IDENTITY_MARKER + b"not-json\n",
            contracts.G2_IDENTITY_MARKER + canonical_bytes({**expected, "role": "dedicated-g1-runtime"}),
        )
        for index, contents in enumerate(cases):
            with tempfile.TemporaryDirectory(prefix=f"sllm-g2-identity-{index}-") as directory:
                root = Path(directory)
                binary = root / contracts.G2_BINARY
                binary.write_bytes(contents)
                binary.chmod(0o755)
                digest = hashlib.sha256(contents).hexdigest()
                (root / contracts.G2_SIDECAR).write_bytes(f"{digest}  {contracts.G2_BINARY}\n".encode())
                with self.subTest(index=index), self.assertRaises(ContractError):
                    builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])
        with tempfile.TemporaryDirectory(prefix="sllm-g2-identity-symlink-") as directory:
            root = Path(directory)
            real = root / "real"
            actual = fresh_g2_binary().read_bytes()
            real.write_bytes(actual)
            binary = root / contracts.G2_BINARY
            binary.symlink_to(real)
            digest = hashlib.sha256(actual).hexdigest()
            (root / contracts.G2_SIDECAR).write_bytes(f"{digest}  {contracts.G2_BINARY}\n".encode())
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])
        with tempfile.TemporaryDirectory(prefix="sllm-g2-identity-directory-") as directory:
            root = Path(directory)
            binary = root / contracts.G2_BINARY
            binary.mkdir()
            (root / contracts.G2_SIDECAR).write_bytes(b"0" * 64 + b"  " + contracts.G2_BINARY.encode() + b"\n")
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])
        with tempfile.TemporaryDirectory(prefix="sllm-g2-identity-nonexec-") as directory:
            root = Path(directory)
            binary = root / contracts.G2_BINARY
            binary.write_bytes(fresh_g2_binary().read_bytes())
            binary.chmod(0o644)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            (root / contracts.G2_SIDECAR).write_bytes(f"{digest}  {contracts.G2_BINARY}\n".encode())
            with self.assertRaises(ContractError):
                builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])

    def test_builder_validator_and_runner_reject_marker_only_python_and_c_elf(self) -> None:
        expected = contracts.expected_build_identity(ROOT)["identity"]
        identity_line = canonical_bytes(expected).decode("utf-8")
        marker = contracts.G2_IDENTITY_MARKER.decode("ascii")
        with tempfile.TemporaryDirectory(prefix="sllm-g2-forged-executables-") as directory:
            root = Path(directory)
            c_source = (
                "#include <stdio.h>\n#include <string.h>\n"
                f"static const char marker[] = {json.dumps(marker)};\n"
                f"static const char identity[] = {json.dumps(identity_line)};\n"
                "int main(int argc, char **argv) {\n"
                "  if (argc == 2 && strcmp(argv[1], \"--query-build-identity\") == 0) { fputs(identity, stdout); return 0; }\n"
                "  fputs(\"HIP unavailable\\n\", stderr); return 1;\n}\n"
            )
            c_binary = root / "forged-c"
            compiled = subprocess.run(
                ["/usr/bin/cc", "-x", "c", "-O0", "-o", str(c_binary), "-"],
                input=c_source.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr.decode("utf-8", "replace"))
            for label, contents in (("python", forged_identity_script()), ("c-elf", c_binary.read_bytes())):
                case_root = root / label
                case_root.mkdir()
                binary = case_root / contracts.G2_BINARY
                binary.write_bytes(contents)
                binary.chmod(0o755)
                digest = hashlib.sha256(binary.read_bytes()).hexdigest()
                sidecar = case_root / contracts.G2_SIDECAR
                sidecar.write_bytes(f"{digest}  {contracts.G2_BINARY}\n".encode())
                with self.subTest(label=label):
                    with self.assertRaises(ContractError):
                        builder.build_artifact("gfx1030", binary, candidate(), root / "artifact.json", prerequisites=artifact("gfx1030", candidate())["prerequisites"])
                    forged_artifact = artifact("gfx1030", candidate())
                    forged_artifact["binary"]["size_bytes"] = binary.stat().st_size
                    forged_artifact["binary"]["sha256"] = digest
                    forged_artifact["binary"]["sidecar_sha256"] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
                    forged_artifact["artifact_id"] = f"rmsnorm-g2-gfx1030-{digest}"
                    with self.assertRaises(ContractError):
                        contracts.validate_artifact(forged_artifact, binary_path=binary)
                    args = self._args(case_root)
                    args.binary = binary
                    args.artifact.write_text(json.dumps(forged_artifact), encoding="utf-8")
                    with self.assertRaises(ContractError):
                        runner.run_row(args)

    def test_pass_report_rejects_identity_health_and_execution_failures(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            root = Path(directory)
            args = self._args(root)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLLM_G2_GPU_EXECUTION", None)
                report = runner.run_row(args)
            report["state"] = "PASS"
            report["scope"]["dispatch_count"] = 6
            report["dispatch"]["dispatch_count"] = 6
            report["execution"]["exit_code"] = 0
            report["health_pre"] = {"available": True, "reliable": True, "state": "OK", "target": "gfx1030", "ras_uncorrectable_count": 0}
            report["health_post"] = copy.deepcopy(report["health_pre"])
            report["process_pre"] = {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []}
            report["process_post"] = copy.deepcopy(report["process_pre"])
            report["execution"]["failure_reason"] = ""
            for case in report["cases"]:
                case["state"] = "PASS"
                case["dispatch_count"] = 1
            report["collection"] = {"expected_cases": 6, "collected_cases": 6, "passed_cases": 6, "failed_cases": 0, "expected_rows": 1, "collected_rows": 1}
            with self.assertRaises(ContractError):
                contracts.validate_report(report)
            for field, value in (("tree_oid", "c" * 40), ("device", {**report["device"], "uuid": "GPU-stale"}), ("execution", {**report["execution"], "timed_out": True}), ("execution", {**report["execution"], "crashed": True})):  # noqa: B007
                changed = copy.deepcopy(report)
                changed[field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ContractError):
                    contracts.validate_report(changed)

    def test_failure_report_rejects_candidate_prerequisite_health_seed_and_nonfinite_probes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g2-runner-") as directory:
            root = Path(directory)
            args = self._args(root)
            report = runner.run_row(args)
            for changed in (
                {**report, "candidate": {**report["candidate"], "tested_sha": "b" * 40}},
                {**report, "prerequisites": [{**report["prerequisites"][0], "row_id": "g0-wrong"}, *report["prerequisites"][1:]]},
                {**report, "health_pre": {**report["health_pre"], "target": "gfx1201"}},
                {**report, "cases": [{**report["cases"][0], "input_seed": 1}, *report["cases"][1:]]},
                {**report, "cases": [{**report["cases"][0], "max_abs_error": float("nan")}, *report["cases"][1:]]},
                {**report, "model": {**report["model"], "slice": {**report["model"]["slice"], "sha256": "0" * 64}}},
            ):
                with self.subTest(changed=changed), self.assertRaises(ContractError):
                    contracts.validate_report(changed)


if __name__ == "__main__":
    unittest.main()
