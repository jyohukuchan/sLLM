"""CLI-only adversarial checks for the semantic G1 controller authority gate."""

from __future__ import annotations

import os
import copy
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = ROOT / "ci/tools/orchestrate_rmsnorm_g1_evidence.py"
sys.path.insert(0, str(ROOT / "ci/tools"))

import validate_rmsnorm_g1_contracts as contracts  # noqa: E402
import orchestrate_rmsnorm_g1_evidence as imported_controller  # noqa: E402
import run_rmsnorm_g1_runtime as runner  # noqa: E402


class SemanticG1ControllerTests(unittest.TestCase):
    def test_uploaded_report_does_not_embed_raw_frame_bytes(self) -> None:
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertNotIn('"response_b64": base64.b64encode(response)', source)
        self.assertIn('"raw_frame_sha256": contracts.sha256_bytes(raw_frames)', source)

    def _closed_environment(self, root: Path, *, extra: dict[str, str] | None = None) -> dict[str, str]:
        reviewed = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout.decode("ascii").strip()
        result = {
            "PATH": "/usr/bin:/bin", "LC_CTYPE": "C.UTF-8", "HOME": os.environ["HOME"], "CI": "true", "GITHUB_ACTIONS": "true",
            "GITHUB_SHA": reviewed, "GITHUB_WORKSPACE": str(ROOT), "RUNNER_TEMP": str(root), "RUN_ROOT": str(root / "run"),
            "REVIEWED_SHA": reviewed, "TESTED_SHA": reviewed, "WORKFLOW_SHA": reviewed,
            "GITHUB_RUN_ID": "1", "GITHUB_RUN_ATTEMPT": "1", "GITHUB_WORKFLOW": "semantic-rmsnorm-g1",
        }
        if extra:
            result.update(extra)
        return result

    @staticmethod
    def _controller_arguments(root: Path) -> tuple[str, ...]:
        reviewed = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout.decode("ascii").strip()
        return (
            "--artifact-root", str(root / "run" / "artifacts"),
            "--output-dir", str(root / "run" / "rmsnorm-semantic-g1-aggregate-1-1"),
            "--run-id", "1", "--run-attempt", "1",
            "--reviewed-sha", reviewed, "--tested-sha", reviewed, "--workflow-sha", reviewed,
        )

    @staticmethod
    def _sealed_controller_command(*controller_args: str) -> list[str]:
        """Exec the controller from a fully sealed source descriptor.

        This mirrors the reviewed workflow's two-stage launcher without
        attempting a Git/GPU run: the current intentionally dirty workspace
        must still fail at the controller's immutable-candidate gate.
        """

        bootstrap = r'''
import fcntl
import os
import sys
data = open(sys.argv[1], "rb").read()
fd = os.memfd_create("semantic-g1-controller-test", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
offset = 0
while offset < len(data):
    offset += os.write(fd, data[offset:])
fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL)
os.set_inheritable(fd, True)
os.execve("/usr/bin/python3", ["/usr/bin/python3", "-I", "-S", f"/proc/self/fd/{fd}", *sys.argv[2:]], {**os.environ, "SLLM_G1_CONTROLLER_FD": str(fd)})
'''
        return ["/usr/bin/python3", "-I", "-S", "-c", bootstrap, str(CONTROLLER), *controller_args]

    def test_importable_controller_has_no_execution_or_emission_api(self) -> None:
        code = (
            "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'ci/tools')); "
            "import orchestrate_rmsnorm_g1_evidence as c; print(hasattr(c, 'run_controller'), hasattr(c, '_emit_only_from_local_controller_rows'), c.main([]))"
        )
        completed = subprocess.run(["/usr/bin/python3", "-c", code], cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "False False 2")
        self.assertIn("importable execution is disabled", completed.stderr.decode("utf-8"))
        source = CONTROLLER.read_text(encoding="utf-8")
        self.assertNotIn("Path(__file__)", source)
        self.assertNotIn("sys.path.insert", source)
        self.assertIn("_fresh_controller_gate", source)
        self.assertIn("_BOOTSTRAP_SOURCES", source)
        self.assertIn("_load_reviewed_module", source)
        self.assertIn("SLLM_G1_CONTROLLER_FD", source)

    def test_fresh_gate_rejects_nonisolated_wrong_executable_runpy_and_same_process(self) -> None:
        commands = [
            ["/usr/bin/python3", str(CONTROLLER)],
            ["/usr/bin/python3.12", "-I", str(CONTROLLER)],
            ["/usr/bin/python3", "-I", "-m", "runpy", str(CONTROLLER)],
            ["/usr/bin/python3", "-I", "-c", f"exec(compile(open({str(CONTROLLER)!r}, 'rb').read(), {str(CONTROLLER)!r}, 'exec'))"],
        ]
        for command in commands:
            completed = subprocess.run(command, cwd=ROOT, env={"PATH": "/usr/bin:/bin"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertNotEqual(completed.returncode, 0)
            if "runpy" not in command:
                self.assertIn("FAIL-CLOSED", completed.stderr.decode("utf-8"))

    def test_closed_environment_rejects_pythonpath_and_dirty_or_copied_authority_before_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-controller-") as temporary:
            root = Path(temporary)
            py_path = subprocess.run(self._sealed_controller_command(*self._controller_arguments(root)), cwd=ROOT, env=self._closed_environment(root, extra={"PYTHONPATH": "/tmp/forged"}), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(py_path.returncode, 2)
            self.assertIn("environment is not the exact closed", py_path.stderr.decode("utf-8"))
            clean_env = subprocess.run(self._sealed_controller_command(*self._controller_arguments(root)), cwd=ROOT, env=self._closed_environment(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(clean_env.returncode, 2)
            self.assertRegex(clean_env.stderr.decode("utf-8"), "dangerous Git local configuration|dirty or copied mutable checkout")
            forged_args = list(self._controller_arguments(root))
            tree_mismatch = subprocess.run(
                self._sealed_controller_command(*forged_args),
                cwd=ROOT,
                env=self._closed_environment(root, extra={"TREE_OID": "forbidden-caller-tree"}),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(tree_mismatch.returncode, 2)
            self.assertIn("environment is not the exact closed", tree_mismatch.stderr.decode("utf-8"))
            self.assertFalse((root / "run" / "artifacts").exists())

    def test_direct_mutable_controller_path_and_unsealed_descriptor_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-controller-") as temporary:
            root = Path(temporary)
            direct = subprocess.run(
            ["/usr/bin/python3", "-I", "-S", str(CONTROLLER)],
                cwd=ROOT,
                env=self._closed_environment(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(direct.returncode, 2)
            self.assertIn("sealed controller-source descriptor", direct.stderr.decode("utf-8"))

    def test_controller_numeric_recomputation_rejects_a_mutated_raw_response(self) -> None:
        # Import mode has no authority or emission route, so this can exercise
        # the independently recomputed tolerance boundary without fabricating
        # a GPU result.  A single BF16 response-word mutation must fail.
        imported_controller.runner = runner
        imported_controller.math = math
        imported_controller.struct = struct
        expected = struct.pack("<H", runner._f32_to_bf16(1.0))
        mutated = struct.pack("<H", runner._f32_to_bf16(1.5))
        with self.assertRaises(imported_controller.ControllerError):
            imported_controller._numerics(mutated, expected, atol=0.0, rtol=0.0)

    def test_offline_saved_response_evidence_recomputes_and_rejects_missing_mutated_or_misbound_bytes(self) -> None:
        imported_controller.runner = runner
        imported_controller.math = math
        imported_controller.struct = struct
        imported_controller.contracts = contracts
        row = {"row_id": contracts.ROWS[0], "target": "gfx1030", **contracts.EXPECTED_BINDINGS["gfx1030"]}
        candidate = {"reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40}

        def response_for(case: dict[str, object], order: int) -> tuple[bytes, dict[str, object]]:
            request, activation, scale, epsilon = imported_controller._case_request(row, case)
            del request
            parsed_output = runner.independent_rmsnorm_oracle(activation, scale, int(case["rows"]), int(case["n"]), epsilon)
            header = bytearray(runner.OUTPUT_HEADER_BYTES)
            offset = 0

            def put(fmt: str, *values: object) -> None:
                nonlocal offset
                struct.pack_into(fmt, header, offset, *values)
                offset += struct.calcsize(fmt)

            def fixed(value: str) -> None:
                nonlocal offset
                encoded = value.encode("ascii")
                header[offset:offset + 64] = encoded + b"\0" * (64 - len(encoded))
                offset += 64

            header[:8] = runner.OUTPUT_MAGIC
            offset = 8
            put("<II", runner.OUTPUT_PROTOCOL_VERSION, runner.OUTPUT_HEADER_BYTES)
            put("<II", 2, 0)
            put("<8Q", int(case["rows"]), int(case["n"]), 0, 0, 0, 0, 0, 0)
            put("<QQQ", int(case["rows"]) * int(case["n"]), int(case["n"]), int(case["rows"]))
            put("<II", struct.unpack("<I", struct.pack("<f", epsilon))[0], 0)
            put("<II", 0, 1)
            put("<QIIII", order + 1, 1, 1, 256, int(case["rows"]))
            put("<II", 0, 0)
            put("<7I", 1, 1, 3, 2, 1, 3, 0)
            put("<4I", 3, 3, 1, 0)
            put("<QQQ", 0, 0, 0)
            fixed(runner.KERNEL_SYMBOL)
            fixed(runner.DEVICE_SYMBOL)
            fixed("gfx1030")
            put("<Q", len(parsed_output))
            self.assertEqual(offset, runner.OUTPUT_HEADER_BYTES)
            response = bytes(header) + parsed_output
            numerical = imported_controller._numerics(parsed_output, parsed_output, atol=0.0078125, rtol=0.015625)
            digest = contracts.sha256_bytes(response)
            evidence = {
                "path": f"rows/{row['row_id']}/raw/case-{order}.bin",
                "sidecar_path": f"rows/{row['row_id']}/raw/case-{order}.bin.sha256",
                "size_bytes": len(response), "sha256": digest,
                "sidecar_sha256": contracts.sha256_bytes(contracts._sidecar_text(digest, f"case-{order}.bin")),
                "candidate_sha256": contracts.sha256_json(candidate), "row_id": row["row_id"], "case_id": case["id"], "order": order,
            }
            return response, {"order": order, **case, "request_sha256": "", "response_sha256": digest, "response_evidence": evidence, "resource_counts": contracts.EXPECTED_CASE_RESOURCE_COUNTS, "dispatch_id": order + 1, "dispatch_count": 1, "kernel_symbol": runner.KERNEL_SYMBOL, "device_symbol": runner.DEVICE_SYMBOL, "numerics": numerical, "controller_started_at": "2026-08-08T00:00:00Z", "controller_finished_at": "2026-08-08T00:00:01Z", "controller_duration_ns": 1}

        with tempfile.TemporaryDirectory(prefix="sllm-g1-offline-evidence-") as temporary:
            evidence_root = Path(temporary)
            cases: list[dict[str, object]] = []
            raw_parts: list[bytes] = []
            for order, case in enumerate(contracts.EXPECTED_CASES):
                response, document = response_for(case, order)
                request, _activation, _scale, _epsilon = imported_controller._case_request(row, case)
                document["request_sha256"] = contracts.sha256_bytes(request)
                path = evidence_root / str(document["response_evidence"]["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(response)
                path.with_name(path.name + ".sha256").write_bytes(contracts._sidecar_text(str(document["response_evidence"]["sha256"]), path.name))
                cases.append(document)
                raw_parts.append(response)
            total = {name: value * len(contracts.EXPECTED_CASES) for name, value in contracts.EXPECTED_CASE_RESOURCE_COUNTS.items()}
            report = {"row_id": row["row_id"], "target": row["target"], "state": "PASS", "candidate": candidate, "cases": cases, "raw_frame_sha256": contracts.sha256_bytes(b"".join(raw_parts)), "resource_counts": total}
            aggregate = {"rows": [{"row_id": row["row_id"], "response_evidence": [case["response_evidence"] for case in cases]}]}
            facts = contracts.recompute_saved_response_evidence(evidence_root, report, aggregate=aggregate, expected_candidate=candidate, repo=ROOT)
            self.assertEqual(facts["case_count"], 15)
            self.assertEqual(facts["resource_counts"], total)

            missing_root = evidence_root.parent / "missing"
            shutil.copytree(evidence_root, missing_root, dirs_exist_ok=True)
            (missing_root / str(cases[0]["response_evidence"]["path"])).unlink()
            with self.assertRaises(contracts.EvidenceError):
                contracts.recompute_saved_response_evidence(missing_root, report, aggregate=aggregate, expected_candidate=candidate, repo=ROOT)

            mutated_root = evidence_root.parent / "mutated"
            shutil.copytree(evidence_root, mutated_root, dirs_exist_ok=True)
            mutated_path = mutated_root / str(cases[1]["response_evidence"]["path"])
            mutated = bytearray(mutated_path.read_bytes()); mutated[-1] ^= 1; mutated_path.write_bytes(mutated)
            with self.assertRaises(contracts.EvidenceError):
                contracts.recompute_saved_response_evidence(mutated_root, report, aggregate=aggregate, expected_candidate=candidate, repo=ROOT)

            misbound = copy.deepcopy(report)
            misbound["cases"][0]["response_evidence"]["case_id"] = misbound["cases"][1]["id"]
            with self.assertRaises(contracts.EvidenceError):
                contracts.recompute_saved_response_evidence(evidence_root, misbound, aggregate=aggregate, expected_candidate=candidate, repo=ROOT)

    def test_runtime_closure_is_recursive_and_rejects_missing_transitive_or_replaced_objects(self) -> None:
        closure = contracts.runtime_dependency_closure(Path("/bin/true"))
        retained: dict[str, tuple[dict[str, object], int]] = {}
        descriptors: list[contracts.SealedDescriptor] = []
        try:
            for item in closure["objects"]:
                descriptor = contracts.snapshot_file(Path(item["record"]["resolved_path"]), item["record"], "closure test object")
                descriptors.append(descriptor)
                retained[str(item["record"]["resolved_path"])] = (dict(descriptor.record), descriptor.fd)
            root = str(closure["root"])
            loader = str(next(item["interpreter"] for item in closure["objects"] if item["record"]["resolved_path"] == root))
            root_data = retained[root][1]
            root_dynamic = contracts._elf_dynamic(contracts.fd_read_all(root_data), root)
            loader_key = next(key for key in retained if key == loader)
            loader_record, loader_fd = retained[loader_key]
            retained[loader_key] = ({**loader_record, "path": str(root_dynamic["interpreter"])}, loader_fd)
            contracts.validate_runtime_dependency_closure(retained, closure, root_path=root, loader_path=loader)

            missing = copy.deepcopy(closure)
            missing["objects"] = missing["objects"][:-1]
            missing["sha256"] = contracts.sha256_json({key: missing[key] for key in ("complete", "algorithm", "root", "objects")})
            with self.assertRaises(contracts.EvidenceError):
                contracts.validate_runtime_dependency_closure(retained, missing, root_path=root, loader_path=loader)

            replaced = copy.deepcopy(closure)
            replaced["objects"][0]["record"]["sha256"] = "0" * 64
            replaced["sha256"] = contracts.sha256_json({key: replaced[key] for key in ("complete", "algorithm", "root", "objects")})
            with self.assertRaises(contracts.EvidenceError):
                contracts.validate_runtime_dependency_closure(retained, replaced, root_path=root, loader_path=loader)
        finally:
            for descriptor in descriptors:
                descriptor.close()

    def test_fixed_worker_bootstrap_uses_only_sealed_stdlib_worker_source(self) -> None:
        code = r'''
import os
import subprocess
import sys
from pathlib import Path
root = Path.cwd()
sys.path.insert(0, str(root / "ci/tools"))
import orchestrate_rmsnorm_g1_evidence as controller
import validate_rmsnorm_g1_contracts as contracts
source = contracts.snapshot_file(root / "ci/tools/run_rmsnorm_g1_runtime.py", None, "host-runner")
try:
    command = ["/usr/bin/python3", "-I", "-S", "-c", controller._SEALED_WORKER_BOOTSTRAP, str(source.fd), str(root / "ci/tools/run_rmsnorm_g1_runtime.py"), "--help"]
    completed = subprocess.run(command, cwd="/", env={"PATH": "/usr/bin:/bin"}, pass_fds=(source.fd,), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0 or b"Fixed, stdlib-only worker" not in completed.stdout:
        raise SystemExit(completed.stderr.decode("utf-8", "replace") or "sealed worker bootstrap failed")
    print("sealed worker bootstrap: PASS")
finally:
    source.close()
'''
        completed = subprocess.run(["/usr/bin/python3", "-c", code], cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "sealed worker bootstrap: PASS")
        worker_source = (ROOT / contracts.RUNNER_RELATIVE_PATH).read_text(encoding="utf-8")
        self.assertNotIn("import common", worker_source)
        self.assertNotIn("import validate_rmsnorm", worker_source)

    def test_reviewed_contract_bytes_ignore_a_post_gate_mutable_path_replacement(self) -> None:
        # This is intentionally a host-only subprocess: it proves the
        # controller-specific binding layer reads the reviewed byte map even
        # when every corresponding checkout path is forged after binding.
        code = r'''
import sys
import tempfile
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "ci/tools"))
import validate_rmsnorm_g1_contracts as contracts
sources = {relative: (root / relative).read_bytes() for relative in contracts.AUTHORITY_SOURCE_FILES}
with tempfile.TemporaryDirectory(prefix="sllm-g1-reviewed-contract-") as temporary:
    repo = Path(temporary) / "repo"
    for relative in contracts.REVIEWED_CONTRACT_FILES:
        forged = repo / relative
        forged.parent.mkdir(parents=True, exist_ok=True)
        forged.write_bytes(b'{"forged":true}')
    contracts.bind_controller_reviewed_sources(repo, sources)
    matrix = contracts.validate_matrix(repo)
    if matrix["suite_id"] != contracts.MATRIX_SUITE_ID:
        raise SystemExit("forged mutable matrix path became authority")
    if contracts.manifest_hashes(repo)["matrix_manifest_sha256"] != contracts.sha256_bytes(sources[contracts.MATRIX_MANIFEST]):
        raise SystemExit("forged mutable schema path became authority")
print("reviewed contract bytes: PASS")
'''
        completed = subprocess.run(
            ["/usr/bin/python3", "-c", code, str(ROOT)],
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "reviewed contract bytes: PASS")

    def test_reviewed_contract_validation_needs_no_site_packages(self) -> None:
        code = r'''
import sys
import copy
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / "ci/tools"), str(root / "ci/tests")]
import validate_rmsnorm_g1_contracts as contracts
import test_rmsnorm_g1_aggregate as aggregate_tests
if "jsonschema" in sys.modules:
    raise SystemExit("jsonschema loaded before controller binding")
sources = {relative: (root / relative).read_bytes() for relative in contracts.AUTHORITY_SOURCE_FILES}
contracts.bind_controller_reviewed_sources(root, sources)
matrix = contracts.validate_matrix(root)
document = aggregate_tests.SemanticG1AggregateTests()._aggregate_document()
try:
    contracts.validate_aggregate_document(
        document,
        identity=document["candidate"],
        repo=root,
        authority=document["authority"],
    )
except contracts.EvidenceError:
    pass
else:
    raise SystemExit("sealed validator accepted fabricated aggregate authority")
if matrix["suite_id"] != contracts.MATRIX_SUITE_ID or "jsonschema" in sys.modules:
    raise SystemExit("sealed validation used an installed dependency")
print("stdlib-only reviewed schema validation: PASS")
'''
        completed = subprocess.run(
            ["/usr/bin/python3", "-I", "-S", "-c", code, str(ROOT)],
            cwd="/",
            env={"PATH": "/usr/bin:/bin"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "stdlib-only reviewed schema validation: PASS")

    def test_git_identity_supports_real_sha1_and_sha256_repositories_and_rejects_replace_refs(self) -> None:
        previous = os.environ.copy()
        try:
            for object_format in ("sha1", "sha256"):
                with tempfile.TemporaryDirectory(prefix=f"sllm-g1-git-{object_format}-") as temporary:
                    repo = Path(temporary) / "repo"
                    subprocess.run(["/usr/bin/git", "init", "--quiet", "--object-format", object_format, str(repo)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    (repo / "tracked.txt").write_bytes(b"real Git object fixture\n")
                    subprocess.run(["/usr/bin/git", "-C", str(repo), "add", "tracked.txt"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    subprocess.run(["/usr/bin/git", "-C", str(repo), "-c", "user.name=semantic-g1", "-c", "user.email=semantic-g1@example.invalid", "commit", "--quiet", "-m", "fixture"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    reviewed = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-C", str(repo), "rev-parse", "HEAD"], check=True, stdout=subprocess.PIPE).stdout.decode("ascii").strip()
                    tree = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-C", str(repo), "rev-parse", "HEAD^{tree}"], check=True, stdout=subprocess.PIPE).stdout.decode("ascii").strip()
                    os.environ.clear()
                    os.environ.update({"GITHUB_WORKSPACE": str(repo), "GITHUB_SHA": reviewed, "REVIEWED_SHA": reviewed, "TESTED_SHA": reviewed, "WORKFLOW_SHA": reviewed})
                    identity = contracts.verify_repository_identity(repo, {"reviewed_sha": reviewed, "tested_sha": reviewed, "workflow_sha": reviewed})
                    self.assertEqual(identity["git_object_format"], object_format)
                    self.assertEqual(identity["git_oid_width"], 40 if object_format == "sha1" else 64)
                    self.assertEqual(identity["git_tree_oid"], tree)
                    blob, width = contracts.git_blob_oid(repo, reviewed, "tracked.txt")
                    self.assertEqual(len(blob), width)
                    replacement = subprocess.run(["/usr/bin/git", "-C", str(repo), "commit-tree", tree, "-m", "replacement"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ, "GIT_AUTHOR_NAME": "semantic-g1", "GIT_AUTHOR_EMAIL": "semantic-g1@example.invalid", "GIT_COMMITTER_NAME": "semantic-g1", "GIT_COMMITTER_EMAIL": "semantic-g1@example.invalid"}).stdout.decode("ascii").strip()
                    subprocess.run(["/usr/bin/git", "-C", str(repo), "replace", reviewed, replacement], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    with self.assertRaises(contracts.EvidenceError):
                        contracts.verify_repository_identity(repo, {"reviewed_sha": reviewed, "tested_sha": reviewed, "workflow_sha": reviewed})
        finally:
            os.environ.clear()
            os.environ.update(previous)

    def test_workflow_validator_rejects_any_topology_or_argv_append(self) -> None:
        contracts.validate_workflow_registration(ROOT)
        with tempfile.TemporaryDirectory(prefix="sllm-g1-workflow-") as temporary:
            mutated = (ROOT / ".github/workflows/semantic-rmsnorm-g1.yml").read_bytes() + b"\n      - run: echo forged\n"
            self.assertNotEqual(contracts.sha256_bytes(mutated), contracts.SEMANTIC_G1_WORKFLOW_SHA256)


if __name__ == "__main__":
    unittest.main()
