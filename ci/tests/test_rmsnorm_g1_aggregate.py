"""Host-safe checks for the validation-only semantic G1 aggregate surface."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import aggregate_rmsnorm_g1_results as aggregate  # noqa: E402
import validate_rmsnorm_g1_contracts as contracts  # noqa: E402


_FIXTURE_TMP = tempfile.TemporaryDirectory(prefix="sllm-g1-real-git-fixture-")
_FIXTURE_REPO = Path(_FIXTURE_TMP.name) / "repo"
subprocess.run(["/usr/bin/git", "init", "--quiet", str(_FIXTURE_REPO)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
for _relative in contracts.AUTHORITY_SOURCE_FILES:
    _source = ROOT / _relative
    _destination = _FIXTURE_REPO / _relative
    _destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_source, _destination)
subprocess.run(["/usr/bin/git", "-C", str(_FIXTURE_REPO), "add", "--", *contracts.AUTHORITY_SOURCE_FILES], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
subprocess.run(["/usr/bin/git", "-C", str(_FIXTURE_REPO), "-c", "user.name=semantic-g1", "-c", "user.email=semantic-g1@example.invalid", "commit", "--quiet", "-m", "real-fixture"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def tearDownModule() -> None:
    _FIXTURE_TMP.cleanup()


def _identity() -> dict[str, object]:
    reviewed = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-C", str(_FIXTURE_REPO), "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=True).stdout.decode("ascii").strip()
    tree = subprocess.run(["/usr/bin/git", "--no-replace-objects", "-C", str(_FIXTURE_REPO), "rev-parse", "HEAD^{tree}"], stdout=subprocess.PIPE, check=True).stdout.decode("ascii").strip()
    return {"reviewed_sha": reviewed, "tested_sha": reviewed, "workflow_sha": reviewed, "git_tree_oid": tree, "git_object_format": contracts.git_object_format(_FIXTURE_REPO), "git_oid_width": len(reviewed), "worktree_clean": True, "revision_input": "full-sha"}


def _authority(identity: dict[str, object]) -> dict[str, object]:
    sources = []
    for path in contracts.AUTHORITY_SOURCE_FILES:
        blob, _width = contracts.git_blob_oid(_FIXTURE_REPO, str(identity["reviewed_sha"]), path)
        data = contracts._git_output_bytes(_FIXTURE_REPO, ("show", f"{identity['reviewed_sha']}:{path}"))
        sources.append({"path": path, "git_blob_oid": blob, "size_bytes": len(data), "sha256": contracts.sha256_bytes(data)})
    records = {str(record["path"]): record for record in sources}
    return {
        "authority_version": contracts.AUTHORITY_VERSION, "candidate": identity,
        "controller": records["ci/tools/orchestrate_rmsnorm_g1_evidence.py"], "workflow": records[contracts.WORKFLOW_PATH],
        "sources": sources,
        "executables": {"python": contracts.CONTROLLER_PYTHON_RECORD, "compiler": contracts.COMPILER_SOURCE_RECORD, "client_interpreter": contracts.COMPILER_CLIENT_INTERPRETER_RECORD},
        "toolchain": {name: {"path": str(path), "resolved_path": str(path.resolve(strict=False)), "size_bytes": 1, "sha256": "0" * 64} for name, path in contracts.CANONICAL_TOOL_PATHS.items()},
    }


def _compiler_execution() -> dict[str, object]:
    empty = ""; empty_sha = contracts.sha256_bytes(b""); nonce = "c" * 64
    result = {"pid": 200, "starttime": 300, "ppid": 1, "pgrp": 200, "exit_code": 0, "stdout_b64": empty, "stderr_b64": empty, "stdout_sha256": empty_sha, "stderr_sha256": empty_sha, "duration_ns": 1, "timed_out": False, "crashed": False, "invocation": {"cwd": "/tmp", "environment_sha256": "d" * 64, "inputs": [], "outputs": [], "policy": "semantic-g1-canonical-compiler-graph-v1"}, "kernel_limits": {"address_space_bytes": 8 * 1024 * 1024 * 1024, "process_count": 4096, "rss_bytes": 6 * 1024 * 1024 * 1024, "enforced_by": "/usr/bin/prlimit", "rss_enforcement": "kernel-prlimit-plus-parent-sampling-fail-closed-v1"}, "exec_identity": {"pid": 200, "starttime": 300, "ppid": 1, "pgrp": 200, "exe_dev": 1, "exe_ino": 2, "sealed_dev": 1, "sealed_ino": 2, "exe_path": "/proc/200/exe", "argv_sha256": contracts.sha256_json(["--version"]), "cwd": "/tmp", "exec_ready": True}}
    event = {"sequence": 0, "request_nonce": nonce, "client_binding": {"pid": 100, "starttime": 150, "uid": 0, "gid": 0}, "argv": ["--version"], "cwd": "/tmp", "argv_sha256": contracts.sha256_json(["--version"]), "environment_sha256": "d" * 64, "request_frame_sha256": "e" * 64, "response_frame_sha256": "f" * 64, "ack_frame_sha256": "a" * 64, "compiler_source_sha256": contracts.COMPILER_SOURCE_RECORD["sha256"], "compiler": result, "started_at_ns": 1, "finished_at_ns": 2, "acknowledged": True}
    events = [event]
    client = {"path": "/tmp/compiler-client.py", "resolved_path": "/tmp/compiler-client.py", "size_bytes": 1, "sha256": "b" * 64}
    helper = {"path": "/tmp/compiler-exec-helper", "resolved_path": "/tmp/compiler-exec-helper", "size_bytes": 1, "sha256": "9" * 64}
    return {"protocol": "parent-owned-compiler-broker-v1", "event_protocol": "rmsnorm-g1-compiler-broker-v1", "source": contracts.COMPILER_SOURCE_RECORD, "client": client, "exec_helper": helper, "session": "1" * 64, "request_count": 1, "events_sha256": contracts.sha256_json(events), "closure": {"state": "closed", "build_root_pid": 100, "build_root_starttime": 125, "build_root_pgrp": 100, "build_tree_reaped": True, "listener_closed": True, "active_requests": 0, "quiescence_rounds": 3, "state_machine": "new-running-closing-closed-v1", "request_count": 1, "last_sequence": 0, "events_sha256": contracts.sha256_json(events)}, "events": events}


class SemanticG1AggregateTests(unittest.TestCase):
    def _aggregate_document(self) -> dict[str, object]:
        identity = _identity(); authority = _authority(identity); compiler = _compiler_execution()
        row_hashes = {"report_sha256": "a" * 64, "binary_sha256": "b" * 64, "companion_sha256": "c" * 64, "loader_sha256": "d" * 64, "runtime_library_sha256": "e" * 64, "runtime_dependency_closure_sha256": "8" * 64, "raw_frame_sha256": "f" * 64, "compiler_execution_sha256": contracts.sha256_json(compiler), "compiler_execution": compiler, "resource_counts": contracts.EXPECTED_ROW_RESOURCE_COUNTS}
        return {"schema_version": "rmsnorm-semantic-g1-aggregate-v1", "aggregate_id": "rmsnorm-semantic-g1-aggregate-host-unit-1", "suite_id": contracts.MATRIX_SUITE_ID, "tier": "tier_g1", "state": "PASS", "required": True, "run_id": "host-unit", "run_attempt": 1, "candidate": identity, "authority": authority, "contracts": contracts.authority_contract_hashes(authority), "artifact_kind": "rmsnorm-semantic-g1-runtime", "expected_rows": list(contracts.ROWS), "rows": [{"row_id": row_id, "target": row_id.rsplit("-", 1)[1], "state": "PASS", **row_hashes} for row_id in contracts.ROWS], "scope": contracts.EXPECTED_SCOPE, "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": 2, "passed_rows": 2, "failed_rows": 0}, "created_at": "2026-08-05T00:00:00Z"}

    def test_import_and_direct_validation_have_no_pass_authority(self) -> None:
        with self.assertRaisesRegex(aggregate.AggregateError, "permanently fail-closed"):
            aggregate.validate_document_only({"state": "PASS"})
        code = (
            "import sys; from pathlib import Path; sys.path.insert(0, str(Path.cwd() / 'ci/tools')); "
            "import aggregate_rmsnorm_g1_results as a; print(hasattr(a, '_emit_controller_evidence'), hasattr(a, '_controller_row_from_live_frames'), a.main([]))"
        )
        completed = subprocess.run(
            ["/usr/bin/python3", "-c", code], cwd=ROOT,
            env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.decode("utf-8").strip(), "False False 2")
        self.assertIn("emission is disabled", completed.stderr.decode("utf-8"))

    def test_fabricated_document_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-g1-aggregate-") as temporary:
            document = Path(temporary) / "aggregate.json"
            document.write_text(json.dumps({"state": "PASS", "compiler_execution": {"request_count": 5}}), encoding="utf-8")
            completed = subprocess.run(
                ["/usr/bin/python3", str(ROOT / "ci/tools/aggregate_rmsnorm_g1_results.py"), "--document", str(document)],
                cwd=ROOT, env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(completed.returncode, 1)
            self.assertIn(b"FAIL", completed.stderr)
            self.assertEqual(sorted(path.name for path in Path(temporary).iterdir()), ["aggregate.json"])

    def test_controller_validator_rejects_fabricated_authority_before_target_replay(self) -> None:
        document = self._aggregate_document()
        identity = document["candidate"]
        authority = document["authority"]
        with self.assertRaises(contracts.EvidenceError):
            contracts.validate_aggregate_document(document, identity=identity, repo=ROOT, authority=authority)


if __name__ == "__main__":
    unittest.main()
