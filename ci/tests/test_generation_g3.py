import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))
import generation_g3 as g3  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(g3._canonical_bytes(value))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_or_command_key(value: object) -> bool:
    if isinstance(value, dict):
        return any(key in {"path", "paths", "command"} or _path_or_command_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_path_or_command_key(item) for item in value)
    return False


def _max_stop() -> dict[str, object]:
    return {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None}


def _raw_report(matrix: dict[str, object], target: str, case: dict[str, object], *, timing_ns: int = 1, submission_delta: int = 0) -> dict[str, object]:
    golden = case["golden"]  # type: ignore[index]
    input_ids = g3._expand_input_spec(golden["input_token_spec"], "fixture input")  # type: ignore[index]
    audit = dict(golden["audit"])  # type: ignore[index]
    audit["submission_count"] += submission_delta
    return {
        "schema_version": "model-frontend-cli-report-v1",
        "command": "generate",
        "state": "PASS",
        "model": copy.deepcopy(matrix["model"]),
        "scope": {"offline": True, "gpu_execution": True, "model_execution": True, "generation": True},
        "result": {
            "kind": "generate",
            "input_kind": case["input_kind"],
            "input_token_ids": input_ids,
            "generated_token_ids": list(golden["generated_token_ids"]),  # type: ignore[index]
            "visible_token_ids": list(golden["visible_token_ids"]),  # type: ignore[index]
            "decode_input_token_ids": list(golden["decode_input_token_ids"]),  # type: ignore[index]
            "output_text": golden["output_text"],  # type: ignore[index]
            "stop_reason": dict(golden["stop_reason"]),  # type: ignore[index]
            "execution": {
                "selected_backend": "hip", "target": target, "device_index": 0,
                "model_fingerprint": matrix["model"]["lock_fingerprint"],  # type: ignore[index]
                "plan_digest": matrix["plan_digest"], "prefill_tokens": len(input_ids),
                "decode_steps": len(golden["decode_input_token_ids"]),  # type: ignore[index]
                "fallback_used": False, "submission_count": audit["submission_count"],
                "kernel_dispatch_count": audit["kernel_dispatch_count"], "all_dispatches_hip": True,
            },
            "timing_ns": timing_ns,
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
        },
    }


def _observation(target: str) -> dict[str, object]:
    identity = g3.TARGET_IDENTITIES[target]
    return {
        "selected_device": {"target": target, **identity, "logical_device_index": 0},
        "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
        "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
    }


class FakeSeams:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw
        self.environments: list[dict[str, str]] = []
        self.commands: list[list[str]] = []
        self.phases: list[str] = []
        self.capture = {
            "stdout": raw, "stderr": b"", "exit_code": 0, "timed_out": False,
            "duration_ns": 42, "term_sent": False, "kill_sent": False, "process_group_gone": True,
        }

    def execute(self, command: list[str], environment: dict[str, str], cwd: Path | None, timeout: int) -> dict[str, object]:
        self.commands.append(command)
        self.environments.append(dict(environment))
        self.timeout = timeout
        return dict(self.capture)

    def observe(self, target: str, phase: str) -> dict[str, object]:
        self.phases.append(phase)
        return _observation(target)


class GenerationG3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        for relative in ("ci/schema/generation-g3-report-v1.schema.json", "ci/schema/generation-g3-aggregate-v1.schema.json"):
            schema = json.loads((ROOT / relative).read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)

    def _fixture(self, root: Path, *, all_reviewed: bool = True) -> tuple[Path, Path, Path, dict[str, object]]:
        matrix = json.loads((ROOT / "ci/matrix/generation-g3-v1.json").read_text(encoding="utf-8"))
        binaries: list[Path] = []
        for target in g3.TARGETS:
            binary = root / f"{target}.bin"
            binary.write_bytes(f"fixture-{target}".encode("ascii"))
            binary.chmod(0o755)
            binaries.append(binary)
        for index, binary in enumerate(binaries):
            matrix["targets"][index]["binary_sha256"] = _sha256(binary)
        if all_reviewed:
            pending = matrix["cases"][5]
            pending["golden"] = {
                "status": "reviewed",
                "input_token_spec": pending["golden"]["input_token_spec"],
                "generated_token_ids": [264], "visible_token_ids": [264], "decode_input_token_ids": [],
                "output_text": "fixture", "stop_reason": _max_stop(),
                "audit": {"prefill_tokens": pending["input_token_length"], "decode_steps": 0, "submission_count": 10, "kernel_dispatch_count": 11},
            }
        matrix_path = root / "fixture-matrix.json"
        _write_json(matrix_path, matrix)
        return matrix_path, binaries[0], binaries[1], matrix

    def _make_manifest(self, root: Path, matrix_path: Path, matrix: dict[str, object], target: str, case: dict[str, object], order: int, *, submission_delta: int = 0) -> Path:
        raw = _raw_report(matrix, target, case, timing_ns=order + 1, submission_delta=submission_delta)
        raw_bytes = g3._canonical_bytes(raw)
        raw_path = root / f"raw-{target}-{case['id']}.json"
        raw_path.write_bytes(raw_bytes)
        binary = root / f"{target}.bin"
        seams = FakeSeams(raw_bytes)
        manifest = g3.run_case(
            target, case["id"], binary, raw_path,
            matrix["candidate"], matrix_path=matrix_path, run_id=f"g3-run-{order + 1:032x}",
            attempt=1, command_runner=seams.execute, observation_provider=seams.observe,
            test_only_fixture_matrix=True,
        )
        manifest_path = root / f"manifest-{target}-{case['id']}.json"
        _write_json(manifest_path, manifest)
        return manifest_path

    def _all_manifests(self, root: Path) -> tuple[list[Path], Path, dict[str, object]]:
        matrix_path, _binary0, _binary1, matrix = self._fixture(root)
        manifests: list[Path] = []
        order = 0
        for target in g3.TARGETS:
            for case in matrix["cases"]:
                manifests.append(self._make_manifest(root, matrix_path, matrix, target, case, order))
                order += 1
        return manifests, matrix_path, matrix

    def test_schema_and_canonical_matrix_pin_preserve_reviewed_goldens(self) -> None:
        matrix = g3.validate_matrix()
        self.assertEqual([case["input_token_length"] for case in matrix["cases"][:5]], [1, 7, 255, 256, 257])
        self.assertEqual([case["golden"]["status"] for case in matrix["cases"]], ["reviewed"] * 6)
        self.assertEqual(matrix["cases"][5]["golden"]["stop_reason"]["token_id"], 248046)
        self.assertEqual(matrix["candidate"], g3.CANDIDATE)
        self.assertEqual(matrix["model"], g3.MODEL)
        self.assertEqual(matrix["plan_digest"], g3.PLAN_DIGEST)
        self.assertEqual(_sha256(ROOT / "ci/matrix/generation-g3-v1.json"), g3.CANONICAL_MATRIX_SHA256)
        self.assertNotIn("248044", json.dumps(matrix, sort_keys=True))

    def test_schemas_reject_unbounded_audit_process_and_cleanup_integers(self) -> None:
        report_schema = json.loads((ROOT / "ci/schema/generation-g3-report-v1.schema.json").read_text(encoding="utf-8"))
        aggregate_schema = json.loads((ROOT / "ci/schema/generation-g3-aggregate-v1.schema.json").read_text(encoding="utf-8"))
        report_audit = report_schema["$defs"]["audit"]["properties"]
        aggregate_audit = aggregate_schema["$defs"]["audit"]["properties"]
        for audit in (report_audit, aggregate_audit):
            self.assertEqual(audit["submission_count"]["maximum"], 18_446_744_073_709_551_615)
            self.assertEqual(audit["kernel_dispatch_count"]["maximum"], 18_446_744_073_709_551_615)
        self.assertEqual(
            report_schema["$defs"]["process"]["properties"]["residual_runner_children"]["items"]["maximum"],
            4_194_304,
        )
        cleanup = report_schema["$defs"]["manifest_cleanup"]["properties"]
        self.assertEqual(cleanup["retryable_cleanup"]["maximum"], 1)
        self.assertEqual(cleanup["durable_quarantine"]["maximum"], 1)

        with tempfile.TemporaryDirectory() as directory:
            manifests, _matrix_path, _matrix = self._all_manifests(Path(directory))
            row = g3.test_only_normalize_fixture_manifest(manifests[0])
            row["audit"]["submission_count"] = 10**100
            self.assertFalse(Draft202012Validator(report_schema).is_valid(row))
            aggregate = g3.test_only_aggregate_fixture_manifests(manifests)
            aggregate["rows"][0]["audit"]["kernel_dispatch_count"] = 10**100
            self.assertFalse(Draft202012Validator(aggregate_schema).is_valid(aggregate))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            manifest["observations"]["post"]["process"]["residual_runner_children"] = [4_194_305]
            self.assertFalse(Draft202012Validator(report_schema).is_valid(manifest))
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            manifest["cleanup"]["retryable_cleanup"] = 2
            self.assertFalse(Draft202012Validator(report_schema).is_valid(manifest))

    def test_expected_commands_use_all_six_exact_existing_cli_shapes(self) -> None:
        executable = Path("/tmp/sllm-g3")
        case_arguments = {
            "g3-prompt-1": ["--prompt", "Hello", "--max-new-tokens", "8"],
            "g3-prompt-7": ["--prompt", " ".join(["a"] * 7), "--max-new-tokens", "8"],
            "g3-prompt-255": ["--prompt", " ".join(["a"] * 255), "--max-new-tokens", "8"],
            "g3-prompt-256": ["--prompt", " ".join(["a"] * 256), "--max-new-tokens", "8"],
            "g3-prompt-257": ["--prompt", " ".join(["a"] * 257), "--max-new-tokens", "1"],
            "g3-unicode-chat-248046": [
                "--message", g3.UNICODE_MESSAGE, "--thinking", "disabled",
                "--max-new-tokens", "8",
            ],
        }
        self.assertEqual(tuple(case_arguments), g3.CASE_IDS)
        for case_id, arguments in case_arguments.items():
            expected = [
                str(executable), "generate",
                "--lock", str(ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"),
                "--cache", str(g3.MODEL_CACHE_PATH),
                *arguments,
                "--device-index", "0", "--target", "gfx1030", "--greedy",
            ]
            command = g3._expected_command(executable, "gfx1030", case_id)
            self.assertEqual(command, expected, case_id)
            self.assertNotIn("--case-id", command)
            self.assertFalse(any(argument.startswith("--g3-") for argument in command))
            self.assertEqual(
                g3._expected_command(executable, "gfx1201", case_id),
                ["gfx1201" if argument == "gfx1030" else argument for argument in expected],
            )

    def test_run_happy_path_isolates_exact_uuid_and_normalize_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, binary, _other, matrix = self._fixture(root)
            case = matrix["cases"][0]
            raw_path = root / "raw.json"
            raw = _raw_report(matrix, "gfx1030", case)
            seams = FakeSeams(g3._canonical_bytes(raw))
            inherited_ld_library_path = "/opt/rocm/lib:/opt/rocm/lib64"
            with mock.patch.dict(g3.os.environ, {"LD_LIBRARY_PATH": inherited_ld_library_path}, clear=False):
                manifest = g3.run_case(
                    "gfx1030", case["id"], binary, raw_path, matrix["candidate"], matrix_path=matrix_path,
                    run_id="g3-run-00000000000000000000000000000001", command_runner=seams.execute,
                    observation_provider=seams.observe, test_only_fixture_matrix=True,
                )
            self.assertEqual(manifest["state"], "PASS")
            self.assertEqual(seams.timeout, 180)
            self.assertEqual(seams.phases, ["pre", "post"])
            environment = seams.environments[0]
            self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], g3.TARGET_IDENTITIES["gfx1030"]["gpu_uuid"])
            self.assertEqual(environment["LD_LIBRARY_PATH"], inherited_ld_library_path)
            self.assertEqual(set(g3.VISIBILITY_NAMES).intersection(environment), {"ROCR_VISIBLE_DEVICES"})
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            row = g3.test_only_normalize_fixture_manifest(manifest_path)
            self.assertEqual(row["state"], "PASS")
            self.assertEqual(row["device"]["gpu_uuid"], g3.TARGET_IDENTITIES["gfx1030"]["gpu_uuid"])
            self.assertNotIn("path", json.dumps(row, sort_keys=True))
            self.assertNotIn("command", json.dumps(row, sort_keys=True))
            mutated = copy.deepcopy(manifest)
            mutated["command"].append("--case-id")
            _write_json(manifest_path, mutated)
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)

    def test_post_observation_retries_transient_driver_cleanup_lag(self) -> None:
        calls: list[str] = []

        def transient_observer(target: str, phase: str) -> dict[str, object]:
            calls.append(phase)
            observation = _observation(target)
            if phase == "post" and calls.count("post") == 1:
                observation["process"] = {
                    "available": True, "reliable": True, "state": "DIRTY",
                    "gpu_processes": [{"record_sha256": "a" * 64}],
                    "residual_runner_children": [],
                }
            return observation

        target_entry = g3.TARGET_IDENTITIES["gfx1201"]
        with mock.patch.object(g3.time, "sleep") as sleep:
            observed = g3._collect_observation(
                "gfx1201", "post", target_entry, transient_observer,
            )
        self.assertEqual(observed, _observation("gfx1201"))
        self.assertEqual(calls, ["post", "post"])
        sleep.assert_called_once_with(g3.POST_OBSERVATION_INTERVAL_SECONDS)

    def test_stale_report_candidate_binary_and_actual_device_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, binary, _other, matrix = self._fixture(root)
            case = matrix["cases"][0]
            raw_path = root / "raw.json"
            raw = _raw_report(matrix, "gfx1030", case)
            seams = FakeSeams(g3._canonical_bytes(raw))
            manifest = g3.run_case("gfx1030", case["id"], binary, raw_path, matrix["candidate"], matrix_path=matrix_path, run_id="g3-run-00000000000000000000000000000002", command_runner=seams.execute, observation_provider=seams.observe, test_only_fixture_matrix=True)
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            raw_path.write_bytes(g3._canonical_bytes({"stale": True}))
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)
            raw_path.write_bytes(g3._canonical_bytes(raw))
            binary.write_bytes(b"modified-binary")
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)
            binary.write_bytes(b"fixture-gfx1030")
            mutated = copy.deepcopy(manifest)
            mutated["candidate"]["commit"] = "c" * 40
            _write_json(manifest_path, mutated)
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)
            mutated = copy.deepcopy(manifest)
            mutated["observations"]["post"]["selected_device"]["gpu_uuid"] = "GPU-a8e9ddefa2d60f55"
            _write_json(manifest_path, mutated)
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)

    def test_forged_timeout_and_missing_health_process_cleanup_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, binary, _other, matrix = self._fixture(root)
            case = matrix["cases"][0]
            raw_path = root / "raw.json"
            raw = _raw_report(matrix, "gfx1030", case)
            seams = FakeSeams(g3._canonical_bytes(raw))
            manifest = g3.run_case("gfx1030", case["id"], binary, raw_path, matrix["candidate"], matrix_path=matrix_path, run_id="g3-run-00000000000000000000000000000003", command_runner=seams.execute, observation_provider=seams.observe, test_only_fixture_matrix=True)
            manifest_path = root / "manifest.json"
            for mutation in (
                lambda value: value["execution"].__setitem__("timed_out", True),
                lambda value: value["observations"]["pre"]["health"].__setitem__("available", False),
                lambda value: value["observations"]["post"]["process"].__setitem__("state", "DIRTY"),
                lambda value: value["cleanup"].__setitem__("process_group_gone", False),
            ):
                mutated = copy.deepcopy(manifest)
                mutation(mutated)
                _write_json(manifest_path, mutated)
                with self.assertRaises(g3.G3Error):
                    g3.test_only_normalize_fixture_manifest(manifest_path)
            forged_raw = copy.deepcopy(raw)
            forged_raw["result"]["timing_ns"] = g3.TIMEOUT_NS + 1
            raw_path.write_bytes(g3._canonical_bytes(forged_raw))
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(manifest_path)

    def test_run_records_external_timeout_and_empty_stderr_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, binary, _other, matrix = self._fixture(root)
            case = matrix["cases"][0]
            raw_path = root / "raw.json"
            raw = _raw_report(matrix, "gfx1030", case)
            seams = FakeSeams(g3._canonical_bytes(raw))
            seams.capture.update({"timed_out": True, "term_sent": True, "kill_sent": True, "process_group_gone": True, "exit_code": -9})
            manifest = g3.run_case("gfx1030", case["id"], binary, raw_path, matrix["candidate"], matrix_path=matrix_path, run_id="g3-run-00000000000000000000000000000004", command_runner=seams.execute, observation_provider=seams.observe, test_only_fixture_matrix=True)
            self.assertEqual(manifest["state"], "FAIL")
            self.assertIn("timeout", manifest["failure_reason"])
            _write_json(root / "manifest.json", manifest)
            with self.assertRaises(g3.G3Error):
                g3.test_only_normalize_fixture_manifest(root / "manifest.json")
            seams.capture.update({"timed_out": False, "term_sent": False, "kill_sent": False, "exit_code": 0, "stderr": b"diagnostic"})

    def test_alternate_matrix_requires_explicit_fixture_api_and_row_never_records_canonical_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix_path, binary, _other, matrix = self._fixture(root)
            with self.assertRaises(g3.G3Error):
                g3.validate_matrix(matrix_path)
            self.assertEqual(g3.test_only_validate_fixture_matrix(matrix_path)["matrix_id"], "generation-g3-v1")
            case = matrix["cases"][0]
            raw_path = root / "raw.json"
            raw = _raw_report(matrix, "gfx1030", case)
            seams = FakeSeams(g3._canonical_bytes(raw))
            manifest = g3.run_case("gfx1030", case["id"], binary, raw_path, matrix["candidate"], matrix_path=matrix_path, run_id="g3-run-00000000000000000000000000000005", command_runner=seams.execute, observation_provider=seams.observe, test_only_fixture_matrix=True)
            manifest_path = root / "manifest.json"
            _write_json(manifest_path, manifest)
            row = g3.test_only_normalize_fixture_manifest(manifest_path)
            self.assertEqual(row["matrix"], {"matrix_id": "generation-g3-v1", "sha256": _sha256(matrix_path)})
            self.assertNotIn("path", row["matrix"])

    def test_aggregate_reopens_all_manifests_revalidates_all_golden_audits_and_rejects_duplicate_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests, _matrix_path, _matrix = self._all_manifests(root)
            aggregate = g3.test_only_aggregate_fixture_manifests(manifests)
            self.assertEqual(aggregate["state"], "PASS")
            self.assertEqual(aggregate["counts"]["reviewed_audit_comparisons"], 48)
            self.assertFalse(_path_or_command_key(aggregate))
            with self.assertRaises(g3.G3Error):
                g3.test_only_aggregate_fixture_manifests(manifests[:-1])
            with self.assertRaises(g3.G3Error):
                g3.test_only_aggregate_fixture_manifests([*manifests[:-1], manifests[0]])

            mutated = json.loads(manifests[0].read_text(encoding="utf-8"))
            raw_path = Path(mutated["raw_report"]["path"])
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            raw["result"]["execution"]["submission_count"] += 1
            raw_path.write_bytes(g3._canonical_bytes(raw))
            mutated["raw_report"]["sha256"] = _sha256(raw_path)
            mutated["raw_report"]["bytes"] = raw_path.stat().st_size
            tampered_manifest = root / "tampered-manifest.json"
            _write_json(tampered_manifest, mutated)
            tampered_paths = [tampered_manifest, *manifests[1:]]
            with self.assertRaises(g3.G3Error):
                g3.test_only_aggregate_fixture_manifests(tampered_paths)

    def test_standalone_normalized_row_cannot_be_aggregated_as_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests, _matrix_path, _matrix = self._all_manifests(root)
            row = g3.test_only_normalize_fixture_manifest(manifests[0])
            row_path = root / "row.json"
            _write_json(row_path, row)
            with self.assertRaises(g3.G3Error):
                g3.test_only_aggregate_fixture_manifests([row_path, *manifests[1:]])

    def test_manifest_duplicate_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"schema_version":"generation-g3-report-v1","schema_version":"generation-g3-report-v1"}\n', encoding="utf-8")
            with self.assertRaises(g3.G3Error):
                g3._read_json(path, "duplicate", 1024)

    def test_cli_validate_matrix_is_host_only(self) -> None:
        self.assertEqual(g3.main(["validate-matrix"]), 0)


if __name__ == "__main__":
    unittest.main()
