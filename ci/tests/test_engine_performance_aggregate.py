from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes  # noqa: E402
import aggregate_engine_performance as aggregator  # noqa: E402
import engine_performance_common as contracts  # noqa: E402
from ci.tests.test_engine_performance_schema import build_identity_for, evidence_for, monitor_capture_for, result_for  # noqa: E402
import run_engine_performance as runner  # noqa: E402


def _observation(target: str) -> dict[str, object]:
    return {
        "selected_device": contracts.expected_device(target),
        "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
        "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
    }


def _allowed_process_record(
    pid: int = 4242, *, name: str = "inert-target", gtt: int = 4096,
    vram: int = 4096, gfx: int = 0, diagnostic: int = 0,
) -> dict[str, object]:
    measurement = lambda value, unit: {"value": value, "unit": unit}
    return {
        "process_info": {
            "name": name,
            "pid": pid,
            "memory_usage": {
                "gtt_mem": measurement(gtt, "B"),
                "cpu_mem": measurement(diagnostic, "B"),
                "vram_mem": measurement(vram, "B"),
            },
            "mem_usage": measurement(diagnostic, "B"),
            "usage": {"gfx": measurement(gfx, "ns"), "enc": measurement(diagnostic, "ns")},
            "sdma_usage": measurement(diagnostic, "us"),
            "cu_occupancy": diagnostic,
            "evicted_time": measurement(diagnostic, "ms"),
        },
    }


def _allowed_observation(target: str, record: dict[str, object]) -> dict[str, object]:
    observation = _observation(target)
    observation["process"]["gpu_processes"] = runner._allowed_process_observation(  # type: ignore[index]
        [record], (record["process_info"]["pid"],),  # type: ignore[index]
    )
    return observation


def _manifest(root: Path, row: dict[str, object], *, raw_mutation=None) -> Path:
    result = result_for(row)
    if raw_mutation:
        raw_mutation(result)
    raw_path = root / f"{row['row_id']}.json"
    raw_bytes = canonical_bytes(result)
    raw_path.write_bytes(raw_bytes)
    model = contracts.expected_model(row["model_size"])
    matrix, matrix_digest = contracts.load_matrix()
    manifest = {
        "benchmark_schema_version": "engine-performance-direct-v1",
        "record_kind": "evidence_manifest",
        "state": "PASS",
        "required": False,
        "failure_reason": None,
        "row_id": row["row_id"],
        "claims": contracts.CLAIMS,
        "matrix": {"path": str(contracts.MATRIX_PATH), "matrix_id": "engine-performance-direct-v1", "sha256": matrix_digest},
        "binary": {"path": str(root / "binary"), "sha256": "1" * 64, "bytes": 1},
        "build_identity": build_identity_for(row),
        "model_lock": {"path": str(root / "lock.json"), "sha256": "2" * 64, "fingerprint": model["lock_fingerprint"]},
        "model_cache": {"path": str(root / "cache"), "sha256": "3" * 64},
        "raw_artifact": {"path": str(raw_path), "sha256": hashlib.sha256(raw_bytes).hexdigest(), "bytes": len(raw_bytes)},
        "observations": {"pre": _observation(row["target"]), "post": _observation(row["target"])},
        "evidence": {
            "version": "engine-performance-evidence-v1", "cadence_seconds": 1,
            "tool": {"path": "/opt/rocm/core-7.14/bin/amd-smi", "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"},
            "definitions": {"clock_variation": "Dynamic clock min/max is observational; no numeric threshold is a violation.", "violation": "When violation accumulators are unavailable, aggregate THROTTLED status is observational; ECC, published thermal/power limits, and exposed active violations remain fail-closed.", "process_ownership": "Every during sample must name only descendants of the benchmark process group."},
            "pre": {key: evidence_for(row["target"])[key] for key in ("static", "metric", "vram_auxiliary", "process_state")},
            "during": {"sample_count": 1, "sample_digest": "sha256:" + "0" * 64, "first": {"timestamp_ns": 1, "metric": evidence_for(row["target"])["metric"], "vram_auxiliary": evidence_for(row["target"])["vram_auxiliary"], "process": {"state": "OWNED", "pids": [123]}, "loader_path_digest": "sha256:" + "0" * 64, "violation": {"power_statuses": ["UNTHROTTLED"], "explicit_violation": False, "accumulator_available": False, "accumulator_reason": "test", "accumulator_digest": "sha256:" + "0" * 64}}, "last": {"timestamp_ns": 1, "metric": evidence_for(row["target"])["metric"], "vram_auxiliary": evidence_for(row["target"])["vram_auxiliary"], "process": {"state": "OWNED", "pids": [123]}, "loader_path_digest": "sha256:" + "0" * 64, "violation": {"power_statuses": ["UNTHROTTLED"], "explicit_violation": False, "accumulator_available": False, "accumulator_reason": "test", "accumulator_digest": "sha256:" + "0" * 64}}, "summary": {"sample_count": 1, "temperature_hotspot_c": {"min": 40, "max": 40}, "temperature_mem_c": {"min": 37, "max": 37}, "gfx_clock_mhz": {"min": 1500, "max": 1500}, "mem_clock_mhz": {"min": 1000, "max": 1000}, "power_w": {"min": 100, "max": 100}, "vram_used_mb": {"min": 1000, "max": 1000}, "vram_aux_used_mb": {"min": 1000, "max": 1000}, "perf_levels": ["AMDSMI_DEV_PERF_LEVEL_AUTO"]}, "process_sample_digest": "sha256:" + "0" * 64, "loader": {"required_rocm_release": "7.14.0", "expected_root": "/opt/rocm/core-7.14", "resolved_paths": ["/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000", "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0"], "path_digest": "sha256:" + "0" * 64, "library_digests": {"/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000": "0" * 64, "/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0": "0" * 64}, "process_ids": [123]}, "violation": {"power_statuses": ["UNTHROTTLED"], "explicit_violation": False, "accumulator_available": False, "accumulator_reason": "test", "accumulator_digest": "sha256:" + "0" * 64}},
            "post": {key: evidence_for(row["target"])[key] for key in ("static", "metric", "vram_auxiliary", "process_state")},
            "checks": {"exact_identity": True, "static_identity_unchanged": True, "profile_unchanged": True, "limits_unchanged": True, "performance_level_unchanged": True, "explicit_violation": False, "vram_auxiliary_complete": True, "process_ownership": True, "loader_paths_verified": True, "monitor_errors": 0, "process_group_cleanup": True},
        },
        "execution": {"exit_code": 0, "timed_out": False, "timeout_seconds": row["timeout_seconds"], "stderr_bytes": 0, "term_sent": False, "kill_sent": False, "process_group_gone": True},
        "cleanup": {"pre_process_clean": True, "post_process_clean": True, "process_group_gone": True, "retryable_cleanup": 0, "durable_quarantine": 0},
    }
    manifest["evidence"] = runner._build_evidence(
        evidence_for(row["target"]), evidence_for(row["target"]), monitor_capture_for(row["target"]), row["target"],
        {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"},
    )
    path = root / f"{row['row_id']}.manifest.json"
    path.write_bytes(canonical_bytes(manifest))
    return path


class EnginePerformanceAggregateTests(unittest.TestCase):
    def _manifests(self, root: Path) -> list[Path]:
        matrix, _ = contracts.load_matrix()
        return [_manifest(root, row) for row in matrix["rows"]]

    def test_aggregate_recomputes_all_rows_stats_and_graph(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-aggregate-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            summary = aggregator.aggregate_manifests(manifests, root / "out", verify_external_digests=False)
            self.assertEqual(summary["state"], "PASS")
            self.assertEqual(summary["counts"], {"expected_rows": 22, "collected_rows": 22, "passed_rows": 22, "expected_samples": 220, "collected_samples": 220})
            self.assertEqual(len(summary["rows"]), 22)
            self.assertEqual(summary["rows"][0]["metrics"]["ttft_ns"]["count"], 10)
            self.assertEqual(summary["rows"][0]["metrics"]["tpot_ns"]["count"], 0)
            self.assertEqual(summary["rows"][0]["metrics"]["resident_vram_bytes"]["count"], 1)
            self.assertEqual(summary["rows"][0]["metrics"]["peak_vram_bytes"]["count"], 1)
            self.assertEqual(summary["identity"]["source"]["source_base_revision"], subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip())
            self.assertEqual(summary["identity"]["build_identity_by_target"]["gfx1030"]["target"], "gfx1030")
            self.assertEqual(summary["identity"]["build_identity_by_target"]["gfx1201"]["build_configuration"]["CMAKE_HIP_ARCHITECTURES"], "gfx1201")
            self.assertTrue((root / "out/summary.json").is_file())
            self.assertTrue((root / "out/graph.csv").is_file())
            self.assertTrue((root / "out/summary.json.sha256").is_file())
            self.assertTrue((root / "out/graph.csv.sha256").is_file())
            self.assertTrue((root / "out/bundle.complete.json").is_file())

    def test_allowlisted_inert_process_accepts_diagnostic_drift_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-allowlisted-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            document = json.loads(manifests[0].read_text(encoding="utf-8"))
            target = document["build_identity"]["target"]
            document["observations"] = {
                "pre": _allowed_observation(target, _allowed_process_record(diagnostic=1)),
                "post": _allowed_observation(target, _allowed_process_record(diagnostic=77)),
            }
            manifests[0].write_bytes(canonical_bytes(document))
            summary = aggregator.aggregate_manifests(
                manifests, root / "out", verify_external_digests=False,
            )
            self.assertEqual(summary["state"], "PASS")

    def test_allowlisted_process_authorization_drift_is_rejected(self) -> None:
        for label, mutation in {
            "pid": lambda record: record["process_info"].__setitem__("pid", 4243),
        }.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="sllm-performance-auth-drift-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                target = document["build_identity"]["target"]
                pre_record = _allowed_process_record()
                post_record = copy.deepcopy(pre_record)
                mutation(post_record)
                document["observations"]["pre"] = _allowed_observation(target, pre_record)
                document["observations"]["post"] = _allowed_observation(target, post_record)
                manifests[0].write_bytes(canonical_bytes(document))
                with self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "out", verify_external_digests=False)

    def test_allowlisted_process_independently_validated_inert_record_drift_is_accepted(self) -> None:
        mutations = {
            "name": lambda record: record["process_info"].__setitem__("name", "changed-target"),
            "gtt": lambda record: record["process_info"]["memory_usage"].__setitem__("gtt_mem", {"value": 8192, "unit": "B"}),
            "vram": lambda record: record["process_info"]["memory_usage"].__setitem__("vram_mem", {"value": 8192, "unit": "B"}),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="sllm-performance-inert-drift-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                target = document["build_identity"]["target"]
                pre_record = _allowed_process_record()
                post_record = copy.deepcopy(pre_record)
                mutation(post_record)
                document["observations"]["pre"] = _allowed_observation(target, pre_record)
                document["observations"]["post"] = _allowed_observation(target, post_record)
                manifests[0].write_bytes(canonical_bytes(document))
                summary = aggregator.aggregate_manifests(
                    manifests, root / "out", verify_external_digests=False,
                )
                self.assertEqual(summary["state"], "PASS")

    def test_allowlisted_process_malformed_extra_and_health_evidence_is_rejected(self) -> None:
        def set_nonzero_gfx(observation: dict) -> None:
            entry = observation["process"]["gpu_processes"][1]
            entry["record"]["process_info"]["usage"]["gfx"] = {"value": 1, "unit": "ns"}
            entry["record_sha256"] = hashlib.sha256(canonical_bytes(entry["record"])).hexdigest()

        mutations = {
            "malformed digest": lambda observation: observation["process"]["gpu_processes"][1].__setitem__("record_sha256", "0" * 64),
            "out-of-range allowlist PID": lambda observation: observation["process"]["gpu_processes"][0].__setitem__("allowlisted_pids", [runner.MAX_LINUX_PID + 1]),
            "unallowlisted extra": lambda observation: observation["process"]["gpu_processes"].append({
                "record": _allowed_process_record(pid=5000),
                "record_sha256": hashlib.sha256(canonical_bytes(_allowed_process_record(pid=5000))).hexdigest(),
            }),
            "nonzero gfx": set_nonzero_gfx,
            "changed health": lambda observation: observation["health"].__setitem__("state", "ERROR"),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="sllm-performance-malformed-process-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                target = document["build_identity"]["target"]
                observation = _allowed_observation(target, _allowed_process_record())
                mutation(observation)
                document["observations"]["pre"] = observation
                document["observations"]["post"] = _allowed_observation(target, _allowed_process_record())
                manifests[0].write_bytes(canonical_bytes(document))
                with self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "out", verify_external_digests=False)

    def test_cross_row_source_and_complete_build_identity_mixing_is_rejected(self) -> None:
        parent_revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^"], text=True).strip()
        parent_tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", f"{parent_revision}^{{tree}}"], text=True).strip()
        mutations = (
            lambda value: value["build_identity"].__setitem__("source_base_revision", parent_revision),
            lambda value: value["build_identity"].__setitem__("semantic_tree", parent_tree),
            lambda value: value["build_identity"].__setitem__("sha256", "4" * 64),
            lambda value: value["build_identity"].__setitem__("build_inputs_digest", "sha256:" + "4" * 64),
        )
        for mutation in mutations:
            with tempfile.TemporaryDirectory(prefix="sllm-performance-mixed-identity-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                mutation(document)
                manifests[0].write_bytes(canonical_bytes(document))
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "out", verify_external_digests=False)

    def test_validation_failure_publishes_nothing_and_sidecars_are_no_replace(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-atomic-validation-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            output = root / "out"
            with mock.patch.object(aggregator, "schema_validate", side_effect=ContractError("synthetic schema failure")):
                with self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, output, verify_external_digests=False)
            self.assertFalse(output.exists())

        for sidecar in ("summary.json.sha256", "graph.csv.sha256"):
            with tempfile.TemporaryDirectory(prefix="sllm-performance-sidecar-no-replace-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                output = root / "out"
                output.mkdir()
                existing = output / sidecar
                existing.write_bytes(b"existing-sidecar\n")
                with self.subTest(sidecar=sidecar), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, output, verify_external_digests=False)
                self.assertEqual(existing.read_bytes(), b"existing-sidecar\n")
                self.assertEqual(sorted(path.name for path in output.iterdir()), [sidecar])

    def test_publication_failure_rolls_back_members_and_never_commits(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-publication-failure-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            output = root / "out"
            real_link = contracts.os.link
            link_count = 0

            def fail_before_commit(source: Path, destination: Path) -> None:
                nonlocal link_count
                link_count += 1
                if link_count == 5:
                    raise OSError("synthetic completion publication failure")
                real_link(source, destination)

            with mock.patch.object(contracts.os, "link", side_effect=fail_before_commit):
                with self.assertRaisesRegex(ContractError, "cannot publish"):
                    aggregator.aggregate_manifests(manifests, output, verify_external_digests=False)
            self.assertFalse((output / contracts.BUNDLE_COMMIT_NAME).exists())
            self.assertEqual(list(output.iterdir()), [])

    def test_missing_duplicate_and_stale_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-aggregate-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            with self.assertRaises(ContractError):
                aggregator.aggregate_manifests(manifests[:-1], root / "missing", verify_external_digests=False)
            with self.assertRaises(ContractError):
                aggregator.aggregate_manifests(manifests[:-1] + [manifests[0]], root / "duplicate", verify_external_digests=False)
            stale = json.loads(manifests[0].read_text(encoding="utf-8")); stale["matrix"]["sha256"] = "f" * 64
            manifests[0].write_bytes(canonical_bytes(stale))
            with self.assertRaises(ContractError):
                aggregator.aggregate_manifests(manifests, root / "stale", verify_external_digests=False)

    def test_manifest_missing_evidence_or_build_identity_is_rejected(self) -> None:
        for field in ("evidence", "build_identity"):
            with tempfile.TemporaryDirectory(prefix="sllm-performance-manifest-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                document.pop(field)
                manifests[0].write_bytes(canonical_bytes(document))
                with self.subTest(field=field), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / field, verify_external_digests=False)

    def test_manifest_build_configuration_drift_is_rejected(self) -> None:
        for mutation in (
            lambda value: value["build_identity"]["build_configuration"].pop("cargo_profile"),
            lambda value: value["build_identity"]["build_configuration"].__setitem__("unknown", "value"),
            lambda value: value["build_identity"]["build_configuration"].__setitem__("cargo_command", "cargo build --release"),
        ):
            with tempfile.TemporaryDirectory(prefix="sllm-performance-build-config-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                mutation(document)
                manifests[0].write_bytes(canonical_bytes(document))
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "bad-config", verify_external_digests=False)

    def test_manifest_evidence_target_throttle_and_loader_drift_are_rejected(self) -> None:
        mutations = (
            lambda value: value["evidence"]["pre"]["static"].__setitem__("gpu_bdf", "0000:07:00.0"),
            lambda value: value["evidence"]["post"]["static"].__setitem__("profile", "tampered-profile"),
            lambda value: value["evidence"]["post"]["static"]["limits"].__setitem__("generation", "tampered"),
            lambda value: value["evidence"]["post"]["metric"].__setitem__("perf_level", "AMDSMI_DEV_PERF_LEVEL_HIGH"),
            lambda value: value["evidence"]["during"]["first"]["metric"]["temperature_c"].__setitem__("hotspot", 100),
            lambda value: value["evidence"]["during"]["first"]["metric"].__setitem__("power_w", 251),
            lambda value: value["evidence"]["during"]["loader"]["resolved_paths"].__setitem__(0, "/opt/rocm/foreign/libamdhip64.so"),
        )
        for mutation in mutations:
            with tempfile.TemporaryDirectory(prefix="sllm-performance-evidence-") as directory:
                root = Path(directory)
                manifests = self._manifests(root)
                document = json.loads(manifests[0].read_text(encoding="utf-8"))
                mutation(document)
                manifests[0].write_bytes(canonical_bytes(document))
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "evidence-drift", verify_external_digests=False)

    def test_wrong_math_wrong_model_target_and_tampered_raw_are_rejected(self) -> None:
        for mutation in (
            lambda value: value["measured"]["samples"][0]["derived"].__setitem__("e2e_ns", 1),
            lambda value: value["identities"]["model"].__setitem__("model_size", "9B"),
            lambda value: value["identities"].__setitem__("target", "gfx1201"),
        ):
            with tempfile.TemporaryDirectory(prefix="sllm-performance-aggregate-") as directory:
                root = Path(directory)
                matrix, _ = contracts.load_matrix()
                manifests = self._manifests(root)
                changed = _manifest(root, matrix["rows"][0], raw_mutation=mutation)
                manifests[0] = changed
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    aggregator.aggregate_manifests(manifests, root / "bad", verify_external_digests=False)

        with tempfile.TemporaryDirectory(prefix="sllm-performance-aggregate-") as directory:
            root = Path(directory)
            manifests = self._manifests(root)
            raw_path = Path(json.loads(manifests[0].read_text(encoding="utf-8"))["raw_artifact"]["path"])
            raw_path.write_bytes(raw_path.read_bytes() + b" ")
            with self.assertRaises(ContractError):
                aggregator.aggregate_manifests(manifests, root / "tampered", verify_external_digests=False)


if __name__ == "__main__":
    unittest.main()
