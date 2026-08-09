from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes, sha256_file, sha256_json  # noqa: E402
import run_rmsnorm_p0_runtime as runner  # noqa: E402
import validate_rmsnorm_p0_contracts as contracts  # noqa: E402


def candidate() -> dict[str, object]:
    return {"reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40, "worktree_clean": True, "revision_input": "full-sha"}


def prerequisites(target: str, value: dict[str, object]) -> list[dict[str, object]]:
    rows = {
        "g0": f"g0-{target}", "private_g1": f"g1-{target}",
        "semantic_g1": f"rmsnorm-semantic-g1-{target}",
        "g2": f"rmsnorm-g2-{target}", "h3": f"h3-rmsnorm-{target}",
    }
    digest = contracts.candidate_sha256(value)
    return [
        {"kind": kind, "row_id": rows[kind], "state": "bound-not-executed-by-p0", "candidate_sha256": digest, "artifact_sha256": f"{index + 1:x}" * 64, "report_sha256": f"{index + 6:x}" * 64}
        for index, kind in enumerate(contracts.PREREQUISITE_KINDS)
    ]


def write_artifact(root: Path, target: str, value: dict[str, object] | None = None) -> tuple[Path, Path, dict[str, object]]:
    value = value or candidate()
    root.mkdir(parents=True, exist_ok=True)
    binary = root / contracts.P0_BINARY
    binary.write_bytes(b"p0-host-scaffold-producer\n")
    binary.chmod(0o755)
    binary_sha = sha256_file(binary)
    sidecar = root / contracts.P0_SIDECAR
    sidecar.write_bytes(f"{binary_sha}  {contracts.P0_BINARY}\n".encode("ascii"))
    document = {
        "schema_version": "rmsnorm-p0-artifact-v1",
        "artifact_id": f"rmsnorm-p0-{target}-{binary_sha}",
        "row_id": f"rmsnorm-p0-{target}", "target": target, "candidate": value,
        "binary": {"role": contracts.P0_BINARY_ROLE, "path": contracts.P0_BINARY, "sidecar_path": contracts.P0_SIDECAR, "size_bytes": binary.stat().st_size, "sha256": binary_sha, "sidecar_sha256": sha256_file(sidecar)},
        "build": {"builder": "ci/tools/build_rmsnorm_p0_runtime.py", "command": list(contracts.P0_BUILD_COMMAND), "profile": "release", "binary_name": contracts.P0_BINARY, "output_path": contracts.P0_BINARY, "fresh_output": True, "substitution_rejected": True, "environment": contracts.p0_build_environment(target)},
        "source_set": contracts.source_set(ROOT),
        "execution_contract": {"public_path": contracts.PUBLIC_PATH, "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": 256, "timing_contract": "rmsnorm-p0-timing-v1", "dtype": dict(contracts.DTYPE_CONTRACT), "producer_status": contracts.PRODUCER_STATUS},
        "scope": {"selected_backend": "hip", "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "hip_only": True, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False},
        "prerequisites": prerequisites(target, value),
    }
    path = root / "artifact.json"
    path.write_bytes(canonical_bytes(document))
    return path, binary, document


def ok_health(target: str, ras: int = 0) -> dict[str, object]:
    return {"available": True, "reliable": True, "state": "OK", "target": target, "ras_uncorrectable_count": ras}


def clean_process() -> dict[str, object]:
    return {"state": "CLEAN", "residual_runner_children": [], "gpu_processes": []}


def runtime_result(target: str, artifact_document: dict[str, object], artifact_path: Path, value: dict[str, object] | None = None) -> dict[str, object]:
    value = value or candidate()
    next_dispatch_id = 1
    cases: list[dict[str, object]] = []
    for order, (case_id, rows, n, seed, classification) in enumerate(contracts.CASE_SPECS):
        warmups = []
        for iteration in range(contracts.WARMUP_ITERATIONS):
            warmups.append({"iteration": iteration, "dispatch_id": next_dispatch_id, "dispatch_count": 1, "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "fallback_used": False})
            next_dispatch_id += 1
        samples = []
        for iteration in range(contracts.MEASUREMENT_ITERATIONS):
            kernel = 1_000 + order * 100 + iteration
            samples.append({"iteration": iteration, "dispatch_id": next_dispatch_id, "dispatch_count": 1, "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "fallback_used": False, "kernel_latency_ns": kernel, "wall_latency_ns": kernel + 500})
            next_dispatch_id += 1
        kernel_values = [sample["kernel_latency_ns"] for sample in samples]
        wall_values = [sample["wall_latency_ns"] for sample in samples]
        cases.append({
            "order": order, "id": case_id, "rows": rows, "n": n, "input_seed": seed,
            "classification": classification, "state": "PASS",
            "warmup_dispatches": warmups, "samples": samples,
            "summary": {"kernel_median_ns": contracts._median(kernel_values), "kernel_mad_ns": contracts._mad(kernel_values), "wall_median_ns": contracts._median(wall_values), "wall_mad_ns": contracts._mad(wall_values), "sample_set_sha256": sha256_json(samples)},
        })
    device = next(item["device"] for item in contracts.expected_matrix()["targets"] if item["target"] == target)
    document = {
        "schema_version": "rmsnorm-p0-runtime-result-v1", "state": "PASS",
        "row_id": f"rmsnorm-p0-{target}", "target": target, "candidate": value,
        "artifact": contracts.artifact_summary(artifact_document, sha256_file(artifact_path)),
        "matrix": {"path": contracts.MATRIX_PATH, "sha256": sha256_file(ROOT / contracts.MATRIX_PATH)},
        "case_set_sha256": contracts.case_set_sha256(ROOT),
        "model_lock": {"path": contracts.MODEL_LOCK_PATH, "sha256": sha256_file(ROOT / contracts.MODEL_LOCK_PATH), "fingerprint": contracts.MODEL_LOCK_FINGERPRINT, "resolved_revision": contracts.RESOLVED_REVISION, "used": False},
        "source_set_sha256": contracts.source_set(ROOT)["sha256"],
        "dtype": dict(contracts.DTYPE_CONTRACT),
        "scope": {"selected_backend": "hip", "gpu_execution": True, "public_rmsnorm_path": True, "semantic_op_used": True, "model_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False},
        "device": {**device, "target": target}, "timing": contracts.expected_matrix()["timing"],
        "dispatch": {"backend": "hip", "kernel_id": 1, "kernel_symbol": "rmsnorm.baseline.wave32.v1", "device_symbol": "sllm_rmsnorm_baseline_wave32_v1", "workgroup_size_x": 256, "dispatch_count": contracts.TOTAL_DISPATCHES, "fallback_allowed": False, "fallback_used": False},
        "cases": cases, "measurement_sha256": sha256_json(cases),
    }
    return document


class P0RunnerTests(unittest.TestCase):
    def _args(self, root: Path, target: str = "gfx1030") -> tuple[Namespace, dict[str, object]]:
        artifact_path, binary, artifact_document = write_artifact(root, target)
        args = Namespace(
            repo=ROOT, target=target, artifact=artifact_path, binary=binary,
            output_dir=root / "out", run_id="p0-test-run", run_attempt=1,
            reviewed_sha="a" * 40, tested_sha="a" * 40, workflow_sha="a" * 40,
            tree_oid="b" * 40, health_pre=None, health_post=None,
            process_pre=None, process_post=None,
        )
        return args, artifact_document

    def test_host_runner_never_executes_or_fabricates_numeric_pass(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-runner-") as directory:
            args, _ = self._args(Path(directory))
            with patch.dict(os.environ, {}, clear=False), patch.object(runner.subprocess, "run") as invoked:
                os.environ.pop("SLLM_P0_GPU_EXECUTION", None)
                report = runner.run_row(args)
            invoked.assert_not_called()
            self.assertEqual(report["state"], "FAIL")
            self.assertEqual(report["collection"]["collected_cases"], 0)
            contracts.validate_report(report)

    def test_complete_runtime_values_are_passed_only_with_clean_external_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-runner-") as directory:
            root = Path(directory)
            args, artifact_document = self._args(root)
            result = runtime_result(args.target, artifact_document, args.artifact)
            completed = runner.subprocess.CompletedProcess([], 0, canonical_bytes(result), b"")
            expected_device = next(
                item["device"] for item in contracts.expected_matrix()["targets"]
                if item["target"] == args.target
            )
            observed = (
                ok_health(args.target), clean_process(),
                {
                    "hip_id": expected_device["physical_hip_index"],
                    "bdf": expected_device["bdf"],
                    "uuid": expected_device["uuid"],
                    "product": expected_device["product"],
                },
            )
            events: list[str] = []

            def observe(*_args: object) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
                events.append("observe")
                return observed

            def execute(*_args: object, **_kwargs: object) -> object:
                events.append("execute")
                return completed

            with patch.dict(os.environ, {"SLLM_P0_GPU_EXECUTION": "1"}), patch.object(runner, "_observe_live", side_effect=observe), patch.object(runner.subprocess, "run", side_effect=execute) as invoked:
                report = runner.run_row(args)
            self.assertEqual(invoked.call_count, 1)
            self.assertEqual(events, ["observe", "execute", "observe"])
            self.assertEqual(invoked.call_args.kwargs["env"]["HIP_VISIBLE_DEVICES"], "1")
            self.assertEqual(
                invoked.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                contracts.P0_RUNTIME_LD_LIBRARY_PATH,
            )
            self.assertIn("--physical-hip-index", invoked.call_args.args[0])
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(report["collection"]["collected_cases"], 5)
            self.assertEqual(report["dispatch"]["dispatch_count"], 130)
            self.assertIn("complete dedicated producer", report["execution"]["failure_reason"])
            contracts.validate_report(report)

    def test_canonical_runner_rejects_precomputed_observation_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-observation-") as directory:
            args, _ = self._args(Path(directory))
            args.health_pre = Path(directory) / "health-pre.json"
            with patch.dict(os.environ, {"SLLM_P0_GPU_EXECUTION": "1"}), patch.object(
                runner, "_observe_live"
            ) as observe:
                with self.assertRaises(ContractError):
                    runner.run_row(args)
            observe.assert_not_called()

    def test_runtime_rejects_non_gpu_zero_dispatch_fallback_and_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-runtime-") as directory:
            root = Path(directory)
            args, artifact_document = self._args(root)
            base = runtime_result(args.target, artifact_document, args.artifact)
            mutations = []
            for section, key, value in (
                ("scope", "gpu_execution", False),
                ("scope", "fallback_used", True),
                ("dispatch", "dispatch_count", 0),
                ("artifact", "binary_sha256", "f" * 64),
                ("model_lock", "sha256", "f" * 64),
            ):
                changed = copy.deepcopy(base)
                changed[section][key] = value
                mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["cases"].reverse()
            changed["measurement_sha256"] = sha256_json(changed["cases"])
            mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["device"]["bdf"] = "0000:00:00.0"
            mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["dtype"]["accumulation"] = "BF16"
            mutations.append(changed)
            for changed in mutations:
                with self.subTest(changed=changed), self.assertRaises(ContractError):
                    contracts.validate_runtime_result(changed, artifact_document, sha256_file(args.artifact))

    def test_runtime_rejects_nan_inf_negative_inconsistent_time_and_dispatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-time-") as directory:
            root = Path(directory)
            args, artifact_document = self._args(root)
            base = runtime_result(args.target, artifact_document, args.artifact)
            mutations = []
            for value in (float("nan"), float("inf"), -1, 0):
                changed = copy.deepcopy(base)
                changed["cases"][0]["samples"][0]["kernel_latency_ns"] = value
                mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["cases"][0]["samples"][0]["wall_latency_ns"] = 1
            mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["cases"][0]["summary"]["kernel_median_ns"] += 1
            mutations.append(changed)
            changed = copy.deepcopy(base)
            changed["cases"][0]["samples"][0]["dispatch_id"] = changed["cases"][0]["warmup_dispatches"][0]["dispatch_id"]
            mutations.append(changed)
            for changed in mutations:
                with self.subTest(changed=changed), self.assertRaises(ContractError):
                    contracts.validate_runtime_result(changed, artifact_document, sha256_file(args.artifact))

    def test_artifact_rechecks_binary_sidecar_source_candidate_and_prerequisites(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-artifact-") as directory:
            root = Path(directory)
            args, artifact_document = self._args(root)
            contracts.validate_artifact(artifact_document, binary_path=args.binary)
            for mutation in (
                lambda value: value["binary"].__setitem__("sha256", "f" * 64),
                lambda value: value["source_set"].__setitem__("sha256", "f" * 64),
                lambda value: value["source_set"]["files"].pop(),
                lambda value: value["source_set"]["files"].reverse(),
                lambda value: value["source_set"]["files"][14].__setitem__(
                    "path", "crates/sllm-core/src/not-op.rs"
                ),
                lambda value: value["source_set"]["files"][14].__setitem__(
                    "sha256", "f" * 64
                ),
                lambda value: value["candidate"].__setitem__("tested_sha", "c" * 40),
                lambda value: value["prerequisites"][0].__setitem__("row_id", "g0-wrong"),
                lambda value: value["execution_contract"]["dtype"].__setitem__("output", "F32"),
            ):
                changed = copy.deepcopy(artifact_document)
                mutation(changed)
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    contracts.validate_artifact(changed, binary_path=args.binary)
            args.binary.write_bytes(b"changed")
            with self.assertRaises(ContractError):
                contracts.validate_artifact(artifact_document, binary_path=args.binary)

    def test_handwritten_pass_report_without_clean_execution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-p0-pass-") as directory:
            root = Path(directory)
            args, artifact_document = self._args(root)
            result = runtime_result(args.target, artifact_document, args.artifact)
            completed = runner.subprocess.CompletedProcess([], 0, canonical_bytes(result), b"")
            expected_device = next(
                item["device"] for item in contracts.expected_matrix()["targets"]
                if item["target"] == args.target
            )
            observed = (
                ok_health(args.target), clean_process(),
                {
                    "hip_id": expected_device["physical_hip_index"],
                    "bdf": expected_device["bdf"],
                    "uuid": expected_device["uuid"],
                    "product": expected_device["product"],
                },
            )
            with patch.dict(os.environ, {"SLLM_P0_GPU_EXECUTION": "1"}), patch.object(runner, "_observe_live", side_effect=[observed, observed]), patch.object(runner.subprocess, "run", return_value=completed):
                report = runner.run_row(args)
            report["state"] = "PASS"
            report["execution"]["stderr_sha256"] = "f" * 64
            with self.assertRaises(ContractError):
                contracts.validate_report(report)


if __name__ == "__main__":
    unittest.main()
