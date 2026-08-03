#!/usr/bin/env python3
"""Host-only negative, boundary, runner, and aggregate tests for G0."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ci/tools"))

from aggregate_g0_results import (  # noqa: E402
    aggregate,
    load_needs,
    validate_aggregate_schema,
    validate_row,
    write_summary,
)
from common import (  # noqa: E402
    ContractError,
    command_content_hash,
    sha256_file,
    sha256_json,
    validate_result_payload,
)
from run_g0_preflight import (  # noqa: E402
    amd_smi_list_json,
    artifact_binding,
    main as runner_main,
    make_report,
    nonblocking_host_lock,
    parse_sysfs_ras_counters,
    parser as runner_parser,
    run_native_provider,
    unavailable_preflight,
    validate_native_provider_json,
)
from validate_g0_contracts import (  # noqa: E402
    AMD_SMI_EXECUTABLE,
    AMD_SMI_LIST_COMMAND,
    EXPECTED_ROWS,
    native_provider_source_contract,
    reject_inherited_visibility_selectors,
    row_by_id,
    validate_g0_matrix,
    validate_native_provider_source_text,
    validate_g0_preflight,
    validate_visibility_environment,
)
from ci.tools import validate_cpp  # noqa: E402
from ci.tests.test_h3_contracts import artifact_fixture, copy_contract_tree  # noqa: E402

G0_FILES = (
    "ci/schema/g0-preflight-v1.schema.json",
    "ci/schema/g0-aggregate-v1.schema.json",
    "ci/matrix/gpu-runtime-v1.json",
    "ci/tools/g0_native_observer.cpp",
)


class CppFilesTests(unittest.TestCase):
    def test_collects_explicit_g0_source_without_opening_ci_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracked = (
                "ci/tools/g0_native_observer.cpp",
                "ci/tools/generated_helper.cpp",
                ".local-artifacts/generated.cpp",
            )
            for relative in (*tracked, "native/hip/src/host.cpp"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("int main() { return 0; }\n", encoding="utf-8")

            listed = subprocess.CompletedProcess(
                args=["git", "ls-files"],
                returncode=0,
                stdout="\n".join(tracked) + "\n",
                stderr="",
            )
            with (
                patch.object(validate_cpp, "ROOT", root),
                patch.object(validate_cpp.subprocess, "run", return_value=listed),
            ):
                collected = [
                    path.relative_to(root).as_posix()
                    for path in validate_cpp.cpp_files()
                ]

        self.assertEqual(
            collected,
            ["ci/tools/g0_native_observer.cpp", "native/hip/src/host.cpp"],
        )


def write_sidecar(path: Path) -> None:
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )


def fixture_repo() -> Path:
    repo = copy_contract_tree()
    for relative in G0_FILES:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return repo


def valid_health(
    row: dict[str, object], timestamp: str = "2026-08-03T08:00:00Z"
) -> dict[str, object]:
    return {
        "available": True,
        "reliable": True,
        "observed_at": timestamp,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1",
        "facts": {
            "device_state": "active",
            "amdgpu_driver_bound": True,
            "runtime_status": "active",
            "ras_uncorrectable_count": 17,
            "sysfs_ras_uncorrectable_count": 17,
            "temperature_c": 47.5,
        },
    }


def valid_process(
    row: dict[str, object], timestamp: str = "2026-08-03T08:00:00Z"
) -> dict[str, object]:
    return {
        "available": True,
        "reliable": True,
        "observed_at": timestamp,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gcnArchName": row["target"],
        "source": "amd-smi-sysfs-read-only-v1",
        "gpu_processes": [],
        "residual_runner_children": [],
    }


def valid_routing(row: dict[str, object], hip_id: int = 17) -> dict[str, object]:
    return {
        "source": "amd-smi-list-e-json-v1",
        "amd_smi": AMD_SMI_EXECUTABLE,
        "argv": list(AMD_SMI_LIST_COMMAND),
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gpu": 3,
        "hip_id": hip_id,
    }


def valid_native_provider(repo: Path) -> dict[str, object]:
    source = native_provider_source_contract(repo)
    return {
        "provider_id": "g0-native-hip-observer-v1",
        "available": True,
        "source_path": source["source_path"],
        "source_sha256": source["source_sha256"],
        "binary_path": None,
        "binary_removed": True,
        "binary_sha256": "1" * 64,
        "compiler_path": "/opt/rocm/core-7.14/bin/amdclang++",
        "compiler_version": "AMD clang version 23.0.0",
        "compile_command_sha256": "2" * 64,
        "runtime_command_sha256": "3" * 64,
    }


def valid_native_device(row: dict[str, object]) -> dict[str, object]:
    return {
        "probe_kind": "hip-identity-only-v1",
        "observed": True,
        "visible_device_count": 1,
        "ordinal": 0,
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "hip_uuid_hex": str(row["uuid"]).removeprefix("GPU-"),
        "gcnArchName": f"{row['target']}:xnack-",
        "exact_target": row["target"],
        "product": row["product"],
        "wave_size": 32,
        "total_global_memory_bytes": 16 * 1024 * 1024 * 1024,
        "rocm_root": row["rocm"]["root"],
        "allocation_count": 0,
        "copy_count": 0,
        "kernel_dispatch_count": 0,
        "dispatch_count": 0,
    }


def valid_runtime(row: dict[str, object]) -> dict[str, object]:
    return {
        "rocm_root": row["rocm"]["root"],
        "release": row["rocm"]["release"],
        "hip_runtime_api_version": row["rocm"]["hip_runtime_api_version"],
        "hip_runtime_library_path": row["rocm"]["hip_runtime_library"],
        "hsa_runtime_library_path": row["rocm"]["hsa_runtime_library"],
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "gcnArchName": row["target"],
    }


def native_provider_document(row: dict[str, object]) -> dict[str, object]:
    device = valid_native_device(row)
    runtime = valid_runtime(row)
    return {
        "provider_id": "g0-native-hip-observer-v1",
        "probe_kind": "hip-identity-only-v1",
        "rocm_root": runtime["rocm_root"],
        "release": runtime["release"],
        "hip_runtime_api_version": runtime["hip_runtime_api_version"],
        "hip_runtime_library_path": runtime["hip_runtime_library_path"],
        "hsa_runtime_library_path": runtime["hsa_runtime_library_path"],
        "visible_device_count": 1,
        "device": {
            key: device[key]
            for key in (
                "ordinal",
                "bdf",
                "uuid",
                "hip_uuid_hex",
                "gcnArchName",
                "exact_target",
                "product",
                "wave_size",
                "total_global_memory_bytes",
            )
        },
        "scope": {
            "allocation_count": 0,
            "copy_count": 0,
            "kernel_dispatch_count": 0,
            "dispatch_count": 0,
        },
    }


def preflight_fixture(
    repo: Path, target: str = "gfx1030", artifact_size: int = 257
) -> tuple[Path, dict[str, object]]:
    run_root, source_metadata_path, metadata = artifact_fixture(repo, target=target, size=artifact_size)
    artifact_path = Path(metadata["artifact"]["path"])
    metadata_path = artifact_path.parent / "hip-artifact-metadata.json"
    shutil.copy2(source_metadata_path, metadata_path)
    write_sidecar(metadata_path)
    write_sidecar(artifact_path)
    matrix = validate_g0_matrix(repo)
    row = row_by_id(matrix, f"g0-{target}")
    candidate = {
        "reviewed_sha": "a" * 40,
        "tested_sha": "a" * 40,
        "workflow_sha": "a" * 40,
        "git_tree_oid": "b" * 40,
        "worktree_clean": True,
        "revision_input": "full-sha",
    }
    metadata_sidecar = metadata_path.with_name(metadata_path.name + ".sha256")
    artifact_sidecar = artifact_path.with_name(artifact_path.name + ".sha256")
    preflight: dict[str, object] = {
        "schema_version": "g0-preflight-v1",
        "candidate": candidate,
        "visibility": {
            "HIP_VISIBLE_DEVICES": "17",
            "CUDA_VISIBLE_DEVICES": None,
            "GPU_DEVICE_ORDINAL": None,
            "security_boundary": False,
        },
        "routing": valid_routing(row),
        "artifact_binding": {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "metadata_sidecar_path": str(metadata_sidecar),
            "metadata_sidecar_sha256": sha256_file(metadata_sidecar),
            "metadata_declared_artifact_path": metadata["artifact"]["path"],
            "artifact_path": str(artifact_path),
            "artifact_sha256": sha256_file(artifact_path),
            "artifact_sidecar_path": str(artifact_sidecar),
            "artifact_sidecar_sha256": sha256_file(artifact_sidecar),
            "h3_matrix_row_id": row["h3_artifact_row_id"],
            "target": target,
            "toolchain_id": "rocm-7.14.0",
            "toolchain_manifest_sha256": metadata["toolchain_manifest_sha256"],
        },
        "provider": valid_native_provider(repo),
        "device": valid_native_device(row),
        "runtime": valid_runtime(row),
        "health_pre": valid_health(row),
        "health_post": valid_health(row, "2026-08-03T08:00:01Z"),
        "process_pre": valid_process(row),
        "process_post": valid_process(row, "2026-08-03T08:00:01Z"),
        "scope": {
            "selected_backend": "hip-preflight",
            "fallback_allowed": False,
            "fallback_used": False,
            "identity_probe_only": True,
            "native_hip_observation_provider": "native-hip-observer-v1",
            "execution_verified": False,
            "numerics_verified": False,
            "performance_verified": False,
            "support_claim": False,
        },
    }
    return run_root, preflight


class G0RasParserTests(unittest.TestCase):
    def test_zero_counters(self) -> None:
        self.assertEqual(
            parse_sysfs_ras_counters("ue: 0\nce: 0\nde: 0\n"),
            {"ue": 0, "ce": 0, "de": 0},
        )

    def test_nonzero_ue_is_returned_without_summing_ce_or_de(self) -> None:
        counters = parse_sysfs_ras_counters("ue: 7\nce: 11\nde: 13\n")
        self.assertEqual(counters["ue"], 7)
        self.assertEqual(counters["ce"], 11)
        self.assertEqual(counters["de"], 13)

    def test_rejects_noncanonical_counter_documents(self) -> None:
        cases = {
            "missing ue": "ce: 0\nde: 0\n",
            "missing de": "ue: 0\nce: 0\n",
            "duplicate ue": "ue: 0\nue: 1\nce: 0\nde: 0\n",
            "unknown key": "ue: 0\nce: 0\nfoo: 0\nde: 0\n",
            "malformed": "ue: zero\nce: 0\nde: 0\n",
            "negative": "ue: -1\nce: 0\nde: 0\n",
            "signed": "ue: +1\nce: 0\nde: 0\n",
            "leading zero": "ue: 01\nce: 0\nde: 0\n",
            "ambiguous whitespace": "ue:\t0\nce: 0\nde: 0\n",
            "trailing whitespace": "ue: 0\nce: 0\nde: 0 \n",
            "overflow": "ue: 18446744073709551616\nce: 0\nde: 0\n",
        }
        for label, text in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ContractError):
                    parse_sysfs_ras_counters(text)


class G0MatrixTests(unittest.TestCase):
    def test_exact_ordered_rows_and_serial_contract(self) -> None:
        matrix = validate_g0_matrix(ROOT)
        self.assertEqual(matrix["rows"], list(EXPECTED_ROWS))
        self.assertEqual(matrix["execution"]["host_lock"]["acquisition"], "nonblocking")
        self.assertFalse(matrix["execution"]["visibility_is_security_boundary"])

    def test_matrix_wrong_missing_duplicate_target_root_and_scope_fail(self) -> None:
        mutations = (
            lambda document: document["rows"].pop(),
            lambda document: document["rows"].append(copy.deepcopy(document["rows"][0])),
            lambda document: document["rows"][0].update(target="gfx1031"),
            lambda document: document["rows"][0]["rocm"].update(root="/opt/rocm"),
            lambda document: document["rows"][0].update(h3_artifact_row_id="h3-gfx1201"),
            lambda document: document["execution"].update(serial=False),
            lambda document: document["execution"]["host_lock"].update(acquisition="blocking"),
            lambda document: document["scope"].update(kernel_attempted=True),
            lambda document: document["scope"].update(native_hip_observation_provider="available"),
            lambda document: document["output"].update(source_tree_output=True),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                repo = fixture_repo()
                try:
                    path = repo / "ci/matrix/gpu-runtime-v1.json"
                    document = json.loads(path.read_text(encoding="utf-8"))
                    mutation(document)
                    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
                    with self.assertRaises(ContractError):
                        validate_g0_matrix(repo)
                finally:
                    shutil.rmtree(repo)


class G0VisibilityTests(unittest.TestCase):
    def test_unset_or_one_normalized_selector(self) -> None:
        unset = {name: None for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
        self.assertFalse(validate_visibility_environment(unset)["security_boundary"])
        for name, token in (
            ("HIP_VISIBLE_DEVICES", "GPU-76a08c022586fed6"),
            ("CUDA_VISIBLE_DEVICES", "0000:03:00.0"),
            ("GPU_DEVICE_ORDINAL", "17"),
        ):
            values = dict(unset)
            values[name] = token
            self.assertEqual(validate_visibility_environment(values)[name], token)

    def test_conflict_malformed_multi_and_aliases_fail(self) -> None:
        cases = (
            {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "0"},
            {"HIP_VISIBLE_DEVICES": "0,1"},
            {"HIP_VISIBLE_DEVICES": " 0"},
            {"GPU_DEVICE_ORDINAL": "00"},
            {"CUDA_VISIBLE_DEVICES": "GPU-76A08C022586FED6"},
            {"HIP_VISIBLE_DEVICES": "gfx1030"},
        )
        for values in cases:
            environment = {name: values.get(name) for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
            with self.subTest(values=values), self.assertRaises(ContractError):
                validate_visibility_environment(environment)

    def test_all_known_inherited_selectors_fail_before_routing(self) -> None:
        for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL", "ROCR_VISIBLE_DEVICES"):
            with self.subTest(name=name), self.assertRaises(ContractError):
                reject_inherited_visibility_selectors({name: "0"})


class G0PreflightNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = fixture_repo()
        self.run_root, self.preflight = preflight_fixture(self.repo)

    def tearDown(self) -> None:
        shutil.rmtree(self.repo)
        shutil.rmtree(self.run_root)

    def reject(self, mutation) -> None:
        document = copy.deepcopy(self.preflight)
        mutation(document)
        with self.assertRaises(ContractError):
            validate_g0_preflight(document, "g0-gfx1030", self.repo)

    def test_valid_exact_preflight_and_boundary_artifact_sizes(self) -> None:
        validate_g0_preflight(self.preflight, "g0-gfx1030", self.repo)
        for size in (255, 256, 257):
            run_root, document = preflight_fixture(self.repo, artifact_size=size)
            try:
                validate_g0_preflight(document, "g0-gfx1030", self.repo)
            finally:
                shutil.rmtree(run_root)

    def test_staged_h3_artifact_rebinding_is_explicit_and_hash_bound(self) -> None:
        metadata_path = Path(self.preflight["artifact_binding"]["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        declared_root = Path(tempfile.mkdtemp(prefix="ullm-h3-declared-")) / "h3-gfx1030"
        declared_artifact = declared_root / "device-code-object-gfx1030.elf"
        metadata["build"].update(
            output_directory=str(declared_root),
            object_path=str(declared_root / "hip-compile-probe-gfx1030.o"),
            link_output_path=str(declared_root / "hip-compile-probe-gfx1030.elf"),
        )
        metadata["artifact"]["path"] = str(declared_artifact)
        metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
        write_sidecar(metadata_path)
        row = row_by_id(validate_g0_matrix(self.repo), "g0-gfx1030")
        self.preflight["artifact_binding"] = artifact_binding(metadata_path, row, self.repo)
        try:
            validate_g0_preflight(self.preflight, "g0-gfx1030", self.repo)
            self.assertNotEqual(
                self.preflight["artifact_binding"]["metadata_declared_artifact_path"],
                self.preflight["artifact_binding"]["artifact_path"],
            )
        finally:
            shutil.rmtree(declared_root.parent)

    def test_wrong_bdf_uuid_target_product_and_root_fail(self) -> None:
        cases = (
            lambda value: value["device"].update(bdf="0000:43:00.0"),
            lambda value: value["device"].update(uuid="GPU-0000000000000000"),
            lambda value: value["device"].update(gcnArchName="gfx1201"),
            lambda value: value["device"].update(exact_target="gfx1201"),
            lambda value: value["device"].update(product="AMD Radeon AI PRO R9700"),
            lambda value: value["device"].update(rocm_root="/opt/rocm"),
            lambda value: value["runtime"].update(rocm_root="/opt/rocm-7.14"),
            lambda value: value["runtime"].update(release="7.13.0"),
            lambda value: value["runtime"].update(bdf="0000:43:00.0"),
            lambda value: value["runtime"].update(uuid="GPU-0000000000000000"),
            lambda value: value["runtime"].update(hip_runtime_library_path="/usr/lib/libamdhip64.so"),
            lambda value: value["device"].update(visible_device_count=2),
            lambda value: value["device"].update(ordinal=1),
            lambda value: value["device"].update(hip_uuid_hex="0" * 16),
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                self.reject(mutation)

    def test_dirty_ref_short_mismatched_sha_and_tree_fail(self) -> None:
        cases = (
            lambda value: value["candidate"].update(worktree_clean=False),
            lambda value: value["candidate"].update(revision_input="branch"),
            lambda value: value["candidate"].update(reviewed_sha="a" * 7),
            lambda value: value["candidate"].update(tested_sha="c" * 40),
            lambda value: value["candidate"].update(git_tree_oid="tree"),
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                self.reject(mutation)

    def test_artifact_hash_target_row_path_and_sidecars_fail(self) -> None:
        cases = (
            lambda value: value["artifact_binding"].update(artifact_sha256="0" * 64),
            lambda value: value["artifact_binding"].update(metadata_sha256="1" * 64),
            lambda value: value["artifact_binding"].update(metadata_declared_artifact_path="/tmp/other.elf"),
            lambda value: value["artifact_binding"].update(target="gfx1201"),
            lambda value: value["artifact_binding"].update(h3_matrix_row_id="h3-gfx1201"),
            lambda value: value["artifact_binding"].update(toolchain_id="rocm-7.13.0"),
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                self.reject(mutation)
        sidecar = Path(self.preflight["artifact_binding"]["artifact_sidecar_path"])
        sidecar.write_text("0" * 64 + "  wrong.elf\n", encoding="ascii")
        with self.assertRaises(ContractError):
            validate_g0_preflight(self.preflight, "g0-gfx1030", self.repo)

    def test_probe_never_allocates_copies_kernels_or_dispatches(self) -> None:
        for field in ("allocation_count", "copy_count", "kernel_dispatch_count", "dispatch_count"):
            with self.subTest(field=field):
                self.reject(lambda value, field=field: value["device"].update({field: 1}))
        self.reject(lambda value: value["scope"].update(execution_verified=True))
        self.reject(lambda value: value["scope"].update(selected_backend="hip"))
        self.reject(lambda value: value["scope"].update(fallback_used=True))

    def test_routing_uses_only_the_canonical_bdf_hip_id_hint(self) -> None:
        cases = (
            lambda value: value["visibility"].update(HIP_VISIBLE_DEVICES="18"),
            lambda value: value["visibility"].update(CUDA_VISIBLE_DEVICES="0"),
            lambda value: value["routing"].update(bdf="0000:43:00.0"),
            lambda value: value["routing"].update(uuid="GPU-0000000000000000"),
            lambda value: value["routing"].update(hip_id=18),
            lambda value: value["routing"].update(argv=[AMD_SMI_EXECUTABLE, "list", "--json"]),
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                self.reject(mutation)

    def test_health_process_unavailable_unreliable_changed_and_residual_fail(self) -> None:
        cases = (
            lambda value: value["health_pre"].update(available=False),
            lambda value: value["health_post"].update(reliable=False),
            lambda value: value["health_post"]["facts"].update(ras_uncorrectable_count=18),
            lambda value: value["health_pre"]["facts"].update(device_state="unknown"),
            lambda value: value["health_pre"].update(bdf="0000:43:00.0"),
            lambda value: value["health_post"].update(uuid="GPU-0000000000000000"),
            lambda value: value["process_pre"].update(available=False),
            lambda value: value["process_post"].update(reliable=False),
            lambda value: value["process_pre"].update(gcnArchName="gfx1201"),
            lambda value: value["process_pre"]["gpu_processes"].append({"pid": 17}),
            lambda value: value["process_post"]["residual_runner_children"].append(257),
            lambda value: value["health_pre"].update(observed_at="2026-08-03T08:00:02Z"),
        )
        for mutation in cases:
            with self.subTest(mutation=mutation):
                self.reject(mutation)


class G0RunnerAggregateTests(unittest.TestCase):
    def test_nonblocking_lock_rejects_competing_row(self) -> None:
        with nonblocking_host_lock(Path("/tmp/ullm-g0.lock")):
            with self.assertRaises(ContractError):
                with nonblocking_host_lock(Path("/tmp/ullm-g0.lock")):
                    pass

    def test_runner_rejects_rocr_before_canonical_routing(self) -> None:
        output_root = Path(tempfile.mkdtemp(prefix="ullm-g0-rocr-"))
        output = output_root / "g0-gfx1030"
        try:
            with patch.dict("run_g0_preflight.os.environ", {"ROCR_VISIBLE_DEVICES": "0"}, clear=True), patch(
                "run_g0_preflight.amd_smi_list_json"
            ) as routing_mock:
                exit_code = runner_main(
                    [
                        "--row", "g0-gfx1030", "--repo", str(ROOT), "--output-dir", str(output),
                        "--trusted-local", "--artifact-metadata", "/tmp/metadata.json",
                        "--run-id", "unit-g0", "--run-attempt", "1",
                        "--reviewed-sha", "a" * 40, "--tested-sha", "a" * 40,
                        "--workflow-sha", "a" * 40,
                    ]
                )
            self.assertEqual(exit_code, 2)
            routing_mock.assert_not_called()
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            validate_result_payload(report)
            self.assertEqual(report["state"], "INFRA_ERROR")
        finally:
            shutil.rmtree(output_root)

    def test_unavailable_identity_report_is_schema_valid_infra_error(self) -> None:
        matrix = validate_g0_matrix(ROOT)
        row = row_by_id(matrix, "g0-gfx1030")
        candidate = {
            "reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40, "worktree_clean": True, "revision_input": "full-sha",
        }
        visibility = validate_visibility_environment({name: None for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")})
        preflight = unavailable_preflight(candidate, visibility, row)
        instant = datetime.now(timezone.utc)
        report = make_report(
            row=row, matrix=matrix, candidate=candidate, preflight=preflight,
            state="INFRA_ERROR", error="HIP identity-only observation provider is unavailable",
            run_id="unit-g0", run_attempt=1, started=instant, finished=instant,
        )
        validate_result_payload(report)
        self.assertEqual(report["state"], "INFRA_ERROR")
        self.assertEqual(report["gpu"]["kernel_dispatch_count"], 0)
        stale_schema = copy.deepcopy(report)
        stale_schema["g0"]["preflight_schema_sha256"] = "0" * 64
        with self.assertRaises(ContractError):
            validate_result_payload(stale_schema)
        invalid_preflight = copy.deepcopy(report)
        invalid_preflight["g0"]["preflight"].pop("scope")
        invalid_preflight["g0"]["preflight_sha256"] = sha256_json(invalid_preflight["g0"]["preflight"])
        with self.assertRaises(ContractError):
            validate_result_payload(invalid_preflight)

    def test_g0_payload_is_rejected_outside_tier_g0(self) -> None:
        matrix = validate_g0_matrix(ROOT)
        row = row_by_id(matrix, "g0-gfx1030")
        candidate = {
            "reviewed_sha": "a" * 40, "tested_sha": "a" * 40, "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40, "worktree_clean": True, "revision_input": "full-sha",
        }
        visibility = validate_visibility_environment(
            {name: None for name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")}
        )
        instant = datetime.now(timezone.utc)
        g0_report = make_report(
            row=row,
            matrix=matrix,
            candidate=candidate,
            preflight=unavailable_preflight(candidate, visibility, row),
            state="INFRA_ERROR",
            error="native provider pending",
            run_id="unit-g0",
            run_attempt=1,
            started=instant,
            finished=instant,
        )
        host_report = copy.deepcopy(g0_report)
        host_report.pop("g0")
        host_report.pop("gpu")
        host_report["tier"] = "tier_h0"
        host_report["artifact"]["content_sha256"] = command_content_hash(host_report["steps"])
        host_report["diagnostic"]["network_disabled"] = True
        validate_result_payload(host_report)
        host_report["g0"] = g0_report["g0"]
        with self.assertRaises(ContractError):
            validate_result_payload(host_report)

    def test_runner_uses_canonical_routing_and_mocked_native_observer(self) -> None:
        output_root = Path(tempfile.mkdtemp(prefix="ullm-g0-runner-"))
        output = output_root / "g0-gfx1030"
        matrix = validate_g0_matrix(ROOT)
        row = row_by_id(matrix, "g0-gfx1030")
        candidate = {
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40,
            "worktree_clean": True,
            "revision_input": "full-sha",
        }
        binding = {
            "metadata_path": "/tmp/h3-gfx1030/hip-artifact-metadata.json",
            "metadata_sha256": "1" * 64,
            "metadata_sidecar_path": "/tmp/h3-gfx1030/hip-artifact-metadata.json.sha256",
            "metadata_sidecar_sha256": "2" * 64,
            "metadata_declared_artifact_path": "/tmp/ullm-h3-build/h3-gfx1030/device-code-object-gfx1030.elf",
            "artifact_path": "/tmp/h3-gfx1030/device-code-object-gfx1030.elf",
            "artifact_sha256": "3" * 64,
            "artifact_sidecar_path": "/tmp/h3-gfx1030/device-code-object-gfx1030.elf.sha256",
            "artifact_sidecar_sha256": "4" * 64,
            "h3_matrix_row_id": "h3-gfx1030",
            "target": "gfx1030",
            "toolchain_id": "rocm-7.14.0",
            "toolchain_manifest_sha256": "5" * 64,
        }
        routing = valid_routing(row)
        try:
            with patch.dict("run_g0_preflight.os.environ", {}, clear=True), patch(
                "run_g0_preflight.now",
                side_effect=[
                    datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc),
                    datetime(2026, 8, 3, 8, 0, 1, tzinfo=timezone.utc),
                ],
            ), patch(
                "run_g0_preflight.git_candidate", return_value=candidate
            ), patch("run_g0_preflight.artifact_binding", return_value=binding), patch(
                "run_g0_preflight.amd_smi_list_json", return_value=routing
            ) as routing_mock, patch(
                "run_g0_preflight.observe_health",
                side_effect=[
                    valid_health(row),
                    valid_health(row, "2026-08-03T08:00:01Z"),
                ],
            ), patch(
                "run_g0_preflight.observe_processes",
                side_effect=[
                    valid_process(row),
                    valid_process(row, "2026-08-03T08:00:01Z"),
                ],
            ), patch(
                "run_g0_preflight.run_native_provider",
                return_value=(
                    valid_native_provider(ROOT),
                    valid_native_device(row),
                    valid_runtime(row),
                ),
            ) as native_mock, patch("run_g0_preflight.validate_g0_preflight"):
                exit_code = runner_main(
                    [
                        "--row",
                        "g0-gfx1030",
                        "--repo",
                        str(ROOT),
                        "--output-dir",
                        str(output),
                        "--trusted-local",
                        "--artifact-metadata",
                        "/tmp/metadata.json",
                        "--run-id",
                        "unit-g0",
                        "--run-attempt",
                        "1",
                        "--reviewed-sha",
                        "a" * 40,
                        "--tested-sha",
                        "a" * 40,
                        "--workflow-sha",
                        "a" * 40,
                    ]
                )
            self.assertEqual(exit_code, 0)
            routing_mock.assert_called_once_with(row, executable=AMD_SMI_EXECUTABLE)
            native_mock.assert_called_once_with(
                ROOT.resolve(),
                row,
                hip_visible_devices="17",
            )
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["state"], "PASS")
            self.assertEqual(report["g0"]["preflight"]["routing"], routing)
            self.assertEqual(
                report["g0"]["preflight"]["visibility"]["HIP_VISIBLE_DEVICES"],
                "17",
            )
        finally:
            shutil.rmtree(output_root)

    def test_runner_fails_closed_when_pre_health_or_process_is_unavailable(self) -> None:
        matrix = validate_g0_matrix(ROOT)
        row = row_by_id(matrix, "g0-gfx1030")
        candidate = {
            "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40,
            "workflow_sha": "a" * 40,
            "git_tree_oid": "b" * 40,
            "worktree_clean": True,
            "revision_input": "full-sha",
        }
        binding = {
            "metadata_path": "/tmp/h3-gfx1030/hip-artifact-metadata.json",
            "metadata_sha256": "1" * 64,
            "metadata_sidecar_path": "/tmp/h3-gfx1030/hip-artifact-metadata.json.sha256",
            "metadata_sidecar_sha256": "2" * 64,
            "metadata_declared_artifact_path": "/tmp/ullm-h3-build/h3-gfx1030/device-code-object-gfx1030.elf",
            "artifact_path": "/tmp/h3-gfx1030/device-code-object-gfx1030.elf",
            "artifact_sha256": "3" * 64,
            "artifact_sidecar_path": "/tmp/h3-gfx1030/device-code-object-gfx1030.elf.sha256",
            "artifact_sidecar_sha256": "4" * 64,
            "h3_matrix_row_id": "h3-gfx1030",
            "target": "gfx1030",
            "toolchain_id": "rocm-7.14.0",
            "toolchain_manifest_sha256": "5" * 64,
        }
        unavailable = valid_health(row)
        unavailable.update(available=False, reliable=False, source=None)
        for observer_name, observer_result in (
            ("observe_health", unavailable),
            ("observe_processes", {**valid_process(row), "available": False, "reliable": False, "source": None}),
        ):
            with self.subTest(observer=observer_name):
                output_root = Path(tempfile.mkdtemp(prefix="ullm-g0-runner-"))
                output = output_root / "g0-gfx1030"
                try:
                    health_result = observer_result if observer_name == "observe_health" else valid_health(row)
                    process_result = observer_result if observer_name == "observe_processes" else valid_process(row)
                    with patch.dict("run_g0_preflight.os.environ", {}, clear=True), patch(
                        "run_g0_preflight.git_candidate", return_value=candidate
                    ), patch("run_g0_preflight.artifact_binding", return_value=binding), patch(
                        "run_g0_preflight.amd_smi_list_json", return_value=valid_routing(row)
                    ), patch(
                        "run_g0_preflight.observe_health", return_value=health_result
                    ) as health_mock, patch(
                        "run_g0_preflight.observe_processes", return_value=process_result
                    ) as process_mock, patch(
                        "run_g0_preflight.run_native_provider"
                    ) as native_mock:
                        exit_code = runner_main(
                            [
                                "--row",
                                "g0-gfx1030",
                                "--repo",
                                str(ROOT),
                                "--output-dir",
                                str(output),
                                "--trusted-local",
                                "--artifact-metadata",
                                "/tmp/metadata.json",
                                "--run-id",
                                "unit-g0",
                                "--run-attempt",
                                "1",
                                "--reviewed-sha",
                                "a" * 40,
                                "--tested-sha",
                                "a" * 40,
                                "--workflow-sha",
                                "a" * 40,
                            ]
                        )
                    self.assertEqual(exit_code, 2)
                    native_mock.assert_not_called()
                    health_mock.assert_called_once()
                    if observer_name == "observe_health":
                        process_mock.assert_not_called()
                    else:
                        process_mock.assert_called_once()
                    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
                    validate_result_payload(report)
                    self.assertEqual(report["state"], "INFRA_ERROR")
                    expected_label = "pre-health" if observer_name == "observe_health" else "pre-process"
                    self.assertEqual(
                        report["diagnostic"]["errors"][0],
                        f"{expected_label} observation is unavailable or unreliable",
                    )
                finally:
                    shutil.rmtree(output_root)

    def test_runner_rejects_external_observation_json_input(self) -> None:
        option_names = {
            option
            for action in runner_parser()._actions
            for option in action.option_strings
        }
        self.assertNotIn("--observation-json", option_names)
        with self.assertRaises(SystemExit) as rejected:
            runner_parser().parse_args(
                [
                    "--row",
                    "g0-gfx1030",
                    "--output-dir",
                    "/tmp/ullm-g0-unit/g0-gfx1030",
                    "--reviewed-sha",
                    "a" * 40,
                    "--tested-sha",
                    "a" * 40,
                    "--workflow-sha",
                    "a" * 40,
                    "--observation-json",
                    "/tmp/untrusted-observation.json",
                ]
            )
        self.assertEqual(rejected.exception.code, 2)

    def test_needs_zero_missing_duplicate_unknown_and_non_success_fail(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="ullm-g0-needs-"))
        path = directory / "needs.json"
        cases = (
            {},
            {"g0-gfx1030": {"result": "success"}},
            {"g0-gfx1030": {"result": "success"}, "g0-gfx1201": {"result": "failure"}},
            {"g0-gfx1030": {"result": "success"}, "g0-gfx1201": {"result": "success"}, "g0-unknown": {"result": "success"}},
        )
        try:
            for document in cases:
                path.write_text(json.dumps(document) + "\n", encoding="utf-8")
                with self.subTest(document=document), self.assertRaises(ContractError):
                    load_needs(path)
            path.write_text(
                '{"g0-gfx1030":{"result":"success"},"g0-gfx1030":{"result":"success"},"g0-gfx1201":{"result":"success"}}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ContractError):
                load_needs(path)
        finally:
            shutil.rmtree(directory)

    def test_aggregate_output_stays_in_private_tmp_and_run_id_is_strict(self) -> None:
        outside = Path(tempfile.mkdtemp(prefix="not-g0-output-"))
        inside = Path(tempfile.mkdtemp(prefix="ullm-g0-summary-"))
        try:
            with self.assertRaises(ContractError):
                write_summary(outside, {})
            write_summary(inside, {})
            self.assertEqual(
                {path.name for path in inside.iterdir()},
                {"aggregate.json", "aggregate.json.sha256"},
            )
        finally:
            shutil.rmtree(outside)
            shutil.rmtree(inside)

    def test_row_aggregation_rejects_skip_zero_duplicate_stale_and_dispatch(self) -> None:
        repo = fixture_repo()
        run_root, preflight = preflight_fixture(repo)
        collection = Path(tempfile.mkdtemp(prefix="ullm-g0-aggregate-"))
        try:
            matrix = validate_g0_matrix(repo)
            row = row_by_id(matrix, "g0-gfx1030")
            instant = datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc)
            finished = datetime(2026, 8, 3, 8, 0, 1, tzinfo=timezone.utc)
            report = make_report(
                row=row, matrix=matrix, candidate=preflight["candidate"], preflight=preflight,
                state="PASS", error=None, run_id="unit-g0", run_attempt=1,
                started=instant, finished=finished,
            )
            row_dir = collection / row["row_id"]
            row_dir.mkdir()
            report_path = row_dir / "report.json"

            def write(document: dict[str, object]) -> None:
                report_path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                write_sidecar(report_path)

            identity = {
                "run_id": "unit-g0", "run_attempt": 1, "reviewed_sha": "a" * 40,
                "tested_sha": "a" * 40, "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40,
            }
            write(report)
            validate_row(row_dir, row, matrix, repo, identity)
            for field, timestamp in (
                ("health_pre", "2026-08-03T07:59:59Z"),
                ("health_post", "2026-08-03T08:00:02Z"),
                ("process_pre", "2026-08-03T07:59:59Z"),
                ("process_post", "2026-08-03T08:00:02Z"),
            ):
                outside_window = copy.deepcopy(report)
                outside_window["g0"]["preflight"][field]["observed_at"] = timestamp
                outside_window["g0"]["preflight_sha256"] = sha256_json(outside_window["g0"]["preflight"])
                write(outside_window)
                with self.subTest(field=field), self.assertRaises(ContractError):
                    validate_row(row_dir, row, matrix, repo, identity)
            write(report)
            mutations = (
                lambda value: value.update(state="SKIP"),
                lambda value: value["counts"].update(selected=0, passed=0, skipped=0, collected=0),
                lambda value: value.update(reviewed_sha="c" * 40),
                lambda value: value["gpu"].update(dispatch_count=1, kernel_dispatch_count=1),
            )
            for mutation in mutations:
                document = copy.deepcopy(report)
                mutation(document)
                write(document)
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    validate_row(row_dir, row, matrix, repo, identity)
            write(report)
            (row_dir / "duplicate.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ContractError):
                validate_row(row_dir, row, matrix, repo, identity)
        finally:
            shutil.rmtree(repo)
            shutil.rmtree(run_root)
            shutil.rmtree(collection)

    def test_aggregate_requires_exact_two_current_rows_and_schema_hashes(self) -> None:
        repo = fixture_repo()
        collection = Path(tempfile.mkdtemp(prefix="ullm-g0-aggregate-"))
        needs_root = Path(tempfile.mkdtemp(prefix="ullm-g0-needs-"))
        run_roots: list[Path] = []
        needs = needs_root / "needs.json"
        needs.write_text(
            json.dumps({row_id: {"result": "success"} for row_id in ("g0-gfx1030", "g0-gfx1201")}) + "\n",
            encoding="utf-8",
        )
        identity = {
            "run_id": "unit-g0", "run_attempt": 1, "reviewed_sha": "a" * 40,
            "tested_sha": "a" * 40, "workflow_sha": "a" * 40, "git_tree_oid": "b" * 40,
        }
        try:
            matrix = validate_g0_matrix(repo)
            for target in ("gfx1030", "gfx1201"):
                run_root, preflight = preflight_fixture(repo, target=target)
                run_roots.append(run_root)
                row = row_by_id(matrix, f"g0-{target}")
                instant = datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc)
                finished = datetime(2026, 8, 3, 8, 0, 1, tzinfo=timezone.utc)
                report = make_report(
                    row=row, matrix=matrix, candidate=preflight["candidate"], preflight=preflight,
                    state="PASS", error=None, run_id="unit-g0", run_attempt=1,
                    started=instant, finished=finished,
                )
                row_dir = collection / row["row_id"]
                row_dir.mkdir()
                report_path = row_dir / "report.json"
                report_path.write_bytes(json.dumps(report, sort_keys=True, separators=(",", ":")).encode() + b"\n")
                write_sidecar(report_path)

            aggregate_kwargs = {
                "needs": needs, "artifact_dir": collection, "repo": repo,
                "run_id": "unit-g0", "run_attempt": 1,
                "reviewed_sha": "a" * 40, "tested_sha": "a" * 40,
                "workflow_sha": "a" * 40, "tree_oid": "b" * 40,
            }
            with patch("aggregate_g0_results.identity", return_value={"commit": "a" * 40, "tree": "b" * 40}), patch(
                "aggregate_g0_results.ensure_clean_worktree"
            ):
                summary = aggregate(**aggregate_kwargs)
                validate_aggregate_schema(summary, repo)
                for mutation in (
                    lambda value: value["rows"].__setitem__(1, copy.deepcopy(value["rows"][0])),
                    lambda value: value["rows"].__setitem__(0, {**value["rows"][0], "target": "gfx1201"}),
                    lambda value: value["rows"].__setitem__(1, {**value["rows"][1], "target": "gfx1030"}),
                ):
                    mutated_rows = copy.deepcopy(summary)
                    mutation(mutated_rows)
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_aggregate_schema(mutated_rows, repo)
                invalid_run = dict(aggregate_kwargs, run_id="not a run id")
                with self.assertRaises(ContractError):
                    aggregate(**invalid_run)
                for mutation in (
                    lambda value: value.update(preflight_schema_sha256="0" * 64),
                    lambda value: value.update(matrix_manifest_sha256="1" * 64),
                ):
                    mutated = copy.deepcopy(summary)
                    mutation(mutated)
                    with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                        validate_aggregate_schema(mutated, repo)

                collection.joinpath("unknown").mkdir()
                with self.assertRaises(ContractError):
                    aggregate(**aggregate_kwargs)
                collection.joinpath("unknown").rmdir()

                report_path = collection / "g0-gfx1030" / "report.json"
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["run_id"] = "stale-run"
                report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
                write_sidecar(report_path)
                with self.assertRaises(ContractError):
                    aggregate(**aggregate_kwargs)

                shutil.rmtree(collection / "g0-gfx1201")
                with self.assertRaises(ContractError):
                    aggregate(**aggregate_kwargs)
        finally:
            shutil.rmtree(repo)
            shutil.rmtree(collection)
            shutil.rmtree(needs_root)
            for run_root in run_roots:
                shutil.rmtree(run_root)


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if os.environ.get("ULLM_EMIT_TEST_COUNTS") == "1":
        selected = result.testsRun
        failed = len(result.failures) + len(result.errors)
        skipped = len(result.skipped)
        print(
            "ULLM_UNITTEST_COUNTS="
            + json.dumps(
                {"collected": selected, "selected": selected, "passed": selected - failed - skipped,
                 "failed": failed, "skipped": skipped, "deselected": 0},
                sort_keys=True, separators=(",", ":"),
            ),
            flush=True,
        )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
