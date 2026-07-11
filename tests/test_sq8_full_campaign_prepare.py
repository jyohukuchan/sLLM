from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import os
import stat
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import sq8_full_campaign_prepare as PREPARE  # noqa: E402
import sq8_full_campaign_identity as IDENTITY  # noqa: E402
import sq8_full_campaign_production as PRODUCTION  # noqa: E402


WORKER_SHA = "a" * 64
COMMIT = "b" * 40
ALL_SOURCE_SHA = "c" * 64
SECRET = b"fourteen-byte-secret"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module(
    "test_sq8_full_campaign_prepare_validator",
    TOOLS / "validate-sq8-openwebui-release.py",
)


@dataclasses.dataclass(frozen=True)
class FakeIndependentIdentity:
    expected_commit: str
    expected_worker_binary_sha256: str


@dataclasses.dataclass(frozen=True)
class FakeSourceCheckout:
    git_commit: str
    source_count: int
    all_source_sha256: str


class FakeAnchor:
    def __init__(self) -> None:
        self.settings = PRODUCTION.production_preflight_settings()
        self.commit = COMMIT
        self.status_raw = b""
        self.revalidations = 0
        self.mutate_commit_on: int | None = None
        self.mutate_status_on: int | None = None
        self.raise_on: int | None = None

    def revalidate(self) -> None:
        self.revalidations += 1
        if self.mutate_commit_on == self.revalidations:
            self.commit = "d" * 40
        if self.mutate_status_on == self.revalidations:
            self.status_raw = b" M tools/drift.py\n"
        if self.raise_on == self.revalidations:
            raise RuntimeError("synthetic Git TOCTOU")


class FakeIndependentValidator:
    def __init__(self) -> None:
        self.identity_error = False
        self.source_error = False
        self.source_count = PREPARE.EXPECTED_SOURCE_COUNT
        self.identity_calls = 0
        self.source_calls = 0

    def validate_campaign_identity(
        self,
        bundle: Path,
        *,
        expected_commit: str,
        expected_worker_binary_sha256: str,
    ) -> FakeIndependentIdentity:
        self.identity_calls += 1
        if self.identity_error:
            raise ValueError("synthetic identity rejection")
        self._require_private_artifact(bundle / "environment.json")
        self._require_private_artifact(bundle / "model-identity.json")
        return FakeIndependentIdentity(expected_commit, expected_worker_binary_sha256)

    def validate_campaign_source_checkout(
        self, identity_data: Any, *, repo_root: Path
    ) -> FakeSourceCheckout:
        self.source_calls += 1
        if self.source_error:
            raise ValueError("synthetic source rejection")
        if repo_root != PRODUCTION.PRODUCTION_REPO_ROOT:
            raise ValueError("repository drift")
        return FakeSourceCheckout(COMMIT, self.source_count, ALL_SOURCE_SHA)

    @staticmethod
    def _require_private_artifact(path: Path) -> None:
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
        ):
            raise ValueError("temporary artifact mode drift")
        if stat.S_IMODE(path.parent.stat().st_mode) != 0o700:
            raise ValueError("temporary directory mode drift")


def process(
    pid: int,
    *,
    ppid: int,
    executable: str,
    executable_sha256: str,
    children: tuple[int, ...],
) -> Any:
    return IDENTITY.ProcessSnapshot(
        pid=pid,
        ppid=ppid,
        uid=1000,
        gid=1000,
        starttime_ticks=pid * 10,
        executable=executable,
        executable_bytes=1234,
        executable_sha256=executable_sha256,
        children=children,
    )


def live_identity() -> Any:
    gateway = process(
        101,
        ppid=1,
        executable="/usr/bin/python3.12",
        executable_sha256="e" * 64,
        children=(202,),
    )
    worker = process(
        202,
        ppid=101,
        executable=os.fspath(PREPARE.PRODUCTION_WORKER_BINARY),
        executable_sha256=WORKER_SHA,
        children=(),
    )
    return IDENTITY.LiveIdentity(
        os_id="ubuntu",
        os_version_id="24.04",
        os_pretty_name="Ubuntu 24.04",
        kernel_sysname="Linux",
        kernel_release="6.17.0-35-generic",
        kernel_version="#1 synthetic",
        kernel_machine="x86_64",
        boot_id="1" * 32,
        cgroup_fs_type="cgroup2fs",
        systemd_major=PREPARE.SYSTEMD_MAJOR,
        systemd_version_line="systemd 255 synthetic",
        python_version_line="Python 3.12.3",
        rustc_version_line="rustc 1.96.0 synthetic",
        cargo_version_line="cargo 1.96.0 synthetic",
        docker_version="28.5.1",
        docker_api_version="1.51",
        docker_os="linux",
        docker_arch="amd64",
        docker_kernel_version="6.17.0-35-generic",
        amd_smi_tool=PREPARE.AMD_SMI_TOOL,
        amd_smi_library=PREPARE.AMD_SMI_LIBRARY,
        rocm_version=PREPARE.ROCM_VERSION,
        amd_smi_version_line="AMD SMI synthetic",
        gpu_index=PREPARE.GPU_INDEX,
        gpu_bdf=PREPARE.GPU_BDF,
        gpu_uuid=PREPARE.GPU_UUID,
        kfd_gpu_id=PREPARE.KFD_GPU_ID,
        gpu_node_id=PREPARE.GPU_NODE_ID,
        gpu_partition_id=PREPARE.GPU_PARTITION_ID,
        service_unit=IDENTITY.SERVICE_UNIT,
        service_user=PREPARE.SERVICE_USER,
        service_group=PREPARE.SERVICE_GROUP,
        service_uid=1000,
        service_gid=1000,
        service_fragment_path=os.fspath(PREPARE.PRODUCTION_EFFECTIVE_SERVICE_UNIT),
        control_group="/system.slice/ullm-openai.service",
        gateway=gateway,
        worker=worker,
        n_restarts=2,
        active_state="active",
        sub_state="running",
        openwebui_version=PREPARE.OPENWEBUI_VERSION,
        openwebui_source_revision=PREPARE.OPENWEBUI_SOURCE_REVISION,
        base_image_digest=PREPARE.OPENWEBUI_BASE_IMAGE_DIGEST,
        base_image_id=PREPARE.OPENWEBUI_BASE_IMAGE_ID,
        derived_image_id=PREPARE.OPENWEBUI_DERIVED_IMAGE_ID,
        patch_sha256=PREPARE.OPENWEBUI_PATCH_SHA256,
        patched_middleware_sha256=PREPARE.OPENWEBUI_PATCHED_MIDDLEWARE_SHA256,
        docker_network_name=IDENTITY.DOCKER_NETWORK_NAME,
        docker_network_id=PREPARE.DOCKER_NETWORK_ID,
        docker_network_subnet=IDENTITY.DOCKER_NETWORK_SUBNET,
        docker_network_gateway=IDENTITY.DOCKER_NETWORK_GATEWAY,
    )


def artifacts() -> Any:
    sources = [
        {
            "role": role,
            "path": relative,
            "bytes": 1,
            "sha256": hashlib.sha256(role.encode("ascii")).hexdigest(),
        }
        for role, relative in IDENTITY.SOURCE_ROLE_PATHS.items()
    ]
    environment = {
        "sources": sources,
        "source_sets": {"all": ALL_SOURCE_SHA},
    }
    model = {"worker": {"binary_sha256": WORKER_SHA}}
    return IDENTITY.IdentityArtifacts(
        environment,
        model,
        b'{"fake":"environment"}\n',
        b'{"fake":"model"}\n',
    )


def promotion_receipt() -> dict[str, Any]:
    return {
        "schema_version": IDENTITY.PROMOTION_SCHEMA,
        "product_root": os.fspath(PRODUCTION.PRODUCTION_PRODUCT_ROOT),
        "created_at": "2026-07-11T12:00:00+00:00",
        "model_revision": IDENTITY.MODEL_REVISION,
        "artifact": {
            "manifest_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "selected_pair_count": 280,
            "payloads_hashed": True,
        },
        "package": {
            "manifest_sha256": "3" * 64,
            "payload_count": 163,
            "payload_bytes": 3_112_499_200,
            "payloads_hashed": True,
        },
        "read_only": True,
        "full_payloads": True,
        "verified": True,
    }


class ProductionIdentityPrepareTests(unittest.TestCase):
    def run_build(
        self,
        *,
        anchor: FakeAnchor | None = None,
        live: Any | None = None,
        built_artifacts: Any | None = None,
        validator: FakeIndependentValidator | None = None,
        receipt: dict[str, Any] | None = None,
        forbidden_values: tuple[bytes, ...] = (SECRET,),
    ) -> Any:
        used_anchor = FakeAnchor() if anchor is None else anchor
        used_live = live_identity() if live is None else live
        used_artifacts = artifacts() if built_artifacts is None else built_artifacts
        used_validator = FakeIndependentValidator() if validator is None else validator
        used_receipt = promotion_receipt() if receipt is None else receipt
        with (
            mock.patch.object(
                IDENTITY,
                "capture_live_identity",
                return_value=used_live,
            ) as capture,
            mock.patch.object(
                IDENTITY,
                "build_identity_artifacts",
                return_value=used_artifacts,
            ) as build,
        ):
            result = PREPARE.build_production_identity_preflight(
                used_anchor,
                used_receipt,
                expected_worker_binary_sha256=WORKER_SHA,
                captured_utc="2026-07-12T00:00:00Z",
                forbidden_values=forbidden_values,
                identity_probe=cast(IDENTITY.IdentityProbe, object()),
                independent_validator=used_validator,
            )
        expectation = capture.call_args.args[1]
        self.assertEqual(expectation.forbidden_values, forbidden_values)
        inputs = build.call_args.args[0]
        self.assertEqual(len(inputs.source_specs), PREPARE.EXPECTED_SOURCE_COUNT)
        self.assertEqual(inputs.git_commit, COMMIT)
        self.assertEqual(inputs.git_status_raw, b"")
        return result

    def test_constants_and_source_contract_match_independent_contracts(self) -> None:
        settings = PREPARE.production_identity_settings()
        self.assertEqual(settings.repo_root, PRODUCTION.PRODUCTION_REPO_ROOT)
        self.assertEqual(settings.product_root, PRODUCTION.PRODUCTION_PRODUCT_ROOT)
        self.assertEqual(
            settings.worker_binary,
            settings.repo_root / "target/release/ullm-sq8-worker",
        )
        self.assertEqual(
            settings.tokenizer_root,
            Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"),
        )
        self.assertEqual(
            settings.effective_service_unit,
            Path("/etc/systemd/system/ullm-openai.service"),
        )
        self.assertEqual(
            settings.effective_environment_file,
            Path("/etc/ullm/openai-gateway.env"),
        )
        with self.assertRaises(PREPARE.ProductionIdentityPreflightError):
            dataclasses.replace(settings, tokenizer_root=Path("/tmp/tokenizer-drift"))
        self.assertEqual(
            IDENTITY.SOURCE_ROLE_PATHS, VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS
        )
        self.assertEqual(len(IDENTITY.SOURCE_ROLE_PATHS), 70)
        self.assertEqual(IDENTITY.HIP_GUARDS, VALIDATOR.HIP_GUARDS)
        self.assertEqual(IDENTITY.UPSTREAM_MODEL_ID, VALIDATOR.UPSTREAM_MODEL_ID)
        self.assertEqual(IDENTITY.SERVED_MODEL_ID, VALIDATOR.SERVED_MODEL_ID)
        self.assertEqual(IDENTITY.MODEL_REVISION, VALIDATOR.MODEL_REVISION)
        self.assertEqual(PREPARE.OPENWEBUI_VERSION, VALIDATOR.OPENWEBUI_VERSION)
        self.assertEqual(
            PREPARE.OPENWEBUI_SOURCE_REVISION, VALIDATOR.OPENWEBUI_SOURCE_REVISION
        )
        self.assertEqual(
            PREPARE.OPENWEBUI_BASE_IMAGE_DIGEST,
            VALIDATOR.OPENWEBUI_BASE_IMAGE_DIGEST,
        )
        self.assertEqual(
            PREPARE.OPENWEBUI_BASE_IMAGE_ID, VALIDATOR.OPENWEBUI_BASE_IMAGE_ID
        )
        self.assertEqual(
            PREPARE.OPENWEBUI_DERIVED_IMAGE_ID,
            VALIDATOR.OPENWEBUI_DERIVED_IMAGE_ID,
        )
        self.assertEqual(
            PREPARE.OPENWEBUI_PATCHED_MIDDLEWARE_SHA256,
            VALIDATOR.OPENWEBUI_PATCHED_MIDDLEWARE_SHA256,
        )
        self.assertEqual(PREPARE.DOCKER_NETWORK_ID, VALIDATOR.DOCKER_NETWORK_ID)
        self.assertEqual(PREPARE.SYSTEMD_MAJOR, VALIDATOR.SYSTEMD_MAJOR)
        self.assertEqual(
            {
                "index": PREPARE.GPU_INDEX,
                "bdf": PREPARE.GPU_BDF,
                "uuid": PREPARE.GPU_UUID,
                "kfd_gpu_id": PREPARE.KFD_GPU_ID,
                "node_id": PREPARE.GPU_NODE_ID,
                "partition_id": PREPARE.GPU_PARTITION_ID,
                "architecture": IDENTITY.DEVICE_ARCHITECTURE,
            },
            VALIDATOR.EXPECTED_GPU_IDENTITY,
        )
        patch = ROOT / IDENTITY.SOURCE_ROLE_PATHS["openwebui_patch"]
        self.assertEqual(
            PREPARE.OPENWEBUI_PATCH_SHA256,
            hashlib.sha256(patch.read_bytes()).hexdigest(),
        )
        live_expectation = PREPARE.production_live_capture_expectation()
        self.assertEqual(live_expectation.hardware.amd_smi_tool, "26.2.2+e1a6bc5663")
        self.assertEqual(live_expectation.hardware.amd_smi_library, "26.2.2")
        self.assertEqual(live_expectation.hardware.rocm_version, "7.2.1")

    def test_build_composes_70_sources_and_independent_validation(self) -> None:
        anchor = FakeAnchor()
        validator = FakeIndependentValidator()
        result = self.run_build(anchor=anchor, validator=validator)

        self.assertEqual(anchor.revalidations, 2)
        self.assertEqual(validator.identity_calls, 1)
        self.assertEqual(validator.source_calls, 1)
        self.assertEqual(result.service_epoch.gateway_pid, 101)
        self.assertEqual(result.service_epoch.worker_pid, 202)
        self.assertEqual(result.service_n_restarts, 2)
        self.assertNotIn(SECRET, repr(result).encode("utf-8"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.service_n_restarts = 3

    def test_operational_expectation_uses_live_identity_and_fixed_endpoints(
        self,
    ) -> None:
        live = live_identity()
        result = PREPARE.build_operational_expectation(
            live,
            container_id="4" * 64,
        )
        self.assertEqual(result.gateway_pid, live.gateway.pid)
        self.assertEqual(result.worker_pid, live.worker.pid)
        self.assertEqual(result.container_id, "4" * 64)
        self.assertEqual(result.image_id, PREPARE.OPENWEBUI_DERIVED_IMAGE_ID)
        self.assertEqual(result.gateway_ready_url, PREPARE.GATEWAY_READY_URL)
        self.assertEqual(result.openwebui_health_url, PREPARE.OPENWEBUI_HEALTH_URL)
        self.assertEqual(result.observer_socket, PREPARE.LIFECYCLE_OBSERVER_SOCKET)

    def test_path_worker_service_openwebui_and_gpu_drift_fail_closed(self) -> None:
        path_anchor = FakeAnchor()
        path_anchor.settings = dataclasses.replace(
            path_anchor.settings, repo_root=Path("/tmp/not-production")
        )
        cases: list[tuple[str, dict[str, Any]]] = [
            ("path", {"anchor": path_anchor}),
            (
                "worker",
                {
                    "live": dataclasses.replace(
                        live_identity(),
                        worker=dataclasses.replace(
                            live_identity().worker, executable_sha256="f" * 64
                        ),
                    )
                },
            ),
            (
                "service",
                {"live": dataclasses.replace(live_identity(), active_state="failed")},
            ),
            (
                "OpenWebUI",
                {
                    "live": dataclasses.replace(
                        live_identity(), derived_image_id="sha256:" + "f" * 64
                    )
                },
            ),
            ("GPU", {"live": dataclasses.replace(live_identity(), gpu_index=1)}),
        ]
        for label, arguments in cases:
            with (
                self.subTest(label=label),
                self.assertRaises(PREPARE.ProductionIdentityPreflightError),
            ):
                self.run_build(**arguments)

    def test_commit_status_source_and_git_toctou_fail_closed(self) -> None:
        commit_anchor = FakeAnchor()
        commit_anchor.mutate_commit_on = 2
        status_anchor = FakeAnchor()
        status_anchor.mutate_status_on = 2
        toctou_anchor = FakeAnchor()
        toctou_anchor.raise_on = 2
        source_list = list(artifacts().environment["sources"])
        source_list.pop()
        source_artifacts = artifacts()._replace(
            environment={
                "sources": source_list,
                "source_sets": {"all": ALL_SOURCE_SHA},
            }
        )
        cases: list[tuple[str, dict[str, Any]]] = [
            ("commit", {"anchor": commit_anchor}),
            ("status", {"anchor": status_anchor}),
            ("source", {"built_artifacts": source_artifacts}),
            ("TOCTOU", {"anchor": toctou_anchor}),
        ]
        for label, arguments in cases:
            with (
                self.subTest(label=label),
                self.assertRaises(PREPARE.ProductionIdentityPreflightError),
            ):
                self.run_build(**arguments)

    def test_receipt_full_flags_and_secret_presence_fail_closed(self) -> None:
        not_full = promotion_receipt()
        not_full["full_payloads"] = False
        not_read_only = promotion_receipt()
        not_read_only["read_only"] = False
        not_verified = promotion_receipt()
        not_verified["verified"] = False
        contains_secret = promotion_receipt()
        contains_secret["created_at"] = SECRET.decode("ascii")
        artifact_secret = artifacts()._replace(
            environment_bytes=b'{"value":"fourteen-byte-secret"}\n'
        )
        cases: list[tuple[str, dict[str, Any]]] = [
            ("full", {"receipt": not_full}),
            ("read-only", {"receipt": not_read_only}),
            ("verified", {"receipt": not_verified}),
            ("receipt secret", {"receipt": contains_secret}),
            ("artifact secret", {"built_artifacts": artifact_secret}),
        ]
        for label, arguments in cases:
            with (
                self.subTest(label=label),
                self.assertRaises(PREPARE.ProductionIdentityPreflightError),
            ):
                self.run_build(**arguments)

    def test_independent_identity_and_source_failures_are_sanitized(self) -> None:
        identity_rejection = FakeIndependentValidator()
        identity_rejection.identity_error = True
        source_rejection = FakeIndependentValidator()
        source_rejection.source_error = True
        source_drift = FakeIndependentValidator()
        source_drift.source_count = 69
        for label, validator in (
            ("identity", identity_rejection),
            ("source", source_rejection),
            ("binding", source_drift),
        ):
            with (
                self.subTest(label=label),
                self.assertRaisesRegex(
                    PREPARE.ProductionIdentityPreflightError,
                    "independent",
                ),
            ):
                self.run_build(validator=validator)

    def test_operational_expectation_rejects_bad_container_id(self) -> None:
        with self.assertRaises(PREPARE.ProductionIdentityPreflightError):
            PREPARE.build_operational_expectation(live_identity(), container_id="short")


if __name__ == "__main__":
    unittest.main()
