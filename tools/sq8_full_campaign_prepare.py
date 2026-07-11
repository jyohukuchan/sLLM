#!/usr/bin/env python3
"""Compose the read-only production identity preflight for one SQ8 campaign."""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast


TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import sq8_full_campaign_identity as identity  # noqa: E402
import sq8_full_campaign_operational as operational  # noqa: E402
import sq8_full_campaign_production as production  # noqa: E402
from sq8_openwebui_campaign import PidEpoch  # noqa: E402


PRODUCTION_TOKENIZER_ROOT = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)
PRODUCTION_WORKER_BINARY = production.PRODUCTION_REPO_ROOT / (
    "target/release/ullm-sq8-worker"
)
PRODUCTION_EFFECTIVE_SERVICE_UNIT = Path("/etc/systemd/system/ullm-openai.service")
PRODUCTION_EFFECTIVE_ENVIRONMENT_FILE = Path("/etc/ullm/openai-gateway.env")

SERVICE_USER = "homelab1"
SERVICE_GROUP = "homelab1"
OPENWEBUI_CONTAINER_NAME = "open-webui"
GATEWAY_READY_URL = "http://172.20.0.1:8000/readyz"
OPENWEBUI_HEALTH_URL = "http://127.0.0.1:3000/health"
LIFECYCLE_OBSERVER_SOCKET = Path("/run/ullm/lifecycle-observer.sock")

OPENWEBUI_VERSION = "0.9.4-ullm.1"
OPENWEBUI_SOURCE_REVISION = "f51d2b026f1b0e7283b15f093412be8b67d24770"
OPENWEBUI_BASE_IMAGE_DIGEST = (
    "sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
)
OPENWEBUI_BASE_IMAGE_REF = (
    "ghcr.io/open-webui/open-webui@" + OPENWEBUI_BASE_IMAGE_DIGEST
)
OPENWEBUI_BASE_IMAGE_ID = (
    "sha256:18247c4608796dd5e416ec1e82f20457837a219ed9c272a8d64b405a262b3399"
)
OPENWEBUI_DERIVED_IMAGE_REF = "ullm/open-webui:0.9.4-ullm.1"
OPENWEBUI_DERIVED_IMAGE_ID = (
    "sha256:ef5ae4fbc06abb662eeefe87e584ea7c69e55838f5f08f637057b9108048b409"
)
OPENWEBUI_PATCH_SHA256 = (
    "20bf654b96f005d5008deff0cdd6f9cd62cbd21fe20a3680b66aecb36190813a"
)
OPENWEBUI_PATCHED_MIDDLEWARE_SHA256 = (
    "b8aa5524fac6971aa8326cbef024b6fe9bcea03b3a00d4e7b0fa559514e0c66a"
)
DOCKER_NETWORK_ID = "79bb7cfca31cb5d76978cbbb229c946662c137b93ea647b5ae6c205af9126dc8"

GPU_INDEX = 2
GPU_BDF = "0000:47:00.0"
GPU_UUID = "a8ff7551-0000-1000-80e9-ddefa2d60f55"
KFD_GPU_ID = 51_545
GPU_NODE_ID = 2
GPU_PARTITION_ID = 0
SYSTEMD_MAJOR = 255
AMD_SMI_TOOL = "26.2.2+e1a6bc5663"
AMD_SMI_LIBRARY = "26.2.2"
ROCM_VERSION = "7.2.1"

EXPECTED_SOURCE_COUNT = 69
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ProductionIdentityPreflightError(RuntimeError):
    """The immutable production identity preflight contract differed."""


def fail(message: str) -> NoReturn:
    raise ProductionIdentityPreflightError(message)


@dataclasses.dataclass(frozen=True, slots=True)
class ProductionIdentitySettings:
    """Fixed deployment paths used to construct the production identity."""

    repo_root: Path
    product_root: Path
    tokenizer_root: Path
    worker_binary: Path
    effective_service_unit: Path
    effective_environment_file: Path

    def __post_init__(self) -> None:
        expected = (
            production.PRODUCTION_REPO_ROOT,
            production.PRODUCTION_PRODUCT_ROOT,
            PRODUCTION_TOKENIZER_ROOT,
            PRODUCTION_WORKER_BINARY,
            PRODUCTION_EFFECTIVE_SERVICE_UNIT,
            PRODUCTION_EFFECTIVE_ENVIRONMENT_FILE,
        )
        actual = (
            self.repo_root,
            self.product_root,
            self.tokenizer_root,
            self.worker_binary,
            self.effective_service_unit,
            self.effective_environment_file,
        )
        if actual != expected or any(
            not isinstance(path, Path)
            or not path.is_absolute()
            or Path(os.path.abspath(path)) != path
            for path in actual
        ):
            fail("production identity path settings differ")


def production_identity_settings() -> ProductionIdentitySettings:
    return ProductionIdentitySettings(
        repo_root=production.PRODUCTION_REPO_ROOT,
        product_root=production.PRODUCTION_PRODUCT_ROOT,
        tokenizer_root=PRODUCTION_TOKENIZER_ROOT,
        worker_binary=PRODUCTION_WORKER_BINARY,
        effective_service_unit=PRODUCTION_EFFECTIVE_SERVICE_UNIT,
        effective_environment_file=PRODUCTION_EFFECTIVE_ENVIRONMENT_FILE,
    )


def production_live_capture_expectation(
    *, forbidden_values: tuple[bytes, ...] = ()
) -> identity.LiveCaptureExpectation:
    """Return the fixed live service, OpenWebUI, and R9700 capture contract."""

    _validate_forbidden_values(forbidden_values)
    openwebui = identity.OpenWebUIExpectation(
        version=OPENWEBUI_VERSION,
        source_revision=OPENWEBUI_SOURCE_REVISION,
        base_image_ref=OPENWEBUI_BASE_IMAGE_REF,
        base_image_digest=OPENWEBUI_BASE_IMAGE_DIGEST,
        base_image_id=OPENWEBUI_BASE_IMAGE_ID,
        derived_image_ref=OPENWEBUI_DERIVED_IMAGE_REF,
        derived_image_id=OPENWEBUI_DERIVED_IMAGE_ID,
        patch_sha256=OPENWEBUI_PATCH_SHA256,
        patched_middleware_sha256=OPENWEBUI_PATCHED_MIDDLEWARE_SHA256,
        docker_network_id=DOCKER_NETWORK_ID,
    )
    hardware = identity.HardwareExpectation(
        gpu_index=GPU_INDEX,
        gpu_bdf=GPU_BDF,
        gpu_uuid=GPU_UUID,
        kfd_gpu_id=KFD_GPU_ID,
        node_id=GPU_NODE_ID,
        partition_id=GPU_PARTITION_ID,
        systemd_major=SYSTEMD_MAJOR,
        amd_smi_tool=AMD_SMI_TOOL,
        amd_smi_library=AMD_SMI_LIBRARY,
        rocm_version=ROCM_VERSION,
    )
    return identity.LiveCaptureExpectation(
        service_unit=identity.SERVICE_UNIT,
        service_user=SERVICE_USER,
        service_group=SERVICE_GROUP,
        service_fragment_path=PRODUCTION_EFFECTIVE_SERVICE_UNIT,
        openwebui=openwebui,
        hardware=hardware,
        forbidden_values=forbidden_values,
    )


class GitAnchorProtocol(Protocol):
    settings: production.ProductionPreflightSettings
    commit: str
    status_raw: bytes

    def revalidate(self) -> None: ...


class IndependentIdentityValidator(Protocol):
    def validate_campaign_identity(
        self,
        bundle: Path,
        *,
        expected_commit: str,
        expected_worker_binary_sha256: str,
    ) -> Any: ...

    def validate_campaign_source_checkout(
        self, identity_data: Any, *, repo_root: Path
    ) -> Any: ...


@dataclasses.dataclass(frozen=True, slots=True)
class ProductionIdentityPreflight:
    """Secret-free identity evidence cached before campaign journal capture."""

    settings: ProductionIdentitySettings
    live_identity: identity.LiveIdentity
    identity_artifacts: identity.IdentityArtifacts
    independent_identity: Any
    source_checkout: Any
    service_epoch: PidEpoch
    service_n_restarts: int


def _validate_forbidden_values(values: tuple[bytes, ...]) -> None:
    if type(values) is not tuple or any(
        type(value) is not bytes or len(value) < 4 for value in values
    ):
        fail("forbidden values must be byte strings of length >= 4")


def _scan_forbidden(values: tuple[bytes, ...], *chunks: bytes) -> None:
    for chunk in chunks:
        if type(chunk) is not bytes:
            fail("secret scan input type differs")
        if any(value in chunk for value in values):
            fail("production identity preflight contains forbidden cleartext")


def _canonical_json(value: Any) -> bytes:
    try:
        raw = (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
        if len(raw) > identity.MAX_DOCUMENT_BYTES:
            fail("promotion validation receipt exceeds its byte bound")
        return raw
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise ProductionIdentityPreflightError(
            "promotion validation receipt is not bounded canonical JSON"
        ) from None


def _validate_full_promotion_receipt(
    receipt: dict[str, Any], settings: ProductionIdentitySettings
) -> None:
    expected_fields = {
        "schema_version",
        "product_root",
        "created_at",
        "model_revision",
        "artifact",
        "package",
        "read_only",
        "full_payloads",
        "verified",
    }
    if (
        type(receipt) is not dict
        or set(receipt) != expected_fields
        or receipt.get("schema_version") != identity.PROMOTION_SCHEMA
        or receipt.get("product_root") != os.fspath(settings.product_root)
        or receipt.get("model_revision") != identity.MODEL_REVISION
        or receipt.get("read_only") is not True
        or receipt.get("full_payloads") is not True
        or receipt.get("verified") is not True
    ):
        fail("full promotion validation receipt differs")
    artifact = receipt.get("artifact")
    package = receipt.get("package")
    artifact_manifest_sha256 = (
        artifact.get("manifest_sha256") if type(artifact) is dict else None
    )
    artifact_content_sha256 = (
        artifact.get("content_sha256") if type(artifact) is dict else None
    )
    package_manifest_sha256 = (
        package.get("manifest_sha256") if type(package) is dict else None
    )
    if (
        type(artifact) is not dict
        or set(artifact)
        != {
            "manifest_sha256",
            "content_sha256",
            "selected_pair_count",
            "payloads_hashed",
        }
        or type(package) is not dict
        or set(package)
        != {
            "manifest_sha256",
            "payload_count",
            "payload_bytes",
            "payloads_hashed",
        }
        or artifact.get("payloads_hashed") is not True
        or package.get("payloads_hashed") is not True
        or type(artifact_manifest_sha256) is not str
        or SHA256_RE.fullmatch(artifact_manifest_sha256) is None
        or type(artifact_content_sha256) is not str
        or SHA256_RE.fullmatch(artifact_content_sha256) is None
        or type(package_manifest_sha256) is not str
        or SHA256_RE.fullmatch(package_manifest_sha256) is None
        or type(artifact.get("selected_pair_count")) is not int
        or artifact["selected_pair_count"] < 1
        or type(package.get("payload_count")) is not int
        or package["payload_count"] < 1
        or type(package.get("payload_bytes")) is not int
        or package["payload_bytes"] < 1
    ):
        fail("full promotion payload validation receipt differs")


def _validate_anchor(
    anchor: GitAnchorProtocol,
    settings: ProductionIdentitySettings,
    *,
    expected_commit: str | None = None,
    expected_status: bytes | None = None,
) -> tuple[str, bytes]:
    anchor_settings = anchor.settings
    if (
        not isinstance(anchor_settings, production.ProductionPreflightSettings)
        or anchor_settings.repo_root != settings.repo_root
        or anchor_settings.product_root != settings.product_root
        or production.GIT_COMMIT_RE.fullmatch(anchor.commit) is None
        or type(anchor.status_raw) is not bytes
        or len(anchor.status_raw) > production.GIT_STATUS_MAX_BYTES
    ):
        fail("production Git anchor binding differs")
    if expected_commit is not None and anchor.commit != expected_commit:
        fail("production Git commit changed during identity preflight")
    if expected_status is not None and anchor.status_raw != expected_status:
        fail("production Git status changed during identity preflight")
    return anchor.commit, bytes(anchor.status_raw)


def _validate_live_identity(
    live: identity.LiveIdentity,
    settings: ProductionIdentitySettings,
    expected_worker_binary_sha256: str,
) -> PidEpoch:
    if not isinstance(live, identity.LiveIdentity):
        fail("live identity result type differs")
    expected_openwebui = production_live_capture_expectation().openwebui
    expected_hardware = production_live_capture_expectation().hardware
    if (
        live.service_unit != identity.SERVICE_UNIT
        or live.service_user != SERVICE_USER
        or live.service_group != SERVICE_GROUP
        or live.service_fragment_path != os.fspath(settings.effective_service_unit)
        or live.active_state != "active"
        or live.sub_state != "running"
        or live.gateway.ppid != 1
        or live.gateway.pid == live.worker.pid
        or live.gateway.children != (live.worker.pid,)
        or live.gateway.uid != live.service_uid
        or live.gateway.gid != live.service_gid
        or live.worker.ppid != live.gateway.pid
        or live.worker.uid != live.service_uid
        or live.worker.gid != live.service_gid
        or live.worker.children
        or live.gateway.starttime_ticks <= 0
        or live.worker.starttime_ticks <= 0
        or Path(live.worker.executable) != settings.worker_binary
        or live.worker.executable_sha256 != expected_worker_binary_sha256
        or type(live.n_restarts) is not int
        or live.n_restarts < 0
        or identity.BOOT_ID_RE.fullmatch(live.boot_id) is None
    ):
        fail("live service epoch or worker identity differs")
    if (
        live.openwebui_version != expected_openwebui.version
        or live.openwebui_source_revision != expected_openwebui.source_revision
        or live.base_image_digest != expected_openwebui.base_image_digest
        or live.base_image_id != expected_openwebui.base_image_id
        or live.derived_image_id != expected_openwebui.derived_image_id
        or live.patch_sha256 != expected_openwebui.patch_sha256
        or live.patched_middleware_sha256
        != expected_openwebui.patched_middleware_sha256
        or live.docker_network_name != expected_openwebui.docker_network_name
        or live.docker_network_id != expected_openwebui.docker_network_id
        or live.docker_network_subnet != expected_openwebui.docker_network_subnet
        or live.docker_network_gateway != expected_openwebui.docker_network_gateway
    ):
        fail("live OpenWebUI identity differs")
    if (
        live.gpu_index != expected_hardware.gpu_index
        or live.gpu_bdf != expected_hardware.gpu_bdf
        or live.gpu_uuid != expected_hardware.gpu_uuid
        or live.kfd_gpu_id != expected_hardware.kfd_gpu_id
        or live.gpu_node_id != expected_hardware.node_id
        or live.gpu_partition_id != expected_hardware.partition_id
        or live.systemd_major != expected_hardware.systemd_major
        or live.amd_smi_tool != expected_hardware.amd_smi_tool
        or live.amd_smi_library != expected_hardware.amd_smi_library
        or live.rocm_version != expected_hardware.rocm_version
    ):
        fail("live R9700 or tool identity differs")
    return PidEpoch(live.gateway.pid, live.worker.pid)


def _validate_artifact_sources(artifacts: identity.IdentityArtifacts) -> str:
    if not isinstance(artifacts, identity.IdentityArtifacts):
        fail("identity artifact result type differs")
    sources = artifacts.environment.get("sources")
    source_sets = artifacts.environment.get("source_sets")
    if type(sources) is not list or type(source_sets) is not dict:
        fail("identity artifact source contract differs")
    roles = [item.get("role") for item in sources if type(item) is dict]
    all_sha = source_sets.get("all")
    if (
        len(sources) != EXPECTED_SOURCE_COUNT
        or len(roles) != EXPECTED_SOURCE_COUNT
        or set(roles) != set(identity.SOURCE_ROLE_PATHS)
        or type(all_sha) is not str
        or SHA256_RE.fullmatch(all_sha) is None
    ):
        fail("69-source identity artifact contract differs")
    assert isinstance(all_sha, str)
    return all_sha


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            fail("temporary identity artifact write made no progress")
        offset += written


def _write_temporary_artifact(directory: Path, name: str, raw: bytes) -> None:
    descriptor = -1
    try:
        descriptor = os.open(
            directory / name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        _write_all(descriptor, raw)
        metadata = os.fstat(descriptor)
        if metadata.st_nlink != 1 or (metadata.st_mode & 0o777) != 0o600:
            fail("temporary identity artifact metadata differs")
    except ProductionIdentityPreflightError:
        raise
    except OSError:
        raise ProductionIdentityPreflightError(
            "failed to stage a temporary identity artifact"
        ) from None
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _run_independent_identity_validation(
    artifacts: identity.IdentityArtifacts,
    validator: IndependentIdentityValidator,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
    repo_root: Path,
) -> tuple[Any, Any]:
    try:
        with tempfile.TemporaryDirectory(prefix="ullm-sq8-identity-") as raw_directory:
            directory = Path(raw_directory)
            os.chmod(directory, 0o700)
            _write_temporary_artifact(
                directory, "environment.json", artifacts.environment_bytes
            )
            _write_temporary_artifact(
                directory, "model-identity.json", artifacts.model_identity_bytes
            )
            independent_identity = validator.validate_campaign_identity(
                directory,
                expected_commit=expected_commit,
                expected_worker_binary_sha256=expected_worker_binary_sha256,
            )
            source_checkout = validator.validate_campaign_source_checkout(
                independent_identity, repo_root=repo_root
            )
            return independent_identity, source_checkout
    except ProductionIdentityPreflightError:
        raise
    except Exception:
        raise ProductionIdentityPreflightError(
            "independent campaign identity validation failed"
        ) from None


def _validate_independent_result(
    independent_identity: Any,
    source_checkout: Any,
    *,
    expected_commit: str,
    expected_worker_binary_sha256: str,
    expected_all_source_sha256: str,
) -> None:
    if (
        getattr(independent_identity, "expected_commit", None) != expected_commit
        or getattr(independent_identity, "expected_worker_binary_sha256", None)
        != expected_worker_binary_sha256
        or getattr(source_checkout, "git_commit", None) != expected_commit
        or getattr(source_checkout, "source_count", None) != EXPECTED_SOURCE_COUNT
        or getattr(source_checkout, "all_source_sha256", None)
        != expected_all_source_sha256
    ):
        fail("independent identity or source checkout binding differs")


def build_production_identity_preflight(
    git_anchor: GitAnchorProtocol,
    promotion_validation_result: dict[str, Any],
    *,
    expected_worker_binary_sha256: str,
    captured_utc: str,
    forbidden_values: tuple[bytes, ...],
    identity_probe: identity.IdentityProbe,
    independent_validator: IndependentIdentityValidator,
) -> ProductionIdentityPreflight:
    """Build and independently verify one secret-free production identity cache."""

    try:
        settings = production_identity_settings()
        _validate_forbidden_values(forbidden_values)
        if SHA256_RE.fullmatch(expected_worker_binary_sha256) is None:
            fail("expected worker binary SHA-256 differs")
        promotion_receipt_raw = _canonical_json(promotion_validation_result)
        _scan_forbidden(forbidden_values, promotion_receipt_raw)
        promotion_receipt = cast(
            dict[str, Any], json.loads(promotion_receipt_raw.decode("ascii"))
        )
        _validate_full_promotion_receipt(promotion_receipt, settings)

        expected_commit, expected_status = _validate_anchor(git_anchor, settings)
        git_anchor.revalidate()
        _validate_anchor(
            git_anchor,
            settings,
            expected_commit=expected_commit,
            expected_status=expected_status,
        )

        live = identity.capture_live_identity(
            identity_probe,
            production_live_capture_expectation(forbidden_values=forbidden_values),
        )
        service_epoch = _validate_live_identity(
            live, settings, expected_worker_binary_sha256
        )
        source_specs = identity.default_source_specs(settings.repo_root)
        if len(source_specs) != EXPECTED_SOURCE_COUNT:
            fail("production source specification count differs")
        artifacts = identity.build_identity_artifacts(
            identity.IdentityBuildInputs(
                repo_root=settings.repo_root,
                product_root=settings.product_root,
                tokenizer_root=settings.tokenizer_root,
                worker_binary=settings.worker_binary,
                effective_service_unit=settings.effective_service_unit,
                effective_environment_file=settings.effective_environment_file,
                promotion_validation_result=promotion_receipt,
                git_commit=expected_commit,
                git_status_raw=expected_status,
                captured_utc=captured_utc,
                source_specs=source_specs,
                forbidden_values=forbidden_values,
            ),
            live,
        )
        all_source_sha256 = _validate_artifact_sources(artifacts)
        _scan_forbidden(
            forbidden_values,
            artifacts.environment_bytes,
            artifacts.model_identity_bytes,
            repr(live).encode("utf-8", errors="strict"),
        )
        independent_identity, source_checkout = _run_independent_identity_validation(
            artifacts,
            independent_validator,
            expected_commit=expected_commit,
            expected_worker_binary_sha256=expected_worker_binary_sha256,
            repo_root=settings.repo_root,
        )
        _validate_independent_result(
            independent_identity,
            source_checkout,
            expected_commit=expected_commit,
            expected_worker_binary_sha256=expected_worker_binary_sha256,
            expected_all_source_sha256=all_source_sha256,
        )
        _scan_forbidden(
            forbidden_values,
            repr(independent_identity).encode("utf-8", errors="strict"),
            repr(source_checkout).encode("utf-8", errors="strict"),
        )

        git_anchor.revalidate()
        _validate_anchor(
            git_anchor,
            settings,
            expected_commit=expected_commit,
            expected_status=expected_status,
        )
        return ProductionIdentityPreflight(
            settings=settings,
            live_identity=live,
            identity_artifacts=artifacts,
            independent_identity=independent_identity,
            source_checkout=source_checkout,
            service_epoch=service_epoch,
            service_n_restarts=live.n_restarts,
        )
    except ProductionIdentityPreflightError:
        raise
    except Exception:
        raise ProductionIdentityPreflightError(
            "production identity preflight failed"
        ) from None


def build_operational_expectation(
    live: identity.LiveIdentity,
    *,
    container_id: str,
) -> operational.OperationalExpectation:
    """Bind injected read-only readers to identities captured before the campaign."""

    if not isinstance(live, identity.LiveIdentity):
        fail("operational live identity type differs")
    expectation = operational.OperationalExpectation(
        service_unit=live.service_unit,
        gateway_pid=live.gateway.pid,
        worker_pid=live.worker.pid,
        container_name=OPENWEBUI_CONTAINER_NAME,
        container_id=container_id,
        image_id=live.derived_image_id,
        network_name=live.docker_network_name,
        network_id=live.docker_network_id,
        gateway_ready_url=GATEWAY_READY_URL,
        openwebui_health_url=OPENWEBUI_HEALTH_URL,
        observer_socket=LIFECYCLE_OBSERVER_SOCKET,
        observer_parent_uid=live.service_uid,
        observer_parent_gid=live.service_gid,
    )
    try:
        operational._validate_expectation(expectation)
    except operational.OperationalError:
        raise ProductionIdentityPreflightError(
            "operational expectation binding differs"
        ) from None
    return expectation


__all__ = [
    "EXPECTED_SOURCE_COUNT",
    "IndependentIdentityValidator",
    "ProductionIdentityPreflight",
    "ProductionIdentityPreflightError",
    "ProductionIdentitySettings",
    "build_operational_expectation",
    "build_production_identity_preflight",
    "production_identity_settings",
    "production_live_capture_expectation",
]
