#!/usr/bin/env python3
"""Build strict, secret-free preflight identity artifacts for the SQ8 campaign."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import hashlib
import json
import math
import os
import pwd
import re
import selectors
import shlex
import stat
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple, NoReturn, Protocol, Sequence, cast


ENVIRONMENT_SCHEMA = "ullm.sq8.full_campaign.environment.v1"
MODEL_IDENTITY_SCHEMA = "ullm.sq8.full_campaign.model_identity.v1"
PROMOTION_SCHEMA = "ullm.sq8_product_promotion.v1"
ARTIFACT_SCHEMA = "sq-fp8-artifact-v0.2"
PACKAGE_SCHEMA = "ullm-prototype-manifest-v0.1"
WORKER_PROTOCOL_SCHEMA = "ullm.worker.v1"

UPSTREAM_MODEL_ID = "Qwen/Qwen3-14B-FP8"
SERVED_MODEL_ID = "ullm-qwen3-14b-sq8"
MODEL_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
SERVICE_UNIT = "ullm-openai.service"
DOCKER_NETWORK_NAME = "open-webui-network"
DOCKER_NETWORK_SUBNET = "172.20.0.0/16"
DOCKER_NETWORK_GATEWAY = "172.20.0.1"

DEVICE_ARCHITECTURE = "gfx1201"
EXECUTION_PROFILE = "rdna4_w8a8_block_ck"
CONTEXT_LENGTH = 4096
MAX_COMPLETION_TOKENS = 512
VOCAB_SIZE = 151_936

COPY_CHUNK_BYTES = 64 << 10
MAX_SOURCE_BYTES = 32 << 20
MAX_DOCUMENT_BYTES = 2 << 20
MAX_TOKENIZER_FILE_BYTES = 64 << 20
MAX_WORKER_BINARY_BYTES = 2 << 30
MAX_PROBE_BYTES = 8 << 20

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_ID_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
BOOT_ID_RE = re.compile(r"[0-9a-f]{32}\Z")
NETWORK_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
BDF_RE = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]\Z")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

HIP_GUARDS = (
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_CACHED_PREFIX_ATTN_F32_FLASH2_KERNEL",
    "ULLM_REQUIRE_HIP_CAUSAL_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
)

SOURCE_ROLE_PATHS = {
    "identity_generator": "tools/sq8_full_campaign_identity.py",
    "product_promotion_validator": "tools/validate-sq8-product-promotion.py",
    "product_promotion_canonical": "tools/sq8_canonical_artifact.py",
    "release_validator": "tools/validate-sq8-openwebui-release.py",
    "release_collector": "tools/collect-sq8-openwebui-release.py",
    "campaign_journal": "tools/sq8_openwebui_campaign.py",
    "campaign_bundle": "tools/sq8_full_campaign_bundle.py",
    "campaign_independent_metrics": "tools/sq8_full_campaign_independent_metrics.py",
    "campaign_independent_views": "tools/sq8_full_campaign_independent_views.py",
    "campaign_orchestrator": "tools/run-sq8-full-openwebui-campaign.py",
    "campaign_production": "tools/sq8_full_campaign_production.py",
    "campaign_prepare": "tools/sq8_full_campaign_prepare.py",
    "campaign_operational": "tools/sq8_full_campaign_operational.py",
    "worker_acceptance": "tools/run-sq8-worker-acceptance.py",
    "campaign_resource": "tools/sq8_full_campaign_resource.py",
    "campaign_renderer": "tools/sq8_full_campaign_renderer.py",
    "campaign_views": "tools/sq8_full_campaign_views.py",
    "gate_api_contract": "tools/run-sq8-api-contract-gate.py",
    "ingest_api_contract": "tools/sq8_api_contract_gate_ingest.py",
    "gate_openwebui_stop": "tools/run-openwebui-stop-gate.py",
    "ingest_openwebui_stop": "tools/sq8_openwebui_stop_gate_ingest.py",
    "gate_openwebui_soak": "tools/run-openwebui-soak-gate.py",
    "ingest_openwebui_gate": "tools/sq8_openwebui_gate_ingest.py",
    "gate_direct_cancel": "tools/run-sq8-direct-cancel-gate.py",
    "gate_openwebui_failure": "tools/run-openwebui-failure-gate.py",
    "gate_openwebui_failure_hook": "tools/run-openwebui-failure-hook.py",
    "ingest_openwebui_failure": "tools/sq8_openwebui_failure_gate_ingest.py",
    "gate_http_latency": "tools/run-sq8-http-latency-gate.py",
    "ingest_http_latency": "tools/sq8_http_latency_gate_ingest.py",
    "http_client": "tools/sq8-openwebui-http-client.py",
    "browser_smoke": "deploy/openwebui/browser-smoke.cjs",
    "browser_stop": "deploy/openwebui/browser-stop-smoke.cjs",
    "browser_failure": "deploy/openwebui/browser-failure-smoke.cjs",
    "browser_soak": "deploy/openwebui/browser-soak.cjs",
    "openwebui_dockerfile": "deploy/openwebui/Dockerfile",
    "openwebui_compose": "deploy/openwebui/compose.yaml",
    "openwebui_configure": "deploy/openwebui/configure.py",
    "openwebui_patch": "deploy/openwebui/provider-stream-error.patch",
    "openwebui_image_validator": "deploy/openwebui/verify-derived-image.sh",
    "systemd_service": "deploy/systemd/ullm-openai.service",
    "systemd_environment_contract": "deploy/systemd/ullm-openai.env.example",
    "gateway_pyproject": "services/openai-gateway/pyproject.toml",
    "gateway_lock": "services/openai-gateway/uv.lock",
    "gateway_init": "services/openai-gateway/src/ullm_openai_gateway/__init__.py",
    "gateway_main": "services/openai-gateway/src/ullm_openai_gateway/__main__.py",
    "gateway_app": "services/openai-gateway/src/ullm_openai_gateway/app.py",
    "gateway_errors": "services/openai-gateway/src/ullm_openai_gateway/errors.py",
    "gateway_schemas": "services/openai-gateway/src/ullm_openai_gateway/schemas.py",
    "gateway_settings": "services/openai-gateway/src/ullm_openai_gateway/settings.py",
    "gateway_tokenizer": "services/openai-gateway/src/ullm_openai_gateway/tokenizer.py",
    "gateway_worker": "services/openai-gateway/src/ullm_openai_gateway/worker.py",
    "worker_cargo_manifest": "crates/ullm-engine/Cargo.toml",
    "worker_entrypoint": "crates/ullm-engine/src/bin/ullm-sq8-worker.rs",
    "worker_backend": "crates/ullm-engine/src/sq8_worker_backend.rs",
    "worker_protocol": "crates/ullm-engine/src/sq8_worker_protocol.rs",
    "worker_runtime": "crates/ullm-engine/src/sq8_worker_runtime.rs",
    "engine_library": "crates/ullm-engine/src/lib.rs",
    "workspace_lock": "Cargo.lock",
    "serving_fixture_manifest": "tests/fixtures/sq8-serving-v0.1/manifest.json",
    "chat_template_fixture_manifest": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/manifest.json"
    ),
    "runtime_oracle_validation": (
        "benchmarks/results/2026-07-10/sq8-serving-v0.1/runtime-oracle-validation.json"
    ),
    "spec_release": "docs/specs/sq8-openwebui-release-v0.1.md",
    "spec_openai_chat_subset": "docs/specs/openai-chat-subset-v0.1.md",
    "spec_worker_protocol": "docs/specs/sq8-worker-protocol-v0.1.md",
    "fixture_ttft_p0032": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0032.json"
    ),
    "fixture_ttft_p0128": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0128.json"
    ),
    "fixture_ttft_p0512": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p0512.json"
    ),
    "fixture_ttft_p2048": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p2048.json"
    ),
    "fixture_ttft_p3584": (
        "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures/exact-p3584.json"
    ),
}

SOURCE_GROUPS = {
    "gateway": (
        "gateway_pyproject",
        "gateway_lock",
        "gateway_init",
        "gateway_main",
        "gateway_app",
        "gateway_errors",
        "gateway_schemas",
        "gateway_settings",
        "gateway_tokenizer",
        "gateway_worker",
    ),
    "worker": (
        "worker_cargo_manifest",
        "worker_entrypoint",
        "worker_backend",
        "worker_protocol",
        "worker_runtime",
        "engine_library",
        "workspace_lock",
    ),
    "collector": ("release_collector", "campaign_journal"),
    "browser": ("browser_smoke", "browser_stop", "browser_failure", "browser_soak"),
    "http_client": ("http_client",),
    "deployment": (
        "openwebui_dockerfile",
        "openwebui_compose",
        "openwebui_configure",
        "openwebui_patch",
        "openwebui_image_validator",
        "systemd_service",
        "systemd_environment_contract",
    ),
    "oracle": (
        "serving_fixture_manifest",
        "chat_template_fixture_manifest",
        "runtime_oracle_validation",
    ),
    "campaign": (
        "identity_generator",
        "product_promotion_validator",
        "product_promotion_canonical",
        "release_validator",
        "release_collector",
        "campaign_journal",
        "campaign_bundle",
        "campaign_independent_metrics",
        "campaign_independent_views",
        "campaign_orchestrator",
        "campaign_production",
        "campaign_prepare",
        "campaign_operational",
        "worker_acceptance",
        "campaign_resource",
        "campaign_renderer",
        "campaign_views",
        "gate_api_contract",
        "ingest_api_contract",
        "gate_openwebui_stop",
        "ingest_openwebui_stop",
        "gate_openwebui_soak",
        "ingest_openwebui_gate",
        "gate_direct_cancel",
        "gate_openwebui_failure",
        "gate_openwebui_failure_hook",
        "ingest_openwebui_failure",
        "gate_http_latency",
        "ingest_http_latency",
        "http_client",
        "browser_smoke",
        "browser_stop",
        "browser_failure",
        "browser_soak",
    ),
    "spec": ("spec_release", "spec_openai_chat_subset", "spec_worker_protocol"),
    "fixture": (
        "serving_fixture_manifest",
        "chat_template_fixture_manifest",
        "fixture_ttft_p0032",
        "fixture_ttft_p0128",
        "fixture_ttft_p0512",
        "fixture_ttft_p2048",
        "fixture_ttft_p3584",
    ),
    "all": tuple(SOURCE_ROLE_PATHS),
}

TTFT_FIXTURE_IDENTITIES = {
    "fixture_ttft_p0032": {
        "path": SOURCE_ROLE_PATHS["fixture_ttft_p0032"],
        "bytes": 1_333,
        "sha256": "c660c7fb3c25d2a3e25693e2beb2abc10295a06935772d17d23cedab04f24c07",
    },
    "fixture_ttft_p0128": {
        "path": SOURCE_ROLE_PATHS["fixture_ttft_p0128"],
        "bytes": 2_776,
        "sha256": "f8fe81bacb8761f3aa10cce1c333a51f9a85d65b5bfc7b02499886fb9f550a37",
    },
    "fixture_ttft_p0512": {
        "path": SOURCE_ROLE_PATHS["fixture_ttft_p0512"],
        "bytes": 8_538,
        "sha256": "e2f53c514a228e9e10871fc0df1867394aae12416215c9716770d2b420a3480f",
    },
    "fixture_ttft_p2048": {
        "path": SOURCE_ROLE_PATHS["fixture_ttft_p2048"],
        "bytes": 31_581,
        "sha256": "cd04c3339542f07731074ac0e00740a83061e620f6caff9c2a7e5316df1ccdcf",
    },
    "fixture_ttft_p3584": {
        "path": SOURCE_ROLE_PATHS["fixture_ttft_p3584"],
        "bytes": 54_622,
        "sha256": "e3cd6c722302f73d688492b73a182298f34cc0a1498def209c262e5e9aa92912",
    },
}

TOKENIZER_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

RUNTIME_ENV_KEYS = (
    "ULLM_WORKER_BINARY",
    "ULLM_PRODUCT_ROOT",
    "ULLM_TOKENIZER_DIR",
    "ULLM_API_KEY_FILE",
    "ULLM_GPU_LOCK_FILE",
    "ULLM_BIND_HOST",
    "ULLM_BIND_PORT",
)


class IdentityError(RuntimeError):
    """A fail-closed identity error that never includes captured evidence."""


def fail(message: str) -> NoReturn:
    raise IdentityError(message)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_text(value: Any, label: str, *, maximum: int = 65_536) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        fail(f"{label} is not bounded non-empty text")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail(f"{label} is not an integer >= {minimum}")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} is not lowercase SHA-256")
    return value


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        fail(f"{label} fields differ")
    return cast(dict[str, Any], value)


def _same_json(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        actual_object = cast(dict[str, Any], actual)
        expected_object = cast(dict[str, Any], expected)
        return set(actual_object) == set(expected_object) and all(
            _same_json(actual_object[key], expected_object[key])
            for key in expected_object
        )
    if type(expected) is list:
        actual_array = cast(list[Any], actual)
        return len(actual_array) == len(expected) and all(
            _same_json(left, right)
            for left, right in zip(actual_array, expected, strict=True)
        )
    return bool(actual == expected)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail("identity JSON contains a duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    fail("identity JSON contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail("identity JSON contains a non-finite number")
    return parsed


def _json_object(
    raw: bytes, label: str, *, maximum: int = MAX_DOCUMENT_BYTES
) -> dict[str, Any]:
    if not raw or len(raw) > maximum:
        fail(f"{label} size differs")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except IdentityError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict JSON")
    if type(value) is not dict:
        fail(f"{label} root is not an object")
    return cast(dict[str, Any], value)


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise IdentityError("identity document cannot be serialized") from error


def _reject_key_recursive(value: Any, key: str) -> None:
    if type(value) is dict:
        if key in value:
            fail(f"identity document contains forbidden key {key}")
        for item in value.values():
            _reject_key_recursive(item, key)
    elif type(value) is list:
        for item in value:
            _reject_key_recursive(item, key)


class _SecretScanner:
    def __init__(self, values: tuple[bytes, ...]):
        for value in values:
            if type(value) is not bytes or len(value) < 4:
                fail("forbidden values must be byte strings of length >= 4")
        self.values = values
        self.overlap = max((len(value) for value in values), default=1) - 1
        self.tail = b""

    def consume(self, chunk: bytes) -> None:
        combined = self.tail + chunk
        if any(value in combined for value in self.values):
            fail("identity input contains forbidden cleartext")
        self.tail = combined[-self.overlap :] if self.overlap else b""


@dataclasses.dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    uid: int
    gid: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for identity capture")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW


def _file_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        fail("O_NOFOLLOW is required for identity capture")
    return os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


class _PinnedFile:
    def __init__(
        self,
        path: Path,
        *,
        maximum: int,
        forbidden_values: tuple[bytes, ...],
        retain: bool,
        expected_sha256: str | None = None,
        require_single_link: bool = True,
    ):
        self.path = Path(os.path.abspath(path))
        self.maximum = maximum
        self.forbidden_values = forbidden_values
        self.retain = retain
        self.expected_sha256 = expected_sha256
        self.require_single_link = require_single_link
        self.parent_fd = -1
        self.fd = -1
        self.identity: _FileIdentity | None = None
        self.raw: bytes | None = None
        self.sha256 = ""
        self.bytes = 0
        if expected_sha256 is not None:
            _sha(expected_sha256, "expected pinned file SHA-256")
        try:
            self.parent_fd = os.open(self.path.parent, _directory_flags())
            entry = _FileIdentity.from_stat(
                os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
            )
            if (
                not stat.S_ISREG(entry.mode)
                or (require_single_link and entry.links != 1)
                or (not require_single_link and entry.links < 1)
                or entry.size < 1
                or entry.size > maximum
            ):
                fail("identity input is not one bounded regular file")
            self.fd = os.open(self.path.name, _file_flags(), dir_fd=self.parent_fd)
            opened = _FileIdentity.from_stat(os.fstat(self.fd))
            if opened != entry:
                fail("identity input changed while opening")
            self.identity = opened
            raw, size, digest = self._snapshot(retain=retain)
            self.raw = raw
            self.bytes = size
            self.sha256 = digest
            if expected_sha256 is not None and digest != expected_sha256:
                fail("identity input SHA-256 differs from its binding")
        except IdentityError:
            self.close()
            raise
        except OSError:
            self.close()
            fail("failed to pin an identity input")

    def _entry(self) -> _FileIdentity:
        try:
            return _FileIdentity.from_stat(
                os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
            )
        except OSError:
            fail("pinned identity input entry is unavailable")

    def _snapshot(self, *, retain: bool) -> tuple[bytes | None, int, str]:
        assert self.identity is not None
        try:
            if _FileIdentity.from_stat(os.fstat(self.fd)) != self.identity:
                fail("identity input changed before reading")
            os.lseek(self.fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            scanner = _SecretScanner(self.forbidden_values)
            total = 0
            while True:
                chunk = os.read(self.fd, COPY_CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > self.maximum:
                    fail("identity input exceeded its streaming bound")
                scanner.consume(chunk)
                digest.update(chunk)
                if retain:
                    chunks.append(chunk)
            if (
                total != self.identity.size
                or _FileIdentity.from_stat(os.fstat(self.fd)) != self.identity
            ):
                fail("identity input changed while reading")
            return b"".join(chunks) if retain else None, total, digest.hexdigest()
        except IdentityError:
            raise
        except OSError:
            fail("failed to stream a pinned identity input")

    def seal(self) -> None:
        assert self.identity is not None
        if (
            _FileIdentity.from_stat(os.fstat(self.fd)) != self.identity
            or self._entry() != self.identity
        ):
            fail("pinned identity input changed before sealing")
        raw, size, digest = self._snapshot(retain=self.retain)
        if size != self.bytes or digest != self.sha256 or raw != self.raw:
            fail("pinned identity input bytes changed before sealing")
        if (
            _FileIdentity.from_stat(os.fstat(self.fd)) != self.identity
            or self._entry() != self.identity
        ):
            fail("pinned identity input changed during sealing")

    def close(self) -> None:
        for fd in (self.fd, self.parent_fd):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self.fd = self.parent_fd = -1


class _PinSet:
    def __init__(self) -> None:
        self.items: list[_PinnedFile] = []

    def open(self, path: Path, **kwargs: Any) -> _PinnedFile:
        item = _PinnedFile(path, **kwargs)
        self.items.append(item)
        return item

    def seal(self) -> None:
        for item in self.items:
            item.seal()

    def close(self) -> None:
        for item in reversed(self.items):
            item.close()
        self.items.clear()


@dataclasses.dataclass(frozen=True)
class SourceFileSpec:
    role: str
    logical_path: str
    path: Path
    expected_sha256: str | None = None
    maximum_bytes: int = MAX_SOURCE_BYTES


@dataclasses.dataclass(frozen=True)
class RuntimeConfiguration:
    worker_binary: Path
    product_root: Path
    tokenizer_root: Path
    api_key_file: Path
    gpu_lock_file: Path
    bind_host: str
    bind_port: int
    hip_visible_devices: str = "1"
    hip_guards: tuple[str, ...] = HIP_GUARDS


@dataclasses.dataclass(frozen=True)
class OpenWebUIExpectation:
    version: str
    source_revision: str
    base_image_ref: str
    base_image_digest: str
    base_image_id: str
    derived_image_ref: str
    derived_image_id: str
    patch_sha256: str
    patched_middleware_sha256: str
    docker_network_id: str
    docker_network_name: str = DOCKER_NETWORK_NAME
    docker_network_subnet: str = DOCKER_NETWORK_SUBNET
    docker_network_gateway: str = DOCKER_NETWORK_GATEWAY


@dataclasses.dataclass(frozen=True)
class HardwareExpectation:
    gpu_index: int
    gpu_bdf: str
    gpu_uuid: str
    kfd_gpu_id: int
    node_id: int
    partition_id: int
    systemd_major: int
    amd_smi_tool: str
    amd_smi_library: str
    rocm_version: str
    cgroup_fs_type: str = "cgroup2fs"


@dataclasses.dataclass(frozen=True)
class LiveCaptureExpectation:
    service_unit: str
    service_user: str
    service_group: str
    service_fragment_path: Path
    openwebui: OpenWebUIExpectation
    hardware: HardwareExpectation
    forbidden_values: tuple[bytes, ...] = ()


@dataclasses.dataclass(frozen=True)
class ProcessSnapshot:
    pid: int
    ppid: int
    uid: int
    gid: int
    starttime_ticks: int
    executable: str
    executable_bytes: int
    executable_sha256: str
    children: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class LiveIdentity:
    os_id: str
    os_version_id: str
    os_pretty_name: str
    kernel_sysname: str
    kernel_release: str
    kernel_version: str
    kernel_machine: str
    boot_id: str
    cgroup_fs_type: str
    systemd_major: int
    systemd_version_line: str
    python_version_line: str
    rustc_version_line: str
    cargo_version_line: str
    docker_version: str
    docker_api_version: str
    docker_os: str
    docker_arch: str
    docker_kernel_version: str
    amd_smi_tool: str
    amd_smi_library: str
    rocm_version: str
    amd_smi_version_line: str
    gpu_index: int
    gpu_bdf: str
    gpu_uuid: str
    kfd_gpu_id: int
    gpu_node_id: int
    gpu_partition_id: int
    service_unit: str
    service_user: str
    service_group: str
    service_uid: int
    service_gid: int
    service_fragment_path: str
    control_group: str
    gateway: ProcessSnapshot
    worker: ProcessSnapshot
    n_restarts: int
    active_state: str
    sub_state: str
    openwebui_version: str
    openwebui_source_revision: str
    base_image_digest: str
    base_image_id: str
    derived_image_id: str
    patch_sha256: str
    patched_middleware_sha256: str
    docker_network_name: str
    docker_network_id: str
    docker_network_subnet: str
    docker_network_gateway: str


@dataclasses.dataclass(frozen=True)
class IdentityBuildInputs:
    repo_root: Path
    product_root: Path
    tokenizer_root: Path
    worker_binary: Path
    effective_service_unit: Path
    effective_environment_file: Path
    promotion_validation_result: dict[str, Any]
    git_commit: str
    git_status_raw: bytes
    captured_utc: str
    source_specs: tuple[SourceFileSpec, ...]
    forbidden_values: tuple[bytes, ...] = ()


class IdentityArtifacts(NamedTuple):
    environment: dict[str, Any]
    model_identity: dict[str, Any]
    environment_bytes: bytes
    model_identity_bytes: bytes


class IdentityProbe(Protocol):
    def os_release(self) -> bytes: ...

    def uname(self) -> tuple[str, str, str, str]: ...

    def boot_id(self) -> bytes: ...

    def cgroup_fs_type(self) -> bytes: ...

    def systemd_version(self) -> bytes: ...

    def python_version(self) -> bytes: ...

    def rustc_version(self) -> bytes: ...

    def cargo_version(self) -> bytes: ...

    def service_show(self, unit: str) -> bytes: ...

    def account_ids(self, user: str) -> tuple[int, int]: ...

    def process(self, pid: int) -> ProcessSnapshot: ...

    def process_starttime(self, pid: int) -> int: ...

    def docker_version(self) -> bytes: ...

    def docker_network(self, name: str) -> bytes: ...

    def docker_images(self, base_ref: str, derived_ref: str) -> bytes: ...

    def amd_smi_version(self) -> bytes: ...

    def amd_smi_list(self) -> bytes: ...


def _validate_source_contract(
    role_paths: dict[str, str] | None = None,
    groups: dict[str, tuple[str, ...]] | None = None,
) -> None:
    checked_paths = SOURCE_ROLE_PATHS if role_paths is None else role_paths
    checked_groups = SOURCE_GROUPS if groups is None else groups
    if type(checked_paths) is not dict or not checked_paths:
        fail("source role contract differs")
    paths: list[str] = []
    for role, path in checked_paths.items():
        pure = PurePosixPath(path) if type(path) is str else None
        if (
            type(role) is not str
            or not role
            or pure is None
            or pure.is_absolute()
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or "\\" in path
        ):
            fail("source role or path contract differs")
        paths.append(path)
    if len(paths) != len(set(paths)):
        fail("source role paths are not unique")
    if (
        type(checked_groups) is not dict
        or not checked_groups
        or "all" not in checked_groups
    ):
        fail("source group contract differs")
    known_roles = set(checked_paths)
    semantic_roles: set[str] = set()
    for group, roles in checked_groups.items():
        if (
            type(group) is not str
            or not group
            or type(roles) is not tuple
            or not roles
            or len(roles) != len(set(roles))
            or any(type(role) is not str or role not in known_roles for role in roles)
        ):
            fail("source group roles differ")
        if group != "all":
            semantic_roles.update(roles)
    if checked_groups["all"] != tuple(checked_paths) or semantic_roles != known_roles:
        fail("source group coverage differs")


def default_source_specs(repo_root: Path) -> tuple[SourceFileSpec, ...]:
    _validate_source_contract()
    root = Path(os.path.abspath(repo_root))
    return tuple(
        SourceFileSpec(role, relative, root / relative)
        for role, relative in SOURCE_ROLE_PATHS.items()
    )


def _read_pseudo_file(path: Path, maximum: int) -> bytes:
    descriptor = -1
    try:
        descriptor = os.open(path, _file_flags())
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            fail("live identity input is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(COPY_CHUNK_BYTES, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                fail("live identity input exceeded its bound")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            fail("live identity input changed while reading")
        return b"".join(chunks)
    except IdentityError:
        raise
    except OSError:
        fail("failed to read a live identity input")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_proc_stat(raw: bytes, expected_pid: int) -> tuple[int, int]:
    try:
        prefix = f"{expected_pid} (".encode("ascii")
        if not raw.startswith(prefix):
            fail("process stat PID differs")
        close = raw.rfind(b") ")
        if close < len(prefix):
            fail("process stat framing differs")
        fields = raw[close + 2 :].split()
        if len(fields) < 20:
            fail("process stat field count differs")
        ppid_raw = fields[1]
        starttime_raw = fields[19]
        if not ppid_raw.isdigit() or not starttime_raw.isdigit():
            fail("process stat numeric identity differs")
        ppid = int(ppid_raw, 10)
        starttime = int(starttime_raw, 10)
    except (UnicodeError, ValueError, OverflowError):
        fail("process stat cannot be decoded")
    if ppid < 0 or starttime <= 0:
        fail("process stat identity is out of range")
    return ppid, starttime


def _parse_proc_status(raw: bytes) -> tuple[int, int]:
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError:
        fail("process status is not ASCII")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Uid", "Gid"}:
            if key in fields:
                fail("process status identity field is duplicated")
            fields[key] = value.strip()
    if set(fields) != {"Uid", "Gid"}:
        fail("process status lacks UID or GID")
    parsed: list[int] = []
    for key in ("Uid", "Gid"):
        values = fields[key].split()
        if len(values) != 4 or any(not value.isdecimal() for value in values):
            fail("process status UID or GID fields differ")
        numbers = [int(value, 10) for value in values]
        if len(set(numbers)) != 1:
            fail("process credential IDs are not stable")
        parsed.append(numbers[0])
    return parsed[0], parsed[1]


def _hash_process_executable(path: Path) -> tuple[int, str]:
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        before = os.fstat(descriptor)
        linked = os.stat(path)
        if not stat.S_ISREG(before.st_mode) or (before.st_dev, before.st_ino) != (
            linked.st_dev,
            linked.st_ino,
        ):
            fail("live process executable identity differs")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, COPY_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_WORKER_BINARY_BYTES:
                fail("live process executable exceeded its bound")
            digest.update(chunk)
        after = os.fstat(descriptor)
        linked_after = os.stat(path)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or total != before.st_size
        ):
            fail("live process executable changed while hashing")
        return total, digest.hexdigest()
    except IdentityError:
        raise
    except OSError:
        fail("failed to hash a live process executable")
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _run_bounded_command(
    argv: Sequence[str],
    *,
    maximum: int = MAX_PROBE_BYTES,
    timeout_seconds: float = 10.0,
) -> bytes:
    if (
        not argv
        or any(type(item) is not str or not item or "\x00" in item for item in argv)
        or maximum < 1
        or timeout_seconds <= 0
    ):
        fail("live identity probe command is invalid")
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LC_ALL": "C"},
            close_fds=True,
        )
        assert process.stdout is not None
        descriptor = process.stdout.fileno()
        selector.register(descriptor, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout_seconds
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                fail("live identity probe timed out")
            events = selector.select(min(remaining, 0.25))
            if not events:
                if process.poll() is not None:
                    chunk = os.read(descriptor, COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                else:
                    continue
            else:
                chunk = os.read(descriptor, COPY_CHUNK_BYTES)
                if not chunk:
                    break
            total += len(chunk)
            if total > maximum:
                process.kill()
                fail("live identity probe output exceeded its bound")
            chunks.append(chunk)
        return_code = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        if return_code != 0:
            fail("live identity probe command failed")
        return b"".join(chunks)
    except IdentityError:
        raise
    except (OSError, subprocess.SubprocessError):
        fail("failed to execute a live identity probe")
    finally:
        selector.close()
        if process is not None and process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.SubprocessError:
                pass


class SystemIdentityProbe:
    """Read-only host probe implementation used by the future orchestrator."""

    def __init__(self, *, proc_root: Path = Path("/proc")):
        self.proc_root = proc_root

    def os_release(self) -> bytes:
        return _read_pseudo_file(Path("/usr/lib/os-release"), 64 << 10)

    def uname(self) -> tuple[str, str, str, str]:
        value = os.uname()
        return value.sysname, value.release, value.version, value.machine

    def boot_id(self) -> bytes:
        return _read_pseudo_file(
            self.proc_root / "sys" / "kernel" / "random" / "boot_id", 128
        )

    def cgroup_fs_type(self) -> bytes:
        return _run_bounded_command(
            ("stat", "-fc", "%T", "/sys/fs/cgroup"), maximum=128
        )

    def systemd_version(self) -> bytes:
        return _run_bounded_command(("systemctl", "--version"), maximum=64 << 10)

    def python_version(self) -> bytes:
        return _run_bounded_command(("/usr/bin/python3", "--version"), maximum=4096)

    def rustc_version(self) -> bytes:
        return _run_bounded_command(("/usr/bin/rustc", "--version"), maximum=4096)

    def cargo_version(self) -> bytes:
        return _run_bounded_command(("/usr/bin/cargo", "--version"), maximum=4096)

    def service_show(self, unit: str) -> bytes:
        return _run_bounded_command(
            (
                "systemctl",
                "show",
                unit,
                "--property=ActiveState",
                "--property=SubState",
                "--property=ControlGroup",
                "--property=MainPID",
                "--property=NRestarts",
                "--property=User",
                "--property=Group",
                "--property=FragmentPath",
                "--no-pager",
            ),
            maximum=4096,
        )

    def account_ids(self, user: str) -> tuple[int, int]:
        try:
            account = pwd.getpwnam(user)
        except KeyError:
            fail("systemd service user does not exist")
        return account.pw_uid, account.pw_gid

    def process_starttime(self, pid: int) -> int:
        raw = _read_pseudo_file(self.proc_root / str(pid) / "stat", 1 << 20)
        return _parse_proc_stat(raw, pid)[1]

    def process(self, pid: int) -> ProcessSnapshot:
        process_root = self.proc_root / str(pid)
        stat_before = _read_pseudo_file(process_root / "stat", 1 << 20)
        ppid, starttime = _parse_proc_stat(stat_before, pid)
        uid, gid = _parse_proc_status(
            _read_pseudo_file(process_root / "status", 1 << 20)
        )
        children_raw = _read_pseudo_file(
            process_root / "task" / str(pid) / "children", 1 << 20
        )
        children: list[int] = []
        for value in children_raw.split():
            if not value.isdigit() or int(value, 10) <= 0:
                fail("process children identity differs")
            children.append(int(value, 10))
        if children != sorted(set(children)):
            fail("process child PID list is not ascending and unique")
        try:
            executable = os.readlink(process_root / "exe")
        except OSError:
            fail("failed to resolve a live process executable")
        if not executable.startswith("/"):
            fail("live process executable path is not absolute")
        executable_bytes, executable_sha256 = _hash_process_executable(
            process_root / "exe"
        )
        stat_after = _read_pseudo_file(process_root / "stat", 1 << 20)
        final_ppid, final_starttime = _parse_proc_stat(stat_after, pid)
        if (final_ppid, final_starttime) != (ppid, starttime):
            fail("live process identity changed during capture")
        return ProcessSnapshot(
            pid,
            ppid,
            uid,
            gid,
            starttime,
            executable,
            executable_bytes,
            executable_sha256,
            tuple(children),
        )

    def docker_version(self) -> bytes:
        return _run_bounded_command(
            ("docker", "version", "--format", "{{json .Server}}")
        )

    def docker_network(self, name: str) -> bytes:
        return _run_bounded_command(("docker", "network", "inspect", name))

    def docker_images(self, base_ref: str, derived_ref: str) -> bytes:
        return _run_bounded_command(
            ("docker", "image", "inspect", base_ref, derived_ref)
        )

    def amd_smi_version(self) -> bytes:
        return _run_bounded_command(
            ("/opt/rocm/bin/amd-smi", "version"), maximum=64 << 10
        )

    def amd_smi_list(self) -> bytes:
        return _run_bounded_command(("/opt/rocm/bin/amd-smi", "list", "--json"))


def _probe_raw(raw: bytes, forbidden_values: tuple[bytes, ...], label: str) -> bytes:
    if type(raw) is not bytes or not raw or len(raw) > MAX_PROBE_BYTES:
        fail(f"{label} output size differs")
    scanner = _SecretScanner(forbidden_values)
    for offset in range(0, len(raw), COPY_CHUNK_BYTES):
        scanner.consume(raw[offset : offset + COPY_CHUNK_BYTES])
    return raw


def _probe_version_line(
    raw: bytes,
    forbidden_values: tuple[bytes, ...],
    label: str,
    prefix: str,
) -> str:
    value = _probe_raw(raw, forbidden_values, label)
    try:
        lines = value.decode("utf-8", errors="strict").splitlines()
    except UnicodeError:
        fail(f"{label} is not strict UTF-8")
    if len(lines) != 1 or not lines[0].startswith(prefix):
        fail(f"{label} line differs")
    return _safe_text(lines[0], label, maximum=4096)


def _key_value_lines(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail(f"{label} is not UTF-8")
    result: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in result:
            fail(f"{label} contains malformed or duplicate fields")
        result[key] = value
    return result


def _os_release(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("os-release is not strict UTF-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, encoded = line.partition("=")
        if not separator or not key or key in values:
            fail("os-release contains malformed or duplicate fields")
        try:
            parsed = shlex.split(encoded, posix=True)
        except ValueError:
            fail("os-release value quoting differs")
        if len(parsed) != 1:
            fail("os-release value field differs")
        values[key] = parsed[0]
    required = {"ID", "VERSION_ID", "PRETTY_NAME"}
    if not required.issubset(values):
        fail("os-release lacks required identity fields")
    return values


def _json_any(raw: bytes, label: str) -> Any:
    if not raw or len(raw) > MAX_PROBE_BYTES:
        fail(f"{label} size differs")
    try:
        return json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_parse_float,
            parse_constant=_reject_constant,
        )
    except IdentityError:
        raise
    except (UnicodeError, ValueError, RecursionError):
        fail(f"{label} is not strict JSON")


def _capture_live_identity(
    probe: IdentityProbe, expectation: LiveCaptureExpectation
) -> LiveIdentity:
    forbidden = expectation.forbidden_values
    _SecretScanner(forbidden)
    if expectation.service_unit != SERVICE_UNIT:
        fail("live service unit differs from the campaign contract")
    hardware = expectation.hardware
    openwebui = expectation.openwebui
    if (
        type(hardware.gpu_index) is not int
        or hardware.gpu_index < 0
        or BDF_RE.fullmatch(hardware.gpu_bdf) is None
        or UUID_RE.fullmatch(hardware.gpu_uuid) is None
        or type(hardware.kfd_gpu_id) is not int
        or hardware.kfd_gpu_id <= 0
        or IMAGE_ID_RE.fullmatch(openwebui.base_image_id) is None
        or IMAGE_ID_RE.fullmatch(openwebui.derived_image_id) is None
        or not openwebui.base_image_digest.startswith("sha256:")
        or SHA256_RE.fullmatch(openwebui.base_image_digest[7:]) is None
        or NETWORK_ID_RE.fullmatch(openwebui.docker_network_id) is None
    ):
        fail("live hardware or container expectation syntax differs")

    service_raw_before = _probe_raw(
        probe.service_show(expectation.service_unit), forbidden, "systemd service"
    )
    service_before = _key_value_lines(service_raw_before, "systemd service")
    service_fields = {
        "ActiveState",
        "SubState",
        "ControlGroup",
        "MainPID",
        "NRestarts",
        "User",
        "Group",
        "FragmentPath",
    }
    if set(service_before) != service_fields:
        fail("systemd service field set differs")
    if (
        service_before["ActiveState"] != "active"
        or service_before["SubState"] != "running"
        or service_before["User"] != expectation.service_user
        or service_before["Group"] != expectation.service_group
        or service_before["FragmentPath"]
        != os.fspath(Path(os.path.abspath(expectation.service_fragment_path)))
        or service_before["ControlGroup"] != f"/system.slice/{expectation.service_unit}"
        or not service_before["MainPID"].isdecimal()
        or not service_before["NRestarts"].isdecimal()
    ):
        fail("systemd service identity differs")
    gateway_pid = int(service_before["MainPID"], 10)
    n_restarts = int(service_before["NRestarts"], 10)
    if gateway_pid <= 0:
        fail("systemd MainPID is invalid")
    service_uid, service_gid = probe.account_ids(expectation.service_user)
    gateway = probe.process(gateway_pid)
    if (
        gateway.pid != gateway_pid
        or gateway.ppid != 1
        or gateway.uid != service_uid
        or gateway.gid != service_gid
        or len(gateway.children) != 1
    ):
        fail("gateway process identity differs")
    worker = probe.process(gateway.children[0])
    if (
        worker.ppid != gateway.pid
        or worker.uid != service_uid
        or worker.gid != service_gid
        or Path(worker.executable).name != "ullm-sq8-worker"
        or worker.children
    ):
        fail("worker process identity differs")
    for process_value, label in ((gateway, "gateway"), (worker, "worker")):
        if (
            process_value.pid <= 0
            or process_value.starttime_ticks <= 0
            or process_value.executable_bytes <= 0
            or not process_value.executable.startswith("/")
            or SHA256_RE.fullmatch(process_value.executable_sha256) is None
        ):
            fail(f"{label} process snapshot is incomplete")

    os_values = _os_release(
        _probe_raw(probe.os_release(), forbidden, "operating system identity")
    )
    uname = probe.uname()
    if len(uname) != 4 or any(type(value) is not str or not value for value in uname):
        fail("kernel identity differs")
    boot_raw = _probe_raw(probe.boot_id(), forbidden, "boot identity")
    try:
        boot_id = boot_raw.decode("ascii", errors="strict").strip().replace("-", "")
    except UnicodeError:
        fail("boot identity is not ASCII")
    if BOOT_ID_RE.fullmatch(boot_id) is None:
        fail("boot identity syntax differs")
    cgroup_raw = _probe_raw(
        probe.cgroup_fs_type(), forbidden, "cgroup filesystem identity"
    )
    try:
        cgroup_fs_type = cgroup_raw.decode("ascii", errors="strict").strip()
    except UnicodeError:
        fail("cgroup filesystem identity is not ASCII")
    if cgroup_fs_type != hardware.cgroup_fs_type:
        fail("cgroup filesystem identity differs")

    systemd_raw = _probe_raw(
        probe.systemd_version(), forbidden, "systemd version identity"
    )
    try:
        systemd_line = systemd_raw.decode("utf-8", errors="strict").splitlines()[0]
    except (UnicodeError, IndexError):
        fail("systemd version identity is invalid")
    match = re.fullmatch(r"systemd ([0-9]+)(?: .*)?", systemd_line)
    if match is None or int(match.group(1), 10) != hardware.systemd_major:
        fail("systemd major version differs")

    python_version_line = _probe_version_line(
        probe.python_version(), forbidden, "Python version identity", "Python "
    )
    rustc_version_line = _probe_version_line(
        probe.rustc_version(), forbidden, "rustc version identity", "rustc "
    )
    cargo_version_line = _probe_version_line(
        probe.cargo_version(), forbidden, "Cargo version identity", "cargo "
    )

    docker_value = _json_any(
        _probe_raw(probe.docker_version(), forbidden, "Docker version identity"),
        "Docker version identity",
    )
    if type(docker_value) is not dict:
        fail("Docker version identity root is not an object")
    docker = cast(dict[str, Any], docker_value)
    docker_version = _safe_text(docker.get("Version"), "Docker version")
    docker_api = _safe_text(docker.get("ApiVersion"), "Docker API version")
    docker_os = _safe_text(docker.get("Os"), "Docker OS")
    docker_arch = _safe_text(docker.get("Arch"), "Docker architecture")
    docker_kernel = _safe_text(docker.get("KernelVersion"), "Docker kernel")
    if docker_os != "linux" or docker_kernel != uname[1]:
        fail("Docker host identity differs from the kernel")

    network_value = _json_any(
        _probe_raw(
            probe.docker_network(openwebui.docker_network_name),
            forbidden,
            "Docker network identity",
        ),
        "Docker network identity",
    )
    if (
        type(network_value) is not list
        or len(network_value) != 1
        or type(network_value[0]) is not dict
    ):
        fail("Docker network inspection shape differs")
    network = network_value[0]
    ipam = network.get("IPAM")
    configs = ipam.get("Config") if type(ipam) is dict else None
    if (
        network.get("Name") != openwebui.docker_network_name
        or network.get("Id") != openwebui.docker_network_id
        or network.get("Driver") != "bridge"
        or type(configs) is not list
        or len(configs) != 1
        or type(configs[0]) is not dict
        or configs[0].get("Subnet") != openwebui.docker_network_subnet
        or configs[0].get("Gateway") != openwebui.docker_network_gateway
    ):
        fail("Docker network content identity differs")

    images_value = _json_any(
        _probe_raw(
            probe.docker_images(openwebui.base_image_ref, openwebui.derived_image_ref),
            forbidden,
            "Docker image identity",
        ),
        "Docker image identity",
    )
    if type(images_value) is not list or len(images_value) != 2:
        fail("Docker image inspection shape differs")
    images = {item.get("Id"): item for item in images_value if type(item) is dict}
    if set(images) != {openwebui.base_image_id, openwebui.derived_image_id}:
        fail("Docker image content identities differ")
    base = cast(dict[str, Any], images[openwebui.base_image_id])
    derived = cast(dict[str, Any], images[openwebui.derived_image_id])
    repo_digests = base.get("RepoDigests")
    if (
        type(repo_digests) is not list
        or openwebui.base_image_ref not in repo_digests
        or not openwebui.base_image_ref.endswith(openwebui.base_image_digest)
    ):
        fail("OpenWebUI base image registry identity differs")
    config = derived.get("Config")
    labels = config.get("Labels") if type(config) is dict else None
    required_labels = {
        "org.opencontainers.image.version": openwebui.version,
        "org.opencontainers.image.revision": openwebui.source_revision,
        "org.opencontainers.image.base.digest": openwebui.base_image_digest,
        "io.ullm.openwebui.base.image.id": openwebui.base_image_id,
        "io.ullm.openwebui.patch.sha256": openwebui.patch_sha256,
        "io.ullm.openwebui.middleware.sha256": openwebui.patched_middleware_sha256,
    }
    if type(labels) is not dict or any(
        labels.get(key) != value for key, value in required_labels.items()
    ):
        fail("derived OpenWebUI image labels differ")

    amd_version_raw = _probe_raw(
        probe.amd_smi_version(), forbidden, "AMD SMI version identity"
    )
    try:
        amd_version_line = amd_version_raw.decode("utf-8", errors="strict").strip()
    except UnicodeError:
        fail("AMD SMI version identity is not UTF-8")
    for expected in (
        hardware.amd_smi_tool,
        hardware.amd_smi_library,
        hardware.rocm_version,
    ):
        if expected not in amd_version_line:
            fail("AMD SMI or ROCm version differs")
    gpu_value = _json_any(
        _probe_raw(probe.amd_smi_list(), forbidden, "AMD SMI GPU identity"),
        "AMD SMI GPU identity",
    )
    if type(gpu_value) is not list:
        fail("AMD SMI GPU list root differs")
    matches = [
        item
        for item in gpu_value
        if type(item) is dict
        and item.get("gpu") == hardware.gpu_index
        and item.get("bdf") == hardware.gpu_bdf
        and item.get("uuid") == hardware.gpu_uuid
        and item.get("kfd_id") == hardware.kfd_gpu_id
        and item.get("node_id") == hardware.node_id
        and item.get("partition_id") == hardware.partition_id
    ]
    if len(matches) != 1:
        fail("AMD SMI physical GPU identity differs")

    service_raw_after = _probe_raw(
        probe.service_show(expectation.service_unit), forbidden, "systemd service"
    )
    service_after = _key_value_lines(service_raw_after, "systemd service")
    if not _same_json(service_after, service_before):
        fail("systemd service identity changed during live capture")
    if (
        probe.process_starttime(gateway.pid) != gateway.starttime_ticks
        or probe.process_starttime(worker.pid) != worker.starttime_ticks
    ):
        fail("gateway or worker identity changed during live capture")

    return LiveIdentity(
        os_values["ID"],
        os_values["VERSION_ID"],
        os_values["PRETTY_NAME"],
        uname[0],
        uname[1],
        uname[2],
        uname[3],
        boot_id,
        cgroup_fs_type,
        hardware.systemd_major,
        systemd_line,
        python_version_line,
        rustc_version_line,
        cargo_version_line,
        docker_version,
        docker_api,
        docker_os,
        docker_arch,
        docker_kernel,
        hardware.amd_smi_tool,
        hardware.amd_smi_library,
        hardware.rocm_version,
        amd_version_line,
        hardware.gpu_index,
        hardware.gpu_bdf,
        hardware.gpu_uuid,
        hardware.kfd_gpu_id,
        hardware.node_id,
        hardware.partition_id,
        expectation.service_unit,
        expectation.service_user,
        expectation.service_group,
        service_uid,
        service_gid,
        os.fspath(Path(os.path.abspath(expectation.service_fragment_path))),
        service_before["ControlGroup"],
        gateway,
        worker,
        n_restarts,
        service_before["ActiveState"],
        service_before["SubState"],
        openwebui.version,
        openwebui.source_revision,
        openwebui.base_image_digest,
        openwebui.base_image_id,
        openwebui.derived_image_id,
        openwebui.patch_sha256,
        openwebui.patched_middleware_sha256,
        openwebui.docker_network_name,
        openwebui.docker_network_id,
        openwebui.docker_network_subnet,
        openwebui.docker_network_gateway,
    )


def capture_live_identity(
    probe: IdentityProbe, expectation: LiveCaptureExpectation
) -> LiveIdentity:
    try:
        return _capture_live_identity(probe, expectation)
    except IdentityError:
        raise
    except Exception as error:
        raise IdentityError("live identity capture failed") from error


def _validate_timestamp(value: Any, label: str) -> str:
    text = _safe_text(value, label, maximum=128)
    if not text.endswith("Z"):
        fail(f"{label} must use UTC Z notation")
    try:
        parsed = datetime.datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError:
        fail(f"{label} is not ISO-8601")
    if parsed.tzinfo is None or parsed.utcoffset() != datetime.timedelta(0):
        fail(f"{label} is not UTC")
    return text


def _parse_runtime_environment(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeError:
        fail("effective service environment is not UTF-8")
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if (
            not separator
            or key not in RUNTIME_ENV_KEYS
            or key in values
            or not value
            or any(character.isspace() for character in value)
            or "\x00" in value
        ):
            fail("effective service environment fields differ")
        values[key] = value
    if set(values) != set(RUNTIME_ENV_KEYS):
        fail("effective service environment key set differs")
    return values


def _runtime_configuration(
    values: dict[str, str], inputs: IdentityBuildInputs
) -> RuntimeConfiguration:
    if not values["ULLM_BIND_PORT"].isdecimal():
        fail("effective service port is invalid")
    configuration = RuntimeConfiguration(
        worker_binary=Path(values["ULLM_WORKER_BINARY"]),
        product_root=Path(values["ULLM_PRODUCT_ROOT"]),
        tokenizer_root=Path(values["ULLM_TOKENIZER_DIR"]),
        api_key_file=Path(values["ULLM_API_KEY_FILE"]),
        gpu_lock_file=Path(values["ULLM_GPU_LOCK_FILE"]),
        bind_host=values["ULLM_BIND_HOST"],
        bind_port=int(values["ULLM_BIND_PORT"], 10),
    )
    expected_paths = (
        (configuration.worker_binary, inputs.worker_binary),
        (configuration.product_root, inputs.product_root),
        (configuration.tokenizer_root, inputs.tokenizer_root),
    )
    if any(
        Path(os.path.abspath(actual)) != Path(os.path.abspath(expected))
        for actual, expected in expected_paths
    ):
        fail("effective runtime paths differ from the preflight bindings")
    if (
        not configuration.api_key_file.is_absolute()
        or not configuration.gpu_lock_file.is_absolute()
        or configuration.bind_host != DOCKER_NETWORK_GATEWAY
        or configuration.bind_port != 8000
    ):
        fail("effective runtime configuration differs from the product contract")
    return configuration


def _source_aggregate(
    entries_by_role: dict[str, dict[str, Any]], roles: Sequence[str]
) -> str:
    value = [entries_by_role[role] for role in sorted(roles)]
    return _sha256(_canonical(value))


def _validate_source_specs(
    inputs: IdentityBuildInputs,
) -> tuple[SourceFileSpec, ...]:
    _validate_source_contract()
    if type(inputs.source_specs) is not tuple:
        fail("source specs must be an immutable tuple")
    roles = [spec.role for spec in inputs.source_specs]
    if len(roles) != len(set(roles)) or set(roles) != set(SOURCE_ROLE_PATHS):
        fail("full campaign source role set differs")
    repo_root = Path(os.path.abspath(inputs.repo_root))
    result: list[SourceFileSpec] = []
    for spec in inputs.source_specs:
        if not isinstance(spec, SourceFileSpec):
            fail("full campaign source spec type differs")
        expected_logical = SOURCE_ROLE_PATHS[spec.role]
        if (
            spec.logical_path != expected_logical
            or Path(os.path.abspath(spec.path)) != repo_root / expected_logical
            or type(spec.maximum_bytes) is not int
            or not 1 <= spec.maximum_bytes <= MAX_SOURCE_BYTES
        ):
            fail("full campaign source path or bound differs")
        if spec.expected_sha256 is not None:
            _sha(spec.expected_sha256, "source binding SHA-256")
        result.append(spec)
    return tuple(result)


def _validate_promotion_receipt(
    result: dict[str, Any],
    promotion: dict[str, Any],
    product_root: Path,
) -> None:
    _exact(
        result,
        {
            "schema_version",
            "product_root",
            "created_at",
            "model_revision",
            "artifact",
            "package",
            "read_only",
            "full_payloads",
            "verified",
        },
        "promotion validator result",
    )
    if (
        result["schema_version"] != PROMOTION_SCHEMA
        or result["product_root"] != os.fspath(Path(os.path.abspath(product_root)))
        or result["created_at"] != promotion["created_at"]
        or result["model_revision"] != MODEL_REVISION
        or result["read_only"] is not True
        or result["full_payloads"] is not True
        or result["verified"] is not True
    ):
        fail("promotion validator result identity or full-validation state differs")
    artifact = _exact(
        result["artifact"],
        {
            "manifest_sha256",
            "content_sha256",
            "selected_pair_count",
            "payloads_hashed",
        },
        "promotion validator artifact result",
    )
    package = _exact(
        result["package"],
        {"manifest_sha256", "payload_count", "payload_bytes", "payloads_hashed"},
        "promotion validator package result",
    )
    if (
        artifact["payloads_hashed"] is not True
        or package["payloads_hashed"] is not True
    ):
        fail("promotion validator receipt is not a full payload validation")
    _sha(artifact["manifest_sha256"], "validated artifact manifest SHA-256")
    _sha(artifact["content_sha256"], "validated artifact content SHA-256")
    _integer(
        artifact["selected_pair_count"], "validated selected pair count", minimum=1
    )
    _sha(package["manifest_sha256"], "validated package manifest SHA-256")
    _integer(package["payload_count"], "validated package payload count", minimum=1)
    _integer(package["payload_bytes"], "validated package payload bytes", minimum=1)


def _promotion_identity(
    result: dict[str, Any],
    promotion_raw: bytes,
    artifact_manifest: _PinnedFile,
    package_manifest: _PinnedFile,
    product_root: Path,
) -> dict[str, Any]:
    promotion = _json_object(promotion_raw, "promotion document")
    _exact(
        promotion,
        {
            "schema_version",
            "created_at",
            "plan_commit",
            "model",
            "artifact",
            "package",
            "copy",
        },
        "promotion document",
    )
    if promotion["schema_version"] != PROMOTION_SCHEMA:
        fail("promotion schema differs")
    created_at = _safe_text(
        promotion["created_at"], "promotion created_at", maximum=128
    )
    try:
        datetime.datetime.fromisoformat(created_at)
    except ValueError:
        fail("promotion created_at is not ISO-8601")
    if promotion["model"] != {"id": UPSTREAM_MODEL_ID, "revision": MODEL_REVISION}:
        fail("promotion model identity differs")
    plan_commit = _safe_text(
        promotion["plan_commit"], "promotion plan commit", maximum=40
    )
    artifact = _exact(
        promotion["artifact"],
        {
            "source",
            "destination",
            "schema_version",
            "manifest_bytes",
            "manifest_sha256",
            "content_sha256",
            "selected_pair_count",
            "payload_bytes",
            "file_count",
            "verified",
        },
        "promotion artifact",
    )
    package = _exact(
        promotion["package"],
        {
            "source",
            "destination",
            "schema_version",
            "manifest_bytes",
            "manifest_sha256",
            "payload_count",
            "file_count",
            "verified",
        },
        "promotion package",
    )
    root = Path(os.path.abspath(product_root))
    if (
        artifact["destination"] != os.fspath(root / "artifact")
        or artifact["schema_version"] != ARTIFACT_SCHEMA
        or artifact["verified"] is not True
        or package["destination"] != os.fspath(root / "package")
        or package["schema_version"] != PACKAGE_SCHEMA
        or package["verified"] is not True
        or not _same_json(
            promotion["copy"],
            {
                "method": "rsync_archive_streaming",
                "source_and_destination_manifests_byte_identical": True,
                "destination_read_only": True,
            },
        )
    ):
        fail("promotion destination, schema, or immutable copy identity differs")
    for value, label in (
        (artifact["manifest_bytes"], "artifact manifest bytes"),
        (artifact["selected_pair_count"], "artifact selected pair count"),
        (artifact["payload_bytes"], "artifact payload bytes"),
        (artifact["file_count"], "artifact file count"),
        (package["manifest_bytes"], "package manifest bytes"),
        (package["payload_count"], "package payload count"),
        (package["file_count"], "package file count"),
    ):
        _integer(value, label, minimum=1)
    for value, label in (
        (artifact["manifest_sha256"], "artifact manifest SHA-256"),
        (artifact["content_sha256"], "artifact content SHA-256"),
        (package["manifest_sha256"], "package manifest SHA-256"),
    ):
        _sha(value, label)
    if (
        artifact_manifest.bytes != artifact["manifest_bytes"]
        or artifact_manifest.sha256 != artifact["manifest_sha256"]
        or package_manifest.bytes != package["manifest_bytes"]
        or package_manifest.sha256 != package["manifest_sha256"]
    ):
        fail("promoted manifest bytes or SHA-256 differ")
    assert artifact_manifest.raw is not None and package_manifest.raw is not None
    artifact_value = _json_object(artifact_manifest.raw, "artifact manifest")
    if (
        artifact_value.get("schema_version") != ARTIFACT_SCHEMA
        or type(artifact_value.get("integrity")) is not dict
        or artifact_value["integrity"].get("content_sha256")
        != artifact["content_sha256"]
        or type(artifact_value.get("coverage")) is not dict
        or artifact_value["coverage"].get("selected_pair_count")
        != artifact["selected_pair_count"]
        or type(artifact_value.get("storage")) is not dict
        or artifact_value["storage"].get("total_payload_bytes")
        != artifact["payload_bytes"]
    ):
        fail("artifact manifest content identity differs")
    package_value = _json_object(package_manifest.raw, "package manifest")
    entries = package_value.get("passthrough_tensors")
    if (
        package_value.get("schema_version") != PACKAGE_SCHEMA
        or package_value.get("tensors") != []
        or type(entries) is not list
        or len(entries) != package["payload_count"]
    ):
        fail("package manifest structure differs")
    payload_bytes = 0
    names: set[str] = set()
    payload_files: set[str] = set()
    for entry in entries:
        if type(entry) is not dict:
            fail("package passthrough entry is not an object")
        name = _safe_text(entry.get("name"), "package tensor name")
        payload_file = _safe_text(entry.get("payload_file"), "package payload file")
        pure = PurePosixPath(payload_file)
        if (
            name in names
            or payload_file in payload_files
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
        ):
            fail("package tensor or payload path identity differs")
        names.add(name)
        payload_files.add(payload_file)
        payload_bytes += _integer(
            entry.get("payload_bytes"), "package payload byte count", minimum=1
        )
        _sha(entry.get("payload_sha256"), "package payload SHA-256")
    _validate_promotion_receipt(result, promotion, root)
    result_artifact = cast(dict[str, Any], result["artifact"])
    result_package = cast(dict[str, Any], result["package"])
    if (
        result_artifact["manifest_sha256"] != artifact["manifest_sha256"]
        or result_artifact["content_sha256"] != artifact["content_sha256"]
        or result_artifact["selected_pair_count"] != artifact["selected_pair_count"]
        or result_package["manifest_sha256"] != package["manifest_sha256"]
        or result_package["payload_count"] != package["payload_count"]
        or result_package["payload_bytes"] != payload_bytes
    ):
        fail("promotion receipt differs from the promoted manifests")
    return {
        "created_at": created_at,
        "plan_commit": plan_commit,
        "artifact": artifact,
        "package": {**package, "payload_bytes": payload_bytes},
    }


def _process_document(value: ProcessSnapshot) -> dict[str, Any]:
    return {
        "pid": value.pid,
        "ppid": value.ppid,
        "uid": value.uid,
        "gid": value.gid,
        "starttime_ticks": value.starttime_ticks,
        "executable": value.executable,
        "executable_bytes": value.executable_bytes,
        "executable_sha256": value.executable_sha256,
        "children": list(value.children),
    }


def _serving_oracle_identity(
    serving_manifest: _PinnedFile,
    chat_manifest: _PinnedFile,
    runtime_validation: _PinnedFile,
    chat_template_identity: dict[str, Any],
) -> dict[str, Any]:
    assert serving_manifest.raw is not None
    value = _json_object(serving_manifest.raw, "serving fixture manifest")
    if value.get("schema_version") != "ullm.sq8.serving_fixtures.v1":
        fail("serving fixture manifest schema differs")
    vllm = _exact(
        value.get("vllm_identity"),
        {
            "async_scheduling",
            "backend",
            "device",
            "dtype",
            "enable_prefix_caching",
            "enforce_eager",
            "max_num_seqs",
            "package_version",
            "pipeline_parallel_size",
            "python_version",
            "rocr_visible_devices",
            "runner",
            "source_revision_from_package_version",
            "tensor_parallel_size",
            "torch_git_version",
            "torch_hip_version",
            "torch_version",
            "transformers_version",
        },
        "serving vLLM identity",
    )
    device = _exact(
        vllm["device"],
        {
            "compute_capability",
            "gfx",
            "name",
            "total_memory_bytes",
            "visible_device_index",
        },
        "serving vLLM device identity",
    )
    if (
        vllm["backend"] != "vLLM"
        or vllm["async_scheduling"] is not False
        or vllm["enable_prefix_caching"] is not False
        or vllm["enforce_eager"] is not True
        or vllm["max_num_seqs"] != 1
        or vllm["pipeline_parallel_size"] != 1
        or vllm["tensor_parallel_size"] != 1
        or vllm["rocr_visible_devices"] != "1"
        or vllm["runner"] != "LLM.generate"
        or device["gfx"] != DEVICE_ARCHITECTURE
        or device["visible_device_index"] != 0
    ):
        fail("serving vLLM execution identity differs")
    for key in (
        "dtype",
        "package_version",
        "python_version",
        "source_revision_from_package_version",
        "torch_git_version",
        "torch_hip_version",
        "torch_version",
        "transformers_version",
    ):
        _safe_text(vllm[key], f"serving vLLM {key}")
    capability = device["compute_capability"]
    if (
        type(capability) is not list
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        fail("serving vLLM compute capability differs")
    _safe_text(device["name"], "serving vLLM device name")
    _integer(device["total_memory_bytes"], "serving vLLM device memory", minimum=1)

    tokenizer = _exact(
        value.get("tokenizer_identity"),
        {
            "chat_template_sha256",
            "chat_template_utf8_bytes",
            "files",
            "revision",
            "tokenizer_class",
        },
        "serving oracle tokenizer identity",
    )
    if (
        tokenizer["revision"] != MODEL_REVISION
        or tokenizer["chat_template_utf8_bytes"] != chat_template_identity["utf8_bytes"]
        or tokenizer["chat_template_sha256"] != chat_template_identity["sha256"]
        or type(tokenizer["files"]) is not list
        or not tokenizer["files"]
    ):
        fail("serving oracle tokenizer or chat template identity differs")
    _safe_text(tokenizer["tokenizer_class"], "serving tokenizer class")
    _sha(tokenizer["chat_template_sha256"], "serving chat template SHA-256")

    chat_fixture = _exact(
        value.get("chat_template_fixture"),
        {
            "directory",
            "exact_prompt_lengths",
            "manifest_file",
            "manifest_sha256",
            "status",
            "validator",
        },
        "chat template fixture identity",
    )
    if (
        chat_fixture["directory"] != "chat-template"
        or chat_fixture["exact_prompt_lengths"] != [32, 128, 512, 2048, 3584]
        or chat_fixture["manifest_file"] != "chat-template/manifest.json"
        or chat_fixture["manifest_sha256"] != chat_manifest.sha256
        or chat_fixture["status"] != "ready_independent_recompute_passed"
        or chat_fixture["validator"] != "tools/validate-sq8-chat-template-fixtures.py"
    ):
        fail("chat template fixture manifest identity differs")
    return {
        "serving_fixture_manifest": {
            "path": SOURCE_ROLE_PATHS["serving_fixture_manifest"],
            "bytes": serving_manifest.bytes,
            "sha256": serving_manifest.sha256,
        },
        "chat_template_fixture_manifest": {
            "path": SOURCE_ROLE_PATHS["chat_template_fixture_manifest"],
            "bytes": chat_manifest.bytes,
            "sha256": chat_manifest.sha256,
        },
        "runtime_oracle_validation": {
            "path": SOURCE_ROLE_PATHS["runtime_oracle_validation"],
            "bytes": runtime_validation.bytes,
            "sha256": runtime_validation.sha256,
        },
        "vllm_identity": copy.deepcopy(vllm),
    }


def _environment_document(
    inputs: IdentityBuildInputs,
    live: LiveIdentity,
    configuration: RuntimeConfiguration,
    source_entries: list[dict[str, Any]],
    source_sets: dict[str, str],
    unit_file: _PinnedFile,
    environment_file: _PinnedFile,
) -> dict[str, Any]:
    return {
        "schema_version": ENVIRONMENT_SCHEMA,
        "record_type": "environment",
        "captured_utc": inputs.captured_utc,
        "git": {
            "commit": inputs.git_commit,
            "dirty": bool(inputs.git_status_raw),
            "status_sha256": _sha256(inputs.git_status_raw),
        },
        "sources": source_entries,
        "source_sets": source_sets,
        "deployment": {
            "service_unit_file": {
                "path": os.fspath(Path(os.path.abspath(inputs.effective_service_unit))),
                "bytes": unit_file.bytes,
                "sha256": unit_file.sha256,
            },
            "environment_file": {
                "path": os.fspath(
                    Path(os.path.abspath(inputs.effective_environment_file))
                ),
                "bytes": environment_file.bytes,
                "sha256": environment_file.sha256,
            },
            "configuration": {
                "worker_binary": os.fspath(
                    Path(os.path.abspath(configuration.worker_binary))
                ),
                "product_root": os.fspath(
                    Path(os.path.abspath(configuration.product_root))
                ),
                "tokenizer_root": os.fspath(
                    Path(os.path.abspath(configuration.tokenizer_root))
                ),
                "api_key_file": os.fspath(configuration.api_key_file),
                "gpu_lock_file": os.fspath(configuration.gpu_lock_file),
                "bind_host": configuration.bind_host,
                "bind_port": configuration.bind_port,
                "hip_visible_devices": configuration.hip_visible_devices,
                "hip_guards": list(configuration.hip_guards),
            },
        },
        "host": {
            "os": {
                "id": live.os_id,
                "version_id": live.os_version_id,
                "pretty_name": live.os_pretty_name,
            },
            "kernel": {
                "sysname": live.kernel_sysname,
                "release": live.kernel_release,
                "version": live.kernel_version,
                "machine": live.kernel_machine,
            },
            "boot_id": live.boot_id,
            "cgroup_fs_type": live.cgroup_fs_type,
            "tools": {
                "systemd_major": live.systemd_major,
                "systemd_version_line": live.systemd_version_line,
                "python_version_line": live.python_version_line,
                "rustc_version_line": live.rustc_version_line,
                "cargo_version_line": live.cargo_version_line,
                "docker_version": live.docker_version,
                "docker_api_version": live.docker_api_version,
                "docker_os": live.docker_os,
                "docker_arch": live.docker_arch,
                "docker_kernel_version": live.docker_kernel_version,
                "amd_smi_tool": live.amd_smi_tool,
                "amd_smi_library": live.amd_smi_library,
                "rocm_version": live.rocm_version,
                "amd_smi_version_line": live.amd_smi_version_line,
            },
            "gpu": {
                "index": live.gpu_index,
                "bdf": live.gpu_bdf,
                "uuid": live.gpu_uuid,
                "kfd_gpu_id": live.kfd_gpu_id,
                "node_id": live.gpu_node_id,
                "partition_id": live.gpu_partition_id,
                "architecture": DEVICE_ARCHITECTURE,
            },
        },
        "service": {
            "unit": live.service_unit,
            "user": live.service_user,
            "group": live.service_group,
            "uid": live.service_uid,
            "gid": live.service_gid,
            "fragment_path": live.service_fragment_path,
            "control_group": live.control_group,
            "gateway": _process_document(live.gateway),
            "worker": _process_document(live.worker),
            "n_restarts": live.n_restarts,
            "active_state": live.active_state,
            "sub_state": live.sub_state,
        },
        "openwebui": {
            "version": live.openwebui_version,
            "source_revision": live.openwebui_source_revision,
            "base_image_digest": live.base_image_digest,
            "base_image_id": live.base_image_id,
            "derived_image_id": live.derived_image_id,
            "Dockerfile_sha256": next(
                item["sha256"]
                for item in source_entries
                if item["role"] == "openwebui_dockerfile"
            ),
            "patch_sha256": live.patch_sha256,
            "patched_middleware_sha256": live.patched_middleware_sha256,
            "network_name": live.docker_network_name,
            "network_id": live.docker_network_id,
            "network_subnet": live.docker_network_subnet,
            "network_gateway": live.docker_network_gateway,
        },
    }


def _model_document(
    inputs: IdentityBuildInputs,
    receipt_raw: bytes,
    promotion_file: _PinnedFile,
    promotion_identity: dict[str, Any],
    artifact_manifest: _PinnedFile,
    package_manifest: _PinnedFile,
    tokenizer_entries: list[dict[str, Any]],
    chat_template_identity: dict[str, Any],
    serving_oracle: dict[str, Any],
    worker_binary: _PinnedFile,
    source_entries_by_role: dict[str, dict[str, Any]],
    source_sets: dict[str, str],
) -> dict[str, Any]:
    artifact = promotion_identity["artifact"]
    package = promotion_identity["package"]
    return {
        "schema_version": MODEL_IDENTITY_SCHEMA,
        "record_type": "model_identity",
        "model": {
            "upstream_id": UPSTREAM_MODEL_ID,
            "served_id": SERVED_MODEL_ID,
            "revision": MODEL_REVISION,
        },
        "promotion_validation": {
            "schema_version": PROMOTION_SCHEMA,
            "result_sha256": _sha256(receipt_raw),
            "validator_source_sha256": source_entries_by_role[
                "product_promotion_validator"
            ]["sha256"],
            "canonical_source_sha256": source_entries_by_role[
                "product_promotion_canonical"
            ]["sha256"],
            "full_payloads": True,
            "read_only": True,
            "verified": True,
        },
        "product": {
            "root": os.fspath(Path(os.path.abspath(inputs.product_root))),
            "promotion": {
                "file": "promotion.json",
                "bytes": promotion_file.bytes,
                "sha256": promotion_file.sha256,
                "created_at": promotion_identity["created_at"],
                "plan_commit": promotion_identity["plan_commit"],
            },
            "artifact": {
                "schema_version": ARTIFACT_SCHEMA,
                "manifest_file": "artifact/sq_manifest.json",
                "manifest_bytes": artifact_manifest.bytes,
                "manifest_sha256": artifact_manifest.sha256,
                "content_sha256": artifact["content_sha256"],
                "selected_pair_count": artifact["selected_pair_count"],
                "payload_bytes": artifact["payload_bytes"],
                "file_count": artifact["file_count"],
                "payloads_hashed": True,
            },
            "package": {
                "schema_version": PACKAGE_SCHEMA,
                "manifest_file": "package/manifest.json",
                "manifest_bytes": package_manifest.bytes,
                "manifest_sha256": package_manifest.sha256,
                "payload_count": package["payload_count"],
                "payload_bytes": package["payload_bytes"],
                "file_count": package["file_count"],
                "payloads_hashed": True,
            },
        },
        "tokenizer": {
            "root": os.fspath(Path(os.path.abspath(inputs.tokenizer_root))),
            "revision": MODEL_REVISION,
            "aggregate_sha256": _sha256(_canonical(tokenizer_entries)),
            "chat_template": chat_template_identity,
            "files": tokenizer_entries,
        },
        "oracle": serving_oracle,
        "worker": {
            "binary": os.fspath(Path(os.path.abspath(inputs.worker_binary))),
            "binary_bytes": worker_binary.bytes,
            "binary_sha256": worker_binary.sha256,
            "source_sha256": source_sets["worker"],
            "protocol_schema": WORKER_PROTOCOL_SCHEMA,
            "device_architecture": DEVICE_ARCHITECTURE,
            "execution_profile": EXECUTION_PROFILE,
            "context_length": CONTEXT_LENGTH,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "vocab_size": VOCAB_SIZE,
            "model_revision": MODEL_REVISION,
            "artifact_content_sha256": artifact["content_sha256"],
            "package_manifest_sha256": package_manifest.sha256,
        },
    }


def build_identity_artifacts(
    inputs: IdentityBuildInputs, live: LiveIdentity
) -> IdentityArtifacts:
    if not isinstance(inputs, IdentityBuildInputs) or not isinstance(
        live, LiveIdentity
    ):
        fail("identity build input type differs")
    if GIT_COMMIT_RE.fullmatch(inputs.git_commit) is None:
        fail("Git commit identity differs")
    _validate_timestamp(inputs.captured_utc, "identity captured_utc")
    if type(inputs.git_status_raw) is not bytes or len(inputs.git_status_raw) > 4 << 20:
        fail("Git status identity exceeds its bound")
    scanner = _SecretScanner(inputs.forbidden_values)
    scanner.consume(inputs.git_status_raw)
    specs = _validate_source_specs(inputs)
    pins = _PinSet()
    try:
        source_entries_by_role: dict[str, dict[str, Any]] = {}
        retained_sources: dict[str, _PinnedFile] = {}
        for spec in specs:
            retain = spec.role in {
                "serving_fixture_manifest",
                "chat_template_fixture_manifest",
                "runtime_oracle_validation",
            }
            pinned = pins.open(
                spec.path,
                maximum=spec.maximum_bytes,
                forbidden_values=inputs.forbidden_values,
                retain=retain,
                expected_sha256=spec.expected_sha256,
            )
            if retain:
                retained_sources[spec.role] = pinned
            source_entries_by_role[spec.role] = {
                "role": spec.role,
                "path": spec.logical_path,
                "bytes": pinned.bytes,
                "sha256": pinned.sha256,
            }
        source_entries = sorted(
            source_entries_by_role.values(),
            key=lambda item: item["path"].encode("utf-8"),
        )
        source_sets = {
            group: _source_aggregate(source_entries_by_role, roles)
            for group, roles in SOURCE_GROUPS.items()
        }
        unit_file = pins.open(
            inputs.effective_service_unit,
            maximum=1 << 20,
            forbidden_values=inputs.forbidden_values,
            retain=True,
        )
        environment_file = pins.open(
            inputs.effective_environment_file,
            maximum=1 << 20,
            forbidden_values=inputs.forbidden_values,
            retain=True,
        )
        if (
            unit_file.sha256 != source_entries_by_role["systemd_service"]["sha256"]
            or environment_file.sha256
            != source_entries_by_role["systemd_environment_contract"]["sha256"]
        ):
            fail("effective systemd deployment differs from tracked configuration")
        assert environment_file.raw is not None
        configuration = _runtime_configuration(
            _parse_runtime_environment(environment_file.raw), inputs
        )

        product_root = Path(os.path.abspath(inputs.product_root))
        promotion_file = pins.open(
            product_root / "promotion.json",
            maximum=MAX_DOCUMENT_BYTES,
            forbidden_values=inputs.forbidden_values,
            retain=True,
        )
        artifact_manifest = pins.open(
            product_root / "artifact" / "sq_manifest.json",
            maximum=MAX_DOCUMENT_BYTES,
            forbidden_values=inputs.forbidden_values,
            retain=True,
        )
        package_manifest = pins.open(
            product_root / "package" / "manifest.json",
            maximum=MAX_DOCUMENT_BYTES,
            forbidden_values=inputs.forbidden_values,
            retain=True,
        )
        receipt_raw = _canonical(inputs.promotion_validation_result)
        receipt_scanner = _SecretScanner(inputs.forbidden_values)
        receipt_scanner.consume(receipt_raw)
        receipt = _json_object(receipt_raw, "promotion validator receipt")
        assert promotion_file.raw is not None
        promotion_identity = _promotion_identity(
            receipt,
            promotion_file.raw,
            artifact_manifest,
            package_manifest,
            product_root,
        )

        tokenizer_entries: list[dict[str, Any]] = []
        tokenizer_config: _PinnedFile | None = None
        for name in TOKENIZER_FILES:
            pinned = pins.open(
                Path(os.path.abspath(inputs.tokenizer_root)) / name,
                maximum=MAX_TOKENIZER_FILE_BYTES,
                forbidden_values=inputs.forbidden_values,
                retain=name == "tokenizer_config.json",
            )
            if name == "tokenizer_config.json":
                tokenizer_config = pinned
            tokenizer_entries.append(
                {"path": name, "bytes": pinned.bytes, "sha256": pinned.sha256}
            )
        assert tokenizer_config is not None and tokenizer_config.raw is not None
        tokenizer_config_value = _json_object(
            tokenizer_config.raw,
            "tokenizer configuration",
            maximum=MAX_TOKENIZER_FILE_BYTES,
        )
        chat_template = _safe_text(
            tokenizer_config_value.get("chat_template"),
            "tokenizer chat template",
            maximum=1 << 20,
        )
        chat_template_raw = chat_template.encode("utf-8", errors="strict")
        chat_template_identity = {
            "utf8_bytes": len(chat_template_raw),
            "sha256": _sha256(chat_template_raw),
        }
        serving_oracle = _serving_oracle_identity(
            retained_sources["serving_fixture_manifest"],
            retained_sources["chat_template_fixture_manifest"],
            retained_sources["runtime_oracle_validation"],
            chat_template_identity,
        )
        worker_binary = pins.open(
            inputs.worker_binary,
            maximum=MAX_WORKER_BINARY_BYTES,
            forbidden_values=inputs.forbidden_values,
            retain=False,
            require_single_link=False,
        )
        if (
            worker_binary.sha256 != live.worker.executable_sha256
            or worker_binary.bytes != live.worker.executable_bytes
            or Path(os.path.abspath(inputs.worker_binary))
            != Path(os.path.abspath(live.worker.executable))
            or Path(os.path.abspath(inputs.effective_service_unit))
            != Path(os.path.abspath(live.service_fragment_path))
            or live.service_unit != SERVICE_UNIT
        ):
            fail("live service executable or deployment identity differs")
        if live.patch_sha256 != source_entries_by_role["openwebui_patch"]["sha256"]:
            fail("live OpenWebUI patch label differs from the tracked patch")

        environment = _environment_document(
            inputs,
            live,
            configuration,
            source_entries,
            source_sets,
            unit_file,
            environment_file,
        )
        model_identity = _model_document(
            inputs,
            receipt_raw,
            promotion_file,
            promotion_identity,
            artifact_manifest,
            package_manifest,
            tokenizer_entries,
            chat_template_identity,
            serving_oracle,
            worker_binary,
            source_entries_by_role,
            source_sets,
        )
        validate_environment_document(environment)
        validate_model_identity_document(model_identity)
        if (
            environment["source_sets"]["worker"]
            != model_identity["worker"]["source_sha256"]
            or environment["service"]["worker"]["executable_sha256"]
            != model_identity["worker"]["binary_sha256"]
        ):
            fail("environment and model worker identities differ")
        environment_sources = {item["role"]: item for item in environment["sources"]}
        for role in (
            "serving_fixture_manifest",
            "chat_template_fixture_manifest",
            "runtime_oracle_validation",
        ):
            if (
                environment_sources[role]["sha256"]
                != model_identity["oracle"][role]["sha256"]
            ):
                fail("environment and model oracle source identities differ")
        environment_raw = _canonical(environment)
        model_raw = _canonical(model_identity)
        for raw in (environment_raw, model_raw):
            output_scanner = _SecretScanner(inputs.forbidden_values)
            output_scanner.consume(raw)
        pins.seal()
        return IdentityArtifacts(
            environment, model_identity, environment_raw, model_raw
        )
    except IdentityError:
        raise
    except Exception as error:
        raise IdentityError("full campaign identity build failed") from error
    finally:
        pins.close()


def _validate_file_identity(
    value: Any, label: str, *, expected_path: str | None = None
) -> dict[str, Any]:
    item = _exact(value, {"path", "bytes", "sha256"}, label)
    path = _safe_text(item["path"], f"{label}.path")
    if expected_path is not None and path != expected_path:
        fail(f"{label}.path differs")
    _integer(item["bytes"], f"{label}.bytes", minimum=1)
    _sha(item["sha256"], f"{label}.sha256")
    return item


def _validate_process_document(value: Any, label: str) -> dict[str, Any]:
    process_value = _exact(
        value,
        {
            "pid",
            "ppid",
            "uid",
            "gid",
            "starttime_ticks",
            "executable",
            "executable_bytes",
            "executable_sha256",
            "children",
        },
        label,
    )
    _integer(process_value["pid"], f"{label}.pid", minimum=1)
    _integer(process_value["ppid"], f"{label}.ppid")
    _integer(process_value["uid"], f"{label}.uid")
    _integer(process_value["gid"], f"{label}.gid")
    _integer(process_value["starttime_ticks"], f"{label}.starttime_ticks", minimum=1)
    executable = _safe_text(process_value["executable"], f"{label}.executable")
    if not Path(executable).is_absolute():
        fail(f"{label}.executable is not absolute")
    _integer(process_value["executable_bytes"], f"{label}.executable_bytes", minimum=1)
    _sha(process_value["executable_sha256"], f"{label}.executable_sha256")
    children = process_value["children"]
    if type(children) is not list:
        fail(f"{label}.children is not an array")
    parsed_children = [
        _integer(child, f"{label}.children[{index}]", minimum=1)
        for index, child in enumerate(children)
    ]
    if parsed_children != sorted(set(parsed_children)):
        fail(f"{label}.children is not ascending and unique")
    return process_value


def validate_environment_document(value: Any) -> dict[str, Any]:
    _validate_source_contract()
    document = _exact(
        value,
        {
            "schema_version",
            "record_type",
            "captured_utc",
            "git",
            "sources",
            "source_sets",
            "deployment",
            "host",
            "service",
            "openwebui",
        },
        "environment document",
    )
    _reject_key_recursive(document, "passed")
    if (
        document["schema_version"] != ENVIRONMENT_SCHEMA
        or document["record_type"] != "environment"
    ):
        fail("environment schema or record type differs")
    _validate_timestamp(document["captured_utc"], "environment captured_utc")
    git = _exact(
        document["git"], {"commit", "dirty", "status_sha256"}, "environment git"
    )
    if type(git["commit"]) is not str or GIT_COMMIT_RE.fullmatch(git["commit"]) is None:
        fail("environment Git commit differs")
    if type(git["dirty"]) is not bool:
        fail("environment Git dirty flag is not boolean")
    _sha(git["status_sha256"], "environment Git status SHA-256")

    sources = document["sources"]
    if type(sources) is not list or len(sources) != len(SOURCE_ROLE_PATHS):
        fail("environment source list count differs")
    by_role: dict[str, dict[str, Any]] = {}
    paths: list[str] = []
    for index, raw in enumerate(sources):
        entry = _exact(
            raw,
            {"role", "path", "bytes", "sha256"},
            f"environment source {index}",
        )
        role = _safe_text(entry["role"], f"environment source {index}.role")
        path = _safe_text(entry["path"], f"environment source {index}.path")
        if (
            role not in SOURCE_ROLE_PATHS
            or role in by_role
            or path != SOURCE_ROLE_PATHS[role]
        ):
            fail("environment source role or path differs")
        _integer(entry["bytes"], f"environment source {index}.bytes", minimum=1)
        _sha(entry["sha256"], f"environment source {index}.sha256")
        by_role[role] = entry
        paths.append(path)
    if set(by_role) != set(SOURCE_ROLE_PATHS) or paths != sorted(
        paths, key=lambda item: item.encode("utf-8")
    ):
        fail("environment sources are not the exact bytewise-sorted set")
    for role, expected in TTFT_FIXTURE_IDENTITIES.items():
        source = by_role[role]
        if any(source[key] != expected[key] for key in ("path", "bytes", "sha256")):
            fail("environment TTFT fixture source differs")
    source_sets = _exact(
        document["source_sets"], set(SOURCE_GROUPS), "environment source sets"
    )
    for group, roles in SOURCE_GROUPS.items():
        digest = _sha(source_sets[group], f"environment source set {group}")
        if digest != _source_aggregate(by_role, roles):
            fail("environment source set aggregate differs")

    deployment = _exact(
        document["deployment"],
        {"service_unit_file", "environment_file", "configuration"},
        "environment deployment",
    )
    unit_file = _validate_file_identity(
        deployment["service_unit_file"], "effective systemd service"
    )
    environment_file = _validate_file_identity(
        deployment["environment_file"], "effective systemd environment"
    )
    for file_value, role in (
        (unit_file, "systemd_service"),
        (environment_file, "systemd_environment_contract"),
    ):
        if (
            not Path(file_value["path"]).is_absolute()
            or file_value["sha256"] != by_role[role]["sha256"]
        ):
            fail("effective deployment file differs from tracked source")
    configuration = _exact(
        deployment["configuration"],
        {
            "worker_binary",
            "product_root",
            "tokenizer_root",
            "api_key_file",
            "gpu_lock_file",
            "bind_host",
            "bind_port",
            "hip_visible_devices",
            "hip_guards",
        },
        "environment runtime configuration",
    )
    for key in (
        "worker_binary",
        "product_root",
        "tokenizer_root",
        "api_key_file",
        "gpu_lock_file",
    ):
        path = _safe_text(configuration[key], f"runtime configuration {key}")
        if not Path(path).is_absolute():
            fail("runtime configuration path is not absolute")
    if (
        configuration["bind_host"] != DOCKER_NETWORK_GATEWAY
        or configuration["bind_port"] != 8000
        or configuration["hip_visible_devices"] != "1"
        or not _same_json(configuration["hip_guards"], list(HIP_GUARDS))
    ):
        fail("runtime network or HIP configuration differs")

    host = _exact(
        document["host"],
        {"os", "kernel", "boot_id", "cgroup_fs_type", "tools", "gpu"},
        "environment host",
    )
    os_value = _exact(host["os"], {"id", "version_id", "pretty_name"}, "environment OS")
    for key in os_value:
        _safe_text(os_value[key], f"environment OS {key}")
    kernel = _exact(
        host["kernel"],
        {"sysname", "release", "version", "machine"},
        "environment kernel",
    )
    for key in kernel:
        _safe_text(kernel[key], f"environment kernel {key}")
    if (
        type(host["boot_id"]) is not str
        or BOOT_ID_RE.fullmatch(host["boot_id"]) is None
    ):
        fail("environment boot ID differs")
    _safe_text(host["cgroup_fs_type"], "environment cgroup filesystem")
    tools_value = _exact(
        host["tools"],
        {
            "systemd_major",
            "systemd_version_line",
            "python_version_line",
            "rustc_version_line",
            "cargo_version_line",
            "docker_version",
            "docker_api_version",
            "docker_os",
            "docker_arch",
            "docker_kernel_version",
            "amd_smi_tool",
            "amd_smi_library",
            "rocm_version",
            "amd_smi_version_line",
        },
        "environment tools",
    )
    _integer(tools_value["systemd_major"], "environment systemd major", minimum=1)
    for key, item in tools_value.items():
        if key != "systemd_major":
            _safe_text(item, f"environment tools {key}")
    if tools_value["docker_kernel_version"] != kernel["release"]:
        fail("Docker kernel differs from host kernel")
    gpu = _exact(
        host["gpu"],
        {
            "index",
            "bdf",
            "uuid",
            "kfd_gpu_id",
            "node_id",
            "partition_id",
            "architecture",
        },
        "environment GPU",
    )
    _integer(gpu["index"], "environment GPU index")
    if type(gpu["bdf"]) is not str or BDF_RE.fullmatch(gpu["bdf"]) is None:
        fail("environment GPU BDF differs")
    if type(gpu["uuid"]) is not str or UUID_RE.fullmatch(gpu["uuid"]) is None:
        fail("environment GPU UUID differs")
    _integer(gpu["kfd_gpu_id"], "environment KFD GPU ID", minimum=1)
    _integer(gpu["node_id"], "environment GPU node ID")
    _integer(gpu["partition_id"], "environment GPU partition ID")
    if gpu["architecture"] != DEVICE_ARCHITECTURE:
        fail("environment GPU architecture differs")

    service = _exact(
        document["service"],
        {
            "unit",
            "user",
            "group",
            "uid",
            "gid",
            "fragment_path",
            "control_group",
            "gateway",
            "worker",
            "n_restarts",
            "active_state",
            "sub_state",
        },
        "environment service",
    )
    if (
        service["unit"] != SERVICE_UNIT
        or service["active_state"] != "active"
        or service["sub_state"] != "running"
        or service["fragment_path"] != unit_file["path"]
        or service["control_group"] != f"/system.slice/{SERVICE_UNIT}"
    ):
        fail("environment service state or path differs")
    _safe_text(service["user"], "environment service user")
    _safe_text(service["group"], "environment service group")
    uid = _integer(service["uid"], "environment service UID")
    gid = _integer(service["gid"], "environment service GID")
    _integer(service["n_restarts"], "environment service restart count")
    gateway = _validate_process_document(service["gateway"], "environment gateway")
    worker = _validate_process_document(service["worker"], "environment worker")
    if (
        gateway["ppid"] != 1
        or gateway["uid"] != uid
        or gateway["gid"] != gid
        or gateway["children"] != [worker["pid"]]
        or worker["ppid"] != gateway["pid"]
        or worker["uid"] != uid
        or worker["gid"] != gid
        or worker["children"] != []
        or Path(worker["executable"]).name != "ullm-sq8-worker"
        or worker["executable"] != configuration["worker_binary"]
    ):
        fail("environment gateway/worker process relationship differs")

    openwebui = _exact(
        document["openwebui"],
        {
            "version",
            "source_revision",
            "base_image_digest",
            "base_image_id",
            "derived_image_id",
            "Dockerfile_sha256",
            "patch_sha256",
            "patched_middleware_sha256",
            "network_name",
            "network_id",
            "network_subnet",
            "network_gateway",
        },
        "environment OpenWebUI",
    )
    _safe_text(openwebui["version"], "OpenWebUI version")
    _safe_text(openwebui["source_revision"], "OpenWebUI source revision")
    for key in ("base_image_id", "derived_image_id"):
        if (
            type(openwebui[key]) is not str
            or IMAGE_ID_RE.fullmatch(openwebui[key]) is None
        ):
            fail("OpenWebUI image content identity differs")
    if (
        type(openwebui["base_image_digest"]) is not str
        or not openwebui["base_image_digest"].startswith("sha256:")
        or SHA256_RE.fullmatch(openwebui["base_image_digest"][7:]) is None
    ):
        fail("OpenWebUI base image digest differs")
    for key in ("Dockerfile_sha256", "patch_sha256", "patched_middleware_sha256"):
        _sha(openwebui[key], f"OpenWebUI {key}")
    if (
        openwebui["Dockerfile_sha256"] != by_role["openwebui_dockerfile"]["sha256"]
        or openwebui["patch_sha256"] != by_role["openwebui_patch"]["sha256"]
        or openwebui["network_name"] != DOCKER_NETWORK_NAME
        or type(openwebui["network_id"]) is not str
        or NETWORK_ID_RE.fullmatch(openwebui["network_id"]) is None
        or openwebui["network_subnet"] != DOCKER_NETWORK_SUBNET
        or openwebui["network_gateway"] != DOCKER_NETWORK_GATEWAY
    ):
        fail("OpenWebUI source or network identity differs")
    return document


def validate_model_identity_document(value: Any) -> dict[str, Any]:
    document = _exact(
        value,
        {
            "schema_version",
            "record_type",
            "model",
            "promotion_validation",
            "product",
            "tokenizer",
            "oracle",
            "worker",
        },
        "model identity document",
    )
    _reject_key_recursive(document, "passed")
    if (
        document["schema_version"] != MODEL_IDENTITY_SCHEMA
        or document["record_type"] != "model_identity"
    ):
        fail("model identity schema or record type differs")
    if document["model"] != {
        "upstream_id": UPSTREAM_MODEL_ID,
        "served_id": SERVED_MODEL_ID,
        "revision": MODEL_REVISION,
    }:
        fail("model identity differs")
    receipt = _exact(
        document["promotion_validation"],
        {
            "schema_version",
            "result_sha256",
            "validator_source_sha256",
            "canonical_source_sha256",
            "full_payloads",
            "read_only",
            "verified",
        },
        "model promotion validation",
    )
    if (
        receipt["schema_version"] != PROMOTION_SCHEMA
        or receipt["full_payloads"] is not True
        or receipt["read_only"] is not True
        or receipt["verified"] is not True
    ):
        fail("model promotion validation state differs")
    _sha(receipt["result_sha256"], "promotion result SHA-256")
    _sha(receipt["validator_source_sha256"], "promotion validator source SHA-256")
    _sha(receipt["canonical_source_sha256"], "promotion canonical source SHA-256")

    product = _exact(
        document["product"],
        {"root", "promotion", "artifact", "package"},
        "model product",
    )
    root = _safe_text(product["root"], "model product root")
    if not Path(root).is_absolute():
        fail("model product root is not absolute")
    promotion = _exact(
        product["promotion"],
        {"file", "bytes", "sha256", "created_at", "plan_commit"},
        "model promotion",
    )
    if promotion["file"] != "promotion.json":
        fail("model promotion file differs")
    _integer(promotion["bytes"], "model promotion bytes", minimum=1)
    _sha(promotion["sha256"], "model promotion SHA-256")
    _safe_text(promotion["created_at"], "model promotion created_at")
    _safe_text(promotion["plan_commit"], "model promotion plan commit", maximum=40)
    artifact = _exact(
        product["artifact"],
        {
            "schema_version",
            "manifest_file",
            "manifest_bytes",
            "manifest_sha256",
            "content_sha256",
            "selected_pair_count",
            "payload_bytes",
            "file_count",
            "payloads_hashed",
        },
        "model artifact",
    )
    if (
        artifact["schema_version"] != ARTIFACT_SCHEMA
        or artifact["manifest_file"] != "artifact/sq_manifest.json"
        or artifact["payloads_hashed"] is not True
    ):
        fail("model artifact schema, path, or validation state differs")
    for key in ("manifest_bytes", "selected_pair_count", "payload_bytes", "file_count"):
        _integer(artifact[key], f"model artifact {key}", minimum=1)
    _sha(artifact["manifest_sha256"], "model artifact manifest SHA-256")
    _sha(artifact["content_sha256"], "model artifact content SHA-256")
    package = _exact(
        product["package"],
        {
            "schema_version",
            "manifest_file",
            "manifest_bytes",
            "manifest_sha256",
            "payload_count",
            "payload_bytes",
            "file_count",
            "payloads_hashed",
        },
        "model package",
    )
    if (
        package["schema_version"] != PACKAGE_SCHEMA
        or package["manifest_file"] != "package/manifest.json"
        or package["payloads_hashed"] is not True
    ):
        fail("model package schema, path, or validation state differs")
    for key in ("manifest_bytes", "payload_count", "payload_bytes", "file_count"):
        _integer(package[key], f"model package {key}", minimum=1)
    _sha(package["manifest_sha256"], "model package manifest SHA-256")

    tokenizer = _exact(
        document["tokenizer"],
        {"root", "revision", "aggregate_sha256", "chat_template", "files"},
        "model tokenizer",
    )
    tokenizer_root = _safe_text(tokenizer["root"], "model tokenizer root")
    if (
        not Path(tokenizer_root).is_absolute()
        or tokenizer["revision"] != MODEL_REVISION
    ):
        fail("model tokenizer root or revision differs")
    files = tokenizer["files"]
    if type(files) is not list or len(files) != len(TOKENIZER_FILES):
        fail("model tokenizer file set differs")
    tokenizer_entries = [
        _validate_file_identity(
            raw, f"model tokenizer file {index}", expected_path=name
        )
        for index, (raw, name) in enumerate(zip(files, TOKENIZER_FILES, strict=True))
    ]
    aggregate = _sha(tokenizer["aggregate_sha256"], "model tokenizer aggregate SHA-256")
    if aggregate != _sha256(_canonical(tokenizer_entries)):
        fail("model tokenizer aggregate differs")
    chat_template = _exact(
        tokenizer["chat_template"],
        {"utf8_bytes", "sha256"},
        "model tokenizer chat template",
    )
    _integer(
        chat_template["utf8_bytes"],
        "model tokenizer chat template UTF-8 bytes",
        minimum=1,
    )
    _sha(chat_template["sha256"], "model tokenizer chat template SHA-256")

    oracle = _exact(
        document["oracle"],
        {
            "serving_fixture_manifest",
            "chat_template_fixture_manifest",
            "runtime_oracle_validation",
            "vllm_identity",
        },
        "model serving oracle",
    )
    for role in (
        "serving_fixture_manifest",
        "chat_template_fixture_manifest",
        "runtime_oracle_validation",
    ):
        _validate_file_identity(
            oracle[role],
            f"model oracle {role}",
            expected_path=SOURCE_ROLE_PATHS[role],
        )
    vllm = _exact(
        oracle["vllm_identity"],
        {
            "async_scheduling",
            "backend",
            "device",
            "dtype",
            "enable_prefix_caching",
            "enforce_eager",
            "max_num_seqs",
            "package_version",
            "pipeline_parallel_size",
            "python_version",
            "rocr_visible_devices",
            "runner",
            "source_revision_from_package_version",
            "tensor_parallel_size",
            "torch_git_version",
            "torch_hip_version",
            "torch_version",
            "transformers_version",
        },
        "model vLLM oracle identity",
    )
    vllm_device = _exact(
        vllm["device"],
        {
            "compute_capability",
            "gfx",
            "name",
            "total_memory_bytes",
            "visible_device_index",
        },
        "model vLLM oracle device",
    )
    if (
        vllm["backend"] != "vLLM"
        or vllm["async_scheduling"] is not False
        or vllm["enable_prefix_caching"] is not False
        or vllm["enforce_eager"] is not True
        or vllm["max_num_seqs"] != 1
        or vllm["pipeline_parallel_size"] != 1
        or vllm["tensor_parallel_size"] != 1
        or vllm["rocr_visible_devices"] != "1"
        or vllm["runner"] != "LLM.generate"
        or vllm_device["gfx"] != DEVICE_ARCHITECTURE
        or vllm_device["visible_device_index"] != 0
    ):
        fail("model vLLM oracle execution identity differs")
    for key in (
        "dtype",
        "package_version",
        "python_version",
        "source_revision_from_package_version",
        "torch_git_version",
        "torch_hip_version",
        "torch_version",
        "transformers_version",
    ):
        _safe_text(vllm[key], f"model vLLM oracle {key}")
    capability = vllm_device["compute_capability"]
    if (
        type(capability) is not list
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        fail("model vLLM oracle compute capability differs")
    _safe_text(vllm_device["name"], "model vLLM oracle device name")
    _integer(
        vllm_device["total_memory_bytes"],
        "model vLLM oracle device memory",
        minimum=1,
    )

    worker = _exact(
        document["worker"],
        {
            "binary",
            "binary_bytes",
            "binary_sha256",
            "source_sha256",
            "protocol_schema",
            "device_architecture",
            "execution_profile",
            "context_length",
            "max_completion_tokens",
            "vocab_size",
            "model_revision",
            "artifact_content_sha256",
            "package_manifest_sha256",
        },
        "model worker",
    )
    binary = _safe_text(worker["binary"], "model worker binary")
    if not Path(binary).is_absolute() or Path(binary).name != "ullm-sq8-worker":
        fail("model worker binary path differs")
    _integer(worker["binary_bytes"], "model worker binary bytes", minimum=1)
    _sha(worker["binary_sha256"], "model worker binary SHA-256")
    _sha(worker["source_sha256"], "model worker source SHA-256")
    expected_worker = {
        "protocol_schema": WORKER_PROTOCOL_SCHEMA,
        "device_architecture": DEVICE_ARCHITECTURE,
        "execution_profile": EXECUTION_PROFILE,
        "context_length": CONTEXT_LENGTH,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "vocab_size": VOCAB_SIZE,
        "model_revision": MODEL_REVISION,
        "artifact_content_sha256": artifact["content_sha256"],
        "package_manifest_sha256": package["manifest_sha256"],
    }
    if any(
        not _same_json(worker[key], expected)
        for key, expected in expected_worker.items()
    ):
        fail("model worker contract differs")
    return document


def serialize_environment_document(value: Any) -> bytes:
    return _canonical(validate_environment_document(value))


def serialize_model_identity_document(value: Any) -> bytes:
    return _canonical(validate_model_identity_document(value))


def write_identity_artifacts(
    directory: Path,
    artifacts: IdentityArtifacts,
    *,
    uid: int | None = None,
    gid: int | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(artifacts, IdentityArtifacts):
        fail("identity artifact result type differs")
    expected = {
        "environment.json": serialize_environment_document(artifacts.environment),
        "model-identity.json": serialize_model_identity_document(
            artifacts.model_identity
        ),
    }
    if (
        artifacts.environment_bytes != expected["environment.json"]
        or artifacts.model_identity_bytes != expected["model-identity.json"]
    ):
        fail("identity artifact serialized bytes differ from their documents")
    directory_fd = -1
    created: list[str] = []
    try:
        directory_fd = os.open(Path(os.path.abspath(directory)), _directory_flags())
        directory_identity = os.fstat(directory_fd)
        if not stat.S_ISDIR(directory_identity.st_mode):
            fail("identity output is not a directory")
        if uid is not None and directory_identity.st_uid != uid:
            fail("identity output directory UID differs")
        if gid is not None and directory_identity.st_gid != gid:
            fail("identity output directory GID differs")
        existing = set(os.listdir(directory_fd))
        if existing.intersection(expected):
            fail("identity output artifact already exists")
        result: dict[str, dict[str, Any]] = {}
        for name, raw in expected.items():
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory_fd,
            )
            created.append(name)
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(
                        descriptor, raw[offset : offset + COPY_CHUNK_BYTES]
                    )
                    if written <= 0:
                        fail("identity artifact write was short")
                    offset += written
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
                expected_uid = directory_identity.st_uid if uid is None else uid
                expected_gid = directory_identity.st_gid if gid is None else gid
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_nlink != 1
                    or metadata.st_uid != expected_uid
                    or metadata.st_gid != expected_gid
                    or metadata.st_size != len(raw)
                ):
                    fail("written identity artifact metadata differs")
            finally:
                os.close(descriptor)
            result[name] = {"bytes": len(raw), "sha256": _sha256(raw)}
        os.fsync(directory_fd)
        return result
    except IdentityError:
        for name in reversed(created):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError:
                pass
        raise
    except OSError:
        for name in reversed(created):
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError:
                pass
        fail("failed to write identity artifacts")
    finally:
        if directory_fd >= 0:
            try:
                os.close(directory_fd)
            except OSError:
                pass


__all__ = [
    "ENVIRONMENT_SCHEMA",
    "MODEL_IDENTITY_SCHEMA",
    "HardwareExpectation",
    "IdentityArtifacts",
    "IdentityBuildInputs",
    "IdentityError",
    "IdentityProbe",
    "LiveCaptureExpectation",
    "LiveIdentity",
    "OpenWebUIExpectation",
    "ProcessSnapshot",
    "RuntimeConfiguration",
    "SourceFileSpec",
    "SystemIdentityProbe",
    "build_identity_artifacts",
    "capture_live_identity",
    "default_source_specs",
    "serialize_environment_document",
    "serialize_model_identity_document",
    "validate_environment_document",
    "validate_model_identity_document",
    "write_identity_artifacts",
]
