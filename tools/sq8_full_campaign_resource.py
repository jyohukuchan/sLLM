#!/usr/bin/env python3
"""Build the pure resource-measurement contract for one full SQ8 campaign."""

from __future__ import annotations

import base64
import copy
import dataclasses
import datetime
import hashlib
import json
import re
from decimal import Decimal
from pathlib import PurePosixPath
from typing import Any, Callable, Generic, NoReturn, Protocol, Sequence, TypeVar, cast

from sq8_full_campaign_identity import (
    IdentityArtifacts,
    serialize_environment_document,
    serialize_model_identity_document,
)


RESOURCE_CONFIG_SCHEMA = "ullm.sq8.full_campaign.resource_config.v1"
RESOURCE_SCHEMA = "ullm.sq8.release_measurement.raw.v1"
TARGET = "/v1/chat/completions"
SERVED_MODEL_ID = "ullm-qwen3-14b-sq8"
SERVICE_UNIT = "ullm-openai.service"
CONFIG_INPUT_PATH = "collector/config.json"
FIXTURE_INPUT_PATH = "collector/resource-chat-fixture.json"
MAX_SEMANTIC_SCAN_NODES = 100_000

RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
UTC_SECONDS_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")

SAMPLED_NORMAL_INDICES = tuple(range(5, 101, 5))
NEGATIVE_SCHEDULE = (
    (25, "context_overflow_1", "one"),
    (50, "malformed_json", None),
    (75, "context_overflow_2", "two"),
)


class ResourceContractError(RuntimeError):
    """A fail-closed resource contract construction error."""


def fail(message: str) -> NoReturn:
    raise ResourceContractError(message)


class NegativeCaseLike(Protocol):
    after_request: int
    name: str
    body: bytes
    expected_status: int


class ResourceSegmentConfigLike(Protocol):
    target: str
    resource_body_template: dict[str, Any]
    negative_cases: tuple[NegativeCaseLike, ...]


class IndependentIdentity(Protocol):
    def validate_session_header(self, record: dict[str, Any]) -> None: ...

    def validate_header_source_inputs(self, input_files: Any) -> None: ...


class IndependentValidatorApi(Protocol):
    def validate_schedule(self, value: Any, label: str) -> dict[str, Any]: ...

    def validate_thresholds(self, value: Any, label: str) -> dict[str, Any]: ...


NegativeCaseT = TypeVar("NegativeCaseT")
ResourceConfigT = TypeVar("ResourceConfigT")


@dataclasses.dataclass(frozen=True)
class ResourceContract(Generic[ResourceConfigT]):
    """All pure inputs written or consumed by the resource campaign phases."""

    segment_config: ResourceConfigT
    config_bytes: bytes
    fixture_bytes: bytes
    session_header_fields: dict[str, Any]
    resource_header: dict[str, Any]


def _canonical_document(value: Any) -> bytes:
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
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("resource contract document is not canonical JSON")


def _canonical_fixture(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        fail("resource fixture is not canonical UTF-8 JSON")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _release_schedule() -> dict[str, Any]:
    return {
        "openwebui_chats": 20,
        "cancel_phases": [
            "after_started_before_progress",
            "prefill_after_128",
            "prefill_after_2048",
            "decode_after_first_content",
            "openwebui_stop_after_visible_content",
        ],
        "normal_warmups": 10,
        "normal_requests": 100,
        "sampled_normal_indices": list(SAMPLED_NORMAL_INDICES),
        "restart_warmups": 10,
        "restart_requests": 20,
        "ttft_fixture_ids": [
            "exact-p0032",
            "exact-p0128",
            "exact-p0512",
            "exact-p2048",
            "exact-p3584",
        ],
        "latency_warmups_per_case": 2,
        "latency_measured_per_case": 10,
        "decode_warmups": 2,
        "decode_measured": 10,
        "idle_settle_ms": 5000,
        "samples_per_point": 5,
        "sample_interval_ms": 1000,
    }


def _resource_schedule() -> dict[str, int]:
    return {
        "normal_warmups": 10,
        "normal_requests": 100,
        "restart_warmups": 10,
        "restart_requests": 20,
        "idle_settle_ms": 5000,
        "samples_per_point": 5,
        "sample_interval_ms": 1000,
    }


def _thresholds() -> dict[str, Any]:
    return {
        "ttft_seconds_maximum": {
            "exact-p0032": {"p50": 2.5, "p95": 3},
            "exact-p0128": {"p50": 4, "p95": 5},
            "exact-p0512": {"p50": 10, "p95": 12},
            "exact-p2048": {"p50": 30, "p95": 35},
            "exact-p3584": {"p50": 50, "p95": 60},
        },
        "decode_p50_tokens_per_second_minimum": 15,
        "decode_p95_inter_content_seconds_maximum": 0.1,
        "cancel_release_max_ns": 5_000_000_000,
        "final_delta_max_bytes": 67_108_864,
        "theil_sen_max_bytes_per_request": 262_144,
    }


def _commands() -> dict[str, str]:
    return {
        "systemd_version": "systemctl --version",
        "service_identity": (
            "systemctl show ullm-openai.service --property=ControlGroup "
            "--property=MainPID --no-pager"
        ),
        "cgroup_type": "stat -fc %T /sys/fs/cgroup",
        "host_memory": "cat /sys/fs/cgroup${ControlGroup}/memory.current",
        "proc_stat": "cat /proc/${PID}/stat",
        "proc_status": "cat /proc/${PID}/status",
        "proc_exe": "readlink /proc/${PID}/exe",
        "proc_fds": "find -P /proc/${PID}/fd -mindepth 1 -maxdepth 1 -printf '%f\\n'",
        "proc_children": "cat /proc/${PID}/task/${PID}/children",
        "amd_smi_version": "amd-smi version",
        "amd_smi_list": "amd-smi list --json",
        "amd_smi_process": "amd-smi process --gpu 2 --general --json",
        "amd_smi_metric": "amd-smi metric --gpu 2 --json",
        "kfd_proc_probe": "test -d /sys/class/kfd/kfd/proc",
        "kfd_processes": (
            "find -P /sys/class/kfd/kfd/proc -mindepth 1 -maxdepth 1 -printf '%f\\n'"
        ),
        "kfd_vram": "cat /sys/class/kfd/kfd/proc/${PID}/vram_51545",
    }


def _fixture() -> dict[str, Any]:
    return {
        "model": SERVED_MODEL_ID,
        "messages": [
            {
                "role": "system",
                "content": "Answer the synthetic resource probe concisely.",
            },
            {"role": "user", "content": "Reply with the word ready."},
        ],
    }


def _positive_request_contract() -> dict[str, Any]:
    return {
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 2,
        "default_sampling": {"temperature": 0, "top_p": 1, "seed": 0},
        "sampled_normal": {
            "request_indices": list(SAMPLED_NORMAL_INDICES),
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": "request_index",
        },
    }


def _overflow_body(marker: str) -> bytes:
    return _canonical_fixture(
        {
            "model": SERVED_MODEL_ID,
            "messages": [
                {
                    "role": "user",
                    "content": marker + (" overflow" * 5000),
                }
            ],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 2,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
        }
    )


def _negative_case_values() -> tuple[tuple[int, str, bytes, int], ...]:
    values: list[tuple[int, str, bytes, int]] = []
    for after_request, name, marker in NEGATIVE_SCHEDULE:
        body = b"{" if marker is None else _overflow_body(marker)
        values.append((after_request, name, body, 400))
    return tuple(values)


def _runtime_identities(artifacts: IdentityArtifacts) -> dict[str, Any]:
    environment = artifacts.environment
    model_identity = artifacts.model_identity
    try:
        openwebui = environment["openwebui"]
        source_sets = environment["source_sets"]
        worker = model_identity["worker"]
        selected_openwebui = {
            key: openwebui[key]
            for key in (
                "version",
                "source_revision",
                "base_image_digest",
                "base_image_id",
                "derived_image_id",
                "Dockerfile_sha256",
                "patch_sha256",
                "patched_middleware_sha256",
            )
        }
        return {
            "environment_file": "environment.json",
            "environment_sha256": _sha256(artifacts.environment_bytes),
            "model_identity_file": "model-identity.json",
            "model_identity_sha256": _sha256(artifacts.model_identity_bytes),
            "openwebui": selected_openwebui,
            "docker_network_id": openwebui["network_id"],
            "gateway_source_sha256": source_sets["gateway"],
            "worker_source_sha256": source_sets["worker"],
            "worker_binary_sha256": worker["binary_sha256"],
        }
    except (KeyError, TypeError):
        fail("identity artifacts lack the resource runtime identities")


def _resource_header(artifacts: IdentityArtifacts) -> dict[str, Any]:
    try:
        environment = artifacts.environment
        host = environment["host"]
        tools = host["tools"]
        gpu = host["gpu"]
        service = environment["service"]
        return {
            "schema_version": RESOURCE_SCHEMA,
            "record_type": "header",
            "service_unit": service["unit"],
            "commands": _commands(),
            "tools": {
                "systemd_major": tools["systemd_major"],
                "systemd_version_line": tools["systemd_version_line"],
                "amd_smi_tool": tools["amd_smi_tool"],
                "amd_smi_library": tools["amd_smi_library"],
                "rocm": tools["rocm_version"],
                "amd_smi_version_output": tools["amd_smi_version_line"],
            },
            "probes": {
                "cgroup_fs_type": host["cgroup_fs_type"],
                "kfd_proc_present": True,
                "gpu_index": gpu["index"],
                "gpu_bdf": gpu["bdf"],
                "gpu_uuid": gpu["uuid"],
                "kfd_gpu_id": gpu["kfd_gpu_id"],
            },
            "schedule": _resource_schedule(),
        }
    except (KeyError, TypeError):
        fail("identity artifacts lack the resource header identities")


def _validate_safe_input(entry: Any, label: str) -> dict[str, Any]:
    if type(entry) is not dict or set(entry) != {"path", "bytes", "sha256"}:
        fail(f"{label} fields differ")
    path = entry["path"]
    if type(path) is not str:
        fail(f"{label}.path is not text")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or "\\" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        fail(f"{label}.path is unsafe")
    if type(entry["bytes"]) is not int or entry["bytes"] < 1:
        fail(f"{label}.bytes differs")
    if type(entry["sha256"]) is not str or SHA256_RE.fullmatch(entry["sha256"]) is None:
        fail(f"{label}.sha256 differs")
    return cast(dict[str, Any], entry)


def _input_files(
    artifacts: IdentityArtifacts, config_bytes: bytes, fixture_bytes: bytes
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        sources = artifacts.environment["sources"]
    except (KeyError, TypeError):
        fail("identity artifacts lack campaign sources")
    if type(sources) is not list:
        fail("identity artifact sources are not an array")
    for index, source in enumerate(sources):
        if type(source) is not dict:
            fail(f"identity source {index} is not an object")
        try:
            entry = {
                "path": source["path"],
                "bytes": source["bytes"],
                "sha256": source["sha256"],
            }
        except KeyError:
            fail(f"identity source {index} fields differ")
        entries.append(_validate_safe_input(entry, f"identity source {index}"))
    entries.extend(
        [
            {
                "path": CONFIG_INPUT_PATH,
                "bytes": len(config_bytes),
                "sha256": _sha256(config_bytes),
            },
            {
                "path": FIXTURE_INPUT_PATH,
                "bytes": len(fixture_bytes),
                "sha256": _sha256(fixture_bytes),
            },
        ]
    )
    entries.sort(key=lambda item: item["path"].encode("utf-8", errors="strict"))
    paths = [entry["path"] for entry in entries]
    if paths != sorted(set(paths), key=lambda item: item.encode("utf-8")):
        fail("resource input paths are not unique")
    return entries


def _config_document(
    fixture: dict[str, Any],
    fixture_bytes: bytes,
    negative_values: Sequence[tuple[int, str, bytes, int]],
) -> dict[str, Any]:
    return {
        "schema_version": RESOURCE_CONFIG_SCHEMA,
        "target": TARGET,
        "fixture": {
            "path": FIXTURE_INPUT_PATH,
            "bytes": len(fixture_bytes),
            "sha256": _sha256(fixture_bytes),
            "value": copy.deepcopy(fixture),
        },
        "positive_request": _positive_request_contract(),
        "negative_cases": [
            {
                "after_request": after_request,
                "name": name,
                "body_base64": base64.b64encode(body).decode("ascii"),
                "body_bytes": len(body),
                "body_sha256": _sha256(body),
                "expected_status": expected_status,
            }
            for after_request, name, body, expected_status in negative_values
        ],
        "schedule": _release_schedule(),
        "resource_schedule": _resource_schedule(),
        "thresholds": _thresholds(),
    }


def _scan_forbidden(value: Any, forbidden_values: Sequence[bytes], label: str) -> None:
    secrets = tuple(forbidden_values)
    if any(type(secret) is not bytes or not secret for secret in secrets):
        fail("forbidden resource values must be non-empty bytes")
    pending: list[Any] = [value]
    visited = 0
    while pending:
        item = pending.pop()
        visited += 1
        if visited > MAX_SEMANTIC_SCAN_NODES:
            fail(f"{label} exceeds the semantic secret-scan bound")
        if type(item) is dict:
            child_count = len(item) * 2
            if visited + len(pending) + child_count > MAX_SEMANTIC_SCAN_NODES:
                fail(f"{label} exceeds the semantic secret-scan bound")
            pending.extend(item.keys())
            pending.extend(item.values())
        elif type(item) in {list, tuple}:
            if visited + len(pending) + len(item) > MAX_SEMANTIC_SCAN_NODES:
                fail(f"{label} exceeds the semantic secret-scan bound")
            pending.extend(item)
        elif type(item) is str:
            raw = item.encode("utf-8", errors="strict")
            if any(secret in raw for secret in secrets):
                fail(f"{label} contains a forbidden value")
        elif type(item) is bytes and any(secret in item for secret in secrets):
            fail(f"{label} contains a forbidden value")


def _validate_fixed_contract(
    contract: ResourceContract[ResourceConfigT], artifacts: IdentityArtifacts
) -> None:
    expected_fixture = _fixture()
    expected_fixture_bytes = _canonical_fixture(expected_fixture)
    negative_values = _negative_case_values()
    expected_config_bytes = _canonical_document(
        _config_document(expected_fixture, expected_fixture_bytes, negative_values)
    )
    if contract.fixture_bytes != expected_fixture_bytes:
        fail("resource fixture bytes drifted")
    if contract.config_bytes != expected_config_bytes:
        fail("resource config bytes drifted")

    config = cast(ResourceSegmentConfigLike, contract.segment_config)
    dataclass_parameters = getattr(type(config), "__dataclass_params__", None)
    if dataclass_parameters is None or dataclass_parameters.frozen is not True:
        fail("resource segment config is not an immutable dataclass")
    if config.target != TARGET or config.resource_body_template != expected_fixture:
        fail("resource segment target, model, or fixture drifted")
    observed_cases = tuple(
        (case.after_request, case.name, case.body, case.expected_status)
        for case in config.negative_cases
    )
    if observed_cases != negative_values:
        fail("resource negative schedule or body drifted")

    expected_inputs = _input_files(
        artifacts, expected_config_bytes, expected_fixture_bytes
    )
    expected_header = {
        "run_id": contract.session_header_fields.get("run_id"),
        "started_utc": contract.session_header_fields.get("started_utc"),
        "clock": "python.time.monotonic_ns",
        "boot_id": artifacts.environment["host"]["boot_id"],
        "identities": _runtime_identities(artifacts),
        "input_files": expected_inputs,
        "schedule": _release_schedule(),
        "thresholds": _thresholds(),
    }
    if contract.session_header_fields != expected_header:
        fail("resource session header drifted")
    if contract.resource_header != _resource_header(artifacts):
        fail("resource runtime header drifted")


def _validate_identity_artifacts(artifacts: IdentityArtifacts) -> None:
    if not isinstance(artifacts, IdentityArtifacts):
        fail("resource identity artifact type differs")
    if artifacts.environment_bytes != serialize_environment_document(
        artifacts.environment
    ) or artifacts.model_identity_bytes != serialize_model_identity_document(
        artifacts.model_identity
    ):
        fail("resource identity artifact bytes differ from their documents")


def validate_resource_contract(
    contract: ResourceContract[ResourceConfigT],
    artifacts: IdentityArtifacts,
    independent_identity: IndependentIdentity,
    independent_validator: IndependentValidatorApi,
    *,
    forbidden_values: Sequence[bytes] = (),
) -> None:
    """Recompute and independently cross-check one pure resource contract."""

    _validate_identity_artifacts(artifacts)
    _validate_fixed_contract(contract, artifacts)

    header = contract.session_header_fields
    run_id = header.get("run_id")
    if type(run_id) is not str or RUN_ID_RE.fullmatch(run_id) is None:
        fail("resource run_id differs")
    started_utc = header.get("started_utc")
    if type(started_utc) is not str or UTC_SECONDS_RE.fullmatch(started_utc) is None:
        fail("resource started_utc differs")
    try:
        parsed = datetime.datetime.strptime(started_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        fail("resource started_utc is not a calendar timestamp")
    if parsed.tzinfo is not None:
        fail("resource started_utc parser state differs")

    independent_validator.validate_schedule(
        header["schedule"], "resource session header.schedule"
    )
    serialized_thresholds = json.loads(
        _canonical_document(header["thresholds"]), parse_float=Decimal
    )
    independent_validator.validate_thresholds(
        serialized_thresholds, "resource session header.thresholds"
    )
    independent_identity.validate_session_header(header)
    independent_identity.validate_header_source_inputs(header["input_files"])

    _scan_forbidden(contract.config_bytes, forbidden_values, "resource config")
    _scan_forbidden(contract.fixture_bytes, forbidden_values, "resource fixture")
    _scan_forbidden(header, forbidden_values, "resource session header")
    _scan_forbidden(
        contract.resource_header, forbidden_values, "resource runtime header"
    )


def build_resource_contract(
    artifacts: IdentityArtifacts,
    independent_identity: IndependentIdentity,
    independent_validator: IndependentValidatorApi,
    *,
    run_id: str,
    started_utc: str,
    negative_case_type: Callable[[int, str, bytes, int], NegativeCaseT],
    resource_config_type: Callable[
        [str, dict[str, Any], tuple[NegativeCaseT, ...]], ResourceConfigT
    ],
    forbidden_values: Sequence[bytes] = (),
) -> ResourceContract[ResourceConfigT]:
    """Build one fixed, serial production resource-measurement contract."""

    _validate_identity_artifacts(artifacts)
    fixture = _fixture()
    fixture_bytes = _canonical_fixture(fixture)
    negative_values = _negative_case_values()
    negative_cases = tuple(
        negative_case_type(after_request, name, body, expected_status)
        for after_request, name, body, expected_status in negative_values
    )
    segment_config = resource_config_type(
        TARGET,
        copy.deepcopy(fixture),
        negative_cases,
    )
    config_bytes = _canonical_document(
        _config_document(fixture, fixture_bytes, negative_values)
    )
    header = {
        "run_id": run_id,
        "started_utc": started_utc,
        "clock": "python.time.monotonic_ns",
        "boot_id": artifacts.environment["host"]["boot_id"],
        "identities": _runtime_identities(artifacts),
        "input_files": _input_files(artifacts, config_bytes, fixture_bytes),
        "schedule": _release_schedule(),
        "thresholds": _thresholds(),
    }
    contract = ResourceContract(
        segment_config=segment_config,
        config_bytes=config_bytes,
        fixture_bytes=fixture_bytes,
        session_header_fields=header,
        resource_header=_resource_header(artifacts),
    )
    validate_resource_contract(
        contract,
        artifacts,
        independent_identity,
        independent_validator,
        forbidden_values=forbidden_values,
    )
    return contract


__all__ = [
    "CONFIG_INPUT_PATH",
    "FIXTURE_INPUT_PATH",
    "ResourceContract",
    "ResourceContractError",
    "build_resource_contract",
    "validate_resource_contract",
]
