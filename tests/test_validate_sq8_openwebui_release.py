import base64
import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate-sq8-openwebui-release.py"
GIT_COMMIT = "a" * 40
WORKER_SHA256 = "b" * 64
BOOT_ID = "11111111111111111111111111111111"
RUN_ID = "synthetic-openwebui-release"


def load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_sq8_openwebui_release", VALIDATOR_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


SCHEDULE = {
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
    "sampled_normal_indices": list(range(5, 101, 5)),
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
THRESHOLDS = {
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


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def identity_canonical(value) -> bytes:
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


def build_identity_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    source_entries = []
    fixed_sources = {
        **VALIDATOR.EXPECTED_ORACLE_FILE_IDENTITIES,
        **VALIDATOR.EXPECTED_TTFT_FIXTURE_IDENTITIES,
    }
    for role, path in VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS.items():
        if role in fixed_sources:
            expected = fixed_sources[role]
            entry = {
                "role": role,
                "path": expected["path"],
                "bytes": expected["bytes"],
                "sha256": expected["sha256"],
            }
        else:
            raw = f"synthetic source {role}\n".encode("ascii")
            entry = {
                "role": role,
                "path": path,
                "bytes": len(raw),
                "sha256": sha256_bytes(raw),
            }
        source_entries.append(entry)
    source_entries.sort(key=lambda item: item["path"].encode("utf-8"))
    by_role = {entry["role"]: entry for entry in source_entries}
    source_sets = {
        group: sha256_bytes(
            identity_canonical([by_role[role] for role in sorted(roles)])
        )
        for group, roles in VALIDATOR.EXPECTED_SOURCE_GROUPS.items()
    }
    worker_binary = "/opt/ullm/bin/ullm-sq8-worker"
    product_root = "/opt/ullm/product/qwen3-14b-sq8"
    tokenizer_root = "/opt/ullm/tokenizer/qwen3-14b-fp8"
    unit_file = {
        "path": "/etc/systemd/system/ullm-openai.service",
        "bytes": by_role["systemd_service"]["bytes"],
        "sha256": by_role["systemd_service"]["sha256"],
    }
    environment_file = {
        "path": "/etc/ullm/openai-gateway.env",
        "bytes": by_role["systemd_environment_contract"]["bytes"],
        "sha256": by_role["systemd_environment_contract"]["sha256"],
    }
    environment = {
        "schema_version": VALIDATOR.ENVIRONMENT_SCHEMA,
        "record_type": "environment",
        "captured_utc": "2026-07-11T12:00:00Z",
        "git": {
            "commit": GIT_COMMIT,
            "dirty": False,
            "status_sha256": sha256_bytes(b""),
        },
        "sources": source_entries,
        "source_sets": source_sets,
        "deployment": {
            "service_unit_file": unit_file,
            "environment_file": environment_file,
            "configuration": {
                "worker_binary": worker_binary,
                "product_root": product_root,
                "tokenizer_root": tokenizer_root,
                "api_key_file": "/etc/ullm/openai-api-key",
                "gpu_lock_file": "/run/ullm/r9700.lock",
                "bind_host": VALIDATOR.DOCKER_NETWORK_GATEWAY,
                "bind_port": 8000,
                "hip_visible_devices": "1",
                "hip_guards": list(VALIDATOR.HIP_GUARDS),
            },
        },
        "host": {
            "os": {
                "id": "ubuntu",
                "version_id": "24.04",
                "pretty_name": "Ubuntu 24.04.4 LTS",
            },
            "kernel": {
                "sysname": "Linux",
                "release": "6.17.0-35-generic",
                "version": "#35-Ubuntu SMP",
                "machine": "x86_64",
            },
            "boot_id": BOOT_ID,
            "cgroup_fs_type": "cgroup2fs",
            "tools": {
                "systemd_major": 255,
                "systemd_version_line": "systemd 255 (255.4-1ubuntu8.12)",
                "python_version_line": "Python 3.12.3",
                "rustc_version_line": "rustc 1.96.0 (synthetic)",
                "cargo_version_line": "cargo 1.96.0 (synthetic)",
                "docker_version": "28.5.1",
                "docker_api_version": "1.51",
                "docker_os": "linux",
                "docker_arch": "amd64",
                "docker_kernel_version": "6.17.0-35-generic",
                "amd_smi_tool": "26.2.2+e1a6bc5663",
                "amd_smi_library": "26.2.2",
                "rocm_version": "7.2.1",
                "amd_smi_version_line": "AMD SMI 26.2.2 ROCm 7.2.1",
            },
            "gpu": {
                "index": 2,
                "bdf": "0000:47:00.0",
                "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
                "kfd_gpu_id": 51545,
                "node_id": 2,
                "partition_id": 0,
                "architecture": VALIDATOR.DEVICE_ARCHITECTURE,
            },
        },
        "service": {
            "unit": VALIDATOR.SERVICE_UNIT,
            "user": "homelab1",
            "group": "homelab1",
            "uid": 1000,
            "gid": 1000,
            "fragment_path": unit_file["path"],
            "control_group": f"/system.slice/{VALIDATOR.SERVICE_UNIT}",
            "gateway": {
                "pid": 1200,
                "ppid": 1,
                "uid": 1000,
                "gid": 1000,
                "starttime_ticks": 10000,
                "executable": "/usr/bin/python3.12",
                "executable_bytes": 64,
                "executable_sha256": "6" * 64,
                "children": [1201],
            },
            "worker": {
                "pid": 1201,
                "ppid": 1200,
                "uid": 1000,
                "gid": 1000,
                "starttime_ticks": 10001,
                "executable": worker_binary,
                "executable_bytes": 123456,
                "executable_sha256": WORKER_SHA256,
                "children": [],
            },
            "n_restarts": 2,
            "active_state": "active",
            "sub_state": "running",
        },
        "openwebui": {
            "version": VALIDATOR.OPENWEBUI_VERSION,
            "source_revision": VALIDATOR.OPENWEBUI_SOURCE_REVISION,
            "base_image_digest": VALIDATOR.OPENWEBUI_BASE_IMAGE_DIGEST,
            "base_image_id": VALIDATOR.OPENWEBUI_BASE_IMAGE_ID,
            "derived_image_id": VALIDATOR.OPENWEBUI_DERIVED_IMAGE_ID,
            "Dockerfile_sha256": by_role["openwebui_dockerfile"]["sha256"],
            "patch_sha256": by_role["openwebui_patch"]["sha256"],
            "patched_middleware_sha256": (
                VALIDATOR.OPENWEBUI_PATCHED_MIDDLEWARE_SHA256
            ),
            "network_name": VALIDATOR.DOCKER_NETWORK_NAME,
            "network_id": VALIDATOR.DOCKER_NETWORK_ID,
            "network_subnet": VALIDATOR.DOCKER_NETWORK_SUBNET,
            "network_gateway": VALIDATOR.DOCKER_NETWORK_GATEWAY,
        },
    }

    tokenizer_files = [
        {"path": path, "bytes": byte_count, "sha256": digest}
        for path, byte_count, digest in VALIDATOR.EXPECTED_TOKENIZER_FILES
    ]
    artifact = deepcopy(VALIDATOR.EXPECTED_ARTIFACT_IDENTITY)
    package = deepcopy(VALIDATOR.EXPECTED_PACKAGE_IDENTITY)
    promotion = {
        "file": "promotion.json",
        "bytes": 1347,
        "sha256": "7" * 64,
        "created_at": "2026-07-10T12:16:25+09:00",
        "plan_commit": VALIDATOR.PROMOTION_PLAN_COMMIT,
    }
    receipt = {
        "schema_version": VALIDATOR.PROMOTION_SCHEMA,
        "product_root": product_root,
        "created_at": promotion["created_at"],
        "model_revision": VALIDATOR.MODEL_REVISION,
        "artifact": {
            "manifest_sha256": artifact["manifest_sha256"],
            "content_sha256": artifact["content_sha256"],
            "selected_pair_count": artifact["selected_pair_count"],
            "payloads_hashed": True,
        },
        "package": {
            "manifest_sha256": package["manifest_sha256"],
            "payload_count": package["payload_count"],
            "payload_bytes": package["payload_bytes"],
            "payloads_hashed": True,
        },
        "read_only": True,
        "full_payloads": True,
        "verified": True,
    }
    model_identity = {
        "schema_version": VALIDATOR.MODEL_IDENTITY_SCHEMA,
        "record_type": "model_identity",
        "model": {
            "upstream_id": VALIDATOR.UPSTREAM_MODEL_ID,
            "served_id": VALIDATOR.SERVED_MODEL_ID,
            "revision": VALIDATOR.MODEL_REVISION,
        },
        "promotion_validation": {
            "schema_version": VALIDATOR.PROMOTION_SCHEMA,
            "result_sha256": sha256_bytes(identity_canonical(receipt)),
            "validator_source_sha256": by_role["product_promotion_validator"]["sha256"],
            "full_payloads": True,
            "read_only": True,
            "verified": True,
        },
        "product": {
            "root": product_root,
            "promotion": promotion,
            "artifact": artifact,
            "package": package,
        },
        "tokenizer": {
            "root": tokenizer_root,
            "revision": VALIDATOR.MODEL_REVISION,
            "aggregate_sha256": sha256_bytes(identity_canonical(tokenizer_files)),
            "chat_template": deepcopy(VALIDATOR.EXPECTED_CHAT_TEMPLATE_IDENTITY),
            "files": tokenizer_files,
        },
        "oracle": {
            **deepcopy(VALIDATOR.EXPECTED_ORACLE_FILE_IDENTITIES),
            "vllm_identity": deepcopy(VALIDATOR.EXPECTED_VLLM_IDENTITY),
        },
        "worker": {
            "binary": worker_binary,
            "binary_bytes": environment["service"]["worker"]["executable_bytes"],
            "binary_sha256": WORKER_SHA256,
            "source_sha256": source_sets["worker"],
            "protocol_schema": VALIDATOR.WORKER_PROTOCOL_SCHEMA,
            "device_architecture": VALIDATOR.DEVICE_ARCHITECTURE,
            "execution_profile": VALIDATOR.EXECUTION_PROFILE,
            "context_length": VALIDATOR.CONTEXT_LENGTH,
            "max_completion_tokens": VALIDATOR.MAX_COMPLETION_TOKENS,
            "vocab_size": VALIDATOR.VOCAB_SIZE,
            "model_revision": VALIDATOR.MODEL_REVISION,
            "artifact_content_sha256": artifact["content_sha256"],
            "package_manifest_sha256": package["manifest_sha256"],
        },
    }
    return environment, model_identity


class EvidenceBuilder:
    def __init__(self, root: Path):
        self.root = root
        self.session_records = []
        self.resource_records = []
        self.journal_records = []
        self.now = 1_000_000_000
        self.cursor_index = 0
        self.fixture = {
            "model": "Qwen3-14B-SQ8",
            "messages": [{"role": "user", "content": "resource fixture"}],
        }

    def write_json(self, relative: str, value) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(compact_json(value), encoding="utf-8")

    def session_add(self, record_type: str, phase: str, case_id, **fields) -> None:
        self.session_records.append(
            {
                "schema_version": VALIDATOR.SESSION_SCHEMA,
                "record_type": record_type,
                "sequence": len(self.session_records),
                "phase": phase,
                "case_id": case_id,
                **fields,
            }
        )

    def gateway_event(self, phase: str, case_id: str, event: dict) -> None:
        self.cursor_index += 1
        cursor = f"s=synthetic;i={self.cursor_index}"
        message = compact_json(event)
        journal_pid = 1200 if phase == "resource_normal" else 2200
        monotonic_usec = event["observed_monotonic_ns"] // 1000
        self.session_add(
            "gateway_event",
            phase,
            case_id,
            journal_cursor=cursor,
            journal_monotonic_usec=monotonic_usec,
            journal_pid=journal_pid,
            message=message,
            message_sha256=sha256_bytes(message.encode("utf-8")),
            event=event,
        )
        self.journal_records.append(
            {
                "__CURSOR": cursor,
                "__MONOTONIC_TIMESTAMP": str(monotonic_usec),
                "_BOOT_ID": BOOT_ID,
                "_PID": str(journal_pid),
                "_SYSTEMD_UNIT": "ullm-openai.service",
                "PRIORITY": "6",
                "MESSAGE": message,
            }
        )

    def lifecycle_probe(self, phase: str, probe: str, segment: str) -> None:
        gateway_pid, worker_pid, gateway_start, worker_start = (
            (1200, 1201, 10_000, 10_001)
            if segment == "normal"
            else (2200, 2201, 20_000, 20_001)
        )
        self.session_add(
            "lifecycle_probe",
            phase,
            probe,
            probe=probe,
            observed_monotonic_ns=self.now,
            service_active=True,
            ready_http_status=200,
            control_group="/system.slice/ullm-openai.service",
            gateway_pid=gateway_pid,
            gateway_starttime_ticks=gateway_start,
            worker_pid=worker_pid,
            worker_starttime_ticks=worker_start,
            n_restarts=2 if segment == "normal" else 3,
        )

    def http_exchange(
        self,
        phase: str,
        case_id: str,
        request_index: int,
        body: bytes,
        response: bytes,
        status: int,
        sent_time: int,
        response_time: int,
    ) -> None:
        request_key = f"p8f-{case_id}"
        self.session_add(
            "http_request",
            phase,
            case_id,
            request_index=request_index,
            request_key=request_key,
            method="POST",
            target="/v1/chat/completions",
            headers={
                "content_type": "application/json",
                "content_length": len(body),
                "authorization_mode": "valid_bearer",
            },
            body_base64=base64.b64encode(body).decode("ascii"),
            body_sha256=sha256_bytes(body),
            body_bytes=len(body),
            connect_completed_monotonic_ns=sent_time - 2,
            write_started_monotonic_ns=sent_time - 1,
            last_body_byte_sent_monotonic_ns=sent_time,
        )
        self.session_add(
            "http_response_start",
            phase,
            case_id,
            request_key=request_key,
            status=status,
            headers=[
                [
                    "Content-Type",
                    "text/event-stream" if status == 200 else "application/json",
                ]
            ],
            observed_monotonic_ns=sent_time + 1,
        )
        self.session_add(
            "http_body_chunk",
            phase,
            case_id,
            request_key=request_key,
            chunk_index=0,
            body_base64=base64.b64encode(response).decode("ascii"),
            body_sha256=sha256_bytes(response),
            body_bytes=len(response),
            observed_monotonic_ns=response_time - 1,
        )
        self.session_add(
            "http_response_end",
            phase,
            case_id,
            request_key=request_key,
            outcome="eof",
            error=None,
            body_bytes=len(response),
            body_sha256=sha256_bytes(response),
            observed_monotonic_ns=response_time,
        )

    def positive_body(self, segment: str, role: str, index: int) -> bytes:
        sampled = (
            segment == "normal"
            and role == "measured"
            and index in SCHEDULE["sampled_normal_indices"]
        )
        return json.dumps(
            {
                "model": self.fixture["model"],
                "messages": self.fixture["messages"],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 2,
                "temperature": 0.6 if sampled else 0,
                "top_p": 0.95 if sampled else 1,
                "seed": index if sampled else 0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

    def lifecycle_request(self, segment: str, role: str, index: int) -> tuple[str, int]:
        phase = f"resource_{segment}"
        request_id = f"req-{segment}-{role}-{index:03d}"
        completion_id = f"chatcmpl-{segment}-{role}-{index:03d}"
        case_id = (
            f"{segment}-warmup-{index:02d}"
            if role == "warmup"
            else f"{segment}-measured-{index:03d}"
        )
        admitted_time = self.now
        started_time = admitted_time + 100_000
        release_time = admitted_time + 1_000_000
        response = (
            b"data: "
            + compact_json(
                {
                    "id": completion_id,
                    "choices": [{"delta": {"content": "x"}}],
                }
            ).encode("utf-8")
            + b"\n\ndata: "
            + compact_json(
                {
                    "id": completion_id,
                    "choices": [],
                    "usage": {"completion_tokens": 2},
                }
            ).encode("utf-8")
            + b"\n\ndata: [DONE]\n\n"
        )
        self.http_exchange(
            phase,
            case_id,
            index,
            self.positive_body(segment, role, index),
            response,
            200,
            admitted_time,
            release_time + 100_000,
        )
        self.gateway_event(
            phase,
            case_id,
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_admitted",
                "observed_monotonic_ns": admitted_time,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 32,
                "max_completion_tokens": 2,
            },
        )
        self.gateway_event(
            phase,
            case_id,
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_started",
                "observed_monotonic_ns": started_time,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 32,
                "admit_to_start_ns": 100_000,
            },
        )
        self.gateway_event(
            phase,
            case_id,
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_first_token",
                "observed_monotonic_ns": started_time + 100_000,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "completion_tokens": 1,
            },
        )
        self.gateway_event(
            phase,
            case_id,
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_released",
                "observed_monotonic_ns": release_time,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "outcome": "length",
                "cancel_reason": None,
                "prompt_tokens": 32,
                "completion_tokens": 2,
                "reset_complete": True,
                "admit_to_start_ns": 100_000,
                "start_to_release_ns": 900_000,
                "admit_to_release_ns": 1_000_000,
            },
        )
        self.now = release_time + 10_000_000_000
        return request_id, release_time

    def negative_request(self, index: int, kind: str) -> None:
        if kind == "malformed_json":
            case_id = "negative-after-050-malformed_json"
            body = b"{"
            code = "invalid_request_error"
            param = None
        else:
            suffix = "1" if index == 25 else "2"
            case_id = f"negative-after-{index:03d}-context_overflow_{suffix}"
            marker = "one" if index == 25 else "two"
            body = compact_json(
                {
                    "model": self.fixture["model"],
                    "messages": [
                        {"role": "user", "content": marker + (" overflow" * 5000)}
                    ],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": 2,
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 0,
                }
            ).encode("utf-8")
            code = "context_length_exceeded"
            param = "messages"
        response = compact_json(
            {
                "error": {
                    "message": "rejected",
                    "type": "invalid_request_error",
                    "param": param,
                    "code": code,
                }
            }
        ).encode("utf-8")
        self.http_exchange(
            "resource_normal",
            case_id,
            index,
            body,
            response,
            400,
            self.now,
            self.now + 1_000_000,
        )
        self.now += 10_000_000

    def process(self, segment: str, kind: str):
        gateway_pid, worker_pid, gateway_start, worker_start = (
            (1200, 1201, 10_000, 10_001)
            if segment == "normal"
            else (2200, 2201, 20_000, 20_001)
        )
        if kind == "gateway":
            return {
                "pid": gateway_pid,
                "ppid": 1,
                "exe": "/usr/bin/python3.12",
                "starttime_ticks_before": gateway_start,
                "starttime_ticks_after": gateway_start,
                "vmrss_kb": 100_000,
                "vmrss_bytes": 102_400_000,
                "threads": 8,
                "fd_count": 32,
                "children": [worker_pid],
            }
        return {
            "pid": worker_pid,
            "ppid": gateway_pid,
            "exe": "/opt/ullm/bin/ullm-sq8-worker",
            "starttime_ticks_before": worker_start,
            "starttime_ticks_after": worker_start,
            "vmrss_kb": 200_000,
            "vmrss_bytes": 204_800_000,
            "threads": 12,
            "fd_count": 24,
            "children": [],
        }

    def resource_point(
        self,
        segment: str,
        phase: str,
        request_index,
        request_id,
        release_time,
    ) -> None:
        gateway_pid = 1200 if segment == "normal" else 2200
        worker_pid = gateway_pid + 1
        ordinal = request_index or 0
        memory = 1_000_000_000 + ordinal * 1024
        vram = 20_000_000_000 + ordinal * 1024
        settle_start = release_time
        for sample_index in range(5):
            self.resource_records.append(
                {
                    "schema_version": VALIDATOR.RESOURCE_SCHEMA,
                    "record_type": "resource_sample",
                    "segment": segment,
                    "phase": phase,
                    "request_index": request_index,
                    "request_id": request_id,
                    "release_outcome": None if phase == "baseline" else "length",
                    "release_observed_monotonic_ns": None
                    if phase == "baseline"
                    else release_time,
                    "reset_complete": None if phase == "baseline" else True,
                    "idle_settle_started_monotonic_ns": settle_start,
                    "sample_index": sample_index,
                    "sample_monotonic_ns": settle_start
                    + 5_000_000_000
                    + sample_index * 1_000_000_000,
                    "systemd": {
                        "control_group_before": "/system.slice/ullm-openai.service",
                        "control_group_after": "/system.slice/ullm-openai.service",
                        "main_pid_before": gateway_pid,
                        "main_pid_after": gateway_pid,
                    },
                    "host": {"memory_current_bytes": memory},
                    "gateway": self.process(segment, "gateway"),
                    "worker": self.process(segment, "worker"),
                    "gpu": {
                        "index": 2,
                        "bdf": "0000:47:00.0",
                        "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
                        "kfd_gpu_id": 51545,
                        "process_record_count": 1,
                        "worker_pid": worker_pid,
                        "mem_usage": {"value": vram, "unit": "B"},
                        "kfd_vram_bytes": vram,
                        "unrelated_process_pids": [],
                    },
                }
            )
        self.now = max(self.now, settle_start + 10_000_000_000)

    def metric(self, segment: str, boundary: str, captured: int) -> None:
        filename = f"amd-smi-metric-{segment}-{boundary}.json"
        self.resource_records.append(
            {
                "schema_version": VALIDATOR.RESOURCE_SCHEMA,
                "record_type": "gpu_metric",
                "segment": segment,
                "boundary": boundary,
                "captured_monotonic_ns": captured,
                "gpu_index": 2,
                "raw_output_file": filename,
                "raw_output_sha256": sha256_file(self.root / filename),
            }
        )

    def resource_header(self):
        return {
            "schema_version": VALIDATOR.RESOURCE_SCHEMA,
            "record_type": "header",
            "service_unit": "ullm-openai.service",
            "commands": deepcopy(VALIDATOR.COMMANDS),
            "tools": {
                "systemd_major": 255,
                "systemd_version_line": "systemd 255 (255.4-1ubuntu8.16)",
                "amd_smi_tool": "26.2.2+e1a6bc5663",
                "amd_smi_library": "26.2.2",
                "rocm": "7.2.1",
                "amd_smi_version_output": (
                    "AMDSMI Tool: 26.2.2+e1a6bc5663 | AMDSMI Library version: "
                    "26.2.2 | ROCm version: 7.2.1 | amdgpu version: 6.16.13"
                ),
            },
            "probes": {
                "cgroup_fs_type": "cgroup2fs",
                "kfd_proc_present": True,
                "gpu_index": 2,
                "gpu_bdf": "0000:47:00.0",
                "gpu_uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
                "kfd_gpu_id": 51545,
            },
            "schedule": {
                "normal_warmups": 10,
                "normal_requests": 100,
                "restart_warmups": 10,
                "restart_requests": 20,
                "idle_settle_ms": 5000,
                "samples_per_point": 5,
                "sample_interval_ms": 1000,
            },
        }

    def segment(self, segment: str, measured_count: int) -> None:
        self.metric(segment, "before", self.now - 1)
        warmup_release = 0
        for index in range(1, 11):
            _, warmup_release = self.lifecycle_request(segment, "warmup", index)
        self.resource_point(segment, "baseline", None, None, warmup_release)
        for index in range(1, measured_count + 1):
            request_id, release_time = self.lifecycle_request(
                segment, "measured", index
            )
            self.resource_point(
                segment, "post_release", index, request_id, release_time
            )
            if segment == "normal" and index in {25, 50, 75}:
                self.negative_request(
                    index, "malformed_json" if index == 50 else "context_overflow"
                )
        final_sample = self.resource_records[-1]["sample_monotonic_ns"]
        self.metric(segment, "after", final_sample + 1)
        self.now = final_sample + 10_000_000_000

    def build(self) -> None:
        self.root.mkdir(parents=True)
        (self.root / "browser").mkdir()
        self.write_json("environment.json", {"synthetic": "environment"})
        self.write_json("model-identity.json", {"synthetic": "model"})
        for segment in ("normal", "restart"):
            for boundary in ("before", "after"):
                (self.root / f"amd-smi-metric-{segment}-{boundary}.json").write_text(
                    compact_json([{"segment": segment, "boundary": boundary}]) + "\n",
                    encoding="utf-8",
                )
        for name in (
            "sampling-results.json",
            "cancel-results.json",
            "prefill-latency-results.json",
            "api-contract-results.json",
            "openwebui-smoke.json",
            "soak-results.json",
        ):
            self.write_json(name, {"derived": name})
        (self.root / "browser/openwebui-stop-before.png").write_bytes(
            b"\x89PNG\r\n\x1a\nstop"
        )
        (self.root / "browser/post-header-failure.png").write_bytes(
            b"\x89PNG\r\n\x1a\nfailure"
        )
        (self.root / "summary.md").write_text(
            "synthetic phase-1 evidence\n", encoding="utf-8"
        )

        self.session_add(
            "header",
            "preflight",
            None,
            run_id=RUN_ID,
            started_utc="2026-07-11T00:00:00Z",
            clock="python.time.monotonic_ns",
            boot_id=BOOT_ID,
            identities={
                "environment_file": "environment.json",
                "environment_sha256": sha256_file(self.root / "environment.json"),
                "model_identity_file": "model-identity.json",
                "model_identity_sha256": sha256_file(self.root / "model-identity.json"),
                "openwebui": {
                    "version": "0.9.4",
                    "source_revision": "synthetic-revision",
                    "base_image_digest": "sha256:" + "1" * 64,
                    "base_image_id": "sha256:" + "2" * 64,
                    "derived_image_id": "sha256:" + "3" * 64,
                    "Dockerfile_sha256": "4" * 64,
                    "patch_sha256": "5" * 64,
                    "patched_middleware_sha256": "6" * 64,
                },
                "docker_network_id": "9" * 64,
                "gateway_source_sha256": "7" * 64,
                "worker_source_sha256": "8" * 64,
                "worker_binary_sha256": WORKER_SHA256,
            },
            input_files=[
                {
                    "path": "collector/config.json",
                    "bytes": 2,
                    "sha256": sha256_bytes(b"{}"),
                },
                {
                    "path": VALIDATOR.RESOURCE_FIXTURE_INPUT_PATH,
                    "bytes": len(
                        json.dumps(
                            self.fixture,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                    "sha256": sha256_bytes(
                        json.dumps(
                            self.fixture,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ),
                },
                {
                    "path": "tools/collect-sq8-openwebui-release.py",
                    "bytes": 1,
                    "sha256": sha256_bytes(b"c"),
                },
                {
                    "path": "tools/sq8-openwebui-http-client.py",
                    "bytes": 1,
                    "sha256": sha256_bytes(b"h"),
                },
            ],
            schedule=deepcopy(SCHEDULE),
            thresholds=deepcopy(THRESHOLDS),
        )
        self.resource_records.append(self.resource_header())
        self.lifecycle_probe("resource_normal", "normal-segment-start", "normal")
        self.segment("normal", 100)
        self.lifecycle_probe(
            "post_header_failure", "post-header-restart-ready", "restart"
        )
        self.lifecycle_probe("resource_restart", "restart-segment-start", "restart")
        self.segment("restart", 20)
        self.lifecycle_probe("final", "final-service-ready", "restart")
        counts = Counter(record["record_type"] for record in self.session_records)
        counts["run_end"] += 1
        self.session_add(
            "run_end",
            "final",
            None,
            completed_utc="2026-07-11T01:00:00Z",
            completed_monotonic_ns=self.now,
            final_git_commit=GIT_COMMIT,
            final_git_status_raw="",
            final_git_status_sha256=sha256_bytes(b""),
            record_counts=dict(counts),
            final_journal_cursor=self.journal_records[-1]["__CURSOR"],
        )
        self.write_jsonl("raw-session-results.jsonl", self.session_records)
        self.write_jsonl("soak-resources.raw.jsonl", self.resource_records)
        self.write_jsonl("service-journal.raw.jsonl", self.journal_records)
        refresh_matrix_and_sums(self.root)

    def write_jsonl(self, relative: str, records) -> None:
        text = "".join(compact_json(record) + "\n" for record in records)
        (self.root / relative).write_text(text, encoding="utf-8")


def refresh_matrix_and_sums(root: Path, matrix_mutator=None) -> None:
    files = []
    for relative in sorted(
        VALIDATOR.EXPECTED_ROLES, key=lambda item: item.encode("utf-8")
    ):
        path = root / relative
        files.append(
            {
                "role": VALIDATOR.EXPECTED_ROLES[relative],
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    matrix = {
        "schema_version": VALIDATOR.MATRIX_SCHEMA,
        "run_id": RUN_ID,
        "files": files,
        "schedule": deepcopy(SCHEDULE),
        "thresholds": deepcopy(THRESHOLDS),
    }
    if matrix_mutator is not None:
        matrix_mutator(matrix)
    (root / "release-matrix.json").write_text(compact_json(matrix), encoding="utf-8")
    paths = sorted(
        VALIDATOR.BUNDLE_FILES - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8")
    )
    sums = "".join(
        f"{sha256_file(root / relative)}  {relative}\n" for relative in paths
    )
    (root / "SHA256SUMS").write_text(sums, encoding="ascii")


def mutate_jsonl(root: Path, relative: str, mutator) -> None:
    path = root / relative
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    mutator(records)
    path.write_text(
        "".join(compact_json(record) + "\n" for record in records), encoding="utf-8"
    )
    refresh_matrix_and_sums(root)


def replace_request_body(record: dict, raw: bytes) -> None:
    record["body_base64"] = base64.b64encode(raw).decode("ascii")
    record["body_sha256"] = sha256_bytes(raw)
    record["body_bytes"] = len(raw)
    record["headers"]["content_length"] = len(raw)


def rewrite_gateway_event_time(
    root: Path, request_id: str, event_name: str, observed_ns: int
) -> None:
    session_path = root / "raw-session-results.jsonl"
    session = [
        json.loads(line)
        for line in session_path.read_text(encoding="utf-8").splitlines()
    ]
    target_cursor = None
    replacement_message = None
    for record in session:
        event = record.get("event")
        if (
            record.get("record_type") == "gateway_event"
            and event.get("request_id") == request_id
            and event.get("event") == event_name
        ):
            event["observed_monotonic_ns"] = observed_ns
            replacement_message = compact_json(event)
            record["message"] = replacement_message
            record["message_sha256"] = sha256_bytes(replacement_message.encode("utf-8"))
            record["journal_monotonic_usec"] = observed_ns // 1000
            target_cursor = record["journal_cursor"]
            break
    if target_cursor is None or replacement_message is None:
        raise AssertionError("gateway event mutation target was not found")
    session_path.write_text(
        "".join(compact_json(record) + "\n" for record in session), encoding="utf-8"
    )
    journal_path = root / "service-journal.raw.jsonl"
    journal = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
    ]
    for record in journal:
        if record["__CURSOR"] == target_cursor:
            record["MESSAGE"] = replacement_message
            record["__MONOTONIC_TIMESTAMP"] = str(observed_ns // 1000)
            break
    journal_path.write_text(
        "".join(compact_json(record) + "\n" for record in journal), encoding="utf-8"
    )
    refresh_matrix_and_sums(root)


def insert_gateway_trace_before_case(
    root: Path,
    before_case_id: str,
    phase: str,
    case_id: str,
    journal_pid: int,
    events: list[dict],
) -> None:
    session_path = root / "raw-session-results.jsonl"
    session = [json.loads(line) for line in session_path.read_text().splitlines()]
    insert_at = next(
        index
        for index, record in enumerate(session)
        if record.get("record_type") == "http_request"
        and record.get("case_id") == before_case_id
    )
    boot_id = session[0]["boot_id"]
    gateway_records = []
    journal_records = []
    for index, event in enumerate(events):
        cursor = f"test-insert-{case_id}-{index}"
        message = compact_json(event)
        gateway_records.append(
            {
                "schema_version": VALIDATOR.SESSION_SCHEMA,
                "record_type": "gateway_event",
                "sequence": 0,
                "phase": phase,
                "case_id": case_id,
                "journal_cursor": cursor,
                "journal_monotonic_usec": event["observed_monotonic_ns"] // 1000,
                "journal_pid": journal_pid,
                "message": message,
                "message_sha256": sha256_bytes(message.encode("utf-8")),
                "event": event,
            }
        )
        journal_records.append(
            {
                "__CURSOR": cursor,
                "__MONOTONIC_TIMESTAMP": str(event["observed_monotonic_ns"] // 1000),
                "_BOOT_ID": boot_id,
                "_PID": str(journal_pid),
                "_SYSTEMD_UNIT": "ullm-openai.service",
                "PRIORITY": "6",
                "MESSAGE": message,
            }
        )
    session[insert_at:insert_at] = gateway_records
    for sequence, record in enumerate(session):
        record["sequence"] = sequence
    counts = {}
    for record in session:
        counts[record["record_type"]] = counts.get(record["record_type"], 0) + 1
    session[-1]["record_counts"] = counts
    session_path.write_text(
        "".join(compact_json(record) + "\n" for record in session),
        encoding="utf-8",
    )
    journal_path = root / "service-journal.raw.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
    journal.extend(journal_records)
    journal.sort(key=lambda record: int(record["__MONOTONIC_TIMESTAMP"]))
    journal_path.write_text(
        "".join(compact_json(record) + "\n" for record in journal),
        encoding="utf-8",
    )
    refresh_matrix_and_sums(root)


class FullCampaignOrderFixture:
    def __init__(self):
        self.records = []
        self.now = 1_000_000
        self.old_gateway_pid = 1200
        self.old_worker_pid = 1201
        self.new_gateway_pid = 2200
        self.new_worker_pid = 2201

    def add(self, record_type: str, phase: str, case_id, **fields) -> None:
        self.records.append(
            {
                "schema_version": VALIDATOR.SESSION_SCHEMA,
                "record_type": record_type,
                "sequence": len(self.records),
                "phase": phase,
                "case_id": case_id,
                **fields,
            }
        )

    def tick(self) -> int:
        self.now += 1_000
        return self.now

    def event(
        self,
        phase: str,
        case_id: str,
        request_id: str,
        event_name: str,
        gateway_pid: int,
        **fields,
    ) -> None:
        self.add(
            "gateway_event",
            phase,
            case_id,
            journal_pid=gateway_pid,
            event={
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": event_name,
                "observed_monotonic_ns": self.tick(),
                "request_id": request_id,
                "completion_id": f"chatcmpl-{request_id}",
                **fields,
            },
        )

    def successful_trace(
        self, phase: str, case_id: str, gateway_pid: int, outcome: str = "stop"
    ) -> None:
        request_id = f"req-{case_id}"
        self.event(phase, case_id, request_id, "request_admitted", gateway_pid)
        self.event(
            phase,
            case_id,
            request_id,
            "request_released",
            gateway_pid,
            outcome=outcome,
            reset_complete=True,
        )

    def lifecycle_probe(self, phase: str, name: str, *, restarted: bool) -> None:
        gateway_pid = self.new_gateway_pid if restarted else self.old_gateway_pid
        worker_pid = self.new_worker_pid if restarted else self.old_worker_pid
        self.add(
            "lifecycle_probe",
            phase,
            name,
            probe=name,
            observed_monotonic_ns=self.tick(),
            service_active=True,
            ready_http_status=200,
            control_group="/system.slice/ullm-openai.service",
            gateway_pid=gateway_pid,
            gateway_starttime_ticks=20_000 if restarted else 10_000,
            worker_pid=worker_pid,
            worker_starttime_ticks=20_001 if restarted else 10_001,
            n_restarts=3 if restarted else 2,
        )

    def cancellation_pair(self, cancel_phase: str, index: int) -> None:
        target_case = f"cancel-target-{index}"
        recovery_case = f"cancel-recovery-{index}"
        request_id = f"req-{target_case}"
        if cancel_phase != "openwebui_stop_after_visible_content":
            self.add("http_request", "cancellation", target_case)
        self.event(
            "cancellation",
            target_case,
            request_id,
            "request_admitted",
            self.old_gateway_pid,
        )
        self.event(
            "cancellation",
            target_case,
            request_id,
            "request_started",
            self.old_gateway_pid,
        )
        if cancel_phase == "prefill_after_128":
            self.event(
                "cancellation",
                target_case,
                request_id,
                "request_progress",
                self.old_gateway_pid,
                processed_prompt_tokens=128,
            )
        elif cancel_phase == "prefill_after_2048":
            for boundary in (128, 2048):
                self.event(
                    "cancellation",
                    target_case,
                    request_id,
                    "request_progress",
                    self.old_gateway_pid,
                    processed_prompt_tokens=boundary,
                )
        elif cancel_phase in {
            "decode_after_first_content",
            "openwebui_stop_after_visible_content",
        }:
            self.event(
                "cancellation",
                target_case,
                request_id,
                "request_first_token",
                self.old_gateway_pid,
            )
        if cancel_phase == "openwebui_stop_after_visible_content":
            wait_started = self.tick()
            wait_completed = self.tick()
            self.add(
                "browser_action",
                "cancellation",
                target_case,
                action="wait_visible",
                started_monotonic_ns=wait_started,
                completed_monotonic_ns=wait_completed,
            )
            click_started = self.tick()
            click_completed = self.tick()
            self.add(
                "browser_action",
                "cancellation",
                target_case,
                action="click_stop",
                started_monotonic_ns=click_started,
                completed_monotonic_ns=click_completed,
            )
        self.event(
            "cancellation",
            target_case,
            request_id,
            "request_cancel_requested",
            self.old_gateway_pid,
        )
        self.event(
            "cancellation",
            target_case,
            request_id,
            "request_released",
            self.old_gateway_pid,
            outcome="cancelled",
            reset_complete=True,
        )
        self.successful_trace(
            "cancellation", recovery_case, self.old_gateway_pid, "length"
        )

    def build(self):
        self.add("header", "preflight", None)
        self.add("http_response_end", "api_contract", "fixed-api-contract")
        for index in range(21):
            case_id = "openwebui-smoke" if index == 0 else f"openwebui-soak-{index:02d}"
            self.successful_trace("openwebui", case_id, self.old_gateway_pid)
        for index, cancel_phase in enumerate(VALIDATOR.CANCEL_PHASES):
            self.cancellation_pair(cancel_phase, index)

        self.lifecycle_probe("resource_normal", "normal-segment-start", restarted=False)
        self.successful_trace(
            "resource_normal", "normal-resource", self.old_gateway_pid, "length"
        )

        failed_case = "post-header-failure"
        failed_request = f"req-{failed_case}"
        self.event(
            "post_header_failure",
            failed_case,
            failed_request,
            "request_admitted",
            self.old_gateway_pid,
        )
        self.event(
            "post_header_failure",
            failed_case,
            failed_request,
            "request_started",
            self.old_gateway_pid,
        )
        fault_started = self.tick()
        fault_completed = self.tick()
        self.add(
            "fault_injection",
            "post_header_failure",
            failed_case,
            injection="post_header_worker_kill",
            target_pid=self.old_worker_pid,
            target_starttime_ticks=10_001,
            signal="SIGKILL",
            started_monotonic_ns=fault_started,
            completed_monotonic_ns=fault_completed,
        )
        self.event(
            "post_header_failure",
            failed_case,
            failed_request,
            "worker_fatal",
            self.old_gateway_pid,
        )
        self.lifecycle_probe(
            "post_header_failure", "post-header-restart-ready", restarted=True
        )
        self.successful_trace(
            "post_header_failure",
            "post-header-recovery",
            self.new_gateway_pid,
        )

        self.lifecycle_probe(
            "resource_restart", "restart-segment-start", restarted=True
        )
        self.successful_trace(
            "resource_restart", "restart-resource", self.new_gateway_pid, "length"
        )
        self.successful_trace(
            "latency", "latency-matrix", self.new_gateway_pid, "length"
        )
        self.lifecycle_probe("final", "final-service-ready", restarted=True)
        self.add("run_end", "final", None)
        return self.records


def resequence_full_campaign(records) -> None:
    for sequence, record in enumerate(records):
        record["sequence"] = sequence


class FullCampaignOrderTest(unittest.TestCase):
    def setUp(self):
        self.records = FullCampaignOrderFixture().build()

    def validate(self):
        return VALIDATOR.validate_full_campaign_order(self.records)

    def assert_invalid(self, text: str):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, text):
            self.validate()

    def test_valid_full_campaign_fixes_order_cardinality_and_epochs(self):
        result = self.validate()
        self.assertEqual(result.phases, VALIDATOR.FULL_CAMPAIGN_PHASE_ORDER)
        self.assertEqual(result.openwebui_successful_requests, 21)
        self.assertEqual(result.cancellation_phases, VALIDATOR.CANCEL_PHASES)
        self.assertEqual(
            (result.normal_gateway_pid, result.restart_gateway_pid), (1200, 2200)
        )
        self.assertEqual(
            (result.normal_worker_pid, result.restart_worker_pid), (1201, 2201)
        )
        self.assertEqual(
            (result.restart_count_before, result.restart_count_after), (2, 3)
        )

    def test_compact_projection_retains_only_order_required_fields(self) -> None:
        compact = [
            VALIDATOR._compact_session_order_record(record) for record in self.records
        ]
        result = VALIDATOR.validate_full_campaign_order(compact)
        self.assertEqual(result.phases, VALIDATOR.FULL_CAMPAIGN_PHASE_ORDER)
        self.assertTrue(VALIDATOR._claims_full_campaign(compact))
        browser = next(
            record for record in compact if record["record_type"] == "browser_action"
        )
        self.assertEqual(
            set(browser),
            {
                "schema_version",
                "record_type",
                "sequence",
                "phase",
                "case_id",
                "action",
                "completed_monotonic_ns",
            },
        )

    def test_compact_projection_rejects_a_missing_full_phase(self) -> None:
        self.records[:] = [
            record for record in self.records if record["phase"] != "latency"
        ]
        resequence_full_campaign(self.records)
        compact = [
            VALIDATOR._compact_session_order_record(record) for record in self.records
        ]
        self.assertTrue(VALIDATOR._claims_full_campaign(compact))
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "phase set/order differs"
        ):
            VALIDATOR.validate_full_campaign_order(compact)

    def test_second_fault_is_rejected(self) -> None:
        fault_index = next(
            index
            for index, record in enumerate(self.records)
            if record["record_type"] == "fault_injection"
        )
        self.records.insert(fault_index + 1, deepcopy(self.records[fault_index]))
        resequence_full_campaign(self.records)
        self.assert_invalid("exactly one fault")

    def test_browser_action_compact_order_is_strict(self) -> None:
        actions = (
            VALIDATOR.BrowserActionData(
                phase="openwebui",
                case_id="case-1",
                browser_case="browser-1",
                action_index=0,
                action="navigate",
                selector=None,
                input_sha256=None,
                started_monotonic_ns=10,
                completed_monotonic_ns=20,
                result_visible=None,
                result_enabled=None,
                result_text_utf8_bytes=None,
                result_text_sha256=None,
                screenshot_file=None,
                screenshot_sha256=None,
            ),
            VALIDATOR.BrowserActionData(
                phase="openwebui",
                case_id="case-1",
                browser_case="browser-1",
                action_index=1,
                action="select_model",
                selector=None,
                input_sha256=None,
                started_monotonic_ns=19,
                completed_monotonic_ns=30,
                result_visible=None,
                result_enabled=None,
                result_text_utf8_bytes=None,
                result_text_sha256=None,
                screenshot_file=None,
                screenshot_sha256=None,
            ),
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "timestamps regress or overlap"
        ):
            VALIDATOR._validate_browser_action_order(actions)
        broken_index = (
            actions[0],
            dataclasses.replace(actions[1], action_index=2, started_monotonic_ns=20),
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "indices are not contiguous"
        ):
            VALIDATOR._validate_browser_action_order(broken_index)

    def test_browser_and_fault_compact_data_retains_validated_fields(self) -> None:
        input_raw = b"http://open-webui/"
        text_raw = "表示済み".encode("utf-8")
        browser = VALIDATOR._validate_browser_action_data(
            {
                "browser_case": "browser-1",
                "action_index": 0,
                "action": "navigate",
                "selector": "#chat-input",
                "input_sha256": sha256_bytes(input_raw),
                "started_monotonic_ns": 10,
                "completed_monotonic_ns": 20,
                "result": {
                    "visible": True,
                    "enabled": False,
                    "text_utf8_bytes": len(text_raw),
                    "text_sha256": sha256_bytes(text_raw),
                },
                "screenshot_file": None,
                "screenshot_sha256": None,
            },
            "openwebui",
            "case-1",
            "browser compact test",
        )
        self.assertEqual(browser.selector, "#chat-input")
        self.assertEqual(browser.input_sha256, sha256_bytes(input_raw))
        self.assertIs(browser.result_visible, True)
        self.assertIs(browser.result_enabled, False)
        self.assertEqual(browser.result_text_utf8_bytes, len(text_raw))
        self.assertEqual(browser.result_text_sha256, sha256_bytes(text_raw))

        command_raw = b"signal.pidfd_send_signal"
        fault = VALIDATOR._validate_fault_injection_data(
            {
                "injection": "post_header_worker_kill",
                "target_pid": 1201,
                "target_starttime_ticks": 10_001,
                "signal": "SIGKILL",
                "command": command_raw.decode("ascii"),
                "started_monotonic_ns": 30,
                "completed_monotonic_ns": 40,
            },
            "post_header_failure",
            "post-header-failure",
            "fault compact test",
        )
        self.assertEqual(fault.command_utf8_bytes, len(command_raw))
        self.assertEqual(fault.command_sha256, sha256_bytes(command_raw))
        with self.assertRaises(VALIDATOR.ValidationError):
            VALIDATOR._validate_fault_injection_data(
                {
                    "injection": "post_header_worker_kill",
                    "target_pid": 1201,
                    "target_starttime_ticks": 10_001,
                    "signal": "SIGKILL",
                    "command": "kill --signal KILL -- 1201",
                    "started_monotonic_ns": 30,
                    "completed_monotonic_ns": 40,
                },
                "post_header_failure",
                "post-header-failure",
                "fault compact test",
            )

    def test_browser_selector_is_bounded_before_compact_retention(self) -> None:
        record = {
            "browser_case": "browser-1",
            "action_index": 0,
            "action": "navigate",
            "selector": "x" * (VALIDATOR.MAX_BROWSER_SELECTOR_BYTES + 1),
            "input_sha256": None,
            "started_monotonic_ns": 10,
            "completed_monotonic_ns": 20,
            "result": {
                "visible": None,
                "enabled": None,
                "text_utf8_bytes": None,
                "text_sha256": None,
            },
            "screenshot_file": None,
            "screenshot_sha256": None,
        }
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "UTF-8 byte bound"):
            VALIDATOR._validate_browser_action_data(
                record, "openwebui", "case-1", "browser compact test"
            )

        record["selector"] = None
        record["browser_case"] = "x" * (VALIDATOR.MAX_SESSION_IDENTIFIER_BYTES + 1)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "UTF-8 byte bound"):
            VALIDATOR._validate_browser_action_data(
                record, "openwebui", "case-1", "browser compact test"
            )

    def test_lifecycle_request_and_completion_ids_are_bounded(self) -> None:
        event = {
            "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
            "event": "request_admitted",
            "observed_monotonic_ns": 1,
            "request_id": "r" * (VALIDATOR.MAX_SESSION_IDENTIFIER_BYTES + 1),
            "completion_id": "chatcmpl-bounded",
            "stream": True,
            "prompt_tokens": 1,
            "max_completion_tokens": 1,
        }
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "UTF-8 byte bound"):
            VALIDATOR.validate_lifecycle(event, "bounded lifecycle")
        event["request_id"] = "request-bounded"
        event["completion_id"] = "c" * (VALIDATOR.MAX_SESSION_IDENTIFIER_BYTES + 1)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "UTF-8 byte bound"):
            VALIDATOR.validate_lifecycle(event, "bounded lifecycle")

    def test_phase_block_cannot_regress(self):
        target = next(
            record
            for record in self.records
            if record["phase"] == "resource_restart"
            and record["record_type"] == "gateway_event"
        )
        target["phase"] = "resource_normal"
        self.assert_invalid("phase order regresses")

    def test_full_campaign_cannot_omit_api_phase(self):
        self.records[:] = [
            record for record in self.records if record["phase"] != "api_contract"
        ]
        resequence_full_campaign(self.records)
        self.assert_invalid("phase set/order differs")

    def test_api_contract_phase_cannot_admit_a_worker_request(self):
        insert_at = next(
            index
            for index, record in enumerate(self.records)
            if record["phase"] == "openwebui"
        )
        fixture = FullCampaignOrderFixture()
        fixture.records = []
        fixture.successful_trace(
            "api_contract", "unexpected-api-admission", 1200, "length"
        )
        self.records[insert_at:insert_at] = fixture.records
        resequence_full_campaign(self.records)
        self.assert_invalid("API contract phase produced a worker lifecycle admission")

    def test_openwebui_soak_requires_exactly_twenty_after_smoke(self):
        self.records[:] = [
            record
            for record in self.records
            if record.get("case_id") != "openwebui-soak-20"
        ]
        resequence_full_campaign(self.records)
        self.assert_invalid("smoke/20-chat cardinality")

    def test_cancellation_trigger_order_is_derived_from_events(self):
        target = next(
            record
            for record in self.records
            if record.get("case_id") == "cancel-target-1"
            and record.get("record_type") == "gateway_event"
            and record["event"]["event"] == "request_progress"
        )
        target["event"]["processed_prompt_tokens"] = 2048
        self.assert_invalid("cancellation phase order differs")

    def test_each_cancellation_requires_immediate_successful_recovery(self):
        target = next(
            record
            for record in self.records
            if record.get("case_id") == "cancel-recovery-0"
            and record.get("record_type") == "gateway_event"
            and record["event"]["event"] == "request_released"
        )
        target["event"]["outcome"] = "cancelled"
        self.assert_invalid("immediate successful recovery")

    def test_first_four_cancellations_require_direct_http_transport(self):
        self.records[:] = [
            record
            for record in self.records
            if not (
                record.get("case_id") == "cancel-target-0"
                and record.get("record_type") == "http_request"
            )
        ]
        resequence_full_campaign(self.records)
        self.assert_invalid("direct cancellation transport differs")

    def test_stop_case_requires_visible_then_click_before_cancel(self):
        self.records[:] = [
            record
            for record in self.records
            if not (
                record.get("case_id") == "cancel-target-4"
                and record.get("record_type") == "browser_action"
                and record.get("action") == "click_stop"
            )
        ]
        resequence_full_campaign(self.records)
        self.assert_invalid("Stop action order differs")

    def test_restart_count_must_increment_exactly_once(self):
        for record in self.records:
            if (
                record.get("record_type") == "lifecycle_probe"
                and record.get("probe") != "normal-segment-start"
            ):
                record["n_restarts"] = 4
        self.assert_invalid("restart identity/count boundary differs")

    def test_restart_must_change_both_process_identities(self):
        for record in self.records:
            if (
                record.get("record_type") == "lifecycle_probe"
                and record.get("probe") != "normal-segment-start"
            ):
                record["gateway_pid"] = 1200
                record["gateway_starttime_ticks"] = 10_000
                record["worker_pid"] = 1201
                record["worker_starttime_ticks"] = 10_001
        self.assert_invalid("restart identity/count boundary differs")

    def test_fault_target_is_bound_to_old_worker_identity(self):
        target = next(
            record
            for record in self.records
            if record.get("record_type") == "fault_injection"
        )
        target["target_starttime_ticks"] += 1
        self.assert_invalid("fault identity/order differs")

    def test_gateway_epoch_cannot_cross_the_restart_boundary(self):
        for record in self.records:
            if (
                record.get("phase") == "resource_restart"
                and record.get("record_type") == "gateway_event"
            ):
                record["journal_pid"] = 1200
        self.assert_invalid("journal PID differs from its lifecycle probe epoch")

    def test_no_second_worker_fatal_is_accepted(self):
        target = next(
            record
            for record in self.records
            if record.get("phase") == "latency"
            and record.get("record_type") == "gateway_event"
            and record["event"]["event"] == "request_released"
        )
        target["event"]["event"] = "worker_fatal"
        self.assert_invalid("sole planned|exactly one")

    def test_post_header_recovery_must_follow_ready_probe(self):
        probe_index = next(
            index
            for index, record in enumerate(self.records)
            if record.get("probe") == "post-header-restart-ready"
        )
        probe = self.records.pop(probe_index)
        insertion = next(
            index
            for index, record in enumerate(self.records)
            if record.get("phase") == "resource_restart"
        )
        self.records.insert(insertion, probe)
        resequence_full_campaign(self.records)
        self.assert_invalid("failure/recovery order differs")


def api_contract_response_body(case) -> bytes:
    if case.expect_models:
        value = {
            "object": "list",
            "data": [
                {
                    "id": VALIDATOR.API_CONTRACT_MODEL_ID,
                    "object": "model",
                    "owned_by": "ullm",
                }
            ],
        }
    else:
        value = {
            "error": {
                "message": case.expected_message,
                "type": "invalid_request_error",
                "param": case.expected_param,
                "code": case.expected_code,
            }
        }
    return compact_json(value).encode("utf-8")


def build_api_contract_http_records() -> list[dict]:
    records = []
    now = 1_000_000
    for case_index, case in enumerate(VALIDATOR.API_CONTRACT_CASES, start=1):
        key = f"api-contract-{case_index:02d}-{case.case_id}"
        response = api_contract_response_body(case)
        common = {
            "schema_version": VALIDATOR.SESSION_SCHEMA,
            "phase": "api_contract",
            "case_id": case.case_id,
        }
        records.extend(
            [
                {
                    **common,
                    "record_type": "http_request",
                    "request_index": case_index,
                    "request_key": key,
                    "method": case.method,
                    "target": case.target,
                    "headers": {
                        "content_type": "application/json",
                        "content_length": len(case.body),
                        "authorization_mode": case.authorization_mode,
                    },
                    "body_base64": base64.b64encode(case.body).decode("ascii"),
                    "body_sha256": sha256_bytes(case.body),
                    "body_bytes": len(case.body),
                    "connect_completed_monotonic_ns": now,
                    "write_started_monotonic_ns": now + 1,
                    "last_body_byte_sent_monotonic_ns": now + 2,
                },
                {
                    **common,
                    "record_type": "http_response_start",
                    "request_key": key,
                    "status": case.expected_status,
                    "headers": [
                        ["Content-Type", "application/json"],
                        ["Content-Length", str(len(response))],
                        ["Server", "synthetic"],
                        *(
                            [["WWW-Authenticate", "Bearer"]]
                            if case.expected_status == 401
                            else []
                        ),
                    ],
                    "observed_monotonic_ns": now + 3,
                },
                {
                    **common,
                    "record_type": "http_body_chunk",
                    "request_key": key,
                    "chunk_index": 0,
                    "body_base64": base64.b64encode(response).decode("ascii"),
                    "body_sha256": sha256_bytes(response),
                    "body_bytes": len(response),
                    "observed_monotonic_ns": now + 4,
                },
                {
                    **common,
                    "record_type": "http_response_end",
                    "request_key": key,
                    "outcome": "eof",
                    "error": None,
                    "body_bytes": len(response),
                    "body_sha256": sha256_bytes(response),
                    "observed_monotonic_ns": now + 5,
                },
            ]
        )
        now += 10
    return records


def replace_api_contract_response(
    records: list[dict], case_id: str, raw: bytes
) -> None:
    chunk = next(
        record
        for record in records
        if record["case_id"] == case_id and record["record_type"] == "http_body_chunk"
    )
    chunk["body_base64"] = base64.b64encode(raw).decode("ascii")
    chunk["body_sha256"] = sha256_bytes(raw)
    chunk["body_bytes"] = len(raw)
    end = next(
        record
        for record in records
        if record["case_id"] == case_id and record["record_type"] == "http_response_end"
    )
    end["body_sha256"] = sha256_bytes(raw)
    end["body_bytes"] = len(raw)
    start = next(
        record
        for record in records
        if record["case_id"] == case_id
        and record["record_type"] == "http_response_start"
    )
    next(pair for pair in start["headers"] if pair[0] == "Content-Length")[1] = str(
        len(raw)
    )


class HttpCompactProjectionTest(unittest.TestCase):
    def state(self) -> Any:
        return VALIDATOR.HttpValidationState(
            fixture_seal=VALIDATOR.InputSeal(size=2, sha256=sha256_bytes(b"{}")),
            requests={},
            response_started=set(),
            response_ended=set(),
            bodies={},
            ordered_keys=[],
        )

    def records(
        self, chunks: list[bytes], *, outcome: str = "eof"
    ) -> list[dict[str, Any]]:
        request_body = b"{}"
        response_body = b"".join(chunks)
        common = {
            "schema_version": VALIDATOR.SESSION_SCHEMA,
            "phase": "latency",
            "case_id": "compact-sse",
        }
        records: list[dict[str, Any]] = [
            {
                **common,
                "record_type": "http_request",
                "request_index": 1,
                "request_key": "compact-sse",
                "method": "POST",
                "target": "/v1/chat/completions",
                "headers": {
                    "content_type": "application/json",
                    "content_length": len(request_body),
                    "authorization_mode": "valid_bearer",
                },
                "body_base64": base64.b64encode(request_body).decode("ascii"),
                "body_sha256": sha256_bytes(request_body),
                "body_bytes": len(request_body),
                "connect_completed_monotonic_ns": 10,
                "write_started_monotonic_ns": 11,
                "last_body_byte_sent_monotonic_ns": 12,
            },
            {
                **common,
                "record_type": "http_response_start",
                "request_key": "compact-sse",
                "status": 200,
                "headers": [["Content-Type", "text/event-stream"]],
                "observed_monotonic_ns": 13,
            },
        ]
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    **common,
                    "record_type": "http_body_chunk",
                    "request_key": "compact-sse",
                    "chunk_index": index,
                    "body_base64": base64.b64encode(chunk).decode("ascii"),
                    "body_sha256": sha256_bytes(chunk),
                    "body_bytes": len(chunk),
                    "observed_monotonic_ns": 20 + index,
                }
            )
        records.append(
            {
                **common,
                "record_type": "http_response_end",
                "request_key": "compact-sse",
                "outcome": outcome,
                "error": None,
                "body_bytes": len(response_body),
                "body_sha256": sha256_bytes(response_body),
                "observed_monotonic_ns": 30 + len(chunks),
            }
        )
        return records

    def validate(self, records: list[dict[str, Any]]) -> Any:
        state = self.state()
        for index, record in enumerate(records):
            VALIDATOR._validate_http_record(
                record, f"compact HTTP record {index}", state
            )
        return state

    def test_crlf_boundary_and_multiline_data_are_compacted(self) -> None:
        first = (
            b'data: {"id":"chatcmpl-compact","choices":[{"delta":{"content":"x"}}]}\r'
        )
        second = (
            b'\n\r\ndata: {"id":"chatcmpl-compact",\r\n'
            b'data: "choices":[],"usage":{"completion_tokens":2}}\r\n\r\n'
            b"data: [DONE]\r\n\r\n"
        )
        state = self.validate(self.records([first, second]))
        self.assertFalse(state.bodies)
        self.assertNotIn("response_body", state.requests["compact-sse"])
        result = state.completed_results["compact-sse"]
        self.assertEqual(result.request_body_bytes, 2)
        self.assertEqual(result.response_body_bytes, len(first) + len(second))
        self.assertEqual(result.response_started_monotonic_ns, 13)
        self.assertEqual(result.response_end_monotonic_ns, 32)
        self.assertIsNotNone(result.sse)
        assert result.sse is not None
        self.assertEqual(result.sse.chunk_count, 2)
        self.assertEqual(result.sse.first_chunk_monotonic_ns, 20)
        self.assertEqual(result.sse.last_chunk_monotonic_ns, 21)
        self.assertEqual(len(result.sse.items), 3)
        completion_id = b"chatcmpl-compact"
        self.assertEqual(
            result.sse.items[0].completion_id_utf8_bytes, len(completion_id)
        )
        self.assertEqual(
            result.sse.items[0].completion_id_sha256, sha256_bytes(completion_id)
        )
        self.assertEqual(result.sse.items[0].content_utf8_bytes, 1)
        self.assertEqual(result.sse.items[1].completion_tokens, 2)
        self.assertTrue(result.sse.items[1].usage_present)
        self.assertIs(result.sse.items[1].usage_is_object, True)
        self.assertTrue(result.sse.items[2].done)
        self.assertNotIn("response_headers", state.requests["compact-sse"])

    def test_client_closed_discards_incomplete_sse_tail(self) -> None:
        state = self.validate(
            self.records([b'data: {"id":"unfinished"'], outcome="client_closed")
        )
        result = state.completed_results["compact-sse"]
        assert result.sse is not None
        self.assertEqual(result.sse.items, ())

    def test_eof_rejects_incomplete_sse_tail(self) -> None:
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "compact SSE data object"
        ):
            self.validate(self.records([b'data: {"id":"unfinished"']))

    def test_empty_sse_body_chunk_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "SSE body chunk is empty"
        ):
            self.validate(self.records([b""]))

    def test_sse_usage_shape_is_retained_without_raw_objects(self) -> None:
        parser = VALIDATOR._CompactSseParser()
        payloads: tuple[dict[str, Any], ...] = (
            {"choices": []},
            {"choices": [], "usage": {}},
            {"choices": [], "usage": "not-an-object"},
        )
        raw = b"".join(
            b"data: " + compact_json(payload).encode("ascii") + b"\n\n"
            for payload in payloads
        )
        parser.feed(raw, 0, 1)
        items = parser.finish(allow_incomplete=False).items
        self.assertEqual(
            [
                (item.usage_present, item.usage_is_object, item.completion_tokens)
                for item in items
            ],
            [(False, None, None), (True, True, None), (True, False, None)],
        )

    def test_http_case_and_request_key_are_bounded(self) -> None:
        for field in ("case_id", "request_key"):
            with self.subTest(field=field):
                records = self.records([b"data: [DONE]\n\n"])
                oversized = "x" * (VALIDATOR.MAX_SESSION_IDENTIFIER_BYTES + 1)
                for record in records:
                    if field == "case_id" or "request_key" in record:
                        record[field] = oversized
                with self.assertRaisesRegex(
                    VALIDATOR.ValidationError, "UTF-8 byte bound"
                ):
                    self.validate(records)

    def test_http_response_headers_are_bounded(self) -> None:
        cases = {
            "name": [
                ["x" * (VALIDATOR.MAX_HTTP_HEADER_NAME_BYTES + 1), "value"],
                ["Content-Type", "text/event-stream"],
            ],
            "value": [
                ["Content-Type", "text/event-stream"],
                ["X-Test", "x" * (VALIDATOR.MAX_HTTP_HEADER_VALUE_BYTES + 1)],
            ],
            "aggregate": [
                ["Content-Type", "text/event-stream"],
                *[
                    [f"X-Test-{index}", "x" * VALIDATOR.MAX_HTTP_HEADER_VALUE_BYTES]
                    for index in range(8)
                ],
            ],
        }
        for name, headers in cases.items():
            with self.subTest(name=name):
                records = self.records([b"data: [DONE]\n\n"])
                start = next(
                    record
                    for record in records
                    if record["record_type"] == "http_response_start"
                )
                start["headers"] = headers
                with self.assertRaisesRegex(
                    VALIDATOR.ValidationError,
                    "UTF-8 byte bound|aggregate byte bound",
                ):
                    self.validate(records)

        records = self.records([b"data: [DONE]\n\n"])
        start = next(
            record
            for record in records
            if record["record_type"] == "http_response_start"
        )
        start["headers"] = [
            ["Content-Type", "text/event-stream"],
            *[
                [f"X-{index}", "x"]
                for index in range(VALIDATOR.MAX_HTTP_RESPONSE_HEADER_COUNT)
            ],
        ]
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "count bound"):
            self.validate(records)

    def test_sse_item_count_is_bounded(self) -> None:
        parser = VALIDATOR._CompactSseParser()
        raw = b"data: {}\n\n" * (VALIDATOR.MAX_SSE_ITEMS_PER_RESPONSE + 1)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "item count exceeds"):
            parser.feed(raw, 0, 1)

    def test_session_sse_item_count_is_bounded(self) -> None:
        original = VALIDATOR.MAX_SESSION_SSE_ITEMS
        VALIDATOR.MAX_SESSION_SSE_ITEMS = 2
        try:
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "session SSE item count exceeds"
            ):
                self.validate(
                    self.records(
                        [
                            b"data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                        ]
                    )
                )
        finally:
            VALIDATOR.MAX_SESSION_SSE_ITEMS = original

    def test_sse_finish_reason_is_bounded_before_retention(self) -> None:
        parser = VALIDATOR._CompactSseParser()
        payload = compact_json(
            {
                "choices": [
                    {"finish_reason": "x" * (VALIDATOR.MAX_SSE_FINISH_REASON_BYTES + 1)}
                ]
            }
        ).encode("ascii")
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "UTF-8 byte bound"):
            parser.feed(b"data: " + payload + b"\n\n", 0, 1)


class ApiContractHttpValidationTest(unittest.TestCase):
    def setUp(self):
        self.records = build_api_contract_http_records()

    def state(self, records: list[dict[str, Any]] | None = None) -> Any:
        state = VALIDATOR.HttpValidationState(
            fixture_seal=VALIDATOR.InputSeal(size=2, sha256=sha256_bytes(b"{}")),
            requests={},
            response_started=set(),
            response_ended=set(),
            bodies={},
            ordered_keys=[],
        )
        for index, record in enumerate(self.records if records is None else records):
            VALIDATOR._validate_http_record(record, f"API test record {index}", state)
        return state

    def validate(self, records: list[dict[str, Any]] | None = None) -> Any:
        state = self.state(records)
        return VALIDATOR.validate_api_contract_http(state)

    def assert_invalid(
        self, text: str, records: list[dict[str, Any]] | None = None
    ) -> None:
        with self.assertRaisesRegex(VALIDATOR.ValidationError, text):
            self.validate(records)

    def record(
        self, records: list[dict[str, Any]], case_id: str, record_type: str
    ) -> dict[str, Any]:
        return next(
            record
            for record in records
            if record["case_id"] == case_id and record["record_type"] == record_type
        )

    def test_exact_ten_case_contract_is_reconstructed(self):
        result = self.validate()
        self.assertEqual(
            result.case_ids,
            tuple(case.case_id for case in VALIDATOR.API_CONTRACT_CASES),
        )
        self.assertEqual(
            result.statuses,
            tuple(case.expected_status for case in VALIDATOR.API_CONTRACT_CASES),
        )
        self.assertEqual(len(result.request_keys), 10)
        self.assertEqual(len(result.cases), 10)
        self.assertEqual(
            result.cases[0],
            {
                "case_index": 1,
                "case_id": "models-valid",
                "method": "GET",
                "target": "/v1/models",
                "authorization_mode": "valid_bearer",
                "request_body_bytes": 0,
                "request_body_sha256": sha256_bytes(b""),
                "connect_completed_monotonic_ns": 1_000_000,
                "write_started_monotonic_ns": 1_000_001,
                "last_body_byte_sent_monotonic_ns": 1_000_002,
                "status": 200,
                "response_started_monotonic_ns": 1_000_003,
                "response_end_monotonic_ns": 1_000_005,
                "content_type": "application/json",
                "content_length": len(
                    api_contract_response_body(VALIDATOR.API_CONTRACT_CASES[0])
                ),
                "www_authenticate": [],
                "response_body_bytes": len(
                    api_contract_response_body(VALIDATOR.API_CONTRACT_CASES[0])
                ),
                "response_body_sha256": sha256_bytes(
                    api_contract_response_body(VALIDATOR.API_CONTRACT_CASES[0])
                ),
                "error": None,
            },
        )

    def test_api_bodies_are_released_after_compact_summary(self) -> None:
        state = self.state()
        result = VALIDATOR.validate_api_contract_http(state)
        self.assertEqual(len(result.cases), 10)
        self.assertFalse(state.bodies)
        self.assertEqual(len(state.completed_results), 10)
        for request in state.requests.values():
            self.assertNotIn("request_body", request)
            self.assertNotIn("response_body", request)
            self.assertIsInstance(request.get("api_response"), dict)
            self.assertIsInstance(request.get("response_headers"), tuple)
        self.assertEqual(
            result.cases[1]["error"],
            {
                "type": "invalid_request_error",
                "code": "invalid_api_key",
                "param": None,
                "message_utf8_bytes": len(
                    "The supplied API key is invalid.".encode("utf-8")
                ),
                "message_sha256": sha256_bytes(
                    "The supplied API key is invalid.".encode("utf-8")
                ),
            },
        )

    def test_validator_schedule_is_frozen_independently_of_the_gate_module(self):
        observed = tuple(
            (
                case.case_id,
                case.method,
                case.target,
                case.body,
                case.authorization_mode,
                case.expected_status,
                case.expected_code,
                case.expected_param,
                case.expected_message,
                case.expect_models,
            )
            for case in VALIDATOR.API_CONTRACT_CASES
        )
        expected = (
            (
                "models-valid",
                "GET",
                "/v1/models",
                b"",
                "valid_bearer",
                200,
                None,
                None,
                None,
                True,
            ),
            (
                "models-missing-auth",
                "GET",
                "/v1/models",
                b"",
                "missing",
                401,
                "invalid_api_key",
                None,
                "The supplied API key is invalid.",
                False,
            ),
            (
                "models-invalid-auth",
                "GET",
                "/v1/models",
                b"",
                "invalid_bearer",
                401,
                "invalid_api_key",
                None,
                "The supplied API key is invalid.",
                False,
            ),
            (
                "models-query",
                "GET",
                "/v1/models?x=1",
                b"",
                "valid_bearer",
                400,
                "invalid_request_error",
                None,
                "Query parameters are not supported.",
                False,
            ),
            (
                "chat-malformed-missing-auth",
                "POST",
                "/v1/chat/completions",
                b'{"broken":',
                "missing",
                401,
                "invalid_api_key",
                None,
                "The supplied API key is invalid.",
                False,
            ),
            (
                "chat-invalid-auth",
                "POST",
                "/v1/chat/completions",
                b'{"messages":[{"content":"API contract preflight","role":"user"}],'
                b'"model":"ullm-qwen3-14b-sq8"}',
                "invalid_bearer",
                401,
                "invalid_api_key",
                None,
                "The supplied API key is invalid.",
                False,
            ),
            (
                "chat-malformed-valid-auth",
                "POST",
                "/v1/chat/completions",
                b'{"broken":',
                "valid_bearer",
                400,
                "invalid_request_error",
                None,
                "The request body is not valid JSON.",
                False,
            ),
            (
                "chat-duplicate-key",
                "POST",
                "/v1/chat/completions",
                b'{"model":"ullm-qwen3-14b-sq8","model":"ullm-qwen3-14b-sq8",'
                b'"messages":[{"role":"user","content":"API contract preflight"}]}',
                "valid_bearer",
                400,
                "invalid_request_error",
                None,
                "The request body is not valid JSON.",
                False,
            ),
            (
                "chat-unsupported-n",
                "POST",
                "/v1/chat/completions",
                b'{"messages":[{"content":"API contract preflight","role":"user"}],'
                b'"model":"ullm-qwen3-14b-sq8","n":2}',
                "valid_bearer",
                400,
                "unsupported_parameter",
                "n",
                "The requested parameter is not supported.",
                False,
            ),
            (
                "chat-missing-model",
                "POST",
                "/v1/chat/completions",
                b'{"messages":[{"content":"API contract preflight","role":"user"}],'
                b'"model":"missing"}',
                "valid_bearer",
                404,
                "model_not_found",
                "model",
                "The requested model does not exist.",
                False,
            ),
        )
        self.assertEqual(observed, expected)

    def test_full_helper_rejects_an_absent_or_incomplete_schedule(self):
        self.assert_invalid("request count", [])
        self.assert_invalid("request count", self.records[:-4])

    def test_request_identity_order_authorization_and_body_are_exact(self):
        mutations = {
            "index": lambda request: request.update({"request_index": 2}),
            "case": lambda request: request.update({"case_id": "models-query"}),
            "target": lambda request: request.update({"target": "/v1/models?x=1"}),
            "authorization": lambda request: request["headers"].update(
                {"authorization_mode": "missing"}
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                records = deepcopy(self.records)
                mutation(self.record(records, "models-valid", "http_request"))
                self.assert_invalid(
                    "request identity, order, authorization, or body", records
                )

        records = deepcopy(self.records)
        request = self.record(records, "chat-invalid-auth", "http_request")
        replace_request_body(request, VALIDATOR.API_CONTRACT_MISSING_MODEL_BODY)
        self.assert_invalid("request identity, order, authorization, or body", records)

    def test_method_body_shape_is_phase_specific(self):
        records = deepcopy(self.records)
        request = self.record(records, "models-valid", "http_request")
        replace_request_body(request, b"{}")
        self.assert_invalid("method/body shape", records)

    def test_status_is_exact_for_every_case(self):
        records = deepcopy(self.records)
        self.record(records, "chat-missing-model", "http_response_start")["status"] = (
            400
        )
        self.assert_invalid("status differs", records)

    def test_response_protocol_headers_are_reconstructed(self):
        mutations = {
            "content-type": (
                "models-valid",
                lambda headers: headers.__setitem__(
                    0, ["Content-Type", "application/json; charset=utf-8"]
                ),
                "Content-Type",
            ),
            "content-length": (
                "models-valid",
                lambda headers: headers.__setitem__(1, ["Content-Length", "1"]),
                "Content-Length",
            ),
            "missing-authenticate": (
                "models-missing-auth",
                lambda headers: headers.__setitem__(
                    slice(None),
                    [pair for pair in headers if pair[0] != "WWW-Authenticate"],
                ),
                "WWW-Authenticate",
            ),
            "unexpected-authenticate": (
                "models-valid",
                lambda headers: headers.append(["WWW-Authenticate", "Bearer"]),
                "WWW-Authenticate",
            ),
            "retry-after": (
                "models-valid",
                lambda headers: headers.append(["Retry-After", "1"]),
                "Retry-After",
            ),
            "transfer-encoding": (
                "models-valid",
                lambda headers: headers.append(["Transfer-Encoding", "chunked"]),
                "Transfer-Encoding",
            ),
        }
        for name, (case_id, mutation, expected) in mutations.items():
            with self.subTest(name=name):
                records = deepcopy(self.records)
                start = self.record(records, case_id, "http_response_start")
                mutation(start["headers"])
                self.assert_invalid(expected, records)

    def test_model_list_is_exact(self):
        records = deepcopy(self.records)
        raw = api_contract_response_body(VALIDATOR.API_CONTRACT_CASES[0]).replace(
            VALIDATOR.API_CONTRACT_MODEL_ID.encode("ascii"), b"other-model"
        )
        replace_api_contract_response(records, "models-valid", raw)
        self.assert_invalid("model list differs", records)

    def test_error_envelope_message_type_code_and_param_are_exact(self):
        for field, replacement in (
            ("message", "different"),
            ("type", "authentication_error"),
            ("code", "different"),
            ("param", "authorization"),
        ):
            with self.subTest(field=field):
                records = deepcopy(self.records)
                body = json.loads(
                    api_contract_response_body(VALIDATOR.API_CONTRACT_CASES[1])
                )
                body["error"][field] = replacement
                replace_api_contract_response(
                    records,
                    "models-missing-auth",
                    compact_json(body).encode("utf-8"),
                )
                self.assert_invalid("error message, type, code, or param", records)

    def test_response_json_duplicate_keys_are_rejected(self):
        records = deepcopy(self.records)
        raw = (
            b'{"object":"list","object":"list","data":[{"id":"'
            + VALIDATOR.API_CONTRACT_MODEL_ID.encode("ascii")
            + b'","object":"model","owned_by":"ullm"}]}'
        )
        replace_api_contract_response(records, "models-valid", raw)
        self.assert_invalid("duplicate JSON key", records)

    def test_empty_api_response_chunk_is_rejected(self):
        records = deepcopy(self.records)
        replace_api_contract_response(records, "models-valid", b"")
        self.assert_invalid("body chunk is empty", records)


class ApiContractQuietCheckValidationTest(unittest.TestCase):
    def fixture(self):
        labels = [case.case_id for case in VALIDATOR.API_CONTRACT_CASES] + [
            "http-client-shutdown",
            "post-observer-close",
            "final-readiness-and-identity",
        ]
        observations = tuple(
            VALIDATOR.ApiJournalObservationData(
                phase="api_contract",
                case_id=f"api-journal-{index + 1:02d}",
                observation_index=index,
                journal_cursor=f"api-cursor-{index + 1:02d}",
                journal_monotonic_usec=100 + index,
                journal_pid=1200,
                message_utf8_bytes=len(f"message-{index}".encode()),
                message_sha256=sha256_bytes(f"message-{index}".encode()),
            )
            for index in range(13)
        )
        checks = tuple(
            VALIDATOR.LifecycleQuietCheckData(
                phase="api_contract",
                case_id=label,
                quiet_sequence=index,
                label=label,
                checked_monotonic_ns=200_000 + index,
                observer_open=index <= 10,
                observer_event_count=0,
                new_journal_record_count=1,
                journal_record_count=index + 1,
                journal_cursor=observations[index].journal_cursor,
            )
            for index, label in enumerate(labels)
        )
        http_results = tuple(
            SimpleNamespace(
                phase="api_contract",
                case_id=case.case_id,
                response_end_monotonic_ns=150_000 + index,
            )
            for index, case in enumerate(VALIDATOR.API_CONTRACT_CASES)
        )
        return checks, observations, http_results

    def test_exact_quiet_schedule_is_bound_to_complete_journal_observations(self):
        checks, observations, http_results = self.fixture()
        self.assertEqual(
            VALIDATOR.validate_api_contract_quiet_checks(
                checks, observations, http_results, 1200
            ),
            checks,
        )

    def test_quiet_schedule_rejects_missing_or_rebound_evidence(self):
        checks, observations, http_results = self.fixture()
        mutations = {
            "missing-check": checks[:-1],
            "rebound-cursor": checks[:5]
            + (dataclasses.replace(checks[5], journal_cursor="api-cursor-01"),)
            + checks[6:],
            "regressed-count": checks[:-1]
            + (
                dataclasses.replace(
                    checks[-1],
                    journal_record_count=12,
                    new_journal_record_count=0,
                    journal_cursor=observations[11].journal_cursor,
                ),
            ),
            "early-check": (
                dataclasses.replace(checks[0], checked_monotonic_ns=99_999),
            )
            + checks[1:],
        }
        for name, mutated in mutations.items():
            with self.subTest(name=name), self.assertRaises(VALIDATOR.ValidationError):
                VALIDATOR.validate_api_contract_quiet_checks(
                    mutated, observations, http_results, 1200
                )

    def test_global_journal_requires_the_complete_contiguous_observation_span(self):
        checks, observations, _http_results = self.fixture()

        def journal_record(cursor, monotonic, message):
            return {
                "__CURSOR": cursor,
                "__MONOTONIC_TIMESTAMP": str(monotonic),
                "_BOOT_ID": BOOT_ID,
                "_PID": "1200",
                "_SYSTEMD_UNIT": "ullm-openai.service",
                "PRIORITY": "6",
                "MESSAGE": message,
            }

        records = [
            journal_record(
                observation.journal_cursor,
                observation.journal_monotonic_usec,
                f"message-{index}",
            )
            for index, observation in enumerate(observations)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "service-journal.raw.jsonl"

            def write(values):
                path.write_text(
                    "".join(compact_json(value) + "\n" for value in values),
                    encoding="utf-8",
                )

            write(records)
            VALIDATOR.validate_service_journal(
                root,
                {},
                BOOT_ID,
                observations[-1].journal_cursor,
                checks,
                observations,
            )

            interrupted = list(records)
            interrupted.insert(
                5,
                journal_record("uncopied-api-cursor", 104, "uncopied API record"),
            )
            write(interrupted)
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "interrupts or reorders"
            ):
                VALIDATOR.validate_service_journal(
                    root,
                    {},
                    BOOT_ID,
                    observations[-1].journal_cursor,
                    checks,
                    observations,
                )


class CampaignIdentityValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "identity-bundle"
        self.root.mkdir()
        self.environment, self.model_identity = build_identity_documents()
        self.write_documents()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_documents(self) -> None:
        (self.root / "environment.json").write_bytes(
            identity_canonical(self.environment)
        )
        (self.root / "model-identity.json").write_bytes(
            identity_canonical(self.model_identity)
        )

    def refresh_source_sets(self) -> None:
        by_role = {item["role"]: item for item in self.environment["sources"]}
        self.environment["source_sets"] = {
            group: sha256_bytes(
                identity_canonical([by_role[role] for role in sorted(roles)])
            )
            for group, roles in VALIDATOR.EXPECTED_SOURCE_GROUPS.items()
        }

    def build_source_checkout(self) -> tuple[Path, str]:
        repo = Path(self.temporary.name) / "source-checkout"
        fixed_sources = {
            **VALIDATOR.EXPECTED_ORACLE_FILE_IDENTITIES,
            **VALIDATOR.EXPECTED_TTFT_FIXTURE_IDENTITIES,
        }
        for role, relative in VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS.items():
            path = repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = (
                (REPO_ROOT / relative).read_bytes()
                if role in fixed_sources
                else f"synthetic source {role}\n".encode("ascii")
            )
            path.write_bytes(raw)
        commands = (
            ("init", "-q"),
            ("config", "user.email", "source-test@example.invalid"),
            ("config", "user.name", "Source Test"),
            ("add", "."),
            ("commit", "-q", "-m", "source fixture"),
        )
        for arguments in commands:
            subprocess.run(
                ("git", "-C", str(repo), *arguments),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        commit = (
            subprocess.run(
                ("git", "-C", str(repo), "rev-parse", "HEAD"),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            .stdout.decode("ascii")
            .strip()
        )
        return repo, commit

    def validate(
        self, *, commit: str = GIT_COMMIT, worker_sha: str = WORKER_SHA256
    ) -> Any:
        return VALIDATOR.validate_campaign_identity(
            self.root,
            expected_commit=commit,
            expected_worker_binary_sha256=worker_sha,
        )

    def assert_invalid(self, text: str) -> None:
        with self.assertRaisesRegex(VALIDATOR.ValidationError, text):
            self.validate()

    def test_valid_identity_reconstructs_fixed_contract(self) -> None:
        result = self.validate()
        self.assertIsInstance(result, VALIDATOR.IdentityData)
        self.assertEqual(result.expected_commit, GIT_COMMIT)
        self.assertEqual(result.expected_worker_binary_sha256, WORKER_SHA256)
        self.assertEqual(
            result.model_worker["artifact_content_sha256"],
            VALIDATOR.EXPECTED_ARTIFACT_IDENTITY["content_sha256"],
        )
        self.assertEqual(
            result.environment_sha256,
            sha256_file(self.root / "environment.json"),
        )

    def test_source_contract_map_and_groups_match_producer(self) -> None:
        generator_path = REPO_ROOT / "tools" / "sq8_full_campaign_identity.py"
        spec = importlib.util.spec_from_file_location(
            "identity_source_contract_parity", generator_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.assertEqual(
            VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS,
            module.SOURCE_ROLE_PATHS,
        )
        self.assertEqual(
            VALIDATOR.EXPECTED_SOURCE_GROUPS,
            module.SOURCE_GROUPS,
        )
        self.assertEqual(
            VALIDATOR.EXPECTED_TTFT_FIXTURE_IDENTITIES,
            module.TTFT_FIXTURE_IDENTITIES,
        )

    def test_source_contract_rejects_unknown_duplicate_and_missing_roles(self) -> None:
        self.assertEqual(len(VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS), 63)
        VALIDATOR._validate_identity_source_contract()
        duplicate_paths = dict(VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS)
        duplicate_paths["campaign_views"] = duplicate_paths["campaign_renderer"]
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "paths are not unique"):
            VALIDATOR._validate_identity_source_contract(
                duplicate_paths, VALIDATOR.EXPECTED_SOURCE_GROUPS
            )

        mutations = []
        unknown = dict(VALIDATOR.EXPECTED_SOURCE_GROUPS)
        unknown["campaign"] = (*unknown["campaign"], "unknown_source")
        mutations.append(unknown)
        duplicate = dict(VALIDATOR.EXPECTED_SOURCE_GROUPS)
        duplicate["campaign"] = (*duplicate["campaign"], duplicate["campaign"][0])
        mutations.append(duplicate)
        missing = dict(VALIDATOR.EXPECTED_SOURCE_GROUPS)
        missing["all"] = missing["all"][:-1]
        mutations.append(missing)
        unclassified = dict(VALIDATOR.EXPECTED_SOURCE_GROUPS)
        unclassified["fixture"] = tuple(
            role for role in unclassified["fixture"] if role != "fixture_ttft_p3584"
        )
        mutations.append(unclassified)
        for groups in mutations:
            with (
                self.subTest(groups=groups),
                self.assertRaises(VALIDATOR.ValidationError),
            ):
                VALIDATOR._validate_identity_source_contract(
                    VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS, groups
                )

    def test_environment_source_set_rejects_missing_extra_and_ttft_mutation(
        self,
    ) -> None:
        pristine = deepcopy(self.environment)
        mutations: tuple[Callable[[dict[str, Any]], object], ...] = (
            lambda value: value["sources"].pop(),
            lambda value: value["sources"].append(
                {
                    "role": "extra",
                    "path": "tools/extra.py",
                    "bytes": 1,
                    "sha256": sha256_bytes(b"x"),
                }
            ),
            lambda value: next(
                item
                for item in value["sources"]
                if item["role"] == "fixture_ttft_p0032"
            ).__setitem__("sha256", "0" * 64),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.environment = deepcopy(pristine)
                mutation(self.environment)
                self.write_documents()
                with self.assertRaises(VALIDATOR.ValidationError):
                    self.validate()

    def test_trusted_git_blob_and_worktree_sources_are_streamed_and_bound(
        self,
    ) -> None:
        repo, commit = self.build_source_checkout()
        self.environment["git"]["commit"] = commit
        self.write_documents()
        identity = self.validate(commit=commit)
        result = VALIDATOR.validate_campaign_source_checkout(identity, repo_root=repo)
        self.assertEqual(result.git_commit, commit)
        self.assertEqual(result.source_count, len(VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS))
        self.assertEqual(result.all_source_sha256, identity.source_sets["all"])

        (repo / "unrelated-dirty-file").write_bytes(b"unrelated\n")
        VALIDATOR.validate_campaign_source_checkout(identity, repo_root=repo)

    def test_trusted_git_blob_rejects_a_recorded_worktree_substitution(self) -> None:
        repo, commit = self.build_source_checkout()
        role = "gate_api_contract"
        path = repo / VALIDATOR.EXPECTED_SOURCE_ROLE_PATHS[role]
        replacement = b"substituted gate source\n"
        path.write_bytes(replacement)
        source = next(
            item for item in self.environment["sources"] if item["role"] == role
        )
        source["bytes"] = len(replacement)
        source["sha256"] = sha256_bytes(replacement)
        self.refresh_source_sets()
        self.environment["git"]["commit"] = commit
        self.write_documents()
        identity = self.validate(commit=commit)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "trusted Git blob"):
            VALIDATOR.validate_campaign_source_checkout(identity, repo_root=repo)

    def test_identity_json_must_be_canonical_and_forbid_passed(self):
        path = self.root / "environment.json"
        path.write_text(json.dumps(self.environment, indent=2), encoding="utf-8")
        self.assert_invalid("canonical identity JSON")
        self.write_documents()
        self.environment["passed"] = True
        self.write_documents()
        self.assert_invalid("forbidden key.*passed")

    def test_source_order_aggregate_and_deployment_copy_are_bound(self):
        mutations = (
            (
                "bytewise-sorted",
                lambda value: value["sources"].reverse(),
            ),
            (
                "source aggregate worker",
                lambda value: value["source_sets"].__setitem__("worker", "0" * 64),
            ),
            (
                "effective deployment",
                lambda value: value["deployment"]["service_unit_file"].__setitem__(
                    "bytes",
                    value["deployment"]["service_unit_file"]["bytes"] + 1,
                ),
            ),
        )
        pristine = deepcopy(self.environment)
        for text, mutation in mutations:
            with self.subTest(text=text):
                self.environment = deepcopy(pristine)
                mutation(self.environment)
                self.write_documents()
                self.assert_invalid(text)

    def test_frozen_gpu_systemd_and_openwebui_values_are_exact(self):
        mutations = (
            (
                "tool or host kernel identity",
                lambda value: value["host"]["tools"].__setitem__("systemd_major", 254),
            ),
            (
                "frozen GPU identity",
                lambda value: value["host"]["gpu"].__setitem__("index", 1),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__("version", "0.9.4-ullm.2"),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__(
                    "source_revision", "e" * 40
                ),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__(
                    "base_image_digest", "sha256:" + "a" * 64
                ),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__(
                    "base_image_id", "sha256:" + "b" * 64
                ),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__(
                    "derived_image_id", "sha256:" + "c" * 64
                ),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__(
                    "patched_middleware_sha256", "d" * 64
                ),
            ),
            (
                "OpenWebUI source or network identity",
                lambda value: value["openwebui"].__setitem__("network_id", "e" * 64),
            ),
        )
        pristine = deepcopy(self.environment)
        for text, mutation in mutations:
            with self.subTest(text=text):
                self.environment = deepcopy(pristine)
                mutation(self.environment)
                self.write_documents()
                self.assert_invalid(text)

    def test_trusted_commit_and_worker_anchors_are_required(self):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "trusted CLI anchor"):
            self.validate(commit="c" * 40)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "worker or runtime"):
            self.validate(worker_sha="c" * 64)

    def test_promotion_receipt_and_fixed_product_are_recomputed(self):
        mutations = (
            (
                "promotion receipt SHA-256",
                lambda value: value["promotion_validation"].__setitem__(
                    "result_sha256", "0" * 64
                ),
            ),
            (
                "promotion validation state",
                lambda value: value["promotion_validation"].__setitem__(
                    "full_payloads", False
                ),
            ),
            (
                "fixed artifact identity",
                lambda value: value["product"]["artifact"].__setitem__(
                    "payload_bytes",
                    value["product"]["artifact"]["payload_bytes"] + 1,
                ),
            ),
            (
                "fixed package identity",
                lambda value: value["product"]["package"].__setitem__(
                    "payload_count",
                    value["product"]["package"]["payload_count"] - 1,
                ),
            ),
        )
        pristine = deepcopy(self.model_identity)
        for text, mutation in mutations:
            with self.subTest(text=text):
                self.model_identity = deepcopy(pristine)
                mutation(self.model_identity)
                self.write_documents()
                self.assert_invalid(text)

    def test_tokenizer_chat_template_and_oracle_are_fixed(self):
        mutations = (
            (
                "tokenizer file tokenizer.json",
                lambda value: value["tokenizer"]["files"][4].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "tokenizer chat template",
                lambda value: value["tokenizer"]["chat_template"].__setitem__(
                    "utf8_bytes", 4167
                ),
            ),
            (
                "oracle runtime_oracle_validation",
                lambda value: value["oracle"]["runtime_oracle_validation"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "vLLM oracle identity",
                lambda value: value["oracle"]["vllm_identity"].__setitem__(
                    "max_num_seqs", 2
                ),
            ),
        )
        pristine = deepcopy(self.model_identity)
        for text, mutation in mutations:
            with self.subTest(text=text):
                self.model_identity = deepcopy(pristine)
                mutation(self.model_identity)
                self.write_documents()
                self.assert_invalid(text)

    def test_environment_and_model_cross_document_bindings_are_required(self):
        pristine_environment = deepcopy(self.environment)
        pristine_model = deepcopy(self.model_identity)
        mutations = (
            (
                "worker or runtime binding",
                lambda: self.model_identity["worker"].__setitem__(
                    "source_sha256", "0" * 64
                ),
            ),
            (
                "worker or runtime binding",
                lambda: self.model_identity["tokenizer"].__setitem__(
                    "root", "/different/tokenizer"
                ),
            ),
            (
                "promotion validator source",
                lambda: self.model_identity["promotion_validation"].__setitem__(
                    "validator_source_sha256", "0" * 64
                ),
            ),
            (
                "worker or runtime binding",
                lambda: self.environment["service"]["worker"].__setitem__(
                    "executable_sha256", "0" * 64
                ),
            ),
        )
        for text, mutation in mutations:
            with self.subTest(text=text):
                self.environment = deepcopy(pristine_environment)
                self.model_identity = deepcopy(pristine_model)
                mutation()
                self.write_documents()
                self.assert_invalid(text)

    def test_identity_data_binds_header_initial_probe_and_run_end(self):
        identity = self.validate()
        source_inputs = sorted(
            (
                {key: source[key] for key in ("path", "bytes", "sha256")}
                for source in identity.source_by_role.values()
            ),
            key=lambda item: item["path"].encode("utf-8"),
        )
        identity.validate_header_source_inputs(source_inputs)
        missing_source = deepcopy(source_inputs[1:])
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "lacks an exact campaign source"
        ):
            identity.validate_header_source_inputs(missing_source)
        changed_source = deepcopy(source_inputs)
        changed_source[0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "lacks an exact campaign source"
        ):
            identity.validate_header_source_inputs(changed_source)

        header = {
            "started_utc": "2026-07-11T12:00:01Z",
            "boot_id": BOOT_ID,
            "identities": {
                "environment_file": "environment.json",
                "environment_sha256": identity.environment_sha256,
                "model_identity_file": "model-identity.json",
                "model_identity_sha256": identity.model_identity_sha256,
                "openwebui": {
                    key: identity.openwebui[key]
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
                },
                "docker_network_id": identity.openwebui["network_id"],
                "gateway_source_sha256": identity.source_sets["gateway"],
                "worker_source_sha256": identity.source_sets["worker"],
                "worker_binary_sha256": WORKER_SHA256,
            },
        }
        identity.validate_session_header(header)
        broken_header = deepcopy(header)
        broken_header["identities"]["gateway_source_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "header identities differ"
        ):
            identity.validate_session_header(broken_header)

        service = identity.service
        probe = {
            "service_active": True,
            "ready_http_status": 200,
            "control_group": service["control_group"],
            "gateway_pid": service["gateway"]["pid"],
            "gateway_starttime_ticks": service["gateway"]["starttime_ticks"],
            "worker_pid": service["worker"]["pid"],
            "worker_starttime_ticks": service["worker"]["starttime_ticks"],
            "n_restarts": service["n_restarts"],
        }
        identity.validate_initial_probe(probe)
        broken_probe = deepcopy(probe)
        broken_probe["worker_pid"] += 1
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "initial lifecycle"):
            identity.validate_initial_probe(broken_probe)

        run_end = {
            "final_git_commit": GIT_COMMIT,
            "final_git_status_raw": "",
            "final_git_status_sha256": sha256_bytes(b""),
        }
        identity.validate_run_end(run_end)
        broken_end = deepcopy(run_end)
        broken_end["final_git_status_raw"] = " M tracked.py\n"
        broken_end["final_git_status_sha256"] = sha256_bytes(
            broken_end["final_git_status_raw"].encode()
        )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "Git status differs"):
            identity.validate_run_end(broken_end)


class ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"
        EvidenceBuilder(self.root).build()

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self):
        return VALIDATOR.validate_phase1(
            self.root,
            expected_commit=GIT_COMMIT,
            expected_worker_binary_sha256=WORKER_SHA256,
        )

    def session(self) -> Any:
        VALIDATOR.validate_bundle_layout(self.root)
        VALIDATOR.validate_sha256sums(self.root)
        matrix = VALIDATOR.validate_matrix(self.root)
        return VALIDATOR.validate_session(
            self.root,
            matrix,
            GIT_COMMIT,
            WORKER_SHA256,
        )

    def test_optional_full_identity_crosschecks_header_probe_and_run_end(self) -> None:
        VALIDATOR.validate_bundle_layout(self.root)
        matrix = VALIDATOR.validate_matrix(self.root)
        identity = mock.create_autospec(VALIDATOR.IdentityData, instance=True)
        VALIDATOR.validate_session(
            self.root,
            matrix,
            GIT_COMMIT,
            WORKER_SHA256,
            identity,
        )
        identity.validate_session_header.assert_called_once()
        identity.validate_header_source_inputs.assert_called_once()
        identity.validate_initial_probe.assert_called_once()
        identity.validate_run_end.assert_called_once()

        rejected = mock.create_autospec(VALIDATOR.IdentityData, instance=True)
        rejected.validate_session_header.side_effect = VALIDATOR.ValidationError(
            "injected header identity rejection"
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "injected header identity rejection"
        ):
            VALIDATOR.validate_session(
                self.root,
                matrix,
                GIT_COMMIT,
                WORKER_SHA256,
                rejected,
            )
        rejected.validate_header_source_inputs.assert_not_called()

    def assert_invalid(self, text: str):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, text):
            self.validate()

    def test_valid_synthetic_bundle_recomputes_resource_gates(self):
        result = self.validate()
        self.assertEqual(result["release_status"], "incomplete")
        self.assertTrue(result["phase1_validated"])
        self.assertEqual(result["raw_counts"]["resource_samples"], 610)
        self.assertEqual(result["resource_segments"]["normal"]["point_count"], 100)
        self.assertEqual(result["resource_segments"]["restart"]["point_count"], 20)
        self.assertGreater(len(result["unimplemented_release_gates"]), 0)

    def test_legacy_phase1_session_keeps_bounded_compact_results(self) -> None:
        session = self.session()
        self.assertIsNone(session.full_campaign_order)
        self.assertIsNone(session.api_contract)
        self.assertFalse(VALIDATOR._claims_full_campaign(session.raw_order_projection))
        self.assertEqual(
            len(session.raw_order_projection), sum(session.record_counts.values())
        )
        self.assertTrue(
            all("body_base64" not in record for record in session.raw_order_projection)
        )
        self.assertEqual(len(session.http_results), 143)
        self.assertTrue(
            all(
                "request_body" not in request and "response_body" not in request
                for request in session.http_requests.values()
            )
        )
        first = session.http_results[0]
        self.assertEqual(first.status, 200)
        self.assertEqual(first.outcome, "eof")
        self.assertIsNotNone(first.sse)
        assert first.sse is not None
        self.assertEqual(len(first.sse.items), 3)

    def test_session_record_count_is_bounded(self) -> None:
        original = VALIDATOR.MAX_SESSION_RECORDS
        VALIDATOR.MAX_SESSION_RECORDS = 1
        try:
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "record-count bound"
            ):
                self.session()
        finally:
            VALIDATOR.MAX_SESSION_RECORDS = original

    def test_fraction_percentile_uses_linear_interpolation(self):
        self.assertEqual(
            VALIDATOR.percentile([0, 10, 20, 30], Fraction(1, 2)), Fraction(15)
        )
        self.assertEqual(
            VALIDATOR.percentile([0, 10, 20, 30], Fraction(19, 20)), Fraction(57, 2)
        )

    def test_duplicate_json_key_is_rejected_after_hashes_match(self):
        path = self.root / "soak-resources.raw.jsonl"
        raw = path.read_text(encoding="utf-8")
        raw = raw.replace(
            '"service_unit":"ullm-openai.service"',
            '"service_unit":"ullm-openai.service","service_unit":"ullm-openai.service"',
            1,
        )
        path.write_text(raw, encoding="utf-8")
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("duplicate JSON key")

    def test_nonfinite_json_number_is_rejected_after_hashes_match(self):
        path = self.root / "soak-resources.raw.jsonl"
        raw = path.read_text(encoding="utf-8")
        raw = raw.replace(
            '"memory_current_bytes":1000000000', '"memory_current_bytes":NaN', 1
        )
        path.write_text(raw, encoding="utf-8")
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("non-finite")

    def test_invalid_utf8_is_rejected_after_hashes_match(self):
        path = self.root / "soak-resources.raw.jsonl"
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"service_unit"', b'"service_\xffunit"', 1))
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("strict UTF-8")

    def test_boolean_is_not_accepted_as_integer(self):
        def mutate(records):
            records[2]["sample_index"] = True

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("must be an integer")

    def test_missing_resource_record_breaks_exact_state_machine(self):
        def mutate(records):
            del records[10]

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("state machine|record count")

    def test_resource_request_id_must_correlate_to_gateway_release(self):
        def mutate(records):
            point = next(
                record
                for record in records
                if record.get("segment") == "normal"
                and record.get("phase") == "post_release"
                and record.get("request_index") == 1
            )
            target_id = point["request_id"]
            for record in records:
                if record.get("request_id") == target_id:
                    record["request_id"] = "req-unmatched"

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("release order differs")

    def test_gateway_journal_pid_must_match_probe_epoch(self):
        session_path = self.root / "raw-session-results.jsonl"
        session = [json.loads(line) for line in session_path.read_text().splitlines()]
        target = next(
            record
            for record in session
            if record.get("record_type") == "gateway_event"
            and record["phase"] == "resource_normal"
        )
        target["journal_pid"] = 1299
        cursor = target["journal_cursor"]
        session_path.write_text(
            "".join(compact_json(record) + "\n" for record in session),
            encoding="utf-8",
        )
        journal_path = self.root / "service-journal.raw.jsonl"
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        next(record for record in journal if record["__CURSOR"] == cursor)["_PID"] = (
            "1299"
        )
        journal_path.write_text(
            "".join(compact_json(record) + "\n" for record in journal),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("journal PID differs from its lifecycle probe epoch")

    def test_probe_gateway_identity_must_match_resource_samples(self):
        session_path = self.root / "raw-session-results.jsonl"
        session = [json.loads(line) for line in session_path.read_text().splitlines()]
        normal_probe = next(
            record
            for record in session
            if record.get("record_type") == "lifecycle_probe"
            and record.get("probe") == "normal-segment-start"
        )
        normal_probe["gateway_pid"] = 1299
        normal_cursors = set()
        for record in session:
            if (
                record.get("record_type") == "gateway_event"
                and record.get("phase") == "resource_normal"
            ):
                record["journal_pid"] = 1299
                normal_cursors.add(record["journal_cursor"])
        session_path.write_text(
            "".join(compact_json(record) + "\n" for record in session),
            encoding="utf-8",
        )
        journal_path = self.root / "service-journal.raw.jsonl"
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        for record in journal:
            if record["__CURSOR"] in normal_cursors:
                record["_PID"] = "1299"
        journal_path.write_text(
            "".join(compact_json(record) + "\n" for record in journal),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("normal resource identity differs from its lifecycle probe")

    def test_old_gateway_event_cannot_exceed_restart_ready_boundary(self):
        evidence = VALIDATOR.GatewayEvidence(
            cursor="cursor",
            journal_monotonic_usec=1,
            journal_pid=1200,
            message="message",
            message_sha256="a" * 64,
            event={"event": "request_released", "observed_monotonic_ns": 101},
            phase="resource_normal",
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "exceeds the post-header restart boundary"
        ):
            VALIDATOR._validate_gateway_event_pids(
                {"cursor": evidence}, 1200, 2200, 100, 200
            )

    def test_post_header_phase_cannot_hide_a_late_restart_gateway_event(self):
        evidence = VALIDATOR.GatewayEvidence(
            cursor="cursor",
            journal_monotonic_usec=1,
            journal_pid=2200,
            message="message",
            message_sha256="a" * 64,
            event={"event": "request_released", "observed_monotonic_ns": 201},
            phase="post_header_failure",
        )
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "exceeds its lifecycle phase boundary"
        ):
            VALIDATOR._validate_gateway_event_pids(
                {"cursor": evidence}, 1200, 2200, 100, 200
            )

    def test_worker_fatal_is_only_allowed_from_planned_old_gateway_boundary(self):
        evidence = VALIDATOR.GatewayEvidence(
            cursor="cursor",
            journal_monotonic_usec=1,
            journal_pid=2200,
            message="message",
            message_sha256="a" * 64,
            event={"event": "worker_fatal", "observed_monotonic_ns": 150},
            phase="post_header_failure",
        )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "sole planned"):
            VALIDATOR._validate_gateway_event_pids(
                {"cursor": evidence}, 1200, 2200, 100, 200
            )

    def test_resource_sampling_body_is_reconstructed(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "normal-measured-005"
            )
            body = json.loads(base64.b64decode(target["body_base64"]))
            body["temperature"] = 0
            replace_request_body(
                target,
                json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("resource sampling settings differ")

    def test_resource_seed_requires_an_integer_json_number(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "normal-measured-005"
            )
            body = json.loads(base64.b64decode(target["body_base64"]))
            body["seed"] = 5.0
            replace_request_body(
                target,
                json.dumps(body, separators=(",", ":")).encode("utf-8"),
            )

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("body.seed must be an integer")

    def test_resource_max_tokens_requires_an_integer_json_number(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "normal-measured-005"
            )
            body = json.loads(base64.b64decode(target["body_base64"]))
            body["max_tokens"] = 2.0
            replace_request_body(
                target,
                json.dumps(body, separators=(",", ":")).encode("utf-8"),
            )

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("body.max_tokens must be an integer")

    def test_context_overflow_negative_body_semantics_are_reconstructed(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "negative-after-025-context_overflow_1"
            )
            replace_request_body(
                target,
                compact_json(
                    {
                        "model": "Qwen3-14B-SQ8",
                        "messages": [{"role": "user", "content": "x"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": 2,
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 0,
                    }
                ).encode("utf-8"),
            )

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("context-overflow request shape differs")

    def test_malformed_negative_must_not_be_valid_json(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "negative-after-050-malformed_json"
            )
            replace_request_body(target, b"{}")

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("must contain malformed JSON")

    def test_malformed_negative_rejects_syntactically_valid_duplicate_keys(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
                and record.get("case_id") == "negative-after-050-malformed_json"
            )
            replace_request_body(target, b'{"duplicate":1,"duplicate":2}')

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("must contain malformed JSON")

    def test_negative_interval_rejects_admission_hidden_under_another_phase(self):
        session_path = self.root / "raw-session-results.jsonl"
        session = [json.loads(line) for line in session_path.read_text().splitlines()]
        negative_index = next(
            index
            for index, record in enumerate(session)
            if record.get("record_type") == "http_response_end"
            and record.get("case_id") == "negative-after-025-context_overflow_1"
        )
        negative_end = session[negative_index]["observed_monotonic_ns"]
        next_request = next(
            record
            for record in session[negative_index + 1 :]
            if record.get("record_type") == "http_request"
        )
        base = next_request["last_body_byte_sent_monotonic_ns"]
        self.assertGreater(base, negative_end)
        shifted_journal = {}
        for record in session:
            if (
                record.get("record_type") == "gateway_event"
                and record.get("case_id") == next_request["case_id"]
            ):
                record["event"]["observed_monotonic_ns"] += 1
                message = compact_json(record["event"])
                record["message"] = message
                record["message_sha256"] = sha256_bytes(message.encode("utf-8"))
                record["journal_monotonic_usec"] = (
                    record["event"]["observed_monotonic_ns"] // 1000
                )
                shifted_journal[record["journal_cursor"]] = (
                    message,
                    record["journal_monotonic_usec"],
                )
        request_id = "req-hidden-negative-admission"
        completion_id = "chatcmpl-hidden-negative-admission"
        events = [
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_admitted",
                "observed_monotonic_ns": base,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 1,
                "max_completion_tokens": 1,
            },
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_started",
                "observed_monotonic_ns": base,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "prompt_tokens": 1,
                "admit_to_start_ns": 0,
            },
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_first_token",
                "observed_monotonic_ns": base,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "completion_tokens": 1,
            },
            {
                "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
                "event": "request_released",
                "observed_monotonic_ns": base,
                "request_id": request_id,
                "completion_id": completion_id,
                "stream": True,
                "outcome": "length",
                "cancel_reason": None,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "reset_complete": True,
                "admit_to_start_ns": 0,
                "start_to_release_ns": 0,
                "admit_to_release_ns": 0,
            },
        ]
        gateway_records = []
        journal_records = []
        boot_id = session[0]["boot_id"]
        for index, event in enumerate(events):
            cursor = f"hidden-negative-cursor-{index}"
            message = "INFO:     " + compact_json(event)
            gateway_records.append(
                {
                    "schema_version": VALIDATOR.SESSION_SCHEMA,
                    "record_type": "gateway_event",
                    "sequence": 0,
                    "phase": "openwebui",
                    "case_id": "hidden-negative-admission",
                    "journal_cursor": cursor,
                    "journal_monotonic_usec": event["observed_monotonic_ns"] // 1000,
                    "journal_pid": 1200,
                    "message": message,
                    "message_sha256": hashlib.sha256(message.encode()).hexdigest(),
                    "event": event,
                }
            )
            journal_records.append(
                {
                    "__CURSOR": cursor,
                    "__MONOTONIC_TIMESTAMP": str(
                        event["observed_monotonic_ns"] // 1000
                    ),
                    "_BOOT_ID": boot_id,
                    "_PID": "1200",
                    "_SYSTEMD_UNIT": "ullm-openai.service",
                    "PRIORITY": "6",
                    "MESSAGE": message,
                }
            )
        session[negative_index + 1 : negative_index + 1] = gateway_records
        for sequence, record in enumerate(session):
            record["sequence"] = sequence
        counts = {}
        for record in session:
            counts[record["record_type"]] = counts.get(record["record_type"], 0) + 1
        session[-1]["record_counts"] = counts
        session_path.write_text(
            "".join(compact_json(record) + "\n" for record in session),
            encoding="utf-8",
        )

        journal_path = self.root / "service-journal.raw.jsonl"
        journal = [json.loads(line) for line in journal_path.read_text().splitlines()]
        for record in journal:
            replacement = shifted_journal.get(record["__CURSOR"])
            if replacement is not None:
                record["MESSAGE"] = replacement[0]
                record["__MONOTONIC_TIMESTAMP"] = str(replacement[1])
        journal.extend(journal_records)
        journal.sort(key=lambda record: int(record["__MONOTONIC_TIMESTAMP"]))
        journal_path.write_text(
            "".join(compact_json(record) + "\n" for record in journal),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid(
            "negative resource request interval contains a worker admission"
        )

    def test_resource_metric_window_rejects_foreign_lifecycle_trace(self):
        resources = [
            json.loads(line)
            for line in (self.root / "soak-resources.raw.jsonl")
            .read_text()
            .splitlines()
        ]
        final_sample = next(
            record
            for record in resources
            if record.get("record_type") == "resource_sample"
            and record.get("segment") == "normal"
            and record.get("request_index") == 1
            and record.get("sample_index") == 4
        )
        session = [
            json.loads(line)
            for line in (self.root / "raw-session-results.jsonl")
            .read_text()
            .splitlines()
        ]
        next_request = next(
            record
            for record in session
            if record.get("record_type") == "http_request"
            and record.get("case_id") == "normal-measured-002"
        )
        base = final_sample["sample_monotonic_ns"] + 1_000_000
        self.assertLess(
            base + 3_000_000, next_request["connect_completed_monotonic_ns"]
        )
        request_id = "req-foreign-resource-gap"
        completion_id = "chatcmpl-foreign-resource-gap"
        common = {
            "schema_version": VALIDATOR.LIFECYCLE_SCHEMA,
            "request_id": request_id,
            "completion_id": completion_id,
            "stream": True,
        }
        events = [
            {
                **common,
                "event": "request_admitted",
                "observed_monotonic_ns": base,
                "prompt_tokens": 1,
                "max_completion_tokens": 1,
            },
            {
                **common,
                "event": "request_started",
                "observed_monotonic_ns": base + 1_000_000,
                "prompt_tokens": 1,
                "admit_to_start_ns": 1_000_000,
            },
            {
                **common,
                "event": "request_first_token",
                "observed_monotonic_ns": base + 2_000_000,
                "completion_tokens": 1,
            },
            {
                **common,
                "event": "request_released",
                "observed_monotonic_ns": base + 3_000_000,
                "outcome": "length",
                "cancel_reason": None,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "reset_complete": True,
                "admit_to_start_ns": 1_000_000,
                "start_to_release_ns": 2_000_000,
                "admit_to_release_ns": 3_000_000,
            },
        ]
        insert_gateway_trace_before_case(
            self.root,
            "normal-measured-002",
            "openwebui",
            "foreign-resource-gap",
            1200,
            events,
        )
        self.assert_invalid("resource metric window contains a foreign lifecycle trace")

    def test_resource_metric_window_rejects_foreign_http_exchange(self):
        resources = [
            json.loads(line)
            for line in (self.root / "soak-resources.raw.jsonl")
            .read_text()
            .splitlines()
        ]
        final_sample = next(
            record
            for record in resources
            if record.get("record_type") == "resource_sample"
            and record.get("segment") == "normal"
            and record.get("request_index") == 1
            and record.get("sample_index") == 4
        )
        session_path = self.root / "raw-session-results.jsonl"
        session = [json.loads(line) for line in session_path.read_text().splitlines()]
        insert_at = next(
            index
            for index, record in enumerate(session)
            if record.get("record_type") == "http_request"
            and record.get("case_id") == "normal-measured-002"
        )
        next_request = session[insert_at]
        base = final_sample["sample_monotonic_ns"] + 1_000_000
        self.assertLess(base + 5, next_request["connect_completed_monotonic_ns"])
        request_body = b"{}"
        response_body = b"{}"
        key = "foreign-resource-gap-http"
        common = {
            "schema_version": VALIDATOR.SESSION_SCHEMA,
            "sequence": 0,
            "phase": "api_contract",
            "case_id": "foreign-resource-gap-http",
        }
        records = [
            {
                **common,
                "record_type": "http_request",
                "request_index": 0,
                "request_key": key,
                "method": "POST",
                "target": "/v1/chat/completions",
                "headers": {
                    "content_type": "application/json",
                    "content_length": len(request_body),
                    "authorization_mode": "valid_bearer",
                },
                "body_base64": base64.b64encode(request_body).decode("ascii"),
                "body_sha256": sha256_bytes(request_body),
                "body_bytes": len(request_body),
                "connect_completed_monotonic_ns": base,
                "write_started_monotonic_ns": base + 1,
                "last_body_byte_sent_monotonic_ns": base + 2,
            },
            {
                **common,
                "record_type": "http_response_start",
                "request_key": key,
                "status": 400,
                "headers": [["Content-Type", "application/json"]],
                "observed_monotonic_ns": base + 3,
            },
            {
                **common,
                "record_type": "http_body_chunk",
                "request_key": key,
                "chunk_index": 0,
                "body_base64": base64.b64encode(response_body).decode("ascii"),
                "body_sha256": sha256_bytes(response_body),
                "body_bytes": len(response_body),
                "observed_monotonic_ns": base + 4,
            },
            {
                **common,
                "record_type": "http_response_end",
                "request_key": key,
                "outcome": "eof",
                "error": None,
                "body_bytes": len(response_body),
                "body_sha256": sha256_bytes(response_body),
                "observed_monotonic_ns": base + 5,
            },
        ]
        session[insert_at:insert_at] = records
        for sequence, record in enumerate(session):
            record["sequence"] = sequence
        counts = {}
        for record in session:
            counts[record["record_type"]] = counts.get(record["record_type"], 0) + 1
        session[-1]["record_counts"] = counts
        session_path.write_text(
            "".join(compact_json(record) + "\n" for record in session),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("resource metric window contains a foreign HTTP request")

    def test_http_chunk_timestamp_regression_is_rejected(self):
        def mutate(records):
            request = next(
                record
                for record in records
                if record.get("record_type") == "http_request"
            )
            chunk = next(
                record
                for record in records
                if record.get("record_type") == "http_body_chunk"
                and record.get("request_key") == request["request_key"]
            )
            chunk["observed_monotonic_ns"] = request["last_body_byte_sent_monotonic_ns"]

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("body chunk timestamps regress")

    def test_next_http_connection_must_follow_prior_response_end(self):
        def mutate(records):
            requests = [
                record
                for record in records
                if record.get("record_type") == "http_request"
            ]
            first, second = requests[:2]
            prior_end = next(
                record
                for record in records
                if record.get("record_type") == "http_response_end"
                and record.get("request_key") == first["request_key"]
            )["observed_monotonic_ns"]
            second["connect_completed_monotonic_ns"] = prior_end - 1
            second["write_started_monotonic_ns"] = prior_end - 1
            second["last_body_byte_sent_monotonic_ns"] = prior_end - 1

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("begins before the prior HTTP response ended")

    def test_restart_probe_count_must_increment_exactly_once(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "lifecycle_probe"
                and record.get("probe") == "post-header-restart-ready"
            )
            target["n_restarts"] += 1

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("post-restart lifecycle probe identities differ")

    def test_lifecycle_probe_name_is_bound_to_its_phase(self):
        def mutate(records):
            target = next(
                record
                for record in records
                if record.get("record_type") == "lifecycle_probe"
                and record.get("probe") == "normal-segment-start"
            )
            target["phase"] = "final"

        mutate_jsonl(self.root, "raw-session-results.jsonl", mutate)
        self.assert_invalid("lifecycle probe identity is duplicated or differs")

    def test_next_admission_must_follow_prior_release_not_client_end(self):
        session = [
            json.loads(line)
            for line in (self.root / "raw-session-results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        releases = [
            record["event"]
            for record in session
            if record.get("record_type") == "gateway_event"
            and record["event"]["event"] == "request_released"
        ]
        second_request_id = releases[1]["request_id"]
        rewrite_gateway_event_time(
            self.root,
            second_request_id,
            "request_admitted",
            releases[0]["observed_monotonic_ns"],
        )
        self.assert_invalid("admitted before the prior lifecycle terminal event")

    def test_post_release_fd_median_must_equal_baseline(self):
        def mutate(records):
            for record in records:
                if (
                    record.get("segment") == "normal"
                    and record.get("request_index") == 1
                ):
                    record["gateway"]["fd_count"] += 1

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("gateway_fds median differs")

    def test_final_memory_delta_gate_is_recomputed(self):
        def mutate(records):
            for record in records:
                if (
                    record.get("segment") == "normal"
                    and record.get("request_index") == 100
                ):
                    record["host"]["memory_current_bytes"] += 100_000_000

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("final MemoryCurrent delta")

    def test_theil_sen_slope_gate_is_recomputed_from_all_pairs(self):
        def mutate(records):
            for record in records:
                if (
                    record.get("segment") == "normal"
                    and record.get("request_index") is not None
                ):
                    record["host"]["memory_current_bytes"] = (
                        1_000_000_000 + record["request_index"] * 300_000
                    )

        mutate_jsonl(self.root, "soak-resources.raw.jsonl", mutate)
        self.assert_invalid("MemoryCurrent Theil-Sen slope")

    def test_release_matrix_passed_key_is_forbidden(self):
        refresh_matrix_and_sums(
            self.root, lambda matrix: matrix.update({"passed": True})
        )
        self.assert_invalid("forbidden key 'passed'")

    def test_release_matrix_file_size_is_independently_checked(self):
        def mutate(matrix):
            matrix["files"][0]["bytes"] += 1

        refresh_matrix_and_sums(self.root, mutate)
        self.assert_invalid("matrix size differs")

    def test_symlink_bundle_member_is_rejected(self):
        path = self.root / "sampling-results.json"
        target = Path(self.temporary.name) / "outside.json"
        target.write_bytes(path.read_bytes())
        path.unlink()
        os.symlink(target, path)
        self.assert_invalid("non-regular file or symlink")

    def test_all_service_journal_cursors_must_be_unique(self):
        path = self.root / "service-journal.raw.jsonl"
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        ordinary = deepcopy(records[-1])
        ordinary["__CURSOR"] = "s=synthetic-extra;i=1"
        ordinary["MESSAGE"] = "ordinary non-lifecycle line"
        records[-1:-1] = [ordinary, deepcopy(ordinary)]
        path.write_text(
            "".join(compact_json(record) + "\n" for record in records),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("journal cursor is duplicated")

    def test_malformed_structured_service_journal_line_is_rejected(self):
        path = self.root / "service-journal.raw.jsonl"
        records = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        malformed = deepcopy(records[-1])
        malformed["__CURSOR"] = "s=synthetic-extra;i=2"
        malformed["MESSAGE"] = "INFO:     {"
        records.insert(-1, malformed)
        path.write_text(
            "".join(compact_json(record) + "\n" for record in records),
            encoding="utf-8",
        )
        refresh_matrix_and_sums(self.root)
        self.assert_invalid("JSON object|failed to decode")

    def test_cli_defaults_to_full_validation_and_phase1_remains_explicit(self):
        command = [
            sys.executable,
            str(VALIDATOR_PATH),
            str(self.root),
            "--expected-commit",
            GIT_COMMIT,
            "--expected-worker-binary-sha256",
            WORKER_SHA256,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 1)
        self.assertIn(
            "environment.json is not canonical identity JSON", completed.stderr
        )
        self.assertFalse((self.root / "release-validation.json").exists())
        phase1 = subprocess.run(
            command + ["--phase1-only"], text=True, capture_output=True, check=False
        )
        self.assertEqual(phase1.returncode, 0, phase1.stderr)
        self.assertEqual(json.loads(phase1.stdout)["release_status"], "incomplete")
        self.assertFalse((self.root / "release-validation.json").exists())


class FullReleaseValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def derived_fixture(self) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        self.root.mkdir(parents=True, exist_ok=True)
        front_documents = {
            "api-contract-results.json": {
                "schema_version": "synthetic.api.v1",
                "case_count": 12,
            },
            "sampling-results.json": {
                "schema_version": "synthetic.sampling.v1",
                "case_count": 20,
            },
            "cancel-results.json": {
                "schema_version": "synthetic.cancel.v1",
                "case_count": 5,
            },
            "openwebui-smoke.json": {
                "schema_version": "synthetic.smoke.v1",
                "case_count": 22,
            },
        }
        front = SimpleNamespace(
            browser_soak_cases=[
                {"case_id": f"chat-{index:02d}"} for index in range(1, 21)
            ],
            canonical_bytes={
                name: VALIDATOR.independent_canonical_json_bytes(value)
                for name, value in front_documents.items()
            },
        )
        latency = {
            "schema_version": "synthetic.latency.v1",
            "request_count": 72,
        }
        resource = {
            "resource_sample_count": 610,
            "gpu_metric_count": 4,
            "segments": {"normal": {}, "restart": {}},
        }
        soak = {
            "schema_version": VALIDATOR.SOAK_RESULTS_SCHEMA,
            "browser": {"chat_count": 20, "cases": front.browser_soak_cases},
            **resource,
        }
        expected = {
            "sampling-results.json": front.canonical_bytes["sampling-results.json"],
            "cancel-results.json": front.canonical_bytes["cancel-results.json"],
            "prefill-latency-results.json": VALIDATOR.independent_canonical_json_bytes(
                latency
            ),
            "api-contract-results.json": front.canonical_bytes[
                "api-contract-results.json"
            ],
            "openwebui-smoke.json": front.canonical_bytes["openwebui-smoke.json"],
            "soak-results.json": VALIDATOR.independent_canonical_json_bytes(soak),
        }
        for relative, raw in expected.items():
            (self.root / relative).write_bytes(raw)
        return front, latency, resource

    def test_six_views_are_reconstructed_in_fixed_product_order(self) -> None:
        front, latency, resource = self.derived_fixture()
        with (
            mock.patch.object(
                VALIDATOR, "reconstruct_front_views", return_value=front
            ) as front_reconstruction,
            mock.patch.object(
                VALIDATOR, "reconstruct_latency_results", return_value=latency
            ),
            mock.patch.object(
                VALIDATOR,
                "reconstruct_soak_resource_results",
                return_value=resource,
            ),
        ):
            evidence, observed_latency, observed_resource = (
                VALIDATOR._validate_full_derived_views(
                    self.root,
                    SimpleNamespace(),
                    forbidden_values=(b"never-present-secret",),
                )
            )
        self.assertEqual(tuple(evidence), VALIDATOR.FULL_DERIVED_VIEW_PATHS)
        self.assertEqual(observed_latency, latency)
        self.assertEqual(observed_resource, resource)
        self.assertEqual(
            front_reconstruction.call_args.kwargs["forbidden_values"],
            (b"never-present-secret",),
        )

    def test_modified_derived_view_and_forbidden_cleartext_are_rejected(self) -> None:
        front, latency, resource = self.derived_fixture()
        (self.root / "cancel-results.json").write_bytes(b"{}\n")
        with (
            mock.patch.object(VALIDATOR, "reconstruct_front_views", return_value=front),
            mock.patch.object(
                VALIDATOR, "reconstruct_latency_results", return_value=latency
            ),
            mock.patch.object(
                VALIDATOR,
                "reconstruct_soak_resource_results",
                return_value=resource,
            ),
            self.assertRaisesRegex(
                VALIDATOR.ValidationError, "cancel-results.json differs"
            ),
        ):
            VALIDATOR._validate_full_derived_views(
                self.root, SimpleNamespace(), forbidden_values=()
            )

        self.derived_fixture()
        leaked_latency = dict(latency, diagnostic="semantic-secret")
        with (
            mock.patch.object(VALIDATOR, "reconstruct_front_views", return_value=front),
            mock.patch.object(
                VALIDATOR,
                "reconstruct_latency_results",
                return_value=leaked_latency,
            ),
            mock.patch.object(
                VALIDATOR,
                "reconstruct_soak_resource_results",
                return_value=resource,
            ),
            self.assertRaisesRegex(
                VALIDATOR.ValidationError, "forbidden semantic cleartext"
            ),
        ):
            VALIDATOR._validate_full_derived_views(
                self.root,
                SimpleNamespace(),
                forbidden_values=(b"semantic-secret",),
            )

    def test_validation_file_is_mode_0600_exclusive_and_failure_cleans_up(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        raw = VALIDATOR.independent_canonical_json_bytes(
            {"schema_version": VALIDATOR.FULL_REPORT_SCHEMA}
        )
        evidence = VALIDATOR._write_release_validation(self.root, raw)
        path = self.root / VALIDATOR.RELEASE_VALIDATION_FILE
        self.assertEqual(evidence, VALIDATOR.FileEvidence(len(raw), sha256_bytes(raw)))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(path.stat().st_uid, self.root.stat().st_uid)
        self.assertEqual(path.stat().st_gid, self.root.stat().st_gid)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "exclusively create"):
            VALIDATOR._write_release_validation(self.root, b'{"changed":true}\n')
        self.assertEqual(path.read_bytes(), raw)

        path.unlink()
        with (
            mock.patch.object(
                VALIDATOR.os, "write", side_effect=OSError("injected write failure")
            ),
            self.assertRaisesRegex(VALIDATOR.ValidationError, "exclusively create"),
        ):
            VALIDATOR._write_release_validation(self.root, raw)
        self.assertFalse(path.exists())

        with (
            mock.patch.object(
                VALIDATOR.os,
                "write",
                side_effect=KeyboardInterrupt("injected interruption"),
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            VALIDATOR._write_release_validation(self.root, raw)
        self.assertFalse(path.exists())

    def test_summary_is_independently_exact_and_scans_forbidden_values(self) -> None:
        raw = VALIDATOR._independent_summary(
            RUN_ID,
            SCHEDULE,
            forbidden_values=(b"not-present-secret",),
        )
        schedule_line = compact_json(SCHEDULE)
        expected = "\n".join(
            [
                "# SQ8 OpenWebUI full campaign",
                "",
                f"Run ID: `{RUN_ID}`",
                "",
                f"Schedule: `{schedule_line}`",
                "",
                "Artifacts:",
                *(
                    f"- `{relative}`"
                    for relative in sorted(
                        VALIDATOR.BUNDLE_FILES,
                        key=lambda value: value.encode("utf-8"),
                    )
                ),
                "",
            ]
        ).encode("ascii")
        self.assertEqual(raw, expected)
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "forbidden cleartext"):
            VALIDATOR._independent_summary(
                "run-semantic-secret",
                SCHEDULE,
                forbidden_values=(b"semantic-secret",),
            )

    def test_orchestrator_adapter_forwards_anchors_and_forbidden_values(self) -> None:
        repo_root = Path(self.temporary.name) / "source"
        adapter = VALIDATOR.FullCampaignIndependentValidator(
            expected_commit=GIT_COMMIT,
            expected_worker_binary_sha256=WORKER_SHA256,
            repo_root=repo_root,
            forbidden_values=(b"api-token-secret",),
        )
        expected = VALIDATOR.FileEvidence(123, "f" * 64)
        with mock.patch.object(
            VALIDATOR, "validate_full_release", return_value=expected
        ) as full_validation:
            self.assertEqual(adapter.validate(self.root), expected)
        full_validation.assert_called_once_with(
            self.root,
            expected_commit=GIT_COMMIT,
            expected_worker_binary_sha256=WORKER_SHA256,
            repo_root=repo_root,
            forbidden_values=(b"api-token-secret",),
        )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "forbidden cleartext"):
            VALIDATOR.FullCampaignIndependentValidator(
                expected_commit=GIT_COMMIT,
                expected_worker_binary_sha256=WORKER_SHA256,
                forbidden_values=(b"",),
            )

    def full_dependencies(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        order = VALIDATOR.FullCampaignOrderResult(
            phases=VALIDATOR.FULL_CAMPAIGN_PHASE_ORDER,
            openwebui_successful_requests=20,
            cancellation_phases=tuple(SCHEDULE["cancel_phases"]),
            normal_gateway_pid=1200,
            restart_gateway_pid=2200,
            normal_worker_pid=1201,
            restart_worker_pid=2201,
            restart_count_before=2,
            restart_count_after=3,
        )
        api = VALIDATOR.ApiContractValidationResult(
            case_ids=tuple(case.case_id for case in VALIDATOR.API_CONTRACT_CASES),
            request_keys=tuple(f"api-{index}" for index in range(12)),
            statuses=tuple(
                case.expected_status for case in VALIDATOR.API_CONTRACT_CASES
            ),
            cases=tuple(
                {"case_id": case.case_id} for case in VALIDATOR.API_CONTRACT_CASES
            ),
        )
        session = SimpleNamespace(
            full_campaign_order=order,
            api_contract=api,
            lifecycle_quiet_checks=tuple(range(13)),
            browser_actions=tuple(range(123)),
        )
        identity = SimpleNamespace(
            expected_commit=GIT_COMMIT,
            expected_worker_binary_sha256=WORKER_SHA256,
            environment_sha256="c" * 64,
            model_identity_sha256="d" * 64,
        )
        source = VALIDATOR.SourceCheckoutData(GIT_COMMIT, 63, "e" * 64)
        resources = VALIDATOR.ResourceResult({}, 610, 4)
        latency = {"request_count": 72}
        reconstructed_resource = {
            "resource_sample_count": 610,
            "gpu_metric_count": 4,
        }
        return (
            session,
            identity,
            source,
            resources,
            latency,
            reconstructed_resource,
        )

    def test_full_entrypoint_requires_source_and_publishes_canonical_report(
        self,
    ) -> None:
        EvidenceBuilder(self.root).build()
        (
            session,
            identity,
            source,
            resources,
            latency,
            reconstructed_resource,
        ) = self.full_dependencies()
        derived = {
            relative: VALIDATOR.FileEvidence(
                (self.root / relative).stat().st_size,
                sha256_file(self.root / relative),
            )
            for relative in VALIDATOR.FULL_DERIVED_VIEW_PATHS
        }
        summary = (self.root / "summary.md").read_bytes()
        repo_root = Path(self.temporary.name) / "source"
        with (
            mock.patch.object(
                VALIDATOR, "validate_campaign_identity", return_value=identity
            ),
            mock.patch.object(
                VALIDATOR,
                "validate_campaign_source_checkout",
                return_value=source,
            ) as source_validation,
            mock.patch.object(
                VALIDATOR, "validate_session", return_value=session
            ) as session_validation,
            mock.patch.object(VALIDATOR, "validate_resources", return_value=resources),
            mock.patch.object(
                VALIDATOR,
                "_validate_full_derived_views",
                return_value=(derived, latency, reconstructed_resource),
            ),
            mock.patch.object(VALIDATOR, "_independent_summary", return_value=summary),
        ):
            evidence = VALIDATOR.validate_full_release(
                self.root,
                expected_commit=GIT_COMMIT,
                expected_worker_binary_sha256=WORKER_SHA256,
                repo_root=repo_root,
                forbidden_values=(b"not-in-report",),
            )
        source_validation.assert_called_once_with(identity, repo_root=repo_root)
        self.assertIs(session_validation.call_args.args[4], identity)
        validation_path = self.root / VALIDATOR.RELEASE_VALIDATION_FILE
        report_raw = validation_path.read_bytes()
        self.assertEqual(evidence.bytes, len(report_raw))
        self.assertEqual(evidence.sha256, sha256_bytes(report_raw))
        self.assertEqual(
            report_raw,
            VALIDATOR.independent_canonical_json_bytes(json.loads(report_raw)),
        )
        report = json.loads(report_raw)
        self.assertEqual(report["release_status"], "complete")
        self.assertTrue(report["full_campaign_validated"])
        self.assertEqual(report["gate_details"]["source_checkout"]["source_count"], 63)

    def test_source_failure_never_creates_validation_file(self) -> None:
        EvidenceBuilder(self.root).build()
        _, identity, _, _, _, _ = self.full_dependencies()
        session_validation = mock.Mock()
        with (
            mock.patch.object(
                VALIDATOR, "validate_campaign_identity", return_value=identity
            ),
            mock.patch.object(
                VALIDATOR,
                "validate_campaign_source_checkout",
                side_effect=VALIDATOR.ValidationError("injected source rejection"),
            ),
            mock.patch.object(VALIDATOR, "validate_session", session_validation),
            self.assertRaisesRegex(VALIDATOR.ValidationError, "source rejection"),
        ):
            VALIDATOR.validate_full_release(
                self.root,
                expected_commit=GIT_COMMIT,
                expected_worker_binary_sha256=WORKER_SHA256,
                repo_root=Path(self.temporary.name) / "source",
            )
        session_validation.assert_not_called()
        self.assertFalse((self.root / VALIDATOR.RELEASE_VALIDATION_FILE).exists())


if __name__ == "__main__":
    unittest.main()
