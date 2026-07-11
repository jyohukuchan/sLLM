import base64
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from copy import deepcopy
from fractions import Fraction
from pathlib import Path


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


class ApiContractHttpValidationTest(unittest.TestCase):
    def setUp(self):
        self.records = build_api_contract_http_records()

    def validate(self, records=None):
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
        return VALIDATOR.validate_api_contract_http(state)

    def assert_invalid(self, text: str, records=None):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, text):
            self.validate(records)

    def record(self, records, case_id: str, record_type: str):
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

    def test_cli_requires_explicit_phase1_and_never_writes_final_validation(self):
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
        self.assertIn("full P8-F release gates are not implemented", completed.stderr)
        self.assertFalse((self.root / "release-validation.json").exists())
        phase1 = subprocess.run(
            command + ["--phase1-only"], text=True, capture_output=True, check=False
        )
        self.assertEqual(phase1.returncode, 0, phase1.stderr)
        self.assertEqual(json.loads(phase1.stdout)["release_status"], "incomplete")
        self.assertFalse((self.root / "release-validation.json").exists())


if __name__ == "__main__":
    unittest.main()
