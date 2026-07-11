import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
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

    def lifecycle_request(self, segment: str, role: str, index: int) -> tuple[str, int]:
        phase = f"resource_{segment}"
        request_id = f"req-{segment}-{role}-{index:03d}"
        completion_id = f"chatcmpl-{segment}-{role}-{index:03d}"
        case_id = f"{segment}-{role}-{index:03d}"
        admitted_time = self.now
        started_time = admitted_time + 100_000
        release_time = admitted_time + 1_000_000
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
            request_id, release_time = self.lifecycle_request(segment, "request", index)
            self.resource_point(
                segment, "post_release", index, request_id, release_time
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
                "docker_network_id": "network-synthetic",
                "gateway_source_sha256": "7" * 64,
                "worker_source_sha256": "8" * 64,
                "worker_binary_sha256": WORKER_SHA256,
            },
            input_files=[],
            schedule=deepcopy(SCHEDULE),
            thresholds=deepcopy(THRESHOLDS),
        )
        self.resource_records.append(self.resource_header())
        self.segment("normal", 100)
        self.segment("restart", 20)
        counts = {"header": 1, "gateway_event": len(self.journal_records), "run_end": 1}
        self.session_add(
            "run_end",
            "final",
            None,
            completed_utc="2026-07-11T01:00:00Z",
            completed_monotonic_ns=self.now,
            final_git_commit=GIT_COMMIT,
            final_git_status_raw="",
            final_git_status_sha256=sha256_bytes(b""),
            record_counts=counts,
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
