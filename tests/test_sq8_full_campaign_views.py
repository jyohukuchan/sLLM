#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools/sq8_full_campaign_views.py"
SPEC = importlib.util.spec_from_file_location("sq8_full_campaign_views", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VIEWS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VIEWS
SPEC.loader.exec_module(VIEWS)

SHA = "a" * 64


def compact(value):
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


def api_view():
    cases = []
    for index in range(1, 11):
        status = 200 if index == 1 else 400
        cases.append(
            {
                "case_index": index,
                "case_id": f"api-case-{index:02d}",
                "status": status,
                "request_body_bytes": index,
                "request_body_sha256": f"{index:064x}",
                "response_body_bytes": index + 10,
                "response_body_sha256": f"{index + 10:064x}",
                "error": None
                if status == 200
                else {
                    "type": "invalid_request_error",
                    "code": f"error-{index:02d}",
                    "param": None,
                    "message_utf8_bytes": 20 + index,
                    "message_sha256": f"{index + 20:064x}",
                },
            }
        )
    return {
        "schema_version": VIEWS.API_INPUT_SCHEMA,
        "case_count": 10,
        "http_record_count": 40,
        "journal_record_count": 13,
        "lifecycle_event_count": 0,
        "quiet_check_count": 13,
        "cases": cases,
        "source_bindings": {"host_source_path": "/not/in/final/view.py"},
    }


def combined_view():
    schedule = []
    cases = []
    for position in range(21):
        case_id = (
            "openwebui_smoke"
            if position == 0
            else f"openwebui_soak_chat_{position:02d}"
        )
        common = {
            "position": position,
            "case_index": position,
            "case_kind": "smoke" if position == 0 else "soak",
            "browser_case": case_id,
        }
        schedule.append(common)
        cases.append(
            {
                **common,
                "browser_case_sha256": SHA,
                "action_count": 5,
                "socket_event_count": 4,
                "chat_id_sha256": SHA,
                "message_id_sha256": SHA,
                "request_id_sha256": SHA,
                "completion_id_sha256": SHA,
                "admitted_monotonic_ns": 100 + position * 10,
                "released_monotonic_ns": 105 + position * 10,
                "outcome": "stop",
                "reset_complete": True,
            }
        )
    return {
        "schema_version": VIEWS.COMBINED_INPUT_SCHEMA,
        "mode": "smoke_then_soak20",
        "schedule": schedule,
        "chat_count": 21,
        "action_count": 105,
        "lifecycle_record_count": 105,
        "maximum_active_requests": 1,
        "stop_release_count": 21,
        "reset_complete_count": 21,
        "component_summary_sha256": SHA,
        "source_bindings": {"source_path": "/not/in/final/browser.py"},
        "cases": cases,
    }


def direct_cancel_view():
    cases = []
    request_index = 0
    for phase in VIEWS.DIRECT_CANCEL_PHASES:
        for role in ("target", "recovery"):
            request_index += 1
            item = {
                "request_index": request_index,
                "phase": phase,
                "role": role,
                "case_id": f"{phase}-{role}",
                "request_body_bytes": 100,
                "request_body_sha256": SHA,
                "http_status": 200,
                "http_outcome": "client_closed" if role == "target" else "eof",
                "response_body_bytes": 20,
                "response_body_sha256": SHA,
                "lifecycle_event_count": 7,
                "request_id_sha256": SHA,
                "completion_id_sha256": SHA,
                "release_observed_monotonic_ns": request_index * 100,
                "release_outcome": "cancelled" if role == "target" else "length",
                "reset_complete": True,
                "completion_tokens": 1 if role == "target" else 2,
            }
            if role == "target":
                item.update(
                    {
                        "trigger_observed_monotonic_ns": request_index * 100 - 20,
                        "cancel_observed_monotonic_ns": request_index * 100 - 10,
                        "cancel_to_release_ns": 10,
                        "progress": None,
                    }
                )
            cases.append(item)
    return {
        "schema_version": VIEWS.DIRECT_CANCEL_INPUT_SCHEMA,
        "phase_order": list(VIEWS.DIRECT_CANCEL_PHASES),
        "request_count": 8,
        "http_record_count": 32,
        "lifecycle_record_count": 55,
        "maximum_active_requests": 1,
        "component_summary_sha256": SHA,
        "source_bindings": {"source_path": "/not/in/final/direct.py"},
        "cases": cases,
    }


def stop_view():
    return {
        "schema_version": VIEWS.STOP_INPUT_SCHEMA,
        "browser_case": "openwebui_stop_after_visible_content",
        "browser_action_count": 9,
        "browser_socket_event_count": 9,
        "lifecycle_event_count": 11,
        "request_count": 2,
        "maximum_active_requests": 1,
        "component_summary_sha256": SHA,
        "screenshot": {
            "file": "browser/openwebui-stop-before.png",
            "bytes": 100,
            "sha256": SHA,
        },
        "source_bindings": {"source_path": "/not/in/final/stop.py"},
        "browser_evidence": {},
        "gateway_evidence": {
            "target_outcome": "cancelled",
            "cancel_reason": "client_disconnect",
            "cancel_to_release_ns": 1234,
            "recovery_outcome": "stop",
            "target_reset_complete": True,
            "recovery_reset_complete": True,
        },
        "raw_artifacts": {},
    }


def failure_view():
    return {
        "schema_version": VIEWS.FAILURE_INPUT_SCHEMA,
        "phase": "post_header_failure",
        "cases": {
            "failure": "post-header-failure",
            "recovery": "post-header-recovery",
        },
        "source_sha256": {"source_path": "/not/in/final/failure.py"},
        "summary_sha256": SHA,
        "service": {},
        "browser": {
            "action_count": 9,
            "socket_event_count": 7,
            "screenshot_file": "browser/post-header-failure.png",
            "screenshot_bytes": 200,
            "screenshot_sha256": SHA,
        },
        "fault": {
            "target_pid": 50,
            "started_monotonic_ns": 10,
            "completed_monotonic_ns": 11,
            "worker_fatal_monotonic_ns": 12,
        },
        "recovery": {
            "ready_completed_monotonic_ns": 13,
            "admitted_monotonic_ns": 14,
            "released_monotonic_ns": 15,
        },
        "journal": {},
    }


def latency_view():
    ttft_samples = []
    sequence = 0
    for fixture_id, prompt_tokens in VIEWS.FIXTURE_ORDER:
        for kind, count in (("warmup", 2), ("measured", 10)):
            for sample_index in range(1, count + 1):
                sequence += 1
                ttft_samples.append(
                    {
                        "sequence": sequence,
                        "case_id": f"ttft-{fixture_id}-{kind}-{sample_index:02d}",
                        "sample_kind": kind,
                        "sample_index": sample_index,
                        "fixture_id": fixture_id,
                        "prompt_tokens": prompt_tokens,
                        "ttft_ns": prompt_tokens * 1000 + sample_index,
                        "content_object_count": 1,
                        "release_outcome": "cancelled",
                        "release_completion_tokens": 1,
                    }
                )
    decode_samples = []
    for kind, count in (("warmup", 2), ("measured", 10)):
        for sample_index in range(1, count + 1):
            sequence += 1
            decode_samples.append(
                {
                    "sequence": sequence,
                    "case_id": f"decode64-{kind}-{sample_index:02d}",
                    "sample_kind": kind,
                    "sample_index": sample_index,
                    "fixture_id": "exact-p0032",
                    "prompt_tokens": 32,
                    "decode_elapsed_ns": 63_000_000,
                    "decode_intervals_ns": [1_000_000] * 63,
                    "decode_tokens_per_second": {
                        "numerator": 63_000_000_000,
                        "denominator": 63_000_000,
                    },
                    "release_outcome": "length",
                    "release_completion_tokens": 64,
                }
            )
    ttft_metrics = {
        fixture_id: {
            "count": 10,
            "p50_ns": 100,
            "p95_ns": 110,
            "p50_maximum_ns": 1_000_000_000,
            "p95_maximum_ns": 2_000_000_000,
        }
        for fixture_id, _prompt_tokens in VIEWS.FIXTURE_ORDER
    }
    return {
        "schema_version": VIEWS.LATENCY_INPUT_SCHEMA,
        "request_count": 72,
        "http_record_count": 288,
        "lifecycle_record_count": 500,
        "journal_record_count": 500,
        "prefill_ttft": {
            "request_count": 60,
            "metrics": ttft_metrics,
            "samples": ttft_samples,
        },
        "decode64": {
            "request_count": 12,
            "metrics": {
                "request_count": 10,
                "interval_count": 630,
                "p50_tokens_per_second": 1000,
                "minimum_p50_tokens_per_second": 15,
                "p95_inter_content_ns": 1_000_000,
                "maximum_p95_inter_content_ns": 100_000_000,
            },
            "samples": decode_samples,
        },
        "source_bindings": {"source_path": "/not/in/final/latency.py"},
    }


def sampling_cases():
    return [
        {
            "request_index": index,
            "temperature": 0.6,
            "top_p": 0.95,
            "seed": index,
            "http_status": 200,
            "http_outcome": "eof",
            "release_outcome": "length",
            "completion_tokens": 2,
            "reset_complete": True,
        }
        for index in range(5, 101, 5)
    ]


def process(pid, ppid, exe, rss_kb, children):
    return {
        "pid": pid,
        "ppid": ppid,
        "exe": exe,
        "starttime_ticks_before": pid * 100,
        "starttime_ticks_after": pid * 100,
        "vmrss_kb": rss_kb,
        "vmrss_bytes": rss_kb * 1024,
        "threads": 4 if exe.endswith("gateway") else 8,
        "fd_count": 10 if exe.endswith("gateway") else 12,
        "children": children,
    }


def resource_sample(segment, phase, request_index, sample_index, point_number):
    gateway_pid, worker_pid = (100, 101) if segment == "normal" else (200, 201)
    settle = point_number * 10_000_000_000
    slope = 0 if request_index is None else request_index
    gateway_rss_kb = 3_000_000 + slope * 6
    worker_rss_kb = 4_000_000 + slope * 8
    return {
        "schema_version": VIEWS.RESOURCE_INPUT_SCHEMA,
        "record_type": "resource_sample",
        "segment": segment,
        "phase": phase,
        "request_index": request_index,
        "request_id": None
        if request_index is None
        else f"resource-{segment}-{request_index:03d}",
        "release_outcome": None if request_index is None else "length",
        "release_observed_monotonic_ns": None if request_index is None else settle,
        "reset_complete": None if request_index is None else True,
        "idle_settle_started_monotonic_ns": settle,
        "sample_index": sample_index,
        "sample_monotonic_ns": settle + 5_000_000_000 + sample_index * 1_000_000_000,
        "systemd": {
            "control_group_before": "/system.slice/ullm-openai.service",
            "control_group_after": "/system.slice/ullm-openai.service",
            "main_pid_before": gateway_pid,
            "main_pid_after": gateway_pid,
        },
        "host": {"memory_current_bytes": 1_000_000_000 + slope * 2048},
        "gateway": process(
            gateway_pid, 1, "/usr/bin/ullm-gateway", gateway_rss_kb, [worker_pid]
        ),
        "worker": process(
            worker_pid,
            gateway_pid,
            "/usr/bin/ullm-sq8-worker",
            worker_rss_kb,
            [],
        ),
        "gpu": {
            "index": 2,
            "bdf": "0000:47:00.0",
            "uuid": "a8ff7551-0000-1000-80e9-ddefa2d60f55",
            "kfd_gpu_id": 51545,
            "process_record_count": 1,
            "worker_pid": worker_pid,
            "mem_usage": {"value": 2_000_000_000 + slope * 4096, "unit": "B"},
            "kfd_vram_bytes": 2_000_000_000 + slope * 4096,
            "unrelated_process_pids": [],
        },
    }


def metric(segment, boundary, point_number):
    return {
        "schema_version": VIEWS.RESOURCE_INPUT_SCHEMA,
        "record_type": "gpu_metric",
        "segment": segment,
        "boundary": boundary,
        "captured_monotonic_ns": point_number * 10_000_000_000,
        "gpu_index": 2,
        "raw_output_file": f"amd-smi-metric-{segment}-{boundary}.json",
        "raw_output_sha256": SHA,
    }


def resource_records():
    records = [
        {
            "schema_version": VIEWS.RESOURCE_INPUT_SCHEMA,
            "record_type": "header",
            "service_unit": "ullm-openai.service",
            "commands": deepcopy(VIEWS.RESOURCE_COMMANDS),
            "tools": {
                "systemd_major": 255,
                "systemd_version_line": "systemd 255 synthetic",
                "amd_smi_tool": "26.2.2+e1a6bc5663",
                "amd_smi_library": "26.2.2",
                "rocm": "7.2.1",
                "amd_smi_version_output": (
                    "AMD SMI Tool: 26.2.2+e1a6bc5663; "
                    "AMD SMI Library: 26.2.2; ROCm: 7.2.1"
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
            "schedule": VIEWS.RESOURCE_SCHEDULE,
        }
    ]
    point_number = 1
    records.append(metric("normal", "before", point_number))
    point_number += 1
    for sample_index in range(5):
        records.append(
            resource_sample("normal", "baseline", None, sample_index, point_number)
        )
    point_number += 1
    for request_index in range(1, 101):
        for sample_index in range(5):
            records.append(
                resource_sample(
                    "normal", "post_release", request_index, sample_index, point_number
                )
            )
        point_number += 1
    records.append(metric("normal", "after", point_number))
    point_number += 1
    records.append(metric("restart", "before", point_number))
    point_number += 1
    for sample_index in range(5):
        records.append(
            resource_sample("restart", "baseline", None, sample_index, point_number)
        )
    point_number += 1
    for request_index in range(1, 21):
        for sample_index in range(5):
            records.append(
                resource_sample(
                    "restart", "post_release", request_index, sample_index, point_number
                )
            )
        point_number += 1
    records.append(metric("restart", "after", point_number))
    assert len(records) == 615
    return records


def write_resource(path, records=None):
    values = resource_records() if records is None else records
    path.write_bytes(b"".join(compact(value) for value in values))
    os.chmod(path, 0o600)


class FullCampaignViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.resources = self.root / "soak-resources.raw.jsonl"
        write_resource(self.resources)

    def build(self, **changes):
        values = {
            "api_contract": api_view(),
            "combined": combined_view(),
            "direct_cancel": direct_cancel_view(),
            "stop": stop_view(),
            "failure": failure_view(),
            "latency": latency_view(),
            "sampling_cases": sampling_cases(),
            "resource_raw_path": self.resources,
        }
        values.update(changes)
        return VIEWS.build_full_campaign_views(**values)

    def test_builds_six_canonical_views_and_omits_component_only_fields(self):
        result = self.build()
        documents = result.documents()
        self.assertEqual(tuple(documents), VIEWS.VIEW_FILENAMES)
        self.assertEqual(
            {name: set(value) for name, value in documents.items()},
            {
                "sampling-results.json": {
                    "schema_version",
                    "sampled_request_count",
                    "sampled_normal_indices",
                    "cases",
                },
                "cancel-results.json": {
                    "schema_version",
                    "phase_count",
                    "request_count",
                    "maximum_active_requests",
                    "phases",
                },
                "prefill-latency-results.json": {
                    "schema_version",
                    "request_count",
                    "prefill_ttft",
                    "decode64",
                },
                "api-contract-results.json": {
                    "schema_version",
                    "case_count",
                    "http_record_count",
                    "quiet_check_count",
                    "cases",
                },
                "openwebui-smoke.json": {
                    "schema_version",
                    "normal",
                    "post_header_failure",
                    "recovery",
                },
                "soak-results.json": {
                    "schema_version",
                    "browser",
                    "resource_sample_count",
                    "gpu_metric_count",
                    "segments",
                },
            },
        )
        encoded = result.serialized()
        self.assertTrue(
            all(
                raw.endswith(b"\n") and raw.count(b"\n") == 1
                for raw in encoded.values()
            )
        )
        self.assertTrue(all(raw.isascii() for raw in encoded.values()))
        joined = b"".join(encoded.values())
        for forbidden in (
            b"component_summary_sha256",
            b"source_bindings",
            b"socket_event_count",
            b"chat_id_sha256",
            b"message_id_sha256",
            b"/not/in/final/",
        ):
            self.assertNotIn(forbidden, joined)

        self.assertEqual(result.api_contract_results["case_count"], 10)
        self.assertEqual(result.cancel_results["phase_count"], 5)
        self.assertEqual(result.cancel_results["request_count"], 10)
        self.assertEqual(result.prefill_latency_results["request_count"], 72)
        self.assertEqual(
            result.prefill_latency_results["decode64"]["samples"][0][
                "decode_tokens_per_second"
            ],
            1000,
        )
        self.assertEqual(
            result.openwebui_smoke["post_header_failure"]["action_count"], 5
        )
        self.assertEqual(result.openwebui_smoke["recovery"]["action_count"], 4)
        self.assertEqual(result.soak_results["browser"]["chat_count"], 20)

    def test_resource_analyzer_derives_complete_exact_metrics(self):
        with mock.patch.object(
            Path, "read_bytes", side_effect=AssertionError("bulk read")
        ):
            result = VIEWS.analyze_soak_resources(self.resources)
        self.assertEqual(result["resource_sample_count"], 610)
        self.assertEqual(result["gpu_metric_count"], 4)
        normal = result["segments"]["normal"]
        restart = result["segments"]["restart"]
        self.assertEqual(normal["measured_point_count"], 100)
        self.assertEqual(restart["measured_point_count"], 20)
        self.assertEqual(
            normal["complete_theil_sen_per_request"],
            {
                "memory_current_bytes": 2048,
                "process_vram_bytes": 4096,
                "gateway_rss_bytes": 6144,
                "worker_rss_bytes": 8192,
            },
        )
        self.assertEqual(
            normal["final_signed_median_delta"]["memory_current_bytes"], 204800
        )
        self.assertEqual(
            restart["final_signed_median_delta"]["process_vram_bytes"], 81920
        )
        self.assertEqual(normal["stable_process_counts"]["gateway"]["children"], 1)
        self.assertEqual(normal["stable_process_counts"]["worker"]["children"], 0)

    def test_resource_header_commands_and_tool_versions_are_frozen(self):
        mutations = {
            "missing-command": (
                lambda header: header["commands"].pop("proc_stat"),
                "resource command identity differs",
            ),
            "extra-command": (
                lambda header: header["commands"].update({"extra": "extra"}),
                "resource command identity differs",
            ),
            "changed-command": (
                lambda header: header["commands"].update(
                    {"proc_stat": "cat /changed/proc/stat"}
                ),
                "resource command identity differs",
            ),
            "changed-systemd-line": (
                lambda header: header["tools"].update(
                    {"systemd_version_line": "systemd 256 synthetic"}
                ),
                "resource systemd, AMD SMI, or ROCm tool identity differs",
            ),
            "changed-tool-version": (
                lambda header: header["tools"].update({"amd_smi_tool": "26.2.3"}),
                "resource systemd, AMD SMI, or ROCm tool identity differs",
            ),
            "missing-version-output": (
                lambda header: header["tools"].update(
                    {"amd_smi_version_output": "26.2.2+e1a6bc5663 26.2.2"}
                ),
                "resource systemd, AMD SMI, or ROCm tool identity differs",
            ),
        }
        for name, (mutate, expected) in mutations.items():
            with self.subTest(name=name):
                path = self.root / f"header-{name}.raw.jsonl"
                header = resource_records()[0]
                mutate(header)
                write_resource(path, [header])
                with self.assertRaisesRegex(VIEWS.FullCampaignViewError, expected):
                    VIEWS.analyze_soak_resources(path)

    def test_resource_accepts_stop_and_cancelled_post_release_outcomes(self):
        records = resource_records()
        for record in records:
            if record.get("record_type") != "resource_sample":
                continue
            if record.get("segment") != "normal":
                continue
            if record.get("request_index") == 1:
                record["release_outcome"] = "stop"
            elif record.get("request_index") == 2:
                record["release_outcome"] = "cancelled"
        path = self.root / "terminal-outcomes.raw.jsonl"
        write_resource(path, records)
        result = VIEWS.analyze_soak_resources(path)
        self.assertEqual(result["resource_sample_count"], 610)
        self.assertEqual(result["segments"]["normal"]["measured_point_count"], 100)

    def test_sampling_schema_fixes_float_types_order_and_outcomes(self):
        cases = sampling_cases()
        view = VIEWS.project_sampling(cases)
        self.assertEqual(view["sampled_normal_indices"], list(range(5, 101, 5)))
        invalid = deepcopy(cases)
        invalid[0]["temperature"] = 0
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_sampling(invalid)
        duplicate = deepcopy(cases)
        duplicate[1]["request_index"] = 5
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_sampling(duplicate)

    def test_component_counts_and_exact_case_schemas_are_enforced(self):
        api = api_view()
        api["case_count"] = 9
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_api_contract(api)

        combined = combined_view()
        combined["cases"][0]["unexpected_cleartext"] = "must not be ignored"
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_browser_soak(combined)

        direct = direct_cancel_view()
        direct["cases"][0]["unexpected_cleartext"] = "must not be ignored"
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_cancellation(direct, stop_view())

        direct = direct_cancel_view()
        direct["http_record_count"] = 33
        VIEWS.project_cancellation(direct, stop_view())
        for invalid_count in (31, 2049):
            direct["http_record_count"] = invalid_count
            with self.assertRaises(VIEWS.FullCampaignViewError):
                VIEWS.project_cancellation(direct, stop_view())

    def test_cancellation_http_deadline_and_stop_recovery_are_exact(self):
        direct = direct_cancel_view()
        direct["cases"][0]["http_outcome"] = "eof"
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_cancellation(direct, stop_view())

        direct = direct_cancel_view()
        direct["cases"][0]["cancel_to_release_ns"] = 5_000_000_001
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_cancellation(direct, stop_view())

        stop = stop_view()
        stop["gateway_evidence"]["recovery_outcome"] = "length"
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_cancellation(direct_cancel_view(), stop)

    def test_failure_timeline_is_kill_then_fatal_then_recovery(self):
        failure = failure_view()
        failure["fault"]["worker_fatal_monotonic_ns"] = 10
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.project_openwebui_smoke(combined_view(), failure)

    def test_canonical_serializer_rejects_nan_and_secret_cleartext(self):
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.canonical_json_bytes({"value": math.nan})
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.canonical_json_bytes(
                {"redacted": "top-secret-value"},
                forbidden_values=(b"top-secret-value",),
            )
        api = api_view()
        api["cases"][1]["error"]["code"] = "top-secret-value"
        with self.assertRaises(VIEWS.FullCampaignViewError):
            self.build(api_contract=api, forbidden_values=(b"top-secret-value",))
        for escaped_secret in (
            b'escaped-secret-"-value',
            b"escaped-secret-\\-value",
            b"escaped-secret-\t-value",
        ):
            with (
                self.subTest(secret=escaped_secret),
                self.assertRaises(VIEWS.FullCampaignViewError),
            ):
                VIEWS.canonical_json_bytes(
                    {"value": escaped_secret.decode("utf-8")},
                    forbidden_values=(escaped_secret,),
                )

    def test_resource_json_rejects_duplicate_keys_and_nonfinite_values(self):
        duplicate = self.root / "duplicate.raw.jsonl"
        lines = self.resources.read_bytes().splitlines(keepends=True)
        lines[0] = lines[0].replace(
            b'"record_type":', b'"schema_version":"duplicate","record_type":', 1
        )
        duplicate.write_bytes(b"".join(lines))
        os.chmod(duplicate, 0o600)
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.analyze_soak_resources(duplicate)

        nonfinite = self.root / "nonfinite.raw.jsonl"
        lines = self.resources.read_bytes().splitlines(keepends=True)
        lines[7] = lines[7].replace(b"1000002048", b"NaN", 1)
        nonfinite.write_bytes(b"".join(lines))
        os.chmod(nonfinite, 0o600)
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.analyze_soak_resources(nonfinite)

    def test_resource_rejects_secret_and_changed_process_counts(self):
        secret_path = self.root / "secret.raw.jsonl"
        records = resource_records()
        records[0]["tools"]["amd_smi_version_output"] = "fixed top-secret-value output"
        write_resource(secret_path, records)
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.analyze_soak_resources(
                secret_path, forbidden_values=(b"top-secret-value",)
            )

        changed_path = self.root / "changed.raw.jsonl"
        records = resource_records()
        for index in range(7, 12):
            records[index]["gateway"]["threads"] += 1
        write_resource(changed_path, records)
        with self.assertRaises(VIEWS.FullCampaignViewError):
            VIEWS.analyze_soak_resources(changed_path)


if __name__ == "__main__":
    unittest.main()
