from __future__ import annotations

import copy
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

from common import ContractError, canonical_bytes  # noqa: E402
import run_engine_performance as runner  # noqa: E402
from create_engine_build_identity import create_identity  # noqa: E402
from ci.tests.test_engine_performance_schema import build_identity_for, evidence_for, first_row, monitor_capture_for, result_for  # noqa: E402
import engine_performance_common as contracts  # noqa: E402


def build_config_records(target: str) -> list[str]:
    return [f"{key}={value}" for key, value in contracts.expected_build_configuration(target).items()]


def amd_smi_process_record(
    pid: int,
    *,
    vram_bytes: int = 32 * 1024,
    gtt_bytes: int = 32 * 1024,
    gfx_usage_ns: int = 0,
    cu_occupancy: object = 0,
) -> dict[str, object]:
    return {
        "process_info": {
            "name": "mprime",
            "pid": pid,
            "memory_usage": {
                "gtt_mem": {"value": gtt_bytes, "unit": "B"},
                "cpu_mem": {"value": 0, "unit": "B"},
                "vram_mem": {"value": vram_bytes, "unit": "B"},
            },
            "mem_usage": {"value": vram_bytes + gtt_bytes, "unit": "B"},
            "usage": {
                "gfx": {"value": gfx_usage_ns, "unit": "ns"},
                "enc": {"value": 0, "unit": "ns"},
            },
            "sdma_usage": {"value": 0, "unit": "us"},
            "cu_occupancy": cu_occupancy,
            "evicted_time": {"value": 0, "unit": "ms"},
        },
    }


class EnginePerformanceRunnerTests(unittest.TestCase):
    def assert_invalid(self, mutation) -> None:
        row = first_row()
        result = result_for(row)
        mutation(result)
        with self.assertRaises(ContractError):
            contracts.validate_cli_result(result, row)

    def test_bounded_process_timeout_and_pipe_capture_are_host_only(self) -> None:
        capture = runner._execute_bounded(
            [sys.executable, "-c", "import sys;sys.stdout.write('out');sys.stderr.write('err')"],
            {}, ROOT, 5,
        )
        self.assertEqual(capture["exit_code"], 0)
        self.assertEqual(capture["stdout"], b"out")
        self.assertEqual(capture["stderr"], b"err")
        timeout = runner._execute_bounded([sys.executable, "-c", "import time;time.sleep(10)"], {}, ROOT, 1)
        self.assertTrue(timeout["timed_out"])
        self.assertTrue(timeout["process_group_gone"])
        descendant = runner._execute_bounded(
            [sys.executable, "-c", "import subprocess,time;subprocess.Popen([\"/bin/sh\",\"-c\",\"sleep 10\"]);time.sleep(10)"],
            {}, ROOT, 1,
        )
        self.assertTrue(descendant["timed_out"])
        self.assertTrue(descendant["process_group_gone"])
        self.assertTrue(descendant["kill_sent"] or descendant["term_sent"])

    def test_stdout_overflow_is_bounded_and_terminates_the_process_group(self) -> None:
        with mock.patch.object(runner, "MAX_RAW_BYTES", 4096), mock.patch.object(
            runner, "TERMINATION_GRACE_SECONDS", 0.2,
        ):
            capture = runner._execute_bounded(
                [sys.executable, "-c", "import os,time;os.write(1,b'x'*8192);time.sleep(10)"],
                {}, ROOT, 5,
            )
        self.assertEqual(capture["output_overflow"], ["stdout"])
        self.assertEqual(len(capture["stdout"]), 4096)
        self.assertEqual(capture["stderr"], b"")
        self.assertFalse(capture["timed_out"])
        self.assertTrue(capture["term_sent"] or capture["kill_sent"])
        self.assertTrue(capture["process_group_gone"])

    def test_stderr_overflow_is_bounded_and_terminates_the_process_group(self) -> None:
        with mock.patch.object(runner, "MAX_RAW_BYTES", 4096), mock.patch.object(
            runner, "TERMINATION_GRACE_SECONDS", 0.2,
        ):
            capture = runner._execute_bounded(
                [sys.executable, "-c", "import os,time;os.write(2,b'e'*8192);time.sleep(10)"],
                {}, ROOT, 5,
            )
        self.assertEqual(capture["output_overflow"], ["stderr"])
        self.assertEqual(capture["stdout"], b"")
        self.assertEqual(len(capture["stderr"]), 4096)
        self.assertFalse(capture["timed_out"])
        self.assertTrue(capture["term_sent"] or capture["kill_sent"])
        self.assertTrue(capture["process_group_gone"])

    def test_output_overflow_produces_a_durable_fail_manifest(self) -> None:
        row = first_row()
        target = str(row["target"])
        device = contracts.expected_device(target)
        observation = {
            "selected_device": device,
            "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
            "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
        }
        with tempfile.TemporaryDirectory(prefix="sllm-performance-overflow-fail-") as directory:
            root = Path(directory)
            binary = root / "engine"
            binary.write_bytes(b"engine")
            binary.chmod(0o755)
            lock = root / "lock.json"
            lock.write_bytes(b"{}")
            cache = root / "cache"
            cache.mkdir()
            build_manifest = root / "build.json"
            build_manifest.write_bytes(b"{}")
            binary_digest = hashlib.sha256(b"engine").hexdigest()
            build_digest = hashlib.sha256(b"{}").hexdigest()
            build_document = build_identity_for(row, binary_sha256=binary_digest)
            build_document.pop("path")
            build_document.pop("sha256")
            capture = monitor_capture_for(target)
            capture.update({
                "stdout": b"x" * 4096, "stderr": b"", "exit_code": -signal.SIGTERM,
                "timed_out": False, "term_sent": True, "kill_sent": False,
                "output_overflow": ["stdout"],
            })
            with mock.patch.object(runner, "_validate_lock", return_value=({"model": {"files": []}}, "6" * 64)), mock.patch.object(
                runner, "_validate_cache", return_value="7" * 64,
            ), mock.patch.object(
                runner, "_validate_build_manifest", return_value=(build_document, build_digest),
            ), mock.patch.object(runner, "cache_digest", return_value="7" * 64):
                manifest = runner.run_row(
                    str(row["row_id"]), binary, lock, cache, root / "output",
                    build_manifest=build_manifest, repo=ROOT,
                    command_runner=lambda *_args: capture,
                    observation_provider=lambda *_args: copy.deepcopy(observation),
                    evidence_provider=lambda *_args: evidence_for(target),
                    tool_provider=lambda: {
                        "path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test",
                        "library_version": "test", "rocm_version": contracts.ROCM_RELEASE,
                    },
                )
        self.assertEqual(manifest["state"], "FAIL")
        self.assertIn("output exceeded the bounded limit on stdout", manifest["failure_reason"])
        self.assertTrue(capture["process_group_gone"])

    def test_large_simultaneous_stdout_and_stderr_do_not_deadlock(self) -> None:
        bytes_per_pipe = 1024 * 1024
        script = (
            "import os,threading;"
            f"n={bytes_per_pipe};"
            "a=threading.Thread(target=lambda:os.write(1,b'o'*n));"
            "b=threading.Thread(target=lambda:os.write(2,b'e'*n));"
            "a.start();b.start();a.join();b.join()"
        )
        capture = runner._execute_bounded([sys.executable, "-c", script], {}, ROOT, 5)
        self.assertEqual(capture["exit_code"], 0)
        self.assertEqual(capture["output_overflow"], [])
        self.assertEqual(capture["stdout"], b"o" * bytes_per_pipe)
        self.assertEqual(capture["stderr"], b"e" * bytes_per_pipe)
        self.assertTrue(capture["process_group_gone"])

    def test_timeout_term_path_cleans_up_process_group_without_orphan(self) -> None:
        with mock.patch.object(runner, "TERMINATION_GRACE_SECONDS", 0.2):
            capture = runner._execute_bounded(
                [sys.executable, "-c", "import subprocess,time;subprocess.Popen(['sleep','10']);time.sleep(10)"],
                {}, ROOT, 0.1,
            )
        self.assertTrue(capture["timed_out"])
        self.assertTrue(capture["term_sent"])
        self.assertFalse(capture["kill_sent"])
        self.assertTrue(capture["process_group_gone"])

    def test_timeout_kill_path_is_bounded_and_leaves_no_process_group(self) -> None:
        script = (
            "import signal,subprocess,sys,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "subprocess.Popen([sys.executable,'-c',"
            "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)']);"
            "time.sleep(10)"
        )
        started = time.monotonic()
        with mock.patch.object(runner, "TERMINATION_GRACE_SECONDS", 0.2):
            capture = runner._execute_bounded([sys.executable, "-c", script], {}, ROOT, 0.1)
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(capture["timed_out"])
        self.assertTrue(capture["term_sent"])
        self.assertTrue(capture["kill_sent"])
        self.assertTrue(capture["process_group_gone"])

    def test_visibility_is_cleared_before_uuid_isolation(self) -> None:
        row = first_row()
        base = {name: "foreign" for name in runner.VISIBILITY_NAMES}
        base["PATH"] = "/usr/bin"
        base["LD_LIBRARY_PATH"] = "/foreign/lib"
        base[runner.PHASE5_ALLOWED_TARGET_PIDS_ENV] = "1325127"
        environment = runner._execution_environment(row["row_id"], row["target"], base)
        self.assertEqual(environment["ROCR_VISIBLE_DEVICES"], contracts.expected_device(row["target"])["gpu_uuid"])
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/opt/rocm/core-7.14/lib")
        self.assertEqual(environment["SLLM_ENGINE_PERFORMANCE_ROW"], row["row_id"])
        self.assertEqual(set(runner.VISIBILITY_NAMES).intersection(environment), {"ROCR_VISIBLE_DEVICES"})
        self.assertNotIn(runner.PHASE5_ALLOWED_TARGET_PIDS_ENV, environment)

    def test_phase5_target_pid_allowlist_parser_is_strict(self) -> None:
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        self.assertEqual(runner._parse_phase5_allowed_target_pids({}), ())
        self.assertEqual(runner._parse_phase5_allowed_target_pids({name: ""}), ())
        self.assertEqual(runner._parse_phase5_allowed_target_pids({name: "7,1325127"}), (7, 1325127))
        for value in ("0", "01", "+1", "-1", " 1", "1 ", "1,", ",1", "1,,2", "1,1", "１２", str(1 << 31)):
            with self.subTest(value=value), self.assertRaises(ContractError):
                runner._parse_phase5_allowed_target_pids({name: value})

    def test_unset_or_empty_allowlist_retains_zero_process_strictness(self) -> None:
        record = amd_smi_process_record(1325127)
        self.assertEqual(runner._allowed_process_observation([], ()), [])
        with self.assertRaises(ContractError):
            runner._allowed_process_observation([record], ())

    def test_fully_typed_inert_allowlisted_process_passes_inclusive_boundaries(self) -> None:
        pid = 1325127
        record = amd_smi_process_record(
            pid,
            vram_bytes=runner.MAX_ALLOWED_TARGET_VRAM_BYTES,
            gtt_bytes=runner.MAX_ALLOWED_TARGET_GTT_BYTES,
        )
        self.assertEqual(runner._validate_inert_allowed_process_record(record), pid)
        self.assertEqual(
            runner._allowed_process_observation([record], (pid,)),
            [
                {"allowlisted_pids": [pid]},
                {"record": record, "record_sha256": hashlib.sha256(canonical_bytes(record)).hexdigest()},
            ],
        )

    def test_actual_amd_smi_inert_process_shape_accepts_na_cu_occupancy_and_preserves_raw_audit(self) -> None:
        pid = 1325127
        record = amd_smi_process_record(pid, cu_occupancy="N/A")
        self.assertEqual(runner._validate_inert_allowed_process_record(record), pid)
        audit = runner._allowed_process_observation([record], (pid,))
        self.assertEqual(audit[1]["record"], record)
        self.assertEqual(
            audit[1]["record_sha256"],
            hashlib.sha256(canonical_bytes(record)).hexdigest(),
        )

    def test_cu_occupancy_rejects_negative_integer_and_all_other_strings_and_types(self) -> None:
        for value in (-1, "0", "1", "NA", "n/a", " N/A", "N/A ", "", 0.0, True, False, None, {}, []):
            record = amd_smi_process_record(1325127, cu_occupancy=value)
            with self.subTest(value=value), self.assertRaises(ContractError):
                runner._validate_inert_allowed_process_record(record)
        self.assertEqual(
            runner._validate_inert_allowed_process_record(amd_smi_process_record(1325127, cu_occupancy=7)),
            1325127,
        )

    def test_allowed_process_boundaries_nonzero_activity_and_malformed_records_fail_closed(self) -> None:
        pid = 1325127
        rejected = (
            amd_smi_process_record(pid, vram_bytes=runner.MAX_ALLOWED_TARGET_VRAM_BYTES + 1),
            amd_smi_process_record(pid, gtt_bytes=runner.MAX_ALLOWED_TARGET_GTT_BYTES + 1),
            amd_smi_process_record(pid, gfx_usage_ns=1),
        )
        malformed = amd_smi_process_record(pid)
        malformed["process_info"]["memory_usage"]["vram_mem"] = {"value": 0.0, "unit": "B"}  # type: ignore[index]
        for record in (*rejected, malformed):
            with self.subTest(record=record), self.assertRaises(ContractError):
                runner._allowed_process_observation([record], (pid,))

    def test_duplicate_and_unallowlisted_target_records_fail_closed(self) -> None:
        allowed = amd_smi_process_record(1325127)
        unallowed = amd_smi_process_record(44)
        with self.assertRaises(ContractError):
            runner._allowed_process_observation([allowed, copy.deepcopy(allowed)], (1325127,))
        with self.assertRaises(ContractError):
            runner._allowed_process_observation([allowed, unallowed], (1325127,))

    def test_pre_post_observation_preserves_raw_record_and_allowlist_for_audit(self) -> None:
        pid = 1325127
        record = amd_smi_process_record(pid)
        process_doc = [{"gpu": 0, "process_list": [record]}]
        metric = {"ecc_uncorrectable": 0}
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        patches = (
            mock.patch.object(runner, "_amd_smi_list_identity", return_value=({}, 0)),
            mock.patch.object(runner, "_static_evidence", return_value={}),
            mock.patch.object(runner, "_metric_evidence", return_value=(metric, {})),
            mock.patch.object(runner, "_run_json_command", return_value=process_doc),
            mock.patch.object(runner, "_child_process_ids", return_value=[]),
            mock.patch.dict(os.environ, {name: str(pid)}, clear=False),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            pre = runner.validate_observation(runner._amd_smi_observation("gfx1030", "pre"), "gfx1030", "pre")
            post = runner.validate_observation(runner._amd_smi_observation("gfx1030", "post"), "gfx1030", "post")
        self.assertEqual(pre, post)
        self.assertEqual(pre["process"]["gpu_processes"][0]["allowlisted_pids"], [pid])
        self.assertEqual(pre["process"]["gpu_processes"][1]["record"], record)
        self.assertEqual(
            pre["process"]["gpu_processes"][1]["record_sha256"],
            hashlib.sha256(canonical_bytes(record)).hexdigest(),
        )
        changed = copy.deepcopy(post)
        changed["process"]["gpu_processes"][1]["record"]["process_info"]["memory_usage"]["gtt_mem"]["value"] += 1
        self.assertNotEqual(pre, changed)
        with mock.patch.dict(os.environ, {name: str(pid)}, clear=False), self.assertRaises(ContractError):
            runner.validate_observation(changed, "gfx1030", "post")

    def test_pre_post_comparison_allows_only_non_authorizing_process_diagnostic_drift(self) -> None:
        pid = 1325127
        pre_record = amd_smi_process_record(pid, cu_occupancy="N/A")
        post_record = copy.deepcopy(pre_record)
        info = post_record["process_info"]
        info["cu_occupancy"] = 7  # type: ignore[index]
        info["evicted_time"]["value"] = 91  # type: ignore[index]
        info["usage"]["enc"]["value"] = 11  # type: ignore[index]
        info["sdma_usage"]["value"] = 13  # type: ignore[index]
        base = {
            "selected_device": contracts.expected_device("gfx1030"),
            "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
            "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
        }
        pre = copy.deepcopy(base)
        post = copy.deepcopy(base)
        pre["process"]["gpu_processes"] = runner._allowed_process_observation([pre_record], (pid,))  # type: ignore[index]
        post["process"]["gpu_processes"] = runner._allowed_process_observation([post_record], (pid,))  # type: ignore[index]
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        with mock.patch.dict(os.environ, {name: str(pid)}, clear=False):
            runner.validate_observation(pre, "gfx1030", "pre")
            runner.validate_observation(post, "gfx1030", "post")
        self.assertNotEqual(pre, post)
        self.assertTrue(runner._observations_have_stable_authorization(pre, post))

    def test_pre_post_comparison_allows_independently_validated_inert_context_presence_drift(self) -> None:
        allowed_pids = (1325127, 1325128)
        base_record = amd_smi_process_record(allowed_pids[0])

        def observation(record: dict[str, object]) -> dict[str, object]:
            return {
                "selected_device": contracts.expected_device("gfx1030"),
                "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
                "process": {
                    "available": True, "reliable": True, "state": "CLEAN",
                    "gpu_processes": runner._allowed_process_observation([record], allowed_pids),
                    "residual_runner_children": [],
                },
            }

        pre = observation(base_record)
        mutations = {
            "pid": lambda info: info.__setitem__("pid", allowed_pids[1]),
            "name": lambda info: info.__setitem__("name", "different"),
            "gtt_memory": lambda info: info["memory_usage"]["gtt_mem"].__setitem__("value", 64 * 1024),
            "vram_memory": lambda info: info["memory_usage"]["vram_mem"].__setitem__("value", 64 * 1024),
            "gfx": lambda info: info["usage"]["gfx"].__setitem__("value", 1),
        }
        for label, mutation in mutations.items():
            post = copy.deepcopy(pre)
            record = post["process"]["gpu_processes"][1]["record"]  # type: ignore[index]
            mutation(record["process_info"])
            post["process"]["gpu_processes"][1]["record_sha256"] = hashlib.sha256(  # type: ignore[index]
                canonical_bytes(record)
            ).hexdigest()
            with self.subTest(label=label):
                name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
                with mock.patch.dict(os.environ, {name: ",".join(str(pid) for pid in allowed_pids)}, clear=False):
                    if label == "gfx":
                        with self.assertRaises(ContractError):
                            runner.validate_observation(post, "gfx1030", "post")
                    else:
                        runner.validate_observation(post, "gfx1030", "post")
                        self.assertTrue(runner._observations_have_stable_authorization(pre, post))
        absent = observation(base_record)
        absent["process"]["gpu_processes"] = [{"allowlisted_pids": list(allowed_pids)}]  # type: ignore[index]
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        with mock.patch.dict(os.environ, {name: ",".join(str(pid) for pid in allowed_pids)}, clear=False):
            runner.validate_observation(absent, "gfx1030", "post")
        self.assertTrue(runner._observations_have_stable_authorization(pre, absent))

    def test_pre_post_phase_evidence_accepts_the_same_validated_inert_record(self) -> None:
        pid = 1325127
        record = amd_smi_process_record(pid)
        process_doc = [{"gpu": 0, "process_list": [record]}]
        static = {"target": "gfx1030"}
        metric = {"ecc_uncorrectable": 0, "throttle_status": "UNTHROTTLED"}
        violation = {"explicit_violation": False}
        vram = {"source": "amd-smi monitor -v"}
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        with mock.patch.object(runner, "_amd_smi_list_identity", return_value=({}, 0)), mock.patch.object(
            runner, "_static_evidence", return_value=static,
        ), mock.patch.object(runner, "_metric_evidence", return_value=(metric, violation)), mock.patch.object(
            runner, "_vram_auxiliary", return_value=vram,
        ), mock.patch.object(runner, "_run_json_command", return_value=process_doc), mock.patch.dict(
            os.environ, {name: str(pid)}, clear=False,
        ):
            pre = runner._amd_smi_phase_evidence("gfx1030", "pre")
            post = runner._amd_smi_phase_evidence("gfx1030", "post")
        self.assertEqual(pre, post)
        self.assertEqual(pre["process_state"], "CLEAN")

    def test_during_monitor_filters_only_validated_inert_allowlisted_record(self) -> None:
        allowed_pid = 1325127
        owned_pid = 123
        allowed = amd_smi_process_record(allowed_pid)
        owned = amd_smi_process_record(owned_pid, vram_bytes=8 * 1024 * 1024, gfx_usage_ns=5)
        process_doc = [{"gpu": 0, "process_list": [allowed, owned]}]
        loader = {
            "path_digest": "sha256:" + "a" * 64,
        }
        metric = {"ecc_uncorrectable": 0, "throttle_status": "UNTHROTTLED"}
        vram = {"source": "amd-smi monitor -v"}
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        with mock.patch.object(runner, "_amd_smi_list_identity", return_value=({}, 0)), mock.patch.object(
            runner, "_metric_evidence", return_value=(metric, {"explicit_violation": False}),
        ), mock.patch.object(runner, "_vram_auxiliary", return_value=vram), mock.patch.object(
            runner, "_run_json_command", return_value=process_doc,
        ), mock.patch.object(runner, "_process_group_members", return_value={owned_pid}), mock.patch.object(
            runner, "_loader_evidence", return_value=loader,
        ), mock.patch.dict(os.environ, {name: str(allowed_pid)}, clear=False):
            sample = runner._amd_smi_monitor_sample("gfx1030", owned_pid, owned_pid)
        self.assertEqual(sample["process"], {"state": "OWNED", "pids": [owned_pid]})

    def test_bounded_child_never_receives_control_plane_allowlist(self) -> None:
        name = runner.PHASE5_ALLOWED_TARGET_PIDS_ENV
        capture = runner._execute_bounded(
            [sys.executable, "-c", f"import os;print(os.environ.get({name!r}, 'missing'))"],
            {name: "1325127"}, ROOT, 5,
        )
        self.assertEqual(capture["exit_code"], 0)
        self.assertEqual(capture["stdout"], b"missing\n")
        self.assertEqual(capture["stderr"], b"")

    @mock.patch("run_engine_performance.subprocess.run")
    def test_amd_smi_version_parses_current_tool_output(self, run: mock.Mock) -> None:
        run.return_value = mock.Mock(
            returncode=0,
            stdout=(
                b"AMDSMI Tool: 26.5.0+2b22ab01 | AMDSMI Library version: 26.5.0 | "
                b"ROCm version: 7.14.0 | Git version: 2b22ab01\n"
            ),
            stderr=b"",
        )
        self.assertEqual(
            runner._amd_smi_version(),
            {
                "tool_version": "26.5.0+2b22ab01",
                "library_version": "26.5.0",
                "rocm_version": "7.14.0",
            },
        )

    def test_metric_telemetry_retries_one_transient_gap_but_fails_persistent_or_nontelemetry_errors(self) -> None:
        success = ({"ecc_uncorrectable": 0}, {"explicit_violation": False})
        transient = ContractError("AMD-SMI socket power is missing or malformed")
        with mock.patch.object(runner, "_metric_evidence_once", side_effect=[transient, success]) as once, mock.patch.object(
            runner.time, "sleep",
        ) as sleep:
            self.assertEqual(runner._metric_evidence("gfx1201", 1), success)
            self.assertEqual(once.call_count, 2)
            sleep.assert_called_once_with(runner.METRIC_TELEMETRY_RETRY_SECONDS)
        with mock.patch.object(runner, "_metric_evidence_once", side_effect=transient) as once, mock.patch.object(
            runner.time, "sleep",
        ):
            with self.assertRaisesRegex(ContractError, "socket power"):
                runner._metric_evidence("gfx1201", 1)
            self.assertEqual(once.call_count, runner.METRIC_TELEMETRY_ATTEMPTS)
        with mock.patch.object(
            runner, "_metric_evidence_once", side_effect=ContractError("AMD-SMI metric evidence selected the wrong device"),
        ) as once:
            with self.assertRaisesRegex(ContractError, "wrong device"):
                runner._metric_evidence("gfx1201", 1)
            once.assert_called_once()
    def test_missing_nonmonotonic_overflow_wrong_math_and_zero_short_are_rejected(self) -> None:
        self.assert_invalid(lambda value: value["measured"]["samples"].pop())
        self.assert_invalid(lambda value: value["measured"]["samples"][0]["events"].__setitem__("stop_ns", value["measured"]["samples"][0]["events"]["first_token_ns"] - 1))
        self.assert_invalid(lambda value: value["measured"]["samples"][0]["events"].__setitem__("first_token_ns", 1 << 63))
        self.assert_invalid(lambda value: value["measured"]["samples"][0]["derived"].__setitem__("ttft_ns", 999999))
        self.assert_invalid(lambda value: value["measured"]["samples"][0]["tokens"].__setitem__("generated_token_ids", []))

    def test_stale_identity_wrong_model_target_and_duplicate_samples_are_rejected(self) -> None:
        self.assert_invalid(lambda value: value["identities"]["model"].__setitem__("lock_fingerprint", "sha256:" + "f" * 64))
        self.assert_invalid(lambda value: value["identities"].__setitem__("target", "gfx1201"))
        self.assert_invalid(lambda value: value["measured"]["samples"].__setitem__(1, copy.deepcopy(value["measured"]["samples"][0])))

    def test_stop_audit_loader_cleanup_throttle_and_interference_are_rejected(self) -> None:
        for mutation in (
            lambda value: value["measured"]["samples"][0]["stop"].__setitem__("token_id", 7),
            lambda value: value["audit"].__setitem__("fallback_used", True),
            lambda value: value["audit"].__setitem__("request_model_load_count", 1),
            lambda value: value["audit"].__setitem__("all_dispatches_hip", False),
            lambda value: value["cleanup"].__setitem__("all_requests_dropped", False),
            lambda value: value["session_cleanup"].__setitem__("retryable_cleanup", 1),
        ):
            with self.subTest(mutation=mutation):
                self.assert_invalid(mutation)

    def test_direct_contract_rejects_load_memory_control_and_timed_path_drift(self) -> None:
        for mutation in (
            lambda value: value["model_load"].__setitem__("start_ns", 1),
            lambda value: value["memory"].__setitem__("resident_vram_bytes", 1000),
            lambda value: value["memory"]["after_model_drop"]["model_resident"].__setitem__("current_bytes", 1),
            lambda value: value["correctness_control"]["tokens"].__setitem__("generated_token_ids", [8]),
            lambda value: value["measured"]["samples"][0].__setitem__("timing_instrumentation", "off"),
            lambda value: value["cleanup"].__setitem__("request_cleanup_count", 13),
        ):
            with self.subTest(mutation=mutation):
                self.assert_invalid(mutation)

    def test_health_missing_unavailable_dirty_and_wrong_device_are_rejected(self) -> None:
        observation = {
            "selected_device": contracts.expected_device("gfx1030"),
            "health": {"available": True, "reliable": True, "state": "OK", "ras_uncorrectable_count": 0},
            "process": {"available": True, "reliable": True, "state": "CLEAN", "gpu_processes": [], "residual_runner_children": []},
        }
        for mutation in (
            lambda value: value.pop("health"),
            lambda value: value["health"].__setitem__("available", False),
            lambda value: value["process"].__setitem__("state", "DIRTY"),
            lambda value: value["selected_device"].__setitem__("target", "gfx1201"),
        ):
            changed = copy.deepcopy(observation)
            mutation(changed)
            with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                runner.validate_observation(changed, "gfx1030", "pre")

    def test_raw_artifact_tamper_is_not_silently_accepted(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-runner-") as directory:
            path = Path(directory) / "raw.json"
            path.write_text('{"broken":', encoding="utf-8")
            with self.assertRaises(ContractError):
                contracts.read_json(path, "tampered raw result")

    def test_setup_failure_writes_durable_schema_valid_fail_manifest(self) -> None:
        row = first_row()
        with tempfile.TemporaryDirectory(prefix="sllm-performance-failure-") as directory:
            root = Path(directory)
            manifest = runner.run_row(
                row["row_id"], root / "missing-binary", root / "missing-lock", root / "missing-cache",
                root / "output", build_manifest=root / "missing-build", repo=ROOT,
            )
            self.assertEqual(manifest["state"], "FAIL")
            self.assertTrue((root / "output/report.json").is_file())
            contracts.schema_validate(manifest, contracts.DIRECT_SCHEMA_PATH, "durable failure manifest", "manifest")

    def test_mocked_monitor_valid_paths_and_compact_summary_pass(self) -> None:
        target = "gfx1030"
        pre = evidence_for(target)
        post = copy.deepcopy(pre)
        capture = monitor_capture_for(target)
        evidence = runner._build_evidence(
            pre, post, capture, target,
            {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "26.5.0+test", "library_version": "26.5.0", "rocm_version": "7.14.0"},
        )
        contracts.schema_validate(evidence, contracts.DIRECT_SCHEMA_PATH, "mocked evidence", "evidence")
        self.assertEqual(evidence["during"]["sample_count"], 1)
        self.assertEqual(evidence["checks"]["explicit_violation"], False)
        runner._validate_loader(evidence["during"]["loader"])

    def test_mocked_monitor_missing_foreign_target_drift_and_explicit_violation_fail_closed(self) -> None:
        target = "gfx1030"
        base = monitor_capture_for(target)
        with self.assertRaises(ContractError):
            runner._validate_monitor_capture({"monitor": {"samples": [], "errors": [], "loader": None}}, target)
        foreign = copy.deepcopy(base)
        foreign["monitor"]["samples"][0]["process"]["pids"] = [999]
        with self.assertRaises(ContractError):
            runner._validate_monitor_capture(foreign, target, 123)
        throttled = copy.deepcopy(base)
        throttled["monitor"]["samples"][0]["violation"]["explicit_violation"] = True
        with self.assertRaises(ContractError):
            runner._validate_monitor_capture(throttled, target, 123)
        too_hot = copy.deepcopy(base)
        too_hot["monitor"]["samples"][0]["metric"]["temperature_c"]["hotspot"] = 100
        with self.assertRaises(ContractError):
            runner._build_evidence(evidence_for(target), evidence_for(target), too_hot, target, {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"})
        at_power_limit = copy.deepcopy(base)
        at_power_limit["monitor"]["samples"][0]["metric"]["power_w"] = 250
        runner._build_evidence(evidence_for(target), evidence_for(target), at_power_limit, target, {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"})
        bounded_telemetry_overshoot = copy.deepcopy(base)
        bounded_telemetry_overshoot["monitor"]["samples"][0]["metric"]["power_w"] = 255
        runner._build_evidence(evidence_for(target), evidence_for(target), bounded_telemetry_overshoot, target, {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"})
        above_published_maximum = copy.deepcopy(base)
        above_published_maximum["monitor"]["samples"][0]["metric"]["power_w"] = 275
        runner._build_evidence(evidence_for(target), evidence_for(target), above_published_maximum, target, {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"})
        drift = evidence_for(target)
        drift["static"]["gpu_bdf"] = "0000:07:00.0"
        with self.assertRaises(ContractError):
            runner._validate_phase_evidence(drift, target, "pre")
        loader_bad = copy.deepcopy(base)
        loader_bad["monitor"]["loader"]["resolved_paths"][0] = "/opt/rocm/foreign/libamdhip64.so"
        with self.assertRaises(ContractError):
            runner._validate_loader(loader_bad["monitor"]["loader"])

    def test_mocked_monitor_malformed_evidence_is_not_a_pass(self) -> None:
        target = "gfx1030"
        pre = evidence_for(target)
        post = copy.deepcopy(pre)
        capture = monitor_capture_for(target)
        del capture["monitor"]["samples"][0]["vram_auxiliary"]
        with self.assertRaises(ContractError):
            runner._build_evidence(pre, post, capture, target, {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"})

    def test_monitor_accepts_pinned_rocm_library_lazy_addition(self) -> None:
        capture = monitor_capture_for("gfx1030")
        first = copy.deepcopy(capture["monitor"]["samples"][0])
        second = copy.deepcopy(first)
        second["timestamp_ns"] += 1_000_000_000
        extra = "/opt/rocm/core-7.14/lib/libamd_comgr.so.3.3.0"
        second_loader = copy.deepcopy(capture["monitor"]["loader"])
        second_loader["resolved_paths"].append(extra)
        second_loader["resolved_paths"].sort()
        second_loader["library_digests"][extra] = "d" * 64
        second_loader["path_digest"] = "sha256:" + hashlib.sha256(
            canonical_bytes(second_loader["resolved_paths"])
        ).hexdigest()
        second["loader_path_digest"] = second_loader["path_digest"]
        capture["monitor"]["samples"] = [first, second]
        capture["monitor"]["loader"] = copy.deepcopy(second_loader)
        capture["monitor"]["loaders"] = [copy.deepcopy(capture["monitor"]["loaders"][0]), second_loader]
        samples, info = runner._validate_monitor_capture(capture, "gfx1030", 123)
        self.assertEqual(len(samples), 2)
        self.assertEqual(info["loader_path_digest"], second["loader_path_digest"])
        self.assertEqual(len(info["loaders"]), 2)
        evidence = runner._build_evidence(
            evidence_for("gfx1030"), evidence_for("gfx1030"), capture, "gfx1030",
            {"path": runner.AMD_SMI_EXECUTABLE, "tool_version": "test", "library_version": "test", "rocm_version": "7.14.0"},
        )
        self.assertNotEqual(
            evidence["during"]["first"]["loader_path_digest"],
            evidence["during"]["loader"]["path_digest"],
        )
        self.assertEqual(len(evidence["during"]["loaders"]), 2)

    def test_monitor_acquisition_waits_only_for_context_and_then_uses_one_second_lane(self) -> None:
        stop = __import__("threading").Event()
        valid = monitor_capture_for("gfx1030")["monitor"]["samples"][0]
        calls = 0

        def provider(_target: str, _pid: int, _group: int) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise runner.MonitorNotReady("context not registered")
            stop.set()
            return dict(valid)

        with mock.patch.object(runner, "_proc_relationship", return_value=(1, 1)):
            capture = runner._monitor_loop("gfx1030", 123, provider, stop)
        self.assertEqual(capture["acquisition"], "acquired")
        self.assertEqual(capture["errors"], [])
        self.assertEqual(len(capture["samples"]), 1)

        stop = __import__("threading").Event()
        calls = 0

        def two_samples(_target: str, _pid: int, _group: int) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 2:
                stop.set()
            return dict(valid)

        with mock.patch.object(runner, "_proc_relationship", return_value=(1, 1)):
            capture = runner._monitor_loop("gfx1030", 123, two_samples, stop)
        self.assertEqual(capture["errors"], [])
        self.assertEqual(len(capture["samples"]), 2)
        self.assertGreaterEqual(
            capture["samples"][1]["timestamp_ns"] - capture["samples"][0]["timestamp_ns"],
            1_000_000_000,
        )

    def test_monitor_foreign_error_and_process_exit_never_become_success(self) -> None:
        stop = __import__("threading").Event()

        def foreign(_target: str, _pid: int, _group: int) -> dict[str, object]:
            raise ContractError("foreign PID")

        with mock.patch.object(runner, "_proc_relationship", return_value=(1, 1)):
            capture = runner._monitor_loop("gfx1030", 123, foreign, stop)
        self.assertEqual(capture["samples"], [])
        self.assertTrue(capture["errors"])

    def test_monitor_allows_only_signaled_teardown_after_loader_acquisition(self) -> None:
        threading = __import__("threading")
        valid = monitor_capture_for("gfx1030")["monitor"]["samples"][0]

        stop = threading.Event()
        calls = 0

        def teardown(_target: str, _pid: int, _group: int) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return dict(valid)
            stop.set()
            raise runner.MonitorNotReady("loader mappings disappeared during teardown")

        with mock.patch.object(runner, "_proc_relationship", return_value=(1, 1)):
            capture = runner._monitor_loop("gfx1030", 123, teardown, stop)
        self.assertEqual(capture["acquisition"], "acquired")
        self.assertEqual(capture["errors"], [])
        self.assertEqual(len(capture["samples"]), 1)

        stop = threading.Event()
        calls = 0

        def live_drift(_target: str, _pid: int, _group: int) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return dict(valid)
            raise runner.MonitorNotReady("loader mappings disappeared while live")

        with mock.patch.object(runner, "_proc_relationship", return_value=(1, 1)):
            capture = runner._monitor_loop("gfx1030", 123, live_drift, stop)
        self.assertEqual(capture["acquisition"], "acquired")
        self.assertTrue(capture["errors"])

        stop = __import__("threading").Event()

        def not_ready(_target: str, _pid: int, _group: int) -> dict[str, object]:
            raise runner.MonitorNotReady("context not registered")

        with mock.patch.object(runner, "_proc_relationship", side_effect=[(1, 1), None]):
            capture = runner._monitor_loop("gfx1030", 123, not_ready, stop)
        self.assertEqual(capture["samples"], [])
        self.assertTrue(capture["errors"])

    def test_immutable_build_manifest_binds_source_tree_target_rocm_and_binary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-build-identity-") as directory:
            root = Path(directory)
            binary = root / "engine"
            binary.write_bytes(b"engine")
            binary.chmod(0o755)
            revision = __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            tree = __import__("subprocess").check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
            document = {
                "schema_version": "sllm-build-identity-v2", "source_root": str(ROOT),
                "source_base_revision": revision, "semantic_tree": tree, "build_inputs_digest": "sha256:" + "1" * 64,
                "build_configuration": contracts.expected_build_configuration("gfx1030"),
                "target": "gfx1030", "backend": "hip", "rocm_release": "7.14.0", "rocm_root": "/opt/rocm/core-7.14",
                "binary_sha256": hashlib.sha256(b"engine").hexdigest(),
            }
            path = root / "build.json"
            path.write_bytes(canonical_bytes(document))
            validated, _digest = runner._validate_build_manifest(path, binary, "gfx1030", ROOT)
            self.assertEqual(validated["semantic_tree"], tree)
            document["target"] = "gfx1201"
            path.write_bytes(canonical_bytes(document))
            with self.assertRaises(ContractError):
                runner._validate_build_manifest(path, binary, "gfx1030", ROOT)

            document["target"] = "gfx1030"
            document["semantic_tree"] = "f" * 40
            path.write_bytes(canonical_bytes(document))
            with self.assertRaises(ContractError):
                runner._validate_build_manifest(path, binary, "gfx1030", ROOT)

            document["semantic_tree"] = tree
            document["binary_sha256"] = "f" * 64
            path.write_bytes(canonical_bytes(document))
            with self.assertRaises(ContractError):
                runner._validate_build_manifest(path, binary, "gfx1030", ROOT)

    def test_build_manifest_rejects_schema_and_configuration_tampering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-performance-build-config-") as directory:
            root = Path(directory)
            binary = root / "engine"
            binary.write_bytes(b"engine")
            binary.chmod(0o755)
            subprocess = __import__("subprocess")
            revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
            tree = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD^{tree}"], text=True).strip()
            valid = {
                "schema_version": "sllm-build-identity-v2", "source_root": str(ROOT),
                "source_base_revision": revision, "semantic_tree": tree, "build_inputs_digest": "sha256:" + "1" * 64,
                "build_configuration": contracts.expected_build_configuration("gfx1030"),
                "target": "gfx1030", "backend": "hip", "rocm_release": "7.14.0", "rocm_root": "/opt/rocm/core-7.14",
                "binary_sha256": hashlib.sha256(b"engine").hexdigest(),
            }
            mutations = (
                lambda value: value.pop("schema_version"),
                lambda value: value.__setitem__("unknown", "value"),
                lambda value: value["build_configuration"].pop("cargo_profile"),
                lambda value: value["build_configuration"].__setitem__("unknown", "value"),
                lambda value: value["build_configuration"].__setitem__("CMAKE_HIP_ARCHITECTURES", "gfx1201"),
                lambda value: value["build_configuration"].__setitem__("ROCM_PATH", "/tmp/rocm"),
                lambda value: value["build_configuration"].__setitem__("cargo_command", "cargo build --release"),
            )
            path = root / "build.json"
            for mutation in mutations:
                document = copy.deepcopy(valid)
                mutation(document)
                path.write_bytes(canonical_bytes(document))
                with self.subTest(mutation=mutation), self.assertRaises(ContractError):
                    runner._validate_build_manifest(path, binary, "gfx1030", ROOT)

    def test_build_config_records_reject_missing_unknown_duplicate_unsafe_and_wrong_values(self) -> None:
        parser = __import__("create_engine_build_identity")._parse_build_configuration
        records = build_config_records("gfx1030")
        self.assertEqual(parser(records, "gfx1030"), contracts.expected_build_configuration("gfx1030"))
        mutations = (
            records[:-1],
            records + ["UNKNOWN=value"],
            records + [records[0]],
            [record if not record.startswith("cargo_profile=") else "cargo_profile=release\nunsafe" for record in records],
            [record if not record.startswith("cargo_profile=") else "cargo_profile=" for record in records],
            [record if not record.startswith("cargo_profile=") else "cargo_profile" for record in records],
            [record if not record.startswith("cargo_command=") else "cargo_command=cargo build --release" for record in records],
            [record if not record.startswith("ROCM_PATH=") else "ROCM_PATH=/tmp/rocm" for record in records],
            [record if not record.startswith("CMAKE_HIP_ARCHITECTURES=") else "CMAKE_HIP_ARCHITECTURES=gfx1201" for record in records],
        )
        for changed in mutations:
            with self.subTest(changed=changed), self.assertRaises(ContractError):
                parser(changed, "gfx1030")

    def test_build_inputs_digest_binds_every_configuration_record(self) -> None:
        digest_helper = __import__("create_engine_build_identity")._build_inputs_digest
        with tempfile.TemporaryDirectory(prefix="sllm-build-digest-") as directory:
            root = Path(directory)
            source = root / "source.rs"
            source.write_bytes(b"source")
            configuration = contracts.expected_build_configuration("gfx1030")
            baseline = digest_helper(root, [source], configuration)
            for key in contracts.BUILD_CONFIGURATION_KEYS:
                changed = dict(configuration)
                changed[key] += "-changed"
                with self.subTest(key=key):
                    self.assertNotEqual(baseline, digest_helper(root, [source], changed))

    def test_build_identity_helper_captures_dirty_candidate_without_staging_real_index(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-build-helper-") as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            subprocess = __import__("subprocess")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.email", "test@example.invalid"], check=True)
            subprocess.run(["git", "-C", str(root), "config", "user.name", "test"], check=True)
            (root / "tracked.rs").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked.rs"], check=True)
            subprocess.run(["git", "-C", str(root), "commit", "-qm", "base"], check=True)
            base_tree = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()
            (root / "tracked.rs").write_text("candidate\n", encoding="utf-8")
            (root / "new.rs").write_text("new candidate\n", encoding="utf-8")
            binary = Path(directory) / "engine"
            binary.write_bytes(b"engine")
            binary.chmod(0o755)
            output = Path(directory) / "build.json"
            document = create_identity(
                root, output, binary, "gfx1030", "hip", "7.14.0", "/opt/rocm/core-7.14",
                None, [], build_config_records("gfx1030"),
            )
            self.assertNotEqual(document["semantic_tree"], base_tree)
            self.assertEqual(document["build_configuration"], contracts.expected_build_configuration("gfx1030"))
            self.assertEqual(subprocess.check_output(["git", "-C", str(root), "cat-file", "-t", document["semantic_tree"]], text=True).strip(), "tree")
            second = create_identity(
                root, Path(directory) / "build-gfx1201.json", binary, "gfx1201", "hip", "7.14.0", "/opt/rocm/core-7.14",
                None, [], build_config_records("gfx1201"),
            )
            self.assertNotEqual(document["build_inputs_digest"], second["build_inputs_digest"])
            (root / "tracked.rs").write_text("second candidate\n", encoding="utf-8")
            third = create_identity(
                root, Path(directory) / "build-source-changed.json", binary, "gfx1030", "hip", "7.14.0", "/opt/rocm/core-7.14",
                None, [], build_config_records("gfx1030"),
            )
            self.assertNotEqual(document["build_inputs_digest"], third["build_inputs_digest"])
            status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
            self.assertIn(" M tracked.rs", status)
            self.assertIn("?? new.rs", status)

    def test_host_fake_invokes_the_real_cli_spelling_and_emits_the_rust_shape(self) -> None:
        row = first_row()
        result = result_for(row)
        with tempfile.TemporaryDirectory(prefix="sllm-performance-cli-fake-") as directory:
            root = Path(directory)
            fake = root / "sllm-fake"
            fake.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "assert sys.argv[1] == 'benchmark'\n"
                "assert '--lane' in sys.argv and sys.argv[sys.argv.index('--lane') + 1] == 'direct'\n"
                "assert '--input-token-ids' in sys.argv\n"
                "print(json.dumps(" + repr(result) + ", separators=(',', ':')))\n",
                encoding="utf-8",
            )
            fake.chmod(fake.stat().st_mode | 0o111)
            command = runner._expected_command(fake, row, root / "lock.json", root / "cache")
            capture = runner._execute_bounded(command, {}, root, row["timeout_seconds"])
            self.assertEqual(capture["exit_code"], 0)
            self.assertEqual(capture["stderr"], b"")
            cli_result = contracts.parse_json_bytes(capture["stdout"], "fake CLI result")
            contracts.validate_cli_result(cli_result, row)
            self.assertEqual(command[1:4], ["benchmark", "--lane", "direct"])
            self.assertEqual(cli_result["lane"], "direct")


if __name__ == "__main__":
    unittest.main()
