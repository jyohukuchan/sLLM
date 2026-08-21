from __future__ import annotations

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import run_phase36_session_a as runner  # noqa: E402


def producer_report(count: int, *, target: str = "gfx942", fallback: bool = False) -> bytes:
    return json.dumps(
        {
            "schema_version": "test",
            "state": "PASS",
            "target": target,
            "selected_backend": "hip",
            "fallback_allowed": fallback,
            "fallback_used": fallback,
            "operations": count,
            "dispatch_count": count,
            "cleanup_retryable": 0,
            "cleanup_durable": 0,
        },
        separators=(",", ":"),
    ).encode()


class SessionARunnerTests(unittest.TestCase):
    def test_dry_run_has_exact_99_logical_cases_and_does_not_spawn(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(runner.subprocess, "run") as invoke:
            summary = runner.run_session_a(
                bin_dir=Path(directory),
                device_index=0,
                target="gfx942",
                output_dir=Path(directory) / "out",
                dry_run=True,
            )
        invoke.assert_not_called()
        self.assertEqual(summary["state"], "DRY_RUN")
        self.assertEqual(summary["expected_cases"], 99)
        self.assertEqual(summary["selected_cases"], 0)
        self.assertEqual([row["expected_cases"] for row in summary["families"]], [2, 17, 21, 8, 19, 16, 6, 7, 3])
        self.assertIn("--phase12-subset", summary["families"][1]["command"])
        self.assertIn("--phase12-subset", summary["families"][5]["command"])
        self.assertIn("--phase12-subset", summary["families"][8]["command"])

    def test_family_execution_parses_count_and_calls_mocked_producer(self) -> None:
        row = runner.matrix(Path("/tmp/bin"), 3, "gfx942")[1]
        report = json.loads(producer_report(17))
        report["device_index"] = 3
        report["cases"] = [{"m": 17, "dispatch_count": 1, "kernel_symbol": "matmul.hipblas.gemm_ex.v2", "device_symbol": "hipblasGemmEx"} for _ in range(17)]
        completed = runner.subprocess.CompletedProcess(row["command"], 0, json.dumps(report).encode(), b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_run_process", return_value=completed) as invoke:
            result = runner._run_family(row, Path(directory), dry_run=False)
        self.assertEqual(result["state"], "PASS")
        self.assertEqual(result["selected_cases"], 17)
        self.assertEqual(result["dispatch_count"], 17)
        invoke.assert_called_once()
        self.assertEqual(invoke.call_args.args, (row["command"],))

    def test_target_rejection_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(runner.SessionAError):
                runner.run_session_a(
                    bin_dir=Path(directory),
                    device_index=0,
                    target="gfx1201",
                    output_dir=Path(directory) / "out",
                    dry_run=True,
                )

    def test_fallback_and_zero_dispatch_are_rejected(self) -> None:
        with self.assertRaises(runner.SessionAError):
            runner._validate_common(json.loads(producer_report(2, fallback=True)), "fp8-matmul", 2)
        with self.assertRaises(runner.SessionAError):
            runner._validate_common({"state": "PASS", "target": "gfx942", "operations": 2, "dispatch_count": 0}, "fp8-matmul", 2)
        with self.assertRaises(runner.SessionAError):
            runner._validate_common({"state": "PASS", "target": "gfx942", "operations": 2, "dispatch_count": 2, "cleanup": {"durable_quarantine": 1}}, "fp8-matmul", 2)

    def test_missing_dispatch_evidence_is_rejected(self) -> None:
        with self.assertRaises(runner.SessionAError):
            runner._validate_common({"state": "PASS", "target": "gfx942", "operations": 2, "cleanup_retryable": 0, "cleanup_durable": 0}, "elementwise", 2)

    def test_positive_no_fallback_markers_are_not_treated_as_fallback(self) -> None:
        document = {
            "state": "PASS",
            "target": "gfx942",
            "device_index": 0,
            "gpu_execution": True,
            "cases": [{"no_fallback_observed": True}],
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "zero_after_shutdown": True},
        }
        self.assertEqual(runner._validate_common(document, "kv-state", 1, 0), 1)

    def test_producer_device_index_mismatch_is_rejected(self) -> None:
        report = json.loads(producer_report(2))
        report["device_index"] = 1
        with self.assertRaises(runner.SessionAError):
            runner._validate_common(report, "elementwise", 2, expected_device_index=0)

    def test_rmsnorm_wrong_output_and_malformed_metadata_fail_closed(self) -> None:
        row = runner.matrix(Path("/tmp/bin"), 0, "gfx942")[7]
        valid = {"output": b"wrong", "dispatch_count": 1, "kernel_id": 2, "resource_counts": {"allocation_count": 3, "copy_count": 3, "kernel_count": 1}}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_rmsnorm_payload", return_value=(b"request", b"oracle", b"input")), patch.object(runner, "parse_response", return_value=valid), patch.object(runner, "_run_process", return_value=runner.subprocess.CompletedProcess(row["command"], 0, b"runtime", b"")):
            result = runner._run_family(row, Path(directory), dry_run=False)
        self.assertEqual(result["state"], "FAIL")
        self.assertIn("byte-match", result["error"])
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_rmsnorm_payload", return_value=(b"request", b"oracle", b"input")), patch.object(runner, "parse_response", side_effect=runner.RunnerError("bad header")), patch.object(runner, "_run_process", return_value=runner.subprocess.CompletedProcess(row["command"], 0, b"runtime", b"")):
            result = runner._run_family(row, Path(directory), dry_run=False)
        self.assertEqual(result["state"], "FAIL")
        self.assertIn("metadata", result["error"])

    def test_rmsnorm_wrong_kernel_provider_fails_closed(self) -> None:
        row = runner.matrix(Path("/tmp/bin"), 0, "gfx942")[7]
        response = {"output": b"oracle", "dispatch_count": 1, "kernel_id": 1, "resource_counts": {"allocation_count": 3, "copy_count": 3, "kernel_count": 1}}
        completed = runner.subprocess.CompletedProcess(row["command"], 0, b"runtime", b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_rmsnorm_payload", return_value=(b"request", b"oracle", b"input")), patch.object(runner, "parse_response", return_value=response), patch.object(runner, "_run_process", return_value=completed):
            result = runner._run_family(row, Path(directory), dry_run=False)
        self.assertEqual(result["state"], "FAIL")
        self.assertIn("kernel provider contract", result["error"])

    def test_rmsnorm_wrong_resource_provider_fails_closed(self) -> None:
        row = runner.matrix(Path("/tmp/bin"), 0, "gfx942")[7]
        response = {"output": b"oracle", "dispatch_count": 1, "kernel_id": 2, "resource_counts": {"allocation_count": 4, "copy_count": 3, "kernel_count": 1}}
        completed = runner.subprocess.CompletedProcess(row["command"], 0, b"runtime", b"")
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "_rmsnorm_payload", return_value=(b"request", b"oracle", b"input")), patch.object(runner, "parse_response", return_value=response), patch.object(runner, "_run_process", return_value=completed):
            result = runner._run_family(row, Path(directory), dry_run=False)
        self.assertEqual(result["state"], "FAIL")
        self.assertIn("provider resource contract", result["error"])

    def test_fail_fast_keeps_bounded_summary_without_scheduling_later_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            rows = runner.matrix(Path(directory), 0, "gfx942")
            first = rows[0]
            failed = {
                "family": first["family"],
                "binary": first["binary"],
                "expected_cases": first["expected_cases"],
                "command": first["command"],
                "state": "FAIL",
                "selected_cases": 0,
                "dispatch_count": 0,
                "fallback_used": False,
                "cleanup_retryable": 0,
                "cleanup_durable": 0,
                "error": "producer failed",
            }
            with patch.object(runner, "_run_family", return_value=failed) as invoke:
                summary = runner.run_session_a(
                    bin_dir=Path(directory),
                    device_index=0,
                    target="gfx942",
                    output_dir=Path(directory) / "out",
                    dry_run=False,
                )
        invoke.assert_called_once()
        self.assertEqual(summary["state"], "FAIL")
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(len(summary["families"]), 9)
        self.assertEqual(summary["families"][0]["error"], "producer failed")
        self.assertTrue(all(row["error"] == runner.NOT_SCHEDULED_ERROR for row in summary["families"][1:]))

    def test_timeout_kills_process_group_and_reaps_process(self) -> None:
        class FakeProcess:
            pid = 4242
            returncode = -signal.SIGKILL

            def __init__(self) -> None:
                self.communicate_calls: list[float | None] = []

            def communicate(self, *, input: bytes | None = None, timeout: float | None = None) -> tuple[bytes, bytes]:
                del input
                self.communicate_calls.append(timeout)
                if timeout == runner.TIMEOUT_SECONDS or timeout == runner.PROCESS_GROUP_TERM_GRACE_SECONDS:
                    raise runner.subprocess.TimeoutExpired(["producer"], timeout)
                return b"", b""

        process = FakeProcess()
        with patch.object(runner.subprocess, "Popen", return_value=process), patch.object(runner.os, "killpg") as killpg:
            with self.assertRaisesRegex(runner.SessionAError, "process group terminated and reaped"):
                runner._run_process(["producer"])
        self.assertEqual(killpg.call_args_list, [
            call(process.pid, signal.SIGTERM),
            call(process.pid, signal.SIGKILL),
        ])
        self.assertEqual(process.communicate_calls, [runner.TIMEOUT_SECONDS, runner.PROCESS_GROUP_TERM_GRACE_SECONDS, runner.PROCESS_GROUP_KILL_GRACE_SECONDS])

    def test_timeout_reports_process_group_cleanup_failure(self) -> None:
        class FakeProcess:
            pid = 4343
            returncode = -signal.SIGKILL

            def communicate(self, *, input: bytes | None = None, timeout: float | None = None) -> tuple[bytes, bytes]:
                del input
                if timeout == runner.TIMEOUT_SECONDS:
                    raise runner.subprocess.TimeoutExpired(["producer"], timeout)
                return b"", b""

        process = FakeProcess()
        with patch.object(runner.subprocess, "Popen", return_value=process), patch.object(
            runner.os, "killpg", side_effect=PermissionError("denied")
        ):
            with self.assertRaisesRegex(runner.SessionAError, "process-group cleanup failed"):
                runner._run_process(["producer"])

    def test_schema_validator_accepts_dry_run_summary(self) -> None:
        try:
            from jsonschema import Draft202012Validator
        except ImportError:  # pragma: no cover - host dependency is pinned
            self.skipTest("jsonschema is not installed")
        with tempfile.TemporaryDirectory() as directory:
            summary = runner.run_session_a(
                bin_dir=Path(directory), device_index=0, target="gfx942", output_dir=Path(directory) / "out", dry_run=True
            )
        schema = json.loads((ROOT / "ci/schema/phase36-mi300x-session-a-summary-v1.schema.json").read_text())
        errors = list(Draft202012Validator(schema).iter_errors(summary))
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
