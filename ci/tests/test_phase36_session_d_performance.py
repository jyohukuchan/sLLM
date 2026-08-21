from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import run_phase36_session_d_performance as runner  # noqa: E402


FAKE_EXECUTABLE = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

args = sys.argv[1:]
row_id = args[args.index("--row-id") + 1]
case_id = args[args.index("--case-id") + 1]
input_ids = [int(value) for value in args[args.index("--input-token-ids") + 1].split(",")]
output_count = int(args[args.index("--max-new-tokens") + 1])
weight = "fp8" if "fp8" in row_id else "bf16"
generated = [23066, 23066] if case_id == "long-10001" else [7] * output_count
encoding = "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32" if weight == "fp8" else "bf16"
provider = "native-fnuz" if weight == "fp8" else None
if os.environ.get("FAKE_MODE") == "malformed":
    print("not-json")
    raise SystemExit(0)
if os.environ.get("FAKE_MODE") == "token-mismatch":
    input_ids = [999] + input_ids[1:]
stop = {"version": 1, "reason_version": 1, "kind": "max_new_tokens", "token_id": None}
if os.environ.get("FAKE_MODE") == "fallback":
    fallback = True
else:
    fallback = False
cleanup_count = 1 if os.environ.get("FAKE_MODE") == "cleanup" else 0
def sample():
    return {
        "tokens": {
            "input_token_ids": input_ids,
            "generated_token_ids": generated,
            "visible_token_ids": generated,
            "decode_input_token_ids": generated[:-1],
        },
        "stop": stop,
        "audit": {"selected_backend": "hip", "target": "gfx942", "all_dispatches_hip": True, "fallback_used": fallback, "weight_encoding": encoding, "fp8_provider": provider},
        "cleanup": {"retryable_cleanup": cleanup_count, "durable_quarantine": 0, "request_dropped": True},
    }
control = {
    "tokens": {
        "input_token_ids": input_ids,
        "generated_token_ids": generated,
        "visible_token_ids": generated,
        "decode_input_token_ids": generated[:-1],
    },
    "stop": stop,
    "audit": {"selected_backend": "hip", "target": "gfx942", "all_dispatches_hip": True, "fallback_used": fallback, "weight_encoding": encoding, "fp8_provider": provider},
    "cleanup": {"retryable_cleanup": cleanup_count, "durable_quarantine": 0, "request_dropped": True},
}
direct = {
    "benchmark_schema_version": "engine-performance-direct-v1",
    "state": "PASS",
    "lane": "direct",
    "row": {"row_id": row_id, "model_size": "4B", "case_id": case_id, "input_token_ids": input_ids},
    "config": {"input_token_ids": input_ids, "input_token_count": len(input_ids), "max_new_tokens": output_count, "greedy": True, "warmups": 3, "measured": 10},
    "identities": {"target": "gfx942", "model": {"model_size": "4B", "repo_id": "Qwen/Qwen3.5-4B"}},
    "audit": {"selected_backend": "hip", "target": "gfx942", "all_dispatches_hip": True, "fallback_used": fallback, "weight_encoding": encoding, "fp8_provider": provider},
    "cleanup": {"correctness_control_request_count": 1, "warmup_request_count": 3, "measured_request_count": 10, "request_cleanup_count": 14, "performance_sample_count": 13, "all_requests_dropped": True, "correctness_control_dropped": True, "retryable_cleanup": cleanup_count, "durable_quarantine": 0},
    "correctness_control": control,
    "warmups": {"count": 3, "samples": [sample(), sample(), sample()]},
    "measured": {"count": 10, "samples": [sample() for _ in range(10)]},
}
counter = os.environ.get("FAKE_COUNTER")
if counter:
    path = pathlib.Path(counter)
    old = int(path.read_text()) if path.exists() else 0
    path.write_text(str(old + 1))
if os.environ.get("FAKE_DIRECT"):
    print(json.dumps(direct, separators=(",", ":")))
else:
    print(json.dumps({"state": "PASS", "result": direct}, separators=(",", ":")))
'''


class Phase36SessionDPerformanceTests(unittest.TestCase):
    def _fixture(self) -> tuple[tempfile.TemporaryDirectory[str], dict[str, str], Path]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        binary = root / "sllm"
        binary.write_text(FAKE_EXECUTABLE, encoding="utf-8")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        models: dict[str, str] = {}
        for name in ("bf16", "fp8"):
            model = root / f"{name}.gguf"
            lock = root / f"{name}.json"
            model.write_bytes((name + " model").encode())
            lock.write_text(json.dumps({"model": name}), encoding="utf-8")
            models[f"{name}_gguf"] = str(model)
            models[f"{name}_lock"] = str(lock)
        sysfs = root / "sysfs"
        sysfs.mkdir()
        (sysfs / "mem_info_vram_used").write_text("123456", encoding="ascii")
        (sysfs / "mem_info_gtt_used").write_text("789", encoding="ascii")
        models["binary"] = str(binary)
        models["sysfs"] = str(sysfs)
        return temporary, models, root

    def _args(self, root: Path, values: dict[str, str]) -> list[str]:
        return [
            "--binary", values["binary"],
            "--bf16-gguf", values["bf16_gguf"], "--bf16-lock", values["bf16_lock"],
            "--fp8-gguf", values["fp8_gguf"], "--fp8-lock", values["fp8_lock"],
            "--output-dir", str(root / "output"), "--sysfs-root", values["sysfs"],
            "--timeout-seconds", "10",
        ]

    def test_matrix_has_fixed_non_aligned_cases_and_long_input(self) -> None:
        rows = runner.matrix()
        self.assertEqual(len(rows), 10)
        self.assertEqual(rows[0]["input_token_ids"], runner.SHORT_ODD)
        self.assertEqual(len(rows[2]["input_token_ids"]), 1024)
        self.assertEqual(rows[2]["input_token_ids"][17], (17 * 7919 + 41) % 248000)
        self.assertEqual(rows[4]["input_token_ids"], [23066] * 10001)
        self.assertEqual(rows[4]["requested_output_tokens"], 2)

    def test_cli_runs_ten_rows_retains_raw_and_resumes_without_overwrite(self) -> None:
        temporary, values, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        counter = root / "counter"
        os.environ["FAKE_COUNTER"] = str(counter)
        self.addCleanup(lambda: os.environ.pop("FAKE_COUNTER", None))
        with redirect_stdout(StringIO()):
            self.assertEqual(runner.main(self._args(root, values)), 0)
        summary_path = root / "output" / "phase36-session-d-performance-v1.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["matrix"]["row_count"], 10)
        self.assertEqual(summary["gpu_uuid"], runner.GPU_UUID)
        self.assertEqual(int(counter.read_text()), 10)
        for row in summary["rows"]:
            row_dir = root / "output" / "raw" / row["row"]["row_id"]
            self.assertTrue((row_dir / "stdout.json").is_file())
            self.assertTrue((row_dir / "stderr.log").is_file())
            self.assertIn("hbm_bytes", row["memory"]["baseline"])
            self.assertEqual(row["environment"]["ROCR_VISIBLE_DEVICES"], runner.GPU_UUID)
            self.assertGreater(row["monitor"]["samples"], 0)
            self.assertIn("--kv-cache-encoding", row["command"])
        with redirect_stdout(StringIO()):
            self.assertEqual(runner.main(self._args(root, values)), 0)
        self.assertEqual(int(counter.read_text()), 10)

    def test_malformed_fallback_cleanup_and_token_mismatch_are_rejected(self) -> None:
        for mode in ("malformed", "fallback", "cleanup", "token-mismatch"):
            temporary, values, root = self._fixture()
            self.addCleanup(temporary.cleanup)
            os.environ["FAKE_MODE"] = mode
            try:
                result = runner.main(self._args(root, values))
            finally:
                os.environ.pop("FAKE_MODE", None)
            self.assertEqual(result, 2, mode)

    def test_direct_cli_result_without_frontend_wrapper_is_accepted(self) -> None:
        temporary, values, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        os.environ["FAKE_DIRECT"] = "1"
        try:
            with redirect_stdout(StringIO()):
                result = runner.main(self._args(root, values))
        finally:
            os.environ.pop("FAKE_DIRECT", None)
        self.assertEqual(result, 0)

    def test_repository_raw_output_is_rejected_and_loader_path_is_closed(self) -> None:
        temporary, values, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        args = runner.build_parser().parse_args(self._args(root, values))
        args.output_dir = str(ROOT / "phase36-session-d-raw-must-not-exist")
        with self.assertRaisesRegex(runner.SessionDError, "outside the repository"):
            runner.run(args)
        env = runner._execution_environment("row", 0, {"LD_LIBRARY_PATH": "/tmp/untrusted"})
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/rocm/lib")

    def test_sysfs_parser_rejects_missing_or_malformed_counters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mem_info_vram_used").write_text("abc", encoding="ascii")
            (root / "mem_info_gtt_used").write_text("1", encoding="ascii")
            with self.assertRaises(runner.SessionDError):
                runner.read_sysfs_memory(root)

    def test_partial_raw_row_is_not_overwritten(self) -> None:
        temporary, values, root = self._fixture()
        self.addCleanup(temporary.cleanup)
        partial = root / "output" / "raw" / "phase36-d-bf16-short-odd"
        partial.mkdir(parents=True)
        (partial / "stdout.json").write_text("partial", encoding="utf-8")
        with redirect_stdout(StringIO()):
            result = runner.main(self._args(root, values))
        self.assertEqual(result, 2)


if __name__ == "__main__":
    unittest.main()
