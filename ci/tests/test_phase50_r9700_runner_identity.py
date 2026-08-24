import importlib.util
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sllm = _load("run_phase50_r9700_sllm", "ci/tools/run_phase50_r9700_sllm.py")
llama = _load("run_phase50_r9700_llama", "ci/tools/run_phase50_r9700_llama.py")


class Phase50R9700RunnerIdentityTest(unittest.TestCase):
    def test_exact_target_identity_is_shared(self):
        expected = ("gfx1201", "GPU-a8e9ddefa2d60f55", "0000:07:00.0")
        self.assertEqual((sllm.TARGET, sllm.GPU_UUID, sllm.GPU_BDF), expected)
        self.assertEqual((llama.TARGET, llama.GPU_UUID, llama.GPU_BDF), expected)

    def test_matrix_has_five_normal_and_two_extended_rows(self):
        sllm_rows = sllm.matrix()
        llama_rows = llama.matrix()
        self.assertEqual([row["case_id"] for row in sllm_rows], [row["case_id"] for row in llama_rows])
        self.assertEqual(len(sllm_rows), 7)
        self.assertEqual({row["case_id"] for row in sllm_rows[:5]}, {"short-odd", "32-32", "prefill-long", "decode-long", "long-10001"})
        self.assertEqual({row["case_id"] for row in sllm_rows[5:]}, {"long-100000", "decode-20000"})
        self.assertTrue(all(row["warmups"] == 3 and row["measured"] == 10 for row in sllm_rows[:5]))
        self.assertTrue(all(row["warmups"] == 1 and row["measured"] == 3 for row in sllm_rows[5:]))

    def test_phase50_row_and_schema_names(self):
        self.assertTrue(all(row["row_id"].startswith("phase50-r9700-sllm-") for row in sllm.matrix()))
        self.assertTrue(all(row["row_id"].startswith("phase50-r9700-llama-") for row in llama.matrix()))
        self.assertEqual(sllm.SCHEMA_VERSION, "phase50-r9700-sllm-v1")
        self.assertEqual(llama.SCHEMA_VERSION, "phase50-r9700-llama-v1")
        self.assertEqual(llama.WRAPPER_SCHEMA_VERSION, "llama-phase50-r9700-v1")

    def test_wrapper_source_binds_r9700(self):
        source = (ROOT / "ci/tools/llama_phase50_r9700_wrapper.cpp").read_text(encoding="utf-8")
        for value in ("llama-phase50-r9700-v1", "gfx1201", "GPU-a8e9ddefa2d60f55"):
            self.assertIn(value, source)
        for value in ("gfx1030", "GPU-76a08c022586fed6", "llama-phase49-v620-v1"):
            self.assertNotIn(value, source)

    def test_oom_is_fail_closed_and_following_rows_continue(self):
        capture = {
            "pid": 1,
            "exit_code": 2,
            "stderr": b"grow virtual KV physical commitment: out of memory",
            "timed_out": False,
            "process_group_gone": True,
        }
        for module in (sllm, llama):
            kind, reason = module._failure_class(capture, None, [], {"hbm_bytes": 1, "gtt_bytes": 2}, {"settled": True, "hbm_bytes": 1, "gtt_bytes": 2})
            self.assertEqual(kind, "oom")
            self.assertIn("out of memory", reason)

        rows = [dict(sllm.matrix()[index]) for index in (0, 5, 6)]
        calls = []
        failure = {"state": "FAIL", "row": rows[1], "failure": {"kind": "oom", "reason": "out of memory"}}
        passed = {"state": "PASS", "row": rows[0]}
        passed_last = {"state": "PASS", "row": rows[2]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "benchmark"
            model = root / "model.gguf"
            lock = root / "model.lock.json"
            for path in (binary, model, lock):
                path.write_bytes(b"fixture")
            binary.chmod(0o755)
            args = argparse.Namespace(
                target=sllm.TARGET,
                gpu_uuid=sllm.GPU_UUID,
                gpu_bdf=sllm.GPU_BDF,
                device_index=0,
                output_dir=str(root / "evidence"),
                binary=str(binary),
                bf16_gguf=str(model),
                bf16_lock=str(lock),
                sysfs_root=str(root),
                amd_smi="",
                timeout_seconds=1.0,
            )
            def fake_run_row(*call_args, **call_kwargs):
                row = call_args[4]
                calls.append(row["case_id"])
                return {"state": "FAIL", "row": row, "failure": {"kind": "oom", "reason": "out of memory"}} if row["case_id"] == rows[1]["case_id"] else {"state": "PASS", "row": row}
            with patch.object(sllm, "matrix", return_value=rows), patch.object(sllm, "run_row", side_effect=fake_run_row):
                summary = sllm.run(args)
            self.assertEqual(calls, [row["case_id"] for row in rows])
            self.assertEqual(summary["state"], "FAIL")


if __name__ == "__main__":
    unittest.main()
