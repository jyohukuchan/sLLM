#!/usr/bin/env python3
"""Host-only tests for the fixed llama.cpp Session D raw producer."""

from __future__ import annotations

import copy
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))
import run_phase36_session_d_llama as producer  # noqa: E402


FAKE = r'''#!/usr/bin/env python3
import json, os, sys

def arg(name):
    return sys.argv[sys.argv.index(name) + 1]

case = arg("--case-id")
ids = [int(x) for x in arg("--input-token-ids").split(",")]
out_n = int(arg("--max-new-tokens"))
sha = arg("--model-sha256")
mode = os.environ.get("FAKE_MODE", "")
generated = [7001, 7002] if case == "long-10001" else list(range(1000, 1000 + out_n))

def sample():
    pubs = [50 + 10 * i for i in range(out_n - 1)]
    last = pubs[-1] if pubs else 40
    return {
        "sample_index": 0,
        "events": {"request_start_ns": 10, "prefill_submit_ns": 20,
                    "prefill_complete_ns": 30, "first_token_ns": 40,
                    "token_publications_ns": pubs, "stop_ns": last + 10,
                    "cleanup_complete_ns": last + 20},
        "tokens": {"input_token_ids": ids, "generated_token_ids": generated,
                   "visible_token_ids": generated, "stop_token_ids_fed_back": [],
                   "bos_inserted": False},
        "stop": {"version": 1, "kind": "max_new_tokens", "token_id": None},
        "derived": {"ttft_ns": 30, "prefill_ns": 10,
                     "prefill_tokens_per_second": len(ids) * 1e9 / 10,
                     "tpot_ns": [10] * (out_n - 1),
                     "decode_tokens": out_n - 1, "decode_ns": pubs[-1] - 40 if pubs else None,
                     "decode_tokens_per_second": (out_n - 1) * 1e9 / (pubs[-1] - 40) if pubs else None,
                     "e2e_ns": (last + 20) - 10},
    }

samples = [sample() for _ in range(13)]
doc = {
    "schema_version": "llama-phase36-session-d-v1", "record_kind": "result", "state": "PASS",
    "llama": {"commit": "3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70", "tag": "b10453"},
    "model": {"path": arg("--model"), "sha256": sha, "format": "GGUF", "weights": "BF16", "kv": "F16"},
    "target": {"exact": "gfx942", "gpu_uuid": "GPU-1228c84fe776f2f4", "logical_device_index": 0},
    "row_id": arg("--row-id"), "case_id": case, "input_token_ids": ids,
    "protocol": {"batch_size": 1, "sequences": 1, "warmup_requests": 3, "measured_requests": 10,
                  "max_new_tokens": out_n, "n_ctx": len(ids) + out_n, "n_batch": 10001,
                  "n_ubatch": 512, "n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0,
                  "offload_kqv": True, "op_offload": True, "greedy": True,
                  "stop_token_ids": [248046, 248044], "bos_inserted": False},
    "offload_evidence": {"gpu_offload_supported": True, "visible_gpu_device_count": 1,
                          "selected_device": {"name": "ROCm0", "description": "fake GPU", "type": "GPU"},
                          "requested": {"n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0,
                                        "offload_kqv": True, "op_offload": True},
                          "observed": {"offloaded_layers": 41, "offloadable_layers": 41}},
    "cleanup": {"request_memory_reset": True, "backend_release_completed": True, "cleanup_failures": 0},
    "warmups": {"count": 3, "samples": samples[:3]},
    "measured": {"count": 10, "samples": samples[3:]},
}
if mode == "target": doc["target"]["gpu_uuid"] = "GPU-wrong"
if mode == "protocol": doc["protocol"]["n_batch"] = 2048
if mode == "offload": doc["offload_evidence"]["observed"]["offloaded_layers"] = 40
if mode == "cleanup": doc["cleanup"]["backend_release_completed"] = False
if mode == "token": doc["input_token_ids"][0] = 999
if mode == "digest": doc["model"]["sha256"] = "0" * 64
print(json.dumps(doc, separators=(",", ":")))
'''


def _fake_setup(root: Path) -> tuple[Path, Path, Path]:
    binary = root / "wrapper.py"
    binary.write_text(FAKE, encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    model = root / "model.gguf"
    model.write_bytes(b"fake-bf16-gguf")
    sysfs = root / "sysfs"
    sysfs.mkdir()
    (sysfs / "mem_info_vram_used").write_text("100\n", encoding="ascii")
    (sysfs / "mem_info_gtt_used").write_text("20\n", encoding="ascii")
    return binary, model, sysfs


class Phase36SessionDLlamaProducerTests(unittest.TestCase):
    def test_matrix_and_command_are_closed(self) -> None:
        rows = producer.matrix()
        self.assertEqual([row["case_id"] for row in rows], ["short-odd", "32x32", "prefill-long", "decode-long", "long-10001"])
        self.assertEqual([row["input_token_count"] for row in rows], [17, 32, 1024, 32, 10001])
        self.assertEqual(rows[-1]["input_token_ids"], [23066] * 10001)
        model = Path("/tmp/model.gguf")
        command = producer.expected_command(Path("/tmp/wrapper"), model, "a" * 64, rows[0])
        self.assertEqual(command[command.index("--n-batch") + 1], "10001")
        self.assertEqual(command[command.index("--benchmark-schema-version") + 1], producer.WRAPPER_SCHEMA_VERSION)
        self.assertIn("--model-sha256", command)

    def test_environment_is_exact_uuid_and_loader_closure(self) -> None:
        env = producer.execution_environment("row", {"HIP_VISIBLE_DEVICES": "bad", "ROCR_VISIBLE_DEVICES": "bad", "LD_LIBRARY_PATH": "old"})
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], producer.GPU_UUID)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/home/hotaisle/phase36-llama-build-gfx942/bin:/opt/rocm/lib")
        self.assertNotIn("HIP_VISIBLE_DEVICES", env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", env)

    def test_repository_raw_output_is_rejected(self) -> None:
        binary = Path("/tmp/wrapper")
        model = Path("/tmp/model.gguf")
        sysfs = Path("/tmp/sysfs")
        args = producer.build_parser().parse_args(["--binary", str(binary), "--model", str(model), "--output-dir", str(ROOT / "raw"), "--sysfs-root", str(sysfs)])
        with patch.object(producer, "regular_file", return_value=binary), patch.object(Path, "is_dir", return_value=True):
            with self.assertRaises(producer.SessionDLlamaError):
                producer.run(args)

    def test_fake_wrapper_runs_all_rows_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model, sysfs = _fake_setup(root)
            args = producer.build_parser().parse_args(["--binary", str(binary), "--model", str(model), "--output-dir", str(root / "out"), "--sysfs-root", str(sysfs), "--timeout-seconds", "10"])
            summary = producer.run(args)
            self.assertEqual(summary["state"], "PASS")
            self.assertEqual(summary["matrix"]["row_count"], 5)
            resumed = producer.run(args)
            self.assertEqual(resumed, summary)
            self.assertTrue((root / "out/phase36-session-d-llama-v1.json").is_file())

    def test_wrong_target_protocol_offload_cleanup_and_token_fail_closed(self) -> None:
        for mode in ("target", "protocol", "offload", "cleanup", "token", "digest"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binary, model, sysfs = _fake_setup(root)
                args = producer.build_parser().parse_args(["--binary", str(binary), "--model", str(model), "--output-dir", str(root / "out"), "--sysfs-root", str(sysfs), "--timeout-seconds", "10"])
                with patch.dict(os.environ, {"FAKE_MODE": mode}):
                    with self.assertRaises(producer.SessionDLlamaError):
                        producer.run(args)

    def test_partial_row_and_tampered_raw_digest_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model, sysfs = _fake_setup(root)
            output = root / "out"
            partial = output / "raw" / producer.matrix()[0]["row_id"]
            partial.mkdir(parents=True)
            (partial / "partial").write_text("incomplete", encoding="ascii")
            args = producer.build_parser().parse_args(["--binary", str(binary), "--model", str(model), "--output-dir", str(output), "--sysfs-root", str(sysfs), "--timeout-seconds", "10"])
            with self.assertRaises(producer.SessionDLlamaError):
                producer.run(args)

    def test_timeout_terminates_wrapper_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "hang.sh"
            script.write_text("#!/bin/sh\nsleep 10\n", encoding="ascii")
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            capture = producer.run_process([str(script)], os.environ.copy(), 0.1)
        self.assertTrue(capture["timed_out"])
        self.assertTrue(capture["termination"]["term_sent"])
        self.assertTrue(capture["process_group_gone"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, model, sysfs = _fake_setup(root)
            args = producer.build_parser().parse_args(["--binary", str(binary), "--model", str(model), "--output-dir", str(root / "out"), "--sysfs-root", str(sysfs), "--timeout-seconds", "10"])
            producer.run(args)
            stdout = root / "out/raw/phase36-d-llama-short-odd/stdout.json"
            stdout.write_bytes(stdout.read_bytes() + b"tamper")
            with self.assertRaises(producer.SessionDLlamaError):
                producer.run(args)


if __name__ == "__main__":
    unittest.main()
