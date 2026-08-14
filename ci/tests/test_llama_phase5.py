#!/usr/bin/env python3
"""Host-only contract tests for the Phase 5 llama.cpp wrapper lane."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import os
import signal
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ci/tools"
sys.path.insert(0, str(TOOLS))

import run_llama_phase5 as runner  # noqa: E402


class LlamaPhase5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix, cls.matrix_digest, cls.direct, cls.direct_digest = runner.load_matrix()
        runner._MATRIX_CACHE = cls.matrix

    def test_contract_cli_is_json_only_on_stdout(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOLS / "run_llama_phase5.py"), "--contract-only"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        document = json.loads(completed.stdout)
        self.assertEqual(document["state"], "PASS")
        self.assertEqual(document["source_commit"], runner.PINNED_COMMIT)
        self.assertEqual(document["sequence_lengths"], [1, 17, 255, 256, 257, 1024, 32])

    def test_tracked_contract_does_not_require_local_reference_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "REFERENCE_PATH", Path(directory) / "missing-llama.cpp"
        ):
            matrix, _, _, _ = runner.load_matrix()
        self.assertEqual(matrix["llama"]["commit"], runner.PINNED_COMMIT)

    def test_tracked_conversion_identity_is_repository_relative(self) -> None:
        conversion = self.matrix["conversion"]
        self.assertEqual(conversion["source"]["path"], "reference/llama.cpp")
        self.assertEqual(
            conversion["tool"]["path"],
            "reference/llama.cpp/convert_hf_to_gguf.py",
        )
        self.assertEqual(
            conversion["arguments"][1],
            "reference/llama.cpp/convert_hf_to_gguf.py",
        )
        self.assertNotIn(str(ROOT.resolve()), json.dumps(conversion, sort_keys=True))

    def test_explicit_local_reference_verification_fails_closed_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
            runner, "REFERENCE_PATH", Path(directory) / "missing-llama.cpp"
        ), self.assertRaises(runner.ContractError):
            runner.load_matrix(verify_reference=True)

    def test_matrix_is_closed_over_direct_recipes_and_devices(self) -> None:
        self.assertEqual(len(self.matrix["rows"]), 14)
        self.assertEqual(
            [case["direct_sequence_id"] for case in self.matrix["cases"]],
            list(runner.CASES),
        )
        self.assertEqual(
            [target["target"] for target in self.matrix["targets"]],
            ["gfx1030", "gfx1201"],
        )
        self.assertEqual(
            [(row["target"], row["case_id"]) for row in self.matrix["rows"][:7]],
            [("gfx1030", case) for case in runner.CASES],
        )
        self.assertEqual(
            next(item for item in self.matrix["targets"] if item["target"] == "gfx1201")["gpu_bdf"],
            "0000:07:00.0",
        )
        self.assertEqual(self.matrix["source_direct_matrix"]["revision"], runner.DIRECT_MATRIX_REVISION)
        self.assertEqual(self.matrix["source_direct_matrix"]["sha256"], runner.DIRECT_MATRIX_SHA256)
        self.assertEqual(self.matrix["source_direct_matrix"]["sha256"], self.direct_digest)

    def test_source_direct_matrix_has_no_pending_digest_escape_hatch(self) -> None:
        self.assertFalse(hasattr(runner, "PENDING_SOURCE_DIRECT_MATRIX_SHA256"))
        self.assertEqual(self.matrix["source_direct_matrix"]["revision"], 4)

    def test_conversion_contract_is_pinned_to_source_lock_and_converter(self) -> None:
        conversion = self.matrix["conversion"]
        self.assertEqual(conversion["source_lock"]["fingerprint"], runner.SOURCE_LOCK_FINGERPRINT)
        self.assertEqual(conversion["source_lock"]["resolved_revision"], runner.SOURCE_MODEL_REVISION)
        self.assertEqual(conversion["tool"]["commit"], runner.PINNED_COMMIT)
        self.assertEqual(conversion["tool"]["sha256"], runner.CONVERTER_SHA256)

    @staticmethod
    def _realistic_actual_manifest_fixture() -> dict:
        return json.loads(runner.CONVERSION_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_realistic_detailed_conversion_manifest_normalizes_to_source_digest(self) -> None:
        fixture = self._realistic_actual_manifest_fixture()
        runner.schema_validate(fixture, "conversion_manifest", "realistic detailed conversion fixture")
        lock = runner.validate_source_lock(runner.SOURCE_LOCK_PATH)
        identity = runner.validate_conversion_manifest(
            runner.CONVERSION_SOURCE_MANIFEST_PATH,
            lock,
            runner.CONVERSION_OUTPUT_PATH,
        )
        self.assertEqual(identity["sha256"], runner.CONVERSION_SOURCE_MANIFEST_SHA256)
        self.assertEqual(identity["path"], str(runner.CONVERSION_SOURCE_MANIFEST_PATH.resolve()))
        self.assertEqual(identity["manifest"], runner.expected_conversion_identity(runner.PINNED_TREE))
        self.assertEqual(identity["manifest"]["arguments"][-3:], ["--outtype", "bf16", "--no-mtp"])
        self.assertIn("--no-mtp", identity["manifest"]["arguments"])
        self.assertEqual(identity["manifest"]["gguf"], {"architecture": "qwen35", "name": runner.SOURCE_MODEL_REVISION, "file_type": 32, "quantization_version": 2, "tensor_count": 426, "mtp_tensor_count": 0})

    def test_detailed_fixture_does_not_bind_primary_dirty_status(self) -> None:
        fixture = self._realistic_actual_manifest_fixture()
        fixture["repository_after"]["primary_repo_status"] = "clean"
        fixture["repository_after"]["primary_repo_status_sha256_at_end"] = "0" * 64
        runner.schema_validate(fixture, "conversion_manifest", "volatile primary-status fixture")

    def test_detailed_conversion_tamper_cases_fail_closed(self) -> None:
        lock = runner.validate_source_lock(runner.SOURCE_LOCK_PATH)
        mutations = {
            "source file set": lambda document: document["model"]["files"][0].__setitem__("sha256", "0" * 64),
            "conversion arguments": lambda document: document["conversion"]["run"]["args"].remove("--no-mtp"),
            "GGUF metadata": lambda document: document["gguf_metadata_validation"].__setitem__("general_file_type", 0),
            "output identity": lambda document: document["conversion"]["run"].__setitem__("output_size_bytes", 1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                fixture = copy.deepcopy(self._realistic_actual_manifest_fixture())
                mutate(fixture)
                encoded = (json.dumps(fixture, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                path = Path(directory) / "manifest.json"
                path.write_bytes(encoded)
                digest = hashlib.sha256(encoded).hexdigest()
                with patch.object(runner, "CONVERSION_SOURCE_MANIFEST_PATH", path), patch.object(runner, "CONVERSION_SOURCE_MANIFEST_SHA256", digest):
                    with self.assertRaises(runner.ContractError):
                        runner.validate_conversion_manifest(path, lock, runner.CONVERSION_OUTPUT_PATH)

    def test_distribution_stats_have_all_required_fields(self) -> None:
        stats = runner.distribution_stats([1, 2, 3, 4, 5])
        self.assertEqual(set(stats), {"median", "p10", "p90", "mad", "min", "max", "count"})
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["mad"], 1.0)

    def test_distribution_stats_reject_empty_input(self) -> None:
        with self.assertRaises(runner.ContractError):
            runner.distribution_stats([])

    def test_tpot_distribution_uses_ten_per_request_medians(self) -> None:
        measured = []
        for index in range(10):
            measured.append({
                "ttft_ns": 10 + index,
                "prefill_ns": 5 + index,
                "prefill_tokens_per_second": 100.0 + index,
                "tpot_ns": [1] if index < 9 else [100] * 100,
                "decode_tokens_per_second": 50.0 + index,
                "e2e_ns": 20 + index,
            })
        distributions = runner.llama_metric_distributions(measured)
        self.assertEqual(distributions["tpot_ns"], [1.0] * 9 + [100.0])
        self.assertEqual(runner.distribution_stats(distributions["tpot_ns"])["median"], 1.0)
        self.assertEqual(len(distributions["tpot_ns"]), 10)
        measured[0]["tpot_ns"] = []
        with self.assertRaises(runner.ContractError):
            runner.llama_metric_distributions(measured)

    @staticmethod
    def _stats(value: float, count: int = 10) -> dict[str, float | int]:
        return {"median": value, "p10": value, "p90": value, "mad": 0.0, "min": value, "max": value, "count": count}

    def _cross_engine_fixtures(self) -> tuple[list[dict], dict]:
        digest = "a" * 64
        direct_rows = []
        llama_rows = []
        for order, row in enumerate(item for item in self.direct["rows"] if item["model_size"] == "4B"):
            direct_rows.append({
                **row,
                "manifest_sha256": digest,
                "raw_result_sha256": digest,
                "binary_sha256": digest,
                "model_lock_sha256": digest,
                "model_lock_fingerprint": runner.SOURCE_LOCK_FINGERPRINT,
                "warmup_count": 3,
                "sample_count": 10,
                "metrics": {
                    "ttft_ns": self._stats(10.0), "prefill_ns": self._stats(5.0),
                    "prefill_token_per_s": self._stats(20.0), "tpot_ns": self._stats(2.0),
                    "decode_token_per_s": self._stats(30.0), "e2e_ns": self._stats(40.0),
                    "resident_vram_bytes": self._stats(1000.0), "peak_vram_bytes": self._stats(2000.0),
                },
            })
            llama_rows.append({
                "order": order, "row_id": f"llama-{row['target']}-{row['case_id']}",
                "target": row["target"], "case_id": row["case_id"],
                "input_tokens": row["input_tokens"], "requested_output_tokens": row["requested_output_tokens"],
                "sample_count": 10,
                "metrics": {
                    "ttft_ns": self._stats(20.0), "prefill_ns": self._stats(10.0),
                    "prefill_tokens_per_second": self._stats(10.0), "tpot_ns": self._stats(4.0),
                    "decode_tokens_per_second": self._stats(15.0), "e2e_ns": self._stats(80.0),
                    "model_load_ns": self._stats(100.0, 1), "available_vram_mb": self._stats(8000.0, 2),
                },
                "manifest_sha256": digest, "raw_sha256": digest, "binary_sha256": digest,
                "build_manifest_sha256": digest, "offload_evidence_sha256": digest,
            })
        return llama_rows, {"rows": direct_rows}

    def test_cross_engine_join_emits_both_distributions_and_only_comparable_ratios(self) -> None:
        llama_rows, direct = self._cross_engine_fixtures()
        rows = runner.cross_engine_rows(llama_rows, direct)
        self.assertEqual(len(rows), 14)
        metrics = {item["metric"]: item for item in rows[0]["metrics"]}
        self.assertEqual(metrics["ttft_ns"]["ratio"], 2.0)
        self.assertEqual(metrics["ttft_ns"]["sllm_distribution"]["count"], 10)
        self.assertEqual(metrics["ttft_ns"]["llama_distribution"]["count"], 10)
        for metric in ("tpot_ns", "decode_tokens_per_second", "e2e_ns", "resident_vram_bytes", "peak_vram_bytes", "model_load_ns", "available_vram_mb"):
            self.assertEqual(metrics[metric]["classification"], "context-only")
            self.assertIsNone(metrics[metric]["ratio"])

    def test_cross_engine_join_rejects_target_case_token_mismatch(self) -> None:
        llama_rows, direct = self._cross_engine_fixtures()
        llama_rows[0]["input_tokens"] += 1
        with self.assertRaises(runner.ContractError):
            runner.cross_engine_rows(llama_rows, direct)

    def test_schema_rejects_ratio_for_context_only_metric(self) -> None:
        record = {
            "metric": "tpot_ns", "classification": "context-only",
            "sllm_distribution": self._stats(2.0), "llama_distribution": self._stats(4.0),
            "ratio": None,
        }
        runner.schema_validate(record, "comparison_metric", "context-only comparison")
        record["ratio"] = 2.0
        with self.assertRaises(runner.ContractError):
            runner.schema_validate(record, "comparison_metric", "context-only comparison with ratio")

    def test_verified_direct_aggregate_accepts_real_contract_and_rejects_stale_matrix(self) -> None:
        import aggregate_engine_performance as direct_aggregator
        from ci.tests.test_engine_performance_aggregate import EnginePerformanceAggregateTests

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = EnginePerformanceAggregateTests()._manifests(root)
            direct_aggregator.aggregate_manifests(manifests, root / "out", verify_external_digests=False)
            summary_path = root / "out/summary.json"
            document, digest = runner.validate_direct_aggregate(summary_path, self.direct, self.direct_digest)
            self.assertEqual(document["state"], "PASS")
            self.assertEqual(len(digest), 64)

            completion = root / "out/bundle.complete.json"
            completion_bytes = completion.read_bytes()
            completion.unlink()
            with self.assertRaisesRegex(runner.ContractError, "completion record"):
                runner.validate_direct_aggregate(summary_path, self.direct, self.direct_digest)
            completion.write_bytes(completion_bytes)

            stale = json.loads(summary_path.read_text(encoding="utf-8"))
            stale["matrix"]["sha256"] = "0" * 64
            encoded = (json.dumps(stale, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
            stale_path = root / "stale.json"
            stale_path.write_bytes(encoded)
            stale_path.with_suffix(".json.sha256").write_text(
                f"{hashlib.sha256(encoded).hexdigest()}  stale.json\n", encoding="ascii",
            )
            with self.assertRaises(runner.ContractError):
                runner.validate_direct_aggregate(stale_path, self.direct, self.direct_digest)

    def test_offload_evidence_is_observed_and_digest_bound(self) -> None:
        evidence = {
            "gpu_offload_supported": True,
            "visible_gpu_device_count": 1,
            "selected_device": {"name": "ROCm0", "description": "AMD GPU", "type": "GPU"},
            "requested": {"n_gpu_layers": -1, "split_mode": "none", "main_gpu": 0, "offload_kqv": True, "op_offload": True},
            "observed": {
                "offloaded_layers": 41, "offloadable_layers": 41, "gpu_model_buffer_mib": 8032.0,
                "device_memory": {"free_before_bytes": 20_000, "total_before_bytes": 30_000, "free_model_ready_bytes": 10_000, "total_model_ready_bytes": 30_000, "observed_decrease_bytes": 10_000},
                "captured_log_bytes": 2048,
            },
        }
        digest = runner.validate_offload_evidence(evidence)
        self.assertEqual(len(digest), 64)
        for mutate in (
            lambda value: value["observed"].__setitem__("offloaded_layers", 40),
            lambda value: value["observed"]["device_memory"].__setitem__("observed_decrease_bytes", 1),
        ):
            changed = copy.deepcopy(evidence)
            mutate(changed)
            with self.assertRaises(runner.ContractError):
                runner.validate_offload_evidence(changed)

    def test_official_commands_render_bench_model_and_rocm0(self) -> None:
        commands = runner.official_commands(self.matrix, Path("/tmp/llama-bench"), Path("/tmp/model.gguf"))
        self.assertEqual(len(commands), 9)
        for _kind, command in commands:
            self.assertEqual(command[0], "/tmp/llama-bench")
            self.assertIn("/tmp/model.gguf", command)
            self.assertEqual(command[command.index("-dev") + 1], "ROCm0")
            if "-pg" in command:
                self.assertEqual(command[command.index("-p") + 1], "0")
                self.assertEqual(command[command.index("-n") + 1], "0")

        self.assertEqual(
            [runner.official_timeout_seconds(command) for _kind, command in commands],
            [10800, 5400, 5400, 5400, 5400, 5400, 5400, 10800, 5400],
        )

    def test_official_startup_stderr_accepts_only_exact_visible_device(self) -> None:
        device = next(target for target in self.matrix["targets"] if target["target"] == "gfx1030")
        stderr = (
            "ggml_cuda_init: found 1 ROCm devices (Total VRAM: 30704 MiB):\n"
            "  Device 0: AMD Radeon Pro V620, gfx1030 (0x1030), VMM: no, "
            "Wave Size: 32, VRAM: 30704 MiB\n"
        ).encode()
        identity = runner.validate_official_startup_stderr(stderr, device)
        self.assertEqual(identity["bytes"], len(stderr))
        with self.assertRaises(runner.ContractError):
            runner.validate_official_startup_stderr(stderr.replace(b"gfx1030", b"gfx1201"), device)

    def test_wrapper_prefill_long_uses_extended_timeout(self) -> None:
        for row in self.matrix["rows"]:
            expected = 10800 if row["case_id"] == "prefill-long" else 5400
            actual = runner.PREFILL_LONG_TIMEOUT_SECONDS if row["case_id"] == "prefill-long" else runner.DEFAULT_TIMEOUT_SECONDS
            self.assertEqual(actual, expected)

    def test_wrapper_process_preserves_cwd_and_environment(self) -> None:
        environment = os.environ.copy()
        environment["SLLM_PHASE5_EXECUTION_TEST"] = "present"
        capture = runner.run_wrapper_process(
            [
                sys.executable, "-c",
                "import os;print(os.getcwd());print(os.environ['SLLM_PHASE5_EXECUTION_TEST'])",
            ],
            environment,
            5,
        )
        self.assertEqual(capture["exit_code"], 0)
        self.assertEqual(capture["stdout"], f"{ROOT}\npresent\n".encode())
        self.assertEqual(capture["stderr"], b"")
        self.assertEqual(capture["output_overflow"], [])
        self.assertTrue(capture["process_group_gone"])

    def test_wrapper_process_drains_simultaneous_stdout_and_stderr(self) -> None:
        bytes_per_pipe = 1024 * 1024
        script = (
            "import os,threading;"
            f"n={bytes_per_pipe};"
            "a=threading.Thread(target=lambda:os.write(1,b'o'*n));"
            "b=threading.Thread(target=lambda:os.write(2,b'e'*n));"
            "a.start();b.start();a.join();b.join()"
        )
        capture = runner.run_wrapper_process(
            [sys.executable, "-c", script], os.environ.copy(), 5,
        )
        self.assertEqual(capture["exit_code"], 0)
        self.assertEqual(capture["stdout"], b"o" * bytes_per_pipe)
        self.assertEqual(capture["stderr"], b"e" * bytes_per_pipe)
        self.assertEqual(capture["output_overflow"], [])
        self.assertTrue(capture["process_group_gone"])

    def test_wrapper_process_overflow_is_bounded_and_cleans_up_group(self) -> None:
        with patch.object(runner.direct_health, "MAX_RAW_BYTES", 4096), patch.object(
            runner.direct_health, "TERMINATION_GRACE_SECONDS", 0.2,
        ):
            capture = runner.run_wrapper_process(
                [sys.executable, "-c", "import os,time;os.write(1,b'x'*8192);time.sleep(10)"],
                os.environ.copy(),
                5,
            )
        self.assertEqual(capture["output_overflow"], ["stdout"])
        self.assertEqual(len(capture["stdout"]), 4096)
        self.assertFalse(capture["timed_out"])
        self.assertTrue(capture["term_sent"] or capture["kill_sent"])
        self.assertTrue(capture["process_group_gone"])

    def test_wrapper_process_timeout_term_and_kill_paths_are_bounded(self) -> None:
        with patch.object(runner.direct_health, "TERMINATION_GRACE_SECONDS", 0.2):
            term_capture = runner.run_wrapper_process(
                [
                    sys.executable, "-c",
                    "import subprocess,time;subprocess.Popen(['sleep','10']);time.sleep(10)",
                ],
                os.environ.copy(),
                0.1,
            )
        self.assertTrue(term_capture["timed_out"])
        self.assertTrue(term_capture["term_sent"])
        self.assertFalse(term_capture["kill_sent"])
        self.assertTrue(term_capture["process_group_gone"])

        script = (
            "import signal,subprocess,sys,time;"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "subprocess.Popen([sys.executable,'-c',"
            "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(10)']);"
            "time.sleep(10)"
        )
        started = time.monotonic()
        with patch.object(runner.direct_health, "TERMINATION_GRACE_SECONDS", 0.2):
            kill_capture = runner.run_wrapper_process(
                [sys.executable, "-c", script], os.environ.copy(), 0.1,
            )
        self.assertLess(time.monotonic() - started, 2)
        self.assertTrue(kill_capture["timed_out"])
        self.assertTrue(kill_capture["term_sent"])
        self.assertTrue(kill_capture["kill_sent"])
        self.assertTrue(kill_capture["process_group_gone"])

    @staticmethod
    def _overflow_capture() -> dict:
        return {
            "stdout": b"x" * 4096,
            "stderr": b"",
            "exit_code": -signal.SIGTERM,
            "timed_out": False,
            "term_sent": True,
            "kill_sent": False,
            "process_group_gone": True,
            "output_overflow": ["stdout"],
            "monitor": {"samples": [], "errors": []},
        }

    def test_wrapper_row_caller_rejects_output_overflow_without_publication(self) -> None:
        row_id = "llama-phase5-4b-gfx1030-minimum"
        build_manifest = {"binary": {"path": "/tmp/wrapper", "sha256": "a" * 64}}
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifacts"
            with patch.object(
                runner, "load_matrix", return_value=(self.matrix, self.matrix_digest, self.direct, self.direct_digest),
            ), patch.object(runner, "validate_model"), patch.object(
                runner, "validate_source_lock", return_value={},
            ), patch.object(runner, "validate_conversion_manifest", return_value={}), patch.object(
                runner, "validate_build_manifest", return_value=build_manifest,
            ), patch.object(runner.direct_health, "_amd_smi_observation", return_value={}), patch.object(
                runner.direct_health, "validate_observation", return_value={"stable": True},
            ), patch.object(runner.direct_health, "_amd_smi_phase_evidence", return_value={}), patch.object(
                runner, "run_wrapper_process", return_value=self._overflow_capture(),
            ):
                with self.assertRaisesRegex(runner.ContractError, "output exceeded the bounded limit"):
                    runner.run_row(
                        row_id, Path("/tmp/build.json"), runner.CONVERSION_OUTPUT_PATH, artifact_root,
                        conversion_manifest=runner.CONVERSION_SOURCE_MANIFEST_PATH,
                    )
            self.assertFalse(artifact_root.exists())

    def test_official_caller_rejects_output_overflow_without_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_root = root / "official-raw"
            output = root / "official.json"
            bench = root / "llama-bench"
            with patch.object(
                runner, "load_matrix", return_value=(self.matrix, self.matrix_digest, self.direct, self.direct_digest),
            ), patch.object(runner, "validate_official_bench", return_value=bench), patch.object(
                runner, "validate_model",
            ), patch.object(runner, "validate_source_lock", return_value={}), patch.object(
                runner, "validate_conversion_manifest", return_value={"path": "/tmp/conversion.json", "sha256": "a" * 64},
            ), patch.object(runner, "reference_identity", return_value={"commit": runner.PINNED_COMMIT, "tree": runner.PINNED_TREE}), patch.object(
                runner.direct_health, "_amd_smi_observation", return_value={}), patch.object(
                runner.direct_health, "validate_observation", return_value={"stable": True},
            ), patch.object(runner.direct_health, "_amd_smi_phase_evidence", return_value={}), patch.object(
                runner, "run_wrapper_process", return_value=self._overflow_capture(),
            ):
                with self.assertRaisesRegex(runner.ContractError, "output exceeded the bounded limit"):
                    runner.run_official_context(
                        "gfx1030", bench, runner.CONVERSION_OUTPUT_PATH,
                        runner.CONVERSION_SOURCE_MANIFEST_PATH, artifact_root, output,
                    )
            self.assertFalse(artifact_root.exists())
            self.assertFalse(output.exists())

    def test_aggregate_bundle_publishes_payload_and_sidecar_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "comparison.json"
            payload = b'{"state":"PASS"}\n'
            published, digest = runner.publish_aggregate_bundle(output, payload)
            self.assertEqual(published, output.resolve())
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(
                output.with_suffix(".json.sha256").read_text(encoding="ascii"),
                f"{digest}  comparison.json\n",
            )
            marker = output.with_suffix(".json.complete.json")
            self.assertTrue(marker.is_file())
            runner.verify_completed_bundle(
                marker, (output, output.with_suffix(".json.sha256")),
                "llama wrapper aggregate",
            )

    def test_aggregate_bundle_preflights_both_destinations_without_overwrite(self) -> None:
        for existing_name in ("comparison.json", "comparison.json.sha256", "comparison.json.complete.json"):
            with self.subTest(existing_name=existing_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "comparison.json"
                existing = root / existing_name
                existing.write_bytes(b"existing\n")
                with self.assertRaises(runner.ContractError):
                    runner.publish_aggregate_bundle(output, b"new\n")
                self.assertEqual(existing.read_bytes(), b"existing\n")
                self.assertEqual(sorted(path.name for path in root.iterdir()), [existing_name])

    def test_aggregate_bundle_rolls_back_first_member_when_sidecar_publish_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "comparison.json"
            real_link = os.link
            link_count = 0

            def fail_second_link(source: Path, destination: Path) -> None:
                nonlocal link_count
                link_count += 1
                if link_count == 2:
                    raise OSError("synthetic sidecar publication failure")
                real_link(source, destination)

            with patch.object(runner.os, "link", side_effect=fail_second_link):
                with self.assertRaisesRegex(runner.ContractError, "cannot publish"):
                    runner.publish_aggregate_bundle(output, b"payload\n")
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json.sha256").exists())
            self.assertFalse(output.with_suffix(".json.complete.json").exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_completed_bundle_consumer_rejects_abrupt_partial_row_and_official_layouts(self) -> None:
        layouts = (
            ("row", lambda root: {
                root / "rows/row/raw-result.json": b"raw\n",
                root / "rows/row/stderr.txt": b"",
                root / "rows/row/manifest.json": b"{}\n",
                root / "rows/row/manifest.json.sha256": b"digest\n",
            }, lambda root: root / "rows/row/bundle.complete.json"),
            ("official", lambda root: {
                root / "official/raw-00-paired.json": b"[]\n",
                root / "official/stderr-00-paired.txt": b"",
                root / "context.json": b"{}\n",
                root / "context.json.sha256": b"digest\n",
            }, lambda root: root / "context.json.complete.json"),
        )
        for label, payload_factory, marker_factory in layouts:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                payloads = payload_factory(root)
                marker = marker_factory(root)
                real_link = os.link
                link_count = 0

                def fail_before_commit(source: Path, destination: Path) -> None:
                    nonlocal link_count
                    link_count += 1
                    if link_count == len(payloads) + 1:
                        raise OSError("synthetic abrupt publication boundary")
                    real_link(source, destination)

                with patch.object(runner.os, "link", side_effect=fail_before_commit):
                    with self.assertRaisesRegex(runner.ContractError, "cannot publish"):
                        runner.publish_completed_bundle(payloads, marker, label)
                self.assertFalse(marker.exists())
                with self.assertRaisesRegex(runner.ContractError, "completion record"):
                    runner.verify_completed_bundle(marker, payloads, label)
                self.assertFalse(any(path.exists() for path in payloads))

    def test_p3_health_comparison_allows_inert_record_drift_but_rejects_allowlist_drift(self) -> None:
        from ci.tests.test_engine_performance_aggregate import _allowed_observation, _allowed_process_record

        pre = _allowed_observation("gfx1030", _allowed_process_record(diagnostic=1))
        post = _allowed_observation("gfx1030", _allowed_process_record(diagnostic=99))
        self.assertTrue(runner.direct_health._observations_have_stable_authorization(pre, post))
        changed = copy.deepcopy(post)
        changed["process"]["gpu_processes"][1]["record"]["process_info"]["name"] = "changed"
        self.assertTrue(runner.direct_health._observations_have_stable_authorization(pre, changed))
        changed["process"]["gpu_processes"][0]["allowlisted_pids"] = [4243]
        self.assertFalse(runner.direct_health._observations_have_stable_authorization(pre, changed))

    def test_official_context_is_marked_context_only(self) -> None:
        definitions = self.matrix["official_llama_bench"]["metric_definitions"]
        self.assertTrue(definitions["context_only"])
        self.assertFalse(definitions["ratio_comparable"])

    def test_official_json_is_closed_over_requested_rows_and_samples(self) -> None:
        commands = runner.official_commands(self.matrix, Path("/tmp/llama-bench"), Path("/tmp/model.gguf"))
        command = commands[0][1]
        fixture = [
            {
                "n_prompt": prompt,
                "n_gen": 0,
                "n_batch": 2048,
                "n_ubatch": 512,
                "main_gpu": 0,
                "split_mode": "none",
                "samples_ns": [1000] * 10,
                "samples_ts": [1.0] * 10,
                "avg_ns": 1000,
                "avg_ts": 1.0,
            }
            for prompt in (1, 17, 32, 255, 256, 257, 1024)
        ]
        runner.validate_official_json(fixture, command)
        for label, mutate in {
            "missing row": lambda value: value.pop(),
            "wrong tokens": lambda value: value[0].__setitem__("n_prompt", 2),
            "short samples": lambda value: value[0].__setitem__("samples_ns", [1000] * 9),
            "wrong batch": lambda value: value[0].__setitem__("n_batch", 512),
        }.items():
            with self.subTest(label=label):
                changed = copy.deepcopy(fixture)
                mutate(changed)
                with self.assertRaises(runner.ContractError):
                    runner.validate_official_json(changed, command)

    def test_official_bench_is_bound_to_target_build_root_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            bench = root / "bin" / "llama-bench"
            bench.write_bytes(b"#!/bin/sh\n")
            os.chmod(bench, 0o755)
            (root / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\n"
                "CMAKE_HIP_ARCHITECTURES:UNINITIALIZED=gfx1030\n"
                "GGML_HIP:BOOL=ON\n",
                encoding="utf-8",
            )
            with patch.dict(runner.BUILD_ROOTS, {"gfx1030": root}):
                self.assertEqual(runner.validate_official_bench("gfx1030", bench), bench.resolve())
                with self.assertRaises(runner.ContractError):
                    runner.validate_official_bench("gfx1030", root / "other")

                (root / "CMakeCache.txt").write_text(
                    "CMAKE_BUILD_TYPE:STRING=Release\n"
                    "CMAKE_HIP_ARCHITECTURES:UNINITIALIZED=gfx1201\n"
                    "GGML_HIP:BOOL=ON\n",
                    encoding="utf-8",
                )
                with self.assertRaises(runner.ContractError):
                    runner.validate_official_bench("gfx1030", bench)

    def test_schema_rejects_failed_record_serialized_as_pass(self) -> None:
        with self.assertRaises(runner.ContractError):
            runner.schema_validate({"schema_version": "llama-phase5-v1", "record_kind": "result", "state": "FAIL"}, "result", "failed result")

    def test_source_lock_path_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "lock.json"
            fake.write_text("{}", encoding="utf-8")
            with self.assertRaises(runner.ContractError):
                runner.validate_source_lock(fake)

    def test_official_llama_bench_uses_logical_rocm_device_with_uuid_isolation(self) -> None:
        official = self.matrix["official_llama_bench"]
        self.assertEqual(
            official["uuid_isolation"],
            {
                "environment_variable": "ROCR_VISIBLE_DEVICES",
                "value": "exact target gpu_uuid",
                "visible_device_count": 1,
                "llama_bench_device": "ROCm0",
            },
        )
        common = official["common_arguments"]
        self.assertEqual(common[common.index("-dev") + 1], "ROCm0")
        commands = official["commands"]
        command_text = " ".join([commands["prompt_processing"], commands["decode"], *commands["paired"]])
        self.assertNotIn("${GPU_UUID}", command_text)
        self.assertEqual(command_text.count("-dev ROCm0"), 9)

    def test_wrapper_timing_boundaries_follow_completed_operations(self) -> None:
        source = (TOOLS / "llama_phase5_wrapper.cpp").read_text(encoding="utf-8")
        body = source[source.index("Sample run_request"):source.index("uint64_t difference")]
        request_start = body.index("sample.request_start_ns = now_ns(origin);")
        self.assertLess(request_start, body.index("llama_memory_clear(memory, true);"))
        self.assertLess(request_start, body.index("llama_sampler_chain_init"))

        synchronize_positions = [
            position for position in range(len(body))
            if body.startswith("llama_synchronize(context);", position)
        ]
        self.assertEqual(len(synchronize_positions), 2)
        prefill_complete = body.index("sample.prefill_complete_ns = after(origin, sample.prefill_submit_ns);")
        self.assertGreater(prefill_complete, synchronize_positions[0])

        sample_positions = []
        search_from = 0
        while True:
            position = body.find("llama_sampler_sample(sampler, context, -1);", search_from)
            if position < 0:
                break
            sample_positions.append(position)
            search_from = position + 1
        self.assertEqual(len(sample_positions), 2)

        first_publication = body.index("sample.first_token_ns = after(origin, sample.prefill_complete_ns);")
        self.assertGreater(first_publication, sample_positions[0])
        self.assertGreater(first_publication, body.index("if (is_stop(token))", sample_positions[0]))
        first_stop = body.index("sample.stop_ns = after(origin, sample.first_token_ns);", first_publication)
        self.assertGreater(first_stop, first_publication)

        later_publication = body.index("const uint64_t publication = after(origin, previous_publication);")
        later_accept = body.index("llama_sampler_accept(sampler, token);", sample_positions[1])
        self.assertGreater(later_publication, sample_positions[1])
        self.assertGreater(later_publication, synchronize_positions[1])
        self.assertGreater(later_publication, later_accept)
        later_stop = body.index("sample.stop_ns = after(origin, publication);", later_publication)
        self.assertGreater(later_stop, later_publication)

        cleanup = body.index("sample.cleanup_ns = after(origin, sample.stop_ns);")
        self.assertLess(body.rindex("llama_sampler_reset(sampler);"), cleanup)
        self.assertLess(body.rindex("llama_sampler_free(sampler);"), cleanup)
        self.assertLess(body.rindex("llama_memory_clear(memory, true);"), cleanup)

    def test_expected_command_uses_exact_tokens_and_fixed_protocol(self) -> None:
        case = next(item for item in self.matrix["cases"] if item["case_id"] == "short-odd")
        row = next(item for item in self.matrix["rows"] if item["case_id"] == "short-odd")
        tokens = runner.direct_tokens(self.direct, case["direct_sequence_id"])
        command = runner.expected_command(Path("/tmp/wrapper"), row, Path("/tmp/model.gguf"), tokens)
        self.assertEqual(command[0], "/tmp/wrapper")
        self.assertEqual(command[command.index("--input-token-ids") + 1], ",".join(map(str, tokens)))
        self.assertEqual(command[command.index("--warmup-requests") + 1], "3")
        self.assertEqual(command[command.index("--measured-requests") + 1], "10")
        self.assertEqual(command[command.index("--batch-size") + 1], "1")
        self.assertEqual(command[command.index("--sequences") + 1], "1")
        self.assertEqual(command[command.index("--main-gpu") + 1], "0")

    def test_valid_stop_sample_passes_host_validation(self) -> None:
        row = next(item for item in self.matrix["rows"] if item["case_id"] == "decode-long")
        tokens = runner.direct_tokens(self.direct, "decode-long")
        sample = {
            "sample_index": 3,
            "events": {
                "request_start_ns": 10,
                "prefill_submit_ns": 20,
                "prefill_complete_ns": 30,
                "first_token_ns": 40,
                "later_token_publications_ns": [50],
                "stop_ns": 60,
                "cleanup_complete_ns": 70,
            },
            "tokens": {
                "input_token_ids": tokens,
                "generated_token_ids": [7, 248046],
                "visible_token_ids": [7],
                "stop_token_ids_fed_back": [],
                "bos_inserted": False,
            },
            "stop": {"version": 1, "kind": "stop_token", "token_id": 248046},
            "derived": {
                "ttft_ns": 30,
                "prefill_ns": 10,
                "prefill_tokens_per_second": len(tokens) * 100_000_000.0,
                "tpot_ns": [10],
                "decode_tokens": 1,
                "decode_tokens_per_second": 100_000_000.0,
                "e2e_ns": 60,
            },
            "audit": {
                "prefill_logits_index": len(tokens) - 1,
                "prefill_logits_position": len(tokens) - 1,
                "decode_first_position": len(tokens),
            },
        }
        self.assertEqual(runner.validate_sample(sample, row, tokens)["decode_tokens"], 1)

    def test_visible_stop_token_is_rejected(self) -> None:
        row = next(item for item in self.matrix["rows"] if item["case_id"] == "decode-long")
        tokens = runner.direct_tokens(self.direct, "decode-long")
        sample = {
            "sample_index": 3,
            "events": {
                "request_start_ns": 10,
                "prefill_submit_ns": 20,
                "prefill_complete_ns": 30,
                "first_token_ns": 40,
                "later_token_publications_ns": [50],
                "stop_ns": 60,
                "cleanup_complete_ns": 70,
            },
            "tokens": {
                "input_token_ids": tokens,
                "generated_token_ids": [7, 248046],
                "visible_token_ids": [7, 248046],
                "stop_token_ids_fed_back": [],
                "bos_inserted": False,
            },
            "stop": {"version": 1, "kind": "stop_token", "token_id": 248046},
            "derived": {
                "ttft_ns": 30,
                "prefill_ns": 10,
                "prefill_tokens_per_second": len(tokens) * 100_000_000.0,
                "tpot_ns": [10],
                "decode_tokens": 1,
                "decode_tokens_per_second": 100_000_000.0,
                "e2e_ns": 60,
            },
            "audit": {
                "prefill_logits_index": len(tokens) - 1,
                "prefill_logits_position": len(tokens) - 1,
                "decode_first_position": len(tokens),
            },
        }
        with self.assertRaises(runner.ContractError):
            runner.validate_sample(sample, row, tokens)

    def test_build_only_requires_explicit_target(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOOLS / "run_llama_phase5.py"), "--build-only"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"--build-only requires --target", completed.stderr)

    def test_run_row_requires_artifact_inputs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "run_llama_phase5.py"),
                "--run-row",
                "llama-phase5-4b-gfx1030-minimum",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"--run-row requires", completed.stderr)

    def test_aggregate_cli_requires_verified_sllm_aggregate(self) -> None:
        completed = subprocess.run(
            [
                sys.executable, str(TOOLS / "run_llama_phase5.py"), "--aggregate",
                "--artifact-dir", "/tmp/llama-rows", "--output", "/tmp/comparison.json",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout, b"")
        self.assertIn(b"--sllm-aggregate", completed.stderr)

    def test_preflight_failure_does_not_create_retry_blocking_row_directory(self) -> None:
        row_id = "llama-phase5-4b-gfx1030-minimum"
        with tempfile.TemporaryDirectory() as directory:
            artifact_root = Path(directory) / "artifacts"
            build_manifest = {"binary": {"path": "/tmp/wrapper", "sha256": "a" * 64}}
            with patch.object(runner, "validate_model"), patch.object(runner, "validate_source_lock", return_value={}), patch.object(
                runner, "validate_conversion_manifest", return_value={}
            ), patch.object(runner, "validate_build_manifest", return_value=build_manifest), patch.object(
                runner.direct_health, "_amd_smi_observation", side_effect=runner.ContractError("preflight failed")
            ):
                with self.assertRaises(runner.ContractError):
                    runner.run_row(
                        row_id, Path("/tmp/build.json"), runner.CONVERSION_OUTPUT_PATH, artifact_root,
                        conversion_manifest=runner.CONVERSION_SOURCE_MANIFEST_PATH,
                    )
            self.assertFalse(artifact_root.exists())

    def test_wrapper_publishes_no_unobserved_fallback_or_dispatch_claim(self) -> None:
        source = (TOOLS / "llama_phase5_wrapper.cpp").read_text(encoding="utf-8")
        self.assertNotIn("fallback_used", source)
        self.assertNotIn("dispatch_count_nonzero", source)
        schema = json.loads(runner.SCHEMA_PATH.read_text(encoding="utf-8"))
        audit = schema["$defs"]["result"]["properties"]["audit"]
        self.assertNotIn("fallback_used", audit["properties"])
        self.assertNotIn("dispatch_count_nonzero", audit["properties"])

    def test_wrapper_sets_populated_batch_length_after_exact_capacity_allocation(self) -> None:
        source = (TOOLS / "llama_phase5_wrapper.cpp").read_text(encoding="utf-8")
        self.assertIn("batch.n_tokens = static_cast<int32_t>(tokens.size());", source)
        self.assertNotIn("batch capacity does not match token count", source)

    def test_tampered_model_and_schema_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            model.write_bytes(b"not-the-locked-model")
            with self.assertRaises(runner.ContractError):
                runner.validate_model(model)

            schema = Path(directory) / "schema.json"
            schema.write_text("{}", encoding="utf-8")
            with patch.object(runner, "SCHEMA_PATH", schema):
                with self.assertRaises(runner.ContractError):
                    runner.schema_validate({}, "matrix", "tampered schema")


if __name__ == "__main__":
    unittest.main()
