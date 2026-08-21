from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci" / "tools"))

import run_phase36_session_b as runner  # noqa: E402


def _identity_files(directory: Path) -> tuple[Path, Path, Path, Path]:
    values = []
    for name, content in (("binary", b"binary"), ("model", b"gguf"), ("lock", b"lock"), ("source", b"source")):
        path = directory / name
        path.write_bytes(content)
        values.append(path)
    return tuple(values)  # type: ignore[return-value]


def _full_report(encoding: str) -> dict[str, object]:
    fp16 = encoding == "fp16-v1"
    cases = []
    for index in range(runner.FULL_ATTENTION_CASES):
        baseline = 4096 + index * 256
        cases.append(
            {
                "id": f"case-{index}",
                "numerical_match": True,
                "metadata_match": True,
                "no_fallback": True,
                "causal_visibility_match": True,
                "gqa_mapping_match": True,
                "memory_kind": "contiguous-resident",
                "committed_bytes_per_plane": baseline if fp16 else baseline // 2,
                "fp16_committed_bytes_per_plane": baseline,
            }
        )
    return {
        "schema_version": "sllm-full-attention-g1-evidence-v2",
        "state": "PASS",
        "pass": True,
        "target": "gfx942",
        "kv_encoding": encoding,
        "selected_backend": "hip",
        "gpu_execution": True,
        "cpu_fallback_used": False,
        "fallback_allowed": False,
        "fallback_used": False,
        "cases": cases,
        "oracle": {
            "scalar_ordered_dot_softmax_v": True,
            "fp16_subnormal_affects_score": True,
            "final_bf16_rne_checked": True,
            "gqa_heads_checked": True,
        },
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True},
    }


def _kv_state_report() -> dict[str, object]:
    return {
        "schema_version": "sllm-kv-state-g1-evidence-v1",
        "state": "PASS",
        "pass": True,
        "target": "gfx942",
        "selected_backend": "hip",
        "gpu_execution": True,
        "cpu_fallback_used": False,
        "fallback_allowed": False,
        "fallback_used": False,
        "cases": [
            {"normal_length_generation": True, "metadata_layout": True, "no_fallback_observed": True, "exact_fp16_storage_observed": True}
            for _ in range(runner.KV_STATE_CASES)
        ],
        "oracle": {
            "special_values_checked": True,
            "rounding_values_checked": True,
            "token_major_placement_checked": True,
            "exact_storage_readback_available": True,
        },
        "transactions": {
            "stale_rejection": True,
            "one_in_flight_rejection": True,
            "timeout_observed": True,
            "drop_cancel_no_publication": True,
            "pending_readback_rejection": True,
        },
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "zero_after_shutdown": True},
    }


def _oracle() -> dict[str, object]:
    return {
        "schema_version": "phase36-lowbit-oracle-v1",
        "state": "PASS",
        "pass": True,
        "implementation": "numpy",
        "backend": "numpy",
        "torch_used": False,
        "gpu_used": False,
        "encodings": ["kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"],
        "cases": 17,
        "all_codes_checked": True,
        "nan_inf_checked": True,
        "rounding_checked": True,
        "saturation_checked": True,
        "no_fp16_mirror": True,
    }


def _rows(raw_dir: Path) -> list[dict[str, object]]:
    input_ids = [runner.EXPECTED_INPUT_ID] * runner.INPUT_TOKENS
    result = []
    for index, (encoding, setting) in enumerate(runner.EXPECTED_ROWS):
        selected = runner.INPUT_TOKENS if setting == "auto" else int(setting)
        count = 1 if setting == "auto" else (runner.INPUT_TOKENS + int(setting) - 1) // int(setting)
        row_id = f"row-{index}-{encoding}-{setting}"
        (raw_dir / f"sysfs-{row_id}.tsv").write_text("sample_ns\thbm_used_bytes\tgtt_used_bytes\n0\t100\t0\n1\t100\t0\n", encoding="utf-8")
        result.append(
            {
                "row_id": row_id,
                "kv_cache_encoding": encoding,
                "chunk_setting": setting,
                "selected_chunk_tokens": selected,
                "chunk_count": count,
                "target": "gfx942",
                "selected_backend": "hip",
                "fallback_used": False,
                "cpu_fallback_used": False,
                "partial_offload": False,
                "state": "PASS",
                "pass": True,
                "weight_dtype": "bf16",
                "input_tokens": runner.INPUT_TOKENS,
                "output_tokens": runner.OUTPUT_TOKENS,
                "input_ids": input_ids,
                "output_ids": [23066, 23066],
                "numerical_match": True,
                "memory_kind": "contiguous-resident",
                "committed_bytes": 100_000,
                "available_bytes": 200_000,
                "request_state_bytes": 379_256_832 if encoding == "fp16-v1" else 217_961_216,
                "arena_high_water_bytes": 5_278_049_280,
                "arena_separate_allocation_bytes": 39_950_821_120,
                "cleanup_retryable": 0,
                "cleanup_durable": 0,
                "terminal_zero": True,
                "sysfs_tsv": f"sysfs-{row_id}.tsv",
            }
        )
    return result


def _populate(directory: Path) -> tuple[Path, Path, Path, Path]:
    binary, model, lock, source = _identity_files(directory)
    raw = directory / "raw"
    raw.mkdir()
    for encoding, names in runner.FULL_REPORT_FILES.items():
        (raw / names[0]).write_text(json.dumps(_full_report(encoding)), encoding="utf-8")
    (raw / runner.KV_STATE_FILES[0]).write_text(json.dumps(_kv_state_report()), encoding="utf-8")
    (raw / runner.ORACLE_FILES[0]).write_text(json.dumps(_oracle()), encoding="utf-8")
    rows = _rows(raw)
    (raw / runner.ROW_FILES[0]).write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return binary, model, lock, source


def _populate_native(directory: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Populate the retained native frontend shape used by Session B."""
    binary, model, lock, source = _populate(directory)
    raw = directory / "raw"
    model_variant = directory / "bf16.gguf"
    model_variant.write_bytes(b"bf16-gguf")
    lock_variant = directory / "bf16.lock.json"
    source_fp = "f" * 64
    lock_variant.write_text(json.dumps({"fingerprint": "sha256:" + "1" * 64, "source_lock_fingerprints": ["sha256:" + source_fp]}), encoding="utf-8")
    input_ids = [runner.EXPECTED_INPUT_ID] * runner.INPUT_TOKENS
    for filename in runner.NATIVE_ROW_FILES:
        stem = Path(filename).stem
        parts = stem.split("-")
        encoding = parts[1]
        setting = "-".join(parts[2:])
        if setting.startswith("chunk-"):
            setting = setting[6:]
        requested = runner.INPUT_TOKENS if setting == "auto" else int(setting)
        selected = min(requested, runner.INPUT_TOKENS)
        count = 1 if setting == "auto" else (runner.INPUT_TOKENS + requested - 1) // requested
        document = {
            "schema_version": "model-frontend-cli-report-v1",
            "state": "PASS",
            "model": {"lock_fingerprint": "sha256:" + source_fp},
            "result": {
                "input_token_ids": input_ids,
                "generated_token_ids": [runner.EXPECTED_INPUT_ID, runner.EXPECTED_INPUT_ID],
                "usage": {"prompt_tokens": runner.INPUT_TOKENS, "completion_tokens": runner.OUTPUT_TOKENS},
                "execution": {
                    "target": "gfx942", "selected_backend": "hip", "fallback_used": False,
                    "all_dispatches_hip": True, "kv_cache_encoding": "fp16" if encoding == "fp16" else "fp8",
                    "prefill_chunk_capacity_tokens": selected, "prefill_chunk_count": count,
                    "placement_required_bytes": 100_000, "placement_available_memory_bytes": 200_000,
                    "placement_request_state_bytes": 1000 if encoding == "fp16" else 500, "workspace_arena_bytes": 5000,
                    "workspace_separate_allocation_bytes": 10000, "weight_encoding": "bf16",
                    "fp8_provider": None, "model_fingerprint": "sha256:" + source_fp,
                },
                "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0},
            },
        }
        (raw / filename).write_text(json.dumps(document), encoding="utf-8")
        (raw / f"{stem}-memory.tsv").write_text(
            "base_vram_bytes\t100\npeak_vram_bytes\t200\nsettled_vram_bytes\t100\n"
            "base_gtt_bytes\t42\npeak_gtt_bytes\t80\nsettled_gtt_bytes\t42\n",
            encoding="utf-8",
        )
    return binary, model, lock, source, model_variant, lock_variant


class Phase36SessionBRunnerTests(unittest.TestCase):
    def test_aggregate_accepts_complete_retained_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source = _populate(directory)
            summary = runner.aggregate(raw_dir=directory / "raw", output_dir=directory / "out", binary=binary, model=model, lock=lock, source_identity=source, target="gfx942")
        self.assertEqual(summary["state"], "PASS")
        self.assertEqual(summary["model_rows"]["selected_rows"], 12)
        self.assertEqual(len(summary["full_attention_reports"]), 4)
        self.assertTrue(summary["memory"]["no_gtt_spill"])
        self.assertEqual(summary["comparisons"]["output_ids"], [23066, 23066])

    def test_exact_target_and_identity_arguments_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source = _identity_files(directory)
            with self.assertRaisesRegex(runner.SessionBError, "exact target"):
                runner.aggregate(raw_dir=directory, output_dir=directory / "out", binary=binary, model=model, lock=lock, source_identity=source, target="gfx1201")
            with self.assertRaises(SystemExit):
                runner.parse_args(["--raw-dir", str(directory), "--output-dir", str(directory), "--target", "gfx942"])

    def test_model_row_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source = _populate(directory)
            rows_path = directory / "raw" / runner.ROW_FILES[0]
            document = json.loads(rows_path.read_text())
            document["rows"][0]["output_ids"] = [23066, 123]
            rows_path.write_text(json.dumps(document))
            with self.assertRaisesRegex(runner.SessionBError, "exact two output"):
                runner.aggregate(raw_dir=directory / "raw", output_dir=directory / "out", binary=binary, model=model, lock=lock, source_identity=source)

    def test_gtt_spill_and_noncontiguous_report_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source = _populate(directory)
            report_path = directory / "raw" / runner.FULL_REPORT_FILES["kv-fp8-v1"][0]
            report = json.loads(report_path.read_text())
            report["cases"][0]["memory_kind"] = "virtual-contiguous"
            report_path.write_text(json.dumps(report))
            with self.assertRaisesRegex(runner.SessionBError, "contiguous-resident"):
                runner.aggregate(raw_dir=directory / "raw", output_dir=directory / "out", binary=binary, model=model, lock=lock, source_identity=source)

    def test_numpy_oracle_rejects_non_numpy_or_missing_encoding(self) -> None:
        bad = copy.deepcopy(_oracle())
        bad["implementation"] = "torch"
        with self.assertRaisesRegex(runner.SessionBError, "independent NumPy"):
            runner.validate_lowbit_oracle(bad)

    def test_native_rows_bind_raw_json_tsv_digests_and_bf16_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source, bf16_model, bf16_lock = _populate_native(directory)
            summary = runner.aggregate(
                raw_dir=directory / "raw", output_dir=directory / "out", binary=binary, model=model, lock=lock,
                source_identity=source, bf16_model=bf16_model, bf16_lock=bf16_lock,
            )
            self.assertEqual(summary["state"], "PASS")
            rows = summary["model_rows"]["rows"]
            self.assertEqual(len(rows), 12)
            for row in rows:
                self.assertEqual(row["raw_sha256"], runner._sha256_file(directory / "raw" / f"{row['row_id']}.json"))
                self.assertEqual(row["sysfs_gtt_baseline_bytes"], 42)
                self.assertEqual(row["sysfs_gtt_incremental_bytes"], 0)
                self.assertEqual(row["settled_gtt_bytes"], 42)
                self.assertEqual(row["weight_dtype"], "bf16")

    def test_native_fp8_gguf_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            binary, model, lock, source, bf16_model, bf16_lock = _populate_native(directory)
            path = directory / "raw" / "long-fp8-auto.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["result"]["execution"]["weight_encoding"] = "e4m3fnuz-converted-from-ocp-e4m3fn-outer-f32"
            document["result"]["execution"]["fp8_provider"] = "native-fnuz"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(runner.SessionBError, "canonical BF16 GGUF"):
                runner.aggregate(
                    raw_dir=directory / "raw", output_dir=directory / "out", binary=binary, model=model, lock=lock,
                    source_identity=source, bf16_model=bf16_model, bf16_lock=bf16_lock,
                )


if __name__ == "__main__":
    unittest.main()
