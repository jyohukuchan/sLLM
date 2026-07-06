from __future__ import annotations

import importlib.util
import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str):
    path = REPO_ROOT / "tools" / filename
    module_name = filename.replace("-", "_").removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = load_tool("build-sq-candidate-runtime-row.py")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class SqCandidateRuntimeRowTests(unittest.TestCase):
    def test_builds_baseline_anchor_row_from_prompt_suite_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "case-a.json"
            summary = root / "summary.json"
            guard = root / "guard-bundle-summary.json"
            write_json(
                report,
                {
                    "backend": "hip",
                    "memory": {
                        "kv_cache_bytes": 4096,
                        "kv_cache_value_dtype": "f32",
                    },
                },
            )
            write_json(
                summary,
                {
                    "schema_version": "package-token-prompt-suite-summary-v0.3",
                    "suite": {"suite_id": "unit-suite"},
                    "tokenizer_dir": "/models/qwen",
                    "device_index": 2,
                    "cases": [
                        {
                            "id": "case-a",
                            "report": str(report),
                            "p50_ms": 50.0,
                        }
                    ],
                    "metrics": {
                        "prefill_tps_mean": 18.0,
                        "decode_tps_mean": 20.0,
                        "decode_tps_min": 19.0,
                        "decode_tps_max": 21.0,
                        "output_ok_count": 1,
                        "output_warn_count": 0,
                        "output_not_evaluated_count": 0,
                        "verified_all": True,
                    },
                },
            )
            write_json(guard, {"schema_version": "unit", "passed": True})

            row = TOOL.build_row(
                types.SimpleNamespace(
                    suite_summary=summary,
                    guard_bundle=guard,
                    output_jsonl=root / "out.jsonl",
                    run_id="run",
                    case_id="case",
                    candidate_id="aq4",
                    format_version="aq4-prototype-current-runtime",
                    description="baseline",
                    package_or_runtime_artifact="/tmp/pkg.ullm.d",
                    source_aq_policy="policy",
                    row_scale_override_policy="preserved",
                    host="WRX80",
                    architecture="RDNA4",
                    gpu_name="R9700",
                    golden_prefix_artifact="golden.jsonl",
                    golden_prefix_verified=True,
                    compact_resident_bytes=None,
                    materialized_working_set_bytes=None,
                    materialization_granularity="current_runtime_mixed_resident_not_measured",
                    materialization_wall_ms=None,
                    whole_model_f32_resident=False,
                    baseline_anchor=True,
                    append=False,
                    note=["unit"],
                )
            )

        self.assertEqual(row["schema_version"], TOOL.SCHEMA_VERSION)
        self.assertEqual(row["hardware"]["backend"], "hip")
        self.assertEqual(row["hardware"]["device_index"], 2)
        self.assertEqual(row["storage"]["kv_cache_bytes"], 4096)
        self.assertEqual(row["timing"]["decode_tps_mean"], 20.0)
        self.assertEqual(row["timing"]["decode_p50_ms_mean"], 50.0)
        self.assertTrue(row["decision"]["comparable_to_baseline"])
        self.assertTrue(row["decision"]["accepted_for_next_iteration"])

    def test_non_anchor_row_requires_storage_timing_and_guard_fields(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "case-a.json"
            summary = root / "summary.json"
            guard = root / "guard-bundle-summary.json"
            write_json(report, {"backend": "hip", "memory": {"kv_cache_value_dtype": "f32"}})
            write_json(
                summary,
                {
                    "suite": {"suite_id": "unit-suite"},
                    "device_index": 2,
                    "cases": [{"id": "case-a", "report": str(report)}],
                    "metrics": {"decode_tps_mean": 20.0, "verified_all": False},
                },
            )
            write_json(guard, {"passed": False})

            row = TOOL.build_row(
                types.SimpleNamespace(
                    suite_summary=summary,
                    guard_bundle=guard,
                    output_jsonl=root / "out.jsonl",
                    run_id="run",
                    case_id="case",
                    candidate_id="sq",
                    format_version="sq-format-v0.1",
                    description="candidate",
                    package_or_runtime_artifact="/tmp/pkg.ullm.d",
                    source_aq_policy="policy",
                    row_scale_override_policy="preserved",
                    host="WRX80",
                    architecture="RDNA4",
                    gpu_name="R9700",
                    golden_prefix_artifact=None,
                    golden_prefix_verified=True,
                    compact_resident_bytes=None,
                    materialized_working_set_bytes=None,
                    materialization_granularity="layer_window",
                    materialization_wall_ms=None,
                    whole_model_f32_resident=False,
                    baseline_anchor=False,
                    append=False,
                    note=[],
                )
            )

        self.assertFalse(row["decision"]["comparable_to_baseline"])
        reason = row["decision"]["reason"]
        self.assertIn("storage.compact_resident_bytes", reason)
        self.assertIn("storage.materialized_working_set_bytes", reason)
        self.assertIn("timing.materialization_wall_ms", reason)
        self.assertIn("quality.verified_all", reason)
        self.assertIn("guards.prompt_guard_bundle.passed", reason)


if __name__ == "__main__":
    unittest.main()
