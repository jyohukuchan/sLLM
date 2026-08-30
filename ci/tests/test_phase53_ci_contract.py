#!/usr/bin/env python3
"""Host-only fail-closed contracts for Phase 53 KV default evidence."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "ci/schema"
POLICY = ROOT / "ci/policy/kv-cache-default-v1.json"
POLICY_V2 = ROOT / "ci/policy/kv-cache-default-v2.json"
TOOL = ROOT / "ci/tools/aggregate_phase53_kv_default.py"
SPEC = importlib.util.spec_from_file_location("aggregate_phase53", TOOL)
assert SPEC and SPEC.loader
aggregate_phase53 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_phase53)


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(character: str) -> str:
    return "sha256:" + character * 64


def policy_digest() -> str:
    return aggregate_phase53.sha256_bytes(POLICY.read_bytes())


def policy_v2_digest() -> str:
    return aggregate_phase53.sha256_bytes(POLICY_V2.read_bytes())


def correctness(target: str = "gfx1030") -> dict[str, object]:
    encoding = "kv-fp8-e5-block16" if target == "gfx1030" else "kv-fp8-e4-block16"
    variant = {"gfx1030": "E5M2-software", "gfx1201": "E4M3-OCP", "gfx942:sramecc+:xnack-": "E4M3-FNUZ"}[target]
    return {
        "$schema": "https://sllm.dev/schema/phase53-kv-fp8-block16-evidence-v1.schema.json",
        "schema_version": "sllm-phase53-kv-fp8-block16-evidence-v1", "state": "PASS",
        "target": target, "device_index": 0, "encoding": encoding, "physical_variant": variant,
        "policy_sha256": policy_digest(), "binary_sha256": digest("a"),
        "host": {"pass": True, "variants": ["E4M3-FNUZ", "E4M3-OCP", "E5M2-software"], "head_dimensions": [15, 16, 17, 255, 256, 257], "special_values": ["zero", "tiny", "max-finite", "nan", "positive-infinity", "negative-infinity", "positive-zero", "negative-zero"], "tail_padding_zero": True, "key_value_scales_independent": True},
        "cases": [{"head_dim": dimension, "value_bytes_exact": True, "key_scales_exact": True, "value_scales_exact": True, "key_value_scales_independent": True, "tail_padding_zero": True, "append_direct": True, "attention_direct": dimension == 256, "numerical_match": True} for dimension in (15, 16, 17, 255, 256, 257)],
        "execution": {"selected_backend": "hip", "gpu_execution": True, "fallback_allowed": False, "fallback_used": False, "append_dispatches": 6, "attention_dispatches": 1},
        "cleanup": {"retryable": 0, "durable": 0, "terminal_zero": True}, "error": None,
    }


def correctness_v2(target: str = "gfx1030") -> dict[str, object]:
    report = correctness(target)
    descriptor = "kv-fp8-e5-block16-v2" if target == "gfx1030" else "kv-fp8-e4-block16-v2"
    report.update({
        "$schema": "https://sllm.dev/schema/phase53-kv-fp8-block16-evidence-v2.schema.json",
        "schema_version": "sllm-phase53-kv-fp8-block16-evidence-v2",
        "descriptor_id": descriptor,
        "scale_recipe": "standard-mx-floor-power-v1",
        "policy_sha256": policy_v2_digest(),
    })
    report["host"]["special_values"].insert(3, "standard-mx-saturation-boundary")
    return report


def mxfp8_correctness(target: str = "gfx1030") -> dict[str, object]:
    return {
        "$schema": "https://sllm.dev/schema/phase53-kv-mxfp8-evidence-v1.schema.json",
        "schema_version": "sllm-phase53-kv-mxfp8-evidence-v1", "state": "PASS",
        "target": target, "device_index": 0, "encoding": "kv-mxfp8-e4", "physical_variant": "E4M3-OCP",
        "policy_sha256": policy_digest(), "binary_sha256": digest("f"),
        "host": {"pass": True, "variants": ["E4M3-OCP"], "head_dimensions": [31, 32, 33, 255, 256, 257], "special_values": ["zero", "tiny", "max-finite", "nan", "positive-infinity", "negative-infinity", "positive-zero", "negative-zero"], "tail_padding_zero": True, "key_value_scales_independent": True},
        "cases": [{"head_dim": dimension, "value_bytes_exact": True, "key_scales_exact": True, "value_scales_exact": True, "key_value_scales_independent": True, "tail_padding_zero": True, "append_direct": True, "attention_direct": dimension == 256, "numerical_match": True} for dimension in (31, 32, 33, 255, 256, 257)],
        "execution": {"selected_backend": "hip", "gpu_execution": True, "fallback_allowed": False, "fallback_used": False, "append_dispatches": 6, "attention_dispatches": 1},
        "cleanup": {"retryable": 0, "durable": 0, "terminal_zero": True}, "error": None,
    }


def quality(target: str = "gfx1030", **metrics: float) -> dict[str, object]:
    encoding = "kv-fp8-e5-block16" if target == "gfx1030" else "kv-fp8-e4-block16"
    comparison_encoding = "kv-mxfp8-e5" if target == "gfx1030" else "kv-mxfp8-e4"
    defaults = {"perplexity_relative_delta": 0.01, "kld_p99": 0.049, "top1_agreement": 0.99, "task_score_delta": 0.02, "long_context_score_delta": 0.02}
    defaults.update(metrics)
    repeats = []
    for repeat in range(1, 4):
        metric = {"selected_count": 20, "metric_sample_counts": {"perplexity": 10, "kld": 20, "top1": 20, "task": 10, "long-context": 6}, **defaults, "hip_dispatches": 1, "fallback_used": False, "all_dispatches_hip": True}
        if target == "gfx942:sramecc+:xnack-":
            repeats.append({"repeat": repeat, "fp16_released_before_block16": True, "block16_released_after_repeat": True, "block16": copy.deepcopy(metric)})
        else:
            repeats.append({"repeat": repeat, "fp16_released_before_block16": True, "block16_released_before_mxfp8": True, "mxfp8_released_after_repeat": True, "block16": copy.deepcopy(metric), "mxfp8": copy.deepcopy(metric)})
    report = {
        "$schema": "https://sllm.dev/schema/phase53-qwen35-kv-quality-candidate-v1.schema.json",
        "schema_version": "sllm-phase53-qwen35-kv-quality-candidate-v1", "state": "PASS",
        "identity": {"policy_sha256": policy_digest(), "dataset_sha256": "sha256:a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d", "model_lock_fingerprint": aggregate_phase53.MODEL_FINGERPRINT, "model_lock_sha256": digest("b"), "derived_lock_fingerprint": digest("c"), "derived_lock_sha256": digest("d"), "binary_sha256": digest("e")},
        "target": target, "device_index": 0, "encoding": encoding, "reference_encoding": "fp16", "sequential_residents": True,
        "selected_count": 20, "repeats": repeats, "cleanup": {"retryable": 0, "durable": 0, "terminal_zero": True},
    }
    if target == "gfx942:sramecc+:xnack-":
        report["mxfp8_comparison"] = {"status": "unsupported", "encoding": None, "reference_only": True, "reason": "gfx942 OCP MXFP8 is intentionally unsupported because CDNA3 FNUZ element bytes differ"}
        report["completely_sequential_order"] = ["fp16", "block16"]
    else:
        report["mxfp8_comparison"] = {"status": "complete", "encoding": comparison_encoding, "reference_only": True}
        report["completely_sequential_order"] = ["fp16", "block16", "mxfp8"]
    return report


def quality_v2(target: str = "gfx1030", **metrics: float) -> dict[str, object]:
    report = quality(target, **metrics)
    descriptor = "kv-fp8-e5-block16-v2" if target == "gfx1030" else "kv-fp8-e4-block16-v2"
    report.update({
        "$schema": "https://sllm.dev/schema/phase53-qwen35-kv-quality-candidate-v2.schema.json",
        "schema_version": "sllm-phase53-qwen35-kv-quality-candidate-v2",
    })
    report["identity"].update({
        "policy_sha256": policy_v2_digest(),
        "descriptor_id": descriptor,
        "scale_recipe": "standard-mx-floor-power-v1",
    })
    return report


def performance_resource(target: str = "gfx1030") -> dict[str, object]:
    encoding = "kv-fp8-e5-block16" if target == "gfx1030" else "kv-fp8-e4-block16"
    cases = (
        ("short-odd", "normal", 17, 17, 3, 10, 34, False),
        ("32-32", "normal", 32, 32, 3, 10, 64, False),
        ("prefill-long", "normal", 1024, 128, 3, 10, 1152, False),
        ("decode-long", "normal", 32, 256, 3, 10, 288, False),
        ("long-10001", "normal", 10001, 2, 3, 10, 10003, False),
        ("long-100000", "long-running", 100000, 2, 1, 3, 131072, False),
        ("decode-20000", "long-running", 32, 20000, 1, 3, 131072, True),
    )
    rows = []
    for row, (case_id, case_class, inputs, outputs, warmups, measured, context, ignore_eos) in enumerate(cases, 1):
        fp16 = {"direct_report_sha256": digest(str(row)), "row_id": f"fp16-{case_id}", "median_e2e_ns": 2.0, "tokens_per_second": 1.0, "logical_kv_bytes": context * 32768, "physical_kv_bytes": 8192, "hbm_peak_delta_bytes": 100, "generated": True, "hip_only": True, "fallback_used": False, "cleanup_empty": True, "hbm_gtt_settled": True}
        candidate = {"direct_report_sha256": digest("abcdef0"[row - 1]), "row_id": f"candidate-{case_id}", "median_e2e_ns": 1.0, "tokens_per_second": 2.0, "logical_kv_bytes": context * 17408, "physical_kv_bytes": 4096, "hbm_peak_delta_bytes": 80, "generated": True, "hip_only": True, "fallback_used": False, "cleanup_empty": True, "hbm_gtt_settled": True}
        rows.append({"row": row, "case_id": case_id, "class": case_class, "input_token_count": inputs, "requested_output_tokens": outputs, "protocol": {"warmups": warmups, "measured": measured, "context_length": context, "ignore_eos": ignore_eos}, "execution_order": ["fp16", "block16"], "fp16": fp16, "candidate": candidate, "candidate_to_fp16_throughput_ratio": 2.0})
    return {
        "$schema": "https://sllm.dev/schema/phase53-performance-resource-evidence-v1.schema.json",
        "schema_version": "sllm-phase53-performance-resource-evidence-v1", "state": "PASS", "target": target, "encoding": encoding,
        "producer": "ci/tools/build_phase53_performance_resource.py", "policy_sha256": policy_digest(), "binary_sha256": digest("7"), "hbm_observation_sha256": digest("8"), "model_lock_fingerprint": aggregate_phase53.MODEL_FINGERPRINT, "selected_count": 7,
        "rows": rows,
        "memory": {"fp16_bytes_per_token_head_plane": 512, "candidate_bytes_per_token_head_plane": 272, "logical_reduction_fraction": 0.46875, "candidate_physical_kv_bytes_max": 4096, "physical_measured": True},
        "decisions": {"performance": "pass", "resource": "pass", "memory": "pass"}, "fallback_used": False,
        "cleanup": {"retryable": 0, "durable": 0, "settled": True},
    }


def performance_resource_v2(target: str = "gfx1030") -> dict[str, object]:
    report = performance_resource(target)
    descriptor = "kv-fp8-e5-block16-v2" if target == "gfx1030" else "kv-fp8-e4-block16-v2"
    report.update({
        "$schema": "https://sllm.dev/schema/phase53-performance-resource-evidence-v2.schema.json",
        "schema_version": "sllm-phase53-performance-resource-evidence-v2",
        "descriptor_id": descriptor,
        "scale_recipe": "standard-mx-floor-power-v1",
        "policy_sha256": policy_v2_digest(),
    })
    return report


def external_hbm(target: str = "gfx1030") -> dict[str, object]:
    cases = []
    timestamp = 1
    for row, case_id in enumerate(("short-odd", "32-32", "prefill-long", "decode-long", "long-10001", "long-100000", "decode-20000"), 1):
        def run(source_digest: str) -> dict[str, object]:
            nonlocal timestamp
            start = timestamp
            timestamp += 10
            return {"direct_report_sha256": source_digest, "started_ns": start, "completed_ns": timestamp, "baseline_hbm_bytes": 100, "peak_hbm_bytes": 200, "settled_hbm_bytes": 100, "baseline_gtt_bytes": 50, "peak_gtt_bytes": 70, "settled_gtt_bytes": 50, "monitor_samples": 2, "process_group_gone": True, "settled": True}
        cases.append({"case_id": case_id, "execution_order": ["fp16", "block16"], "fp16": run(digest(str(row))), "candidate": run(digest("abcdef0"[row - 1]))})
    return {"$schema": "https://sllm.dev/schema/phase53-external-hbm-observation-v1.schema.json", "schema_version": "sllm-phase53-external-hbm-observation-v1", "target": target, "binary_sha256": digest("7"), "cases": cases}


class Phase53CiContractTests(unittest.TestCase):
    def assert_valid(self, schema: str, value: dict[str, object]) -> None:
        validator = Draft202012Validator(load(SCHEMA / schema), format_checker=FormatChecker())
        errors = list(validator.iter_errors(value))
        self.assertEqual(errors, [], "; ".join(error.message for error in errors))

    def write(self, root: Path, name: str, value: dict[str, object]) -> Path:
        path = root / name
        path.write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
        return path

    def decide(self, candidate: dict[str, object], oracle: dict[str, object] | None = None) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qpath = self.write(root, "quality.json", candidate)
            cpath = self.write(root, "correctness.json", oracle or correctness())
            ppath = self.write(root, "performance.json", performance_resource())
            summary, _ = aggregate_phase53.aggregate(POLICY, [cpath], [qpath], [ppath])
            return summary["targets"][2]["decision"]

    def decide_v2(self, candidate: dict[str, object], oracle: dict[str, object] | None = None) -> str:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qpath = self.write(root, "quality.json", candidate)
            cpath = self.write(root, "correctness.json", oracle or correctness_v2())
            ppath = self.write(root, "performance.json", performance_resource_v2())
            summary, _ = aggregate_phase53.aggregate(POLICY_V2, [cpath], [qpath], [ppath])
            return summary["targets"][2]["decision"]

    def test_phase53_schemas_are_draft202012_and_fixtures_validate(self) -> None:
        for path in sorted(SCHEMA.glob("phase53-*.schema.json")):
            Draft202012Validator.check_schema(load(path))
        self.assert_valid("phase53-kv-fp8-block16-evidence-v1.schema.json", correctness())
        for target in ("gfx942", "gfx942:sramecc+:xnack-", "gfx1201", "gfx1030"):
            self.assert_valid("phase53-kv-mxfp8-evidence-v1.schema.json", mxfp8_correctness(target))
        self.assert_valid("phase53-qwen35-kv-quality-candidate-v1.schema.json", quality())
        self.assert_valid("phase53-qwen35-kv-quality-candidate-v1.schema.json", quality("gfx942:sramecc+:xnack-"))
        for target in aggregate_phase53.TARGETS:
            self.assert_valid("phase53-kv-fp8-block16-evidence-v2.schema.json", correctness_v2(target))
            self.assert_valid("phase53-qwen35-kv-quality-candidate-v2.schema.json", quality_v2(target))
            self.assert_valid("phase53-performance-resource-evidence-v2.schema.json", performance_resource_v2(target))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, mapping = aggregate_phase53.aggregate(POLICY, [self.write(root, "c.json", correctness())], [self.write(root, "q.json", quality())], [self.write(root, "p.json", performance_resource())])
        self.assert_valid("phase53-kv-default-summary-v1.schema.json", summary)
        self.assert_valid("phase53-runtime-mapping-candidate-v1.schema.json", mapping)
        self.assert_valid("phase53-performance-resource-evidence-v1.schema.json", performance_resource())
        self.assert_valid("phase53-external-hbm-observation-v1.schema.json", external_hbm())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary_v2, mapping_v2 = aggregate_phase53.aggregate(
                POLICY_V2,
                [self.write(root, "c-v2.json", correctness_v2())],
                [self.write(root, "q-v2.json", quality_v2())],
                [self.write(root, "p-v2.json", performance_resource_v2())],
            )
        self.assert_valid("phase53-kv-default-summary-v2.schema.json", summary_v2)
        self.assert_valid("phase53-runtime-mapping-candidate-v2.schema.json", mapping_v2)

    def test_v2_policy_inherits_pre_candidate_freeze_and_only_advances_descriptor(self) -> None:
        v1 = load(POLICY)
        v2 = load(POLICY_V2)
        self.assert_valid("kv-cache-default-v2.schema.json", v2)
        for field in ("dataset", "context_matrix", "metrics", "repeats", "baseline_requirements"):
            self.assertEqual(v2["freeze"][field], v1["freeze"][field])
        for field in ("thresholds", "failure_rules"):
            self.assertEqual(v2[field], v1[field])
        for old, revised in zip(v1["targets"], v2["targets"]):
            for field in ("target", "baseline"):
                self.assertEqual(revised[field], old[field])
            self.assertEqual(revised["kv_descriptor"], old["kv_descriptor"].replace("-v1", "-v2"))
        self.assertIn("inherited unchanged", v2["freeze"]["rationale"])

    def test_v2_recipe_evidence_is_strictly_version_paired(self) -> None:
        self.assertEqual(self.decide_v2(quality_v2()), "adopt")
        for field, value in (
            ("descriptor_id", "kv-fp8-e4-block16-v2"),
            ("scale_recipe", "legacy-max-ratio"),
        ):
            with self.subTest(kind="quality", field=field):
                report = quality_v2()
                report["identity"][field] = value
                with self.assertRaises(aggregate_phase53.ContractError):
                    self.decide_v2(report)
            with self.subTest(kind="correctness", field=field):
                report = correctness_v2()
                report[field] = value
                with self.assertRaises(aggregate_phase53.ContractError):
                    self.decide_v2(quality_v2(), report)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(aggregate_phase53.ContractError):
                aggregate_phase53.aggregate(
                    POLICY_V2,
                    [self.write(root, "c-v1.json", correctness())],
                    [self.write(root, "q-v2.json", quality_v2())],
                )
            with self.assertRaises(aggregate_phase53.ContractError):
                aggregate_phase53.aggregate(
                    POLICY,
                    [self.write(root, "c-v2.json", correctness_v2())],
                    [self.write(root, "q-v1.json", quality())],
                )

    def test_cli_and_benchmark_selection_manifest_is_closed_and_additive(self) -> None:
        selection = {"requested": "kv-mxfp8-e4", "resolved": "kv-mxfp8-e4", "selection_source": "process-explicit", "reason": "explicit reviewed target selection", "physical_variant": "E4M3-OCP", "descriptor_id": "kv-mxfp8-e4-v1", "policy_version": 1}
        for schema_name in ("model-frontend-cli-report-v1.schema.json", "engine-performance-direct-v2.schema.json"):
            document = load(SCHEMA / schema_name)
            Draft202012Validator.check_schema(document)
            validator = Draft202012Validator(document["$defs"]["kv_cache_selection"])
            self.assertEqual(list(validator.iter_errors(selection)), [])
            unknown = copy.deepcopy(selection)
            unknown["pointer"] = "forbidden"
            self.assertTrue(list(validator.iter_errors(unknown)))

    def test_missing_target_evidence_is_insufficient_not_cross_target_failure(self) -> None:
        summary, mapping = aggregate_phase53.aggregate(POLICY, [], [])
        self.assertEqual([row["decision"] for row in summary["targets"]], ["insufficient-evidence"] * 3)
        self.assertEqual(mapping["mappings"], [])

    def test_missing_zero_nonfinite_and_digest_mismatch_fail_closed(self) -> None:
        missing = quality()
        del missing["repeats"][0]["block16"]["kld_p99"]
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(missing)
        zero = quality()
        zero["repeats"][0]["block16"]["metric_sample_counts"]["task"] = 0
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(zero)
        nonfinite = quality()
        nonfinite["repeats"][0]["block16"]["perplexity_relative_delta"] = math.nan
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(nonfinite)
        nonfinite_mxfp8 = quality()
        nonfinite_mxfp8["repeats"][0]["mxfp8"]["kld_p99"] = math.inf
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(nonfinite_mxfp8)
        mismatch = quality()
        mismatch["identity"]["policy_sha256"] = digest("9")
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(mismatch)

    def test_correctness_host_cases_dispatches_and_variant_fail_closed(self) -> None:
        mutations = {
            "host-pass-false": lambda report: report["host"].__setitem__("pass", False),
            "empty-cases": lambda report: report.__setitem__("cases", []),
            "zero-append-dispatches": lambda report: report["execution"].__setitem__("append_dispatches", 0),
            "zero-attention-dispatches": lambda report: report["execution"].__setitem__("attention_dispatches", 0),
            "wrong-physical-variant": lambda report: report.__setitem__("physical_variant", "E4M3-OCP"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = correctness()
                mutate(report)
                with self.assertRaises(aggregate_phase53.ContractError):
                    self.decide(quality(), report)

    def test_failed_quality_cleanup_and_malformed_quality_fail_closed(self) -> None:
        cleanup = quality()
        cleanup["cleanup"]["terminal_zero"] = False
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(cleanup)

        malformed = quality()
        malformed["selected_count"] = "20"
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(malformed)

    def test_every_aggregate_input_rejects_unknown_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = load(POLICY)
            policy["unknown"] = True
            with self.assertRaises(aggregate_phase53.ContractError):
                aggregate_phase53.aggregate(self.write(root, "policy.json", policy), [], [])

        for kind in ("correctness", "quality", "performance/resource"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                oracle = correctness()
                candidate = quality()
                performance = performance_resource()
                value = {"correctness": oracle, "quality": candidate, "performance/resource": performance}[kind]
                value["unknown"] = True
                with self.assertRaises(aggregate_phase53.ContractError):
                    aggregate_phase53.aggregate(
                        POLICY,
                        [self.write(root, "correctness.json", oracle)],
                        [self.write(root, "quality.json", candidate)],
                        [self.write(root, "performance.json", performance)],
                    )

    def test_inclusive_and_exclusive_threshold_boundaries(self) -> None:
        self.assertEqual(self.decide(quality()), "adopt")
        self.assertEqual(self.decide(quality(kld_p99=0.05)), "retain-fp16")
        self.assertEqual(self.decide(quality(perplexity_relative_delta=0.010000001)), "retain-fp16")
        self.assertEqual(self.decide(quality(top1_agreement=0.989999999)), "retain-fp16")
        self.assertEqual(self.decide(quality(task_score_delta=0.020000001)), "retain-fp16")
        self.assertEqual(self.decide(quality(long_context_score_delta=0.020000001)), "retain-fp16")

    def test_correctness_fallback_and_cleanup_contract_fail_closed(self) -> None:
        fallback = correctness()
        fallback["execution"]["fallback_used"] = True
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(quality(), fallback)
        cleanup = correctness()
        cleanup["cleanup"]["terminal_zero"] = False
        with self.assertRaises(aggregate_phase53.ContractError):
            self.decide(quality(), cleanup)

    def test_performance_resource_missing_is_insufficient_and_failure_retains(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, _ = aggregate_phase53.aggregate(POLICY, [self.write(root, "c.json", correctness())], [self.write(root, "q.json", quality())])
        self.assertEqual(summary["targets"][2]["decision"], "insufficient-evidence")
        failed = performance_resource()
        failed["state"] = "FAIL"
        failed["decisions"]["performance"] = "fail"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, _ = aggregate_phase53.aggregate(POLICY, [self.write(root, "c.json", correctness())], [self.write(root, "q.json", quality())], [self.write(root, "p.json", failed)])
        self.assertEqual(summary["targets"][2]["decision"], "retain-fp16")

    def test_performance_rows_and_protocol_are_frozen(self) -> None:
        swapped = performance_resource()
        swapped["rows"][0], swapped["rows"][1] = swapped["rows"][1], swapped["rows"][0]
        with self.assertRaises(aggregate_phase53.ContractError):
            aggregate_phase53.performance_resource_result(swapped)
        modified = performance_resource()
        modified["rows"][5]["protocol"]["warmups"] = 3
        with self.assertRaises(aggregate_phase53.ContractError):
            aggregate_phase53.performance_resource_result(modified)

    def test_explicit_failure_does_not_require_performance_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed_correctness = correctness()
            failed_correctness["state"] = "FAIL"
            summary, _ = aggregate_phase53.aggregate(
                POLICY,
                [self.write(root, "c.json", failed_correctness)],
                [self.write(root, "q.json", quality())],
            )
        self.assertEqual(summary["targets"][2]["decision"], "retain-fp16")
        self.assertEqual(summary["targets"][2]["performance"], "insufficient-evidence")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed_quality = quality()
            failed_quality["state"] = "FAIL"
            summary, _ = aggregate_phase53.aggregate(
                POLICY,
                [self.write(root, "c.json", correctness())],
                [self.write(root, "q.json", failed_quality)],
            )
        self.assertEqual(summary["targets"][2]["decision"], "retain-fp16")

    def test_gfx942_unsupported_mxfp8_does_not_block_block16_decision(self) -> None:
        target = "gfx942:sramecc+:xnack-"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, _ = aggregate_phase53.aggregate(
                POLICY,
                [self.write(root, "c.json", correctness(target))],
                [self.write(root, "q.json", quality(target))],
                [self.write(root, "p.json", performance_resource(target))],
            )
        self.assertEqual(summary["targets"][0]["decision"], "adopt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
