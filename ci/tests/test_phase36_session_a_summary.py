from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "ci" / "matrix" / "phase36-mi300x-session-a-final-v1.json"
SCHEMA_PATH = ROOT / "ci" / "schema" / "phase36-mi300x-session-a-final-v1.schema.json"


class Phase36SessionASummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def assert_schema_rejects(self, document: dict) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        self.assertTrue(list(Draft202012Validator(self.schema).iter_errors(document)))

    def test_final_summary_passes_strict_jsonschema(self) -> None:
        if Draft202012Validator is None:
            self.skipTest("jsonschema is not installed")
        self.assertEqual(list(Draft202012Validator(self.schema).iter_errors(self.summary)), [])

    def test_schema_rejects_unknown_root_field(self) -> None:
        invalid = copy.deepcopy(self.summary)
        invalid["unexpected"] = True
        self.assert_schema_rejects(invalid)

    def test_initial_digest_is_historical_and_final_identity_is_authoritative(self) -> None:
        identity = self.summary["identity"]
        self.assertEqual(identity["initial_semantic_digest"], "fa2c82c936f61c897c87cee82cb92b0aa100cb0b0c766734e42d24c8df2bc892")
        self.assertEqual(identity["initial_semantic_digest_scope"], "historical initial-99-operator-run")
        self.assertEqual(identity["final_semantic_digest"], "8836bd3ec07464904a657b846c79e21f3ffb8c975b55a65dca7ea720e993a57a")
        self.assertEqual(identity["operator_summary_sha256"], "5daa5869932513490c50cbb9ff330cf47fb581aa333fc1133fc0261a1192222d")

    def test_stage_scopes_are_bound_to_the_correct_stage(self) -> None:
        expected = {
            "A0": "VM, hardware, and target identity",
            "A1": "ROCm root and exact artifact build/load",
            "A2": "tiny runtime, library probe, and profiler",
            "A3": "99-case native operator matrix",
            "A4": "BF16/FP8 model verify, load, reuse, and generation smoke",
            "A5": "cleanup, provider restoration, and retention",
        }
        self.assertEqual({key: value["scope"] for key, value in self.summary["stages"].items()}, expected)
        invalid = copy.deepcopy(self.summary)
        invalid["stages"]["A0"]["scope"], invalid["stages"]["A1"]["scope"] = invalid["stages"]["A1"]["scope"], invalid["stages"]["A0"]["scope"]
        self.assert_schema_rejects(invalid)

    def test_exact_target_artifacts_have_full_code_object_contract(self) -> None:
        contract = self.summary["artifact_contract"]
        self.assertEqual(contract["device_bundle"], "hipv4-amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-")
        self.assertEqual(contract["host_bundle"], "host-x86_64-unknown-linux-gnu-")
        self.assertEqual((contract["abi_version"], contract["code_object_version"], contract["e_flags"], contract["wavefront_size"]), (4, 6, "0xE4C", 64))
        self.assertEqual(contract["other_device_bundles"], [])
        self.assertEqual(contract["generic_targets"], [])
        self.assertNotEqual(contract["device_bundle"], "gfx942")
        self.assert_schema_rejects({**copy.deepcopy(self.summary), "artifact_contract": {**contract, "device_bundle": "gfx942"}})
        self.assertEqual(len(self.summary["artifacts"]), 11)
        self.assertTrue(all(len(item["sha256"]) == 64 and item["size_bytes"] > 0 for item in self.summary["artifacts"].values()))

    def test_wrong_target_fails_before_dispatch_with_exact_error(self) -> None:
        negative = self.summary["wrong_target_negative"]
        self.assertEqual(negative["requested_target"], "gfx1201")
        self.assertEqual(negative["exit_code"], 1)
        self.assertFalse(negative["dispatch_started"])
        self.assertEqual(negative["stderr"], "matmul-g1 evidence failed: owned HIP execution-session open failed: backend status 259: requested device gcnArchName does not match exactly")

    def test_a2_tiny_library_and_profiler_evidence_is_explicit(self) -> None:
        a2 = self.summary["a2"]
        self.assertEqual(a2["preflight"]["tiny_runtime"], {"input": 41, "output": 42, "allocation_count": 1, "copy_count": 2, "dispatch_count": 1})
        self.assertEqual((a2["fnuz_probe"]["solution_count"], a2["fnuz_probe"]["workspace_bytes"]), (8, 0))
        self.assertTrue(a2["rocprof"]["increment_captured"])
        self.assertEqual(a2["host_fp8_conversion_oracle"]["evidence"], "relevant core tests")

    def test_operator_dispatch_is_not_faked_from_case_count(self) -> None:
        operator = self.summary["operator"]
        self.assertEqual(operator["family_case_counts"], [2, 17, 21, 8, 19, 16, 6, 7, 3])
        self.assertEqual(operator["dispatch_counts"], [4, 17, 21, 8, 19, 16, 6, 7, 6])
        self.assertNotEqual(operator["dispatch_counts"], operator["family_case_counts"])
        self.assertEqual(sum(operator["family_case_counts"]), operator["expected_cases"])
        self.assertTrue(operator["hip_only"])
        self.assertFalse(operator["fallback_used"])
        invalid = copy.deepcopy(self.summary)
        invalid["operator"]["dispatch_counts"] = invalid["operator"]["family_case_counts"]
        self.assert_schema_rejects(invalid)

    def test_a4_verification_memory_load_reuse_and_after_drop_are_required(self) -> None:
        expected = {
            "bf16": ("5e6d31c89da0c6eb6c5dbc187740dc87e44194e01d4549d95a2f68586490fb28", 10826068306, 8411592192, 8477011968, "9dc379e3f89a29db6f040b6c17bda4c919ee0799f72e86574bf0191cab110f59"),
            "fp8": ("8628cf4100f54254939fa483eb8f53036b2cce4ecbf46f20f3b4e4d86fcc156f", 16937844619, 4847029760, 4912449536, "f80fa8ed3a68448a3eb1a59c07290fbec6df59627e28e1596a4642233c71c0db"),
        }
        for name, (plan, load_ns, resident, peak, digest) in expected.items():
            model = self.summary["models"][name]
            self.assertEqual((model["verify"]["plan_digest"], model["verify"]["loadable_entries"], model["verify"]["weight_entries"]), (plan, 426, 738))
            benchmark = model["benchmark"]
            self.assertEqual((benchmark["sha256"], benchmark["warmups"], benchmark["measured"], benchmark["total_requests"]), (digest, 3, 10, 14))
            self.assertEqual((benchmark["model_load_count"], benchmark["model_load_ns"], benchmark["resident_bytes"], benchmark["peak_bytes"], benchmark["after_model_drop_bytes"]), (1, load_ns, resident, peak, 0))
            self.assertTrue(benchmark["model_reused"])
            self.assertEqual((benchmark["cleanup_retryable"], benchmark["cleanup_durable"]), (0, 0))
        invalid = copy.deepcopy(self.summary)
        del invalid["models"]["bf16"]["benchmark"]["after_model_drop_bytes"]
        self.assert_schema_rejects(invalid)

    def test_a5_cleanup_store_retention_and_exclusions_are_explicit(self) -> None:
        cleanup = self.summary["cleanup"]
        self.assertEqual((cleanup["process_count"], cleanup["model_handles"], cleanup["provider_rocm_smi_gpu_processes"], cleanup["provider_rocm_smi_vram_percent"], cleanup["provider_amd_smi_used_vram_mb"], cleanup["provider_amd_smi_gfx_activity_percent"]), (0, 0, 0, 0, 285, 0))
        self.assertEqual(cleanup["ecc"], {"scope": "all sysfs blocks", "correctable_total": 0, "uncorrectable_total": 0})
        self.assertFalse(cleanup["execution_amd_smi_metric_available"])
        self.assertTrue(cleanup["post_provider_metric_available"])
        self.assertTrue(cleanup["provider_restored"])
        self.assertEqual(self.summary["artifact_store"]["identifier"], "/home/homelab1/.local/share/sllm-evidence/phase36/session-a/enc1-gpuvm015-2026-08-21/final")
        self.assertEqual(self.summary["retention"], {"owner": "hotaisle account / local operator", "purpose": "Session B", "review_or_expiry_at": "2026-08-28T23:59:59+09:00", "unless_extended": True, "automatic_deletion_claim": False})
        self.assertEqual(self.summary["unexecuted_scope"]["sessions"], ["Session B", "Session C", "Session D", "Session E"])
        for exclusion in ("low-bit KV full matrix", "chunked prefill", "RDMA-RoCE", "Responses API", "Gemma", "MoE"):
            self.assertIn(exclusion, self.summary["unexecuted_scope"]["session_a_exclusions"])


if __name__ == "__main__":
    unittest.main()
