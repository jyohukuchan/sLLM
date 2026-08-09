from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import aggregate_rmsnorm_h3_results as aggregate
import validate_rmsnorm_h3_contracts as validator


class RmsNormH3AggregateTests(unittest.TestCase):
    def _assert_symlinked_target_rejected(self, filename: str, content: bytes) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-rmsnorm-h3-symlink-target-") as directory:
            root = Path(directory)
            external = root / f"external-{filename}"
            external.write_bytes(content)
            target = root / filename
            target.symlink_to(external)
            sidecar = root / f"{filename}.sha256"
            sidecar.write_text(f"{validator.sha256_file(external)}  {filename}\n", encoding="ascii")

            with self.assertRaisesRegex(validator.ContractError, "symlink"):
                validator._sidecar(sidecar, target, f"{filename} sidecar")

    def test_missing_and_unknown_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-rmsnorm-h3-aggregate-test-") as directory:
            root = Path(directory)
            (root / "h3-rmsnorm-gfx1030").mkdir()
            with self.assertRaises(validator.ContractError):
                validator.validate_artifacts(ROOT, root, expected_sha=None, expected_tree=None, strict=False)
            (root / "unknown-row").mkdir()
            with self.assertRaises(validator.ContractError):
                validator.validate_artifacts(ROOT, root, expected_sha=None, expected_tree=None, strict=False)

    def test_sidecar_format_is_exact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sllm-rmsnorm-h3-sidecar-test-") as directory:
            path = Path(directory) / "value.json"
            path.write_text("{}\n", encoding="utf-8")
            sidecar = path.with_name(path.name + ".sha256")
            sidecar.write_text("not-a-digest value.json\n", encoding="ascii")
            with self.assertRaises(validator.ContractError):
                validator._sidecar(sidecar, path, "fixture sidecar")

    def test_symlinked_host_artifact_target_is_rejected(self) -> None:
        self._assert_symlinked_target_rejected("host-bundle-gfx1030.elf", b"host artifact\n")

    def test_symlinked_device_artifact_target_is_rejected(self) -> None:
        self._assert_symlinked_target_rejected("device-code-object-gfx1030.elf", b"device artifact\n")

    def test_symlinked_metadata_target_is_rejected(self) -> None:
        self._assert_symlinked_target_rejected("rmsnorm-h3-artifact.json", b"{}\n")

    def test_symlinked_report_target_is_rejected(self) -> None:
        self._assert_symlinked_target_rejected("rmsnorm-h3-report.json", b"{}\n")

    def test_aggregate_schema_requires_both_rows_and_compile_only_scope(self) -> None:
        schema = validator._schema(ROOT, "aggregate")
        document = {
            "schema_version": "rmsnorm-h3-aggregate-v1",
            "aggregate_id": "rmsnorm-h3-aggregate-test",
            "suite_id": "h3-rmsnorm-compile-only",
            "tier": "tier_h3_rmsnorm",
            "state": "PASS",
            "required": False,
            "evidence_mode": "local-nonstrict",
            "run_id": "test",
            "run_attempt": 1,
            "reviewed_sha": "0" * 40,
            "tested_sha": "0" * 40,
            "workflow_sha": "0" * 40,
            "git_tree_oid": "0" * 40,
            "matrix_id": "rmsnorm-h3-compile-v1",
            "matrix_manifest_sha256": "0" * 64,
            "workflow_file_sha256": "0" * 64,
            "expected_rows": ["h3-rmsnorm-gfx1030", "h3-rmsnorm-gfx1201"],
            "rows": [],
            "source_sets": {"device": {}, "host_abi": {}, "binding_build": {}, "ci_contract": {}},
            "source_symbol_map": [],
            "toolchain": {},
            "container": {},
            "codegen": {},
            "logical_kernel": "rmsnorm.baseline.wave32.v1",
            "device_symbol": "sllm_rmsnorm_baseline_wave32_v1",
            "case_manifest": {"id": "rmsnorm-h3-compile-link-extract-inspect-v1", "selected_count": 2, "collected_count": 2},
            "scope": {"compile_only": True, "execution_attempted": False, "gpu_execution": False, "model_used": False, "network_used": False, "fallback_allowed": False, "fallback_used": False, "cpu_fallback_used": False, "fake_hip": False, "emulation": False},
            "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": 2, "passed_rows": 0, "failed_rows": 0},
            "errors": [],
            "timestamps": {"started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:00Z"},
        }
        errors = list(validator.Draft202012Validator(schema).iter_errors(document))
        self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
