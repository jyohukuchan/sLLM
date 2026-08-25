import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sllm = _load("run_phase51_mi300x_sllm", "ci/tools/run_phase51_mi300x_sllm.py")
llama = _load("run_phase51_mi300x_llama", "ci/tools/run_phase51_mi300x_llama.py")
aggregate = _load("aggregate_phase51_mi300x", "ci/tools/aggregate_phase51_mi300x.py")


class Phase51MI300XRunnerIdentityTest(unittest.TestCase):
    def test_exact_mi300x_tuple_and_toolchain(self):
        expected = (
            "gfx942",
            "gfx942:sramecc+:xnack-",
            64,
            "GPU-6104e2a75685060a",
            "0000:ff:00.0",
            "/opt/rocm",
            "7.14.0",
        )
        for module in (sllm, llama, aggregate):
            self.assertEqual(
                (
                    module.TARGET,
                    module.ACTUAL_ARCH,
                    module.WAVEFRONT_SIZE,
                    module.GPU_RUNTIME_UUID,
                    module.GPU_BDF,
                    module.ROCM_ROOT,
                    module.ROCM_VERSION,
                ),
                expected,
            )
            self.assertEqual(module.GPU_AMD_SMI_UUID, "61ff74b5-0000-1000-8004-e2a75685060a")
            self.assertEqual(module.ROCM_SOURCE_ROOT, "/opt/rocm-7.2.4/core-7.14")

    def test_matrix_has_fixed_seven_rows_and_repetition_protocol(self):
        sllm_rows = sllm.matrix()
        llama_rows = llama.matrix()
        self.assertEqual([row["case_id"] for row in sllm_rows], [row["case_id"] for row in llama_rows])
        self.assertEqual(len(sllm_rows), 7)
        self.assertEqual(len(llama_rows), 7)
        self.assertEqual(
            {row["case_id"] for row in sllm_rows[:5]},
            {"short-odd", "32-32", "prefill-long", "decode-long", "long-10001"},
        )
        self.assertEqual({row["case_id"] for row in sllm_rows[5:]}, {"long-100000", "decode-20000"})
        self.assertTrue(all(row["warmups"] == 3 and row["measured"] == 10 for row in sllm_rows[:5]))
        self.assertTrue(all(row["warmups"] == 1 and row["measured"] == 3 for row in sllm_rows[5:]))
        self.assertTrue(all(row["row_id"].startswith("phase51-mi300x-sllm-") for row in sllm_rows))
        self.assertTrue(all(row["row_id"].startswith("phase51-mi300x-llama-") for row in llama_rows))
        self.assertTrue(all(row["target"] == "gfx942" for row in sllm_rows + llama_rows))

    def test_summary_identity_fields_are_frozen(self):
        expected = {
            "actual_arch": "gfx942:sramecc+:xnack-",
            "wavefront_size": 64,
            "rocm_root": "/opt/rocm",
            "rocm_source_root": "/opt/rocm-7.2.4/core-7.14",
            "rocm_version": "7.14.0",
        }
        for module in (sllm, llama):
            # The producer writes these exact fields at the publication
            # boundary; inspect source to keep this unit test host-only.
            source = (ROOT / ("ci/tools/run_phase51_mi300x_sllm.py" if module is sllm else "ci/tools/run_phase51_mi300x_llama.py")).read_text(encoding="utf-8")
            for key, value in expected.items():
                self.assertIn(key, source)
                self.assertIn(str(value), source)

    def test_visibility_environment_is_uuid_bound_and_rocm_pinned(self):
        env = sllm._execution_environment("phase51-mi300x-sllm-short-odd", 0, base={"PATH": "/bin"})
        self.assertEqual(env["ROCR_VISIBLE_DEVICES"], sllm.GPU_UUID)
        self.assertEqual(env["LD_LIBRARY_PATH"], "/opt/rocm/lib")
        self.assertEqual(env["SLLM_PHASE51_MI300X_ROW"], "phase51-mi300x-sllm-short-odd")
        self.assertNotIn("HIP_VISIBLE_DEVICES", env)

    def test_aggregate_rejects_wrong_identity_and_summary_stats_are_finite(self):
        self.assertEqual(aggregate.summary_stats([1.0, 2.0, 3.0], "metric")["median"], 2.0)
        with self.assertRaises(aggregate.Phase51Error):
            aggregate._validate_summary_header(
                {"schema_version": aggregate.SLLM_SCHEMA, "state": "PASS", "target": "gfx1030", "gpu_uuid": aggregate.GPU_UUID, "gpu_bdf": aggregate.GPU_BDF},
                "sllm",
            )

    def test_atomic_publication_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "row.json"
            path.write_text("existing", encoding="utf-8")
            with self.assertRaises(sllm.SessionDError):
                sllm._atomic_write(path, b"new", "row report")


if __name__ == "__main__":
    unittest.main()
