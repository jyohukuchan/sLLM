#!/usr/bin/env python3
"""Host-only contract checks for the Phase 36 Session D llama wrapper.

These checks deliberately do not load a model or require a GPU.  They protect
the fixed comparison identity and compile the public-API consumer when the
local b10453 reference headers are available.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ci/tools/llama_phase36_session_d_wrapper.cpp"
REFERENCE_INCLUDE = ROOT / "reference/llama.cpp/include"
GGML_INCLUDE = ROOT / "reference/llama.cpp/ggml/include"


class LlamaPhase36SessionDWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_source_has_current_provenance_and_exact_gpu_identity(self) -> None:
        self.assertIn("3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70", self.source)
        self.assertIn('constexpr const char *kLlamaTag = "b10453"', self.source)
        self.assertIn('constexpr const char *kTarget = "gfx942"', self.source)
        self.assertIn('constexpr const char *kGpuUuid = "GPU-1228c84fe776f2f4"', self.source)
        self.assertNotIn("f5919bf458ef190468b5c329bb293f8a54a1e69c", self.source)
        self.assertNotIn("GPU-cb0412d4d88cfa69", self.source)

    def test_closed_case_recipe_and_long_input_guard_are_present(self) -> None:
        expected = (
            '{"short-odd", 17, 17}',
            '{"32x32", 32, 32}',
            '{"prefill-long", 1024, 128}',
            '{"decode-long", 32, 256}',
            '{"long-10001", kLongInput, 2}',
        )
        for item in expected:
            self.assertIn(item, self.source)
        self.assertIn("kLongToken = 23066", self.source)
        self.assertIn("requires token ID 23066 repeated 10001 times", self.source)

    def test_fixed_protocol_includes_long_batch_and_fp16_kv(self) -> None:
        self.assertIn("constexpr int32_t kNBatch = 10001", self.source)
        self.assertIn("constexpr int32_t kNUbatch = 512", self.source)
        self.assertIn("options.n_batch != kNBatch", self.source)
        self.assertIn("context_params.type_k = GGML_TYPE_F16", self.source)
        self.assertIn("context_params.type_v = GGML_TYPE_F16", self.source)
        self.assertIn("model_params.n_gpu_layers = -1", self.source)
        self.assertIn("model_params.split_mode = LLAMA_SPLIT_MODE_NONE", self.source)
        self.assertIn("model_params.load_mtp = false", self.source)
        self.assertIn("context_params.n_ctx = static_cast<uint32_t>(options.input.size() + options.max_new_tokens)", self.source)

    def test_json_contract_retains_inputs_events_derived_offload_and_cleanup(self) -> None:
        for field in (
            '\\"input_token_ids\\"',
            '\\"events\\"',
            '\\"derived\\"',
            '\\"ttft_ns\\"',
            '\\"prefill_ns\\"',
            '\\"tpot_ns\\"',
            '\\"decode_ns\\"',
            '\\"e2e_ns\\"',
            '\\"offload_evidence\\"',
            '\\"cleanup\\"',
            '\\"backend_release_completed\\"',
        ):
            self.assertIn(field, self.source)
        self.assertIn("sample.generated.push_back(token)", self.source)
        self.assertIn("sample.visible.push_back(token)", self.source)
        self.assertIn("stop_tokens_not_fed_back", self.source)

    def test_public_api_consumer_compiles_against_b10453_headers(self) -> None:
        if shutil.which("g++") is None:
            self.skipTest("g++ is unavailable")
        if not (REFERENCE_INCLUDE / "llama.h").exists() or not GGML_INCLUDE.exists():
            self.skipTest("reference/llama.cpp headers are not checked out")
        completed = subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-fsyntax-only",
                f"-I{REFERENCE_INCLUDE}",
                f"-I{GGML_INCLUDE}",
                str(SOURCE),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_input_parser_contract_rejects_non_exact_long_row(self) -> None:
        # This mirrors the wrapper's fail-closed long-row rule without loading
        # llama.cpp: malformed length, empty values, and a wrong token are all
        # rejected before any model or GPU side effect.
        def parse(value: str) -> list[int]:
            if not value:
                raise ValueError("empty")
            result = [int(item) for item in value.split(",")]
            if len(result) != 10001 or any(item != 23066 for item in result):
                raise ValueError("long-10001 recipe mismatch")
            return result

        valid = parse(",".join(["23066"] * 10001))
        self.assertEqual(len(valid), 10001)
        for value in ("23066", ",".join(["23066"] * 10000 + ["23067"]), "23066,"):
            with self.assertRaises((ValueError, TypeError)):
                parse(value)


if __name__ == "__main__":
    unittest.main()
