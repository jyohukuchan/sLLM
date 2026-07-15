from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/src/ullm_runtime_parts/part_00.inc"


class UllmRuntimeDirectionalCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_runtime_resolves_only_directional_async_copy_symbols(self) -> None:
        self.assertNotIn("hipMemcpyAsync", self.source)
        for symbol in (
            "hipMemcpyHtoDAsync",
            "hipMemcpyDtoHAsync",
            "hipMemcpyDtoDAsync",
        ):
            self.assertEqual(self.source.count(f'dlsym(handle_, "{symbol}")'), 1)

    def test_copy_dispatch_maps_exact_kinds_and_rejects_unknown_kind(self) -> None:
        start = self.source.index("    bool copy_async(")
        end = self.source.index("\n    bool memset_async(", start)
        implementation = self.source[start:end]

        self.assertIn("if (kind < 1 || kind > 3)", implementation)
        self.assertLess(
            implementation.index("if (kind < 1 || kind > 3)"),
            implementation.index("if (bytes == 0)"),
        )
        for kind, member in (
            (1, "hip_memcpy_htod_async_"),
            (2, "hip_memcpy_dtoh_async_"),
            (3, "hip_memcpy_dtod_async_"),
        ):
            case_start = implementation.index(f"case {kind}:")
            case_end = implementation.find("case ", case_start + 1)
            if case_end < 0:
                case_end = implementation.index("default:", case_start)
            self.assertIn(member, implementation[case_start:case_end])
        self.assertIn("default:\n            return false;", implementation)

    def test_rocm_header_signatures_are_checked_at_compile_time(self) -> None:
        for symbol in (
            "hipMemcpyHtoDAsync",
            "hipMemcpyDtoHAsync",
            "hipMemcpyDtoDAsync",
        ):
            self.assertIn(f"decltype(&{symbol})", self.source)


if __name__ == "__main__":
    unittest.main()
