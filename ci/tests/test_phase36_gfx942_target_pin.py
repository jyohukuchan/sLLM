"""Host contracts for the Phase 36 MI300X codegen-target pin."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD_RS = ROOT / "crates/sllm-hip-sys/build.rs"
CMAKE = ROOT / "native/hip/CMakeLists.txt"


class Phase36Gfx942TargetPinTests(unittest.TestCase):
    def test_build_script_maps_only_logical_gfx942_to_the_feature_tuple(self) -> None:
        source = BUILD_RS.read_text(encoding="utf-8")
        self.assertIn('println!("cargo:rerun-if-env-changed=SLLM_HIP_TARGET");', source)
        self.assertIn('"gfx942:sramecc+:xnack-".to_owned()', source)
        self.assertIn('"-DSLLM_HIP_TARGET={}", configuration.target', source)
        self.assertRegex(
            source,
            r'"-DSLLM_HIP_COMPILE_TARGET=\{\}"\s*,\s*\n\s*configuration\.target',
        )
        self.assertIn('"-DCMAKE_HIP_ARCHITECTURES={}",\n                configuration.codegen_target', source)
        self.assertIn(
            "// Preserve the required logical entry point for CMake.",
            source,
        )
        self.assertRegex(
            source,
            r"HipConfiguration\s*\{\s*// Preserve the required logical entry point for CMake\.[\s\S]*?rocm_path,",
        )
        self.assertNotIn("rocm_path: canonical_rocm", source)

        # A feature-suffixed target is not a logical target input. The source
        # contract must retain the explicit allowlist and must not normalize an
        # arbitrary suffix by splitting at ':'.
        target_allowlist = re.search(
            r'matches!\(target\.as_str\(\),\s*"gfx1030"\s*\|\s*"gfx1201"\s*\|\s*"gfx942"\)',
            source,
        )
        self.assertIsNotNone(target_allowlist)
        self.assertNotIn("split(':')", source)

    def test_cmake_separates_logical_target_from_codegen_architecture(self) -> None:
        source = CMAKE.read_text(encoding="utf-8")
        self.assertIn('set(SLLM_EXPECTED_HIP_ARCHITECTURE "gfx942:sramecc+:xnack-")', source)
        self.assertIn(
            'NOT CMAKE_HIP_ARCHITECTURES STREQUAL SLLM_EXPECTED_HIP_ARCHITECTURE',
            source,
        )
        self.assertIn(
            'NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx942:sramecc+:xnack-"',
            source,
        )
        self.assertIn(
            'SLLM_TEST_EXPECTED_TARGET="${SLLM_HIP_COMPILE_TARGET}"',
            source,
        )
        self.assertIn(
            'OUTPUT_NAME "hip-compile-probe-${SLLM_HIP_COMPILE_TARGET}"',
            source,
        )

        # The exact suffix is the sole CDNA3 codegen spelling. Other suffixes
        # must not appear in the allowlist or be accepted by a broad colon
        # check.
        self.assertNotIn('MATCHES "^gfx942:', source)
        self.assertNotIn('MATCHES "^gfx9', source)
        self.assertIn(
            'CMAKE_HIP_ARCHITECTURES MATCHES ":"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
