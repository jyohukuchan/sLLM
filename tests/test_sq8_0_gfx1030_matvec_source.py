"""Source-level non-interference gates for the SQ8_0 gfx1030 matvec specialization."""

from __future__ import annotations

import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc"

# This is the exact legacy single-matvec body before the gfx1030-only branch.
# Keeping it byte-stable guarantees that gfx1201 compiles the previous generic
# symbol body; static offline code-object comparison is recorded separately.
GFX1201_LEGACY_BODY_SHA256 = "7b10c5d38ba6cc79ce346d81f9a2382bbdb8cfe5adac2b5504c6c799ff66368a"
GFX1201_LEGACY_BATCH_BODY_SHA256 = "4aee9d87c84d6f744469c46fb896240da057a189b49b1c2227aba5c57d16ccdb"


def test_gfx1030_specialization_is_preprocessor_isolated() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    guard = "#if defined(__gfx1030__)"
    legacy_marker = '#else\n\nextern "C" __global__ void ullm_sq_fp8_matvec_f32_kernel('
    batch_legacy_marker = '#else\n\nextern "C" __global__ void ullm_sq_fp8_matvec_batch_f32_kernel('
    assert source.count(guard) == 2
    assert source.count(legacy_marker) == 1
    assert source.count(batch_legacy_marker) == 1
    assert source.index(guard) < source.index(legacy_marker)
    assert source.rindex(guard) < source.index(batch_legacy_marker)
    assert source.index("#endif  // defined(__gfx1030__)") < source.index(
        'extern "C" __global__ void ullm_sq_fp8_matvec_batch_f32_kernel('
    )


def test_gfx1201_legacy_single_matvec_body_is_unchanged() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    marker = '#else\n\nextern "C" __global__ void ullm_sq_fp8_matvec_f32_kernel('
    start = source.index(marker) + len("#else\n\n")
    end = source.index("\n\n#endif  // defined(__gfx1030__)", start)
    body = source[start:end]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == GFX1201_LEGACY_BODY_SHA256


def test_gfx1201_legacy_batch_matvec_body_is_unchanged() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    marker = '#else\n\nextern "C" __global__ void ullm_sq_fp8_matvec_batch_f32_kernel('
    start = source.index(marker) + len("#else\n\n")
    end = source.index("\n\n#endif  // defined(__gfx1030__)", start)
    body = source[start:end]
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == GFX1201_LEGACY_BATCH_BODY_SHA256


def test_gfx1030_candidate_retains_the_legacy_symbol_and_scale_fallback() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    candidate = source[source.index("#if defined(__gfx1030__)") : source.index("#else\n\nextern", source.index("#if defined(__gfx1030__)"))]
    assert 'extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_f32_kernel(' in candidate
    assert "ullm_sq_fp8_wave32_sum" in candidate
    assert "ullm_sq_fp8_uint4" in candidate
    assert "__shared__ float wave_partial[8]" in candidate
    assert "segment_scale && wide_aligned" in candidate
    assert "for (unsigned int index = 0u; index < count; ++index)" in candidate
    assert 'extern "C" __global__ __launch_bounds__(256) void ullm_sq_fp8_matvec_batch_f32_kernel(' in source
