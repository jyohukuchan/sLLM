from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/capture-aq4-resident-executor-record.py"
SPEC = importlib.util.spec_from_file_location("capture_aq4_sq8_promotion", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def telemetry(**projection_overrides: int) -> dict:
    projection = {
        "single_matvec_count": 0,
        "batch_matvec_count": 24,
        "pair_matvec_count": 48,
        "triple_matvec_count": 0,
        "fallback_count": 0,
    }
    projection.update(projection_overrides)
    return {
        "schema_version": "ullm.qwen35_aq4.sq8_promotion_telemetry.v1",
        "projection": projection,
        "diagnostic_host_staging": {
            "read_count": 0,
            "write_count": 0,
            "read_bytes": 0,
            "write_bytes": 0,
        },
    }


def test_valid_sq8_promotion_telemetry_requires_observed_batch_and_pair() -> None:
    value = telemetry()
    assert TOOL.validate_sq8_promotion_telemetry(value) is value
    for key in ("batch_matvec_count", "pair_matvec_count"):
        with pytest.raises(TOOL.CaptureError, match="batch and pair"):
            TOOL.validate_sq8_promotion_telemetry(telemetry(**{key: 0}))


@pytest.mark.parametrize(
    "key",
    ("single_matvec_count", "triple_matvec_count", "fallback_count"),
)
def test_sq8_promotion_telemetry_rejects_unexpected_projection_paths(key: str) -> None:
    with pytest.raises(TOOL.CaptureError, match=key):
        TOOL.validate_sq8_promotion_telemetry(telemetry(**{key: 1}))


def test_sq8_promotion_telemetry_rejects_host_staging_and_shape_extensions() -> None:
    staged = telemetry()
    staged["diagnostic_host_staging"]["write_count"] = 1
    with pytest.raises(TOOL.CaptureError, match="host staging"):
        TOOL.validate_sq8_promotion_telemetry(staged)

    extended = telemetry()
    extended["projection"]["unknown"] = 0
    with pytest.raises(TOOL.CaptureError, match="shape differs"):
        TOOL.validate_sq8_promotion_telemetry(extended)


def test_output_token_identity_is_domain_separated_and_order_sensitive() -> None:
    expected = hashlib.sha256(b"ullm.sq8-promotion-output-token-ids.v1\0")
    expected.update((7).to_bytes(8, "little"))
    expected.update((9).to_bytes(8, "little"))
    assert TOOL.token_identity_digest([7, 9]) == expected.hexdigest()
    assert TOOL.token_identity_digest([7, 9]) != TOOL.token_identity_digest([9, 7])
    with pytest.raises(TOOL.CaptureError, match="invalid token id"):
        TOOL.token_identity_digest([-1])


def test_sq8_promotion_marker_is_not_a_caller_control_surface() -> None:
    caller = {
        TOOL.SQ8_PROMOTION_REQUEST_ENV: "caller-controlled",
        "HIP_VISIBLE_DEVICES": "0,1,2",
        "ULLM_HIP_VISIBLE_DEVICES": "0",
        "ROCR_VISIBLE_DEVICES": "2",
    }
    ordinary = TOOL.configure_sq8_promotion_environment(
        caller, enabled=False, request_id="internal"
    )
    assert TOOL.SQ8_PROMOTION_REQUEST_ENV not in ordinary

    promotion = TOOL.configure_sq8_promotion_environment(
        caller, enabled=True, request_id="internal"
    )
    assert promotion[TOOL.SQ8_PROMOTION_REQUEST_ENV] == "internal"
    assert promotion["HIP_VISIBLE_DEVICES"] == "1"
    assert promotion["ULLM_HIP_VISIBLE_DEVICES"] == "1"
    assert "ROCR_VISIBLE_DEVICES" not in promotion
