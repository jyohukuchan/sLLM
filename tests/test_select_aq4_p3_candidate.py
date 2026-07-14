from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_aq4_p3_candidate",
    ROOT / "tools/select-aq4-p3-candidate.py",
)
assert SPEC and SPEC.loader
SELECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SELECTOR
try:
    SPEC.loader.exec_module(SELECTOR)
finally:
    sys.modules.pop(SPEC.name, None)


IDENTITY_SHA = "1" * 64


def seal(value: dict[str, object]) -> dict[str, object]:
    value["evidence_sha256"] = SELECTOR.semantic_sha256(value)
    return value


def raw_fixture(candidate_id: str = "paged-kv-table-validation-v1") -> dict[str, object]:
    family = SELECTOR.CANDIDATES[candidate_id]["family"]
    measurements = []
    resolved_ms = [128, 64, 128, 32, 128, 16, 8]
    for index, resolved_m in enumerate(resolved_ms):
        measurements.append(
            {
                "candidate_id": candidate_id,
                "family": family,
                "prompt_id": f"prompt-{index}",
                "case_sha256": f"{index + 2:x}" * 64,
                "identity_sha256": IDENTITY_SHA,
                "resolved_m": resolved_m,
                "baseline_p50_ms": 100.0,
                "baseline_cv": 0.01,
                "ci95_halfwidth_ms": 1.0,
                "recoverable_family_exclusive_ms": 10.0,
                "d2h_count": 2 if candidate_id == "paged-kv-table-validation-v1" else None,
                "stream_sync_count": 2
                if candidate_id == "paged-kv-table-validation-v1"
                else None,
            }
        )
    pairs = [
        {
            "candidate_id": candidate_id,
            "pair_id": f"pair-{index}",
            "case_sha256": f"{index + 10:x}" * 64,
            "identity_sha256": IDENTITY_SHA,
            "baseline_ms": 100.0 + index,
            "candidate_ms": 90.0 + index,
        }
        for index in range(5)
    ]
    return seal(
        {
            "schema_version": SELECTOR.RAW_SCHEMA,
            "status": "complete",
            "measurement_eligible": True,
            "smoke_only": False,
            "promotion_eligible": True,
            "evidence_sha256": None,
            "identity": {
                "identity_sha256": IDENTITY_SHA,
                "case_manifest_sha256": "a" * 64,
                "binary_sha256": "b" * 64,
                "package_content_sha256": "c" * 64,
            },
            "capabilities": {
                "family_exclusive_timing": True,
                "d2h_count": True,
                "stream_sync_count": True,
            },
            "representative_prompt_count": 7,
            "measurements": measurements,
            "full_model_pairs": pairs,
        }
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def select_raw(tmp_path: Path, value: dict[str, object]) -> dict[str, object]:
    path = tmp_path / "raw.json"
    write_json(path, value)
    snapshot = SELECTOR.capture(path)
    return SELECTOR.select([(snapshot, SELECTOR.parse_json(snapshot))])


def candidate(result: dict[str, object], candidate_id: str) -> dict[str, object]:
    return next(
        item for item in result["candidates"] if item["candidate_id"] == candidate_id
    )


def test_eligible_fixture_selects_paged_kv_and_recomputes_all_gates(
    tmp_path: Path,
) -> None:
    result = select_raw(tmp_path, raw_fixture())
    item = candidate(result, "paged-kv-table-validation-v1")
    assert result["status"] == "selected"
    assert result["selected_candidate_id"] == "paged-kv-table-validation-v1"
    assert item["eligible"] is True
    assert item["reason_codes"] == []
    assert item["recoverable_share_e"] == 0.1
    assert item["noise_floor_n"] == 0.05
    assert item["representative"]["above_noise_count"] == 7
    assert item["representative"]["m128_above_noise"] is True
    assert item["representative"]["non_m128_above_noise"] is True
    assert item["paired_full_model_95ci"]["ci95_lower_ms"] == 10.0


def test_noise_floor_uses_each_maximum_term(tmp_path: Path) -> None:
    value = raw_fixture()
    rows = value["measurements"]
    rows[0]["baseline_cv"] = 0.01
    rows[0]["ci95_halfwidth_ms"] = 1.0
    rows[1]["baseline_cv"] = 0.03
    rows[1]["ci95_halfwidth_ms"] = 1.0
    rows[2]["baseline_cv"] = 0.01
    rows[2]["ci95_halfwidth_ms"] = 6.0
    seal(value)
    item = candidate(select_raw(tmp_path, value), "paged-kv-table-validation-v1")
    prompts = {row["prompt_id"]: row for row in item["representative"]["prompts"]}
    assert prompts["prompt-0"]["noise_floor_n"] == 0.05
    assert prompts["prompt-1"]["noise_floor_n"] == 0.09
    assert prompts["prompt-2"]["noise_floor_n"] == 0.12


def test_e_equal_to_n_is_not_above_noise(tmp_path: Path) -> None:
    value = raw_fixture()
    for row in value["measurements"]:
        row["recoverable_family_exclusive_ms"] = 5.0
    seal(value)
    item = candidate(select_raw(tmp_path, value), "paged-kv-table-validation-v1")
    assert item["eligible"] is False
    assert item["representative"]["above_noise_count"] == 0
    assert "aggregate_e_not_above_n" in item["reason_codes"]
    assert "representative_above_noise_lt_4" in item["reason_codes"]


def test_exactly_four_prompts_with_m128_and_other_m_are_sufficient(
    tmp_path: Path,
) -> None:
    value = raw_fixture()
    for row in value["measurements"][4:]:
        row["recoverable_family_exclusive_ms"] = 4.0
    seal(value)
    item = candidate(select_raw(tmp_path, value), "paged-kv-table-validation-v1")
    assert item["eligible"] is True
    assert item["representative"]["above_noise_count"] == 4


def test_four_prompts_without_non_m128_support_fail(tmp_path: Path) -> None:
    value = raw_fixture()
    for index, row in enumerate(value["measurements"]):
        row["recoverable_family_exclusive_ms"] = 10.0 if index < 4 else 4.0
        if index < 4:
            row["resolved_m"] = 128
    seal(value)
    item = candidate(select_raw(tmp_path, value), "paged-kv-table-validation-v1")
    assert item["eligible"] is False
    assert "non_m128_above_noise_missing" in item["reason_codes"]


def test_full_model_paired_ci_must_be_strictly_positive(tmp_path: Path) -> None:
    value = raw_fixture()
    for pair in value["full_model_pairs"]:
        pair["candidate_ms"] = pair["baseline_ms"]
    seal(value)
    item = candidate(select_raw(tmp_path, value), "paged-kv-table-validation-v1")
    assert item["paired_full_model_95ci"]["ci95_lower_ms"] == 0.0
    assert item["eligible"] is False
    assert "paired_full_model_ci95_not_positive" in item["reason_codes"]


def test_measurement_and_pair_order_do_not_change_selection(tmp_path: Path) -> None:
    first = raw_fixture()
    second = json.loads(json.dumps(first))
    second["measurements"] = list(reversed(second["measurements"]))
    second["full_model_pairs"] = list(reversed(second["full_model_pairs"]))
    seal(second)
    assert first["evidence_sha256"] == second["evidence_sha256"]

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_json(first_path, first)
    write_json(second_path, second)
    first_snapshot = SELECTOR.capture(first_path)
    second_snapshot = SELECTOR.capture(second_path)
    first_result = SELECTOR.select(
        [(first_snapshot, SELECTOR.parse_json(first_snapshot))]
    )
    second_result = SELECTOR.select(
        [(second_snapshot, SELECTOR.parse_json(second_snapshot))]
    )
    assert first_result == second_result


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("measurement_eligible", False, "measurement_eligible"),
        ("smoke_only", True, "smoke_only"),
        ("promotion_eligible", False, "promotion_eligible"),
    ],
)
def test_ineligible_source_flags_fail_closed(
    field: str, replacement: object, message: str
) -> None:
    value = raw_fixture()
    value[field] = replacement
    seal(value)
    with pytest.raises(SELECTOR.SelectionError, match=message):
        SELECTOR.validate_raw(value)


@pytest.mark.parametrize("mutation", ("missing", "unknown", "record_missing"))
def test_missing_and_unknown_fields_fail_closed(mutation: str) -> None:
    value = raw_fixture()
    if mutation == "missing":
        del value["capabilities"]
    elif mutation == "unknown":
        value["trusted"] = True
    else:
        del value["measurements"][0]["baseline_cv"]
    with pytest.raises(SELECTOR.SelectionError, match="fields differ"):
        SELECTOR.validate_raw(value)


def test_hash_swap_fails_closed_even_when_outer_hash_is_resealed() -> None:
    changed_without_reseal = raw_fixture()
    changed_without_reseal["measurements"][0]["baseline_p50_ms"] = 101.0
    with pytest.raises(SELECTOR.SelectionError, match="semantic SHA-256 differs"):
        SELECTOR.validate_raw(changed_without_reseal)

    identity_swap = raw_fixture()
    identity_swap["measurements"][0]["identity_sha256"] = "9" * 64
    seal(identity_swap)
    with pytest.raises(SELECTOR.SelectionError, match="differs from raw identity"):
        SELECTOR.validate_raw(identity_swap)


def test_nonfinite_json_is_rejected_before_selection(tmp_path: Path) -> None:
    path = tmp_path / "nonfinite.json"
    path.write_text('{"schema_version":"x","value":NaN}\n', encoding="utf-8")
    snapshot = SELECTOR.capture(path)
    with pytest.raises(SELECTOR.SelectionError, match="non-finite JSON number"):
        SELECTOR.parse_json(snapshot)


def test_current_diagnostic_profile_cannot_qualify_paged_kv(tmp_path: Path) -> None:
    profile = {
        "schema_version": SELECTOR.PROFILE_SCHEMA,
        "status": "profiled_diagnostic",
        "measurement_eligible": False,
        "promotion": False,
        "binding": {"identity": {"identity_sha256": IDENTITY_SHA}},
        "profiler": {},
        "trace": {},
        "mapping": {},
        "timing_ns": {
            "prefill": {
                "families": {
                    "paged_validation": {
                        "exclusive_ns": 1_000_000,
                        "non_overlap_ns": 1_000_000,
                        "active_union_ns": 1_000_000,
                    }
                }
            }
        },
        "timing_ms": {},
        "eligibility_blockers": ["diagnostic only"],
        "schedule_separation": {},
    }
    path = tmp_path / "profile.json"
    write_json(path, profile)
    snapshot = SELECTOR.capture(path)
    result = SELECTOR.select([(snapshot, SELECTOR.parse_json(snapshot))])
    item = candidate(result, "paged-kv-table-validation-v1")
    assert result["status"] == "no_eligible_candidate"
    assert item["eligible"] is False
    assert "eligible_raw_evidence_missing" in item["reason_codes"]
    assert "paged_kv_d2h_count_missing" in item["reason_codes"]
    assert "paged_kv_stream_sync_count_missing" in item["reason_codes"]
    assert result["input_warnings"] == [
        "diagnostic family-exclusive profiles are not measurement eligible and do not provide D2H/stream-sync counts"
    ]


def test_cli_writes_deterministic_json_and_refuses_overwrite(tmp_path: Path) -> None:
    evidence = tmp_path / "raw.json"
    output = tmp_path / "selection.json"
    write_json(evidence, raw_fixture())
    assert SELECTOR.main(["--evidence", str(evidence), "--output", str(output)]) == 0
    raw = output.read_bytes()
    parsed = json.loads(raw)
    assert parsed["selected_candidate_id"] == "paged-kv-table-validation-v1"
    assert raw == (
        json.dumps(parsed, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("ascii")
    assert SELECTOR.main(["--evidence", str(evidence), "--output", str(output)]) == 2


def test_paged_kv_requires_counts_and_observed_transfer_or_sync(tmp_path: Path) -> None:
    missing = raw_fixture()
    missing["capabilities"]["d2h_count"] = False
    for row in missing["measurements"]:
        row["d2h_count"] = None
    seal(missing)
    item = candidate(select_raw(tmp_path, missing), "paged-kv-table-validation-v1")
    assert "paged_kv_d2h_count_missing" in item["reason_codes"]

    zero = raw_fixture()
    for row in zero["measurements"]:
        row["d2h_count"] = 0
        row["stream_sync_count"] = 0
    seal(zero)
    other = tmp_path / "other"
    other.mkdir()
    item = candidate(select_raw(other, zero), "paged-kv-table-validation-v1")
    assert "paged_kv_transfer_or_sync_not_observed" in item["reason_codes"]


def test_candidate_ranking_uses_e_minus_n_then_stable_ties(tmp_path: Path) -> None:
    paged = raw_fixture("paged-kv-table-validation-v1")
    projection = raw_fixture("aq4-register-bm8-v1")
    for row in projection["measurements"]:
        row["recoverable_family_exclusive_ms"] = 15.0
    seal(projection)
    paged_path = tmp_path / "paged.json"
    projection_path = tmp_path / "projection.json"
    write_json(paged_path, paged)
    write_json(projection_path, projection)
    snapshots = [SELECTOR.capture(paged_path), SELECTOR.capture(projection_path)]
    result = SELECTOR.select(
        [(snapshot, SELECTOR.parse_json(snapshot)) for snapshot in reversed(snapshots)]
    )
    assert result["selected_candidate_id"] == "aq4-register-bm8-v1"
    assert result["eligible_candidate_ids"] == [
        "aq4-register-bm8-v1",
        "paged-kv-table-validation-v1",
    ]
