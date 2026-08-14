from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "ci/tools/run_phase11_mi300x_candidate.py"
SPEC = importlib.util.spec_from_file_location("phase11_mi300x_candidate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_profiles_dry_run_is_ordered_and_model_free_first() -> None:
    report = MODULE.build_dry_run([])
    assert report["state"] == "PASS"
    assert report["execution_attempted"] is False
    assert [profile["name"] for profile in report["profiles"]] == list(MODULE.PROFILE_ORDER)
    assert report["profiles"][0]["requires_model"] is False
    assert report["profiles"][1]["boundary_values"][-3:] == [1023, 1024, 1025]
    assert report["totals"] == {"selected_profiles": 6, "estimated_minutes": 435}


def test_profile_subset_preserves_requested_order() -> None:
    report = MODULE.build_dry_run(["preflight", "operator"])
    assert [profile["name"] for profile in report["profiles"]] == ["preflight", "operator"]
    assert report["totals"] == {"selected_profiles": 2, "estimated_minutes": 75}
