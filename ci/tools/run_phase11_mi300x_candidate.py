#!/usr/bin/env python3
"""Validate and dry-run the Phase 11 exact-gfx942 MI300X candidate profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, read_json  # noqa: E402

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Phase 11 MI300X runner dependency missing: {exc}") from exc


MANIFEST = ROOT / "ci/matrix/phase11-mi300x-candidate-v1.json"
SCHEMA = ROOT / "ci/schema/phase11-mi300x-dry-run-v1.schema.json"
PROFILE_ORDER = ("preflight", "operator", "slice", "full-model", "service", "performance")


def build_dry_run(selected: list[str]) -> dict[str, Any]:
    manifest = read_json(MANIFEST)
    target = manifest.get("target", {})
    if target != {
        "exact_arch": "gfx942",
        "code_object": 6,
        "wave_size": 64,
        "xnack": "off",
        "sramecc": "on",
        "generic_processor_version": 0,
        "gpu_count": 1,
    }:
        raise ContractError("Phase 11 candidate exact gfx942 target tuple drifted")
    if manifest.get("expected_capabilities", {}).get("silent_fallback_allowed") is not False:
        raise ContractError("Phase 11 candidate must reject silent fallback")
    indexed = {profile["name"]: profile for profile in manifest.get("profiles", [])}
    if tuple(indexed) != PROFILE_ORDER:
        raise ContractError("Phase 11 candidate profile order or membership drifted")
    if not selected:
        selected = list(PROFILE_ORDER)
    if len(selected) != len(set(selected)) or any(name not in indexed for name in selected):
        raise ContractError("Phase 11 candidate selected profile is invalid or duplicated")
    profiles = [
        {
            "name": name,
            "estimated_minutes": indexed[name]["estimated_minutes"],
            "requires_model": indexed[name]["requires_model"],
            "steps": indexed[name]["steps"],
            "boundary_values": manifest["boundary_values"] if name in {"operator", "slice"} else [],
        }
        for name in selected
    ]
    report = {
        "schema_version": "phase11-mi300x-dry-run-v1",
        "state": "PASS",
        "execution_attempted": False,
        "candidate_id": manifest["candidate_id"],
        "target": {
            "exact_arch": target["exact_arch"],
            "wave_size": target["wave_size"],
            "gpu_count": target["gpu_count"],
        },
        "profiles": profiles,
        "totals": {
            "selected_profiles": len(profiles),
            "estimated_minutes": sum(profile["estimated_minutes"] for profile in profiles),
        },
    }
    errors = sorted(Draft202012Validator(read_json(SCHEMA)).iter_errors(report), key=lambda item: list(item.path))
    if errors:
        raise ContractError("Phase 11 MI300X dry-run schema failed: " + "; ".join(error.message for error in errors[:5]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="append", choices=PROFILE_ORDER, default=[])
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="plan only; GPU commands are deliberately not executed in Phase 11")
    arguments = parser.parse_args()
    try:
        report = build_dry_run(arguments.profile)
    except (ContractError, KeyError, TypeError, ValueError) as exc:
        print(f"phase11 MI300X candidate: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
