#!/usr/bin/env python3
"""Validate and resolve the versioned Phase 7 CI/CD lifecycle profiles."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import ContractError, ROOT, canonical_bytes, read_json  # type: ignore[no-redef]
else:
    from .common import ContractError, ROOT, canonical_bytes, read_json

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"phase7 lifecycle dependency missing: {exc}") from exc


PROFILE_PATH = ROOT / "ci/matrix/phase7-ci-profiles-v1.json"
PROFILE_SCHEMA_PATH = ROOT / "ci/schema/phase7-ci-profiles-v1.schema.json"
COMPATIBILITY_PATH = ROOT / "ci/matrix/phase7-compatibility-v1.json"
COMPATIBILITY_SCHEMA_PATH = ROOT / "ci/schema/phase7-compatibility-v1.schema.json"
TUPLE_SCHEMA_PATH = ROOT / "ci/schema/compatibility-tuple-v1.schema.json"
EXPECTED_TARGETS = [
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
    "gfx1036", "gfx1200", "gfx1201", "gfx942",
]
EXPECTED_TUPLES = [
    "local-v620-gfx1030-rocm714-hwe617",
    "local-r9700-gfx1201-rocm714-hwe617",
]
EXPECTED_GPU_TIERS = ["tier_g0", "tier_g3", "tier_g4", "tier_p1"]


def _fail(message: str) -> None:
    raise ContractError(message)


def _validate_schema(document: Any, schema: dict[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        _fail(f"{label} schema validation failed: " + "; ".join(error.message for error in errors[:5]))


def _load(path: Path, label: str) -> dict[str, Any]:
    document = read_json(path)
    if not isinstance(document, dict):
        _fail(f"{label} must be a JSON object")
    return document


def validate_contracts(root: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_path = root / PROFILE_PATH.relative_to(ROOT)
    profile_schema_path = root / PROFILE_SCHEMA_PATH.relative_to(ROOT)
    compatibility_path = root / COMPATIBILITY_PATH.relative_to(ROOT)
    compatibility_schema_path = root / COMPATIBILITY_SCHEMA_PATH.relative_to(ROOT)
    tuple_schema_path = root / TUPLE_SCHEMA_PATH.relative_to(ROOT)

    profile = _load(profile_path, "Phase 7 profile matrix")
    profile_schema = _load(profile_schema_path, "Phase 7 profile schema")
    compatibility = _load(compatibility_path, "Phase 7 compatibility matrix")
    compatibility_schema = _load(compatibility_schema_path, "Phase 7 compatibility schema")
    tuple_schema = _load(tuple_schema_path, "compatibility tuple schema")
    Draft202012Validator.check_schema(profile_schema)
    Draft202012Validator.check_schema(compatibility_schema)
    Draft202012Validator.check_schema(tuple_schema)
    _validate_schema(profile, profile_schema, "Phase 7 profile matrix")

    if set(compatibility) != {"schema_version", "matrix_id", "revision", "tuples"}:
        _fail("Phase 7 compatibility matrix has missing or unknown keys")
    if compatibility["schema_version"] != "phase7-compatibility-v1" or compatibility["matrix_id"] != "phase7-compatibility-v1":
        _fail("Phase 7 compatibility matrix identity is stale")
    tuples = compatibility["tuples"]
    if not isinstance(tuples, list) or len(tuples) != 2:
        _fail("Phase 7 compatibility matrix must contain exactly two canonical tuples")
    for index, record in enumerate(tuples):
        _validate_schema(record, tuple_schema, f"Phase 7 compatibility tuple {index}")

    if profile["compile_targets"] != EXPECTED_TARGETS:
        _fail("Phase 7 compile target order or set drifted")
    profiles = profile["profiles"]
    if [item["name"] for item in profiles] != ["daily", "weekly", "release"]:
        _fail("Phase 7 profile order or set drifted")
    if [item["tuple_id"] for item in tuples] != EXPECTED_TUPLES:
        _fail("Phase 7 canonical tuple order or set drifted")
    tuple_ids = {item["tuple_id"] for item in tuples}
    for item in profiles:
        if not set(item["gpu_tuples"]).issubset(tuple_ids):
            _fail(f"Phase 7 profile {item['name']} references an unknown tuple")
        if item["gpu_tiers"] != EXPECTED_GPU_TIERS:
            _fail(
                f"Phase 7 profile {item['name']} claims GPU tiers outside the direct full-model observation"
            )
    daily, weekly, release = profiles
    if daily["compile_targets"] != ["gfx1030", "gfx1201"] or daily["gpu_tuples"] != EXPECTED_TUPLES:
        _fail("daily profile must select both canonical tuples")
    if weekly["compile_targets"] != EXPECTED_TARGETS or release["compile_targets"] != EXPECTED_TARGETS:
        _fail("weekly/release profiles must compile every planned exact target")
    if weekly["gpu_tuples"] != EXPECTED_TUPLES or release["gpu_tuples"] != EXPECTED_TUPLES:
        _fail("weekly/release profiles must select both canonical tuples")
    if any(item["blocking"] for item in (daily, weekly)) or not release["blocking"]:
        _fail("only the explicit release profile may be blocking")
    if any(item["retention_days"] != 30 for item in (daily, weekly)) or release["retention_days"] != 90:
        _fail("Phase 7 artifact retention drifted")
    if profile["claims"] != {
        "performance_hard_gate": False,
        "compile_proves_runtime": False,
        "cpu_proves_gpu": False,
    }:
        _fail("Phase 7 non-claim contract drifted")
    return profile, compatibility


def resolve_profile(
    profile: dict[str, Any], *, event: str, schedule: str | None = None,
    requested_profile: str | None = None, release_action: str | None = None,
) -> dict[str, Any]:
    workflow = profile["workflow"]
    if event == "schedule":
        if requested_profile:
            _fail("scheduled execution cannot override its profile")
        if schedule == workflow["daily_cron"]:
            selected = "daily"
        elif schedule == workflow["weekly_cron"]:
            selected = "weekly"
        else:
            _fail("unknown Phase 7 schedule")
    elif event == "workflow_dispatch":
        if requested_profile not in workflow["manual_profiles"]:
            _fail("manual Phase 7 execution requires a registered profile")
        selected = requested_profile
    elif event == "release":
        if release_action != workflow["release_event"]:
            _fail("only a published release may select the release profile")
        if requested_profile and requested_profile != "release":
            _fail("release event cannot select a non-release profile")
        selected = "release"
    else:
        _fail(f"unsupported Phase 7 event: {event}")

    record = next(item for item in profile["profiles"] if item["name"] == selected)
    return {
        "schema_version": "phase7-profile-selection-v1",
        "state": "PASS",
        "profile": selected,
        "event": event,
        "host_rows": record["host_rows"],
        "compile_targets": record["compile_targets"],
        "gpu_tuples": record["gpu_tuples"],
        "gpu_tiers": record["gpu_tiers"],
        "performance_lane": record["performance_lane"],
        "retention_days": record["retention_days"],
        "timeout_minutes": record["timeout_minutes"],
        "blocking": record["blocking"],
        "claims": profile["claims"],
    }


def _write_output(path: Path, document: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        _fail(f"refusing to overwrite Phase 7 output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(document))


def _write_github_output(path: Path, selection: dict[str, Any]) -> None:
    lines = {
        "profile": selection["profile"],
        "host_rows": json.dumps(selection["host_rows"], separators=(",", ":")),
        "compile_targets": json.dumps(selection["compile_targets"], separators=(",", ":")),
        "gpu_tuples": json.dumps(selection["gpu_tuples"], separators=(",", ":")),
        "retention_days": str(selection["retention_days"]),
        "timeout_minutes": str(selection["timeout_minutes"]),
        "blocking": str(selection["blocking"]).lower(),
    }
    with path.open("a", encoding="utf-8") as stream:
        for key, value in lines.items():
            if "\n" in value or "\r" in value:
                _fail("Phase 7 GitHub output contains a newline")
            stream.write(f"{key}={value}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--event", required=True, choices=("schedule", "workflow_dispatch", "release"))
    resolve.add_argument("--schedule")
    resolve.add_argument("--requested-profile")
    resolve.add_argument("--release-action")
    resolve.add_argument("--output", type=Path, required=True)
    resolve.add_argument("--github-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile, _compatibility = validate_contracts()
        if args.command == "validate":
            print("phase7 lifecycle contracts: PASS")
            return 0
        selection = resolve_profile(
            profile,
            event=args.event,
            schedule=args.schedule,
            requested_profile=args.requested_profile,
            release_action=args.release_action,
        )
        _write_output(args.output, selection)
        if args.github_output:
            _write_github_output(args.github_output, selection)
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"phase7 lifecycle: FAIL: {exc}", file=sys.stderr)
        return 1
    print(f"phase7 lifecycle selection: PASS profile={selection['profile']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
