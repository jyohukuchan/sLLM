#!/usr/bin/env python3
"""Prepare and launch isolated SQ8 v0.2 GPU capture plans.

This tool never changes a production default.  It reads the frozen v0.2 JSON
and a read-only reference index, writes a create-new capture plan, and (only
when explicitly asked) launches the dedicated capture binary in a subprocess
whose experimental-selector environment is scrubbed and reconstructed from
that plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


EXPECTED_GATE_SHA256 = "64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf"
GATE_SCHEMA = "ullm.sq8.numerical_gate.relative_fp32.v0.2"
REFERENCE_INDEX_SCHEMA = "ullm.sq8.gate.v0.2.reference-index.v1"
PLAN_SCHEMA = "ullm.sq8.gate.v0.2.capture-plan.v1"
BLOCKED_PLAN_SCHEMA = "ullm.sq8.gate.v0.2.capture-plan-blocked.v1"

SELECTOR_ENVIRONMENTS = (
    "ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE",
    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE",
    "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE",
)


class PlanError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read JSON {path}: {exc}") from exc


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PlanError(f"refusing to overwrite existing output {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frozen_gate(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_GATE_SHA256:
        raise PlanError(
            f"frozen gate SHA-256 mismatch: expected={EXPECTED_GATE_SHA256} actual={digest}"
        )
    value = json.loads(raw)
    if value.get("schema_version") != GATE_SCHEMA:
        raise PlanError(f"unexpected frozen gate schema {value.get('schema_version')!r}")
    return value, digest


def selector_definition(candidate: str, role: str) -> dict[str, Any]:
    if role == "control":
        return {
            "enabled": False,
            "kind": "matched_ck_or_direct_control",
            "configuration": {
                "experimental_environment": "all SQ8 candidate selector variables scrubbed",
                "ordinary_profile": "CK/direct selected by the matched build",
            },
            "environment": {},
        }
    definitions = {
        "flash2-staged-wave32": {
            "kind": "flash2_staged_wave32_reduction",
            "environment": {"ULLM_USE_SQ8_0_FLASH2_STAGED_WAVE32_PROTOTYPE": "1"},
            "build_feature": "rocm-ck-gfx1201",
            "scope": "SQ8_0 full-model; isolated process environment",
        },
        "paged-decode-source-tile-128": {
            "kind": "paged_decode_source_tile_split",
            "environment": {
                "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE": "128",
                "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE": "1",
            },
            "build_feature": "rocm-ck-gfx1201",
            "scope": "SQ8_0 full-model; explicit evaluation-only containment bypass",
        },
        "paged-decode-source-tile-256": {
            "kind": "paged_decode_source_tile_split",
            "environment": {
                "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_TILE": "256",
                "ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE": "1",
            },
            "build_feature": "rocm-ck-gfx1201",
            "scope": "SQ8_0 full-model; explicit evaluation-only containment bypass",
        },
        "handwritten-wmma-projection": {
            "kind": "handwritten_wmma_projection_prototype",
            "environment": {},
            "build_feature": "rocm-handwritten-projection-gfx1201",
            "scope": "SQ8_0 M=1-only private session selector",
        },
    }
    try:
        chosen = definitions[candidate]
    except KeyError as exc:
        raise PlanError(f"unsupported candidate selector {candidate!r}") from exc
    return {
        "enabled": True,
        "kind": chosen["kind"],
        "configuration": {
            "candidate_id": candidate,
            "build_feature": chosen["build_feature"],
            "scope": chosen["scope"],
            "all_other_experimental_selectors": "disabled",
        },
        "environment": chosen["environment"],
    }


def capture_position_id(mode: str, case_id: str, phase: str, phase_index: int) -> str:
    return f"{mode}:{case_id}:{phase}:{phase_index:05}"


def frozen_capture_position_ids(gate: Mapping[str, Any]) -> set[str]:
    """Derive the complete v0.2 capture set directly from the frozen JSON."""

    corpus = gate["corpus"]
    cases = {
        str(case["id"]): case
        for case in [*corpus["primary_decode_streams"], *corpus["required_boundary_cases"]]
    }
    expected: set[str] = set()
    for stream in corpus["primary_decode_streams"]:
        expected.update(
            capture_position_id("sequential_m1", stream["id"], "decode", decode_index)
            for decode_index in range(int(stream["forced_decode_tokens"]))
        )
    for boundary in corpus["required_boundary_cases"]:
        expected.update(
            capture_position_id("sequential_m1", boundary["id"], "decode", decode_index)
            for decode_index in range(int(boundary["forced_decode_tokens"]))
        )
    for case_id in corpus["prefill_coverage"]["m128_inputs"]:
        case = cases[str(case_id)]
        prompt_tokens = int(case["prompt_tokens"])
        total_tokens = prompt_tokens + int(case["forced_decode_tokens"])
        ordinals = list(range(127, total_tokens, 128))
        if not ordinals or ordinals[-1] != total_tokens - 1:
            ordinals.append(total_tokens - 1)
        for ordinal in ordinals:
            if ordinal < prompt_tokens:
                expected.add(
                    capture_position_id("m128_chunks_with_declared_tail", str(case_id), "prompt", ordinal)
                )
            else:
                expected.add(
                    capture_position_id(
                        "m128_chunks_with_declared_tail",
                        str(case_id),
                        "decode",
                        ordinal - prompt_tokens,
                    )
                )
    return expected


def reference_index_qualification(gate: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    """Reject partial indexes before they can consume an isolated GPU window."""

    positions = index.get("positions")
    if not isinstance(positions, list):
        raise PlanError("reference index positions must be a list")
    actual_ids: list[str] = []
    for position in positions:
        if not isinstance(position, dict) or not isinstance(position.get("id"), str):
            raise PlanError("reference index has a position without a stable id")
        actual_ids.append(position["id"])
    if len(set(actual_ids)) != len(actual_ids):
        raise PlanError("reference index repeats a capture position id")
    expected_ids = frozen_capture_position_ids(gate)
    actual = set(actual_ids)
    missing = sorted(expected_ids - actual)
    unexpected = sorted(actual - expected_ids)
    return {
        "complete": actual == expected_ids,
        "actual_positions": len(actual),
        "required_positions": len(expected_ids),
        "missing_positions": len(missing),
        "unexpected_positions": len(unexpected),
        "missing_examples": missing[:5],
        "unexpected_examples": unexpected[:5],
    }


def materialize_prompt(gate: Mapping[str, Any], case: Mapping[str, Any], gate_path: Path) -> list[int]:
    fixture_relative = gate["corpus"]["fixture_root"]
    fixture_root = None
    for ancestor in gate_path.resolve().parents:
        candidate = ancestor / fixture_relative
        if candidate.is_dir():
            fixture_root = candidate
            break
    if fixture_root is None:
        raise PlanError(f"cannot resolve fixture root {fixture_relative!r} from {gate_path}")
    manifest = fixture_root / "manifest.json"
    if sha256_file(manifest) != gate["corpus"]["fixture_manifest_sha256"]:
        raise PlanError(f"fixture manifest SHA-256 mismatch: {manifest}")
    prompt_tokens = int(case["prompt_tokens"])
    relative = case.get("input")
    if relative is None:
        return list(range(1, prompt_tokens + 1))
    path = fixture_root / str(relative)
    raw = path.read_bytes()
    expected = case.get("input_sha256")
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None and actual != expected:
        raise PlanError(f"fixture SHA-256 mismatch at {path}: expected={expected} actual={actual}")
    if path.suffix == ".u32le":
        if len(raw) % 4:
            raise PlanError(f"raw prompt has non-u32 size: {path}")
        values = [int.from_bytes(raw[offset : offset + 4], "little") for offset in range(0, len(raw), 4)]
        if values != list(range(1, prompt_tokens + 1)):
            raise PlanError(f"raw range prompt is not the frozen [1..N] sequence: {path}")
        return values
    if path.suffix == ".json":
        value = json.loads(raw)
        expected_case = value.get("expected", {})
        values = expected_case.get("token_ids")
        if expected_case.get("prompt_tokens") != prompt_tokens or not isinstance(values, list):
            raise PlanError(f"chat fixture has unexpected frozen shape: {path}")
        if len(values) != prompt_tokens or any(not isinstance(item, int) for item in values):
            raise PlanError(f"chat fixture token IDs are invalid: {path}")
        return values
    raise PlanError(f"unsupported fixture type {path}")


def read_teacher_inputs(reference_root: Path, mode: str, case_id: str, forced: int) -> tuple[list[int], str]:
    case_root = reference_root / "cases" / mode / case_id
    run = load_json(case_root / "run.json")
    if run.get("status") != "complete":
        raise PlanError(f"reference case is not complete: {case_root}")
    path = case_root / "teacher-forced-tokens.u32le"
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if run.get("teacher_forced_tokens_u32le_sha256") != actual:
        raise PlanError(f"teacher-forced token SHA-256 mismatch: {path}")
    if len(raw) % 4:
        raise PlanError(f"teacher-forced file is not u32le: {path}")
    values = [int.from_bytes(raw[index : index + 4], "little") for index in range(0, len(raw), 4)]
    if len(values) != forced + 1:
        raise PlanError(
            f"teacher-forced token count mismatch at {path}: expected={forced + 1} actual={len(values)}"
        )
    return values[:forced], actual


def build_plan(args: argparse.Namespace) -> int:
    gate, gate_sha = frozen_gate(args.gate)
    index = load_json(args.reference)
    if index.get("schema_version") != REFERENCE_INDEX_SCHEMA:
        raise PlanError(f"reference index schema mismatch: {args.reference}")
    if index.get("frozen_gate", {}).get("sha256") != gate_sha:
        raise PlanError("reference index is not bound to the frozen gate")
    candidate = args.candidate
    if candidate == "sq8_1-w8a8":
        write_json_new(
            args.output,
            {
                "schema_version": BLOCKED_PLAN_SCHEMA,
                "status": "excluded_by_frozen_scope",
                "frozen_gate_sha256": gate_sha,
                "candidate_id": candidate,
                "reason": "v0.2 scope.format_id is SQ8_0; SQ8_1 W8A8 is a separate quality gate.",
            },
        )
        return 0
    reference_qualification = reference_index_qualification(gate, index)
    if not reference_qualification["complete"]:
        write_json_new(
            args.output,
            {
                "schema_version": BLOCKED_PLAN_SCHEMA,
                "status": "blocked_reference_or_capture",
                "frozen_gate_sha256": gate_sha,
                "candidate_id": candidate,
                "role": args.role,
                "reason": "reference index does not yet contain every frozen v0.2 capture position; no GPU plan was prepared",
                "coverage": reference_qualification,
            },
        )
        return 0
    recorded_reference_completion = index.get("reference_qualification")
    if isinstance(recorded_reference_completion, dict) and recorded_reference_completion.get("complete") is not True:
        write_json_new(
            args.output,
            {
                "schema_version": BLOCKED_PLAN_SCHEMA,
                "status": "blocked_reference_or_capture",
                "frozen_gate_sha256": gate_sha,
                "candidate_id": candidate,
                "role": args.role,
                "reason": "reference index records an incomplete strict-F32 reference qualification; no GPU plan was prepared",
                "reference_qualification": recorded_reference_completion,
            },
        )
        return 0
    if (
        candidate == "handwritten-wmma-projection"
        and args.role == "candidate"
        and not args.diagnostic_only
    ):
        write_json_new(
            args.output,
            {
                "schema_version": BLOCKED_PLAN_SCHEMA,
                "status": "blocked_reference_or_capture",
                "frozen_gate_sha256": gate_sha,
                "candidate_id": candidate,
                "reason": "The private handwritten WMMA selector is M=1-only and cannot satisfy required M=128 prefill coverage. Use --diagnostic-only only for non-qualifying M=1 evidence.",
            },
        )
        return 0
    selector = selector_definition(candidate, args.role)
    if args.role == "control":
        candidate_identity = "matched-ck-or-direct-control"
    else:
        candidate_identity = candidate
    case_specs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for position in index.get("positions", []):
        if not isinstance(position, dict):
            raise PlanError("reference index contains a non-object position")
        mode = str(position["mode"])
        case_id = str(position["case_id"])
        if candidate == "handwritten-wmma-projection" and mode != "sequential_m1":
            continue
        case_specs.setdefault((mode, case_id), []).append(position)
    if not case_specs:
        raise PlanError("reference index did not provide any capture positions for this plan")
    frozen_cases = {
        str(case["id"]): case
        for case in [
            *gate["corpus"]["primary_decode_streams"],
            *gate["corpus"]["required_boundary_cases"],
        ]
    }
    reference_root = Path(str(index["reference_root"]))
    cases: list[dict[str, Any]] = []
    teacher_hashes: dict[str, str] = {}
    for (mode, case_id), positions in sorted(case_specs.items()):
        case = frozen_cases.get(case_id)
        if case is None:
            raise PlanError(f"reference index case is not frozen: {case_id}")
        forced = int(case["forced_decode_tokens"])
        prompt = materialize_prompt(gate, case, args.gate)
        teacher_inputs, teacher_hash = read_teacher_inputs(reference_root, mode, case_id, forced)
        teacher_hashes[f"{mode}:{case_id}"] = teacher_hash
        compact_positions = []
        for position in sorted(positions, key=lambda item: (int(item["ordinal"]), str(item["id"]))):
            expected_input = int(position["input_token_id"])
            ordinal = int(position["ordinal"])
            if ordinal < len(prompt):
                actual_input = prompt[ordinal]
            else:
                decode_index = ordinal - len(prompt)
                if decode_index >= len(teacher_inputs):
                    raise PlanError(f"reference ordinal exceeds teacher stream: {position['id']}")
                actual_input = teacher_inputs[decode_index]
            if expected_input != actual_input:
                raise PlanError(
                    f"reference input token does not match frozen teacher stream at {position['id']}: "
                    f"reference={expected_input} materialized={actual_input}"
                )
            compact_positions.append(
                {
                    key: position[key]
                    for key in (
                        "id",
                        "case_id",
                        "mode",
                        "ordinal",
                        "phase",
                        "phase_index",
                        "position",
                        "input_token_id",
                        "layer_required",
                    )
                }
            )
        cases.append(
            {
                "case_id": case_id,
                "mode": mode,
                "prompt_token_ids": prompt,
                "teacher_forced_input_tokens": teacher_inputs,
                "teacher_forced_tokens_u32le_sha256": teacher_hash,
                "positions": compact_positions,
            }
        )
    identity = dict(index["identity"])
    plan = {
        "schema_version": PLAN_SCHEMA,
        "frozen_gate": {"path": str(args.gate.resolve()), "sha256": gate_sha},
        "reference_index": str(args.reference.resolve()),
        "role": args.role,
        "candidate": {"id": candidate_identity},
        "selector": selector,
        "artifact": str(args.artifact.resolve()),
        "package": str(args.package.resolve()),
        "identity": {
            "artifact_content_sha256": identity["artifact_content_sha256"],
            "fixture_manifest_sha256": identity["fixture_manifest_sha256"],
            "materialized_token_hashes": identity["materialized_token_hashes"],
            "reference_executable_sha256": identity["reference_executable_sha256"],
            "reference_identity": identity.get("reference_identity"),
            "teacher_forced_tokens_u32le_sha256": teacher_hashes,
        },
        "qualification": {
            "reference_index_coverage_complete": reference_qualification["complete"],
            "reference_index_coverage": reference_qualification,
            "diagnostic_only": bool(args.diagnostic_only),
            "handwritten_m128_unavailable": candidate == "handwritten-wmma-projection",
        },
        "cases": cases,
    }
    write_json_new(args.output, plan)
    print(json.dumps({"output": str(args.output), "cases": len(cases), "positions": sum(len(item["positions"]) for item in cases)}, sort_keys=True))
    return 0


def run_capture(args: argparse.Namespace) -> int:
    plan = load_json(args.plan)
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise PlanError(f"capture plan is not runnable: {args.plan}")
    if args.output.exists():
        raise PlanError(f"refusing to overwrite capture output {args.output}")
    environment = os.environ.copy()
    for name in SELECTOR_ENVIRONMENTS:
        environment.pop(name, None)
    for name, value in plan.get("selector", {}).get("environment", {}).items():
        environment[str(name)] = str(value)
    command = [str(args.capture_binary.resolve()), "--plan", str(args.plan.resolve()), "--output", str(args.output.resolve())]
    completed = subprocess.run(command, env=environment, check=False)
    return int(completed.returncode)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="write a create-new capture plan")
    prepare.add_argument("--gate", type=Path, required=True)
    prepare.add_argument("--reference", type=Path, required=True)
    prepare.add_argument("--artifact", type=Path, required=True)
    prepare.add_argument("--package", type=Path, required=True)
    prepare.add_argument("--role", choices=("control", "candidate"), required=True)
    prepare.add_argument(
        "--candidate",
        choices=(
            "flash2-staged-wave32",
            "paged-decode-source-tile-128",
            "paged-decode-source-tile-256",
            "handwritten-wmma-projection",
            "sq8_1-w8a8",
        ),
        required=True,
    )
    prepare.add_argument("--diagnostic-only", action="store_true")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(function=build_plan)
    run = sub.add_parser("run", help="launch a prepared plan in an isolated subprocess")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--capture-binary", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.set_defaults(function=run_capture)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.function(args))
    except PlanError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
